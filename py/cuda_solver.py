from datetime import datetime
from typing import List, Tuple

import numpy as np

# Strict CUDA requirement – no CPU fallback here.
try:
    from numba import cuda
except Exception as e:
    raise ImportError(
        "cuda_solver requires Numba CUDA support. Install numba and CUDA toolkit/drivers."
    ) from e

from puzzle import Piece, Board, Vector


def _board_to_bitmask(board: Board) -> Tuple[np.ndarray, int, int]:
    """
    Pack board free cells into a bitmask array of uint64 words.
    Returns (bitmask, width, height).
    Bit value 1 == free (None), 0 == occupied/blocked.
    """
    height = len(board._board)
    width = len(board._board[0])
    # Fixed 3 words (13x13=169 bits); safe for our board size
    mask = np.zeros((3,), dtype=np.uint64)

    idx = 0
    for y in range(height):
        for x in range(width):
            if board._board[y][x] is None:
                word = idx >> 6
                bit = idx & 63
                if word < 3:
                    mask[word] |= np.uint64(1) << np.uint64(bit)
            idx += 1

    return mask, width, height


def _encode_candidates(
    pieces: List[Piece],
    board_width: int,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[Piece, int, List[Vector]]]]:
    """
    Build candidate relative offsets for bitboard kernel.
    Returns:
      offsets: (N, Lmax) int32 – flattened offsets relative to anchor cell
      lengths: (N,) int32 – number of cells per candidate
      meta:    list of (piece, origin, vec_list) for CPU-side placement
    """
    candidates_meta: List[Tuple[Piece, int, List[Vector]]] = []
    offsets: List[List[int]] = []

    for piece in pieces:
        if not hasattr(piece, "_cached_candidates"):
            cached = []
            relTrans = piece.relevantTrans()
            for origin in range(len(piece)):
                for trans in relTrans:
                    if (
                        (piece.sides == "front" and trans.isFront())
                        or (piece.sides == "back" and trans.isBack())
                        or piece.sides == "both"
                    ):
                        vecs = piece._transform(trans)
                        cached.append((origin, [Vector(v.x, v.y) for v in vecs]))
            piece._cached_candidates = cached

        for origin, vecs in piece._cached_candidates:
            coords = [(0, 0)]

            cx = cy = 0
            for idx in range(origin - 1, -1, -1):
                v = vecs[idx]
                cx -= v.x
                cy -= v.y
                coords.append((cx, cy))

            cx = cy = 0
            for idx in range(origin, len(vecs)):
                v = vecs[idx]
                cx += v.x
                cy += v.y
                coords.append((cx, cy))

            offsets.append([cy * board_width + cx for (cx, cy) in coords])
            candidates_meta.append((piece, origin, vecs))

    if not offsets:
        return (
            np.empty((0, 0), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            candidates_meta,
        )

    max_len = max(len(c) for c in offsets)
    N = len(offsets)
    offsets_arr = np.zeros((N, max_len), dtype=np.int32)
    lengths = np.zeros((N,), dtype=np.int32)
    for i, offs in enumerate(offsets):
        lengths[i] = len(offs)
        offsets_arr[i, : len(offs)] = offs

    return offsets_arr, lengths, candidates_meta


class PlacementPiece:
    """
    Lightweight view for placement to avoid deepcopy.
    Provides the minimal interface used by Board.putPiece.
    """
    def __init__(self, name: str, vecs: List[Vector]):
        self.name = name
        self._currShape = vecs
        self._origin = 0

    def setOrigin(self, origin: int):
        self._origin = origin

    def __len__(self) -> int:
        return len(self._currShape) + 1

    def __getitem__(self, idx: int):
        # Mirror Piece.__getitem__ semantics for compatibility with Board.putPiece
        ret = None
        if idx > 0:
            idx -= 1
        if (idx + self._origin) < len(self._currShape) and (idx + self._origin) >= 0:
            ret = self._currShape[idx + self._origin]
        return ret


@cuda.jit
def _kernel_check_candidates(
    board_bits: np.ndarray,  # uint64 bitmask of free cells
    width: int,
    height: int,
    base_idx: int,  # flattened index of anchor cell
    offsets_flat: np.ndarray,  # (N*Lmax,) flattened offsets relative to anchor
    stride_L: int,  # Lmax used to index into flat array
    lengths: np.ndarray,  # (N,)
    results: np.ndarray,  # (N,) – 1 valid, 0 invalid
) -> None:
    i = cuda.grid(1)
    if i >= lengths.shape[0]:
        return

    total_cells = width * height
    L = lengths[i]

    for j in range(L):
        cell_idx = base_idx + offsets_flat[i * stride_L + j]
        if cell_idx < 0 or cell_idx >= total_cells:
            results[i] = 0
            return

        word = cell_idx >> 6
        bit = cell_idx & 63
        if (board_bits[word] >> bit) & 1 == 0:
            results[i] = 0
            return

    results[i] = 1


class PuzzleSolver:
    """
    CUDA-accelerated puzzle solver with the same API as solver.PuzzleSolver.
    Falls back to CPU when CUDA/Numba is unavailable.
    """

    def __init__(self, board: Board, pieces: List[Piece]):
        self._board = board
        self._pieces = pieces
        self._nbPieces = len(self._pieces)
        self._startTime = None
        self._nbTries = 0
        self._nbPcsPut = 0
        self._findAll = False
        self._stop = False
        self._print = True
        # Enforce CUDA device presence – no CPU fallback.
        try:
            available = bool(getattr(cuda, "is_available", lambda: False)())
        except Exception:
            available = False
        if not available:
            raise RuntimeError(
                "CUDA device not available. Use solver.PuzzleSolver for CPU or install CUDA drivers."
            )
        # Preallocated device buffers (capacity managed dynamically)
        self._d_offsets_flat = None
        self._d_lengths = None
        self._d_valids = None
        self._cap_N = 0
        self._cap_L = 0
        # Fixed 3-word mask buffer
        self._d_mask_words = cuda.device_array(3, dtype=np.uint64)

    def _ensure_buffers(self, N: int, max_len: int):
        # Ensure device buffers have enough capacity; allocate if needed
        need_offsets = (self._d_offsets_flat is None) or (N > self._cap_N) or (max_len > self._cap_L)
        if need_offsets:
            self._cap_N = max(N, self._cap_N or N)
            self._cap_L = max(max_len, self._cap_L or max_len)
            self._d_offsets_flat = cuda.device_array(self._cap_N * self._cap_L, dtype=np.int32)
        if (self._d_lengths is None) or (N > self._cap_N):
            self._d_lengths = cuda.device_array(self._cap_N, dtype=np.int32)
        if (self._d_valids is None) or (N > self._cap_N):
            self._d_valids = cuda.device_array(self._cap_N, dtype=np.int32)

    def solve(self, findAll: bool = False, printSol: bool = True):
        self._findAll = findAll
        self._print = printSol
        solutions: List[Board] = []
        self._startTime = datetime.now()
        solutions = self._solve(self._board, self._pieces, solutions)
        return solutions, self._nbTries, self._nbPcsPut

    def _solve(self, board: Board, pieces: List[Piece], solutions: List[Board]):
        """Depth-first search using bitboard kernel."""
        nbPcs = len(pieces)
        if nbPcs:
            pos = board.nextAvailablePos()
            if pos is None:
                return solutions

            offsets, lengths, meta = _encode_candidates(pieces, len(board._board[0]))
            N = lengths.shape[0]
            if N == 0:
                return solutions

            mask, width, height = _board_to_bitmask(board)
            pos_abs_x = board._origin.x + pos.x
            pos_abs_y = board._origin.y + pos.y
            base_idx = pos_abs_y * width + pos_abs_x

            self._nbTries += N

            h_offsets = np.ascontiguousarray(offsets)
            h_lengths = np.ascontiguousarray(lengths)
            h_mask = np.ascontiguousarray(mask)

            # Ensure and fill device buffers
            self._ensure_buffers(N, h_offsets.shape[1])
            h_offsets_flat = h_offsets.reshape(N * h_offsets.shape[1])
            self._d_offsets_flat[: N * h_offsets.shape[1]].copy_to_device(h_offsets_flat)
            self._d_lengths[:N].copy_to_device(h_lengths)
            self._d_mask_words.copy_to_device(h_mask)

            threads_per_block = 512  # larger block size for better occupancy
            blocks = (N + threads_per_block - 1) // threads_per_block

            _kernel_check_candidates[blocks, threads_per_block](
                self._d_mask_words, width, height, base_idx,
                self._d_offsets_flat,
                h_offsets.shape[1],
                self._d_lengths[:N],
                self._d_valids[:N]
            )

            valids_host = self._d_valids[:N].copy_to_host()

            for i, is_valid in enumerate(valids_host):
                if is_valid:
                    piece_i, origin_i, vecs_i = meta[i]
                    # Use lightweight placement view to avoid deepcopy costs
                    piece_view = PlacementPiece(piece_i.name, vecs_i)
                    piece_view.setOrigin(origin_i)

                    newBoard = board.putPiece(piece_view, pos)
                    if newBoard is not None:
                        self._nbPcsPut += 1
                        # Shallow copy and remove by identity; fallback to name if needed
                        newPieces = pieces.copy()
                        try:
                            newPieces.remove(piece_i)
                        except ValueError:
                            newPieces = [p for p in newPieces if p.name != piece_i.name]
                        solutions = self._solve(newBoard, newPieces, solutions)
        else:
            if self._print:
                print(
                    "\nSolution found in {} after testing {} combinations and putting {} pieces:".format(
                        str(datetime.now() - self._startTime)[:-7],
                        self._nbTries,
                        self._nbPcsPut,
                    )
                )
                print(board, flush=True)
            if not self._findAll:
                self._stop = True
            solutions.append(board)
        return solutions
