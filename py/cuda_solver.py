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
) -> Tuple[List[List[int]], List[int], List[Tuple[Piece, int, List[Vector]]], int]:
    """
    Build candidate relative offsets for bitboard kernel with caching.
    Returns:
      offsets_list: list of lists of offsets
      lengths_list: list of candidate lengths
      meta:         list of (piece, origin, vec_list) for CPU-side placement
      max_len:      maximum candidate length
    """
    candidates_meta: List[Tuple[Piece, int, List[Vector]]] = []
    offsets_list: List[List[int]] = []
    lengths_list: List[int] = []
    max_len = 0

    for piece in pieces:
        # Cache transforms once
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

        # Cache offsets per board width (offset values depend on width)
        width_cache = getattr(piece, "_cached_offsets", None)
        if width_cache is None:
            width_cache = {}
            piece._cached_offsets = width_cache

        cached_for_width = width_cache.get(board_width)
        if cached_for_width is None:
            entries = []
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

                offs = [(cy * board_width + cx) for (cx, cy) in coords]
                entries.append((origin, vecs, offs))
            width_cache[board_width] = entries
            cached_for_width = entries

        for origin, vecs, offs in cached_for_width:
            candidates_meta.append((piece, origin, vecs))
            offsets_list.append(offs)
            L = len(offs)
            lengths_list.append(L)
            if L > max_len:
                max_len = L

    return offsets_list, lengths_list, candidates_meta, max_len


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
    offsets_flat: np.ndarray,  # (N × max_len) padded offsets
    max_len: int,
    lengths: np.ndarray,  # (N,)
    results: np.ndarray,  # (N,) – 1 valid, 0 invalid
) -> None:
    i = cuda.grid(1)
    if i >= lengths.shape[0]:
        return

    total_cells = width * height
    L = lengths[i]
    offset_base = i * max_len

    for j in range(L):
        cell_idx = base_idx + offsets_flat[offset_base + j]
        if cell_idx < 0 or cell_idx >= total_cells:
            results[i] = 0
            return

        word = cell_idx >> 6
        bit = cell_idx & 63
        if (board_bits[word] >> bit) & 1 == 0:
            results[i] = 0
            return

    results[i] = 1


@cuda.jit
def _kernel_check_candidates_preloaded(
    board_bits: np.ndarray,  # uint64 bitmask (3,)
    width: int,
    height: int,
    base_idx: int,
    all_offsets: np.ndarray,  # (total_candidates, max_len) on GPU
    all_lengths: np.ndarray,  # (total_candidates,) on GPU
    candidate_indices: np.ndarray,  # (N,) indices into all_offsets/all_lengths
    results: np.ndarray,  # (N,) valid flags
    valid_indices: np.ndarray,  # (N,) output: indices of valid candidates
    valid_count: np.ndarray,  # (1,) output: count of valid
) -> None:
    i = cuda.grid(1)
    if i >= candidate_indices.shape[0]:
        return

    total_cells = width * height
    cand_idx = candidate_indices[i]
    L = all_lengths[cand_idx]

    is_valid = 1
    for j in range(L):
        cell_idx = base_idx + all_offsets[cand_idx, j]
        if cell_idx < 0 or cell_idx >= total_cells:
            is_valid = 0
            break

        word = cell_idx >> 6
        bit = cell_idx & 63
        if (board_bits[word] >> bit) & 1 == 0:
            is_valid = 0
            break

    results[i] = is_valid
    if is_valid:
        # Atomically increment valid count and store index
        idx = cuda.atomic.add(valid_count, 0, 1)
        valid_indices[idx] = i


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
        self._cap_max_len = 0
        # Fixed 3-word mask buffer
        self._d_mask_words = cuda.device_array(3, dtype=np.uint64)
        # Host buffer pool (pinned for faster H2D copies)
        self._h_offsets = None
        self._h_lengths = None
        self._cap_h_N = 0
        self._cap_h_max_len = 0

    def _ensure_buffers(self, N: int, max_len: int):
        # Ensure device buffers have enough capacity; allocate if needed
        if (self._d_offsets_flat is None) or (N > self._cap_N) or (max_len > self._cap_max_len):
            self._cap_N = max(N, self._cap_N or N)
            self._cap_max_len = max(max_len, self._cap_max_len or max_len)
            self._d_offsets_flat = cuda.device_array(self._cap_N * self._cap_max_len, dtype=np.int32)
        if (self._d_lengths is None) or (N > self._cap_N):
            self._d_lengths = cuda.device_array(self._cap_N, dtype=np.int32)
        if (self._d_valids is None) or (N > self._cap_N):
            self._d_valids = cuda.device_array(self._cap_N, dtype=np.int32)

    def _ensure_host_buffers(self, N: int, max_len: int):
        # Host-side pool to reduce allocations (pinned for faster transfers)
        if (self._h_offsets is None) or (N > self._cap_h_N) or (max_len > self._cap_h_max_len):
            self._cap_h_N = max(N, self._cap_h_N or N)
            self._cap_h_max_len = max(max_len, self._cap_h_max_len or max_len)
            self._h_offsets = cuda.pinned_array((self._cap_h_N, self._cap_h_max_len), dtype=np.int32)
        if (self._h_lengths is None) or (N > self._cap_h_N):
            self._h_lengths = cuda.pinned_array(self._cap_h_N, dtype=np.int32)

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
            # Encode candidates for current level
            offsets_list, lengths_list, meta, max_len = _encode_candidates(pieces, len(board._board[0]))
            N = len(lengths_list)
            if N == 0:
                return solutions

            mask, width, height = _board_to_bitmask(board)
            pos_abs_x = board._origin.x + pos.x
            pos_abs_y = board._origin.y + pos.y
            base_idx = pos_abs_y * width + pos_abs_x

            self._nbTries += N

            # Fill host buffers with padded format
            self._ensure_host_buffers(N, max_len)
            for i, offs in enumerate(offsets_list):
                self._h_offsets[i, :len(offs)] = offs
            self._h_lengths[:N] = lengths_list

            # Copy to device
            self._ensure_buffers(N, max_len)
            h_offsets_flat = self._h_offsets[:N, :max_len].reshape(N * max_len)
            self._d_offsets_flat[:N * max_len].copy_to_device(h_offsets_flat)
            self._d_lengths[:N].copy_to_device(self._h_lengths[:N])
            self._d_mask_words.copy_to_device(mask)

            threads_per_block = 512
            blocks = (N + threads_per_block - 1) // threads_per_block

            _kernel_check_candidates[blocks, threads_per_block](
                self._d_mask_words, width, height, base_idx,
                self._d_offsets_flat,
                max_len,
                self._d_lengths[:N],
                self._d_valids[:N]
            )

            # Copy results back
            valids_host = self._d_valids[:N].copy_to_host()

            # Process valid placements
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
