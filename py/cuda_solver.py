from copy import deepcopy
from datetime import datetime
from sys import stdout
from typing import List, Tuple

import numpy as np

# Strict CUDA requirement – no CPU fallback here.
try:
    from numba import cuda
except Exception as e:
    raise ImportError(
        "cuda_solver requires Numba CUDA support. Install numba and CUDA toolkit/drivers."
    ) from e

from puzzle import Piece, Board, Trans, Coordinate, Vector


def _board_to_free_array(board: Board) -> np.ndarray:
    """
    Convert internal board representation (list of lists with None/0/name)
    into an int array of shape (H, W) where 1 means free (None), 0 means occupied or blocked.
    """
    arr = np.zeros((len(board._board), len(board._board[0])), dtype=np.int32)
    for y in range(len(board._board)):
        for x in range(len(board._board[y])):
            arr[y, x] = 1 if board._board[y][x] is None else 0
    return arr


def _encode_candidates(
    pieces: List[Piece],
    pos: Coordinate,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[Piece, int, List[Vector]]]]:
    """
    Build a batch of candidate placements at current position:
    - For every piece, origin and relevant transformation (respecting sides)
    Returns:
      vectors: (N, Lmax, 2) int32 – transformed vector chains for the kernel
      lengths: (N,) int32 – per-candidate vector chain length
      origins: (N,) int32 – origin index inside the chain
      meta:    list of (piece, origin, vec_list) for CPU-side placement without deepcopy
    """
    candidates_meta: List[Tuple[Piece, int, List[Vector]]] = []
    chains: List[List[Tuple[int, int]]] = []
    origins: List[int] = []

    for piece in pieces:
        # Cache all allowed (origin, transformed vectors) pairs per piece
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
            chains.append([(v.x, v.y) for v in vecs])
            origins.append(origin)
            candidates_meta.append((piece, origin, vecs))

    if not chains:
        return (
            np.empty((0, 0, 2), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            candidates_meta,
        )

    max_len = max(len(c) for c in chains)
    N = len(chains)
    vectors = np.zeros((N, max_len, 2), dtype=np.int32)
    lengths = np.zeros((N,), dtype=np.int32)
    for i, ch in enumerate(chains):
        lengths[i] = len(ch)
        for j, (dx, dy) in enumerate(ch):
            vectors[i, j, 0] = dx
            vectors[i, j, 1] = dy

    return vectors, lengths, np.asarray(origins, dtype=np.int32), candidates_meta


@cuda.jit
def _kernel_check_candidates(
    board_free: np.ndarray,
    pos_abs_x: int,
    pos_abs_y: int,
    vectors: np.ndarray,  # (N, Lmax, 2)
    lengths: np.ndarray,  # (N,)
    origins: np.ndarray,  # (N,)
    results: np.ndarray,  # (N,) – 1 valid, 0 invalid
) -> None:
    i = cuda.grid(1)
    if i >= lengths.shape[0]:
        return

    H = board_free.shape[0]
    W = board_free.shape[1]

    x = pos_abs_x
    y = pos_abs_y

    # Anchor cell must be free
    if y < 0 or y >= H or x < 0 or x >= W or board_free[y, x] == 0:
        results[i] = 0
        return

    L = lengths[i]
    origin = origins[i]

    # Walk negative direction: idx origin-1..0, subtract vectors
    cx = x
    cy = y
    for idx in range(origin - 1, -1, -1):
        dx = vectors[i, idx, 0]
        dy = vectors[i, idx, 1]
        cx -= dx
        cy -= dy
        if cy < 0 or cy >= H or cx < 0 or cx >= W or board_free[cy, cx] == 0:
            results[i] = 0
            return

    # Walk positive direction: idx origin..L-1, add vectors
    cx = x
    cy = y
    for idx in range(origin, L):
        dx = vectors[i, idx, 0]
        dy = vectors[i, idx, 1]
        cx += dx
        cy += dy
        if cy < 0 or cy >= H or cx < 0 or cx >= W or board_free[cy, cx] == 0:
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

    def _ensure_buffers(self, N: int, max_len: int):
        # Deprecated placeholder retained for interface; no-op now
        return

    def solve(self, findAll: bool = False, printSol: bool = True):
        self._findAll = findAll
        self._print = printSol
        solutions: List[Board] = []
        self._startTime = datetime.now()
        solutions = self._solve(self._board, self._pieces, solutions)
        return solutions, self._nbTries, self._nbPcsPut

    def _solve(self, board: Board, pieces: List[Piece], solutions: List[Board]):
        nbPcs = len(pieces)
        if nbPcs:
            pos = board.nextAvailablePos()
            if pos is None:
                # No available pos but still pieces – dead end
                return solutions
            # Batch all candidates for this board position in a single kernel launch
            vectors, lengths, origins, meta = _encode_candidates(pieces, pos)

            N = lengths.shape[0]
            if N == 0:
                return solutions

            free = _board_to_free_array(board)
            pos_abs_x = board._origin.x + pos.x
            pos_abs_y = board._origin.y + pos.y

            self._nbTries += N

            h_vec = np.ascontiguousarray(vectors)
            h_len = np.ascontiguousarray(lengths)
            h_org = np.ascontiguousarray(origins)

            d_vectors = cuda.to_device(h_vec)
            d_lengths = cuda.to_device(h_len)
            d_origins = cuda.to_device(h_org)
            d_board = cuda.to_device(free)

            threads_per_block = 128
            blocks = (N + threads_per_block - 1) // threads_per_block
            _kernel_check_candidates[blocks, threads_per_block](
                d_board, pos_abs_x, pos_abs_y, d_vectors, d_lengths, d_origins, cuda.device_array(N, dtype=np.int32)
            )

            valids = cuda.device_array(N, dtype=np.int32)
            _kernel_check_candidates[blocks, threads_per_block](
                d_board, pos_abs_x, pos_abs_y, d_vectors, d_lengths, d_origins, valids
            )
            valids_host = valids.copy_to_host()

            for i, is_valid in enumerate(valids_host):
                if is_valid:
                    piece_i, origin_i, vecs_i = meta[i]
                    # Work on a dedicated copy for safety
                    piece_copy = deepcopy(piece_i)
                    piece_copy._currShape = vecs_i
                    piece_copy.setOrigin(origin_i)

                    newBoard = board.putPiece(piece_copy, pos)
                    if newBoard is not None:
                        self._nbPcsPut += 1
                        newPieces = deepcopy(pieces)
                        newPieces.remove(piece_i)
                        solutions = self._solve(newBoard, newPieces, solutions)

            # Top-level progress display like original solver (approximate, using candidate count)
            if nbPcs == self._nbPieces and not self._stop:
                execDuration = str(datetime.now() - self._startTime)
                if execDuration.rfind(".") != -1:
                    execDuration = execDuration[: execDuration.rfind(".")]
                pct = 100 * (self._nbTries / (nbPcs * max(len(p) for p in pieces) * len(Trans)))
                stdout.write(
                    "\r{0} - {1:.2f}% - {2} sol. over {3} pcs put with {4} tested combi.".format(
                        execDuration,
                        pct,
                        len(solutions),
                        self._nbPcsPut,
                        self._nbTries,
                    )
                )
                stdout.flush()

            if nbPcs == self._nbPieces:
                print("\n")
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
