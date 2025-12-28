#!/usr/bin/env python3
"""
Profile CUDA solver to identify bottlenecks.
Adds timing instrumentation to understand where time is spent.
"""

from datetime import datetime
import sys
import time

sys.path.insert(0, './py')

from puzzle import Vector, Piece, Board
from cuda_solver import PuzzleSolver


def GenerateBoard(date):
    board = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, None, None, None, None, None, None, 0, 0, 0, 0],
        [0, 0, 0, None, None, None, None, None, None, 0, 0, 0, 0],
        [0, 0, 0, None, None, None, None, None, None, None, 0, 0, 0],
        [0, 0, 0, None, None, None, None, None, None, None, 0, 0, 0],
        [0, 0, 0, None, None, None, None, None, None, None, 0, 0, 0],
        [0, 0, 0, None, None, None, None, None, None, None, 0, 0, 0],
        [0, 0, 0, None, None, None, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    dayNum = date.day - 1
    monthNum = date.month - 1
    board[int(monthNum / 6) + 3][(monthNum % 6) + 3] = date.strftime("%b")
    board[int(dayNum / 7) + 5][(dayNum % 7) + 3] = date.strftime("%d")
    return Board(board)


def CreatePieces():
    O = Piece(
        shape=[Vector(1, 0), Vector(1, 0), Vector(0, 1), Vector(-1, 0), Vector(-1, 0)],
        name="O"
    )
    t = Piece(
        shape=[Vector(1, 0), Vector(1, 0), Vector(1, 0), Vector(-1, 1)],
        name="t"
    )
    Q = Piece(
        shape=[Vector(0, 1), Vector(1, 0), Vector(0, 1), Vector(-1, 0)],
        name="Q"
    )
    BigS = Piece(
        shape=[Vector(1, 0), Vector(0, 1), Vector(0, 1), Vector(1, 0)],
        name="S",
        sides="both"
    )
    SmallsTail = Piece(
        shape=[Vector(1, 0), Vector(0, 1), Vector(1, 0), Vector(1, 0)],
        name="sl"
    )
    BigL = Piece(
        shape=[Vector(0, 1), Vector(1, 0), Vector(1, 0), Vector(1, 0)],
        name="L"
    )
    U = Piece(
        shape=[Vector(0, 1), Vector(1, 0), Vector(1, 0), Vector(0, -1)],
        name="U"
    )
    Lequal = Piece(
        shape=[Vector(0, 1), Vector(0, 1), Vector(1, 0), Vector(1, 0)],
        name="LL"
    )
    return [O, t, Q, BigS, SmallsTail, BigL, U, Lequal]


# Instrumented version to track timing
class ProfiledSolver(PuzzleSolver):
    def __init__(self, board, pieces):
        super().__init__(board, pieces)
        self.time_encode = 0.0
        self.time_bitmask = 0.0
        self.time_h2d = 0.0
        self.time_kernel = 0.0
        self.time_d2h = 0.0
        self.time_placement = 0.0
        self.kernel_calls = 0
        self.total_candidates = 0

    def _solve(self, board, pieces, solutions):
        nbPcs = len(pieces)
        if nbPcs:
            pos = board.nextAvailablePos()
            if pos is None:
                return solutions

            # Build list of candidate indices for current pieces
            piece_to_idx = {id(self._pieces[i]): i for i in range(len(self._pieces))}

            cand_indices = []
            for piece in pieces:
                piece_idx = piece_to_idx.get(id(piece))
                if piece_idx is not None:
                    start_idx, count = self._piece_ranges[piece_idx]
                    cand_indices.extend(range(start_idx, start_idx + count))

            N = len(cand_indices)
            if N == 0:
                return solutions

            # Time bitmask creation
            t0 = time.perf_counter()
            from cuda_solver import _board_to_bitmask
            mask, width, height = _board_to_bitmask(board)
            pos_abs_x = board._origin.x + pos.x
            pos_abs_y = board._origin.y + pos.y
            base_idx = pos_abs_y * width + pos_abs_x
            self.time_bitmask += time.perf_counter() - t0

            self._nbTries += N
            self.total_candidates += N

            # Time H2D transfer (only indices and mask - offsets already on GPU!)
            t0 = time.perf_counter()
            self._h_candidate_indices[:N] = cand_indices
            self._d_candidate_indices[:N].copy_to_device(self._h_candidate_indices[:N])
            self._d_mask_words.copy_to_device(mask)
            self.time_h2d += time.perf_counter() - t0

            # Time kernel execution
            threads_per_block = 512
            blocks = (N + threads_per_block - 1) // threads_per_block

            t0 = time.perf_counter()
            from cuda_solver import _kernel_check_candidates_preloaded
            _kernel_check_candidates_preloaded[blocks, threads_per_block](
                self._d_mask_words, width, height, base_idx,
                self._d_all_offsets,
                self._all_max_len,
                self._d_all_lengths,
                self._d_candidate_indices[:N],
                self._d_results[:N]
            )
            self.time_kernel += time.perf_counter() - t0
            self.kernel_calls += 1

            # Time D2H transfer
            t0 = time.perf_counter()
            results_host = self._d_results[:N].copy_to_host()
            self.time_d2h += time.perf_counter() - t0

            # Time placement (but NOT recursion - that's measured separately)
            for i, is_valid in enumerate(results_host):
                if is_valid:
                    cand_idx = cand_indices[i]
                    piece_idx, origin_i, vecs_i = self._all_meta[cand_idx]
                    piece_i = self._pieces[piece_idx]
                    from cuda_solver import PlacementPiece

                    t0 = time.perf_counter()
                    piece_view = PlacementPiece(piece_i.name, vecs_i)
                    piece_view.setOrigin(origin_i)
                    newBoard = board.putPiece(piece_view, pos)
                    self.time_placement += time.perf_counter() - t0

                    if newBoard is not None:
                        self._nbPcsPut += 1
                        t0 = time.perf_counter()
                        newPieces = pieces.copy()
                        try:
                            newPieces.remove(piece_i)
                        except ValueError:
                            newPieces = [p for p in newPieces if p.name != piece_i.name]
                        self.time_placement += time.perf_counter() - t0
                        solutions = self._solve(newBoard, newPieces, solutions)
        else:
            if self._print:
                print(f"\nSolution found after {self._nbTries} tries")
            solutions.append(board)
        return solutions


if __name__ == "__main__":
    date = datetime.strptime("28/12/2025", "%d/%m/%Y")
    prettyDate = date.strftime("%A, %d %B %Y")
    puzzle = GenerateBoard(date)
    pieces = CreatePieces()

    print(f"Profiling CUDA solver for {prettyDate}")
    print("=" * 60)
    print(f"Finding first solution...")

    t_start = time.perf_counter()
    solver = ProfiledSolver(puzzle, pieces)
    t_init = time.perf_counter() - t_start

    starttime = datetime.now()
    solutions, tries, nbPcsPut = solver.solve(findAll=False, printSol=False)
    total_time = (datetime.now() - starttime).total_seconds()

    print(f"\n{'='*60}")
    print(f"GPU initialization: {t_init:.3f}s (preloading candidates)")
    print(f"Total solutions: {len(solutions)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total tries: {tries:,}")
    print(f"Pieces placed: {nbPcsPut:,}")
    print(f"\n{'='*60}")
    print("Time breakdown:")
    print(f"  Creating bitmasks:    {solver.time_bitmask:7.3f}s ({100*solver.time_bitmask/total_time:5.1f}%)")
    print(f"  Host-to-Device copy:  {solver.time_h2d:7.3f}s ({100*solver.time_h2d/total_time:5.1f}%)")
    print(f"  Kernel execution:     {solver.time_kernel:7.3f}s ({100*solver.time_kernel/total_time:5.1f}%)")
    print(f"  Device-to-Host copy:  {solver.time_d2h:7.3f}s ({100*solver.time_d2h/total_time:5.1f}%)")
    print(f"  Placement & recursion:{solver.time_placement:7.3f}s ({100*solver.time_placement/total_time:5.1f}%)")

    measured = (solver.time_encode + solver.time_bitmask + solver.time_h2d +
                solver.time_kernel + solver.time_d2h + solver.time_placement)
    overhead = total_time - measured
    print(f"  Other overhead:       {overhead:7.3f}s ({100*overhead/total_time:5.1f}%)")

    print(f"\n{'='*60}")
    print(f"Kernel statistics:")
    print(f"  Total kernel calls:   {solver.kernel_calls:,}")
    print(f"  Total candidates:     {solver.total_candidates:,}")
    print(f"  Avg candidates/call:  {solver.total_candidates/solver.kernel_calls:.1f}")
    print(f"  Avg time per kernel:  {1000*solver.time_kernel/solver.kernel_calls:.3f}ms")
    print(f"{'='*60}")
