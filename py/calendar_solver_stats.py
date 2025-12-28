#!/usr/bin/env python3
"""
Generate statistics showing the number of solutions for each day of the year.
Outputs a CSV file with date and solution count.
"""

from datetime import timedelta
import argparse
import time
import csv
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

from numba.core.errors import NumbaPerformanceWarning
from puzzle import Vector, Piece, Board
from datetime import datetime as dt

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


def create_pieces():
    """Create puzzle pieces (same as in dragonFjordDailyCalendarSolver.py)."""
    O = Piece(  # noqa: E741
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
        sides="back"
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


def generate_board(date):
    """Generate board for a given date (same as in dragonFjordDailyCalendarSolver.py)."""
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


def count_solutions(date, pieces, use_cuda=True):
    """Count the number of solutions for a given date."""
    try:
        puzzle = generate_board(date)

        if use_cuda:
            from cuda_solver import PuzzleSolver as CudaPuzzleSolver
            solver = CudaPuzzleSolver(puzzle, pieces)
        else:
            from solver import PuzzleSolver as CPUSolver
            solver = CPUSolver(puzzle, pieces)

        solutions, _, _ = solver.solve(findAll=True, printSol=False, printProgress=False)
        return len(solutions)
    except Exception:
        return -1


def count_solutions_with_date(args):
    """Wrapper for parallel processing."""
    date, pieces, use_cuda = args
    return date, count_solutions(date, pieces, use_cuda)


def generate_csv(solutions_by_date, output_file):
    """Generate a CSV file with date and solution count."""
    # Sort dates
    sorted_dates = sorted(solutions_by_date.keys())

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'Solutions'])

        for date in sorted_dates:
            count = solutions_by_date[date]
            writer.writerow([date.strftime('%Y-%m-%d'), count])

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate a CSV file with solutions per day"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Year to calculate (default: 2025)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="calendar_solutions.csv",
        help="Output CSV file (default: calendar_solutions.csv)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU solver instead of CUDA (CUDA is default)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date to calculate (format: MM-DD), if not provided starts from Jan 1"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date to calculate (format: MM-DD), if not provided goes to Dec 31"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of parallel threads for processing (default: 4)"
    )

    args = parser.parse_args()

    pieces = create_pieces()
    solutions_by_date = {}

    # Determine date range
    if args.start_date:
        parts = args.start_date.split('-')
        start_date = dt(args.year, int(parts[0]), int(parts[1]))
    else:
        start_date = dt(args.year, 1, 1)

    if args.end_date:
        parts = args.end_date.split('-')
        end_date = dt(args.year, int(parts[0]), int(parts[1]))
    else:
        end_date = dt(args.year, 12, 31)

    current_date = start_date
    total_days = (end_date - start_date).days + 1

    print(f"Calculating solutions for {total_days} days ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})...")
    print(f"Using {'CPU' if args.cpu else 'CUDA'} solver with {args.threads} parallel threads")
    print()

    # Build list of dates to process
    dates_to_process = []
    current_date = start_date
    while current_date <= end_date:
        dates_to_process.append(current_date)
        current_date += timedelta(days=1)

    # Parallel processing with ProcessPoolExecutor
    start_time = time.time()
    day_num = 0

    with ProcessPoolExecutor(max_workers=args.threads) as executor:
        # Submit all tasks
        futures = {
            executor.submit(count_solutions_with_date, (date, pieces, not args.cpu)): date
            for date in dates_to_process
        }

        # Process results as they complete
        print()  # Newline for progress bar
        for future in as_completed(futures):
            day_num += 1
            elapsed = time.time() - start_time
            date, count = future.result()
            solutions_by_date[date] = count

            progress = (day_num / total_days) * 100

            # Calculate estimated time remaining
            if day_num > 0:
                avg_time_per_day = elapsed / day_num
                remaining_days = total_days - day_num
                eta_seconds = avg_time_per_day * remaining_days
                eta_str = f"{int(eta_seconds // 3600):02d}:{int((eta_seconds % 3600) // 60):02d}:{int(eta_seconds % 60):02d}"
            else:
                eta_str = "--:--:--"

            elapsed_str = f"{int(elapsed // 3600):02d}:{int((elapsed % 3600) // 60):02d}:{int(elapsed % 60):02d}"

            # Progress bar
            bar_length = 40
            filled = int(bar_length * day_num / total_days)
            bar = "█" * filled + "░" * (bar_length - filled)

            # Print progress bar on same line (overwrite)
            print(
                f"\r[{bar}] {day_num:3d}/{total_days} ({progress:5.1f}%) | Elapsed: {elapsed_str} | ETA: {eta_str}",
                end='',
                flush=True,
            )
            # Print last completed date below, clearly labeled as the most recent result
            print(
                f"\nLast completed: {date.strftime('%Y-%m-%d')} ({count:3d} solutions)",
                end='',
                flush=True,
            )
            # Move cursor up one line for next update (skip on final iteration)
            if day_num != total_days:
                print("\033[F", end='', flush=True)

    total_time = time.time() - start_time
    total_str = f"{int(total_time // 3600):02d}:{int((total_time % 3600) // 60):02d}:{int(total_time % 60):02d}"
    print()  # finish progress line
    print(f"\n✓ Completed all {total_days} days in {total_str}")

    # Generate CSV
    output_path = generate_csv(solutions_by_date, args.output)
    print(f"✓ CSV file saved to: {output_path}")


if __name__ == "__main__":
    main()
