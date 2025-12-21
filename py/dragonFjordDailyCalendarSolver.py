from puzzle import Vector, Piece, Board
from solver import PuzzleSolver
from datetime import datetime
import argparse
import sys


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
    O = Piece(  # noqa: E741
        shape=[Vector(1, 0), Vector(1, 0), Vector(0, 1), Vector(-1, 0), Vector(-1, 0)],
        name="O"
    )  # 6 squares rectangle
    t = Piece(
        shape=[Vector(1, 0), Vector(1, 0), Vector(1, 0), Vector(-1, 1)],
        name="t"
    )  # 5 squares small t shaped
    Q = Piece(
        shape=[Vector(0, 1), Vector(1, 0), Vector(0, 1), Vector(-1, 0)],
        name="Q"
    )  # 5 squares in square with teeth
    BigS = Piece(
        shape=[Vector(1, 0), Vector(0, 1), Vector(0, 1), Vector(1, 0)],
        name="S",
        sides="both"
    )  # big S shaped
    SmallsTail = Piece(
        shape=[Vector(1, 0), Vector(0, 1), Vector(1, 0), Vector(1, 0)],
        name="sl"
    )  # 5 squares small S with long tail
    BigL = Piece(
        shape=[Vector(0, 1), Vector(1, 0), Vector(1, 0), Vector(1, 0)],
        name="L"
    )  # big L
    U = Piece(
        shape=[Vector(0, 1), Vector(1, 0), Vector(1, 0), Vector(0, -1)],
        name="U"
    )  # U shaped 5 squares - symmetric
    Lequal = Piece(
        shape=[Vector(0, 1), Vector(0, 1), Vector(1, 0), Vector(1, 0)],
        name="LL"
    )  # L with equal length arms - symmetric
    return [O, t, Q, BigS, SmallsTail, BigL, U, Lequal]


def PrintPieces(pieces):
    """
    Print detailed information about puzzle pieces with visual representation.

    Args:
        pieces: List of Piece objects to print
    """
    print("\n" + "=" * 60)
    print("Dragon Fjord Puzzle Pieces")
    print("=" * 60)

    for i, piece in enumerate(pieces, 1):
        print(f"\nPiece {i}: {piece.name}")
        print(f"  Number of squares: {len(piece.shape) + 1}")
        if hasattr(piece, 'sides') and piece.sides:
            print(f"  Sides: {piece.sides}")

        # Draw piece graphically
        print("  Shape:")
        piece.draw()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Dragon Fjord Calendar Puzzle Solver"
    )
    parser.add_argument(
        "--print-pieces",
        action="store_true",
        help="Print puzzle pieces information before solving"
    )
    args = parser.parse_args()

    pieces = CreatePieces()

    # Print pieces if flag is set and exit
    if args.print_pieces:
        PrintPieces(pieces)
        print("\n" + "=" * 60 + "\n")
        sys.exit(0)

    userDate = input(
        "Calendar puzzle date to solve (ex: 31/01/2022), leave empty for today's date): "
    )

    if len(userDate) == 0:
        userDate = datetime.now().strftime("%d/%m/%Y")
        print("Solving current date {}".format(userDate))
    try:
        date = datetime.strptime(userDate, "%d/%m/%Y")
    except ValueError:
        print("The date '{}' is not in the expected format dd/mm/yyyy".format(userDate))
    else:
        prettyDate = date.strftime("%A, %d %B %Y")
        puzzle = GenerateBoard(date)
        solver = PuzzleSolver(puzzle, pieces)
        print("Start solving puzzle for {}".format(prettyDate))
        starttime = datetime.now()
        solutions, tries, nbPcsPut = solver.solve(
            findAll=True, printSol=True
        )
        print(
            "{} solutions found for {} in {} after {} tries and placing {} pieces".format(
                len(solutions), prettyDate, datetime.now() - starttime, tries, nbPcsPut
            )
        )
        print("\n" + "=" * 60 + "\n")
