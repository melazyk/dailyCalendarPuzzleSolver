"""Tests for MultiThreadPuzzleSolver class."""
import pytest
from datetime import datetime
from puzzle import Vector, Piece, Board
from multithreadssolver import MultiThreadPuzzleSolver


class TestMultiThreadPuzzleSolver:
    """Test suite for MultiThreadPuzzleSolver."""

    def test_multithread_solve_basic(self, small_board, simple_pieces):
        """Test basic multi-threaded solving."""
        solver = MultiThreadPuzzleSolver(small_board, [simple_pieces['A'], simple_pieces['B']])
        solutions, tries, nb_pcs_put = solver.solve(findAll=True, printSol=False)

        assert len(solutions) > 0, f"Expected at least one solution, got {len(solutions)}"
        assert tries > 0, "Should have made attempts"
        assert nb_pcs_put > 0, "Should have placed pieces"

    def test_multithread_solve_with_sides_parameter(self, small_board, simple_pieces):
        """Test multi-threaded solving with sides parameter."""
        # Test with front side only
        solver_front = MultiThreadPuzzleSolver(small_board, [simple_pieces['A'], simple_pieces['B']])
        solutions_front, _, _ = solver_front.solve(findAll=True, printSol=False, sides="front")

        # Test with both sides
        solver_both = MultiThreadPuzzleSolver(small_board, [simple_pieces['A'], simple_pieces['B']])
        solutions_both, _, _ = solver_both.solve(findAll=True, printSol=False, sides="both")

        assert len(solutions_front) > 0, "Should find solutions with front side"
        assert len(solutions_both) >= len(solutions_front), "Both sides should find at least as many solutions as front only"

    @pytest.mark.slow
    def test_solve_calendar_puzzle(self, calendar_pieces):
        """Test calendar puzzle solving (slower test)."""
        date = datetime.strptime("06/05/2022", "%d/%m/%Y")
        puzzle = self._generate_board(date)

        solver = MultiThreadPuzzleSolver(puzzle, calendar_pieces)
        solutions, tries, pieces_put = solver.solve(
            findAll=True, sides="front", printSol=False
        )

        assert len(solutions) > 0, "Should find at least one solution"
        assert tries > 0
        assert pieces_put > 0

    def test_create_calendar_pieces(self):
        """Test creation of calendar puzzle pieces."""
        pieces = [
            Piece(shape=[Vector(0, 1), Vector(0, 1), Vector(0, 1)], name="I"),
            Piece(shape=[Vector(0, 1), Vector(1, 0), Vector(0, 1)], name="s"),
        ]

        assert len(pieces) > 0, "Should create at least one piece"
        assert all(isinstance(p, Piece) for p in pieces), "All should be Piece objects"
        assert all(p.name for p in pieces), "All pieces should have names"

    def test_generate_board_for_date(self):
        """Test board generation for specific dates."""
        test_date = datetime(2024, 1, 15)  # Jan 15, 2024 (Monday)
        board = self._generate_board(test_date)

        assert isinstance(board, Board), "Should return a Board object"

    @pytest.mark.slow
    @pytest.mark.integration
    def test_full_calendar_puzzle_with_output(self, calendar_pieces, capsys):
        """Full integration test for calendar puzzle with detailed output."""
        date = datetime.strptime("06/05/2022", "%d/%m/%Y")
        pretty_date = date.strftime("%A, %d %B %Y")
        puzzle = self._generate_board(date)

        solver = MultiThreadPuzzleSolver(puzzle, calendar_pieces)
        start_time = datetime.now()
        solutions, tries, nb_pcs_put = solver.solve(findAll=True, sides="front", printSol=False)
        duration = datetime.now() - start_time

        assert len(solutions) > 0, f"Should find at least one solution for {pretty_date}"
        assert tries > 0
        assert nb_pcs_put > 0

        # Verify the solution fits the date
        print(f"\n{len(solutions)} solutions found for {pretty_date}")
        print(f"Duration: {duration}")
        print(f"Tries: {tries}, Pieces put: {nb_pcs_put}")

    @staticmethod
    def _generate_board(date):
        """Helper method to generate board for a specific date."""
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
            [0, 0, 0, None, None, None, None, None, None, None, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, None, None, None, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]

        week_day_num = date.weekday()
        day_num = date.day - 1
        month_num = date.month - 1

        board[int(month_num / 6) + 3][(month_num % 6) + 3] = date.strftime("%b")
        board[int(day_num / 7) + 5][(day_num % 7) + 3] = date.strftime("%d")

        if week_day_num == 6:
            board[9][6] = date.strftime("%a")
        else:
            board[int(week_day_num / 3) + 9][(week_day_num % 3) + 7] = date.strftime("%a")

        return Board(board)
