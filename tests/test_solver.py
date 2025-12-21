"""Tests for PuzzleSolver class."""
import pytest
from puzzle import Vector, Piece, Board
from solver import PuzzleSolver


class TestPuzzleSolver:
    """Test suite for PuzzleSolver."""

    def test_solve_with_two_pieces(self, small_board, simple_pieces):
        """Test solving puzzle with 2 pieces."""
        solver = PuzzleSolver(small_board, [simple_pieces['A'], simple_pieces['B']])
        solutions, nb_tries, nb_pcs_put = solver.solve(findAll=True, printSol=False)

        assert len(solutions) > 0, "Should find at least one solution"
        assert nb_tries > 0, "Number of tries should be positive"
        assert nb_pcs_put > 0, "Number of pieces placed should be positive"

    def test_solve_with_three_pieces_expects_eight_solutions(self, medium_board):
        """Test solving puzzle with 3 pieces expecting 8 solutions."""
        pieces = [
            Piece(shape=[Vector(0, 1), Vector(0, 1), Vector(1, -1)], name="A"),
            Piece(shape=[Vector(0, 1)], name="B"),
            Piece(shape=[Vector(0, 1), Vector(1, 0)], name="C"),
        ]

        solver = PuzzleSolver(medium_board, pieces)
        solutions, nb_tries, nb_pcs_put = solver.solve(findAll=True, printSol=False)

        assert len(solutions) == 8, f"Expected 8 solutions, got {len(solutions)}"
        assert nb_tries > 0
        assert nb_pcs_put > 0

    def test_solve_with_empty_piece(self, medium_board):
        """Test that solver handles empty pieces correctly."""
        pieces = [
            Piece(shape=[Vector(0, 1), Vector(0, 1), Vector(1, 0)], name="A"),
            Piece(shape=[Vector(0, 1), Vector(0, 1), Vector(1, 0)], name="B"),
            Piece(shape=[], name="Empty"),
        ]

        solver = PuzzleSolver(medium_board, pieces)
        solutions, _, _ = solver.solve(findAll=True, printSol=False)

        assert isinstance(solutions, list), "Solutions should be a list"
        assert len(solutions) > 0, "Should find at least one solution"
        assert all(isinstance(sol, Board) for sol in solutions), "All solutions should be Board objects"

    @pytest.mark.parametrize("find_all,expected_min_solutions", [
        (True, 1),
        (False, 1),
    ])
    def test_solve_find_all_parameter(self, small_board, simple_pieces, find_all, expected_min_solutions):
        """Test findAll parameter behavior."""
        solver = PuzzleSolver(small_board, [simple_pieces['A'], simple_pieces['B']])
        solutions, _, _ = solver.solve(findAll=find_all, printSol=False)

        assert len(solutions) >= expected_min_solutions, f"Should find at least {expected_min_solutions} solution(s)"

    def test_solver_returns_correct_tuple(self, small_board, simple_pieces):
        """Test that solve returns correct tuple structure."""
        solver = PuzzleSolver(small_board, [simple_pieces['A'], simple_pieces['B']])
        result = solver.solve(findAll=True, printSol=False)

        assert isinstance(result, tuple), "Result should be a tuple"
        assert len(result) == 3, "Result should have 3 elements"
        solutions, tries, pieces_put = result
        assert isinstance(solutions, list), "First element should be list of solutions"
        assert isinstance(tries, int), "Second element should be int (tries)"
        assert isinstance(pieces_put, int), "Third element should be int (pieces put)"

    def test_solve_with_print_enabled(self, small_board, simple_pieces, capsys):
        """Test solving with print output enabled."""
        solver = PuzzleSolver(small_board, [simple_pieces['A'], simple_pieces['B']])
        solutions, _, _ = solver.solve(findAll=True, printSol=True)

        captured = capsys.readouterr()
        assert len(solutions) > 0, "Should find solutions"
        assert "sol." in captured.out or len(captured.out) > 0, "Should print output"