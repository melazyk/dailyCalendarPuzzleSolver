"""Shared test fixtures and configuration."""
import pytest
from puzzle import Vector, Piece, Board


@pytest.fixture
def simple_pieces():
    """Create simple test pieces."""
    return {
        'A': Piece(shape=[Vector(0, 1), Vector(0, 1), Vector(1, 0)], name="A"),
        'B': Piece(shape=[Vector(0, 1)], name="B"),
        'C': Piece(shape=[Vector(0, 1), Vector(1, 0)], name="C"),
    }


@pytest.fixture
def small_board():
    """Create a small test board."""
    return Board([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, None, None, 0, 0],
        [0, 0, None, None, 0, 0],
        [0, 0, None, None, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ])


@pytest.fixture
def medium_board():
    """Create a medium test board."""
    return Board([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, None, None, None, 0, 0],
        [0, 0, None, None, None, 0, 0],
        [0, 0, None, None, None, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])


@pytest.fixture
def calendar_pieces():
    """Create standard calendar puzzle pieces."""
    return [
        Piece(shape=[Vector(0, 1), Vector(0, 1), Vector(0, 1)], name="I"),
        Piece(shape=[Vector(0, 1), Vector(1, 0), Vector(0, 1)], name="s"),
        Piece(shape=[Vector(0, 1), Vector(1, 0), Vector(1, 0)], name="Ls"),
        Piece(shape=[Vector(1, 0), Vector(1, 0), Vector(-1, 1), Vector(0, 1)], name="T"),
        Piece(shape=[Vector(0, 1), Vector(1, 0), Vector(0, 1), Vector(-1, 0)], name="Q"),
        Piece(shape=[Vector(1, 0), Vector(0, 1), Vector(0, 1), Vector(1, 0)], name="S"),
        Piece(shape=[Vector(1, 0), Vector(0, 1), Vector(1, 0), Vector(1, 0)], name="sl"),
        Piece(shape=[Vector(0, 1), Vector(1, 0), Vector(1, 0), Vector(1, 0)], name="L"),
        Piece(shape=[Vector(0, 1), Vector(1, 0), Vector(1, 0), Vector(0, -1)], name="U"),
        Piece(shape=[Vector(0, 1), Vector(0, 1), Vector(1, 0), Vector(1, 0)], name="LL"),
    ]
