"""Test package for dailyCalendarPuzzleSolver."""
import sys
from pathlib import Path

# Add py directory to Python path
py_dir = Path(__file__).parent.parent / "py"
if str(py_dir) not in sys.path:
    sys.path.insert(0, str(py_dir))
