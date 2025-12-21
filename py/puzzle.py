from enum import Enum
from copy import deepcopy


class Coordinate:
    """
    Class representing a coordinate on the puzzle board
    """

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __eq__(self, other):
        """
        required to be able to use == operator on Coordinate object
        """
        return other is not None and self.x == other.x and self.y == other.y

    def move(self, vector):
        self.x += vector.x
        self.y += vector.y

    def __repr__(self):
        return f"(x={self.x},y={self.y})"


class Vector(Coordinate):
    """
    Class representing a vector, and used to define a puzzle piece
    """

    def __init__(self, x=0, y=0):
        super().__init__(x, y)


class Trans(Enum):
    """
    Class representing all physically possible puzzle pieces transformations
    There are 4 orientations with original piece side,
    and 4 more when the piece is flipped upside down
    """

    UpFront = 1
    RightFront = 2
    DownFront = 3
    LeftFront = 4
    UpBack = 5
    RightBack = 6
    DownBack = 7
    LeftBack = 8

    def isFront(self):
        return self.value < 5

    def isBack(self):
        return self.value > 4


class Piece:
    """
    Class used to represent a puzzle piece
    A puzzle pieces is coded with a list of Vector, a name and a list of relevant transformations
    The list of Vector represent the necessary moves to run through all the squares of the piece
    For example, this piece:      O                 ^ y axis
                                OOO    ---> x axis  |
    will be represented by 3 Vectors : two x=1,y=0 for the bottom part,
    and one x=0,y=1 for the leg of this L shaped piece
    Of course, a piece of only 1 square will be represented with an empty Vector list
    The name of the piece is used to represent it on the puzzle board,
    in the example above this is "O"
    """

    def __init__(self, shape, name, sides="both"):
        self._baseShape = shape
        self._currShape = shape
        self.name = name
        self._origin = 0

        # Automatically detect symmetry and optimize
        if self._isSymmetric():
            self.sides = "front"
        else:
            self.sides = sides

        if len(self._baseShape):
            self._relevantTrans = self._listRelevantTransform()
        else:
            self._relevantTrans = [Trans.UpFront]

    @property
    def shape(self):
        """
        Current shape vectors of the piece (list of Vector).
        Exposed as a property for convenience in UIs/printing.
        """
        return self._currShape

    def _isSymmetric(self):
        """
        Checks if the piece is symmetric (front and back sides are identical)
        """
        frontShapes = set()
        backShapes = set()

        for trans in Trans:
            transformed = tuple((v.x, v.y) for v in self._transform(trans))
            if trans.isFront():
                frontShapes.add(transformed)
            else:
                backShapes.add(transformed)

        # If all back transformations exist in front - piece is symmetric
        return backShapes.issubset(frontShapes)

    def __repr__(self):
        return (
            f"(base={self._baseShape}\n"
            f"current={self._currShape}\n"
            f"name={self.name}\n"
            f"origin=({self._origin})\n"
            f"relevantTrans={self._relevantTrans})"
        )

    def __len__(self):
        # Add one because a single square piece has no vector
        return len(self._currShape) + 1

    def __getitem__(self, idx):
        """
        required to be able to use [] operator on Piece object
        """
        ret = None
        if idx > 0:
            idx -= 1
        if (idx + self._origin) < len(self._currShape) and (idx + self._origin) >= 0:
            ret = self._currShape[idx + self._origin]
        return ret

    def __eq__(self, other):
        """
        required to be able to use remove() operator on list of Piece objects
        As two pieces with the same name could not be distinguished once on a board,
        they are considered as the same as soon as they have the same name
        """
        return self.name == other.name

    def setOrigin(self, origin):
        if origin >= 0 and origin <= len(self._currShape):
            self._origin = origin

    def relevantTrans(self):
        return self._relevantTrans

    def _listRelevantTransform(self):
        """
        Find all unique transformations of the piece (removes duplicates from symmetry).
        Returns list of Trans enum values that produce unique orientations.
        """
        seen_frames = []
        relevant_transforms = []

        for trans in Trans:
            frame = self._get_piece_frame(trans)
            if frame not in seen_frames:
                seen_frames.append(frame)
                relevant_transforms.append(trans)

        return relevant_transforms

    def _get_piece_frame(self, trans):
        """
        Convert transformed piece to a normalized 2D grid representation.
        This allows comparison to detect duplicate orientations.
        """
        transformed = self._transform(trans)

        # Find bounding box
        min_x = min_y = max_x = max_y = 0
        pos = Coordinate()

        for vector in transformed:
            pos.move(vector)
            min_x = min(min_x, pos.x)
            max_x = max(max_x, pos.x)
            min_y = min(min_y, pos.y)
            max_y = max(max_y, pos.y)

        # Create normalized grid
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        frame = [[0 for _ in range(width)] for _ in range(height)]

        # Mark occupied cells
        pos = Coordinate(-min_x, -min_y)
        frame[pos.y][pos.x] = 1

        for vector in transformed:
            pos.move(vector)
            frame[pos.y][pos.x] = 1

        return frame

    def transform(self, transformation):
        self._currShape = self._transform(transformation)

    def _transform(self, transformation):
        """
        Apply a transformation to the base shape.
        Each transformation maps (x, y) to a new coordinate based on rotation/flip.
        """
        # Transformation rules: input coord -> output coord
        transform_map = {
            Trans.UpFront: lambda c: Vector(c.x, c.y),  # no change
            Trans.RightFront: lambda c: Vector(c.y, -c.x),  # rotate 90° CW
            Trans.DownFront: lambda c: Vector(-c.x, -c.y),  # rotate 180°
            Trans.LeftFront: lambda c: Vector(-c.y, c.x),  # rotate 90° CCW
            Trans.UpBack: lambda c: Vector(-c.x, c.y),  # flip horizontal
            Trans.RightBack: lambda c: Vector(c.y, c.x),  # flip diagonal
            Trans.DownBack: lambda c: Vector(c.x, -c.y),  # flip vertical
            Trans.LeftBack: lambda c: Vector(-c.y, -c.x),  # flip anti-diagonal
        }

        transform_func = transform_map[transformation]
        return [transform_func(coord) for coord in self._baseShape]

    def draw(self, indent: str = "    "):
        """
        Print an ASCII rendering of the current piece orientation using
        the same border style as Board.__repr__: vertical '|' and horizontal '_'.
        """
        # Handle single-square piece
        if len(self._currShape) == 0:
            print(f"{indent}||\n{indent}||")
            return

        # Enumerate occupied coordinates of the piece
        coords = [(0, 0)]
        x, y = 0, 0
        for vect in self._currShape:
            x += vect.x
            y += vect.y
            coords.append((x, y))

        min_x = min(c[0] for c in coords)
        max_x = max(c[0] for c in coords)
        min_y = min(c[1] for c in coords)
        max_y = max(c[1] for c in coords)

        # Normalize to start at (0,0)
        norm = [(cx - min_x, cy - min_y) for (cx, cy) in coords]
        width = max_x - min_x + 1
        height = max_y - min_y + 1

        # Build padded occupancy grid (height+2) x (width+2)
        grid = [[0 for _ in range(width + 2)] for __ in range(height + 2)]
        for cx, cy in norm:
            grid[cy + 1][cx + 1] = 1

        # Draw borders like Board.__repr__
        for y in range(0, height + 1):
            line = indent
            for x in range(1, width + 2):
                c = " "
                if grid[y][x] != grid[y][x - 1]:
                    c = "|"
                line += c
                c = " "
                if grid[y][x] != grid[y + 1][x]:
                    c = "_"
                line += c
            print(line)


class Board:
    """
    This class represent the board of the puzzle
    A board is initialized by a two dimensions array (a list of lists)
    This array shall contains "None" on any available square, and 0
    on squares where no pieces can be put
    As the puzzle solver will try to put pieces on boad borders, the array
    provided to build this board shall have enough unavailable square all
    around to prevent the solver from trying to put a piece square outside
    the board array. The number of unavailable squares around the "None"
    filled cells of the board array depend on the length of the bigger
    piece of the piece list.
    Any shape of board is possible, providing that the "None" cells
    are surrounded by enough not availables cells (0 filled)
    to create a 2 dimensions array.
    The system of corrdinates of the puzzle is the following:
    board parameter of the contructor is a list of list such as it is a list
    of X axis lines. The coordinate x=0,y=0 correspond to the top left corner
    of the puzzle, and positive X are growing to the right, and positive Y are
    growing down, e.g
    board =  [ [x=0 y=0, x=1 y=0],
               [x=0 y=1, x=1 y=1] ]
    Due to the, the internal _board of this class is used with Y coordinates first
    """

    def __init__(self, board):
        if isinstance(board, Board):
            # Copy constructor
            self._board = deepcopy(board._board)
            self._origin = board._origin
        else:
            self._board = board
            self._origin = self._getOrigin()

    def _getOrigin(self):
        xMin = len(self._board[0])
        yMin = len(self._board)
        for y in range(len(self._board)):
            for x in range(len(self._board[y])):
                if self._board[y][x] != 0:
                    if x < xMin:
                        xMin = x
                    if y < yMin:
                        yMin = y
        return Coordinate(xMin, yMin)

    def __repr__(self):
        ret = "\n"
        xMax = 0
        xMin = len(self._board[0])
        yMax = 0
        yMin = len(self._board)
        for y in range(len(self._board)):
            for x in range(len(self._board[y])):
                if self._board[y][x] != 0:
                    if x < xMin:
                        xMin = x
                    if x > xMax:
                        xMax = x
                    if y < yMin:
                        yMin = y
                    if y > yMax:
                        yMax = y
        for y in range(len(self._board)):
            for x in range(len(self._board[y])):
                if (
                    x >= (xMin - 1)
                    and x <= (xMax + 1)
                    and y >= (yMin - 1)
                    and y <= yMax
                ):
                    c = " "
                    if self._board[y][x] != self._board[y][x - 1]:
                        c = "|"
                    ret += c
                    c = " "
                    if self._board[y][x] != self._board[y + 1][x]:
                        c = "_"
                    ret += c
            if y >= (yMin - 1) and y <= yMax:
                ret += "\n"
        return ret

    def _putPieceSquare(self, name, pos):
        if self._board[self._origin.y + pos.y][self._origin.x + pos.x] is None:
            self._board[self._origin.y + pos.y][self._origin.x + pos.x] = name
            return self
        else:
            return None

    def putPiece(self, piece, pos):
        newBoard = Board(self)
        newBoard = newBoard._putPieceSquare(piece.name, pos)
        if newBoard is None:
            return None
        idx = -1
        currPos = pos
        vect = piece[idx]
        while vect is not None:
            nextPos = Coordinate(currPos.x - vect.x, currPos.y - vect.y)
            newBoard = newBoard._putPieceSquare(piece.name, nextPos)
            if newBoard is None:
                return None
            currPos = nextPos
            idx -= 1
            vect = piece[idx]
        idx = 1
        currPos = pos
        vect = piece[idx]
        while vect is not None:
            nextPos = Coordinate(currPos.x + vect.x, currPos.y + vect.y)
            newBoard = newBoard._putPieceSquare(piece.name, nextPos)
            if newBoard is None:
                return None
            currPos = nextPos
            idx += 1
            vect = piece[idx]
        return newBoard

    def nextAvailablePos(self):
        ret = None
        x = self._origin.x
        y = self._origin.y
        while y < len(self._board) and ret is None:
            while x < len(self._board[y]) and ret is None:
                pos = self._board[y][x]
                if pos is None:
                    ret = Coordinate(x - self._origin.x, y - self._origin.y)
                x += 1
            x = 0
            y += 1

        return ret
