#ifndef PUZZLE_COMMON_H
#define PUZZLE_COMMON_H

#include <iostream>
#include <string>

using namespace std;

enum Trans { up, right, down, left, upBack, leftBack, downBack, rightBack};

// ============================================================================
// Transformation arrays (commonly used combinations)
// ============================================================================
extern Trans allTrans[8];
extern Trans allFaceTrans[4];
extern Trans upRightTrans[4];

// ============================================================================
// Coord - Puzzle board coordinates
// ============================================================================
class Coord
{
    public:
        int x;
        int y;

        Coord(int cx, int cy) { x = cx; y = cy; }
        Coord() { x = 0; y = 0; }
        virtual ~Coord() {}
        void AssignFrom(const Coord& other) { x = other.x; y = other.y; }
        Coord(const Coord& other) { AssignFrom(other); }
        Coord& operator=(const Coord& other) { AssignFrom(other); return *this; }
        bool operator==(const Coord& other) {
            if((x == other.x) && (y == other.y)) return true;
            else return false;
        }
        virtual ostream& put(ostream & o, char sep=' ') const {
            return o << "(x=" << x << "y=" << y << ")" << sep;
        }

        friend ostream& operator<<(ostream& o, const Coord& p) { return p.put(o); }
};

// ============================================================================
// Vect - Vector used for puzzle piece definition
// ============================================================================
class Vect: public Coord
{
    public:
        Vect(int vx, int vy) : Coord(vx, vy) {}
        Vect() : Coord() {}

        inline bool isNull() {
            if(x == 0 && y == 0) return true;
            else return false;
        }
};

// ============================================================================
// Piece - Puzzle piece class
// ============================================================================
class Piece
{
    protected:
        Vect * baseShape;
        Vect * currShape;

    public:
        int shapeLength;
        int origin;
        unsigned char value;
        Trans* relevantTrans;
        int nbRelevantTrans;

        Piece(Vect shape[], int shapeLen, unsigned char val, Trans relTrans[], int nbRelTrans);
        virtual ~Piece();
        void transform(Trans transformation);
        Vect operator[](int index);
        virtual ostream& put(ostream & o, char sep=' ') const;
        friend ostream& operator<<(ostream& o, const Piece& p) { return p.put(o); }
};

inline Piece::Piece(Vect shape[], int shapeLen, unsigned char val, Trans relTrans[], int nbRelTrans)
{
    int i;
    shapeLength = shapeLen;
    baseShape = new Vect[shapeLength];
    currShape = new Vect[shapeLength];
    for(i = 0; i < shapeLength; i++) {
        baseShape[i] = shape[i];
        currShape[i] = shape[i];
    }
    origin = 0;
    value = val;
    nbRelevantTrans = nbRelTrans;
    relevantTrans = new Trans[nbRelTrans];
    for(i = 0; i < nbRelevantTrans; i++) {
        relevantTrans[i] = relTrans[i];
    }
}

inline Piece::~Piece()
{
    if(baseShape) {
        delete[] baseShape;
        baseShape = NULL;
    }
    if(currShape) {
        delete[] currShape;
        currShape = NULL;
    }
    if(relevantTrans) {
        delete[] relevantTrans;
        relevantTrans = NULL;
    }
}

inline Vect Piece::operator[](int index)
{
    int actualIdx = index + origin;
    if(index > 0) actualIdx -= 1;
    if( (actualIdx >= 0) && (actualIdx < shapeLength) )
        return currShape[actualIdx];
    else
        return Vect();
}

inline void Piece::transform(Trans transformation)
{
    for(int i = 0; i < shapeLength; i++) {
        switch(transformation) {
            case Trans::up :
                currShape[i] = baseShape[i];
                break;
            case Trans::right :
                currShape[i].x = baseShape[i].y;
                currShape[i].y = -baseShape[i].x;
                break;
            case Trans::down :
                currShape[i].x = -baseShape[i].x;
                currShape[i].y = -baseShape[i].y;
                break;
            case Trans::left :
                currShape[i].x = -baseShape[i].y;
                currShape[i].y = baseShape[i].x;
                break;
            case Trans::upBack :
                currShape[i].x = -baseShape[i].x;
                currShape[i].y = baseShape[i].y;
                break;
            case Trans::rightBack :
                currShape[i].x = baseShape[i].y;
                currShape[i].y = baseShape[i].x;
                break;
            case Trans::downBack :
                currShape[i].x = baseShape[i].x;
                currShape[i].y = -baseShape[i].y;
                break;
            case Trans::leftBack :
                currShape[i].x = -baseShape[i].y;
                currShape[i].y = -baseShape[i].x;
                break;
            default:
                cout << "Invalid transformation " << transformation << " for piece: " << to_string(value) << endl;
        }
    }
}

inline ostream & Piece::put(ostream & o, char sep) const
{
    return o << "(Piece " << to_string(value) << ")" << sep;
}

// ============================================================================
// BoardBase - generic board operations (CRTP) parametrized by dimensions
// ============================================================================
template <typename Derived, int BXL, int BYL, int BOX, int BOY, int BDXL, int BDYL>
class BoardBase
{
    protected:
        int boardArray[BYL][BXL];
        Coord arrayOrigin;

    public:
        Derived * next;

        BoardBase() : next(NULL) { arrayOrigin.x = BOX; arrayOrigin.y = BOY; }
        BoardBase(const BoardBase& other) { AssignFrom(other); }
        BoardBase& operator=(const BoardBase& other) { AssignFrom(other); return *this; }
        virtual ~BoardBase() {}

        void nextAvailablePos(Coord * pos)
        {
            int y = arrayOrigin.y;
            int x;
            bool found = false;
            while(y < (BDYL + arrayOrigin.y) && (found == false)) {
                x = arrayOrigin.x;
                while(x < (BDXL + arrayOrigin.x) && (found == false)) {
                    if(boardArray[y][x] == 0) {
                        pos->y = y - arrayOrigin.y;
                        pos->x = x - arrayOrigin.x;
                        found = true;
                    }
                    x++;
                }
                y++;
            }
        }

        void print()
        {
            int y, x;
            bool firstY, firstX;
            cout << "[";
            firstY = true;
            for(y = arrayOrigin.y; y < (BDYL + arrayOrigin.y); y++) {
                if(firstY == true) {
                    cout << "[";
                    firstY = false;
                } else {
                    cout << ",[";
                }
                firstX = true;
                for(x = arrayOrigin.x; x < (BDXL + arrayOrigin.x); x++) {
                    if(firstX == false) cout << ",";
                    else firstX = false;
                    cout << to_string(boardArray[y][x]);
                }
                cout << "]";
            }
            cout << "]";
        }

        Derived * putPiece(Piece &piece, Coord pos)
        {
            Derived * newBoard = new Derived(static_cast<Derived&>(*this));
            bool success;
            Coord currPos;
            int index = -1;
            Vect currVect;
            currPos = pos;
            success = putPieceSquare(*newBoard, piece.value, currPos);
            if(success) {
                currVect = piece[index];
                while(!currVect.isNull()) {
                    currPos.x = currPos.x - currVect.x;
                    currPos.y = currPos.y - currVect.y;
                    success = putPieceSquare(*newBoard, piece.value, currPos);
                    if(!success) {
                        delete newBoard;
                        return NULL;
                    }
                    index -= 1;
                    currVect = piece[index];
                }
                index = 1;
                currPos = pos;
                currVect = piece[index];
                while(!currVect.isNull()) {
                    currPos.x = currPos.x + currVect.x;
                    currPos.y = currPos.y + currVect.y;
                    success = putPieceSquare(*newBoard, piece.value, currPos);
                    if(!success) {
                        delete newBoard;
                        return NULL;
                    }
                    index += 1;
                    currVect = piece[index];
                }
            } else {
                delete newBoard;
                return NULL;
            }
            return newBoard;
        }

        virtual ostream& put(ostream & o, char sep=' ') const
        {
            int y, x;
            for(y = arrayOrigin.y - 1; y < (BDYL + arrayOrigin.y); y++) {
                for(x = arrayOrigin.x; x < (BDXL + arrayOrigin.x + 1); x++) {
                    if(boardArray[y][x - 1] != boardArray[y][x]) {
                        o << "|";
                    } else {
                        o << " ";
                    }
                    if(boardArray[y + 1][x] != boardArray[y][x]) {
                        o << "_";
                    } else {
                        o << " ";
                    }
                }
                o << endl;
            }
            return o << sep;
        }

        friend ostream& operator<<(ostream& o, const BoardBase& p) { return p.put(o); }

    protected:
        void AssignFrom(const BoardBase& other)
        {
            int x, y;
            next = other.next;
            arrayOrigin = other.arrayOrigin;
            for(x = 0; x < BXL; x++) {
                for(y = 0; y < BYL; y++) {
                    boardArray[y][x] = other.boardArray[y][x];
                }
            }
        }

        bool putPieceSquare(BoardBase &board, int value, Coord pos)
        {
            if(board.boardArray[board.arrayOrigin.y + pos.y][board.arrayOrigin.x + pos.x] == 0) {
                board.boardArray[board.arrayOrigin.y + pos.y][board.arrayOrigin.x + pos.x] = value;
                return true;
            }
            else return false;
        }
};

// ============================================================================
// Solve - Generic backtracking solver (works with any Board type)
// ============================================================================
template<typename BoardType>
bool Solve(BoardType& board, Piece * pieces[], int nbPieces, BoardType ** sols, int * nbTries, int * nbPlPcs, bool printSol, bool isUniqueSol, bool &keepSearching)
{
    bool isSolution = false;
    Coord pos;
    Trans* trans;
    BoardType* newBoard;
    Piece ** newPieces;
    int i, j;
    int newNbPieces = nbPieces - 1;
    if(nbPieces != 0) {
        board.nextAvailablePos(&pos);
        int pIdx = 0;
        while(pIdx < nbPieces && keepSearching) {
            int origin = 0;
            while(origin <= pieces[pIdx]->shapeLength && keepSearching) {
                trans = pieces[pIdx]->relevantTrans;
                int tIdx = 0;
                while(tIdx < pieces[pIdx]->nbRelevantTrans && keepSearching) {
                    pieces[pIdx]->origin = origin;
                    pieces[pIdx]->transform(*trans);
                    newBoard = board.putPiece(*pieces[pIdx], pos);
                    (*nbTries)++;
                    if(newBoard) {
                        (*nbPlPcs)++;
                        if(newNbPieces > 0) {
                            newPieces = new Piece*[newNbPieces];
                            j = 0;
                            for(i = 0; i < nbPieces; i++) {
                                if(i != pIdx) {
                                    newPieces[j] = pieces[i];
                                    j++;
                                }
                            }
                        } else {
                            newPieces = NULL;
                        }
                        isSolution = Solve(*newBoard, newPieces, newNbPieces, sols, nbTries, nbPlPcs, printSol, isUniqueSol, keepSearching);
                        if(isSolution != true) {
                            delete newBoard;
                        }
                        if(isUniqueSol == true && isSolution == true) {
                            keepSearching = false;
                        }
                        isSolution = false;
                        delete newPieces;
                    }
                    tIdx++;
                    trans++;
                }
                origin++;
            }
            pIdx++;
        }
    } else {
        if(*sols) {
            board.next = *sols;
        }
        *sols = &board;
        isSolution = true;
        if(printSol == true) {
            cout << board << endl;
        }
    }
    return isSolution;
}

// ============================================================================
// Utility functions for printing
// ============================================================================
inline void printMonth(int monthNum)
{
    switch(monthNum) {
        case 1: cout << "January"; break;
        case 2: cout << "February"; break;
        case 3: cout << "March"; break;
        case 4: cout << "April"; break;
        case 5: cout << "May"; break;
        case 6: cout << "June"; break;
        case 7: cout << "July"; break;
        case 8: cout << "August"; break;
        case 9: cout << "September"; break;
        case 10: cout << "October"; break;
        case 11: cout << "November"; break;
        case 12: cout << "December"; break;
        default: cout << "InvalidMonthNumber";
    }
}

inline void printWeekday(int weekdayNum)
{
    switch(weekdayNum) {
        case 1: cout << "Monday"; break;
        case 2: cout << "Tuesday"; break;
        case 3: cout << "Wednesday"; break;
        case 4: cout << "Thursday"; break;
        case 5: cout << "Friday"; break;
        case 6: cout << "Saturday"; break;
        case 7: cout << "Sunday"; break;
        default: cout << "InvalidWeekdayNum";
    }
}

inline void printError(string msg, string arg="")
{
    cerr << msg << " " << arg << endl;
}

#endif // PUZZLE_COMMON_H
