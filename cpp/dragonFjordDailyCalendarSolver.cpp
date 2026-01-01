#include <iostream>
#include <string>
#include <time.h>
#include <stdexcept>
#include <ctime>
#include <chrono>
#include <stdlib.h>
#include "puzzleCommon.h"

using namespace std;

// ============================================================================
// Transformation arrays definitions
// ============================================================================
Trans allTrans[8] = {Trans::up, Trans::right, Trans::down, Trans::left, Trans::upBack, Trans::rightBack, Trans::downBack, Trans::leftBack};
Trans allFaceTrans[4] = {Trans::up, Trans::right, Trans::down, Trans::left};
Trans upRightTrans[4] = {Trans::up, Trans::right, Trans::upBack, Trans::rightBack};

// Puzzle board sizes definition for Dragon Fjord (different from Poodle)
// The playable area is 7 rows high (y=0..6) and up to 7 columns wide
// (x=0..6) with the first two rows only 6 wide; we keep the outer padding
// to reuse the generic printer/solver.
#define BXL 13 // X len of the puzzle board
#define BYL 13 // Y len of the puzzle board
#define BOX 3 // Board origin in X, to prevent pieces from spanning out of board array
#define BOY 3 // Board origin in Y, to prevent pieces from spanning out of board array
#define BDXL 7 // X span from board origin BOX to display area border
#define BDYL 7 // Y span from board origin BOY to display area boarder

// ============================================================================
// Board - Puzzle board for Dragon Fjord puzzle
// ============================================================================
class Board : public BoardBase<Board, BXL, BYL, BOX, BOY, BDXL, BDYL>
{
    public:
        Board(int monthDay, int month);
        Board(const Board& other) : BoardBase(other) {}
        virtual ~Board() {}
        Board& operator=(const Board& other) { BoardBase::operator=(other); return *this; }
};

Board::Board(int monthDay, int month)
{
    // monthDay from 1 to 31
    // month from 1 for January to 12 for december
    int x, y;
    next = NULL;
    arrayOrigin.x = BOX;
    arrayOrigin.y = BOY;

    // Default everything to blocked
    for(x = 0; x < BXL; x++) {
        for(y = 0; y < BYL; y++) {
            boardArray[y][x] = -1;
        }
    }

    // Playable area shape (mirrors py/dragonFjordDailyCalendarSolver.GenerateBoard)
    // Rows 0-1: width 6 (x=0..5)
    // Rows 2-5: width 7 (x=0..6)
    // Row 6: width 3 (x=0..2)
    for(y = 0; y < BDYL; y++) {
        int maxX;
        if(y < 2) {
            maxX = 5;  // Rows 0-1: 6 columns
        } else if(y < 6) {
            maxX = 6;  // Rows 2-5: 7 columns
        } else {
            maxX = 2;  // Row 6: 3 columns
        }
        for(x = 0; x <= maxX; x++) {
            boardArray[arrayOrigin.y + y][arrayOrigin.x + x] = 0;
        }
    }

    // Mark unavailable positions for day and month (occupy the hole squares)
    boardArray[((monthDay - 1) / 7) + BOY + 2][((monthDay - 1) % 7) + BOX] = -1;
    boardArray[((month - 1) / 6) + BOY][((month - 1) % 6) + BOX] = -1;
}


// ============================================================================
// Utility print functions (for puzzle-specific help)
// ============================================================================
void printHelp(string prog)
{
    cout << "Synopsis:\n\n    [day month] [-h] [-u] [-i]\n\nDescription:\n\n    Solver for Dragon Fjord Daily Calendar Puzzle with month and day of month. Return value is 0 if at least one solution has been found or -h option is used.\n\nArguments:\n\n    day month\n          Date to solve. day from 1 to 31 and month from 1 to 12.\n\n    -h    Print this help\n\n    -u    Stop looking for solutions when a first one has been found\n\n    -i    Make the results appear once all of them has been found in a single line of python syntax\n\nExample\n\n    To solve 23rd of January\n\n    " << prog << " 23 1\n" << endl;
}

int main(int argc, char* argv[])
{
    auto startTime = std::chrono::high_resolution_clock::now();
    int dayNum = 0;
    int monthNum = 0;
    int oTransLen = 2;
    int tTransLen = 8;
    int qTransLen = 8;
    int bigSTransLen = 4;
    int smallSTailTransLen = 8;
    int bigLTransLen = 8;
    int uTransLen = 4;
    int lEqualTransLen = 4;
    bool inLine = false;
    Board * sols = NULL;
    bool isArgsValid = true;
    bool isArgValid;
    bool isHelpOpt = false;
    bool isUniqueSol = false;
    string prog(argv[0]);

    for(int i = 1; i < argc; i++) {
        isArgValid = false;
        string arg(argv[i]);
        if(isArgsValid && (dayNum == 0 || monthNum == 0)) {
            int p;
            bool isNum = true;
            try {
                p = stoi(arg);
            } catch (std::invalid_argument& e) {
                isNum = false;
            }
            if(isNum && isArgsValid && !isArgValid && dayNum == 0) {
                if(p >= 1 && p <= 31) {
                    dayNum = p;
                    isArgValid = true;
                } else {
                    printError("Day number out of [1-31] range:", arg);
                    isArgsValid = false;
                }
            }
            if(isNum && isArgsValid && !isArgValid && monthNum == 0) {
                if(p >= 1 && p <= 12) {
                    monthNum = p;
                    isArgValid = true;
                } else {
                    printError("Month number out of [1-12] range:", arg);
                    isArgsValid = false;
                }
            }
        }
        if(!isArgValid && isArgsValid && arg.compare("-i") == 0) {
            inLine = true;
            isArgValid = true;
        }
        if(!isArgValid && isArgsValid && arg.compare("-u") == 0) {
            isUniqueSol = true;
            isArgValid = true;
        }
        if(!isArgValid && isArgsValid && arg.compare("-h") == 0) {
            isArgValid = true;
            isHelpOpt = true;
        }
        if(!isArgValid && isArgsValid) {
            printError("Unknown argument:", arg);
            isArgsValid = false;
        }
    }

    if(dayNum == 0 || monthNum == 0) {
        if(dayNum == 0 && monthNum == 0 && isArgsValid && !isHelpOpt) {
            auto now = std::chrono::system_clock::now();
            std::time_t time = std::chrono::system_clock::to_time_t(now);
            std::tm* timeinfo = std::localtime(&time);
            dayNum = timeinfo->tm_mday;
            monthNum = timeinfo->tm_mon + 1;
            cout << "No date provided, solving current date." << endl;
        } else {
            if(isArgsValid && !isHelpOpt) {
                isArgsValid = false;
                printError("Missing at least one of the following numbers : day month");
            }
        }
    }

    if(!isArgsValid || isHelpOpt) {
        printHelp(prog);
        if(isHelpOpt) {
            exit(0);
        } else {
            exit(1);
        }
    } else {
        // Create the 8 pieces for Dragon Fjord puzzle
        Vect OArray[5] = {Vect(1, 0), Vect(1, 0), Vect(0, 1), Vect(-1, 0), Vect(-1, 0)};
        Piece O(OArray, 5, 1, upRightTrans, oTransLen);

        Vect tArray[4] = {Vect(1, 0), Vect(1, 0), Vect(1, 0), Vect(-1, 1)};
        Piece t(tArray, 4, 2, allTrans, tTransLen);

        Vect QArray[4] = {Vect(0, 1), Vect(1, 0), Vect(0, 1), Vect(-1, 0)};
        Piece Q(QArray, 4, 3, allTrans, qTransLen);

        Vect BigSArray[4] = {Vect(1, 0), Vect(0, 1), Vect(0, 1), Vect(1, 0)};
        Piece BigS(BigSArray, 4, 4, upRightTrans, bigSTransLen);

        Vect SmallSTailArray[4] = {Vect(1, 0), Vect(0, 1), Vect(1, 0), Vect(1, 0)};
        Piece SmallSTail(SmallSTailArray, 4, 5, allTrans, smallSTailTransLen);

        Vect BigLArray[4] = {Vect(0, 1), Vect(1, 0), Vect(1, 0), Vect(1, 0)};
        Piece BigL(BigLArray, 4, 6, allTrans, bigLTransLen);

        Vect UArray[4] = {Vect(0, 1), Vect(1, 0), Vect(1, 0), Vect(0, -1)};
        Piece U(UArray, 4, 7, allFaceTrans, uTransLen);

        Vect LequalArray[4] = {Vect(0, 1), Vect(0, 1), Vect(1, 0), Vect(1, 0)};
        Piece Lequal(LequalArray, 4, 8, allFaceTrans, lEqualTransLen);

        // Create the board
        Board puzzle(dayNum, monthNum);

        // Solving
        int nbTries = 0;
        int nbSols = 0;
        int nbPlPcs = 0;
        Board * nextSol;

        // Create the list of pieces to give to the solver
        Piece * puzzlePieces[8] = {&O, &t, &Q, &BigS, &SmallSTail, &BigL, &U, &Lequal};

        if(inLine == false) {
            cout << "Solutions:" << endl;
        }
        bool keepSearching = true;
        Solve(puzzle, puzzlePieces, 8, &sols, &nbTries, &nbPlPcs, (inLine == false), isUniqueSol, keepSearching);

        if(inLine == false) {
            if(sols) {
                nbSols++;
                nextSol = sols->next;
                while(nextSol) {
                    nbSols++;
                    nextSol = nextSol->next;
                }
            }
            cout << nbSols << " solutions found after " << nbTries << " tries and " << nbPlPcs << " pieces placed for ";
            cout << dayNum << " ";
            printMonth(monthNum);
            cout << endl;
        } else {
            cout << "\"" << dayNum;
            cout << " ";
            printMonth(monthNum);
            cout << "\": { \"nbPcsPlaced\": " << nbPlPcs << ",\"nbTries\":" << nbTries << ", \"sols\": [";
            if(sols) {
                (*sols).print();
                nbSols++;
                nextSol = sols->next;
                while(nextSol) {
                    cout << ",";
                    (*nextSol).print();
                    nbSols++;
                    nextSol = nextSol->next;
                }
            }
            cout << "], \"nbSol\": " << nbSols << "}";
        }
    }

    if(inLine == false) {
        auto endTime = std::chrono::high_resolution_clock::now();
        cout << "End of program reached, execution duration: " << (float)(std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count()) / 1000 << " seconds" << endl;
    }
    if(!sols) {
        exit(1);
    } else {
        exit(0);
    }
}
