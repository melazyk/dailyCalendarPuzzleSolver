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

// Puzzle board sizes definition, to be customize for other types of puzzle board
#define BXL 13 // X len of the puzzle board
#define BYL 14 // Y len of the puzzle board
#define BOX 3 // Board origin in X, to prevent pieces from spanning out of board array
#define BOY 3 // Board origin in Y, to prevent pieces from spanning out of board array
// BOX and BOY depends of the max dimension of the biggest piece
#define BDXL 7 // X span from board origin BOX to display area border
#define BDYL 8 // Y span from board origin BOY to display area boarder
// The board display area correspond to real life puzzle board
// the board not displayed area is a technical area required to place pieces without spanning out of the array implementing the board

// Puzzle board, only the constructor needs to be customized to change the type of puzzle board
class Board : public BoardBase<Board, BXL, BYL, BOX, BOY, BDXL, BDYL>
{
public:
    Board(int weekday, int monthDay, int month);
    Board(const Board& other): BoardBase(other){}
    virtual ~Board(){}
    Board& operator=(const Board& other){BoardBase::operator=(other);return *this;}
};

Board::Board(int weekday, int monthDay, int month)
{
    // weekday from 1 to 7 with 1=Monday..7=Sunday
    // monthDay from 1 to 31
    // month from 1 for January to 12 for december
    int x,y;
    next = NULL;
    arrayOrigin.x= BOX;
    arrayOrigin.y = BOY;
    for(x=0;x<BXL;x++){
        for(y=0;y<BYL;y++){
            boardArray[y][x] = -1;
        }
    }
    for(x=arrayOrigin.x;x<(arrayOrigin.x+BDXL);x++){
        for(y=arrayOrigin.y;y<(arrayOrigin.y+BDYL);y++){
            boardArray[y][x] = 0;
        }
    }
    boardArray[3][9] = -1;
    boardArray[4][9] = -1;
    boardArray[10][3] = -1;
    boardArray[10][4] = -1;
    boardArray[10][5] = -1;
    boardArray[10][6] = -1;
    boardArray[((month-1)/6)+3][((month-1)%6)+3] = -1;
    boardArray[((monthDay-1)/7)+5][((monthDay-1)%7)+3] = -1;
    if((weekday-1) == 6) boardArray[9][6] = -1;
    else boardArray[((weekday-1)/3)+9][((weekday-1)%3)+7] = -1;
}


void printHelp(string prog)
{
    cout << "Synopsis:\n\n    [weekday day month] [-h] [-u] [-i] [-s] [-t [PcsNb] [PcsNb] [PcsNb]...]\n\nDescription:\n\n    Solver for Daily Calendar Puzzle with month, day of month and day of week. Return value is 0 if at least one solution has been found or -h option is used.\n\nArguments:\n\n    weekday day month\n          Date to solve. weekday from 1=Monday to 7=Sunday, day from 1 to 31 and month from 1 to 12.\n\n    -h    Print this help\n\n    -u    Stop looking for solutions when a first one has been found\n\n    -i    Make the results appear once all of them has been found in a single line of python syntax\n\n    -s     Make the pieces be used with their smooth side up as reference side, instead their frosted side\n\n    -t    Specify which pieces can be flipped (use both sides instead reference side only)\n\n    PcsNb Number of a piece in range 1-10 added after '-t' to specify a pieces which shall be used both sides during solutions searching.\n\n          Pieces numbers are these ones:\n\n          7  7     4  4  4\n          1  7  7  7  4  2\n          1  6  6  3  4  2  2\n          1 10  6  3  3  3  2\n          1 10  6  6  9  9  9\n          8 10 10 10  9     9\n          8  8  8  8     5  5\n                      5  5  5\n\nExample\n\n    To solve Friday 23rd of January\n\n    " << prog << " 5 23 1\n" << endl;
}


int main(int argc, char* argv[])
{
    auto startTime = std::chrono::high_resolution_clock::now();
    int wdayNum = 0;
    int dayNum = 0;
    int monthNum = 0;
    int fourFlatTransLen=2;
    int smallSTransLen=2;
    int smallLTransLen=4;
    int tTransLen=4;
    int uTransLen=4;
    int bigSTransLen=2;
    int smallStailTransLen=4;
    int bigLTransLen=4;
    int qTransLen=4;
    int lEqualTransLen=4;
    bool inLine = false;
    bool fSide=true;
    Board * sols = NULL;
    bool isArgsValid=true;
    bool isArgValid;
    bool isHelpOpt=false;
    bool isArgPcsFlip=false;
    bool isUniqueSol=false;
    string prog(argv[0]);
    for(int i=1; i<argc;i++){
        isArgValid=false;
        string arg(argv[i]);
        if( isArgsValid && !isArgPcsFlip && (wdayNum == 0 || dayNum == 0 || monthNum == 0 ) ){
            int p;
            bool isNum=true;
            try {
                p = stoi(arg);// convert argument to a number
            } catch ( std::invalid_argument& e ){
                isNum=false;
            }
            if( isNum &&  isArgsValid && !isArgValid && wdayNum == 0 ){
                if( p>=1 && p<=7){
                    wdayNum = p;// week day number, monday=1, tuesday=2...
                    isArgValid=true;
                } else {
                    printError("Week day number out of [1-7] range:",arg);
                    isArgsValid=false;
                }
            }
            if( isNum && isArgsValid && !isArgValid && dayNum == 0 ){
                if( p>= 1 && p<=31 ){
                    dayNum = p;//day number from 1 to 31
                    isArgValid=true;
                } else {
                    printError("Month day number out of [1-31] range:",arg);
                    isArgsValid=false;
                }
            }
            if( isNum && isArgsValid && !isArgValid && monthNum == 0 ){
                if( p>= 1 && p<=12 ){
                    monthNum = p;// month number, january=1, february=2...
                    isArgValid=true;
                 } else {
                    printError("Month number out of [1-12] range:",arg);
                    isArgsValid=false;
                }
            }
        }
        if( !isArgValid && isArgsValid && arg.compare("-i")==0) {
            inLine=true;//inline python processable printing of the results
            isArgValid=true;
        }
        if( !isArgValid && isArgsValid && arg.compare("-s")==0){
            fSide=false;//front side not used, back side of pieces are the base side to use
            isArgValid=true;
        }
        if( !isArgValid && isArgsValid && arg.compare("-t")==0){
            isArgPcsFlip = true; // next arguments are pieces numbers
            isArgValid=true;
        }
        if( !isArgValid && isArgsValid && arg.compare("-u")==0){
            isUniqueSol = true;
            isArgValid=true;
        }
         if( !isArgValid && isArgsValid && arg.compare("-h")==0 ){
            isArgValid=true;
            isHelpOpt=true;
        }
        if( !isArgValid && isArgsValid && isArgPcsFlip ){
            int p;
            try {
                p = stoi(arg);// convert argument to a number to identify the piece to use on both sides
            } catch ( std::invalid_argument& e){
                printError("Invalid piece number of range [1-10]:",arg);
                isArgsValid=false;
            }
            if( p<1 || p>10){
                printError("Invalid piece number of range [1-10]:",arg);
                isArgsValid=false;
            } else {
                isArgValid=true;
            }
            switch(p){
                case 2:
                    smallSTransLen=4;
                    break;
                case 3:
                    smallLTransLen=8;
                    break;
                case 6:
                    bigSTransLen=4;
                    break;
                case 7:
                    smallStailTransLen=8;
                    break;
                case 8:
                    bigLTransLen=8;
                    break;
                case 5:
                    qTransLen=8;
                    break;
                default:
                    break;//other pieces are the same once returned, so they are not returned to avoid creating identical solutions
            }
        }
        if( !isArgValid && isArgsValid  ){
            printError("Unkown argument:",arg);
            isArgsValid=false;
        }
    }
    if( wdayNum == 0 || dayNum == 0 || monthNum == 0 ){
        if(wdayNum == 0 && dayNum == 0 && monthNum == 0 && isArgsValid && !isHelpOpt){
            // get the current time
            auto now = std::chrono::system_clock::now();
            std::time_t time = std::chrono::system_clock::to_time_t(now);
             // get the current day of week, day in month, and month number
            std::tm* timeinfo = std::localtime(&time);
            wdayNum = timeinfo->tm_wday == 0 ? 7 : timeinfo->tm_wday;
            dayNum = timeinfo->tm_mday;
            monthNum = timeinfo->tm_mon + 1;
            cout << "No date provided, solving current date." << endl;
        } else {
            if( isArgsValid && !isHelpOpt ){
                isArgsValid = false;
                printError("Missing at least one of the following numbers : weekday day month");
            }
        }
    }
    if( !isArgsValid || isHelpOpt ) {
        printHelp(prog);
        if( isHelpOpt){
            exit(0);
        } else {
            exit(1);
        }
    } else {
        if(fSide==false){
            // update relevant transformations lists to put back side first to use it as base side
            allTrans[0] = Trans::upBack;
            allTrans[1] = Trans::rightBack;
            allTrans[2] = Trans::downBack;
            allTrans[3] = Trans::leftBack;
            allTrans[4] = Trans::up;
            allTrans[5] = Trans::right;
            allTrans[6] = Trans::down;
            allTrans[7] = Trans::left;
            allFaceTrans[0] = Trans::upBack;
            allFaceTrans[1] = Trans::rightBack;
            allFaceTrans[2] = Trans::downBack;
            allFaceTrans[3] = Trans::leftBack;
            upRightTrans[0] = Trans::upBack;
            upRightTrans[1] = Trans::rightBack;
            upRightTrans[2] = Trans::up;
            upRightTrans[3] = Trans::right;
        }
        // Create the 10 pieces
        Vect FourFlatArray[3]= {Vect(0,1),Vect(0,1),Vect(0,1)};
        Piece FourFlat(FourFlatArray, 3, 1, upRightTrans, fourFlatTransLen);
        Vect SmallSArray[3]=  {Vect(0,1),Vect(1,0),Vect(0,1)};;
        Piece SmallS(SmallSArray, 3, 2, upRightTrans, smallSTransLen);
        Vect SmallLArray[3] =  {Vect(0,1),Vect(1,0),Vect(1,0)};
        Piece SmallL(SmallLArray, 3, 3, allTrans, smallLTransLen);
        Vect TArray[4] =  {Vect(1,0),Vect(1,0),Vect(-1,1),Vect(0,1)};
        Piece T(TArray, 4, 4, allFaceTrans, tTransLen);
        Vect QArray[4] = {Vect(0,1),Vect(1,0),Vect(0,1),Vect(-1,0)};
        Piece Q(QArray, 4, 5, allTrans, qTransLen);
        Vect BigSArray[4] = {Vect(1,0),Vect(0,1),Vect(0,1),Vect(1,0)};
        Piece BigS(BigSArray, 4, 6, upRightTrans, bigSTransLen);
        Vect SmallsTailArray[4] = {Vect(1,0),Vect(0,1),Vect(1,0),Vect(1,0)};
        Piece SmallsTail(SmallsTailArray, 4, 7, allTrans, smallStailTransLen);
        Vect BigLArray[4] = {Vect(0,1),Vect(1,0),Vect(1,0),Vect(1,0)};
        Piece BigL(BigLArray, 4, 8, allTrans, bigLTransLen);
        Vect UArray[4] = {Vect(0,1),Vect(1,0),Vect(1,0),Vect(0,-1)};
        Piece U(UArray, 4, 9, allFaceTrans, uTransLen);
        Vect LequalArray[4] = {Vect(0,1),Vect(0,1),Vect(1,0),Vect(1,0)};
        Piece Lequal(LequalArray, 4, 10, allFaceTrans, lEqualTransLen);

        //  Create the board
        Board puzzle(wdayNum,dayNum,monthNum);
        // Solving
        int nbTries=0;
        int nbSols=0;
        int nbPlPcs=0;
        Board * nextSol;
        // Create the list of pieces to give to the solver
        Piece * puzzlePieces[10] = {&FourFlat,&U,&Q,&SmallsTail,&SmallL,&BigL,&SmallS,&Lequal,&BigS,&T};//this order is from the most to least frequent appearance in first case, when pieces are used on their front side, fSide=true
        if(inLine==false){
            cout << "Solutions:" << endl;
        }
        bool keepSearching = true;
        // Call the solving function
        Solve(puzzle, puzzlePieces, 10, &sols,&nbTries,&nbPlPcs,(inLine==false),isUniqueSol,keepSearching);

        if(inLine==false){
            // Print the solution as human readable
            if(sols){
                nbSols ++;
                nextSol = sols->next;
                while(nextSol){
                    nbSols++;
                    nextSol = nextSol->next;
                }
             }
            cout << nbSols << " solutions found after " << nbTries << " tries and "<<  nbPlPcs << " pieces placed for ";
            printWeekday(wdayNum);
            cout << " " << dayNum << " ";
            printMonth(monthNum);
            cout << endl;
        } else {
            // print the solution as python dict entry
            cout << "\"";
            printWeekday(wdayNum);
            cout << " " << dayNum;
            printMonth(monthNum);
            cout << "\": { \"nbPcsPlaced\": " << nbPlPcs << ",\"nbTries\":" << nbTries << ", \"sols\": [";
            if(sols){
                (*sols).print();
                nbSols ++;
                nextSol = sols->next;
                while(nextSol){
                    cout << ",";
                    (*nextSol).print();
                    nbSols++;
                    nextSol = nextSol->next;
                }
            }
            cout  << "], \"nbSol\": " << nbSols << "}";
        }
        if(!sols){
            if(fSide){
                // turning pieces 5 is enough to get solution for all dates like wed 27 when using frosted side only, except sun 6th apr, for which another piece need to be returned
                cout << "Try with option: -t 5 2 " << endl;
            } else {
                // turning piece 5 is enough when using smooth side as reference to get a solution for all dates like mon 27th which have no solution smooth side only
                cout << "Try with option: -t 5" << endl;
            }
        }
    }
    if(inLine==false){
        auto endTime = std::chrono::high_resolution_clock::now();
        cout << "End of program reached, execution duration: " << (float)(std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count())/1000 << " seconds" << endl;
    }
    if ( !sols ){
        exit(1);
    } else {
        exit(0);
    }
}
