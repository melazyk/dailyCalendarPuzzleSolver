all: poodlepuzzleDailyCalendarSolver.bin dragonFjordDailyCalendarSolver.bin

clean:
	rm -f poodlepuzzleDailyCalendarSolver.bin poodlepuzzleDailyCalendarSolver.o
	rm -f dragonFjordDailyCalendarSolver.bin dragonFjordDailyCalendarSolver.o

poodlepuzzleDailyCalendarSolver.o: poodlepuzzleDailyCalendarSolver.cpp puzzleCommon.h
	g++ -Wall -std=c++11 -o poodlepuzzleDailyCalendarSolver.o -c poodlepuzzleDailyCalendarSolver.cpp

poodlepuzzleDailyCalendarSolver.bin: poodlepuzzleDailyCalendarSolver.o
	g++ -o poodlepuzzleDailyCalendarSolver.bin poodlepuzzleDailyCalendarSolver.o

dragonFjordDailyCalendarSolver.o: dragonFjordDailyCalendarSolver.cpp puzzleCommon.h
	g++ -Wall -std=c++11 -o dragonFjordDailyCalendarSolver.o -c dragonFjordDailyCalendarSolver.cpp

dragonFjordDailyCalendarSolver.bin: dragonFjordDailyCalendarSolver.o
	g++ -o dragonFjordDailyCalendarSolver.bin dragonFjordDailyCalendarSolver.o
