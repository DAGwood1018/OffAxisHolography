# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -O2

# FFTW library
LIBS = -lfftw3_threads -lfftw3 -lm

DFT : DFT.o
	$(CXX) $(CXXFLAGS) -o DFT DFT.o $(LIBS)

DFT.o : DFT.cpp
	$(CXX) $(CXXFLAGS) -c -o DFT.o DFT.cpp

clean:
	rm -f DFT DFT.o

