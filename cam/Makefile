# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -O2

# FFTW library
LIBS = -lSpinnaker_v140
PATHS = -I/mnt/c/Program\ Files/FLIR\ Systems/Spinnaker/include -L/mnt/c/Program\ Files/FLIR\ Systems/Spinnaker/lib64

spin : spin.o
	$(CXX) $(CXXFLAGS) -o spin spin.o $(PATHS) $(LIBS)

spin.o : spinnaker_test.cpp
	$(CXX) $(CXXFLAGS) -c -o spin.o spinnaker_test.cpp

clean:
	rm -f spin spin.o