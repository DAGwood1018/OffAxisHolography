CXX = g++
CXXFLAGS = -Wall -O3
LDFLAGS = -lfftw3_threads -lfftw3 -lm

TARGET = dft_test
SRCS = main.cpp DFT.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp DFT.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
