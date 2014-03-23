CXX=g++
CXXFLAGS=-std=c++11 -Wall $$(pkg-config --cflags opencv)
LDFLAGS=$$(pkg-config --libs opencv)
TARGET=main

all: $(TARGET) Makefile

$(TARGET): src/main.cpp Makefile
	$(CXX) -o $@ $< $(LDFLAGS) $(CXXFLAGS)

run: all Makefile
	./$(TARGET) ../data/img{1..50}.jpg

clean:
	$(RM) $(TARGET) *.o

