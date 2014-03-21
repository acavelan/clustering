CXX=g++
CXXFLAGS=-std=c++11 $$(pkg-config --cflags opencv)
LDFLAGS=$$(pkg-config --libs opencv)
TARGET=main


all: $(TARGET) Makefile

$(TARGET): main.cpp Makefile
	$(CXX) -o $@ $< $(LDFLAGS) $(CXXFLAGS)

run-good: all Makefile
	./$(TARGET) ../data/img1.jpg ../data/img7.jpg

run-bad: all Makefile
	./$(TARGET) ../data/img1.jpg ../data/img12.jpg

clean:
	$(RM) $(TARGET) *.o

