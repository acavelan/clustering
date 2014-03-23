CXX=g++
CXXFLAGS=-std=c++11 -Wall $$(pkg-config --cflags opencv) -Isrc/local
LDFLAGS=$$(pkg-config --libs opencv)

all: means sift Makefile

means: src/means.cpp Makefile
	$(CXX) -o $@ $< src/local/opencv2/nonfree/*.cpp $(LDFLAGS) $(CXXFLAGS)

sift: src/sift.cpp Makefile
	$(CXX) -o $@ $< src/local/opencv2/nonfree/*.cpp $(LDFLAGS) $(CXXFLAGS)

run-means: means Makefile
	./means data/img{1..50}.jpg

run-sift: sift Makefile
	./sift data/img{1..50}.jpg

clean:
	$(RM) means sift *.o

