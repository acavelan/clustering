CXX=g++
NONFREE=local/opencv2/nonfree
CXXFLAGS=$$(pkg-config --libs opencv)

all: Makefile
	$(CXX) -o surf surf.cpp -Ilocal $(NONFREE)/*.cpp $(CXXFLAGS)

run-good: all Makefile
	./surf data/img1.jpg data/img7.jpg

run-bad: all Makefile
	./surf data/img1.jpg data/img12.jpg

