CXX=g++
NONFREE=local/opencv2/nonfree
CXXFLAGS=$$(pkg-config --libs opencv)

all: 
	$(CXX) -o surf surf.cpp -Ilocal $(NONFREE)/*.cpp $(CXXFLAGS)


