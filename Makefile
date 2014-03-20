CXX=g++
NONFREE=local/opencv2/nonfree
CXXFLAGS=$$(pkg-config --libs opencv)

all: 
	$(CXX) -o surf surf.cpp -I. $(NONFREE)/nonfree_init.cpp  $(NONFREE)/surf.cpp $(NONFREE)/sift.cpp $(CXXFLAGS)