CXX=g++
NONFREE=src/local/opencv2/nonfree
CXXFLAGS=-std=c++11 -Wall $$(pkg-config --cflags opencv) -Isrc/local -O3
LDFLAGS=$$(pkg-config --libs opencv)
SRC_NONFREE=$(wildcard $(NONFREE)/*.cpp)
OBJ_NONFREE=$(SRC_NONFREE:.cpp=.o)
PROG_OBJ=src/descriptors.o src/utils.o $(OBJ_NONFREE)

all: mean sift surf huMoments gabor

src/%.o: src/%.cpp
	$(CXX) -c $^ -o $@ $(CXXFLAGS)

$(NONFREE)/%.o: $(NONFREE)/%.cpp
	$(CXX) -c $^ -o $@ $(CXXFLAGS)

mean: $(PROG_OBJ) bin
	$(CXX) -o bin/$@ src/main.cpp $(PROG_OBJ) $(CXXFLAGS) $(LDFLAGS) -DCREATE_DESCRIPTORS=meanDescriptor

sift: $(PROG_OBJ) bin
	$(CXX) -o bin/$@ src/main.cpp $(PROG_OBJ) $(CXXFLAGS) $(LDFLAGS) -DCREATE_DESCRIPTORS=siftDescriptor

surf: $(PROG_OBJ) bin
	$(CXX) -o bin/$@ src/main.cpp $(PROG_OBJ) $(CXXFLAGS) $(LDFLAGS) -DCREATE_DESCRIPTORS=surfDescriptor

huMoments: $(PROG_OBJ) bin
	$(CXX) -o bin/$@ src/main.cpp $(PROG_OBJ) $(CXXFLAGS) $(LDFLAGS) -DCREATE_DESCRIPTORS=huMomentsDescriptor

gabor: $(PROG_OBJ) bin
	$(CXX) -o bin/$@ src/main.cpp $(PROG_OBJ) $(CXXFLAGS) $(LDFLAGS) -DCREATE_DESCRIPTORS=gaborDescriptor

bin:
	mkdir -p bin

cleanfiles:
	$(RM) *.yml data*/*.yml

clean: cleanfiles
	$(RM) bin/mean bin/sift bin/surf src/*.o $(NONFREE)/*.o

