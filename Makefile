CXX=g++
NONFREE=src/local/opencv2/nonfree
CXXFLAGS=-std=c++11 -Wall $$(pkg-config --cflags opencv) -Isrc/local -O3
LDFLAGS=$$(pkg-config --libs opencv)
SRC_NONFREE=$(wildcard $(NONFREE)/*.cpp)
OBJ_NONFREE=$(SRC_NONFREE:.cpp=.o)
PROG_OBJ=src/descriptors.o src/utils.o $(OBJ_NONFREE)

all: bin/mean bin/sift bin/surf

src/%.o: src/%.cpp
	$(CXX) -c $^ -o $@ $(CXXFLAGS)

$(NONFREE)/%.o: $(NONFREE)/%.cpp
	$(CXX) -c $^ -o $@ $(CXXFLAGS)

bin/mean: $(PROG_OBJ) bin
	$(CXX) -o $@ src/main.cpp $(PROG_OBJ) $(CXXFLAGS) $(LDFLAGS) -DCREATE_DESCRIPTORS=meanDescriptor

bin/sift: $(PROG_OBJ) bin
	$(CXX) -o $@ src/main.cpp $(PROG_OBJ) $(CXXFLAGS) $(LDFLAGS) -DCREATE_DESCRIPTORS=siftDescriptor

bin/surf: $(PROG_OBJ) bin
	$(CXX) -o $@ src/main.cpp $(PROG_OBJ) $(CXXFLAGS) $(LDFLAGS) -DCREATE_DESCRIPTORS=surfDescriptor

bin:
	mkdir -p bin

cleanfiles:
	$(RM) *.yml data*/*.yml

clean: cleanfiles
	$(RM) bin/mean bin/sift bin/surf src/*.o $(NONFREE)/*.o

