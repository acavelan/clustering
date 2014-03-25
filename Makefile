CXX=g++
NONFREE=src/local/opencv2/nonfree
CXXFLAGS=-std=c++11 -Wall $$(pkg-config --cflags opencv) -Isrc/local -O3
LDFLAGS=$$(pkg-config --libs opencv)
SRC_NONFREE=$(wildcard $(NONFREE)/*.cpp)
OBJ_NONFREE=$(SRC_NONFREE:.cpp=.o)


all: dirs mean sift surf

src/%.o: src/%.cpp
	$(CXX) -c $^ -o $@ $(CXXFLAGS)

$(NONFREE)/%.o: $(NONFREE)/%.cpp
	$(CXX) -c $^ -o $@ $(CXXFLAGS)

mean: src/descriptors.o src/utils.o
	$(CXX) -o bin/$@ src/main.cpp $^ $(CXXFLAGS) $(LDFLAGS) -DCREATE_DESCRIPTORS=meanDescriptor

sift: src/descriptors.o src/utils.o
	$(CXX) -o bin/$@ src/main.cpp $^ $(CXXFLAGS) $(LDFLAGS) -DCREATE_DESCRIPTORS=siftDescriptor

surf: src/descriptors.o src/utils.o
	$(CXX) -o bin/$@ src/main.cpp $^ $(CXXFLAGS) $(LDFLAGS) -DCREATE_DESCRIPTORS=surfDescriptor

dirs:
	mkdir -p bin

cleanfiles:
	$(RM) *.yml data/*.yml data2/*.yml

clean: cleanfiles
	$(RM) bin/mean bin/sift bin/surf src/*.o $(NONFREE)/*.o

