CXX=g++
NONFREE=src/local/opencv2/nonfree
CXXFLAGS=-std=c++11 -Wall $$(pkg-config --cflags opencv) -Isrc/local -O3
LDFLAGS=$$(pkg-config --libs opencv)
SRC_NONFREE=$(wildcard $(NONFREE)/*.cpp)
OBJ_NONFREE=$(SRC_NONFREE:.cpp=.o)


all: means

src/%.o: src/%.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

$(NONFREE)/%.o: $(NONFREE)/%.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

means: src/means.o src/utils.o $(OBJ_NONFREE)
	$(CXX) $^ -o $@ $(LDFLAGS)

run-means: means
	./means data/img*.jpg

cleanfiles:
	$(RM) dictionary.yml data/*.yml data2/*.yml

clean: cleanfiles
	$(RM) means sift src/*.o $(NONFREE)/*.o dictionary.yml data/*.yml

