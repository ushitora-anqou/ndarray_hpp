CXXFLAGS=-std=c++17 -Wall -Wextra -pedantic
CXXFLAGS_DEBUG=$(CXXFLAGS) -g3 -O0
CXXFLAGS_RELEASE=$(CXXFLAGS) -O3 -march=native
INC=-I mnist/include -I cereal/include -I range-v3/include -I ../../sequences/include -I cifar-10/include -I ../../
LIB= -lopenblas -lpthread

main: main.cpp
	#clang++ $(CXXFLAGS_SANITIZE) -o $@ $< $(INC) $(LIB)
	#clang++ $(CXXFLAGS_DEBUG) -o $@ $< $(INC) $(LIB)
	clang++ $(CXXFLAGS_RELEASE) -o $@ $< $(INC) $(LIB)
