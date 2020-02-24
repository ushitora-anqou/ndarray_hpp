CXXFLAGS=-std=c++17 -Wall -Wextra -pedantic
CXXFLAGS_DEBUG=$(CXXFLAGS) -g3 -O0
CXXFLAGS_RELEASE=$(CXXFLAGS) -O3 -march=native
INC=-I sequences/include
LIB= -lopenblas -lpthread

test0: test.cpp ndarray.hpp
	#clang++ $(CXXFLAGS_SANITIZE) -o $@ $< $(INC) $(LIB)
	#clang++ $(CXXFLAGS_DEBUG) -o $@ $< $(INC) $(LIB)
	clang++ $(CXXFLAGS_RELEASE) -o $@ $< $(INC) $(LIB)
