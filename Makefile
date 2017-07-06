CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))

CC=g++
#CC=nvcc -ccbin g++

FLAGS=-std=c++11
# FLAGS=-Wall -Wextra -Werror
#FLAGS=-std=c++11 -gencode arch=compute_35,code=sm_35

INCLUDES=-I/usr/local/cuda-8.0/include -Iinclude -I../common/inc

LIBS=-lOpenCL

all: clReduction

debugc: FLAGS += -DDEBUG -g -O0
debugc: clReduction

clReduction: $(OBJ_FILES) 
	$(CC) -o $@ $^ $(FLAGS) $(LIBS) $(INCLUDES)

obj/%.o: src/%.cpp
	$(CC) -c -o $@ $< $(FLAGS) $(LIBS) $(INCLUDES)

run: clReduction
	./clReduction
	
.PHONY clean:
	rm -f clReduction ./obj/*

debug: $(OBJ_FILES) 
	$(CC) -g -o $@ $^ $(FLAGS) $(LIBS) $(INCLUDES)	