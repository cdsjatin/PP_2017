CC=nvcc

all: test-cuda

test-cuda: test-cuda.o
	$(CC) -o test-cuda test-cuda.o

test-cuda.o: test-cuda.cu
	$(CC) -o test-cuda.o -c test-cuda.cu

clean:
	rm -f test-cuda.o test-cuda

