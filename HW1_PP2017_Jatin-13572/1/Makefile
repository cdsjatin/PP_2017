
MPICC?=mpicc
CC = gcc

all: collComm
	qsub sample-script
	

collComm: collComm.c
	${MPICC} -std=c99 -o collComm collComm.c -lm

clean:
	rm -f *.o *.out *.txt myjob* collComm
