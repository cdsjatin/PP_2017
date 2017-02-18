CC=nvcc

all: hw1-b-AOS hw1-b-SOA
	./hw1-b-AOS 128
	./hw1-b-AOS 129
	./hw1-b-AOS 140
	./hw1-b-AOS 150
	./hw1-b-AOS 160		
	./hw1-b-SOA 128
	./hw1-b-SOA 129
	./hw1-b-SOA 140
	./hw1-b-SOA 150
	./hw1-b-SOA 160	

hw1-b-AOS: hw1-b-AOS.o
	$(CC) -o hw1-b-AOS hw1-b-AOS.o

hw1-b-AOS.o: hw1-b-AOS.cu
	$(CC) -o hw1-b-AOS.o -c hw1-b-AOS.cu

hw1-b-SOA: hw1-b-SOA.o
	$(CC) -o hw1-b-SOA hw1-b-SOA.o

hw1-b-SOA.o: hw1-b-AOS.cu
	$(CC) -o hw1-b-SOA.o -c hw1-b-SOA.cu

clean:
	rm -f hw1-b-AOS.o hw1-b-AOS hw1-b-SOA.o hw1-b-SOA

