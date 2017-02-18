#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <unistd.h>

// Change number of core dynamically in a makefile
int NUM_CORE=8;
int MAX_INT_SIZE = 1024;	// could be any integer.


int main(int argc, char** argv){

MPI_Init(&argc, &argv);
srand(clock());
                    
int comm_size,myrank,k=7,s_no=1,i;
int TOTAL_ARRAY_LENGTH,RECV_ARRAY_LENGTH;
double time_t;

//no. of processes and rank
MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
NUM_CORE = comm_size;

	if(!myrank){
	printf("////////////////// %d Num cores //////////////////\n",NUM_CORE);
	printf("S.no \t Size(KB) \t Time(R.S.)(sec) \t Time(R.H.)(sec) \t Speedup (R.S./R.H.) \t Validity \n");
	printf("------\t----------\t------------------\t------------------\t-------------------\t----------\n");
	}

//repeat whole process for different message sizes(1k,2k,4k,8k)(2^19/(4 bytes of int))
while(++k<19){
MPI_Barrier(MPI_COMM_WORLD);
int *send_buffer,*recv_buffer,*final_sum;

// Generate total message on all processes excluding sizeof(int)
	TOTAL_ARRAY_LENGTH = pow(2,k);
	send_buffer = malloc(TOTAL_ARRAY_LENGTH*sizeof(int));
	
	if(!myrank) final_sum = malloc(TOTAL_ARRAY_LENGTH*sizeof(int));
	for(i =0;i< TOTAL_ARRAY_LENGTH;++i)
		send_buffer[i] = rand()%MAX_INT_SIZE;


// scatter message to all the process using MPI_SCATTER

RECV_ARRAY_LENGTH = TOTAL_ARRAY_LENGTH/NUM_CORE;
recv_buffer = malloc(RECV_ARRAY_LENGTH*sizeof(int));

time_t = MPI_Wtime();

// reduce using mpi call
MPI_Reduce((void *)send_buffer,final_sum,TOTAL_ARRAY_LENGTH,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

 //TOdo changed RECV_ARRAY_LENGTH
MPI_Scatter(final_sum,RECV_ARRAY_LENGTH,MPI_INT,recv_buffer,RECV_ARRAY_LENGTH,MPI_INT,0,MPI_COMM_WORLD);

time_t = MPI_Wtime() - time_t;
MPI_Barrier(MPI_COMM_WORLD);




/////////////////////////////////////////////////////////////////////////
/// 	RECURSIVE HALVING 											/////
/////////////////////////////////////////////////////////////////////////

double r_time = MPI_Wtime();
int SEND_ARRAY_LENGTH = TOTAL_ARRAY_LENGTH;
int sendr,recvr,k=0,q = NUM_CORE,p,g,arr_offset,m=0,*recv_buffer_r ;

recv_buffer_r = malloc(TOTAL_ARRAY_LENGTH*sizeof(int));
MPI_Request req;
MPI_Barrier(MPI_COMM_WORLD);

	while(q-1 != 0){
		q = q/2;		// communication group size
		p = myrank/q;	// pseudo communication group number
		g = p*q;	// communicating group leader (1st node in every pseudo communication group)
		

		sendr = myrank;
		recvr = ((myrank-g)+(q/2))%q + g;
		
		// to update only that half of the array which must be reduced 
		SEND_ARRAY_LENGTH = SEND_ARRAY_LENGTH/2;
		arr_offset = (TOTAL_ARRAY_LENGTH/NUM_CORE)*g;
		
		// offset send buffer with the group leader specifies the
		MPI_Isend((void *)(send_buffer+arr_offset),SEND_ARRAY_LENGTH,MPI_INT,recvr,q,MPI_COMM_WORLD,&req);
		MPI_Recv((void *)(recv_buffer_r+arr_offset),SEND_ARRAY_LENGTH,MPI_INT,recvr,q,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		
		//reduction by each process after receiving new value
		MPI_Barrier(MPI_COMM_WORLD);
		
	for(int i= arr_offset;i < arr_offset + SEND_ARRAY_LENGTH;i++){
		send_buffer[i] += recv_buffer_r[i];
			}
	
	}

	r_time = MPI_Wtime()-r_time;
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////// check and Print
///////////////////////////////////////////////////////////////////////////////////////
	int j=0,isvalid = 1,isallvalid = 0;
	
	for( i = myrank*RECV_ARRAY_LENGTH;i<(myrank+1)*RECV_ARRAY_LENGTH;i++){		
	if(recv_buffer[j++] != send_buffer[i]){isvalid = 0; break;}
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	// to check all the array have valid data comparison of recursive halving with reduce-scatter
	MPI_Reduce(&isvalid,&isallvalid,1,MPI_INT,MPI_LAND,0,MPI_COMM_WORLD);

	if(!myrank) printf("  %d \t %d\t\t%f\t\t%f \t \t %f \t \t %d \t \n",s_no++,TOTAL_ARRAY_LENGTH/256,time_t,r_time,time_t/r_time,isallvalid);

// free up memory
free(recv_buffer_r);
if(!myrank) free(final_sum);
free(recv_buffer);
free(send_buffer);
}

MPI_Finalize();


}//** end program **//

