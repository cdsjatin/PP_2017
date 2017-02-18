#include<stdio.h>
#include<sys/time.h>

// time of execution AOS will be more than SOA as AOS contains other data
// which is not yet loaded.

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

const int numThreads = 1000000;
int numThreadsPerBlock = 16;


// Array of structure a structure occupies 72 B
typedef struct {
	double3 pos;
	double3 vel;
	double3 force;
}atom;

//////////////////////////////////////////////////////
// 		KERNEL
//////////////////////////////////////////////////////
__global__ void updateAtomKernel(const atom *d_in,atom *d_out ,const int N){
	
	//int t_idx = threadIdx.x;	// thread index
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	
	//Colesced memory access.

	d_out[idx].pos.x =  2*d_in[idx].pos.x ;
	d_out[idx].pos.y =  2*d_in[idx].pos.y ;
	d_out[idx].pos.z =  2*d_in[idx].pos.z ;

	d_out[idx].vel.x =  2*d_in[idx].vel.x;
	d_out[idx].vel.y =  2*d_in[idx].vel.y;
	d_out[idx].vel.z =  2*d_in[idx].vel.z;

	d_out[idx].force.x = 2*d_in[idx].force.x;
	d_out[idx].force.y = 2*d_in[idx].force.y;
	d_out[idx].force.z = 2*d_in[idx].force.z;
}


////////////////////////////////////////////////////////////
//		Main Program
////////////////////////////////////////////////////////////
int main(int argc,char **argv){

	if(argc >= 1)  numThreadsPerBlock = atoi(argv[1]) ;

	// assign host memory variable and size
	atom *h_aos;						// number of threads
	int sizeA = numThreads*sizeof(atom); 				// size of host memory
	timeval t;
	double time;

	// assign device memory address
	atom *d_a;
	atom *d_b;

	// assign number of blocks and num of threads
	//int numThreadsPerBlock = ThreadsPerBlock;
	int numBlocks = numThreads/numThreadsPerBlock;

	// allocate space to host memory and device
	int memSize = sizeA;
	h_aos = (atom*)malloc(sizeA);
	cudaMalloc((void **)&d_a,memSize);
	cudaMalloc((void **)&d_b,memSize);

	//intialize host device
	for(int i=0;i<numThreads;i++){
		h_aos[i].pos.x = 1;
		h_aos[i].pos.y = 1;
		h_aos[i].pos.z = 1;

		h_aos[i].vel.x = 1;
		h_aos[i].vel.y = 1; 
		h_aos[i].vel.z = 1; 

		h_aos[i].force.x = 1; 
		h_aos[i].force.y = 1; 
		h_aos[i].force.z = 1; 

	}

	//copy host to device all the memory
	cudaMemcpy(d_a,h_aos,memSize,cudaMemcpyHostToDevice);
	
	gettimeofday(&t,NULL);
	time = t.tv_sec*1000.0 + (t.tv_usec/1000.0);

	//launch kernel
	dim3 dimGrid(numBlocks);
	dim3 dimBlock(numThreadsPerBlock);
	updateAtomKernel<<<dimGrid,dimBlock>>>(d_a,d_b,numThreads);

	//let the threads complete
	cudaThreadSynchronize();

	// check if any error
	checkCUDAError("Invocation kernel");
	
	gettimeofday(&t,NULL);
	time = t.tv_sec*1000.0 + (t.tv_usec/1000.0) - time;	
	
    // device to host copy
    cudaMemcpy( h_aos, d_b, memSize, cudaMemcpyDeviceToHost );

    // To validate result. must be all = 2
	/*
    for(int i=0;i<numThreads;i++){
	  	
		printf("new pos x: %f \n",h_aos[i].pos.x);
    		printf("new pos y: %f \n",h_aos[i].pos.y);
    		printf("new pos x: %f \n",h_aos[i].pos.z);

    		printf("new vel y: %f \n",h_aos[i].vel.x);
    		printf("new vel x: %f \n",h_aos[i].vel.y);
    		printf("new vel y: %f \n",h_aos[i].vel.z);

    		printf("new force x: %f \n",h_aos[i].force.x);
    		printf("new force y: %f \n",h_aos[i].force.y);
    		printf("new force z: %f \n",h_aos[i].force.z);
	
    	}*/
	
	printf("Time taken in ThreadsPerBlock: %d is %f msec\n", numThreadsPerBlock,time);

    // Free some memory
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_aos);

}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}
