#include<stdio.h>
#include<sys/time.h>

 const int numThreads = 100000;
 int numThreadsPerBlock = 16;

// catch error if any
void checkCUDAError(const char* msg);

struct atom{
	 double posx[numThreads];
	 double posy[numThreads],posz[numThreads];
	 double velx[numThreads],vely[numThreads],velz[numThreads];
	 double forcex[numThreads],forcey[numThreads],forcez[numThreads];
};

//////////////////////////////////////////////////////
// 		KERNEL
//////////////////////////////////////////////////////
__global__ void updateAtomKernel(const atom *d_in,atom *d_out){

	// thread id
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	//coalesced memory acces
	d_out->posx[idx] = 2*d_in->posx[idx];
	d_out->posy[idx] = 2*d_in->posy[idx];
	d_out->posz[idx] = 2*d_in->posz[idx];

	d_out->velx[idx] = 2*d_in->velx[idx];
	d_out->vely[idx] = 2*d_in->vely[idx];
	d_out->velz[idx] = 2*d_in->velz[idx];

	d_out->forcex[idx] = 2*d_in->forcex[idx];
	d_out->forcey[idx] = 2*d_in->forcey[idx];
	d_out->forcez[idx] = 2*d_in->forcez[idx];

}

////////////////////////////////////////////////////////////
//		Main Program
////////////////////////////////////////////////////////////
int main(int argc,char **argv){

	if(argc >= 1) numThreadsPerBlock = atoi(argv[1]); 


	// assign host memory variable and size
	atom *h_soa;
	int sizeA = sizeof(atom);
	double time;
	timeval t;
	// assign device memory address
	atom *d_a;
	atom *d_b;

	// assign number of blocks and num of threads
	int numBlocks = numThreads/numThreadsPerBlock;

	// allocate space to host memory and device
	int memSize = sizeA;
	h_soa = (atom *)malloc(sizeA);
	cudaMalloc((void **)&d_a,memSize);
	cudaMalloc((void **)&d_b,memSize);

	//intialize host device
	for(int i=0;i<numThreads;i++){
		h_soa->posx[i] = 1;
		h_soa->posy[i] = 1;
		h_soa->posz[i] = 1;

		h_soa->velx[i] = 1;
		h_soa->vely[i] = 1;
		h_soa->velz[i] = 1;

		h_soa->forcex[i] = 1;
		h_soa->forcey[i] = 1;
		h_soa->forcez[i] = 1;
 	}

 	gettimeofday(&t,NULL);
	time = t.tv_sec*1000.0 + (t.tv_usec/1000.0);

	//copy host to device all the memory
	cudaMemcpy(d_a,h_soa,memSize,cudaMemcpyHostToDevice);

	//launch kernel
	dim3 dimGrid(numBlocks);
	dim3 dimBlock(numThreadsPerBlock);
	updateAtomKernel<<<dimGrid,dimBlock>>>(d_a,d_b);

	//synchronize threads
	cudaThreadSynchronize();

	// check if any error
	checkCUDAError("Invocation kernel");

    // device to host copy
    cudaMemcpy( h_soa, d_b, memSize, cudaMemcpyDeviceToHost );

    // To validate result. must be all = 2
    /*
	for(int i=0;i<numThreads;i++){
		printf("h at x pos %d = %f \n",i,h_soa->posx[i]);
		printf("h at y pos %d = %f \n",i,h_soa->posy[i]);
		printf("h at z pos %d = %f \n",i,h_soa->posz[i]);

		printf("h at x vel %d = %f \n",i,h_soa->velx[i]);
		printf("h at y vel %d = %f \n",i,h_soa->vely[i]);
		printf("h at z vel %d = %f \n",i,h_soa->velz[i]);

		printf("h at x force %d = %f \n",i,h_soa->forcex[i]);
		printf("h at y force %d = %f \n",i,h_soa->forcey[i]);
		printf("h at z force %d = %f \n",i,h_soa->forcez[i]);

	}*/

		gettimeofday(&t,NULL);
	time = t.tv_sec*1000.0 + (t.tv_usec/1000.0) - time;

	// print time taken
	printf("Time taken in ThreadsPerBlock: %d is %f msec\n", numThreadsPerBlock,time);
	
	//free memory
	cudaFree(d_a);
	cudaFree(d_b);
	free(h_soa);


}

// error catch statement to ease debug
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}
