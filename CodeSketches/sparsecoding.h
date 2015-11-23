#ifndef SPARSECODING_H
#define SPARSECODING_H
#include <stdio.h>
#include <math.h>
#include <vector>
#include "utils.hpp"
#define BLOCK_SIZE 1024 

// the reduction operator
// a and b are K-element arrays sorted in descending magnitude i.e. [5 -4 3 1 0 0]
// buf is a K-element array used as buffer to store results before copying back to a
// index is copied along in the same order as its corresponding element.
/*
 Example Input:
 a : [5 -4 3 1 0 0]
 b : [-9 2 -2 1 0 0]
 buf : don't care
 Example Output:
 a=buf : [-9 5 -4 3 2 -2]
 b : [-9 2 -2 1 0 0]
*/
// note that the data type needs to be integral!
template <typename T>
__device__ void merge_maxabs(T* a, T *b,int* idxA, int* idxB, T* buf, int* idxbuf, const int N, const int K)
{
	int ia = 0;
	int ib = 0;
	for(int i=0;i<K;i++)
	{
		T aval = a[ia];		 T bval = b[ib];
		int aidx = idxA[ia];	 int bidx = idxB[ib];
		T amag = fabs(aval); 	 T bmag = fabs(bval);
		int isequal = amag==bmag;
		int incb = isequal ? (bidx<aidx) : (bmag>amag);
		buf[i] = incb ? bval : aval;
		idxbuf[i] = incb ? bidx : aidx;
		ia += !incb;
		ib += incb;
	}
	for(int i=0;i<K;i++)
	{
		a[i] = buf[i];
		idxA[i] = idxbuf[i];
	}
}
// thresholding on BLOCK_SIZE chunks of data in a sample
// can be used recursively on output until K max magnitudes are left
// Do we need to preserve information about the order of elements?
// input[N]
// output[num_blocks x K]
// output is K max magnitudes of each block, in descending order
template<typename T> 
__global__ void max_abs_k(T *input, T *output, int* idxinput, const int initialize, const int N, const int K)
{ 
  extern __shared__ T data[];
  int tx = threadIdx.x, bx = blockIdx.x, dimx = BLOCK_SIZE;
  int gx = tx/K;
  int offset = gx*K; // offset within block
  int block_offset = bx*dimx;
  int* idxdata = (int*) &data[dimx];
  T* buf = (T*) &idxdata[dimx];
  int* idxbuf = (int*) &buf[dimx];
  unsigned int i = block_offset + tx;  
  int count = 0;
  T myval = data[tx] = i<N ? input[i] : 0;       
  int myidx = idxdata[tx] = i<N ? ( initialize ? i : idxinput[i] ): i; 
  T absmyval = fabs(myval);  
  int numgroups = dimx/K;
  if(initialize)
  {       
	for(int j=0;j<K;j++)
	  {	  
		  int p = offset+j;
		  T absthatval = fabs(data[p]);		  
		  int  thatidx = idxdata[p];
		  count += ((myidx < N && absmyval < absthatval) || ((absmyval == absthatval || myidx>=N) && myidx>thatidx));			 
 	  }
	  __syncthreads();    
	  data[offset+count] = myval;
	  idxdata[offset+count] = myidx;  
	
  }  
  __syncthreads();  
  int self_offset = tx*K;
  for(unsigned int stride=(numgroups>>1)+(numgroups&1);stride>0;stride=(stride>>1)+((stride&1) && stride!=1))
  {	  	
	  int stride_offset = self_offset+stride*K;	  
	  if(tx<stride && tx+stride < numgroups){
		//perform min absolute K reduction operator		
		merge_maxabs(data+self_offset,data+stride_offset,idxdata+self_offset, idxdata+stride_offset, buf+self_offset, idxbuf+stride_offset,N,K);  	 
	  }
	  numgroups = stride;
	  __syncthreads();
  }  
  if(tx<K)
  {
	output[bx*K+tx] = data[tx];
	idxinput[bx*K+tx] = idxdata[tx];
  }  
}


// multiply MxM matrix by a sparse vector with K nonzero elements, indexes are
// given correspondingly. The kernel will be called with size <<<M,K>>>
// the algorithm can be faster to load if we pass a constant memory for W.
template <typename T>
__global__ void sparseMult(T* W,T* x, int* idx, T* y, const int M, const int K)
{
	extern __shared__ T p[]; // p for product, this will have a size of M*K*sizeof(T)
	int tx = threadIdx.x, bx = blockIdx.x;	
	T value = x[tx];
	__syncthreads();
	int column = idx[tx];
	__syncthreads();
	p[tx] = W[bx*M + column] * value;
	__syncthreads();	
	int numgroups = K;
  	for(unsigned int stride=(numgroups>>1)+(numgroups&1);stride>0;stride=(stride>>1)+((stride&1) && stride!=1))
	  {	  	
		  if(tx<stride && tx+stride < numgroups){
		  	p[tx] += p[tx+stride];
		  }
		  numgroups = stride;
		  __syncthreads();
	  }  
	if(tx==0)
	{
		y[bx] = p[0];
	}
}
namespace sparsecoding{
//
// kernel call wrapper.
// One sample is probably already large enough for a GPU, so maybe each sample (or a fixed limited number of them) should go to a different GPU on different machines?
// input : N
// output : num_blocks x K
template <typename T>
void klargest(const std::vector<T>& initial_input,std::vector<T>& final_output,std::vector<int>& idxoutput, const int K )
{
	int N = initial_input.size();
	int n = N;
	int num_blocks = (int) ceil(n*1.0/BLOCK_SIZE);		
	T *d_input,*input,*output;		
	int* idxinput;
	cudaMalloc(&d_input,sizeof(T)*N);
	input = d_input;
	cudaMalloc(&output,sizeof(T)*K*num_blocks);
	cudaMalloc(&idxinput,sizeof(int)*N);
	cudaMemcpy(input,&initial_input[0],sizeof(T)*N,cudaMemcpyHostToDevice);
	int shared_size_bytes = (sizeof(T)+sizeof(int))*2*BLOCK_SIZE; // 4 parts: data(T), index data(int), data buffer(T), index buffer (int)
	int init_index = 1; //use direct index for the first round
	do
	{		
		max_abs_k<<<num_blocks,BLOCK_SIZE,shared_size_bytes>>>(input,output,idxinput,init_index,n,K);
		init_index = 0;
		n = K*num_blocks;
		num_blocks = (int) ceil(n*1.0/BLOCK_SIZE);		
		input = output;
	}while(n!=K);
	cudaDeviceSynchronize();
	final_output.resize(K);
	idxoutput.resize(K);
	cudaMemcpy(&final_output[0],output,sizeof(T)*K,cudaMemcpyDeviceToHost);
	cudaMemcpy(&idxoutput[0],idxinput,sizeof(int)*K,cudaMemcpyDeviceToHost);
	cudaFree(d_input); cudaFree(output); cudaFree(idxinput);
	// now final_output[0..K-1] contains the max k elements 
	// now idxoutput[0..K-1] storing corresponding indexes
}
template <typename T>
void kmult(const std::vector<T>& W, const std::vector<T>& x, const std::vector<int>& idx, std::vector<T>& y)
{
	int M = (int) round(sqrt(W.size()));
	int K = x.size();
	T *d_W, *d_x, *d_y;
	int* d_idx;
	cudaMalloc(&d_W,sizeof(T)*M*M); //maybe change this to constant mem
	cudaMalloc(&d_x,sizeof(T)*K);
	cudaMalloc(&d_idx,sizeof(int)*K);
	cudaMalloc(&d_y,sizeof(T)*M);
	cudaMemcpy(d_W,&W[0],sizeof(T)*M*M,cudaMemcpyHostToDevice);
	cudaMemcpy(d_x,&x[0],sizeof(T)*K,cudaMemcpyHostToDevice);
	cudaMemcpy(d_idx,&idx[0],sizeof(int)*K,cudaMemcpyHostToDevice);
	sparseMult<<<M,K,K*sizeof(T)>>>(d_W,d_x,d_idx,d_y,M,K);
	cudaDeviceSynchronize();
	y.resize(M);
	cudaMemcpy(&y[0],d_y,sizeof(T)*M,cudaMemcpyDeviceToHost);
	cudaFree(d_W); cudaFree(d_x); cudaFree(d_idx); cudaFree(d_y);
}


int deviceQuery()
{
  int deviceCount;

  cudaGetDeviceCount(&deviceCount);

  printf("start - Getting GPU Data.\n"); //@@ start a timer

  for (int dev = 0; dev < deviceCount; dev++) {
    cudaDeviceProp deviceProp;

    cudaGetDeviceProperties(&deviceProp, dev);

    if (dev == 0) {
      if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
        printf("No CUDA GPU has been detected\n");
        return -1;
      } else if (deviceCount == 1) {
        printf("There is 1 device supporting CUDA\n");
      } else {
        printf("There are %d devices supporting CUDA\n", deviceCount);
      }
    }
    printf("Device %d: %s\n", dev, deviceProp.name);
    printf("Computational Capabilities: %d.%d\n", deviceProp.major, deviceProp.minor);    
    printf(" Maximum global memory size: %lu\n", deviceProp.totalGlobalMem);
    printf(" Maximum constant memory size: %lu\n", deviceProp.totalConstMem);
    printf(" Maximum shared memory size per block: %d\n", (int)deviceProp.sharedMemPerBlock);
    printf(" Maximum block dimensions: %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf(" Maximum grid dimensions: %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf(" Warp size: %d\n", deviceProp.warpSize);
  }

  printf("stop - Getting GPU Data.\n"); //@@ stop the timer
  return 0;
}

} //end namespace
#endif

