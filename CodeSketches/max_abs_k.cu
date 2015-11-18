#define BLOCK_SIZE 1024
#include <stdio.h>
#include <stdlib.h>

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
__device__ void merge_maxabs(float* a, float *b,int* idxA, int* idxB, float* buf, int* idxbuf, const int N, const int K)
{
	int ia = 0;
	int ib = 0;
	for(int i=0;i<K;i++)
	{
		float aval = a[ia];		 float bval = b[ib];
		float aidx = idxA[ia];	 float bidx = idxB[ib];
		float amag = fabs(aval); float bmag = fabs(bval);
		float maxmag = fmax(amag,bmag);
		int iszeros = aval==0 && bval==0;		
		int incb = iszeros ?  bidx<aidx : (maxmag==bmag) && (amag!=bmag);
		
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
__global__ void max_abs_k(float *input, float *output, int* idxinput, int init_index, const int N, const int K) {  
  extern __shared__ float data[];
  int tx = threadIdx.x, bx = blockIdx.x;
  int offset = tx*K;
  int* idxdata = (int*) &data[BLOCK_SIZE*K];
  float* buf = (float*) &idxdata[BLOCK_SIZE*K];
  int* idxbuf = (int*) &buf[BLOCK_SIZE*K];
  unsigned int i = bx*BLOCK_SIZE + tx;
  data[offset] = i<N ? input[i] : 0;       
  idxdata[offset] = i<N ? ( init_index ? i : idxinput[i] ): N;       
  for(int j=1;j<K;j++)
  {
	data[offset+j] = 0;	    	
	idxdata[offset+j] = N;
  }		
  __syncthreads();  
  for(unsigned int stride=(BLOCK_SIZE>>1);stride>0;stride>>=1)
  {	  
	  if(tx<stride){
		//perform min absolute K reduction operator
		int stride_offset = offset+stride*K;
		merge_maxabs(data+offset,data+stride_offset,idxdata+offset, idxdata+stride_offset, buf+offset, idxbuf+offset,N,K);  	 
	  }
	  __syncthreads();
  }  
  if(tx<K)
  {
	  int count = 0;
	  float my_val = data[tx];
	  int my_idx = idxdata[tx];
	  for(int j=0;j<K;j++)
	  {
		  count += (my_idx > idxdata[j]);
	  }
	output[bx*K+count] = my_val;
	idxinput[bx*K+count] = my_idx;
  }  
}
//
// kernel call wrapper. Since this is a sketch it will only handle one sample at a time.
// One sample is probably already large enough for a GPU, so maybe each sample (or a fixed limited number of them) should go to a different GPU on different machines?
// input : N
// output : num_blocks x K
__host__ void do_max_abs_k(float *initial_input, float *output,int* idxinput, const int N, const int K)
{
	int n = N;
	float* input = initial_input;		
	int shared_size_bytes = (sizeof(float)+sizeof(int))*2*BLOCK_SIZE*K; // 4 parts: data(float), index data(int), data buffer(float), index buffer (int)
	int num_blocks = (int) ceil(n*1.0/BLOCK_SIZE);		
	int init_index = 1;
	int round = 0;
	do
	{		
		printf("doing round %d: size=%d (%d blocks)\n", round, n, num_blocks);
		max_abs_k<<<num_blocks,BLOCK_SIZE,shared_size_bytes>>>(input,output,idxinput,init_index,n,K);
		init_index = 0;
		n = K*num_blocks;
		num_blocks = (int) ceil(n*1.0/BLOCK_SIZE);		
		input = output;
	}while(n!=K);
	// now output[0..K-1] contains the max k elements 
	// now idxinput[0..K-1] storing corresponding index (index order preserved)
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
    printf(" Maximum shared memory size per block: %d\n", deviceProp.sharedMemPerBlock);
    printf(" Maximum block dimensions: %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf(" Maximum grid dimensions: %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf(" Warp size: %d\n", deviceProp.warpSize);
  }

  printf("stop - Getting GPU Data.\n"); //@@ stop the timer
  return 0;
}

int main(int argc, char **argv) 
{
	if(deviceQuery())
	{
		return 0;
	}
	int input_size = 10;
	int k = 5;
	int num_blocks = (int) ceil(input_size*1.0/BLOCK_SIZE);
	int output_size = k*num_blocks;
	float* hInput = (float*) malloc(sizeof(float)*input_size);
	float* hOutput = (float*) malloc(sizeof(float)*k);
	printf("initializing input\n");
	for(int i=0;i<input_size;i++)
	{
		hInput[i] = ( i&1 ? 1 : -1 )*(0.123*i);
		printf("%f\n",hInput[i]);
	}
	printf("done\n");
	float* dInput;
	int* dIdxInput;
	float* dOutput;	
	cudaMalloc(&dInput, sizeof(float)*input_size);
	cudaMalloc(&dIdxInput, sizeof(int)*input_size);
	cudaMalloc(&dOutput, sizeof(float)*output_size);	
	cudaMemcpy(dInput,hInput,sizeof(float)*input_size,cudaMemcpyHostToDevice);
	do_max_abs_k(dInput,dOutput,dIdxInput,input_size,k);
	cudaDeviceSynchronize();
	cudaMemcpy(hOutput,dOutput, sizeof(float)*k, cudaMemcpyDeviceToHost);	
	for(int i=0;i<k;i++)
	{
		printf("%f\n", hOutput[i]);
	}
	free(hInput);
	free(hOutput);
	cudaFree(dInput);
	cudaFree(dIdxInput);
	cudaFree(dOutput);
	return 0;
}
