#define BLOCK_SIZE 1024 
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <map>
#include <set>


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

//
// kernel call wrapper. Since this is a sketch it will only handle one sample at a time.
// One sample is probably already large enough for a GPU, so maybe each sample (or a fixed limited number of them) should go to a different GPU on different machines?
// input : N
// output : num_blocks x K
template <typename T>
__host__ void do_max_abs_k(T *initial_input, T *output,int* idxinput, const int N, const int K)
{
	int n = N;
	T* input = initial_input;		
	int shared_size_bytes = (sizeof(T)+sizeof(int))*2*BLOCK_SIZE; // 4 parts: data(T), index data(int), data buffer(T), index buffer (int)
	int num_blocks = (int) ceil(n*1.0/BLOCK_SIZE);		
	int init_index = 1; //use direct index for the first round
	do
	{		
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
    printf(" Maximum shared memory size per block: %d\n", (int)deviceProp.sharedMemPerBlock);
    printf(" Maximum block dimensions: %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf(" Maximum grid dimensions: %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf(" Warp size: %d\n", deviceProp.warpSize);
  }

  printf("stop - Getting GPU Data.\n"); //@@ stop the timer
  return 0;
}
template<typename K, typename V>
__host__ void mergekv(K* keys, V* values, int start, int mid, int end)
{	
	int size = end - start;
	int ia = start;
	int ib = mid;
	int i = 0;
	K* ktmp = (K*)malloc(sizeof(K)*size);
	V* vtmp = (V*)malloc(sizeof(V)*size);
	while(ia < mid && ib < end)
	{
		int inca = keys[ia] < keys[ib];
		ktmp[i] = inca ? keys[ia] : keys[ib];
		vtmp[i] = inca ? values[ia]: values[ib];
		ia += inca;
		ib += !inca;
		i++;
	}
	while(ia < mid) 
	{
		ktmp[i] = keys[ia];
		vtmp[i] = values[ia];
		ia++;
		i++;
	}
	while(ib < end)
	{
		ktmp[i] = keys[ib];
		vtmp[i] = values[ib];
		ib++;
		i++;
	}
	for(i=0;i<size;i++)
	{
		keys[start+i] = ktmp[i];
		values[start+i] = vtmp[i];
	}
	free(ktmp);
	free(vtmp);
}
template<typename K, typename V>
__host__ void sortkv(K* keys, V* values,int start, int end)
{
	int size = end-start;
	if(size<2) return;
	if(size==2) 
	{
		if(keys[start+1] < keys[start])
		{
			K tmpk = keys[start];
			keys[start] = keys[start+1];
			keys[start+1] = tmpk;
			V tmpv = values[start];
			values[start] = values[start+1];
			values[start+1] = tmpv;
		}
	} else {
		int mid = (start+end)/2;
		sortkv(keys,values,start, mid);
		sortkv(keys,values,mid, end);
		mergekv(keys,values,start,mid,end);
	}
}

/*
* Performs an in-place sort, sorted in ascending order (not really relevant)
*/
template<typename K, typename V>
__host__ void do_sortkv(K* keys, V* values, int N)
{
	sortkv(keys,values,0,N);	
}
// TODO take this entry point away and implement entry point for matlab mex
int main() 
{
	int input_size;
	int k ;
	//get data attributes
	std::cin >> input_size;
	std::cin >> k;
	if(k<0 || input_size < 0 || k > input_size)
	{
		printf("error: input_size and k must be positive and k > input_size\n");
		return -1;
	}
	int num_blocks = (int) ceil(input_size*1.0/BLOCK_SIZE);
	int output_size = k*num_blocks;
	float* hInput = (float*) malloc(sizeof(float)*input_size);
	float* hOutput = (float*) malloc(sizeof(float)*k);
	int* hIdx = (int*) malloc(sizeof(int)*k);
	std::map<int,float> expected;
	std::set<int> found;
	//populate input
	for(int i=0;i<input_size;i++) std::cin >> hInput[i];
	//populate expected result from test case
	for(int i=0;i<k;i++) 
	{
		int idx;
		float val;
		std::cin >> idx;
		std::cin >> val;
		expected[idx] = val;
	}
	float* dInput;
	int* dIdxInput;
	float* dOutput;	
	cudaMalloc(&dInput, sizeof(float)*input_size);
	cudaMalloc(&dIdxInput, sizeof(int)*input_size);
	cudaMalloc(&dOutput, sizeof(float)*output_size);	
	cudaMemcpy(dInput,hInput,sizeof(float)*input_size,cudaMemcpyHostToDevice);
	do_max_abs_k(dInput,dOutput,dIdxInput,input_size,k);

	cudaDeviceSynchronize();
	cudaMemcpy(hOutput,dOutput,sizeof(float)*k,cudaMemcpyDeviceToHost);
	cudaMemcpy(hIdx,dIdxInput,sizeof(int)*k,cudaMemcpyDeviceToHost);
	bool pass = true;
	for(int i=0;i<k;i++)
	{
		int idx = hIdx[i];
		float val = hOutput[i];
		printf("%d %f\n",idx,val);
		if(expected.count(idx))
		{
			if(expected[idx]!=val)
			{
				pass = false;
			}
		}
		else
		{
			pass = false;
		}

		if(!found.count(idx))
		{
			found.insert(idx);
		} else
		{
			pass =  false;
		}
	}
	if(pass) std::cout << "correct!" << std::endl;
	else std::cout << "incorrect." << std::endl;
	free(hInput);
	free(hOutput);
	cudaFree(dInput);
	cudaFree(dIdxInput);
	cudaFree(dOutput);
	return 0;
}
