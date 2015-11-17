#define BLOCK_SIZE 1024
// the reduction operator
// a and b are K-element arrays sorted in descending magnitude i.e. [5 -4 3 1 0 0]
// buf is a K-element array used as buffer to store results before copying back to a
/*
 Example Input:
 a : [5 -4 3 1 0 0]
 b : [-9 2 -2 1 0 0]
 buf : don't care
 Example Output:
 a=buf : [-9 5 -4 3 2 -2]
 b : [-9 2 -2 1 0 0]
*/
__device__ void merge_maxabs(float* a, float *b,float* idxA, float* idxB, float* buf, float* idxbuf, const int N, const int K)
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
		int incb = iszeros ?  aidx==N : (maxmag==bmag) && (amag!=bmag);
		
		buf[i] = incb ? bval : aval;				
		idxbuf[i] = incb ? bidx : aidx;
		ia += !incb;
		ib += incb;
	}
	memcpy(a,buf,sizeof(float)*K);
	memcpy(idxA,idxbuf,sizeof(float)*K);
	/* OR
	for(int i=0;i<K;i++)
	{
		a[i] = buf[i];
		idxA[i] = idxbuf[i];
	}
	//*/
}
// thresholding on BLOCK_SIZE chunks of data in a sample
// can be used recursively on output until K max magnitudes are left
// Do we need to preserve information about the order of elements?
// input[N]
// output[num_blocks x K]
// output is K max magnitudes of each block, in descending order
__global__ void max_abs_k(float *input, float *output, int* idxinput, int init_index, const int N, int K) {
  //__shared__ float p[BLOCK_SIZE+1][K]; /*!! kernel code cannot use variable here in array declaration, need to pass through kernel call!!*/
  extern __shared__ float p[];
  float* data = p+ tx*K;
  float* idxdata = p+(BLOCK_SIZE+tx)*K;
  float* buf = p+(BLOCK_SIZE*2+tx)*K;
  float* idxbuf = p+(BLOCK_SIZE*3+tx)*K;
  int tx = threadIdx.x, bx = blockIdx.x;
  unsigned int i = bx*BLOCK_SIZE + tx;
  data[0] = i<N ? input[i] : 0;       
  idxdata[0] = i<N ? ( init_index ? i : idxinput[i] ): N;       
  for(int j=1;j<K;j++)
  {
	data[j] = 0;	    	
	idxdata[j] = N;
  }		
  __syncthreads();  
  for(unsigned int stride=(BLOCK_SIZE>>1);stride>0;stride>>=1)
  {	  
	  if(tx<stride){
		//perform min absolute k
		merge_maxabs(data,data+stride*K,idxdata, idxdata+stride*K, buf, idxbuf,N,K);  	 
	  }
	  __syncthreads();
  }  
  if(tx<K)
  {
	  int count = 0;
	  float my_val = p[tx];
	  int my_idx = p[BLOCK_SIZE*K+tx];
	  for(int j=0;j<K;j++)
	  {
		  count += (my_idx > p[BLOCK_SIZE*K+j]);
	  }
	  output[bx*K+count] = my_val;
	  idxinput[bx*K+count] = my_idx;
  }  
}

// kernel call wrapper. Since this is a sketch it will only handle one sample at a time.
// One sample is probably already large enough for a GPU, so maybe each sample (or a fixed limited number of them) should go to a different GPU on different machines?
// input : N
// output : num_blocks x K
__host__ void do_max_abs_k(float *initial_input, float *output,int* idxinput, const int N, const int K)
{
	int n = N;
	float* input = initial_input;	
	int shared_size_bytes = sizeof(float)*BLOCK_SIZE*K*4; // three parts: data, index data, data buffer, index buffer
	int num_blocks = ceil(n*1.0/NUM_BLOCKS);		
	int init_index = 1;
	do
	{		
		max_abs_k<<<num_blocks,BLOCK_SIZE,shared_size_bytes>>>(input,output,idxinput,init_index,n,K);
		init_index = 0;
		n = K*num_blocks;
		num_blocks = ceil(n*1.0/NUM_BLOCKS);		
		input = output;
	}while(n!=K);
	// now output[0..K-1] contains the max k elements 
	// now idxinput[0..K-1] storing corresponding index (index order preserved)
}