#ifndef SPARSECODING_H
#define SPARSECODING_H
template <typename T>
__device__ void merge_maxabs(T* a, T *b,int* idxA, int* idxB, T* buf, int* idxbuf, const int N, const int K);

template<typename T> 
__global__ void max_abs_k(T *input, T *output, int* idxinput, const int initialize, const int N, const int K);

template <typename T>
__host__ void do_max_abs_k(T *initial_input, T *output,int* idxinput, const int N, const int K);

int deviceQuery();

#endif