#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "sparsecoding.h"
#include "utils.hpp"

typedef elem<float> Elem;

int main()
{
	std::srand(std::time(NULL));
	int M;
	int K;	
	std::cin >> M;
	std::cin >> K;
	float W[M][M];
	float x[M];
	float x2[K];
	int idx[K];
	float y[M];
	float expected[M];
	std::cout << "W" << std::endl;
	for(int i=0;i<M;i++)
	{

		x[i] = (0.5-(((float)std::rand())/((float)RAND_MAX))) * 20;
		for(int j=0;j<M;j++)
		{
			W[i][j] = (0.5-(((float)std::rand())/((float)RAND_MAX))) * 20;
			std::cout << W[i][j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "x" << std::endl;
	for(int i=0; i<M;i++){
		std::cout << x[i] << std::endl;
	}
	float* d_W;
	float* d_x;
	int* d_idx;
	float* d_y;
	cudaMalloc(&d_W, sizeof(float)*M*M);
	cudaMalloc(&d_x, sizeof(float)*M);
	cudaMalloc(&d_idx, sizeof(int)*M);
	cudaMalloc(&d_y, sizeof(float)*M);
	cudaMemcpy(d_W,W,sizeof(float)*M*M,cudaMemcpyHostToDevice);
	cudaMemcpy(d_x,x,sizeof(float)*M,cudaMemcpyHostToDevice);
	do_max_abs_k(d_x,d_x,d_idx,M,K);
	do_sparseMult(d_W,d_x,d_idx,d_y,M,K);
	cudaDeviceSynchronize();
	cudaMemcpy(x2,d_x,sizeof(float)*K,cudaMemcpyDeviceToHost);
	cudaMemcpy(idx,d_idx,sizeof(int)*K,cudaMemcpyDeviceToHost);
	cudaMemcpy(y,d_y,sizeof(float)*M,cudaMemcpyDeviceToHost);
	std::cout << "projected x" << std::endl;
	for(int i=0; i<K; i++)
	{
		std::cout << idx[i] << " " << x2[i] << std::endl;
	}
	std::cout << "y" << std::endl;
	bool pass = true;	
	for(int i=0;i<M;i++)
	{
		expected[i]=0;
		for(int j=0;j<K;j++)
		{
			expected[i] += W[i][idx[j]] * x2[j];
		}
		bool correct = approx(expected[i],y[i]);
		pass = pass && correct;
		if(correct) std::cout << y[i] << std::endl;
		else std::cout << "expected " << expected[i] << " got " << y[i] << std::endl;
	}

	if(pass) std::cout<< "correct!" << std::endl;
	else std::cout << "incorrect." << std::endl;

	cudaFree(d_W);
	cudaFree(d_x);
	cudaFree(d_idx);
	cudaFree(d_y);
	return 0;
}
