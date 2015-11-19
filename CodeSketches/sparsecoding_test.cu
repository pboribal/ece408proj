#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <map>
#include <set>
#include "utils.hpp"
#include "sparsecoding.h"

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
