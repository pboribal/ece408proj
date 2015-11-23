#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <map>
#include <set>
#include "utils.hpp"
#include "sparsecoding.h"
#define BLOCK_SIZE 1024

//TODO take this entry point away and implement entry point for matlab mex
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
	std::vector<float> hInput;
	std::vector<float> hOutput;
	std::vector<int> hIdx;
	std::map<int,float> expected;
	std::set<int> found;
	hInput.resize(input_size);
	//populate input
	for(int i=0;i<input_size;i++)
		std::cin >> hInput[i];
	//populate expected result from test case
	for(int i=0;i<k;i++) 
	{
		int idx;
		float val;
		std::cin >> idx;
		std::cin >> val;
		expected[idx] = val;
	}
	sparsecoding::klargest(hInput,hOutput,hIdx,k);
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
	return 0;
}
