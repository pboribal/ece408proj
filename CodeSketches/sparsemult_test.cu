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
	std::vector<float> W,x,x2,y;
	std::vector<int> idx;	
	std::cin >> M;
	std::cin >> K;
	std::cout << "W" << std::endl;
	for(int i=0;i<M;i++)
	{

		x.push_back( (0.5-(((float)std::rand())/((float)RAND_MAX))) * 20);
		for(int j=0;j<M;j++)
		{
			float t = (0.5-(((float)std::rand())/((float)RAND_MAX))) * 20;
			W.push_back(t);
			std::cout << t << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "x" << std::endl;
	for(int i=0; i<M;i++){
		std::cout << x[i] << std::endl;
	}
	//my::sparsecoding<float> spc;
	//spc.klargest(x,x2,idx,K);
	//spc.kmult(W,x2,idx,y);
	my::klargest(x,x2,idx,K);
	my::kmult(W,x2,idx,y);
	std::cout << "projected x" << std::endl;
	for(int i=0; i<K; i++)
	{
		std::cout << idx[i] << " " << x2[i] << std::endl;
	}
	std::vector<float> expected;
	expected.resize(M,0.f);
	std::cout << "y" << std::endl;
	bool pass = true;	
	for(int i=0;i<M;i++)
	{
		expected[i]=0;
		for(int j=0;j<K;j++)
		{
			expected[i] += W[i*M+idx[j]] * x2[j];
		}
		bool correct = approx(expected[i],y[i]);
		pass = pass && correct;
		if(correct) std::cout << y[i] << std::endl;
		else std::cout << "expected " << expected[i] << " got " << y[i] << std::endl;
	}

	if(pass) std::cout<< "correct!" << std::endl;
	else std::cout << "incorrect." << std::endl;

	return 0;
}
