#include <iostream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include "utils.hpp"

typedef elem<float> Elem;
typedef struct {
	bool operator()(const Elem& a, const Elem& b)
	{
		return std::abs(a.val) > std::abs(b.val) || (std::abs(a.val)==std::abs(b.val) && a.idx < b.idx );
	}
} elemcmp;

int main()
{
	std::srand(std::time(NULL));
	int n;
	int k;
	std::cin >> n;
	std::cin >> k;
	std::cout << n << " " << k << std::endl;
	std::vector<Elem> f;
	for(int i=0;i<n;i++)
	{
		Elem fi;
		fi.val = (0.5-((float)std::rand())/((float)RAND_MAX))*20;
		fi.idx = i;
		std::cout << fi.val << std::endl;
		f.push_back(fi);	
	}
	std::cout << std::endl;
	elemcmp e;
	std::sort(f.begin(),f.end(),e);
	for(int i=0;i<k;i++)
	{
		std::cout << f[i].idx << "  " << f[i].val << std::endl;
	}
	return 0;
}
