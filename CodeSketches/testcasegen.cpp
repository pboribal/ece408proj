#include <iostream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

typedef std::pair<float,int> elem;
typedef struct {
	bool operator()(const elem& a, const elem& b)
	{
		return std::abs(a.first) > std::abs(b.first) || (std::abs(a.first)==std::abs(b.first) && a.second < b.second );
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
	std::vector<elem> f;
	for(int i=0;i<n;i++)
	{
		elem fi;
		fi.first = (0.5-((float)std::rand())/((float)RAND_MAX))*20;
		fi.second = i;
		std::cout << fi.first << std::endl;
		f.push_back(fi);	
	}
	std::cout << std::endl;
	elemcmp e;
	std::sort(f.begin(),f.end(),e);
	for(int i=0;i<k;i++)
	{
		std::cout << f[i].second << "  " << f[i].first << std::endl;
	}
	return 0;
}
