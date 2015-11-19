#include <iostream>
#include <stdlib.h>
#include <cmath>
int main()
{
	srand(time(NULL));
	int n;
	int k;
	std::cin >> n;
	std::cin >> k;
	std::cout << n << " " << k << std::endl;
	float f[n];
	int idx[n];
	for(int i=0;i<n;i++)
	{
		f[i] = (0.5-((float)rand())/((float)RAND_MAX))*20;
		idx[i] = i;
		std::cout << f[i] << std::endl;
	}
	std::cout << std::endl;
	// Bubble sort is enough, it's just a test case generator
	// no need to be performance critical here
	volatile bool swapped = false;
	do
	{
		swapped = false;
		for(int i=1;i<n;i++)
		{
			if( std::abs(f[i-1]) < std::abs(f[i]) )
			{
				float t = f[i-1];
				f[i-1] = f[i];
				f[i] = t;
				int j = idx[i-1];
				idx[i-1] = idx[i];
				idx[i] = j;
				swapped = true;
			}
		}		
	}while(swapped);

	for(int i=0;i<k;i++)
	{
		std::cout << idx[i] << "  " << f[i] << std::endl;
	}
	return 0;
}
