#include <iostream>
#include <stdlib.h>
#include <math.h>
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
		f[i] = (0.5-((float)rand())/((float)RAND_MAX))*2;
		idx[i] = i;
		std::cout << f[i] << " ";
	}
	std::cout << std::endl;
	// Bubble sort is enough, it's just a test case generator
	// no need to be performance critical here
	bool pass = false;
	while(!pass)
	{
		pass = true;
		for(int i=0;i<n-1;i++)
		{
			if(fabs(f[i])<fabs(f[i+1]))
			{
				float t = f[i];
				f[i] = f[i+1];
				f[i+1] = t;
				int j = idx[i];
				idx[i] = idx[i+1];
				idx[i+1] = j;
				pass = false;
			}
		}
	}
	for(int i=0;i<k;i++)
	{
		std::cout << idx[i] << "  " << f[i] << std::endl;
	}
	return 0;
}
