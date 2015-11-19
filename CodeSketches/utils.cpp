#include "utils.hpp"
#include <stdlib.h>
#include <algorithm>

template<typename K, typename V>
void mergekv(K* keys, V* values, int start, int mid, int end, bool ascending)
{	
	int size = end - start;
	int ia = start;
	int ib = mid;
	int i = 0;
	K* ktmp = (K*)malloc(sizeof(K)*size);
	V* vtmp = (V*)malloc(sizeof(V)*size);
	while(ia < mid && ib < end)
	{
		int inca = (ascending && keys[ia] < keys[ib]) || (!ascending && keys[ia]>keys[ib]) || (keys[ia]==keys[ib]);
		ktmp[i] = inca ? keys[ia] : keys[ib];
		vtmp[i] = inca ? values[ia]: values[ib];
		ia += inca;
		ib += !inca;
		i++;
	}
	while(ia < mid) 
	{
		ktmp[i] = keys[ia];
		vtmp[i] = values[ia];
		ia++;
		i++;
	}
	while(ib < end)
	{
		ktmp[i] = keys[ib];
		vtmp[i] = values[ib];
		ib++;
		i++;
	}
	for(i=0;i<size;i++)
	{
		keys[start+i] = ktmp[i];
		values[start+i] = vtmp[i];
	}
	free(ktmp);
	free(vtmp);
}
template<typename K, typename V>
void sortkv_r(K* keys, V* values,int start, int end,bool ascending)
{
	int size = end-start;
	if(size<2) return;
	if(size==2) 
	{
		if( (ascending && keys[start+1] < keys[start])||(!ascending && keys[start+1] > keys[start]) )
		{			
			std::swap(keys[start],keys[start+1]);
			std::swap(values[start],values[start+1]);
		}
	} else {
		int mid = (start+end)/2;
		sortkv_r(keys,values,start, mid,ascending);
		sortkv_r(keys,values,mid, end,ascending);
		mergekv(keys,values,start,mid,end,ascending);
	}
}

/*
* Performs a sort in ascending order
*/
template<typename K, typename V>
void sortkv(K* keys, V* values, int N,bool ascending)
{
	sortkv_r(keys,values,0,N,ascending);	
}

