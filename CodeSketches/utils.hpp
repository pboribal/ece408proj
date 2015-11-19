#ifndef UTILS_HPP
#define UTILS_HPP
// SEQUENTIAL sort
// entry function for sorting two arrays simultaneously using 
template<typename K, typename V>
void sortkv(K* keys, V* values, int N,bool ascending=true);

template<typename K, typename V>
void mergekv(K* keys, V* values, int start, int mid, int end,bool ascending);

template<typename K, typename V>
void sortkv_r(K* keys, V* values,int start, int end,bool ascending);


#endif

