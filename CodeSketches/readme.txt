README

usage:

make - create executables
make clean - clear output files and executables
make testcase - creates test cases 

qsub runX.pbs - run test case X. qsub take a while to return result. Results are stored in outX.txt. The sparse-coding-test.* files are indicators that qsub is done, feel free to remove them if it clutters the folder.

edit files testcaseXparams.txt to specify N and K, respectively.

CAUTION:
Currently the algorithm supports high N but has problems with K. The maximum value of K is approximately inverse proportional to the constant BLOCK_SIZE in maxabsk_test.cu, namely K<4 for BLOCK_SIZE==1024 and K<13 for BLOCK_SIZE==256. This is solely due to the limit in shared memory available per block. We need to restructure the algorithm a bit to lift this constraint...

Possible optimizations (I will do this thursday):
- try to merge_max_abs_k in-place, if succeed will increase upper bound of K
  by 2x
- try to not pad each element into K slots, instead pack and pre-sort chunks
  of K elements in-place right in the original array. If success, will reduce
shared memory usage by K times! (better goal than above)

