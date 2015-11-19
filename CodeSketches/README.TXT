README

usage:

make - create executables
make clean - clear output files and executables
make testcase - creates test cases 

qsub runX.pbs - run test case X. qsub take a while to return result. Results are stored in outX.txt. The sparse-coding-test.* files are indicators that qsub is done, feel free to remove them if it clutters the folder.

edit files testcaseXparams.txt to specify N and K, respectively.

CAUTION:

Currently the algorithm supports very large N but has limitations on the value of K. The constraint as of now is that BLOCK_SIZE must be multiple of K, meaning K is also limited by the amount of shared memory available. A workaround (not yet implemented) could be to see if K>BLOCK_SIZE. If yes then start by sorting, in descending order, blocks of size BLOCK_SIZE. Then merge those sorted blocks to get K elements. Each merged part, then, are concatenated and and calls max_abs_k recursively until final K elements is obtained.

Optimizations (IP = In Progress, OK = Done):

OK - strip common expressions and store into registers

OK - Pre-sort every K elements before actual reduction, so we can allocate just BLOCK_SIZE elements instead of BLOCK_SIZE*K elements

IP - try to merge_max_abs_k in-place, if succeed will increase upper bound of K
  by 2x
