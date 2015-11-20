README

files:
sparsecoding_test.cu - entry point for test program
sparsecoding.h - the parallel algorithm templates for sparse coding + device query code
testcasegen.cpp - test case generation code
utils.cpp - generic SEQUENTIAL algorithms

usage:

make - create executables
	=> sparsecoding_test - receives a test case (input + expected result) as input, computes using parallel implementation and compare it to the sequential algorithm results for correctness
	=> testcasegen - receives two numbers, N and K as inputs and outputs a test case file to be fed to the sparsecoding_test program.
 
make clean - clear output files and executables

make testcase - creates test cases 

qsub test.pbs - run test cases. qsub take a while to return result. Results are stored in outX.txt. The sparse-coding-test.* files are indicators that qsub is done, feel free to remove them if it clutters the folder.

edit files testcaseXparams.txt to specify N and K, respectively, then "make testcase" to generate the test cases.

Caveats:

Currently the algorithm supports very large N but has limitations on the value of K. The constraint as of now is that BLOCK_SIZE must be multiple of K, meaning K is also limited by the amount of shared memory available. A workaround (not yet implemented) could be to see if K>BLOCK_SIZE. If yes then start by sorting, in descending order, blocks of size BLOCK_SIZE. Then merge those sorted blocks to get K elements. Each merged part, then, are concatenated and and calls max_abs_k recursively until final K elements is obtained.

*** if K>BLOCK_SIZE works then test case 5 must pass 

