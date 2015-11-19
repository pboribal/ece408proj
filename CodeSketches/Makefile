all: sparsecoding_test.cu utils.cpp
	nvcc -O3 -ccbin=g++ -Xcompiler -Wall,-Wextra,-Wshadow -o sparsecoding_test $^
	g++ -Wall -Werror testcasegen.cpp -o testcasegen
testcase:testcasegen
	./testcasegen < testcase1params.txt > testcase1.txt
	./testcasegen < testcase2params.txt > testcase2.txt	
	./testcasegen < testcase3params.txt > testcase3.txt
	./testcasegen < testcase4params.txt > testcase4.txt
	./testcasegen < testcase5params.txt > testcase5.txt
clean:
	rm -f sparsecoding_test
	rm -f testcasegen
	rm -f out*.txt
	rm -f qsub_sp_result.*
