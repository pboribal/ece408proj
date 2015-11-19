all: sparsecoding_test.cu utils.cpp
	nvcc -O3 -ccbin=g++ -Xcompiler -Wall,-Wextra,-Wshadow -o sparsecoding_test $^
	g++ testcasegen.cpp -o testcasegen
testcase:testcasegen testcase1params.txt testcase2params.txt testcase3params.txt
	./testcasegen < testcase1params.txt > testcase1.txt
	./testcasegen < testcase2params.txt > testcase2.txt
	./testcasegen < testcase3params.txt > testcase3.txt
clean:
	rm -f sparsecoding_test
	rm -f testcasegen
	rm -f out1.txt
	rm -f out2.txt
	rm -f out3.txt
	rm -f qsub_sp_result.*
