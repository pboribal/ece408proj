all:
	nvcc -O3 -ccbin=g++ -Xcompiler -Wall sparsecoding_test.cu -o sparsecoding_test
	nvcc -O3 -ccbin=g++ -Xcompiler -Wall sparsemult_test.cu -o sparsemult_test
	g++ -O3 -Wall -Werror testcasegen.cpp -o testcasegen	
testcase:testcasegen
	echo "256 32" | ./testcasegen  > testcase1.txt
	echo "1024 128" | ./testcasegen  > testcase2.txt
	echo "65536 128" | ./testcasegen  > testcase3.txt
	echo "262144 512" | ./testcasegen  > testcase4.txt
clean:
	rm -f sparsecoding_test
	rm -f sparsemult_test
	rm -f testcasegen
	rm -f out*.txt
	rm -f qsub_sp_result.*
