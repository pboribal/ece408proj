all: max_abs_k.cu
	nvcc -O3 -ccbin=g++ -Xcompiler -Wall,-Wextra,-Wshadow -o max_abs_k $^
clean:
	rm -f max_abs_k
