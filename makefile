CC		  = gcc
LDFLAGS	 = -lm

NVCC		= nvcc
NVCC_FLAGS  = -O3
LD_FLAGS	= -lcudart

default: err_sample

err_sample.o: err_sampling.cu calc_delta.cuh
	$(NVCC) -c -o err_sample.o err_sampling.cu $(NVCC_FLAGS)

err_sample: err_sample.o
	$(NVCC) err_sample.o -o err_sample $(LD_FLAGS)

clean:
	rm -rf *.o err_sample output/* delta.dat
