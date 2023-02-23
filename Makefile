NVCOMP_DIR ?= /home/parallelio_development/src/nvcomp/nvcomp
CUDA_DIR ?= /usr/local/cuda
HDF5_DIR ?= /home/parallelio_development/sw

all: high_level_quickstart_example low_level_quickstart_example

%: %.cpp Makefile
	unset CUDA_HOME && nvc++ -o $@ $< -I$(NVCOMP_DIR)/include/ -L$(CUDA_DIR)/lib64 -lcuda -lcudart -Wl,-rpath,$(CUDA_DIR)/lib64 -L$(NVCOMP_DIR)/lib -lnvcomp -Wl,-rpath,$(NVCOMP_DIR)/lib

%: %.c Makefile
	unset CUDA_HOME && nvc -o $@ $< -I$(NVCOMP_DIR)/include/ -L$(CUDA_DIR)/lib64 -lcuda -lcudart -Wl,-rpath,$(CUDA_DIR)/lib64 -L$(NVCOMP_DIR)/lib -lnvcomp -Wl,-rpath,$(NVCOMP_DIR)/lib

clean:
	rm -f *_example *~
