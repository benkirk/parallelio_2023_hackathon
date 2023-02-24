NVCOMP_DIR ?= /home/parallelio_development/src/nvcomp/nvcomp
CUDA_DIR ?= /usr/local/cuda
HDF5_DIR ?= /home/parallelio_development/sw

CXX ?= nvc++
CC  ?= nvc

all: high_level_quickstart_example low_level_quickstart_example filter_example nvcomp_gds

%: %.cpp Makefile
	unset CUDA_HOME && $(CXX) -o $@ $< -I$(HDF5_DIR)/include -I$(CUDA_DIR)/include -I$(NVCOMP_DIR)/include/ -L$(CUDA_DIR)/lib64 -lcuda -lcudart -lcufile -Wl,-rpath,$(CUDA_DIR)/lib64 -L$(NVCOMP_DIR)/lib -lnvcomp -Wl,-rpath,$(NVCOMP_DIR)/lib -L$(HDF5_DIR)/lib -lhdf5 -Wl,-rpath,$(HDF5_DIR)/lib

%: %.cu Makefile
	unset CUDA_HOME && $(CXX) -o $@ $< -I$(HDF5_DIR)/include -I$(CUDA_DIR)/include -I$(NVCOMP_DIR)/include/ -L$(CUDA_DIR)/lib64 -lcuda -lcudart -lcufile -Wl,-rpath,$(CUDA_DIR)/lib64 -L$(NVCOMP_DIR)/lib -lnvcomp -Wl,-rpath,$(NVCOMP_DIR)/lib -L$(HDF5_DIR)/lib -lhdf5 -Wl,-rpath,$(HDF5_DIR)/lib

%: %.c Makefile
	unset CUDA_HOME && $(CC) -o $@ $< -I$(HDF5_DIR)/include -I$(CUDA_DIR)/include -I$(NVCOMP_DIR)/include/ -L$(CUDA_DIR)/lib64 -lcuda -lcudart -lcufile -Wl,-rpath,$(CUDA_DIR)/lib64 -L$(NVCOMP_DIR)/lib -lnvcomp -Wl,-rpath,$(NVCOMP_DIR)/lib -L$(HDF5_DIR)/lib -lhdf5 -Wl,-rpath,$(HDF5_DIR)/lib

clean:
	rm -f *_example *~ $$(find . -maxdepth 1 -type f -executable)
