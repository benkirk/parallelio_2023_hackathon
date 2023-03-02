NVCOMP_DIR ?= /home/parallelio_development/src/nvcomp/nvcomp
CUDA_DIR ?= /usr/local/cuda
HDF5_DIR ?= /home/parallelio_development/sw

CXX := nvc++
CC  := nvc

CPPFLAGS ?= -I$(HDF5_DIR)/include -I$(CUDA_DIR)/include -I$(NVCOMP_DIR)/include -DCOMPILE_MAIN
CFLAGS ?= -g -fPIC
CXXFLAGS ?= -g -fPIC
LDFLAGS ?= -L$(CUDA_DIR)/lib64 -lcuda -lcudart -lcufile -Wl,-rpath,$(CUDA_DIR)/lib64 -L$(NVCOMP_DIR)/lib -lnvcomp -Wl,-rpath,$(NVCOMP_DIR)/lib -L$(HDF5_DIR)/lib -lhdf5 -Wl,-rpath,$(HDF5_DIR)/lib

all: high_level_quickstart_example low_level_quickstart_example filter_example nvcomp_gds gds_helloworld hdf5_hl_filter trivial_filter

# %: %.cpp Makefile
# 	unset CUDA_HOME && $(CXX) -o $@ $< -I$(HDF5_DIR)/include -I$(CUDA_DIR)/include -I$(NVCOMP_DIR)/include/ -L$(CUDA_DIR)/lib64 -lcuda -lcudart -lcufile -Wl,-rpath,$(CUDA_DIR)/lib64 -L$(NVCOMP_DIR)/lib -lnvcomp -Wl,-rpath,$(NVCOMP_DIR)/lib -L$(HDF5_DIR)/lib -lhdf5 -Wl,-rpath,$(HDF5_DIR)/lib

%: %.cu Makefile
	unset CUDA_HOME && $(CXX) -o $@ $< $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS)

# %: %.c Makefile
# 	unset CUDA_HOME && $(CC) -o $@ $< -I$(HDF5_DIR)/include -I$(CUDA_DIR)/include -I$(NVCOMP_DIR)/include/ -L$(CUDA_DIR)/lib64 -lcuda -lcudart -lcufile -Wl,-rpath,$(CUDA_DIR)/lib64 -L$(NVCOMP_DIR)/lib -lnvcomp -Wl,-rpath,$(NVCOMP_DIR)/lib -L$(HDF5_DIR)/lib -lhdf5 -Wl,-rpath,$(HDF5_DIR)/lib

clean:
	rm -f *_example *~ $$(find . -maxdepth 1 -type f -executable) *.o


filter_example: filter_example.o filter_example_driver.o

hdf5_hl_filter: hdf5_hl_filter.o hdf5_hl_filter_driver.o
	$(CXX) -o $@ $^ $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS)
