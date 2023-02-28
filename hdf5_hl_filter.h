#ifndef HDF5_HL_FILTER_H
#define HDF5_HL_FILTER_H

#include <random>
#include <assert.h>
#include <iostream>

#include "nvcomp/lz4.hpp"
#include "nvcomp.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

#ifdef CUDA_CHECK
#  undef CUDA_CHECK
#endif
#define CUDA_CHECK(cond)                                                  \
  do {                                                                    \
    cudaError_t err = cond;                                               \
    if (err != cudaSuccess) {                                             \
      std::cerr << "Failure: "                                            \
                <<  __FILE__ << __LINE__ << std::endl;                    \
      exit(1);                                                            \
    }                                                                     \
  } while (false)

void decomp_compressed_with_manager_factory_example(uint8_t* device_input_ptrs, const size_t input_buffer_len);
void comp_decomp_with_single_manager(uint8_t* device_input_ptrs, const size_t input_buffer_len);
void multi_comp_decomp_example(const std::vector<uint8_t*>& device_input_ptrs, std::vector<size_t>& input_buffer_lengths);
void multi_comp_decomp_example_comp_config(const std::vector<uint8_t*>& device_input_ptrs, std::vector<size_t>& input_buffer_lengths);
void comp_decomp_with_single_manager_with_checksums(uint8_t* device_input_ptrs, const size_t input_buffer_len);
void decomp_compressed_with_manager_factory_with_checksums(uint8_t* device_input_ptrs, const size_t input_buffer_len);


#endif // #define HDF5_HL_FILTER_H
