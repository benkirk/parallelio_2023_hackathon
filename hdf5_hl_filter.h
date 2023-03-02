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

#define NVCOMP_FILTER_IDX 306;

extern "C"
{
  void register_nvcomp_filter();
}

void decomp_compressed_with_manager_factory_example(uint8_t* input_buf, const size_t input_buffer_len);

size_t nvcomp_filter( unsigned int flags, size_t cd_nelmts, const unsigned int cd_values[],
                      size_t nbytes, size_t *buf_size, void **buf);
#endif // #define HDF5_HL_FILTER_H
