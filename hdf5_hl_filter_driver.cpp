#include <random>
#include <assert.h>
#include <iostream>

#include "nvcomp/lz4.hpp"
#include "nvcomp.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

#include "hdf5_hl_filter.h"


int main()
{
  // Initialize a random array of chars
  const size_t input_buffer_len = 1000000, chunksize = input_buffer_len/100;

  std::vector<uint8_t> uncompressed_data(input_buffer_len);

  std::mt19937 random_gen(42);

  // char specialization of std::uniform_int_distribution is
  // non-standard, and isn't available on MSVC, so use short instead,
  // but with the range limited, and then cast below.
  std::uniform_int_distribution<short> uniform_dist(0, 255);
  short rand_val = uniform_dist(random_gen);
  for (size_t ix = 0; ix < input_buffer_len; ++ix) {
    if (ix % chunksize == 0)
      rand_val = uniform_dist(random_gen);
    uncompressed_data[ix] = static_cast<uint8_t>(rand_val);
    if (ix % 10 == 0)
      std::cout << "uncompressed_data[" << ix << "]=" << rand_val << "\n";
  }

  uint8_t* device_input_ptrs;
  CUDA_CHECK(cudaMalloc(&device_input_ptrs, input_buffer_len));
  CUDA_CHECK(cudaMemcpy(device_input_ptrs, uncompressed_data.data(), input_buffer_len, cudaMemcpyDefault));

  // Four roundtrip examples
  decomp_compressed_with_manager_factory_example(device_input_ptrs, input_buffer_len);
  decomp_compressed_with_manager_factory_with_checksums(device_input_ptrs, input_buffer_len);
  comp_decomp_with_single_manager(device_input_ptrs, input_buffer_len);
  comp_decomp_with_single_manager_with_checksums(device_input_ptrs, input_buffer_len);

  CUDA_CHECK(cudaFree(device_input_ptrs));

  // Multi buffer example
  const size_t num_buffers = 10;

  std::vector<uint8_t*> gpu_buffers(num_buffers);
  std::vector<size_t> input_buffer_lengths(num_buffers);

  std::vector<std::vector<uint8_t>> uncompressed_buffers(num_buffers);
  for (size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
    uncompressed_buffers[ix_buffer].resize(input_buffer_len);
    for (size_t ix_byte = 0; ix_byte < input_buffer_len; ++ix_byte) {
      uncompressed_buffers[ix_buffer][ix_byte] = static_cast<uint8_t>(uniform_dist(random_gen));
    }
    CUDA_CHECK(cudaMalloc(&gpu_buffers[ix_buffer], input_buffer_len));
    CUDA_CHECK(cudaMemcpy(gpu_buffers[ix_buffer], uncompressed_buffers[ix_buffer].data(), input_buffer_len, cudaMemcpyDefault));
    input_buffer_lengths[ix_buffer] = input_buffer_len;
  }

  multi_comp_decomp_example(gpu_buffers, input_buffer_lengths);
  multi_comp_decomp_example_comp_config(gpu_buffers, input_buffer_lengths);

  for (size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
    CUDA_CHECK(cudaFree(gpu_buffers[ix_buffer]));
  }
  return 0;
}
