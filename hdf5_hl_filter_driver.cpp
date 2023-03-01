#include <random>
#include <assert.h>
#include <iostream>

#include "nvcomp/lz4.hpp"
#include "nvcomp.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

#include "hdf5_hl_filter.h"

#include <hdf5.h>
#include <H5Zpublic.h>


int main()
{
  // Initialize a random array of chars
  size_t input_buffer_len = 1000000;
  const size_t chunksize = input_buffer_len/100;

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

  uint8_t* device_input_data;
  CUDA_CHECK(cudaMalloc(&device_input_data, input_buffer_len));
  CUDA_CHECK(cudaMemcpy(device_input_data, uncompressed_data.data(), input_buffer_len, cudaMemcpyDefault));

  // Single roundtrip examples
  decomp_compressed_with_manager_factory_example(device_input_data, input_buffer_len);

  // HDF5 filter: test output:
  const unsigned int cd_values[] = {0,1,2};
  const size_t cd_nelmts = sizeof(cd_nelmts) / sizeof(unsigned int);
  unsigned int flags = 0;

  std::cout << "Calling nvcomp_filter (compress)" << std::endl;

  const size_t ncompressed_bytes =
    nvcomp_filter(flags, cd_nelmts, cd_values,
                  input_buffer_len, &input_buffer_len, (void**) &device_input_data);

  std::cout << "compressed #bytes = " << ncompressed_bytes << std::endl;

  // HDF5 filter: test input:
  flags = H5Z_FLAG_REVERSE;

  std::cout << "Calling nvcomp_filter (uncompress) NOT IMPLEMENTED!!" << std::endl;

  const size_t uncompressed_bytes =
    nvcomp_filter(flags, cd_nelmts, cd_values,
                  input_buffer_len, &input_buffer_len, (void**) &device_input_data);

  std::cout << "uncompressed #bytes = " << uncompressed_bytes << std::endl;

  return 0;
}
