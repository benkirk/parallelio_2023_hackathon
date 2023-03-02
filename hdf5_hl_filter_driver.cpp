#include <random>
#include <assert.h>
#include <string.h>
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
  size_t input_buffer_len = 10000000;
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
    //if (ix % 10 == 0)
    //  std::cout << "uncompressed_data[" << ix << "]=" << rand_val << "\n";
  }

  // nvcomp example; shoudl work!!
  // {
  //   uint8_t *device_input_data;
  //   CUDA_CHECK(cudaMalloc(&device_input_data, input_buffer_len));
  //   CUDA_CHECK(cudaMemcpy(device_input_data, uncompressed_data.data(), input_buffer_len, cudaMemcpyDefault));
  //
  //   // Single roundtrip example
  //   decomp_compressed_with_manager_factory_example(device_input_data, input_buffer_len);
  //   CUDA_CHECK(cudaFree(device_input_data));
  //   std::cout << std::endl;
  // }

  // HDF5 filter: test cases
  const unsigned int cd_values[] = {0,1,2};
  const size_t cd_nelmts = sizeof(cd_nelmts) / sizeof(unsigned int);

  {
    // Path 1: device buffers in/out of filter
    {
      uint8_t *device_input_data;
      CUDA_CHECK(cudaMalloc(&device_input_data, input_buffer_len));
      CUDA_CHECK(cudaMemcpy(device_input_data, uncompressed_data.data(), input_buffer_len, cudaMemcpyDefault));

      // HDF5 filter: test device output:
      {
        std::cout << "Calling nvcomp_filter (compress)" << std::endl;

        const size_t ncompressed_bytes =
          nvcomp_filter(/* flags = */ 0, cd_nelmts, cd_values,
                        input_buffer_len, &input_buffer_len, (void**) &device_input_data);

        std::cout << "compressed #bytes = " << ncompressed_bytes << "\n\n";
      }

      // HDF5 filter: test device input:
      {
        std::cout << "Calling nvcomp_filter (uncompress)" << std::endl;

        const size_t uncompressed_bytes =
          nvcomp_filter(/* flags = */ H5Z_FLAG_REVERSE, cd_nelmts, cd_values,
                        input_buffer_len, &input_buffer_len, (void**) &device_input_data);

        std::cout << "uncompressed #bytes = " << uncompressed_bytes << "\n\n";

        assert (uncompressed_bytes == input_buffer_len);
      }
      CUDA_CHECK(cudaFree(device_input_data));
    }

    // Path 2: host buffers in/out of filter
    {
      uint8_t *host_input_data = (uint8_t *) malloc(sizeof(uint8_t) * input_buffer_len);
      memcpy (host_input_data, uncompressed_data.data(), sizeof(uint8_t) * input_buffer_len);

      // HDF5 filter: test host output:
      {
        std::cout << "Calling nvcomp_filter (compress)" << std::endl;

        input_buffer_len = uncompressed_data.size();

        const size_t ncompressed_bytes =
          nvcomp_filter(/* flags = */ 0, cd_nelmts, cd_values,
                        input_buffer_len, &input_buffer_len, (void**) &host_input_data);

        std::cout << "compressed #bytes = " << ncompressed_bytes << "\n\n";
      }

      // HDF5 filter: test host input:
      {
        std::cout << "Calling nvcomp_filter (uncompress)" << std::endl;

        const size_t uncompressed_bytes =
          nvcomp_filter(/* flags = */ H5Z_FLAG_REVERSE, cd_nelmts, cd_values,
                        input_buffer_len, &input_buffer_len, (void**) &host_input_data);

        std::cout << "uncompressed #bytes = " << uncompressed_bytes << "\n\n";

        assert (uncompressed_bytes == input_buffer_len);
      }

      for (size_t ix = 0; ix < input_buffer_len; ++ix)
        assert (uncompressed_data[ix] == host_input_data[ix]);

      free(host_input_data);
    }
  }


  return 0;
}
