/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <random>
#include <assert.h>
#include <iostream>

#include "nvcomp/lz4.hpp"
#include "nvcomp.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <hdf5.h>
#include <H5Zpublic.h>


#include "hdf5_hl_filter.h"

using namespace nvcomp;


// /**
//  * In this example, we:
//  *  1) compress the input data
//  *  2) construct a new manager using the input data for demonstration purposes
//  *  3) decompress the input data
//  */
// void decomp_compressed_with_manager_factory_example(uint8_t* device_input_ptrs, const size_t input_buffer_len)
// {
//   cudaStream_t stream;
//   CUDA_CHECK(cudaStreamCreate(&stream));
//
//   const int chunk_size = 1 << 16;
//   nvcompType_t data_type = NVCOMP_TYPE_CHAR;
//
//   //LZ4Manager nvcomp_manager{chunk_size, data_type, stream};
//   BitcompManager nvcomp_manager{data_type, 0, stream};
//   CompressionConfig comp_config = nvcomp_manager.configure_compression(input_buffer_len);
//
//   uint8_t* comp_buffer;
//   CUDA_CHECK(cudaMalloc(&comp_buffer, comp_config.max_compressed_buffer_size));
//
//   nvcomp_manager.compress(device_input_ptrs, comp_buffer, comp_config);
//
//   // Construct a new nvcomp manager from the compressed buffer.
//   // Note we could use the nvcomp_manager from above, but here we demonstrate how to create a manager
//   // for the use case where a buffer is received and the user doesn't know how it was compressed
//   // Also note, creating the manager in this way synchronizes the stream, as the compressed buffer must be read to
//   // construct the manager
//   auto decomp_nvcomp_manager = create_manager(comp_buffer, stream);
//
//   DecompressionConfig decomp_config = decomp_nvcomp_manager->configure_decompression(comp_buffer);
//
//   // BSK: print uncompressed & compressed file sizes
//   std::cout << "input_buffer_len = " << comp_config.uncompressed_buffer_size
//             << ", compressed_buffer_len = " << decomp_nvcomp_manager->get_compressed_output_size(comp_buffer) << std::endl;
//
//
//   uint8_t* res_decomp_buffer;
//   CUDA_CHECK(cudaMalloc(&res_decomp_buffer, decomp_config.decomp_data_size));
//
//   decomp_nvcomp_manager->decompress(res_decomp_buffer, comp_buffer, decomp_config);
//
//   // Debug: compare device_input_ptrs & res_decomp_buffer
//   {
//     uint8_t *host_input_buffer, *host_res_buffer;
//     CUDA_CHECK(cudaMallocHost((void**) &host_input_buffer, input_buffer_len));
//     CUDA_CHECK(cudaMallocHost((void**) &host_res_buffer, input_buffer_len));
//     cudaMemcpy((void*) host_input_buffer, (void*) device_input_ptrs, input_buffer_len, cudaMemcpyDeviceToHost);
//     cudaMemcpy((void*) host_res_buffer,   (void*) res_decomp_buffer, input_buffer_len, cudaMemcpyDeviceToHost);
//
//     for (size_t i=0; i<input_buffer_len; i++)
//       {
//         //if (i % 10 == 0)
//         //  std::cout << "host_input_buffer[" << i << "]=" << static_cast<short>(host_input_buffer[i])
//         //            << ", host_res_buffer[" << i << "]=" << static_cast<short>(host_res_buffer[i]) << '\n';
//         assert(host_input_buffer[i] == host_res_buffer[i]);
//       }
//     std::cout << "SUCCESS: decomp matches comp!!\n";
//     CUDA_CHECK(cudaFreeHost(host_input_buffer));
//     CUDA_CHECK(cudaFreeHost(host_res_buffer));
//   }
//
//   CUDA_CHECK(cudaFree(comp_buffer));
//   CUDA_CHECK(cudaFree(res_decomp_buffer));
//
//   CUDA_CHECK(cudaStreamSynchronize(stream));
//
//   CUDA_CHECK(cudaStreamDestroy(stream));
// }



bool nvc_is_device_pointer (const void *ptr)
{
    struct cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, ptr);
    return (attributes.devicePointer != NULL);
}


// ref: https://docs.hdfgroup.org/hdf5/develop/_f_i_l_t_e_r.html
size_t nvcomp_filter( unsigned int flags,
                      size_t cd_nelmts,
                      const unsigned int cd_values[],
                      size_t nbytes,
                      size_t *buf_size, void **buf)
{
  const bool reading_input = flags & H5Z_FLAG_REVERSE;

  cudaStream_t stream;
  const int chunk_size = 1 << 16;
  const nvcompType_t data_type = NVCOMP_TYPE_CHAR;
  const size_t input_buffer_len = nbytes;

  fprintf(stderr,
          "nvcomp_filter called with reading_input=%d, buf_size=%d, nbytes=%d, cd_nelmts=%d\n",
          reading_input,
          *buf_size,
          nbytes,
          cd_nelmts);

  const bool input_buf_on_device = nvc_is_device_pointer(*buf);

  fprintf(stderr,
          input_buf_on_device ? "input buf is on device\n" : "input buf on host\n");


  /* Reading_Input: we will decompress buf and replace, in place */
  if (reading_input)
    {
      CUDA_CHECK(cudaStreamCreate(&stream));

      // **buf could live anywhere - handle accordingly
      // if its already on the device, use in place.
      // if not,  we'll allocate a device buffer and copy the contents
      uint8_t *device_data_to_uncompress = input_buf_on_device ? (uint8_t*) *buf : NULL;

      if (device_data_to_uncompress == NULL)
        {
          CUDA_CHECK(cudaMalloc(&device_data_to_uncompress, nbytes));
          CUDA_CHECK(cudaMemcpy(device_data_to_uncompress, *buf, input_buffer_len, cudaMemcpyHostToDevice));
        }

      // Construct a new nvcomp manager from the compressed buffer.
      // Creating the manager in this way synchronizes the stream, as the compressed buffer must be read to
      // construct the manager
      auto decomp_nvcomp_manager = create_manager(device_data_to_uncompress, stream);

      DecompressionConfig decomp_config = decomp_nvcomp_manager->configure_decompression(device_data_to_uncompress);

      uint8_t* decomp_buffer;
      const size_t nbytes_uncompressed = decomp_config.decomp_data_size;

      CUDA_CHECK(cudaMalloc(&decomp_buffer, nbytes_uncompressed));

      decomp_nvcomp_manager->decompress(decomp_buffer, device_data_to_uncompress, decomp_config);

      if (!input_buf_on_device)
        {
          CUDA_CHECK(cudaFree(device_data_to_uncompress));
        }

      std::cout << "nbytes_uncompressed = " << nbytes_uncompressed << std::endl;

      // reallocation required
      if (nbytes_uncompressed > *buf_size)
        {
          // ... on the host
          if (!input_buf_on_device)
            {
              *buf = realloc(*buf, nbytes_uncompressed);
              *buf_size = nbytes_uncompressed;
            }
          // ... on the device
          else
            {
              // Ben: fix this. redundant, just use decomp_buffer??
              CUDA_CHECK(cudaFree(*buf));
              CUDA_CHECK(cudaMalloc(buf, nbytes_uncompressed));
              *buf_size = nbytes_uncompressed;
            }
        }

      // put decomp_buffer into buf, wherever it lives
      CUDA_CHECK(cudaMemcpy(*buf, decomp_buffer, nbytes_uncompressed,
                            input_buf_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost));

      CUDA_CHECK(cudaFree(decomp_buffer));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaStreamDestroy(stream));

      return nbytes_uncompressed;
    }

  /* Output: we will compress buf and replace, in place */
  else
    {
      CUDA_CHECK(cudaStreamCreate(&stream));

      // **buf could live anywhere - handle accordingly
      // if its already on the device, use in place.
      // if not,  we'll allocate a device buffer and copy the contents
      uint8_t *device_data_to_compress = input_buf_on_device ? (uint8_t*) *buf : NULL;

      if (device_data_to_compress == NULL)
        {
          CUDA_CHECK(cudaMalloc(&device_data_to_compress, nbytes));
          CUDA_CHECK(cudaMemcpy(device_data_to_compress, *buf, nbytes, cudaMemcpyHostToDevice));
        }

      //LZ4Manager nvcomp_manager{chunk_size, data_type, stream};
      BitcompManager nvcomp_manager{data_type, 0, stream};
      CompressionConfig comp_config = nvcomp_manager.configure_compression(input_buffer_len);

      uint8_t* comp_buffer;
      CUDA_CHECK(cudaMalloc(&comp_buffer, comp_config.max_compressed_buffer_size));

      nvcomp_manager.compress(device_data_to_compress, comp_buffer, comp_config);

      if (!input_buf_on_device)
        {
          CUDA_CHECK(cudaFree(device_data_to_compress));
        }

      const size_t nbytes_compressed = nvcomp_manager.get_compressed_output_size(comp_buffer);

      std::cout << "nbytes (after Bitcomp compression): " << nbytes_compressed << std::endl;

      // make sure the compressed buffer is smaller than the input buffer, otherwise we'll need to increase buf
      //assert (nbytes <= input_buffer_len);
      //assert (nbytes <= *buf_size);

      // reallocation required
      if (nbytes_compressed > *buf_size)
        {
          // ... on the host
          if (!input_buf_on_device)
            {
              *buf = realloc(*buf, nbytes_compressed);
              *buf_size = nbytes_compressed;
            }
          // ... on the device
          else
            {
              // Ben: fix this. redundant, just use decomp_buffer
              CUDA_CHECK(cudaFree(*buf));
              CUDA_CHECK(cudaMalloc(buf, nbytes_compressed));
              *buf_size = nbytes_compressed;
            }
        }

      CUDA_CHECK(cudaMemcpy(*buf, comp_buffer, nbytes_compressed,
                            input_buf_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost));

      CUDA_CHECK(cudaFree(comp_buffer));

      CUDA_CHECK(cudaStreamSynchronize(stream));

      CUDA_CHECK(cudaStreamDestroy(stream));

      return nbytes_compressed;
    }

  /*fail*/
  return 0;
}

extern "C"
{
  void register_nvcomp_filter()
  {
  // ref: https://docs.hdfgroup.org/archive/support/HDF5/doc/RM/RM_H5Z.html

  /*
   typedef struct H5Z_class2_t {
            int version;
            H5Z_filter_t id;
            unsigned encoder_present;
            unsigned decoder_present;
            const char  *name;
            H5Z_can_apply_func_t can_apply;
            H5Z_set_local_func_t set_local;
            H5Z_func_t filter;
        } H5Z_class2_t;
  */

  /* registrer the filter */
  H5Z_class_t filter_spec;
  filter_spec.version = H5Z_CLASS_T_VERS;

  filter_spec.id = NVCOMP_FILTER_IDX;
  filter_spec.name = "nvcomp filter";
  filter_spec.can_apply = NULL;
  filter_spec.set_local = NULL;
  filter_spec.filter = nvcomp_filter;


  herr_t status = H5Zregister(&filter_spec);
}
}
