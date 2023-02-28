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

#include "hdf5_hl_filter.h"

using namespace nvcomp;


/**
 * In this example, we:
 *  1) compress the input data
 *  2) construct a new manager using the input data for demonstration purposes
 *  3) decompress the input data
 */
void decomp_compressed_with_manager_factory_example(uint8_t* device_input_ptrs, const size_t input_buffer_len)
{
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const int chunk_size = 1 << 16;
  nvcompType_t data_type = NVCOMP_TYPE_CHAR;

  LZ4Manager nvcomp_manager{chunk_size, data_type, stream};
  CompressionConfig comp_config = nvcomp_manager.configure_compression(input_buffer_len);

  uint8_t* comp_buffer;
  CUDA_CHECK(cudaMalloc(&comp_buffer, comp_config.max_compressed_buffer_size));

  nvcomp_manager.compress(device_input_ptrs, comp_buffer, comp_config);

  // Construct a new nvcomp manager from the compressed buffer.
  // Note we could use the nvcomp_manager from above, but here we demonstrate how to create a manager
  // for the use case where a buffer is received and the user doesn't know how it was compressed
  // Also note, creating the manager in this way synchronizes the stream, as the compressed buffer must be read to
  // construct the manager
  auto decomp_nvcomp_manager = create_manager(comp_buffer, stream);

  DecompressionConfig decomp_config = decomp_nvcomp_manager->configure_decompression(comp_buffer);

  // BSK: print uncompressed & compressed file sizes
  std::cout << "input_buffer_len = " << comp_config.uncompressed_buffer_size
            << ", compressed_buffer_len = " << decomp_nvcomp_manager->get_compressed_output_size(comp_buffer) << std::endl;


  uint8_t* res_decomp_buffer;
  CUDA_CHECK(cudaMalloc(&res_decomp_buffer, decomp_config.decomp_data_size));

  decomp_nvcomp_manager->decompress(res_decomp_buffer, comp_buffer, decomp_config);

  // TODO: compare device_input_ptrs & res_decomp_buffer
  {
    uint8_t *host_input_buffer, *host_res_buffer;
    CUDA_CHECK(cudaMallocHost((void**) &host_input_buffer, input_buffer_len));
    CUDA_CHECK(cudaMallocHost((void**) &host_res_buffer, input_buffer_len));
    cudaMemcpy((void*) host_input_buffer, (void*) device_input_ptrs, input_buffer_len, cudaMemcpyDeviceToHost);
    cudaMemcpy((void*) host_res_buffer,   (void*) res_decomp_buffer, input_buffer_len, cudaMemcpyDeviceToHost);

    for (size_t i=0; i<input_buffer_len; i++)
      {
        if (i % 10 == 0)
          std::cout << "host_input_buffer[" << i << "]=" << static_cast<short>(host_input_buffer[i])
                    << ", host_res_buffer[" << i << "]=" << static_cast<short>(host_res_buffer[i]) << '\n';
        assert(host_input_buffer[i] == host_res_buffer[i]);
      }
    std::cout << "SUCCESS: decomp matches comp!!\n";
    CUDA_CHECK(cudaFreeHost(host_input_buffer));
    CUDA_CHECK(cudaFreeHost(host_res_buffer));
  }

  CUDA_CHECK(cudaFree(comp_buffer));
  CUDA_CHECK(cudaFree(res_decomp_buffer));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaStreamDestroy(stream));
}

/**
 * In this example, we:
 *  1) construct an nvcompManager
 *  2) compress the input data
 *  3) decompress the input data
 */
void comp_decomp_with_single_manager(uint8_t* device_input_ptrs, const size_t input_buffer_len)
{
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const int chunk_size = 1 << 16;
  nvcompType_t data_type = NVCOMP_TYPE_CHAR;

  LZ4Manager nvcomp_manager{chunk_size, data_type, stream};
  CompressionConfig comp_config = nvcomp_manager.configure_compression(input_buffer_len);

  uint8_t* comp_buffer;
  CUDA_CHECK(cudaMalloc(&comp_buffer, comp_config.max_compressed_buffer_size));

  nvcomp_manager.compress(device_input_ptrs, comp_buffer, comp_config);

  DecompressionConfig decomp_config = nvcomp_manager.configure_decompression(comp_buffer);
  uint8_t* res_decomp_buffer;
  CUDA_CHECK(cudaMalloc(&res_decomp_buffer, decomp_config.decomp_data_size));

  nvcomp_manager.decompress(res_decomp_buffer, comp_buffer, decomp_config);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaFree(comp_buffer));
  CUDA_CHECK(cudaFree(res_decomp_buffer));

  CUDA_CHECK(cudaStreamDestroy(stream));
}

/**
 * Additionally, we can use the same manager to execute multiple streamed compressions / decompressions
 * In this example we configure the multiple decompressions by inspecting the compressed buffers
 */
void multi_comp_decomp_example(const std::vector<uint8_t*>& device_input_ptrs, std::vector<size_t>& input_buffer_lengths)
{
  size_t num_buffers = input_buffer_lengths.size();

  using namespace std;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const int chunk_size = 1 << 16;
  nvcompType_t data_type = NVCOMP_TYPE_CHAR;

  LZ4Manager nvcomp_manager{chunk_size, data_type, stream};

  std::vector<uint8_t*> comp_result_buffers(num_buffers);

  for(size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
    uint8_t* input_data = device_input_ptrs[ix_buffer];
    size_t input_length = input_buffer_lengths[ix_buffer];

    auto comp_config = nvcomp_manager.configure_compression(input_length);

    CUDA_CHECK(cudaMalloc(&comp_result_buffers[ix_buffer], comp_config.max_compressed_buffer_size));
    nvcomp_manager.compress(input_data, comp_result_buffers[ix_buffer], comp_config);
  }

  std::vector<uint8_t*> decomp_result_buffers(num_buffers);
  for(size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
    uint8_t* comp_data = comp_result_buffers[ix_buffer];

    auto decomp_config = nvcomp_manager.configure_decompression(comp_data);

    CUDA_CHECK(cudaMalloc(&decomp_result_buffers[ix_buffer], decomp_config.decomp_data_size));

    nvcomp_manager.decompress(decomp_result_buffers[ix_buffer], comp_data, decomp_config);
  }

  for (size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
    CUDA_CHECK(cudaFree(decomp_result_buffers[ix_buffer]));
    CUDA_CHECK(cudaFree(comp_result_buffers[ix_buffer]));
  }
}

/**
 * Additionally, we can use the same manager to execute multiple streamed compressions / decompressions
 * In this example we configure the multiple decompressions by storing the comp_config's and inspecting those
 */
void multi_comp_decomp_example_comp_config(const std::vector<uint8_t*>& device_input_ptrs, std::vector<size_t>& input_buffer_lengths)
{
  size_t num_buffers = input_buffer_lengths.size();

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const int chunk_size = 1 << 16;
  nvcompType_t data_type = NVCOMP_TYPE_CHAR;

  LZ4Manager nvcomp_manager{chunk_size, data_type, stream};

  std::vector<CompressionConfig> comp_configs;
  comp_configs.reserve(num_buffers);

  std::vector<uint8_t*> comp_result_buffers(num_buffers);

  for(size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
    uint8_t* input_data = device_input_ptrs[ix_buffer];
    size_t input_length = input_buffer_lengths[ix_buffer];

    comp_configs.push_back(nvcomp_manager.configure_compression(input_length));
    auto& comp_config = comp_configs.back();

    CUDA_CHECK(cudaMalloc(&comp_result_buffers[ix_buffer], comp_config.max_compressed_buffer_size));

    nvcomp_manager.compress(input_data, comp_result_buffers[ix_buffer], comp_config);
  }

  std::vector<uint8_t*> decomp_result_buffers(num_buffers);
  for(size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
    auto decomp_config = nvcomp_manager.configure_decompression(comp_configs[ix_buffer]);

    CUDA_CHECK(cudaMalloc(&decomp_result_buffers[ix_buffer], decomp_config.decomp_data_size));

    nvcomp_manager.decompress(decomp_result_buffers[ix_buffer], comp_result_buffers[ix_buffer], decomp_config);
  }

  for (size_t ix_buffer = 0; ix_buffer < num_buffers; ++ix_buffer) {
    CUDA_CHECK(cudaFree(decomp_result_buffers[ix_buffer]));
    CUDA_CHECK(cudaFree(comp_result_buffers[ix_buffer]));
  }
}

/**
 * In this example, we:
 *  1) construct an nvcompManager with checksum support enabled
 *  2) compress the input data
 *  3) decompress the input data
 */
void comp_decomp_with_single_manager_with_checksums(uint8_t* device_input_ptrs, const size_t input_buffer_len)
{
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const int chunk_size = 1 << 16;
  nvcompType_t data_type = NVCOMP_TYPE_CHAR;

  /*
   * There are 5 possible modes for checksum processing as
   * described below.
   *
   * Mode: NoComputeNoVerify
   * Description:
   *   - During compression, do not compute checksums
   *   - During decompression, do not verify checksums
   *
   * Mode: ComputeAndNoVerify
   * Description:
   *   - During compression, compute checksums
   *   - During decompression, do not attempt to verify checksums
   *
   * Mode: NoComputeAndVerifyIfPresent
   * Description:
   *   - During compression, do not compute checksums
   *   - During decompression, verify checksums if they were included
   *
   * Mode: ComputeAndVerifyIfPresent
   * Description:
   *   - During compression, compute checksums
   *   - During decompression, verify checksums if they were included
   *
   * Mode: ComputeAndVerify
   * Description:
   *   - During compression, compute checksums
   *   - During decompression, verify checksums. A runtime error will be
   *     thrown upon configure_decompression if checksums were not
   *     included in the compressed buffer.
   */

  int gpu_num = 0;

  // manager constructed with checksum mode as final argument
  LZ4Manager nvcomp_manager{chunk_size, data_type, stream, gpu_num, ComputeAndVerify};
  CompressionConfig comp_config = nvcomp_manager.configure_compression(input_buffer_len);

  uint8_t* comp_buffer;
  CUDA_CHECK(cudaMalloc(&comp_buffer, comp_config.max_compressed_buffer_size));

  // Checksums are computed and stored for uncompressed and compressed buffers during compression
  nvcomp_manager.compress(device_input_ptrs, comp_buffer, comp_config);

  DecompressionConfig decomp_config = nvcomp_manager.configure_decompression(comp_buffer);
  uint8_t* res_decomp_buffer;
  CUDA_CHECK(cudaMalloc(&res_decomp_buffer, decomp_config.decomp_data_size));

  // Checksums are computed for compressed and decompressed buffers and verified against those
  // stored during compression
  nvcomp_manager.decompress(res_decomp_buffer, comp_buffer, decomp_config);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  /*
   * After synchronizing the stream, the nvcomp status can be checked to see if
   * the checksums were successfully verified. Provided no unrelated nvcomp errors occurred,
   * if the checksums were successfully verified, the status will be nvcompSuccess. Otherwise,
   * it will be nvcompErrorBadChecksum.
   */
  nvcompStatus_t final_status = *decomp_config.get_status();
  if(final_status == nvcompErrorBadChecksum) {
    throw std::runtime_error("One or more checksums were incorrect.\n");
  }

  CUDA_CHECK(cudaFree(comp_buffer));
  CUDA_CHECK(cudaFree(res_decomp_buffer));

  CUDA_CHECK(cudaStreamDestroy(stream));
}



void decomp_compressed_with_manager_factory_with_checksums(
  uint8_t* device_input_ptrs, const size_t input_buffer_len)
{
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const int chunk_size = 1 << 16;
  nvcompType_t data_type = NVCOMP_TYPE_CHAR;

  int gpu_num = 0;

  /*
   * For a full description of the checksum modes, see the above example. Here, the
   * constructed manager will compute checksums on compression, but not verify them
   * on decompression.
   */
  LZ4Manager nvcomp_manager{chunk_size, data_type, stream, gpu_num, ComputeAndNoVerify};
  CompressionConfig comp_config = nvcomp_manager.configure_compression(input_buffer_len);

  uint8_t* comp_buffer;
  CUDA_CHECK(cudaMalloc(&comp_buffer, comp_config.max_compressed_buffer_size));

  nvcomp_manager.compress(device_input_ptrs, comp_buffer, comp_config);

  // Construct a new nvcomp manager from the compressed buffer.
  // Note we could use the nvcomp_manager from above, but here we demonstrate how to create a manager
  // for the use case where a buffer is received and the user doesn't know how it was compressed
  // Also note, creating the manager in this way synchronizes the stream, as the compressed buffer must be read to
  // construct the manager. This manager is configured to verify checksums on decompression if they were
  // supplied in the compressed buffer. For a full description of the checksum modes, see the
  // above example.
  auto decomp_nvcomp_manager = create_manager(comp_buffer, stream, gpu_num, NoComputeAndVerifyIfPresent);

  DecompressionConfig decomp_config = decomp_nvcomp_manager->configure_decompression(comp_buffer);
  uint8_t* res_decomp_buffer;
  CUDA_CHECK(cudaMalloc(&res_decomp_buffer, decomp_config.decomp_data_size));

  decomp_nvcomp_manager->decompress(res_decomp_buffer, comp_buffer, decomp_config);

  CUDA_CHECK(cudaFree(comp_buffer));
  CUDA_CHECK(cudaFree(res_decomp_buffer));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  /*
   * After synchronizing the stream, the nvcomp status can be checked to see if
   * the checksums were successfully verified. Provided no unrelated nvcomp errors occurred,
   * if the checksums were successfully verified, the status will be nvcompSuccess. Otherwise,
   * it will be nvcompErrorBadChecksum.
   */
  nvcompStatus_t final_status = *decomp_config.get_status();
  if(final_status == nvcompErrorBadChecksum) {
    throw std::runtime_error("One or more checksums were incorrect.\n");
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
}
