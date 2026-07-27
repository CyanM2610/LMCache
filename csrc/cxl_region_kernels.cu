// SPDX-License-Identifier: Apache-2.0

#include "cxl_region_kernels.cuh"

void cxl_region_block_kv_transfer(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<int64_t> region_object_ptrs, const torch::Tensor& block_ids,
    const torch::Device& device, PageBufferShapeDesc shape_desc,
    TransferDirection direction, int lmcache_chunk_size,
    EngineKVFormat engine_kv_format, int skip_prefix_n_blocks) {
  multi_layer_block_kv_transfer(
      paged_buffer_ptrs_tensor, std::move(region_object_ptrs), block_ids,
      device, direction, shape_desc, lmcache_chunk_size, engine_kv_format,
      skip_prefix_n_blocks);
}
