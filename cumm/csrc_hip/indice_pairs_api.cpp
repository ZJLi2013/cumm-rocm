#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

namespace cumm_hip {

void build_implicit_gemm_mask_gpu(
    const int* indice_pairs, const int* indice_pair_num,
    int kv, int N, int num_act_out, int block_m,
    int* sorted_inp, int* sorted_out, int* sorted_kv,
    int* mask, int* pair_start, int* pair_end,
    int* inp_row_lut,
    int total_pairs, int num_tiles,
    hipStream_t stream);

}  // namespace cumm_hip

// Build mask for output-tile-centric implicit GEMM.
// Input: indice_pairs [kv, 2, N], indice_pair_num [kv], num_act_out, block_m
// Returns: (sorted_inp, sorted_out, sorted_kv, mask, pair_start, pair_end)
//   sorted_inp/out/kv: [total_pairs] sorted by (kv, out_index)
//   mask:       [num_tiles, kv]  — 1 if active
//   pair_start: [num_tiles, kv]  — start in sorted arrays
//   pair_end:   [num_tiles, kv]  — end (exclusive) in sorted arrays
std::vector<torch::Tensor> build_implicit_gemm_mask(
    torch::Tensor indice_pairs,
    torch::Tensor indice_pair_num,
    int64_t num_act_out,
    int64_t block_m)
{
    TORCH_CHECK(indice_pairs.is_cuda(), "indice_pairs must be on GPU");
    TORCH_CHECK(indice_pairs.dim() == 3, "indice_pairs must be [kv, 2, N]");

    int kv = indice_pairs.size(0);
    int N = indice_pairs.size(2);
    int num_tiles = ((int)num_act_out + (int)block_m - 1) / (int)block_m;

    // Compute total_pairs on CPU
    auto pn_cpu = indice_pair_num.to(torch::kCPU, torch::kInt32);
    int total_pairs = 0;
    for (int k = 0; k < kv; k++)
        total_pairs += pn_cpu.data_ptr<int>()[k];

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(indice_pairs.device());

    if (total_pairs == 0) {
        auto empty = torch::zeros({0}, options);
        auto mask = torch::zeros({num_tiles, kv}, options);
        auto ps = torch::full({num_tiles, kv}, 0x7FFFFFFF, options);
        auto pe = torch::zeros({num_tiles, kv}, options);
        auto lut = torch::full({num_tiles, kv, (int)block_m}, -1, options);
        return {empty, empty, empty, mask, ps, pe, lut};
    }

    auto sorted_inp = torch::empty({total_pairs}, options);
    auto sorted_out = torch::empty({total_pairs}, options);
    auto sorted_kv = torch::empty({total_pairs}, options);
    auto mask = torch::zeros({num_tiles, kv}, options);
    auto pair_start = torch::full({num_tiles, kv}, 0x7FFFFFFF, options);
    auto pair_end = torch::zeros({num_tiles, kv}, options);
    auto inp_row_lut = torch::full({num_tiles, kv, (int)block_m}, -1, options);

    hipStream_t stream = (hipStream_t)at::cuda::getCurrentCUDAStream().stream();

    cumm_hip::build_implicit_gemm_mask_gpu(
        indice_pairs.data_ptr<int>(),
        indice_pair_num.data_ptr<int>(),
        kv, N, (int)num_act_out, (int)block_m,
        sorted_inp.data_ptr<int>(),
        sorted_out.data_ptr<int>(),
        sorted_kv.data_ptr<int>(),
        mask.data_ptr<int>(),
        pair_start.data_ptr<int>(),
        pair_end.data_ptr<int>(),
        inp_row_lut.data_ptr<int>(),
        total_pairs, num_tiles,
        stream);

    return {sorted_inp, sorted_out, sorted_kv, mask, pair_start, pair_end, inp_row_lut};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_implicit_gemm_mask", &build_implicit_gemm_mask,
          "Build mask for output-tile-centric implicit GEMM");
}
