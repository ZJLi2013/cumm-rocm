#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include "hash_table.h"

namespace cumm_hip {

constexpr int kMaxNDim = 4;

template <typename T>
__host__ __device__ __forceinline__
T coord_to_linear(const int* coord, const int* dims, int ndim) {
    T key = (T)coord[0];
    for (int d = 0; d < ndim; d++) {
        key = key * (T)dims[d] + (T)coord[d + 1];
    }
    return key;
}

__host__ __device__ __forceinline__
void filter_offset_to_ndim(int filter_offset, int* r, const int* ksize, int ndim) {
    for (int d = ndim - 1; d >= 0; d--) {
        r[d] = filter_offset % ksize[d];
        filter_offset /= ksize[d];
    }
}

// ============================================================================
// SubM convolution: hash table build + indice pairs
// ============================================================================

template <typename K>
__global__ void build_subm_hash_table_kernel(
    spconv_hip::LinearHashTable<K, int> table,
    const int* __restrict__ indices_in,
    const int* __restrict__ dims,
    int ndim, int num_indices)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_indices; i += gridDim.x * blockDim.x) {
        K key = coord_to_linear<K>(indices_in + i * (ndim + 1), dims, ndim);
        table.insert(key, i);
    }
}

template <typename K>
__global__ void calc_subm_conv_indices_kernel(
    spconv_hip::LinearHashTable<K, int> table,
    const int* __restrict__ indices_in,
    int* __restrict__ indice_pairs,
    int* __restrict__ indice_num_per_loc,
    const int* __restrict__ input_dims,
    const int* __restrict__ ksize_d,
    const int* __restrict__ dilation_d,
    int ndim, int num_indices, int indices_pair_size, int kv)
{
    int fo = blockIdx.y;
    int fo_mul_ips = fo * indices_pair_size;
    int mir = kv - 1 - fo;
    int mir_mul_ips = mir * indices_pair_size;
    int ips_mul_kv = indices_pair_size * kv;

    if (fo == kv / 2) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < num_indices; i += gridDim.x * blockDim.x) {
            indice_pairs[fo_mul_ips + i] = i;
            indice_pairs[ips_mul_kv + fo_mul_ips + i] = i;
        }
    } else {
        int r[kMaxNDim];
        filter_offset_to_ndim(fo, r, ksize_d, ndim);
        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < num_indices; i += gridDim.x * blockDim.x) {
            const int* cur = indices_in + i * (ndim + 1);
            int nb[kMaxNDim + 1];
            nb[0] = cur[0];
            bool valid = true;
            for (int d = 0; d < ndim; d++) {
                nb[d + 1] = cur[d + 1] + (r[d] - ksize_d[d] / 2) * dilation_d[d];
                if (nb[d + 1] < 0 || nb[d + 1] >= input_dims[d]) {
                    valid = false; break;
                }
            }
            if (valid) {
                K key = coord_to_linear<K>(nb, input_dims, ndim);
                int slot = table.lookup_offset(key);
                if (slot != -1) {
                    int v = table.values[slot];
                    int old = atomicAdd(indice_num_per_loc + fo, 1);
                    indice_pairs[fo_mul_ips + old] = v;
                    indice_pairs[ips_mul_kv + fo_mul_ips + old] = i;
                    indice_pairs[mir_mul_ips + old] = i;
                    indice_pairs[ips_mul_kv + mir_mul_ips + old] = v;
                }
            }
        }
    }
}

template <typename K>
void generate_subm_conv_inds(
    const int* indices_in, int* indice_pairs, int* indice_num_per_loc,
    int num_act_in, int batch_size,
    const int* input_dims_h, const int* ksize_h, const int* dilation_h,
    int ndim, hipStream_t stream)
{
    int kv = 1;
    for (int d = 0; d < ndim; d++) kv *= ksize_h[d];

    int hash_size = num_act_in * 2;
    if (hash_size < 128) hash_size = 128;

    K* hash_keys; int* hash_vals;
    hipMalloc(&hash_keys, hash_size * sizeof(K));
    hipMalloc(&hash_vals, hash_size * sizeof(int));

    int* d_dims; int* d_ks; int* d_dil;
    hipMalloc(&d_dims, ndim * sizeof(int));
    hipMalloc(&d_ks, ndim * sizeof(int));
    hipMalloc(&d_dil, ndim * sizeof(int));
    hipMemcpyAsync(d_dims, input_dims_h, ndim * sizeof(int), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_ks, ksize_h, ndim * sizeof(int), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_dil, dilation_h, ndim * sizeof(int), hipMemcpyHostToDevice, stream);

    spconv_hip::LinearHashTable<K, int> table(hash_keys, hash_vals, hash_size);
    int threads = 256;

    hipLaunchKernelGGL(spconv_hip::clear_hash_table_kernel<K>,
        dim3((hash_size + threads - 1) / threads), dim3(threads), 0, stream,
        hash_keys, hash_size);

    hipLaunchKernelGGL(build_subm_hash_table_kernel<K>,
        dim3((num_act_in + threads - 1) / threads), dim3(threads), 0, stream,
        table, indices_in, d_dims, ndim, num_act_in);

    int center = kv / 2;
    hipMemcpyAsync(indice_num_per_loc + center, &num_act_in, sizeof(int),
                   hipMemcpyHostToDevice, stream);

    dim3 grid((num_act_in + threads - 1) / threads, kv / 2 + 1);
    hipLaunchKernelGGL(calc_subm_conv_indices_kernel<K>,
        grid, dim3(threads), 0, stream,
        table, indices_in, indice_pairs, indice_num_per_loc,
        d_dims, d_ks, d_dil, ndim, num_act_in, num_act_in, kv);

    hipStreamSynchronize(stream);
    for (int k = 0; k < kv / 2; k++) {
        int mirror = kv - 1 - k;
        int count;
        hipMemcpy(&count, indice_num_per_loc + k, sizeof(int), hipMemcpyDeviceToHost);
        hipMemcpy(indice_num_per_loc + mirror, &count, sizeof(int), hipMemcpyHostToDevice);
    }

    hipFree(hash_keys); hipFree(hash_vals);
    hipFree(d_dims); hipFree(d_ks); hipFree(d_dil);
}

// ============================================================================
// Mask generation for output-tile-centric implicit GEMM
// ============================================================================

// Flatten indice_pairs[kv, 2, N] → contiguous (inp, out, kv_id) arrays.
// Also produces composite sort key = kv_id * stride + out_index.
__global__ void flatten_pairs_kernel(
    const int* __restrict__ indice_pairs,  // [kv, 2, N] — layout: [k][0=inp/1=out][p]
    const int* __restrict__ cum_offsets,    // [kv+1] cumulative pair counts
    const int* __restrict__ pair_num,       // [kv]
    int N,                                 // max pairs per kv (indices_pair_size)
    int kv,
    int sort_stride,                       // num_act_out + 1
    int* __restrict__ flat_inp,            // [total_pairs]
    int* __restrict__ flat_out,            // [total_pairs]
    int* __restrict__ flat_kv,             // [total_pairs]
    int64_t* __restrict__ sort_keys,       // [total_pairs]
    int* __restrict__ sort_vals)            // [total_pairs] = original index
{
    int total_slots = N * kv;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total_slots;
         i += gridDim.x * blockDim.x) {
        int k = i / N;
        int p = i % N;
        int nhot = pair_num[k];
        if (p >= nhot) continue;

        int dst = cum_offsets[k] + p;
        // [kv, 2, N] layout: inp at k*2*N + 0*N + p, out at k*2*N + 1*N + p
        int inp_val = indice_pairs[k * 2 * N + p];
        int out_val = indice_pairs[k * 2 * N + N + p];
        flat_inp[dst] = inp_val;
        flat_out[dst] = out_val;
        flat_kv[dst] = k;
        sort_keys[dst] = (int64_t)k * (int64_t)sort_stride + (int64_t)out_val;
        sort_vals[dst] = dst;
    }
}

// After sorting by (kv, out_index), reorder inp/out/kv arrays according to permutation.
__global__ void apply_permutation_kernel(
    const int* __restrict__ src_inp,
    const int* __restrict__ src_out,
    const int* __restrict__ src_kv,
    const int* __restrict__ perm,  // sort_vals after sort
    int* __restrict__ dst_inp,
    int* __restrict__ dst_out,
    int* __restrict__ dst_kv,
    int total)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += gridDim.x * blockDim.x) {
        int src_idx = perm[i];
        dst_inp[i] = src_inp[src_idx];
        dst_out[i] = src_out[src_idx];
        dst_kv[i] = src_kv[src_idx];
    }
}

// Build mask and pair ranges from sorted (kv, out_index) order.
// Each pair knows its out_index → tile = out_index / block_m, and its kv.
// For each (tile, kv): track whether active, and min/max position in sorted array.
__global__ void build_mask_kernel(
    const int* __restrict__ sorted_out,  // [total_pairs]
    const int* __restrict__ sorted_kv,   // [total_pairs]
    int* __restrict__ mask,              // [num_tiles * kv], init to 0
    int* __restrict__ pair_start,        // [num_tiles * kv], init to INT_MAX
    int* __restrict__ pair_end,          // [num_tiles * kv], init to 0
    int total_pairs, int num_tiles, int kv, int block_m)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total_pairs; i += gridDim.x * blockDim.x) {
        int out_idx = sorted_out[i];
        int k = sorted_kv[i];
        int tile = out_idx / block_m;
        if (tile >= num_tiles) continue;

        int slot = tile * kv + k;
        atomicMax(mask + slot, 1);
        atomicMin(pair_start + slot, i);
        atomicMax(pair_end + slot, i + 1);
    }
}

// Build inp_row_lut: for each (tile, kv, local_row), store the input row index.
// sorted pairs are ordered by (kv, out_index). For SubM conv each (out_row, kv)
// has at most 1 pair, so we can directly scatter into lut.
__global__ void build_lut_kernel(
    const int* __restrict__ sorted_inp,   // [total_pairs]
    const int* __restrict__ sorted_out,   // [total_pairs]
    const int* __restrict__ sorted_kv,    // [total_pairs]
    int* __restrict__ inp_row_lut,        // [num_tiles * kv * block_m], init to -1
    int total_pairs, int num_tiles, int kv, int block_m)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total_pairs; i += gridDim.x * blockDim.x) {
        int out_idx = sorted_out[i];
        int k = sorted_kv[i];
        int tile = out_idx / block_m;
        int local_row = out_idx % block_m;
        if (tile >= num_tiles) continue;
        inp_row_lut[tile * kv * block_m + k * block_m + local_row] = sorted_inp[i];
    }
}

// Host function: takes indice_pairs + indice_pair_num (outputs of get_indice_pairs),
// produces sorted pair arrays + mask + pair ranges + inp_row_lut for the implicit GEMM kernel.
void build_implicit_gemm_mask_gpu(
    const int* indice_pairs,
    const int* indice_pair_num,
    int kv, int N, int num_act_out, int block_m,
    // Outputs (caller allocates):
    int* sorted_inp, int* sorted_out, int* sorted_kv,
    int* mask, int* pair_start, int* pair_end,
    int* inp_row_lut,
    int total_pairs, int num_tiles,
    hipStream_t stream)
{
    if (total_pairs == 0) return;

    int threads = 256;

    // 1. Compute cumulative offsets on host
    int* pair_num_h = (int*)malloc(kv * sizeof(int));
    hipMemcpyAsync(pair_num_h, indice_pair_num, kv * sizeof(int), hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);

    int* cum_h = (int*)malloc((kv + 1) * sizeof(int));
    cum_h[0] = 0;
    for (int k = 0; k < kv; k++)
        cum_h[k + 1] = cum_h[k] + pair_num_h[k];

    int* d_cum; int* d_pnum;
    hipMalloc(&d_cum, (kv + 1) * sizeof(int));
    hipMalloc(&d_pnum, kv * sizeof(int));
    hipMemcpyAsync(d_cum, cum_h, (kv + 1) * sizeof(int), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_pnum, indice_pair_num, kv * sizeof(int), hipMemcpyDeviceToDevice, stream);

    // 2. Flatten + build sort keys
    int64_t* sort_keys; int* sort_vals;
    hipMalloc(&sort_keys, total_pairs * sizeof(int64_t));
    hipMalloc(&sort_vals, total_pairs * sizeof(int));

    // Temporary unsorted arrays
    int* tmp_inp; int* tmp_out; int* tmp_kv;
    hipMalloc(&tmp_inp, total_pairs * sizeof(int));
    hipMalloc(&tmp_out, total_pairs * sizeof(int));
    hipMalloc(&tmp_kv, total_pairs * sizeof(int));

    int sort_stride = num_act_out + 1;
    int total_slots = kv * N;
    int blocks_flat = (total_slots + threads - 1) / threads;
    hipLaunchKernelGGL(flatten_pairs_kernel,
        dim3(blocks_flat), dim3(threads), 0, stream,
        indice_pairs, d_cum, d_pnum,
        N, kv, sort_stride,
        tmp_inp, tmp_out, tmp_kv, sort_keys, sort_vals);

    // 3. Sort by composite key using hipCUB
    int64_t* sorted_keys;
    int* sorted_perm;
    hipMalloc(&sorted_keys, total_pairs * sizeof(int64_t));
    hipMalloc(&sorted_perm, total_pairs * sizeof(int));

    size_t sort_temp_bytes = 0;
    hipcub::DeviceRadixSort::SortPairs(nullptr, sort_temp_bytes,
        sort_keys, sorted_keys, sort_vals, sorted_perm,
        total_pairs, 0, sizeof(int64_t) * 8, stream);
    void* sort_temp;
    hipMalloc(&sort_temp, sort_temp_bytes);
    hipcub::DeviceRadixSort::SortPairs(sort_temp, sort_temp_bytes,
        sort_keys, sorted_keys, sort_vals, sorted_perm,
        total_pairs, 0, sizeof(int64_t) * 8, stream);

    // 4. Apply permutation to get sorted inp/out/kv
    int blocks_perm = (total_pairs + threads - 1) / threads;
    hipLaunchKernelGGL(apply_permutation_kernel,
        dim3(blocks_perm), dim3(threads), 0, stream,
        tmp_inp, tmp_out, tmp_kv, sorted_perm,
        sorted_inp, sorted_out, sorted_kv,
        total_pairs);

    // 5. Build mask and pair ranges
    // Initialize mask=0, pair_start=INT_MAX, pair_end=0
    hipMemsetAsync(mask, 0, num_tiles * kv * sizeof(int), stream);
    // pair_start needs INT_MAX init
    int* h_init = (int*)malloc(num_tiles * kv * sizeof(int));
    for (int i = 0; i < num_tiles * kv; i++) h_init[i] = 0x7FFFFFFF;
    hipMemcpyAsync(pair_start, h_init, num_tiles * kv * sizeof(int), hipMemcpyHostToDevice, stream);
    free(h_init);
    hipMemsetAsync(pair_end, 0, num_tiles * kv * sizeof(int), stream);

    int blocks_mask = (total_pairs + threads - 1) / threads;
    hipLaunchKernelGGL(build_mask_kernel,
        dim3(blocks_mask), dim3(threads), 0, stream,
        sorted_out, sorted_kv,
        mask, pair_start, pair_end,
        total_pairs, num_tiles, kv, block_m);

    // 6. Build inp_row_lut
    hipMemsetAsync(inp_row_lut, 0xFF, num_tiles * kv * block_m * sizeof(int), stream);  // -1
    hipLaunchKernelGGL(build_lut_kernel,
        dim3(blocks_mask), dim3(threads), 0, stream,
        sorted_inp, sorted_out, sorted_kv,
        inp_row_lut,
        total_pairs, num_tiles, kv, block_m);

    // Cleanup temporaries
    hipFree(sort_keys); hipFree(sort_vals);
    hipFree(sorted_keys); hipFree(sorted_perm);
    hipFree(sort_temp);
    hipFree(tmp_inp); hipFree(tmp_out); hipFree(tmp_kv);
    hipFree(d_cum); hipFree(d_pnum);
    free(pair_num_h); free(cum_h);
}

// Explicit instantiations
template void generate_subm_conv_inds<int32_t>(
    const int*, int*, int*, int, int, const int*, const int*, const int*, int, hipStream_t);
template void generate_subm_conv_inds<int64_t>(
    const int*, int*, int*, int, int, const int*, const int*, const int*, int, hipStream_t);

}  // namespace cumm_hip
