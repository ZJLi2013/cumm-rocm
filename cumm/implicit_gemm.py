"""FlyDSL Implicit GEMM — output-tile-centric fused gather+GEMM+scatter.

Replaces 81 kernel launches (27 x gather + mm + scatter) with a single kernel.

Design (output-tile-centric):
  - Grid: (num_output_tiles,) where tile_id maps to output[tile*BLOCK_M : (tile+1)*BLOCK_M]
  - Block: (BLOCK_M,) — each thread owns one output row
  - Per workgroup:
    for kv in 0..KV-1 where mask[tile, kv] == 1:
      1. Cooperative load weight[kv] → LDS
      2. Each thread: lookup pair via sorted arrays → get inp_row
      3. Each thread: dot product features[inp_row, :] × weight_lds[:, :]
      4. Accumulate into register
    5. Store output (no atomicAdd needed — each output row owned by exactly one thread)

  Preprocessing (in C++/HIP):
    - Sort pairs by (kv, out_index) using hipCUB
    - Build mask[num_tiles, kv] + pair_start/pair_end ranges

Supports: fp32, fp16, bf16. Accumulation in f32.
"""
import math
import os
import warnings
from typing import Optional, Dict, Tuple

import torch

_COMPILED_KERNELS: Dict = {}
_HIP_MODULE = None


def _get_hip_module():
    """JIT compile the HIP indice pairs + mask generation extension."""
    global _HIP_MODULE
    if _HIP_MODULE is not None:
        return _HIP_MODULE

    try:
        from torch.utils.cpp_extension import load
        csrc_dir = os.path.join(os.path.dirname(__file__), 'csrc_hip')
        _HIP_MODULE = load(
            name='cumm_hip_indice',
            sources=[
                os.path.join(csrc_dir, 'indice_pairs_api.cpp'),
                os.path.join(csrc_dir, 'indice_pairs_kernel.hip'),
            ],
            extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3'],
            verbose=False,
        )
        return _HIP_MODULE
    except Exception as e:
        warnings.warn(f"Failed to compile HIP indice pairs module: {e}")
        return None


def _compile_implicit_gemm(c_in: int, c_out: int, kv: int, dtype_str: str):
    """Compile output-tile-centric FlyDSL implicit GEMM kernel."""
    import sys
    _flydsl_root = os.path.dirname(os.path.dirname(__import__('flydsl').__file__))
    if _flydsl_root not in sys.path:
        sys.path.insert(0, _flydsl_root)
    if '/opt/FlyDSL' not in sys.path and os.path.isdir('/opt/FlyDSL'):
        sys.path.insert(0, '/opt/FlyDSL')

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import gpu, arith, range_constexpr, const_expr, buffer_ops, rocdl
    from flydsl.expr.typing import T
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import scf, llvm
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.compiler.protocol import fly_values
    from flydsl._mlir.dialects import fly
    from kernels.tensor_shim import GTensor, STensor

    C_IN = c_in
    C_OUT = c_out
    KV = kv
    BLOCK_M = 64
    W_ELEMS = C_IN * C_OUT

    if dtype_str == 'f32':
        DT_BYTES = 4
    elif dtype_str in ('f16', 'bf16'):
        DT_BYTES = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    OUT_DT_BYTES = 4  # output is always f32

    allocator = SmemAllocator(None, arch="gfx942", global_sym_name="smem_ig_v2")
    smem_w_offset = allocator._align(allocator.ptr, 16)
    W_BYTES = W_ELEMS * DT_BYTES
    allocator.ptr = smem_w_offset + W_BYTES

    LOAD_ITERS = math.ceil(W_ELEMS / BLOCK_M)

    def _make_kernel_f32():
        @flyc.kernel(known_block_size=[BLOCK_M, 1, 1])
        def kernel(features: fx.Tensor, weights: fx.Tensor, output: fx.Tensor,
                   sorted_inp: fx.Tensor, sorted_out: fx.Tensor,
                   mask: fx.Tensor, pair_start: fx.Tensor, pair_end: fx.Tensor,
                   num_act_out_val: fx.Int32):
            _rc = range_constexpr
            dt = T.f32
            tid = fx.Int32(gpu.thread_idx.x)
            bid = fx.Int32(gpu.block_idx.x)

            feat_ = GTensor(features, dtype=dt, shape=(-1,))
            w_ = GTensor(weights, dtype=dt, shape=(-1,))
            sinp_ = GTensor(sorted_inp, dtype=T.i32, shape=(-1,))
            sout_ = GTensor(sorted_out, dtype=T.i32, shape=(-1,))
            mask_ = GTensor(mask, dtype=T.i32, shape=(-1,))
            ps_ = GTensor(pair_start, dtype=T.i32, shape=(-1,))
            pe_ = GTensor(pair_end, dtype=T.i32, shape=(-1,))

            base_ptr = allocator.get_base()
            smem_w_ptr = SmemPtr(base_ptr, smem_w_offset, dt, shape=(W_ELEMS,))
            w_lds = STensor(smem_w_ptr, dt, shape=(W_ELEMS,))

            my_out_row = fx.Int32(bid) * fx.Int32(const_expr(BLOCK_M)) + tid
            is_valid_out = arith.cmpi(arith.CmpIPredicate.slt, my_out_row, num_act_out_val)

            # Accumulator for output: C_OUT values per thread
            # Use raw pointer for output store (non-atomic)
            _ptr_type = ir.Type.parse("!llvm.ptr<1>")
            out_raw = fly_values(output)[0]
            out_base_ptr = fly.extract_aligned_pointer_as_index(_ptr_type, out_raw)
            out_base_int = llvm.PtrToIntOp(T.i64, out_base_ptr).result

            # Initialize accumulators
            # We'll accumulate across all kv positions, then store once
            # Since range_constexpr C_OUT can be large, use array in registers
            # FlyDSL doesn't have array; use output pointer directly after accumulation

            # For each kv position, check mask and accumulate
            for k in range_constexpr(KV):
                mask_idx = fx.Index(bid) * fx.Index(const_expr(KV)) + fx.Index(const_expr(k))
                is_active = arith.cmpi(arith.CmpIPredicate.ne,
                                       mask_.load(mask_idx),
                                       fx.Int32(arith.constant(0, type=T.i32)))
                kv_if = scf.IfOp(is_active, results_=[], has_else=False)
                with ir.InsertionPoint(kv_if.then_block):
                    # Cooperative weight load to LDS
                    w_base = fx.Index(const_expr(k * W_ELEMS))
                    for wi in range_constexpr(LOAD_ITERS):
                        w_idx = fx.Index(const_expr(wi * BLOCK_M)) + fx.Index(tid)
                        w_valid = arith.cmpi(arith.CmpIPredicate.ult, w_idx, fx.Index(const_expr(W_ELEMS)))
                        w_if = scf.IfOp(w_valid, results_=[], has_else=False)
                        with ir.InsertionPoint(w_if.then_block):
                            w_val = w_.load(w_base + w_idx)
                            w_lds[w_idx] = w_val
                            scf.YieldOp([])
                    gpu.barrier()

                    # Each thread: find my pair in sorted arrays for this (tile, kv)
                    ps_idx = fx.Index(bid) * fx.Index(const_expr(KV)) + fx.Index(const_expr(k))
                    p_start = ps_.load(ps_idx)
                    p_end = pe_.load(ps_idx)

                    valid_thread = arith.cmpi(arith.CmpIPredicate.slt, my_out_row, num_act_out_val)
                    thread_if = scf.IfOp(valid_thread, results_=[], has_else=False)
                    with ir.InsertionPoint(thread_if.then_block):
                        p_start_idx = arith.index_cast(T.index, p_start)
                        p_end_idx = arith.index_cast(T.index, p_end)
                        step_idx = arith.constant(1, type=T.index)

                        loop = scf.ForOp(p_start_idx, p_end_idx, step_idx, iter_args=[])
                        with ir.InsertionPoint(loop.body):
                            pi = loop.induction_variable
                            pair_out = sout_.load(pi)
                            is_mine = arith.cmpi(arith.CmpIPredicate.eq, pair_out, my_out_row)
                            mine_if = scf.IfOp(is_mine, results_=[], has_else=False)
                            with ir.InsertionPoint(mine_if.then_block):
                                inp_row = sinp_.load(pi)
                                feat_base = fx.Index(inp_row) * fx.Index(const_expr(C_IN))
                                out_row_base = fx.Index(my_out_row) * fx.Index(const_expr(C_OUT))
                                for j in range_constexpr(C_OUT):
                                    acc = arith.constant(0.0, type=T.f32)
                                    for c in range_constexpr(C_IN):
                                        f_off = feat_base + fx.Index(const_expr(c))
                                        f_val = feat_.load(f_off)
                                        w_lds_idx = fx.Index(const_expr(c * C_OUT + j))
                                        w_val = w_lds[w_lds_idx]
                                        acc = f_val * w_val + acc
                                    out_off = out_row_base + fx.Index(const_expr(j))
                                    byte_off = arith.index_cast(T.i64, out_off * fx.Index(const_expr(OUT_DT_BYTES)))
                                    addr_i64 = llvm.AddOp(out_base_int, byte_off, llvm.IntegerOverflowFlags(0)).result
                                    addr_ptr = llvm.IntToPtrOp(_ptr_type, addr_i64).result
                                    addr_v = addr_ptr._value if const_expr(hasattr(addr_ptr, "_value")) else addr_ptr
                                    acc_v = acc._value if const_expr(hasattr(acc, "_value")) else acc
                                    llvm.AtomicRMWOp(llvm.AtomicBinOp.fadd, addr_v, acc_v,
                                                     llvm.AtomicOrdering.monotonic, syncscope="agent", alignment=4)
                                scf.YieldOp([])
                            scf.YieldOp([])
                        scf.YieldOp([])
                    gpu.barrier()
                    scf.YieldOp([])
        return kernel

    def _make_kernel_fp16(use_f16=True):
        @flyc.kernel(known_block_size=[BLOCK_M, 1, 1])
        def kernel(features: fx.Tensor, weights: fx.Tensor, output: fx.Tensor,
                   sorted_inp: fx.Tensor, sorted_out: fx.Tensor,
                   mask: fx.Tensor, pair_start: fx.Tensor, pair_end: fx.Tensor,
                   num_act_out_val: fx.Int32):
            _rc = range_constexpr
            dt = T.f16 if const_expr(use_f16) else T.bf16
            tid = fx.Int32(gpu.thread_idx.x)
            bid = fx.Int32(gpu.block_idx.x)

            feat_ = GTensor(features, dtype=dt, shape=(-1,))
            w_ = GTensor(weights, dtype=dt, shape=(-1,))
            sinp_ = GTensor(sorted_inp, dtype=T.i32, shape=(-1,))
            sout_ = GTensor(sorted_out, dtype=T.i32, shape=(-1,))
            mask_ = GTensor(mask, dtype=T.i32, shape=(-1,))
            ps_ = GTensor(pair_start, dtype=T.i32, shape=(-1,))
            pe_ = GTensor(pair_end, dtype=T.i32, shape=(-1,))

            base_ptr = allocator.get_base()
            smem_w_ptr = SmemPtr(base_ptr, smem_w_offset, dt, shape=(W_ELEMS,))
            w_lds = STensor(smem_w_ptr, dt, shape=(W_ELEMS,))

            my_out_row = fx.Int32(bid) * fx.Int32(const_expr(BLOCK_M)) + tid

            _ptr_type = ir.Type.parse("!llvm.ptr<1>")
            out_raw = fly_values(output)[0]
            out_base_ptr = fly.extract_aligned_pointer_as_index(_ptr_type, out_raw)
            out_base_int = llvm.PtrToIntOp(T.i64, out_base_ptr).result

            for k in range_constexpr(KV):
                mask_idx = fx.Index(bid) * fx.Index(const_expr(KV)) + fx.Index(const_expr(k))
                is_active = arith.cmpi(arith.CmpIPredicate.ne,
                                       mask_.load(mask_idx),
                                       fx.Int32(arith.constant(0, type=T.i32)))
                kv_if = scf.IfOp(is_active, results_=[], has_else=False)
                with ir.InsertionPoint(kv_if.then_block):
                    w_base = fx.Index(const_expr(k * W_ELEMS))
                    for wi in range_constexpr(LOAD_ITERS):
                        w_idx = fx.Index(const_expr(wi * BLOCK_M)) + fx.Index(tid)
                        w_valid = arith.cmpi(arith.CmpIPredicate.ult, w_idx, fx.Index(const_expr(W_ELEMS)))
                        w_if = scf.IfOp(w_valid, results_=[], has_else=False)
                        with ir.InsertionPoint(w_if.then_block):
                            w_val = w_.load(w_base + w_idx)
                            w_lds[w_idx] = w_val
                            scf.YieldOp([])
                    gpu.barrier()

                    ps_idx = fx.Index(bid) * fx.Index(const_expr(KV)) + fx.Index(const_expr(k))
                    p_start = ps_.load(ps_idx)
                    p_end = pe_.load(ps_idx)

                    valid_thread = arith.cmpi(arith.CmpIPredicate.slt, my_out_row,
                                              num_act_out_val)
                    thread_if = scf.IfOp(valid_thread, results_=[], has_else=False)
                    with ir.InsertionPoint(thread_if.then_block):
                        p_start_idx = arith.index_cast(T.index, p_start)
                        p_end_idx = arith.index_cast(T.index, p_end)
                        step_idx = arith.constant(1, type=T.index)

                        loop = scf.ForOp(p_start_idx, p_end_idx, step_idx, iter_args=[])
                        with ir.InsertionPoint(loop.body):
                            pi = loop.induction_variable
                            pair_out = sout_.load(pi)
                            is_mine = arith.cmpi(arith.CmpIPredicate.eq, pair_out, my_out_row)
                            mine_if = scf.IfOp(is_mine, results_=[], has_else=False)
                            with ir.InsertionPoint(mine_if.then_block):
                                inp_row = sinp_.load(pi)
                                feat_base = fx.Index(inp_row) * fx.Index(const_expr(C_IN))
                                out_row_base = fx.Index(my_out_row) * fx.Index(const_expr(C_OUT))
                                for j in range_constexpr(C_OUT):
                                    acc = arith.constant(0.0, type=T.f32)
                                    for c in range_constexpr(C_IN):
                                        f_off = feat_base + fx.Index(const_expr(c))
                                        f_val = feat_.load(f_off)
                                        f_f32 = arith.extf(T.f32, f_val)
                                        w_lds_idx = fx.Index(const_expr(c * C_OUT + j))
                                        w_val = w_lds[w_lds_idx]
                                        w_f32 = arith.extf(T.f32, w_val)
                                        acc = f_f32 * w_f32 + acc
                                    out_off = out_row_base + fx.Index(const_expr(j))
                                    byte_off = arith.index_cast(T.i64, out_off * fx.Index(const_expr(OUT_DT_BYTES)))
                                    addr_i64 = llvm.AddOp(out_base_int, byte_off, llvm.IntegerOverflowFlags(0)).result
                                    addr_ptr = llvm.IntToPtrOp(_ptr_type, addr_i64).result
                                    addr_v = addr_ptr._value if const_expr(hasattr(addr_ptr, "_value")) else addr_ptr
                                    acc_v = acc._value if const_expr(hasattr(acc, "_value")) else acc
                                    llvm.AtomicRMWOp(llvm.AtomicBinOp.fadd, addr_v, acc_v,
                                                     llvm.AtomicOrdering.monotonic, syncscope="agent", alignment=4)
                                scf.YieldOp([])
                            scf.YieldOp([])
                        scf.YieldOp([])
                    gpu.barrier()
                    scf.YieldOp([])
        return kernel

    if dtype_str == 'f32':
        implicit_gemm_kernel = _make_kernel_f32()
    elif dtype_str == 'f16':
        implicit_gemm_kernel = _make_kernel_fp16(use_f16=True)
    else:
        implicit_gemm_kernel = _make_kernel_fp16(use_f16=False)

    @flyc.jit
    def launch_fn(
        features: fx.Tensor,
        weights: fx.Tensor,
        output: fx.Tensor,
        sorted_inp: fx.Tensor,
        sorted_out: fx.Tensor,
        mask: fx.Tensor,
        pair_start: fx.Tensor,
        pair_end: fx.Tensor,
        num_tiles: fx.Int32,
        num_act_out: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        implicit_gemm_kernel(
            features, weights, output,
            sorted_inp, sorted_out, mask, pair_start, pair_end,
            num_act_out,
        ).launch(
            grid=(num_tiles,),
            block=(BLOCK_M,),
            stream=stream,
        )

    return launch_fn, BLOCK_M


def implicit_gemm_forward(
    features: torch.Tensor,
    filters: torch.Tensor,
    indice_pairs: torch.Tensor,
    indice_pair_num: torch.Tensor,
    num_activate_out: int,
) -> Optional[torch.Tensor]:
    """Fused gather+GEMM+scatter via output-tile-centric FlyDSL implicit GEMM."""
    try:
        import flydsl
    except ImportError:
        return None

    hip = _get_hip_module()
    if hip is None:
        return None

    kv = filters.shape[0]
    c_in = features.shape[1]
    c_out = filters.shape[2]
    device = features.device
    dtype = features.dtype

    if c_in % 4 != 0 or c_out % 4 != 0:
        return None

    if dtype == torch.float32:
        dtype_str = 'f32'
    elif dtype == torch.float16:
        dtype_str = 'f16'
    elif dtype == torch.bfloat16:
        dtype_str = 'bf16'
    else:
        return None

    BLOCK_M = 64
    num_tiles = (num_activate_out + BLOCK_M - 1) // BLOCK_M

    # Build mask via C++/HIP (sort + mask generation on GPU)
    sorted_inp, sorted_out, sorted_kv, mask, pair_start, pair_end = \
        hip.build_implicit_gemm_mask(indice_pairs, indice_pair_num,
                                     num_activate_out, BLOCK_M)

    if sorted_inp.shape[0] == 0:
        return torch.zeros(num_activate_out, c_out, dtype=torch.float32, device=device)

    # Compile kernel (cached by config)
    key = (c_in, c_out, kv, dtype_str)
    if key not in _COMPILED_KERNELS:
        try:
            launch_fn, block_m = _compile_implicit_gemm(c_in, c_out, kv, dtype_str)
            _COMPILED_KERNELS[key] = (launch_fn, block_m)
        except Exception as e:
            import traceback
            warnings.warn(f"Failed to compile implicit GEMM kernel: {e}\n{traceback.format_exc()}")
            return None
    else:
        launch_fn, block_m = _COMPILED_KERNELS[key]

    out_features = torch.zeros(num_activate_out, c_out, dtype=torch.float32, device=device)
    weights_flat = filters.reshape(-1).contiguous()
    stream = torch.cuda.current_stream()

    launch_fn(
        features.contiguous(),
        weights_flat,
        out_features,
        sorted_inp, sorted_out,
        mask.reshape(-1).contiguous(),
        pair_start.reshape(-1).contiguous(),
        pair_end.reshape(-1).contiguous(),
        num_tiles, num_activate_out,
        stream,
    )

    if dtype != torch.float32:
        out_features = out_features.to(dtype)

    return out_features


# Keep old preprocess_pairs for backward compat / testing
def preprocess_pairs(
    indice_pairs: torch.Tensor,
    indice_pair_num: torch.Tensor,
    block_m: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Legacy preprocessing (kv-centric). Kept for testing."""
    kv = indice_pairs.shape[0]
    device = indice_pairs.device
    pair_num_cpu = indice_pair_num.cpu().int().numpy()

    total_pairs = int(pair_num_cpu.sum())
    if total_pairs == 0:
        empty = torch.zeros(0, dtype=torch.int32, device=device)
        return empty, empty, empty, empty, 0

    num_tiles_per_kv = [(int(n) + block_m - 1) // block_m if n > 0 else 0
                        for n in pair_num_cpu]
    num_tiles = sum(num_tiles_per_kv)
    if num_tiles == 0:
        empty = torch.zeros(0, dtype=torch.int32, device=device)
        return empty, empty, empty, empty, 0

    all_inp = torch.empty(total_pairs, dtype=torch.int32, device=device)
    all_out = torch.empty(total_pairs, dtype=torch.int32, device=device)
    all_kv = torch.empty(total_pairs, dtype=torch.int32, device=device)

    offset = 0
    for k in range(kv):
        nhot = int(pair_num_cpu[k])
        if nhot <= 0:
            continue
        all_inp[offset:offset + nhot] = indice_pairs[k, 0, :nhot]
        all_out[offset:offset + nhot] = indice_pairs[k, 1, :nhot]
        all_kv[offset:offset + nhot] = k
        offset += nhot

    max_idx = int(all_inp.max().item()) + 1
    sort_key = all_kv.long() * max_idx + all_inp.long()
    sorted_order = torch.argsort(sort_key)
    all_inp = all_inp[sorted_order]
    all_out = all_out[sorted_order]

    total_padded = num_tiles * block_m
    inp_flat = torch.zeros(total_padded, dtype=torch.int32, device=device)
    out_flat = torch.zeros(total_padded, dtype=torch.int32, device=device)
    tile_kpos = torch.empty(num_tiles, dtype=torch.int32, device=device)
    tile_pair_count = torch.empty(num_tiles, dtype=torch.int32, device=device)

    tile_idx = 0
    flat_offset = 0
    src_offset = 0
    for k in range(kv):
        nhot = int(pair_num_cpu[k])
        if nhot <= 0:
            continue
        inp_flat[flat_offset:flat_offset + nhot] = all_inp[src_offset:src_offset + nhot]
        out_flat[flat_offset:flat_offset + nhot] = all_out[src_offset:src_offset + nhot]
        n_tiles_k = num_tiles_per_kv[k]
        tile_kpos[tile_idx:tile_idx + n_tiles_k] = k
        for t in range(n_tiles_k):
            n_valid = min(block_m, nhot - t * block_m)
            tile_pair_count[tile_idx + t] = n_valid
        tile_idx += n_tiles_k
        flat_offset += n_tiles_k * block_m
        src_offset += nhot

    return inp_flat, out_flat, tile_kpos, tile_pair_count, num_tiles
