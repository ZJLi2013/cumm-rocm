"""FlyDSL Implicit GEMM — fused gather+GEMM+scatter for sparse convolution.

Replaces 81 kernel launches (27 x gather + mm + scatter) with a single kernel.

Design:
  - Grid: (num_tiles,) where tiles = all (kv_position, tile_m_start) pairs
  - Block: (BLOCK_M,) — each thread handles one pair
  - Per workgroup:
    1. Load weight[k_pos] to LDS (shared, coalesced)
    2. Each thread: indirect-load features[inp_idx[pair], :] (quasi-coalesced after sorting)
    3. Each thread: compute dot products with weight in LDS
    4. Each thread: atomic scatter output[out_idx[pair], :] (f32 atomic)

  Host preprocessing sorts pairs within each kv group by inp_indices
  to restore quasi-coalesced access patterns.

Supports: fp32, fp16, bf16. Accumulation in f32; output written as f32 atomicAdd.
"""
import math
import warnings
from typing import Optional, Dict, Tuple

import torch

_COMPILED_KERNELS: Dict = {}


def _compile_implicit_gemm(c_in: int, c_out: int, dtype_str: str):
    """Compile FlyDSL fused gather+GEMM+scatter kernel.

    Thread mapping:
      BLOCK_M threads per workgroup, each thread handles one (inp, out) pair.
      Workgroups are tiled over kv positions and pair ranges.

    Weight loading:
      Cooperative: all BLOCK_M threads load weight[k_pos, :, :] into LDS.
      Weight is row-major in LDS: w_lds[c * C_OUT + j] for coalesced reads
      in the inner j loop.

    Feature loading (indirect):
      Each thread loads features[inp_indices[global_pair_idx], c] — indirect access.
      Because pairs are sorted by inp_indices in preprocessing, neighboring threads
      access nearby feature rows → quasi-coalesced.

    Output scatter (atomic):
      Each thread atomically adds its result to output[out_indices[pair], j].
      We always accumulate in f32 and use f32 atomicAdd for simplicity.
    """
    import sys
    import os
    # FlyDSL's kernels/ lives under the FlyDSL install root (e.g. /opt/FlyDSL)
    _flydsl_root = os.path.dirname(os.path.dirname(__import__('flydsl').__file__))
    if _flydsl_root not in sys.path:
        sys.path.insert(0, _flydsl_root)
    # Also check /opt/FlyDSL (common Docker location)
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
    BLOCK_M = 64
    W_ELEMS = C_IN * C_OUT

    # Output always in f32 for atomic correctness
    OUT_DT_BYTES = 4

    if dtype_str == 'f32':
        dt = T.f32
        DT_BYTES = 4
    elif dtype_str == 'f16':
        dt = T.f16
        DT_BYTES = 2
    elif dtype_str == 'bf16':
        dt = T.bf16
        DT_BYTES = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    # LDS for weight tile (manual offset management, matching hgemm_splitk pattern)
    allocator = SmemAllocator(None, arch="gfx942", global_sym_name="smem_ig")
    smem_w_offset = allocator._align(allocator.ptr, 16)
    W_BYTES = W_ELEMS * DT_BYTES
    allocator.ptr = smem_w_offset + W_BYTES

    # How many iterations each thread does to cooperatively load weight to LDS
    LOAD_ITERS = math.ceil(W_ELEMS / BLOCK_M)

    @flyc.kernel(known_block_size=[BLOCK_M, 1, 1])
    def implicit_gemm_kernel(
        features: fx.Tensor,        # [N_in, C_in], dt
        weights: fx.Tensor,         # [kv * C_in * C_out], dt (flattened)
        output: fx.Tensor,          # [N_out, C_out], f32
        inp_indices: fx.Tensor,     # [total_padded_pairs], i32
        out_indices: fx.Tensor,     # [total_padded_pairs], i32
        tile_kpos: fx.Tensor,       # [num_tiles], i32
        tile_pair_count: fx.Tensor, # [num_tiles], i32
        total_pairs_val: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        bid = fx.Int32(gpu.block_idx.x)

        # Wrap tensors
        feat_ = GTensor(features, dtype=dt, shape=(-1,))
        w_ = GTensor(weights, dtype=dt, shape=(-1,))
        out_ = GTensor(output, dtype=T.f32, shape=(-1,))
        inp_idx_ = GTensor(inp_indices, dtype=T.i32, shape=(-1,))
        out_idx_ = GTensor(out_indices, dtype=T.i32, shape=(-1,))
        kpos_ = GTensor(tile_kpos, dtype=T.i32, shape=(-1,))
        tpc_ = GTensor(tile_pair_count, dtype=T.i32, shape=(-1,))

        # Read tile metadata
        k_pos = kpos_.load(fx.Index(bid))
        n_valid = tpc_.load(fx.Index(bid))

        # LDS for weight: SmemPtr → STensor (matches hgemm_splitk pattern)
        base_ptr = allocator.get_base()
        smem_w_ptr = SmemPtr(base_ptr, smem_w_offset, dt, shape=(W_ELEMS,))
        w_lds = STensor(smem_w_ptr, dt, shape=(W_ELEMS,))

        # --- Cooperative weight loading to LDS ---
        w_base_offset = fx.Index(k_pos) * fx.Index(const_expr(W_ELEMS))
        for wi in range_constexpr(LOAD_ITERS):
            w_idx = fx.Index(const_expr(wi * BLOCK_M)) + fx.Index(tid)
            is_valid = arith.cmpi(arith.CmpIPredicate.ult, w_idx, fx.Index(const_expr(W_ELEMS)))
            w_if = scf.IfOp(is_valid, results_=[], has_else=False)
            with ir.InsertionPoint(w_if.then_block):
                w_val = w_.load(w_base_offset + w_idx)
                w_lds[w_idx] = w_val
                scf.YieldOp([])

        gpu.barrier()

        # --- Per-thread: gather + GEMM + scatter ---
        is_active = arith.cmpi(arith.CmpIPredicate.slt, fx.Int32(tid), n_valid)
        active_if = scf.IfOp(is_active, results_=[], has_else=False)
        with ir.InsertionPoint(active_if.then_block):
            global_pair_idx = fx.Index(bid) * fx.Index(const_expr(BLOCK_M)) + fx.Index(tid)

            # Indirect index lookups
            inp_row = inp_idx_.load(global_pair_idx)
            out_row = out_idx_.load(global_pair_idx)

            feat_row_base = fx.Index(inp_row) * fx.Index(const_expr(C_IN))
            out_row_base = fx.Index(out_row) * fx.Index(const_expr(C_OUT))

            # Extract output base pointer once (hoisted from inner loop)
            _ptr_type = ir.Type.parse("!llvm.ptr<1>")
            out_raw = fly_values(output)[0]
            out_base_ptr = fly.extract_aligned_pointer_as_index(_ptr_type, out_raw)
            out_base_int = llvm.PtrToIntOp(T.i64, out_base_ptr).result

            # Compute output[out_row, j] += sum_c(features[inp_row, c] * weight[c, j])
            for j in range_constexpr(C_OUT):
                acc = arith.constant(0.0, type=T.f32)

                for c in range_constexpr(C_IN):
                    f_off = feat_row_base + fx.Index(const_expr(c))
                    f_val = feat_.load(f_off)
                    if const_expr(dtype_str != 'f32'):
                        f_f32 = arith.ExtFOp(T.f32, f_val)
                    else:
                        f_f32 = f_val

                    w_lds_idx = fx.Index(const_expr(c * C_OUT + j))
                    w_val = w_lds[w_lds_idx]
                    if const_expr(dtype_str != 'f32'):
                        w_f32 = arith.ExtFOp(T.f32, w_val)
                    else:
                        w_f32 = w_val

                    acc = arith.FMAOp(f_f32, w_f32, acc,
                                      fastmath=arith.FastMathFlags.fast)

                # Atomic f32 add to output
                out_elem_off = out_row_base + fx.Index(const_expr(j))
                byte_off = arith.index_cast(T.i64, out_elem_off * fx.Index(const_expr(OUT_DT_BYTES)))
                addr_i64 = llvm.AddOp(out_base_int, byte_off, llvm.IntegerOverflowFlags(0)).result
                addr_ptr = llvm.IntToPtrOp(_ptr_type, addr_i64).result

                addr_v = addr_ptr._value if const_expr(hasattr(addr_ptr, "_value")) else addr_ptr
                acc_v = acc._value if const_expr(hasattr(acc, "_value")) else acc
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.fadd,
                    addr_v, acc_v,
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent", alignment=4)

            scf.YieldOp([])

    @flyc.jit
    def launch_fn(
        features: fx.Tensor,
        weights: fx.Tensor,
        output: fx.Tensor,
        inp_indices: fx.Tensor,
        out_indices: fx.Tensor,
        tile_kpos: fx.Tensor,
        tile_pair_count: fx.Tensor,
        num_tiles: fx.Int32,
        total_pairs: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        implicit_gemm_kernel(
            features, weights, output,
            inp_indices, out_indices, tile_kpos, tile_pair_count,
            total_pairs,
        ).launch(
            grid=(num_tiles,),
            block=(BLOCK_M,),
            stream=stream,
        )

    return launch_fn, BLOCK_M


def preprocess_pairs(
    indice_pairs: torch.Tensor,     # [kv, 2, N_max]
    indice_pair_num: torch.Tensor,  # [kv]
    block_m: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Sort pairs within each kv group by inp_indices for coalesced access.

    Returns:
        inp_indices_flat: [num_tiles * block_m] int32, padded
        out_indices_flat: [num_tiles * block_m] int32, padded
        tile_kpos: [num_tiles] int32
        tile_pair_count: [num_tiles] int32
        num_tiles: int
    """
    kv = indice_pairs.shape[0]
    device = indice_pairs.device
    pair_num_cpu = indice_pair_num.cpu().int().numpy()

    tile_kpos_list = []
    tile_pair_count_list = []
    inp_flat_list = []
    out_flat_list = []

    for k in range(kv):
        nhot = int(pair_num_cpu[k])
        if nhot <= 0:
            continue

        inp_k = indice_pairs[k, 0, :nhot]
        out_k = indice_pairs[k, 1, :nhot]

        sorted_order = torch.argsort(inp_k)
        inp_k = inp_k[sorted_order]
        out_k = out_k[sorted_order]

        for tile_start in range(0, nhot, block_m):
            tile_end = min(tile_start + block_m, nhot)
            n_valid = tile_end - tile_start

            inp_tile = inp_k[tile_start:tile_end]
            out_tile = out_k[tile_start:tile_end]

            if n_valid < block_m:
                pad = block_m - n_valid
                inp_tile = torch.cat([inp_tile, torch.zeros(pad, dtype=torch.int32, device=device)])
                out_tile = torch.cat([out_tile, torch.zeros(pad, dtype=torch.int32, device=device)])

            inp_flat_list.append(inp_tile)
            out_flat_list.append(out_tile)
            tile_kpos_list.append(k)
            tile_pair_count_list.append(n_valid)

    num_tiles = len(tile_kpos_list)
    if num_tiles == 0:
        empty = torch.zeros(0, dtype=torch.int32, device=device)
        return empty, empty, empty, empty, 0

    inp_flat = torch.cat(inp_flat_list).int().contiguous()
    out_flat = torch.cat(out_flat_list).int().contiguous()
    tile_kpos = torch.tensor(tile_kpos_list, dtype=torch.int32, device=device)
    tile_pair_count = torch.tensor(tile_pair_count_list, dtype=torch.int32, device=device)

    return inp_flat, out_flat, tile_kpos, tile_pair_count, num_tiles


def implicit_gemm_forward(
    features: torch.Tensor,         # [N_in, C_in]
    filters: torch.Tensor,          # [kv, C_in, C_out]
    indice_pairs: torch.Tensor,     # [kv, 2, N_max]
    indice_pair_num: torch.Tensor,  # [kv]
    num_activate_out: int,
) -> Optional[torch.Tensor]:
    """Fused gather+GEMM+scatter via FlyDSL implicit GEMM.

    Returns output tensor, or None if FlyDSL is not available.
    """
    try:
        import flydsl
    except ImportError:
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

    key = (c_in, c_out, dtype_str)
    if key not in _COMPILED_KERNELS:
        try:
            launch_fn, block_m = _compile_implicit_gemm(c_in, c_out, dtype_str)
            _COMPILED_KERNELS[key] = (launch_fn, block_m)
        except Exception as e:
            warnings.warn(f"Failed to compile implicit GEMM kernel: {e}")
            return None
    else:
        launch_fn, block_m = _COMPILED_KERNELS[key]

    inp_flat, out_flat, tile_kpos, tile_pc, num_tiles = preprocess_pairs(
        indice_pairs, indice_pair_num, block_m)

    if num_tiles == 0:
        return torch.zeros(num_activate_out, c_out, dtype=torch.float32, device=device)

    # Output always f32 for atomic correctness
    out_features = torch.zeros(num_activate_out, c_out, dtype=torch.float32, device=device)
    total_pairs = inp_flat.shape[0]

    stream = torch.cuda.current_stream()

    # Flatten weights for contiguous access
    weights_flat = filters.reshape(-1).contiguous()

    launch_fn(
        features.contiguous(),
        weights_flat,
        out_features,
        inp_flat, out_flat,
        tile_kpos, tile_pc,
        num_tiles, total_pairs,
        stream,
    )

    # Cast back to input dtype if needed
    if dtype != torch.float32:
        out_features = out_features.to(dtype)

    return out_features
