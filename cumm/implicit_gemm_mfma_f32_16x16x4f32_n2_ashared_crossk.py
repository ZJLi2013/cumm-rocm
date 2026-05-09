"""Cross-kv K-fused implicit GEMM (Step 8).

Merges kv × C_IN into a unified K_total reduction with a runtime scf.for loop,
replacing the compile-time unrolled per-kv structure.  This enables larger
BLOCK_K and true software-pipeline steady state.

Key differences from _kpipe.py:
  - Host precomputes active_kv_ids[m_tile, max_active] + active_count[m_tile]
  - Kernel uses runtime scf.for over K_total = active_count * C_IN
  - Row map reload is conditional on kv boundary (c_offset_in_kv == 0)
  - BLOCK_K constrained to <= C_IN (no cross-kv-boundary tiles)
"""
import math
import warnings
from typing import Dict, Optional

import torch

from cumm.implicit_gemm_common import (
    _dtype_to_str,
    _ensure_flydsl_path,
    _get_hip_module,
    _pack_weights,
)


CROSSK_COMPILED_KERNELS: Dict = {}


def _build_active_kv_ids(mask: torch.Tensor, kv: int):
    """Build active_kv_ids[num_tiles, kv] and active_count[num_tiles] from mask.

    mask: [num_tiles, kv] int32, nonzero = active.
    Returns:
        active_kv_ids: [num_tiles, kv] int32, packed active original kv indices
        active_count:  [num_tiles] int32, number of active kv per tile
    """
    num_tiles = mask.shape[0]
    mask_2d = mask.reshape(num_tiles, kv)
    active_count = (mask_2d != 0).sum(dim=1).to(torch.int32)
    active_kv_ids = torch.zeros_like(mask_2d)
    for t in range(num_tiles):
        ids = torch.nonzero(mask_2d[t], as_tuple=False).squeeze(-1).to(torch.int32)
        active_kv_ids[t, : ids.shape[0]] = ids
    return active_kv_ids.contiguous(), active_count.contiguous()


def _compile_crossk(c_in: int, c_out: int, kv: int, dtype_str: str, block_k: int):
    _ensure_flydsl_path()

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl, vector
    from flydsl.expr.typing import T
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import scf
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
    from kernels.tensor_shim import GTensor, STensor

    if dtype_str != "f32":
        raise ValueError("crossk supports f32 only")
    if block_k > c_in:
        raise ValueError(f"block_k={block_k} > c_in={c_in}: cross-kv boundary not supported yet")
    if c_in % block_k != 0:
        raise ValueError(f"c_in={c_in} not divisible by block_k={block_k}")

    C_IN = c_in
    C_OUT = c_out
    KV = kv
    BLOCK_M = 16
    BLOCK_K = block_k
    C_OUT_TILE = 16
    WAVES_PER_BLOCK = 2
    BLOCK_N = C_OUT_TILE * WAVES_PER_BLOCK
    N_C_OUT_TILES = C_OUT // C_OUT_TILE
    N_C_OUT_TILE_GROUPS = C_OUT // BLOCK_N
    BLOCK_THREADS = 64 * WAVES_PER_BLOCK
    BLOCKS_PER_KV = C_IN // BLOCK_K

    ROW_MAP_ELEMS = BLOCK_M * 2
    ROW_MAP_BYTES = ROW_MAP_ELEMS * 4
    A_LDS_STRIDE = BLOCK_K + 4
    A_STAGE_ELEMS = BLOCK_M * BLOCK_K
    A_STAGE_ELEMS_PADDED = BLOCK_M * A_LDS_STRIDE
    A_STAGE_BYTES = A_STAGE_ELEMS_PADDED * 4
    W_STAGE_ELEMS = BLOCK_K * C_OUT_TILE
    W_STAGE_ELEMS_PER_BLOCK = W_STAGE_ELEMS * WAVES_PER_BLOCK
    W_STAGE_BYTES = W_STAGE_ELEMS_PER_BLOCK * 4

    A_VEC = 4 if (BLOCK_K % 4 == 0) else 1
    A_VEC_GROUPS = BLOCK_M * (BLOCK_K // A_VEC) if A_VEC > 1 else A_STAGE_ELEMS
    A_VEC_LOAD_PER_BLOCK = math.ceil(A_VEC_GROUPS / BLOCK_THREADS)
    W_VEC = 4 if (C_OUT_TILE % 4 == 0 and BLOCK_K % 4 == 0) else 1
    W_VEC_GROUPS = BLOCK_K * (C_OUT_TILE // W_VEC) if W_VEC > 1 else W_STAGE_ELEMS
    W_VEC_LOAD_PER_WAVE = math.ceil(W_VEC_GROUPS / 64)

    allocator = SmemAllocator(
        None, arch="gfx942",
        global_sym_name=f"smem_crossk_bk{BLOCK_K}",
    )
    smem_row_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = smem_row_offset + ROW_MAP_BYTES
    smem_a_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = smem_a_offset + A_STAGE_ELEMS_PADDED * 4
    smem_w_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = smem_w_offset + W_STAGE_ELEMS_PER_BLOCK * 4

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        features: fx.Tensor,
        weights_packed: fx.Tensor,
        output: fx.Tensor,
        inp_row_lut: fx.Tensor,
        active_kv_ids: fx.Tensor,
        active_count: fx.Tensor,
        num_act_out_val: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        bid = fx.Int32(gpu.block_idx.x)

        feat_ = GTensor(features, dtype=T.f32, shape=(-1,))
        wp_ = GTensor(weights_packed, dtype=T.f32, shape=(-1,))
        out_ = GTensor(output, dtype=T.f32, shape=(-1,))
        lut_ = GTensor(inp_row_lut, dtype=T.i32, shape=(-1,))
        akv_ = GTensor(active_kv_ids, dtype=T.i32, shape=(-1,))
        acnt_ = GTensor(active_count, dtype=T.i32, shape=(-1,))

        base_ptr = allocator.get_base()
        smem_row_ptr = SmemPtr(base_ptr, smem_row_offset, T.i32, shape=(ROW_MAP_ELEMS,))
        row_lds = STensor(smem_row_ptr, T.i32, shape=(ROW_MAP_ELEMS,))
        smem_a_ptr = SmemPtr(base_ptr, smem_a_offset, T.f32, shape=(A_STAGE_ELEMS_PADDED,))
        a_lds = STensor(smem_a_ptr, T.f32, shape=(A_STAGE_ELEMS_PADDED,))
        smem_w_ptr = SmemPtr(base_ptr, smem_w_offset, T.f32, shape=(W_STAGE_ELEMS_PER_BLOCK,))
        w_lds = STensor(smem_w_ptr, T.f32, shape=(W_STAGE_ELEMS_PER_BLOCK,))

        wave_id = tid // fx.Int32(const_expr(64))
        lane = tid % fx.Int32(const_expr(64))

        m_tile = bid // fx.Int32(const_expr(N_C_OUT_TILE_GROUPS))
        ct_group = bid % fx.Int32(const_expr(N_C_OUT_TILE_GROUPS))
        ct = ct_group * fx.Int32(const_expr(WAVES_PER_BLOCK)) + wave_id
        row_tile_base = m_tile * fx.Int32(const_expr(BLOCK_M))
        c_out_offset = fx.Index(ct) * fx.Index(const_expr(C_OUT_TILE))

        mfma_row = lane % fx.Int32(const_expr(16))
        mfma_col = lane % fx.Int32(const_expr(16))
        mfma_k_lane = lane // fx.Int32(const_expr(16))
        c_row_vec_base = (lane // fx.Int32(const_expr(16))) * fx.Int32(const_expr(4))

        zero_f32 = arith.constant(0.0, type=T.f32)
        zero_i32 = fx.Int32(arith.constant(0, type=T.i32))
        one_i32 = fx.Int32(arith.constant(1, type=T.i32))
        zero_acc = arith.constant_vector(0.0, T.vec(4, T.f32))

        acc_reg_ty = fx.MemRefType.get(
            T.f32, fx.LayoutType.get(const_expr(4), 1), fx.AddressSpace.Register
        )
        acc_reg_lay = fx.make_layout(const_expr(4), 1)
        acc_reg = fx.memref_alloca(acc_reg_ty, acc_reg_lay)
        fx.memref_store_vec(zero_acc, acc_reg)
        wave_w_offset = fx.Index(wave_id) * fx.Index(const_expr(W_STAGE_ELEMS))

        n_active = acnt_.load(fx.Index(m_tile))
        n_blocks_i32 = n_active * fx.Int32(const_expr(BLOCKS_PER_KV))
        n_blocks_idx = fx.Index(arith.index_cast(T.index, n_blocks_i32))
        zero_idx = fx.Index(arith.constant(0, type=T.index))
        one_idx = fx.Index(arith.constant(1, type=T.index))
        blocks_per_kv_idx = fx.Index(const_expr(BLOCKS_PER_KV))

        for blk_idx in range(zero_idx, n_blocks_idx, one_idx):
            kv_seq = blk_idx // blocks_per_kv_idx
            c_blk_in_kv = blk_idx % blocks_per_kv_idx
            c_offset_idx = c_blk_in_kv * fx.Index(const_expr(BLOCK_K))
            c_offset = fx.Int32(arith.index_cast(T.i32, c_offset_idx))

            c_blk_zero = fx.Index(arith.constant(0, type=T.index))
            kv_changed = arith.cmpi(arith.CmpIPredicate.eq, c_blk_in_kv, c_blk_zero)

            akv_idx = fx.Index(m_tile) * fx.Index(const_expr(KV)) + kv_seq
            orig_kv = akv_.load(akv_idx)

            # --- row map reload on kv boundary (no barrier inside) ---
            rm_if = scf.IfOp(kv_changed, results_=[], has_else=False)
            with ir.InsertionPoint(rm_if.then_block):
                row_tid_valid = arith.cmpi(
                    arith.CmpIPredicate.slt, tid, fx.Int32(const_expr(BLOCK_M))
                )
                row_if = scf.IfOp(row_tid_valid, results_=[], has_else=False)
                with ir.InsertionPoint(row_if.then_block):
                    row_idx = fx.Index(tid)
                    lut_idx = (
                        fx.Index(m_tile) * fx.Index(const_expr(KV * BLOCK_M))
                        + fx.Index(orig_kv) * fx.Index(const_expr(BLOCK_M))
                        + row_idx
                    )
                    inp_row = lut_.load(lut_idx)
                    out_row = row_tile_base + tid
                    row_in_bounds = arith.cmpi(
                        arith.CmpIPredicate.slt, out_row, num_act_out_val
                    )
                    has_pair = arith.cmpi(arith.CmpIPredicate.sge, inp_row, zero_i32)
                    row_valid = arith.andi(row_in_bounds, has_pair)
                    safe_inp_row = arith.select(row_valid, inp_row, zero_i32)
                    row_valid_i32 = arith.select(row_valid, one_i32, zero_i32)
                    safe_row_base = safe_inp_row * fx.Int32(const_expr(C_IN))
                    row_lds.store(row_idx, safe_row_base)
                    row_lds.store(
                        fx.Index(const_expr(BLOCK_M)) + row_idx, row_valid_i32
                    )
                    scf.YieldOp([])
                scf.YieldOp([])

            # Unified barrier: ensures row_map LDS writes visible +
            # previous iteration's MFMA reads complete before A/W overwrites.
            gpu.barrier()

            # --- A staging (vec4 when BLOCK_K divisible by 4) ---
            if const_expr(A_VEC == 4):
                for ai in range_constexpr(A_VEC_LOAD_PER_BLOCK):
                    a_vec_idx = fx.Index(const_expr(ai * BLOCK_THREADS)) + fx.Index(tid)
                    a_vec_valid = arith.cmpi(
                        arith.CmpIPredicate.ult, a_vec_idx,
                        fx.Index(const_expr(A_VEC_GROUPS)),
                    )
                    a_vec_if = scf.IfOp(a_vec_valid, results_=[], has_else=False)
                    with ir.InsertionPoint(a_vec_if.then_block):
                        a_row_idx = a_vec_idx // fx.Index(const_expr(BLOCK_K // A_VEC))
                        a_k_vec_idx = a_vec_idx % fx.Index(const_expr(BLOCK_K // A_VEC))
                        safe_row_base = row_lds.load(a_row_idx)
                        row_valid_i32 = row_lds.load(
                            fx.Index(const_expr(BLOCK_M)) + a_row_idx
                        )
                        row_valid = arith.cmpi(
                            arith.CmpIPredicate.ne, row_valid_i32, zero_i32
                        )
                        feat_off = (
                            fx.Index(safe_row_base)
                            + fx.Index(c_offset)
                            + a_k_vec_idx * fx.Index(const_expr(A_VEC))
                        )
                        a_vec = feat_.vec_load((feat_off,), const_expr(A_VEC))
                        a_lds_base = (
                            a_row_idx * fx.Index(const_expr(A_LDS_STRIDE))
                            + a_k_vec_idx * fx.Index(const_expr(A_VEC))
                        )
                        for vi in range_constexpr(A_VEC):
                            a_val = vector.extract(
                                a_vec,
                                static_position=[const_expr(vi)],
                                dynamic_position=[],
                            )
                            a_val = arith.select(row_valid, a_val, zero_f32)
                            a_lds.store(a_lds_base + fx.Index(const_expr(vi)), a_val)
                        scf.YieldOp([])
            else:
                a_load_per_block = const_expr(math.ceil(A_STAGE_ELEMS / BLOCK_THREADS))
                for ai in range_constexpr(a_load_per_block):
                    a_elem_idx = fx.Index(const_expr(ai * BLOCK_THREADS)) + fx.Index(tid)
                    a_valid = arith.cmpi(
                        arith.CmpIPredicate.ult, a_elem_idx,
                        fx.Index(const_expr(A_STAGE_ELEMS)),
                    )
                    a_if = scf.IfOp(a_valid, results_=[], has_else=False)
                    with ir.InsertionPoint(a_if.then_block):
                        a_row_idx = a_elem_idx // fx.Index(const_expr(BLOCK_K))
                        a_k_idx = a_elem_idx % fx.Index(const_expr(BLOCK_K))
                        safe_row_base = row_lds.load(a_row_idx)
                        row_valid_i32 = row_lds.load(
                            fx.Index(const_expr(BLOCK_M)) + a_row_idx
                        )
                        row_valid = arith.cmpi(
                            arith.CmpIPredicate.ne, row_valid_i32, zero_i32
                        )
                        feat_off = fx.Index(safe_row_base) + fx.Index(c_offset) + a_k_idx
                        a_val = feat_.load(feat_off)
                        a_val = arith.select(row_valid, a_val, zero_f32)
                        a_lds_base = a_row_idx * fx.Index(const_expr(A_LDS_STRIDE)) + a_k_idx
                        a_lds.store(a_lds_base, a_val)
                        scf.YieldOp([])

            # --- W staging (vec4) ---
            w_base = (
                fx.Index(orig_kv) * fx.Index(const_expr(N_C_OUT_TILES * C_IN * C_OUT_TILE))
                + fx.Index(ct) * fx.Index(const_expr(C_IN * C_OUT_TILE))
            )
            if const_expr(W_VEC == 4):
                for wi in range_constexpr(W_VEC_LOAD_PER_WAVE):
                    w_vec_idx = fx.Index(const_expr(wi * 64)) + fx.Index(lane)
                    w_vec_valid = arith.cmpi(
                        arith.CmpIPredicate.ult, w_vec_idx,
                        fx.Index(const_expr(W_VEC_GROUPS)),
                    )
                    w_vec_if = scf.IfOp(w_vec_valid, results_=[], has_else=False)
                    with ir.InsertionPoint(w_vec_if.then_block):
                        w_k_idx = w_vec_idx // fx.Index(const_expr(C_OUT_TILE // W_VEC))
                        w_col_vec = w_vec_idx % fx.Index(const_expr(C_OUT_TILE // W_VEC))
                        w_src = (
                            w_base
                            + (fx.Index(c_offset) + w_k_idx)
                            * fx.Index(const_expr(C_OUT_TILE))
                            + w_col_vec * fx.Index(const_expr(W_VEC))
                        )
                        w_vec = wp_.vec_load((w_src,), const_expr(W_VEC))
                        w_lds_base = (
                            wave_w_offset
                            + w_k_idx * fx.Index(const_expr(C_OUT_TILE))
                            + w_col_vec * fx.Index(const_expr(W_VEC))
                        )
                        for vi in range_constexpr(W_VEC):
                            w_val = vector.extract(
                                w_vec,
                                static_position=[const_expr(vi)],
                                dynamic_position=[],
                            )
                            w_lds.store(w_lds_base + fx.Index(const_expr(vi)), w_val)
                        scf.YieldOp([])
            else:
                w_load_per_wave = const_expr(math.ceil(W_STAGE_ELEMS / 64))
                for wi in range_constexpr(w_load_per_wave):
                    w_elem_idx = fx.Index(const_expr(wi * 64)) + fx.Index(lane)
                    w_valid = arith.cmpi(
                        arith.CmpIPredicate.ult, w_elem_idx,
                        fx.Index(const_expr(W_STAGE_ELEMS)),
                    )
                    w_if = scf.IfOp(w_valid, results_=[], has_else=False)
                    with ir.InsertionPoint(w_if.then_block):
                        w_k_idx = w_elem_idx // fx.Index(const_expr(C_OUT_TILE))
                        w_col_idx = w_elem_idx % fx.Index(const_expr(C_OUT_TILE))
                        w_src = (
                            w_base
                            + (fx.Index(c_offset) + w_k_idx)
                            * fx.Index(const_expr(C_OUT_TILE))
                            + w_col_idx
                        )
                        w_val = wp_.load(w_src)
                        w_lds_base = (
                            wave_w_offset
                            + w_k_idx * fx.Index(const_expr(C_OUT_TILE))
                            + w_col_idx
                        )
                        w_lds.store(w_lds_base, w_val)
                        scf.YieldOp([])

            gpu.barrier()

            # --- MFMA ---
            for kk in range_constexpr(0, BLOCK_K, 4):
                a_off = (
                    fx.Index(mfma_row) * fx.Index(const_expr(A_LDS_STRIDE))
                    + fx.Index(const_expr(kk))
                    + fx.Index(mfma_k_lane)
                )
                a_val = a_lds.load(a_off)
                b_off = (
                    wave_w_offset
                    + fx.Index(const_expr(kk * C_OUT_TILE))
                    + fx.Index(mfma_k_lane) * fx.Index(const_expr(C_OUT_TILE))
                    + fx.Index(mfma_col)
                )
                b_val = w_lds.load(b_off)
                cur_acc = fx.memref_load_vec(acc_reg)
                new_acc = rocdl.mfma_f32_16x16x4f32(
                    T.vec(4, T.f32), a_val, b_val, cur_acc, 0, 0, 0
                )
                fx.memref_store_vec(new_acc, acc_reg)

        # --- Epilogue: direct store ---
        final_acc = fx.memref_load_vec(acc_reg)
        for ri in range_constexpr(4):
            store_row = row_tile_base + c_row_vec_base + fx.Int32(const_expr(ri))
            store_ok = arith.cmpi(arith.CmpIPredicate.slt, store_row, num_act_out_val)
            store_if = scf.IfOp(store_ok, results_=[], has_else=False)
            with ir.InsertionPoint(store_if.then_block):
                val = vector.extract(
                    final_acc, static_position=[const_expr(ri)], dynamic_position=[]
                )
                out_off = (
                    fx.Index(store_row) * fx.Index(const_expr(C_OUT))
                    + c_out_offset
                    + fx.Index(mfma_col)
                )
                out_.store(out_off, val)
                scf.YieldOp([])

    @flyc.jit
    def launch_fn(
        features: fx.Tensor,
        weights_packed: fx.Tensor,
        output: fx.Tensor,
        inp_row_lut: fx.Tensor,
        active_kv_ids: fx.Tensor,
        active_count: fx.Tensor,
        num_tiles: fx.Int32,
        num_act_out: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        kernel(
            features,
            weights_packed,
            output,
            inp_row_lut,
            active_kv_ids,
            active_count,
            num_act_out,
        ).launch(
            grid=(num_tiles * N_C_OUT_TILE_GROUPS,),
            block=(BLOCK_THREADS,),
            stream=stream,
        )

    return launch_fn, BLOCK_M


def _compile_crossk_prefetch(c_in: int, c_out: int, kv: int, dtype_str: str, block_k: int,
                             use_xor_swizzle: bool = False):
    """Crossk with VMEM-MFMA overlap: issue loads before MFMA, ds_write after."""
    _ensure_flydsl_path()

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl, vector
    from flydsl.expr.typing import T
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import scf
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
    from kernels.tensor_shim import GTensor, STensor

    if dtype_str != "f32":
        raise ValueError("crossk prefetch supports f32 only")
    if block_k > c_in or c_in % block_k != 0:
        raise ValueError(f"block_k={block_k} invalid for c_in={c_in}")

    C_IN = c_in
    C_OUT = c_out
    KV = kv
    BLOCK_M = 16
    BLOCK_K = block_k
    C_OUT_TILE = 16
    WAVES_PER_BLOCK = 2
    N_C_OUT_TILES = C_OUT // C_OUT_TILE
    N_C_OUT_TILE_GROUPS = C_OUT // (C_OUT_TILE * WAVES_PER_BLOCK)
    BLOCK_THREADS = 64 * WAVES_PER_BLOCK
    BLOCKS_PER_KV = C_IN // BLOCK_K

    USE_XOR = use_xor_swizzle
    K_SLOTS_16B = BLOCK_K * 4 // 16  # number of 16-byte slots per row

    ROW_MAP_ELEMS = BLOCK_M * 2
    ROW_MAP_BYTES = ROW_MAP_ELEMS * 4
    A_LDS_STRIDE = BLOCK_K if USE_XOR else (BLOCK_K + 4)
    A_STAGE_ELEMS_PADDED = BLOCK_M * A_LDS_STRIDE
    W_STAGE_ELEMS = BLOCK_K * C_OUT_TILE
    W_STAGE_ELEMS_PER_BLOCK = W_STAGE_ELEMS * WAVES_PER_BLOCK

    A_VEC = 4
    A_VEC_GROUPS = BLOCK_M * (BLOCK_K // A_VEC)
    A_VEC_LOAD_PER_BLOCK = math.ceil(A_VEC_GROUPS / BLOCK_THREADS)
    W_VEC = 4
    W_VEC_GROUPS = BLOCK_K * (C_OUT_TILE // W_VEC)
    W_VEC_LOAD_PER_WAVE = math.ceil(W_VEC_GROUPS / 64)

    allocator = SmemAllocator(
        None, arch="gfx942",
        global_sym_name=f"smem_crossk_pf{'_xor' if USE_XOR else ''}_bk{BLOCK_K}",
    )
    smem_row_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = smem_row_offset + ROW_MAP_BYTES
    smem_a_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = smem_a_offset + A_STAGE_ELEMS_PADDED * 4
    smem_w_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = smem_w_offset + W_STAGE_ELEMS_PER_BLOCK * 4

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel_pf(
        features: fx.Tensor,
        weights_packed: fx.Tensor,
        output: fx.Tensor,
        inp_row_lut: fx.Tensor,
        active_kv_ids: fx.Tensor,
        active_count: fx.Tensor,
        num_act_out_val: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        bid = fx.Int32(gpu.block_idx.x)

        feat_ = GTensor(features, dtype=T.f32, shape=(-1,))
        wp_ = GTensor(weights_packed, dtype=T.f32, shape=(-1,))
        out_ = GTensor(output, dtype=T.f32, shape=(-1,))
        lut_ = GTensor(inp_row_lut, dtype=T.i32, shape=(-1,))
        akv_ = GTensor(active_kv_ids, dtype=T.i32, shape=(-1,))
        acnt_ = GTensor(active_count, dtype=T.i32, shape=(-1,))

        base_ptr = allocator.get_base()
        smem_row_ptr = SmemPtr(base_ptr, smem_row_offset, T.i32, shape=(ROW_MAP_ELEMS,))
        row_lds = STensor(smem_row_ptr, T.i32, shape=(ROW_MAP_ELEMS,))
        smem_a_ptr = SmemPtr(base_ptr, smem_a_offset, T.f32, shape=(A_STAGE_ELEMS_PADDED,))
        a_lds = STensor(smem_a_ptr, T.f32, shape=(A_STAGE_ELEMS_PADDED,))
        smem_w_ptr = SmemPtr(base_ptr, smem_w_offset, T.f32, shape=(W_STAGE_ELEMS_PER_BLOCK,))
        w_lds = STensor(smem_w_ptr, T.f32, shape=(W_STAGE_ELEMS_PER_BLOCK,))

        wave_id = tid // fx.Int32(const_expr(64))
        lane = tid % fx.Int32(const_expr(64))

        m_tile = bid // fx.Int32(const_expr(N_C_OUT_TILE_GROUPS))
        ct_group = bid % fx.Int32(const_expr(N_C_OUT_TILE_GROUPS))
        ct = ct_group * fx.Int32(const_expr(WAVES_PER_BLOCK)) + wave_id
        row_tile_base = m_tile * fx.Int32(const_expr(BLOCK_M))
        c_out_offset = fx.Index(ct) * fx.Index(const_expr(C_OUT_TILE))

        mfma_row = lane % fx.Int32(const_expr(16))
        mfma_col = lane % fx.Int32(const_expr(16))
        mfma_k_lane = lane // fx.Int32(const_expr(16))
        c_row_vec_base = (lane // fx.Int32(const_expr(16))) * fx.Int32(const_expr(4))

        zero_f32 = arith.constant(0.0, type=T.f32)
        zero_vec4 = arith.constant_vector(0.0, T.vec(4, T.f32))
        zero_i32 = fx.Int32(arith.constant(0, type=T.i32))
        one_i32 = fx.Int32(arith.constant(1, type=T.i32))

        acc_reg_ty = fx.MemRefType.get(
            T.f32, fx.LayoutType.get(const_expr(4), 1), fx.AddressSpace.Register
        )
        acc_reg_lay = fx.make_layout(const_expr(4), 1)
        acc_reg = fx.memref_alloca(acc_reg_ty, acc_reg_lay)
        fx.memref_store_vec(zero_vec4, acc_reg)
        wave_w_offset = fx.Index(wave_id) * fx.Index(const_expr(W_STAGE_ELEMS))

        n_active = acnt_.load(fx.Index(m_tile))
        n_blocks_i32 = n_active * fx.Int32(const_expr(BLOCKS_PER_KV))
        n_blocks_idx = fx.Index(arith.index_cast(T.index, n_blocks_i32))
        one_idx = fx.Index(arith.constant(1, type=T.index))
        blocks_per_kv_idx = fx.Index(const_expr(BLOCKS_PER_KV))

        has_work = arith.cmpi(arith.CmpIPredicate.sgt, n_blocks_i32, zero_i32)
        work_if = scf.IfOp(has_work, results_=[], has_else=False)
        with ir.InsertionPoint(work_if.then_block):

            # === Helper: XOR swizzle for A LDS ===
            # Returns the swizzled base column offset for a vec4 group.
            # row_idx: row index; vec_group_idx: which vec4 group (== akv).
            def _a_swizzle_vec_base(row_idx, vec_group_idx):
                if not USE_XOR:
                    return vec_group_idx * fx.Index(const_expr(A_VEC))
                row_mask = fx.Index(arith.andi(row_idx, fx.Index(const_expr(K_SLOTS_16B - 1))))
                swz_group = fx.Index(arith.xori(vec_group_idx, row_mask))
                return swz_group * fx.Index(const_expr(A_VEC))

            # Scalar version for MFMA reads (kk + k_lane is not necessarily vec4-aligned).
            def _a_swizzle_scalar(row_idx, col_dword):
                if not USE_XOR:
                    return col_dword
                group = col_dword // fx.Index(const_expr(4))
                row_mask = fx.Index(arith.andi(row_idx, fx.Index(const_expr(K_SLOTS_16B - 1))))
                swz_group = fx.Index(arith.xori(group, row_mask))
                within = col_dword % fx.Index(const_expr(4))
                return swz_group * fx.Index(const_expr(4)) + within

            # === Helper: compute blk_idx → kv_seq, c_offset, orig_kv ===
            def _blk_params(blk_idx_idx):
                kv_seq = blk_idx_idx // blocks_per_kv_idx
                c_blk = blk_idx_idx % blocks_per_kv_idx
                c_off_idx = c_blk * fx.Index(const_expr(BLOCK_K))
                c_off = fx.Int32(arith.index_cast(T.i32, c_off_idx))
                akv_i = fx.Index(m_tile) * fx.Index(const_expr(KV)) + kv_seq
                okv = akv_.load(akv_i)
                kv_ch = arith.cmpi(
                    arith.CmpIPredicate.eq, c_blk,
                    fx.Index(arith.constant(0, type=T.index)),
                )
                return kv_seq, c_off, okv, kv_ch

            # === Helper: reload row_map ===
            def _reload_row_map(okv):
                row_tid_valid = arith.cmpi(
                    arith.CmpIPredicate.slt, tid, fx.Int32(const_expr(BLOCK_M))
                )
                rif = scf.IfOp(row_tid_valid, results_=[], has_else=False)
                with ir.InsertionPoint(rif.then_block):
                    ri = fx.Index(tid)
                    li = (
                        fx.Index(m_tile) * fx.Index(const_expr(KV * BLOCK_M))
                        + fx.Index(okv) * fx.Index(const_expr(BLOCK_M))
                        + ri
                    )
                    inp_r = lut_.load(li)
                    out_r = row_tile_base + tid
                    rib = arith.cmpi(arith.CmpIPredicate.slt, out_r, num_act_out_val)
                    hp = arith.cmpi(arith.CmpIPredicate.sge, inp_r, zero_i32)
                    rv = arith.andi(rib, hp)
                    sir = arith.select(rv, inp_r, zero_i32)
                    rvi = arith.select(rv, one_i32, zero_i32)
                    srb = sir * fx.Int32(const_expr(C_IN))
                    row_lds.store(ri, srb)
                    row_lds.store(fx.Index(const_expr(BLOCK_M)) + ri, rvi)
                    scf.YieldOp([])

            # === Helper: standard A+W load to LDS (blocking) ===
            def _load_aw_to_lds(c_off, okv):
                for ai in range_constexpr(A_VEC_LOAD_PER_BLOCK):
                    a_vi = fx.Index(const_expr(ai * BLOCK_THREADS)) + fx.Index(tid)
                    a_vv = arith.cmpi(
                        arith.CmpIPredicate.ult, a_vi,
                        fx.Index(const_expr(A_VEC_GROUPS)),
                    )
                    aif = scf.IfOp(a_vv, results_=[], has_else=False)
                    with ir.InsertionPoint(aif.then_block):
                        ari = a_vi // fx.Index(const_expr(BLOCK_K // A_VEC))
                        akv = a_vi % fx.Index(const_expr(BLOCK_K // A_VEC))
                        srb = row_lds.load(ari)
                        rvi = row_lds.load(fx.Index(const_expr(BLOCK_M)) + ari)
                        rv = arith.cmpi(arith.CmpIPredicate.ne, rvi, zero_i32)
                        fo = (
                            fx.Index(srb)
                            + fx.Index(c_off)
                            + akv * fx.Index(const_expr(A_VEC))
                        )
                        av = feat_.vec_load((fo,), const_expr(A_VEC))
                        swz_base = _a_swizzle_vec_base(ari, akv)
                        ab = ari * fx.Index(const_expr(A_LDS_STRIDE)) + swz_base
                        for vi in range_constexpr(A_VEC):
                            val = vector.extract(av, static_position=[const_expr(vi)], dynamic_position=[])
                            val = arith.select(rv, val, zero_f32)
                            a_lds.store(ab + fx.Index(const_expr(vi)), val)
                        scf.YieldOp([])

                wb = (
                    fx.Index(okv) * fx.Index(const_expr(N_C_OUT_TILES * C_IN * C_OUT_TILE))
                    + fx.Index(ct) * fx.Index(const_expr(C_IN * C_OUT_TILE))
                )
                for wi in range_constexpr(W_VEC_LOAD_PER_WAVE):
                    wvi = fx.Index(const_expr(wi * 64)) + fx.Index(lane)
                    wvv = arith.cmpi(
                        arith.CmpIPredicate.ult, wvi,
                        fx.Index(const_expr(W_VEC_GROUPS)),
                    )
                    wif = scf.IfOp(wvv, results_=[], has_else=False)
                    with ir.InsertionPoint(wif.then_block):
                        wki = wvi // fx.Index(const_expr(C_OUT_TILE // W_VEC))
                        wcv = wvi % fx.Index(const_expr(C_OUT_TILE // W_VEC))
                        ws = (
                            wb
                            + (fx.Index(c_off) + wki) * fx.Index(const_expr(C_OUT_TILE))
                            + wcv * fx.Index(const_expr(W_VEC))
                        )
                        wv = wp_.vec_load((ws,), const_expr(W_VEC))
                        wlb = (
                            wave_w_offset
                            + wki * fx.Index(const_expr(C_OUT_TILE))
                            + wcv * fx.Index(const_expr(W_VEC))
                        )
                        for vi in range_constexpr(W_VEC):
                            val = vector.extract(wv, static_position=[const_expr(vi)], dynamic_position=[])
                            w_lds.store(wlb + fx.Index(const_expr(vi)), val)
                        scf.YieldOp([])

            # === Helper: MFMA on current LDS data ===
            def _mfma():
                for kk in range_constexpr(0, BLOCK_K, 4):
                    a_col = fx.Index(const_expr(kk)) + fx.Index(mfma_k_lane)
                    swz_a_col = _a_swizzle_scalar(fx.Index(mfma_row), a_col)
                    ao = (
                        fx.Index(mfma_row) * fx.Index(const_expr(A_LDS_STRIDE))
                        + swz_a_col
                    )
                    av = a_lds.load(ao)
                    bo = (
                        wave_w_offset
                        + fx.Index(const_expr(kk * C_OUT_TILE))
                        + fx.Index(mfma_k_lane) * fx.Index(const_expr(C_OUT_TILE))
                        + fx.Index(mfma_col)
                    )
                    bv = w_lds.load(bo)
                    ca = fx.memref_load_vec(acc_reg)
                    na = rocdl.mfma_f32_16x16x4f32(T.vec(4, T.f32), av, bv, ca, 0, 0, 0)
                    fx.memref_store_vec(na, acc_reg)

            # === Helper: issue VMEM loads unconditionally, return SSA vecs ===
            def _issue_vmem_loads(c_off, okv):
                a_vecs = []
                a_meta = []
                for ai in range_constexpr(A_VEC_LOAD_PER_BLOCK):
                    a_vi = fx.Index(const_expr(ai * BLOCK_THREADS)) + fx.Index(tid)
                    ari = a_vi // fx.Index(const_expr(BLOCK_K // A_VEC))
                    akv = a_vi % fx.Index(const_expr(BLOCK_K // A_VEC))
                    srb = row_lds.load(ari)
                    rvi = row_lds.load(fx.Index(const_expr(BLOCK_M)) + ari)
                    rv = arith.cmpi(arith.CmpIPredicate.ne, rvi, zero_i32)
                    a_valid = arith.cmpi(
                        arith.CmpIPredicate.ult, a_vi,
                        fx.Index(const_expr(A_VEC_GROUPS)),
                    )
                    combined_valid = arith.andi(rv, a_valid)
                    fo = (
                        fx.Index(srb)
                        + fx.Index(c_off)
                        + akv * fx.Index(const_expr(A_VEC))
                    )
                    safe_fo = arith.select(combined_valid, fo, fx.Index(arith.constant(0, type=T.index)))
                    av = feat_.vec_load((safe_fo,), const_expr(A_VEC))
                    swz_base = _a_swizzle_vec_base(ari, akv)
                    ab = ari * fx.Index(const_expr(A_LDS_STRIDE)) + swz_base
                    a_vecs.append(av)
                    a_meta.append((ab, combined_valid))

                w_vecs = []
                w_meta = []
                wb = (
                    fx.Index(okv) * fx.Index(const_expr(N_C_OUT_TILES * C_IN * C_OUT_TILE))
                    + fx.Index(ct) * fx.Index(const_expr(C_IN * C_OUT_TILE))
                )
                for wi in range_constexpr(W_VEC_LOAD_PER_WAVE):
                    wvi = fx.Index(const_expr(wi * 64)) + fx.Index(lane)
                    wki = wvi // fx.Index(const_expr(C_OUT_TILE // W_VEC))
                    wcv = wvi % fx.Index(const_expr(C_OUT_TILE // W_VEC))
                    w_valid = arith.cmpi(
                        arith.CmpIPredicate.ult, wvi,
                        fx.Index(const_expr(W_VEC_GROUPS)),
                    )
                    ws = (
                        wb
                        + (fx.Index(c_off) + wki) * fx.Index(const_expr(C_OUT_TILE))
                        + wcv * fx.Index(const_expr(W_VEC))
                    )
                    safe_ws = arith.select(w_valid, ws, fx.Index(arith.constant(0, type=T.index)))
                    wv = wp_.vec_load((safe_ws,), const_expr(W_VEC))
                    wlb = (
                        wave_w_offset
                        + wki * fx.Index(const_expr(C_OUT_TILE))
                        + wcv * fx.Index(const_expr(W_VEC))
                    )
                    w_vecs.append(wv)
                    w_meta.append((wlb, w_valid))

                return a_vecs, a_meta, w_vecs, w_meta

            # === Helper: drain prefetched vecs to LDS ===
            def _drain_to_lds(a_vecs, a_meta, w_vecs, w_meta):
                for av, (ab, cv) in zip(a_vecs, a_meta):
                    for vi in range_constexpr(A_VEC):
                        val = vector.extract(av, static_position=[const_expr(vi)], dynamic_position=[])
                        val = arith.select(cv, val, zero_f32)
                        a_lds.store(ab + fx.Index(const_expr(vi)), val)
                for wv, (wlb, wvl) in zip(w_vecs, w_meta):
                    for vi in range_constexpr(W_VEC):
                        val = vector.extract(wv, static_position=[const_expr(vi)], dynamic_position=[])
                        val = arith.select(wvl, val, zero_f32)
                        w_lds.store(wlb + fx.Index(const_expr(vi)), val)

            # ============ PROLOGUE: load iter 0 to LDS ============
            zero_blk = fx.Index(arith.constant(0, type=T.index))
            _, c_off0, okv0, _ = _blk_params(zero_blk)
            _reload_row_map(okv0)
            gpu.barrier()
            _load_aw_to_lds(c_off0, okv0)
            gpu.barrier()

            # ============ MAIN LOOP: iter 1..N-1 ============
            for blk_idx in range(one_idx, n_blocks_idx, one_idx):
                _, c_off, okv, kv_ch = _blk_params(blk_idx)

                rm_if = scf.IfOp(kv_ch, results_=[], has_else=False)
                with ir.InsertionPoint(rm_if.then_block):
                    _reload_row_map(okv)
                    scf.YieldOp([])
                gpu.barrier()

                a_vecs, a_meta, w_vecs, w_meta = _issue_vmem_loads(c_off, okv)

                _mfma()

                gpu.barrier()
                _drain_to_lds(a_vecs, a_meta, w_vecs, w_meta)
                gpu.barrier()

            # ============ EPILOGUE: MFMA on last data ============
            _mfma()

            scf.YieldOp([])

        # --- Epilogue: direct store ---
        final_acc = fx.memref_load_vec(acc_reg)
        for ri in range_constexpr(4):
            store_row = row_tile_base + c_row_vec_base + fx.Int32(const_expr(ri))
            store_ok = arith.cmpi(arith.CmpIPredicate.slt, store_row, num_act_out_val)
            store_if = scf.IfOp(store_ok, results_=[], has_else=False)
            with ir.InsertionPoint(store_if.then_block):
                val = vector.extract(
                    final_acc, static_position=[const_expr(ri)], dynamic_position=[]
                )
                out_off = (
                    fx.Index(store_row) * fx.Index(const_expr(C_OUT))
                    + c_out_offset
                    + fx.Index(mfma_col)
                )
                out_.store(out_off, val)
                scf.YieldOp([])

    @flyc.jit
    def launch_fn(
        features: fx.Tensor,
        weights_packed: fx.Tensor,
        output: fx.Tensor,
        inp_row_lut: fx.Tensor,
        active_kv_ids: fx.Tensor,
        active_count: fx.Tensor,
        num_tiles: fx.Int32,
        num_act_out: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        kernel_pf(
            features,
            weights_packed,
            output,
            inp_row_lut,
            active_kv_ids,
            active_count,
            num_act_out,
        ).launch(
            grid=(num_tiles * N_C_OUT_TILE_GROUPS,),
            block=(BLOCK_THREADS,),
            stream=stream,
        )

    return launch_fn, BLOCK_M


def implicit_gemm_crossk_prefetch_forward(
    features: torch.Tensor,
    filters: torch.Tensor,
    indice_pairs: torch.Tensor,
    indice_pair_num: torch.Tensor,
    num_activate_out: int,
    block_k: int = 32,
    use_xor_swizzle: bool = False,
) -> Optional[torch.Tensor]:
    """Run the cross-kv K-fused implicit GEMM kernel with prefetch overlap."""
    try:
        import flydsl  # noqa: F401
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

    dtype_str = _dtype_to_str(dtype)
    if dtype_str != "f32":
        return None
    if c_in % 4 != 0 or c_out % 32 != 0:
        return None
    if block_k > c_in or c_in % block_k != 0:
        return None

    block_m = 16
    num_tiles = (num_activate_out + block_m - 1) // block_m
    sorted_inp, sorted_out, sorted_kv, mask, pair_start, pair_end, inp_row_lut = (
        hip.build_implicit_gemm_mask(indice_pairs, indice_pair_num, num_activate_out, block_m)
    )
    if sorted_inp.shape[0] == 0:
        return torch.zeros(num_activate_out, c_out, dtype=torch.float32, device=device)

    mask_2d = mask.reshape(num_tiles, kv)
    active_kv_ids, active_count = _build_active_kv_ids(mask_2d, kv)

    swz_tag = "_xor" if use_xor_swizzle else ""
    key = ("crossk_pf" + swz_tag, c_in, c_out, kv, dtype_str, block_k)
    if key not in CROSSK_COMPILED_KERNELS:
        try:
            launch_fn, block_m = _compile_crossk_prefetch(
                c_in, c_out, kv, dtype_str, block_k, use_xor_swizzle=use_xor_swizzle
            )
            CROSSK_COMPILED_KERNELS[key] = (launch_fn, block_m)
        except Exception as e:
            import traceback
            warnings.warn(
                f"Failed to compile crossk prefetch{swz_tag} kernel: {e}\n{traceback.format_exc()}"
            )
            return None
    else:
        launch_fn, block_m = CROSSK_COMPILED_KERNELS[key]

    weights_packed = _pack_weights(filters, 16)
    out_features = torch.zeros(num_activate_out, c_out, dtype=torch.float32, device=device)
    stream = torch.cuda.current_stream()

    launch_fn(
        features.contiguous(),
        weights_packed,
        out_features,
        inp_row_lut.reshape(-1).contiguous(),
        active_kv_ids.reshape(-1).contiguous(),
        active_count.reshape(-1).contiguous(),
        num_tiles,
        num_activate_out,
        stream,
    )
    return out_features


def implicit_gemm_crossk_forward(
    features: torch.Tensor,
    filters: torch.Tensor,
    indice_pairs: torch.Tensor,
    indice_pair_num: torch.Tensor,
    num_activate_out: int,
    block_k: int = 32,
) -> Optional[torch.Tensor]:
    """Run the cross-kv K-fused implicit GEMM kernel."""
    try:
        import flydsl  # noqa: F401
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

    dtype_str = _dtype_to_str(dtype)
    if dtype_str != "f32":
        return None
    if c_in % 4 != 0 or c_out % 32 != 0:
        return None
    if block_k > c_in or c_in % block_k != 0:
        return None

    block_m = 16
    num_tiles = (num_activate_out + block_m - 1) // block_m
    sorted_inp, sorted_out, sorted_kv, mask, pair_start, pair_end, inp_row_lut = (
        hip.build_implicit_gemm_mask(indice_pairs, indice_pair_num, num_activate_out, block_m)
    )
    if sorted_inp.shape[0] == 0:
        return torch.zeros(num_activate_out, c_out, dtype=torch.float32, device=device)

    mask_2d = mask.reshape(num_tiles, kv)
    active_kv_ids, active_count = _build_active_kv_ids(mask_2d, kv)

    key = ("crossk", c_in, c_out, kv, dtype_str, block_k)
    if key not in CROSSK_COMPILED_KERNELS:
        try:
            launch_fn, block_m = _compile_crossk(c_in, c_out, kv, dtype_str, block_k)
            CROSSK_COMPILED_KERNELS[key] = (launch_fn, block_m)
        except Exception as e:
            import traceback
            warnings.warn(
                f"Failed to compile crossk kernel: {e}\n{traceback.format_exc()}"
            )
            return None
    else:
        launch_fn, block_m = CROSSK_COMPILED_KERNELS[key]

    weights_packed = _pack_weights(filters, 16)
    out_features = torch.zeros(num_activate_out, c_out, dtype=torch.float32, device=device)
    stream = torch.cuda.current_stream()

    launch_fn(
        features.contiguous(),
        weights_packed,
        out_features,
        inp_row_lut.reshape(-1).contiguous(),
        active_kv_ids.reshape(-1).contiguous(),
        active_count.reshape(-1).contiguous(),
        num_tiles,
        num_activate_out,
        stream,
    )
    return out_features
