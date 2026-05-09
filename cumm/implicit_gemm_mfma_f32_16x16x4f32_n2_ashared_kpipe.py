"""mfma_f32_16x16x4f32_n2_ashared_kpipe implicit GEMM family member.

This keeps the current per-kv BLOCK_K pipeline, but factors sparse A row lookup
out of the per-element A staging loop:
  - once per active kv, stage 16 safe input row bases + valid flags into LDS
  - each A[16 x BLOCK_K] stage reuses the row map

It is a minimal step toward a reusable sparse A/K stager, not a full iterator
or a separate dispatch candidate.
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


MFMA_F32_16X16X4F32_N2_ASHARED_KPIPE_COMPILED_KERNELS: Dict = {}


def _compile_implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe(
    c_in: int, c_out: int, kv: int, dtype_str: str, block_k: int, epilogue: str = "direct"
):
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
        raise ValueError("mfma_f32_16x16x4f32_n2_ashared_kpipe supports f32 only")
    if block_k not in (16, 32):
        raise ValueError(f"unsupported block_k={block_k}, expected 16 or 32")
    if epilogue not in ("direct", "remap"):
        raise ValueError(f"unsupported epilogue={epilogue}, expected direct or remap")

    C_IN = c_in
    C_OUT = c_out
    KV = kv
    EPILOGUE = epilogue
    EPILOGUE_REMAP = EPILOGUE == "remap"
    BLOCK_M = 16
    BLOCK_K = block_k
    C_OUT_TILE = 16
    WAVES_PER_BLOCK = 2
    BLOCK_N = C_OUT_TILE * WAVES_PER_BLOCK
    N_C_OUT_TILES = C_OUT // C_OUT_TILE
    N_C_OUT_TILE_GROUPS = C_OUT // BLOCK_N
    BLOCK_THREADS = 64 * WAVES_PER_BLOCK

    ROW_MAP_ELEMS = BLOCK_M * 2  # safe_row_base[16] + row_valid_i32[16]
    ROW_MAP_BYTES = ROW_MAP_ELEMS * 4
    A_LDS_STRIDE = BLOCK_K + 1
    W_LDS_STRIDE = C_OUT_TILE + 1
    A_STAGE_ELEMS = BLOCK_M * BLOCK_K
    A_STAGE_ELEMS_PADDED = BLOCK_M * A_LDS_STRIDE
    A_STAGE_BYTES = A_STAGE_ELEMS_PADDED * 4
    W_STAGE_ELEMS = BLOCK_K * C_OUT_TILE
    W_STAGE_ELEMS_PADDED = BLOCK_K * W_LDS_STRIDE
    W_STAGE_ELEMS_PER_BLOCK = W_STAGE_ELEMS_PADDED * WAVES_PER_BLOCK
    W_STAGE_BYTES = W_STAGE_ELEMS_PER_BLOCK * 4
    EPI_STAGE_ELEMS = BLOCK_M * C_OUT_TILE
    EPI_STAGE_ELEMS_PER_BLOCK = EPI_STAGE_ELEMS * WAVES_PER_BLOCK
    EPI_STAGE_BYTES = EPI_STAGE_ELEMS_PER_BLOCK * 4

    allocator = SmemAllocator(
        None,
        arch="gfx942",
        global_sym_name=(
            f"smem_ig_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk{BLOCK_K}_{EPILOGUE}"
        ),
    )
    smem_row_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = smem_row_offset + ROW_MAP_BYTES
    smem_a_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = smem_a_offset + A_STAGE_BYTES
    smem_w_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = smem_w_offset + W_STAGE_BYTES
    if EPILOGUE_REMAP:
        smem_epi_offset = allocator._align(allocator.ptr, 16)
        allocator.ptr = smem_epi_offset + EPI_STAGE_BYTES

    A_LOAD_PER_BLOCK = math.ceil(A_STAGE_ELEMS / BLOCK_THREADS)
    A_VEC = 4
    A_VEC_GROUPS = BLOCK_M * (BLOCK_K // A_VEC)
    A_VEC_LOAD_PER_BLOCK = math.ceil(A_VEC_GROUPS / BLOCK_THREADS)
    W_LOAD_PER_WAVE = math.ceil(W_STAGE_ELEMS / 64)
    W_VEC = 4
    W_VEC_GROUPS = BLOCK_K * (C_OUT_TILE // W_VEC)
    W_VEC_LOAD_PER_WAVE = math.ceil(W_VEC_GROUPS / 64)

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        features: fx.Tensor,
        weights_packed: fx.Tensor,
        output: fx.Tensor,
        inp_row_lut: fx.Tensor,
        mask: fx.Tensor,
        num_act_out_val: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        bid = fx.Int32(gpu.block_idx.x)

        feat_ = GTensor(features, dtype=T.f32, shape=(-1,))
        wp_ = GTensor(weights_packed, dtype=T.f32, shape=(-1,))
        out_ = GTensor(output, dtype=T.f32, shape=(-1,))
        lut_ = GTensor(inp_row_lut, dtype=T.i32, shape=(-1,))
        mask_ = GTensor(mask, dtype=T.i32, shape=(-1,))

        base_ptr = allocator.get_base()
        smem_row_ptr = SmemPtr(base_ptr, smem_row_offset, T.i32, shape=(ROW_MAP_ELEMS,))
        row_lds = STensor(smem_row_ptr, T.i32, shape=(ROW_MAP_ELEMS,))
        smem_a_ptr = SmemPtr(base_ptr, smem_a_offset, T.f32, shape=(A_STAGE_ELEMS_PADDED,))
        a_lds = STensor(smem_a_ptr, T.f32, shape=(A_STAGE_ELEMS_PADDED,))
        smem_w_ptr = SmemPtr(base_ptr, smem_w_offset, T.f32, shape=(W_STAGE_ELEMS_PER_BLOCK,))
        w_lds = STensor(smem_w_ptr, T.f32, shape=(W_STAGE_ELEMS_PER_BLOCK,))
        if const_expr(EPILOGUE_REMAP):
            smem_epi_ptr = SmemPtr(
                base_ptr,
                smem_epi_offset,
                T.f32,
                shape=(EPI_STAGE_ELEMS_PER_BLOCK,),
            )
            epi_lds = STensor(smem_epi_ptr, T.f32, shape=(EPI_STAGE_ELEMS_PER_BLOCK,))

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

        wave_w_offset = fx.Index(wave_id) * fx.Index(const_expr(W_STAGE_ELEMS_PADDED))

        for k in range_constexpr(KV):
            mask_idx = fx.Index(m_tile) * fx.Index(const_expr(KV)) + fx.Index(const_expr(k))
            is_active = arith.cmpi(
                arith.CmpIPredicate.ne,
                mask_.load(mask_idx),
                zero_i32,
            )
            kv_if = scf.IfOp(is_active, results_=[], has_else=False)
            with ir.InsertionPoint(kv_if.then_block):
                # Prepare sparse A row bases once per kv, then reuse them for all c_blocks.
                row_tid_valid = arith.cmpi(
                    arith.CmpIPredicate.slt,
                    tid,
                    fx.Int32(const_expr(BLOCK_M)),
                )
                row_if = scf.IfOp(row_tid_valid, results_=[], has_else=False)
                with ir.InsertionPoint(row_if.then_block):
                    row_idx = fx.Index(tid)
                    lut_idx = (
                        fx.Index(m_tile) * fx.Index(const_expr(KV * BLOCK_M))
                        + fx.Index(const_expr(k * BLOCK_M))
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
                        fx.Index(const_expr(BLOCK_M)) + row_idx,
                        row_valid_i32,
                    )
                    scf.YieldOp([])
                gpu.barrier()

                for c_block in range_constexpr(0, C_IN, BLOCK_K):
                    if const_expr(BLOCK_K == 16 and c_block + BLOCK_K <= C_IN):
                        # Stage A with vec4 global loads. Sparse rows are irregular,
                        # but channels within each row are contiguous.
                        for ai in range_constexpr(A_VEC_LOAD_PER_BLOCK):
                            a_vec_idx = fx.Index(const_expr(ai * BLOCK_THREADS)) + fx.Index(tid)
                            a_vec_valid = arith.cmpi(
                                arith.CmpIPredicate.ult,
                                a_vec_idx,
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
                                    arith.CmpIPredicate.ne,
                                    row_valid_i32,
                                    zero_i32,
                                )
                                feat_off = (
                                    fx.Index(safe_row_base)
                                    + fx.Index(const_expr(c_block))
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
                                    a_lds.store(
                                        a_lds_base + fx.Index(const_expr(vi)),
                                        a_val,
                                    )
                                scf.YieldOp([])
                    else:
                        # Stage A[16 x BLOCK_K], reusing the prepared safe row map.
                        for ai in range_constexpr(A_LOAD_PER_BLOCK):
                            a_elem_idx = fx.Index(const_expr(ai * BLOCK_THREADS)) + fx.Index(tid)
                            a_valid = arith.cmpi(
                                arith.CmpIPredicate.ult,
                                a_elem_idx,
                                fx.Index(const_expr(A_STAGE_ELEMS)),
                            )
                            a_if = scf.IfOp(a_valid, results_=[], has_else=False)
                            with ir.InsertionPoint(a_if.then_block):
                                a_row_idx = a_elem_idx // fx.Index(const_expr(BLOCK_K))
                                a_k_idx = a_elem_idx % fx.Index(const_expr(BLOCK_K))
                                a_k_i32 = fx.Int32(arith.index_cast(T.i32, a_k_idx))
                                c_idx_i32 = a_k_i32 + fx.Int32(const_expr(c_block))
                                c_valid = arith.cmpi(
                                    arith.CmpIPredicate.slt,
                                    c_idx_i32,
                                    fx.Int32(const_expr(C_IN)),
                                )
                                safe_c_i32 = arith.select(c_valid, c_idx_i32, zero_i32)
                                safe_row_base = row_lds.load(a_row_idx)
                                row_valid_i32 = row_lds.load(
                                    fx.Index(const_expr(BLOCK_M)) + a_row_idx
                                )
                                row_valid = arith.cmpi(
                                    arith.CmpIPredicate.ne,
                                    row_valid_i32,
                                    zero_i32,
                                )
                                load_a = arith.andi(row_valid, c_valid)
                                feat_off = fx.Index(safe_row_base) + fx.Index(safe_c_i32)
                                a_val = feat_.load(feat_off)
                                a_val = arith.select(load_a, a_val, zero_f32)
                                a_lds.store(
                                    a_row_idx * fx.Index(const_expr(A_LDS_STRIDE)) + a_k_idx,
                                    a_val,
                                )
                                scf.YieldOp([])

                    w_base = fx.Index(
                        const_expr(k * N_C_OUT_TILES * C_IN * C_OUT_TILE)
                    ) + fx.Index(ct) * fx.Index(const_expr(C_IN * C_OUT_TILE))
                    if const_expr(BLOCK_K == 16 and c_block + BLOCK_K <= C_IN and C_OUT_TILE % W_VEC == 0):
                        for wi in range_constexpr(W_VEC_LOAD_PER_WAVE):
                            w_vec_idx = fx.Index(const_expr(wi * 64)) + fx.Index(lane)
                            w_vec_valid = arith.cmpi(
                                arith.CmpIPredicate.ult,
                                w_vec_idx,
                                fx.Index(const_expr(W_VEC_GROUPS)),
                            )
                            w_vec_if = scf.IfOp(w_vec_valid, results_=[], has_else=False)
                            with ir.InsertionPoint(w_vec_if.then_block):
                                w_k_idx = w_vec_idx // fx.Index(const_expr(C_OUT_TILE // W_VEC))
                                w_col_vec = w_vec_idx % fx.Index(const_expr(C_OUT_TILE // W_VEC))
                                w_src = (
                                    w_base
                                    + (fx.Index(const_expr(c_block)) + w_k_idx) * fx.Index(const_expr(C_OUT_TILE))
                                    + w_col_vec * fx.Index(const_expr(W_VEC))
                                )
                                w_vec = wp_.vec_load((w_src,), const_expr(W_VEC))
                                w_lds_base = (
                                    wave_w_offset
                                    + w_k_idx * fx.Index(const_expr(W_LDS_STRIDE))
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
                        for wi in range_constexpr(W_LOAD_PER_WAVE):
                            w_elem_idx = fx.Index(const_expr(wi * 64)) + fx.Index(lane)
                            w_valid = arith.cmpi(
                                arith.CmpIPredicate.ult,
                                w_elem_idx,
                                fx.Index(const_expr(W_STAGE_ELEMS)),
                            )
                            w_if = scf.IfOp(w_valid, results_=[], has_else=False)
                            with ir.InsertionPoint(w_if.then_block):
                                w_k_idx = w_elem_idx // fx.Index(const_expr(C_OUT_TILE))
                                w_col_idx = w_elem_idx % fx.Index(const_expr(C_OUT_TILE))
                                w_k_i32 = fx.Int32(arith.index_cast(T.i32, w_k_idx))
                                c_idx_i32 = w_k_i32 + fx.Int32(const_expr(c_block))
                                c_valid = arith.cmpi(
                                    arith.CmpIPredicate.slt,
                                    c_idx_i32,
                                    fx.Int32(const_expr(C_IN)),
                                )
                                safe_c_i32 = arith.select(c_valid, c_idx_i32, zero_i32)
                                w_src = (
                                    w_base
                                    + fx.Index(safe_c_i32) * fx.Index(const_expr(C_OUT_TILE))
                                    + w_col_idx
                                )
                                b_val = wp_.load(w_src)
                                b_val = arith.select(c_valid, b_val, zero_f32)
                                w_lds.store(
                                    wave_w_offset
                                    + w_k_idx * fx.Index(const_expr(W_LDS_STRIDE))
                                    + w_col_idx,
                                    b_val,
                                )
                                scf.YieldOp([])
                    gpu.barrier()

                    for kk in range_constexpr(0, BLOCK_K, 4):
                        a_off = (
                            fx.Index(mfma_row) * fx.Index(const_expr(A_LDS_STRIDE))
                            + fx.Index(const_expr(kk))
                            + fx.Index(mfma_k_lane)
                        )
                        a_val = a_lds.load(a_off)
                        b_off = (
                            wave_w_offset
                            + fx.Index(const_expr(kk * W_LDS_STRIDE))
                            + fx.Index(mfma_k_lane) * fx.Index(const_expr(W_LDS_STRIDE))
                            + fx.Index(mfma_col)
                        )
                        b_val = w_lds.load(b_off)
                        cur_acc = fx.memref_load_vec(acc_reg)
                        new_acc = rocdl.mfma_f32_16x16x4f32(
                            T.vec(4, T.f32), a_val, b_val, cur_acc, 0, 0, 0
                        )
                        fx.memref_store_vec(new_acc, acc_reg)

                    if const_expr(c_block + BLOCK_K < C_IN):
                        gpu.barrier()
                scf.YieldOp([])

        final_acc = fx.memref_load_vec(acc_reg)
        if const_expr(EPILOGUE_REMAP):
            wave_epi_offset = fx.Index(wave_id) * fx.Index(const_expr(EPI_STAGE_ELEMS))
            for ri in range_constexpr(4):
                epi_row = c_row_vec_base + fx.Int32(const_expr(ri))
                epi_off = (
                    wave_epi_offset
                    + fx.Index(epi_row) * fx.Index(const_expr(C_OUT_TILE))
                    + fx.Index(mfma_col)
                )
                val = vector.extract(
                    final_acc,
                    static_position=[const_expr(ri)],
                    dynamic_position=[],
                )
                epi_lds.store(epi_off, val)
            gpu.barrier()

            epi_store_row = lane // fx.Int32(const_expr(4))
            epi_col_vec = lane % fx.Int32(const_expr(4))
            store_row = row_tile_base + epi_store_row
            store_ok = arith.cmpi(arith.CmpIPredicate.slt, store_row, num_act_out_val)
            store_if = scf.IfOp(store_ok, results_=[], has_else=False)
            with ir.InsertionPoint(store_if.then_block):
                epi_vec_off = (
                    wave_epi_offset
                    + fx.Index(epi_store_row) * fx.Index(const_expr(C_OUT_TILE))
                    + fx.Index(epi_col_vec) * fx.Index(const_expr(4))
                )
                epi_vec = epi_lds.vec_load((epi_vec_off,), const_expr(4))
                out_off = (
                    fx.Index(store_row) * fx.Index(const_expr(C_OUT))
                    + c_out_offset
                    + fx.Index(epi_col_vec) * fx.Index(const_expr(4))
                )
                out_.vec_store((out_off,), epi_vec, const_expr(4))
                scf.YieldOp([])
        else:
            for ri in range_constexpr(4):
                store_row = row_tile_base + c_row_vec_base + fx.Int32(const_expr(ri))
                store_ok = arith.cmpi(arith.CmpIPredicate.slt, store_row, num_act_out_val)
                store_if = scf.IfOp(store_ok, results_=[], has_else=False)
                with ir.InsertionPoint(store_if.then_block):
                    val = vector.extract(final_acc, static_position=[const_expr(ri)], dynamic_position=[])
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
        mask: fx.Tensor,
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
            mask,
            num_act_out,
        ).launch(
            grid=(num_tiles * N_C_OUT_TILE_GROUPS,),
            block=(BLOCK_THREADS,),
            stream=stream,
        )

    return launch_fn, BLOCK_M


def _implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_forward(
    features: torch.Tensor,
    filters: torch.Tensor,
    indice_pairs: torch.Tensor,
    indice_pair_num: torch.Tensor,
    num_activate_out: int,
    block_k: int,
    epilogue: str = "direct",
) -> Optional[torch.Tensor]:
    """Run the f32 16x16x4 MFMA N2 A-shared K-pipe kernel."""
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

    block_m = 16
    num_tiles = (num_activate_out + block_m - 1) // block_m
    sorted_inp, sorted_out, sorted_kv, mask, pair_start, pair_end, inp_row_lut = (
        hip.build_implicit_gemm_mask(indice_pairs, indice_pair_num, num_activate_out, block_m)
    )
    if sorted_inp.shape[0] == 0:
        return torch.zeros(num_activate_out, c_out, dtype=torch.float32, device=device)

    if epilogue == "direct":
        name = f"mfma_f32_16x16x4f32_n2_ashared_kpipe_bk{block_k}"
    else:
        name = f"mfma_f32_16x16x4f32_n2_ashared_kpipe_bk{block_k}_{epilogue}"
    key = (name, c_in, c_out, kv, dtype_str, block_k, epilogue)
    if key not in MFMA_F32_16X16X4F32_N2_ASHARED_KPIPE_COMPILED_KERNELS:
        try:
            launch_fn, block_m = _compile_implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe(
                c_in, c_out, kv, dtype_str, block_k, epilogue
            )
            MFMA_F32_16X16X4F32_N2_ASHARED_KPIPE_COMPILED_KERNELS[key] = (
                launch_fn,
                block_m,
            )
        except Exception as e:
            import traceback

            warnings.warn(
                "Failed to compile mfma_f32_16x16x4f32_n2_ashared_kpipe "
                f"kernel: {e}\n{traceback.format_exc()}"
            )
            return None
    else:
        launch_fn, block_m = (
            MFMA_F32_16X16X4F32_N2_ASHARED_KPIPE_COMPILED_KERNELS[key]
        )

    weights_packed = _pack_weights(filters, 16)
    out_features = torch.zeros(num_activate_out, c_out, dtype=torch.float32, device=device)
    stream = torch.cuda.current_stream()

    launch_fn(
        features.contiguous(),
        weights_packed,
        out_features,
        inp_row_lut.reshape(-1).contiguous(),
        mask.reshape(-1).contiguous(),
        num_tiles,
        num_activate_out,
        stream,
    )
    return out_features


def implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_forward(
    features: torch.Tensor,
    filters: torch.Tensor,
    indice_pairs: torch.Tensor,
    indice_pair_num: torch.Tensor,
    num_activate_out: int,
) -> Optional[torch.Tensor]:
    """Run the f32 16x16x4 MFMA N2 A-shared K-pipe BK16 kernel."""
    return _implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_forward(
        features, filters, indice_pairs, indice_pair_num, num_activate_out, 16
    )


def implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk16_remap_forward(
    features: torch.Tensor,
    filters: torch.Tensor,
    indice_pairs: torch.Tensor,
    indice_pair_num: torch.Tensor,
    num_activate_out: int,
) -> Optional[torch.Tensor]:
    """Run the BK16 K-pipe kernel with LDS epilogue remap."""
    return _implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_forward(
        features, filters, indice_pairs, indice_pair_num, num_activate_out, 16, "remap"
    )


def implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk32_forward(
    features: torch.Tensor,
    filters: torch.Tensor,
    indice_pairs: torch.Tensor,
    indice_pair_num: torch.Tensor,
    num_activate_out: int,
) -> Optional[torch.Tensor]:
    """Run the f32 16x16x4 MFMA N2 A-shared K-pipe BK32 kernel."""
    return _implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_forward(
        features, filters, indice_pairs, indice_pair_num, num_activate_out, 32
    )


def implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_forward(
    features: torch.Tensor,
    filters: torch.Tensor,
    indice_pairs: torch.Tensor,
    indice_pair_num: torch.Tensor,
    num_activate_out: int,
) -> Optional[torch.Tensor]:
    """Compatibility alias for the BK32 K-pipe kernel."""
    return implicit_gemm_mfma_f32_16x16x4f32_n2_ashared_kpipe_bk32_forward(
        features, filters, indice_pairs, indice_pair_num, num_activate_out
    )
