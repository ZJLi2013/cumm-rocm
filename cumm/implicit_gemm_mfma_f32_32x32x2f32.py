"""mfma_f32_32x32x2f32 implicit GEMM family member.

This is the first 32x32 MFMA candidate:
  - f32 only
  - one wave owns one 32 x 32 output tile
  - A/features: global/LUT gather -> VGPR scalar fragment
  - B/weight: global -> LDS -> VGPR scalar fragment
  - C: VGPR accumulator fragment updated by mfma_f32_32x32x2f32
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

MFMA_F32_32X32X2F32_COMPILED_KERNELS: Dict = {}


def _compile_implicit_gemm_mfma_f32_32x32x2f32(
    c_in: int, c_out: int, kv: int, dtype_str: str
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
        raise ValueError("mfma_f32_32x32x2f32 currently supports f32 only")

    C_IN = c_in
    C_OUT = c_out
    KV = kv
    BLOCK_M = 32
    C_OUT_TILE = 32
    N_C_OUT_TILES = C_OUT // C_OUT_TILE
    BLOCK_THREADS = 64

    W_TILE_ELEMS = C_IN * C_OUT_TILE
    W_TILE_BYTES = W_TILE_ELEMS * 4
    allocator = SmemAllocator(None, arch="gfx942", global_sym_name="smem_ig_mfma_f32_32x32x2f32")
    smem_w_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = smem_w_offset + W_TILE_BYTES

    LDG_VEC = min(4, W_TILE_ELEMS)
    W_VEC_LOAD_PER_THREAD = math.ceil(W_TILE_ELEMS / (BLOCK_THREADS * LDG_VEC))

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
        smem_w_ptr = SmemPtr(base_ptr, smem_w_offset, T.f32, shape=(W_TILE_ELEMS,))
        w_lds = STensor(smem_w_ptr, T.f32, shape=(W_TILE_ELEMS,))

        m_tile = bid // fx.Int32(const_expr(N_C_OUT_TILES))
        ct = bid % fx.Int32(const_expr(N_C_OUT_TILES))
        row_tile_base = m_tile * fx.Int32(const_expr(BLOCK_M))
        c_out_offset = fx.Index(ct) * fx.Index(const_expr(C_OUT_TILE))

        # MFMA 32x32x2f32 lane layout mirrors the 16x16 kernel, but each lane
        # owns 16 accumulator rows for one output column.
        lane = tid
        mfma_row = lane % fx.Int32(const_expr(32))
        mfma_col = lane % fx.Int32(const_expr(32))
        mfma_k_lane = lane // fx.Int32(const_expr(32))
        c_row_vec_base = (lane // fx.Int32(const_expr(32))) * fx.Int32(const_expr(16))

        zero_f32 = arith.constant(0.0, type=T.f32)
        zero_acc = arith.constant_vector(0.0, T.vec(16, T.f32))

        acc_reg_ty = fx.MemRefType.get(
            T.f32, fx.LayoutType.get(const_expr(16), 1), fx.AddressSpace.Register
        )
        acc_reg_lay = fx.make_layout(const_expr(16), 1)
        acc_reg = fx.memref_alloca(acc_reg_ty, acc_reg_lay)
        fx.memref_store_vec(zero_acc, acc_reg)

        for k in range_constexpr(KV):
            mask_idx = fx.Index(m_tile) * fx.Index(const_expr(KV)) + fx.Index(const_expr(k))
            is_active = arith.cmpi(
                arith.CmpIPredicate.ne,
                mask_.load(mask_idx),
                fx.Int32(arith.constant(0, type=T.i32)),
            )
            kv_if = scf.IfOp(is_active, results_=[], has_else=False)
            with ir.InsertionPoint(kv_if.then_block):
                w_base = fx.Index(
                    const_expr(k * N_C_OUT_TILES * W_TILE_ELEMS)
                ) + fx.Index(ct) * fx.Index(const_expr(W_TILE_ELEMS))
                for wi in range_constexpr(W_VEC_LOAD_PER_THREAD):
                    w_elem_idx = (
                        fx.Index(const_expr(wi * BLOCK_THREADS * LDG_VEC))
                        + fx.Index(tid) * fx.Index(const_expr(LDG_VEC))
                    )
                    w_valid = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        w_elem_idx,
                        fx.Index(const_expr(W_TILE_ELEMS)),
                    )
                    w_if = scf.IfOp(w_valid, results_=[], has_else=False)
                    with ir.InsertionPoint(w_if.then_block):
                        w_vec = wp_.vec_load((w_base + w_elem_idx,), const_expr(LDG_VEC))
                        w_lds.vec_store((w_elem_idx,), w_vec, const_expr(LDG_VEC))
                        scf.YieldOp([])
                gpu.barrier()

                lut_idx = (
                    fx.Index(m_tile) * fx.Index(const_expr(KV * BLOCK_M))
                    + fx.Index(const_expr(k * BLOCK_M))
                    + fx.Index(mfma_row)
                )
                inp_row = lut_.load(lut_idx)
                out_row = row_tile_base + mfma_row
                row_in_bounds = arith.cmpi(
                    arith.CmpIPredicate.slt, out_row, num_act_out_val
                )
                has_pair = arith.cmpi(
                    arith.CmpIPredicate.sge,
                    inp_row,
                    fx.Int32(arith.constant(0, type=T.i32)),
                )
                load_a = arith.andi(row_in_bounds, has_pair)
                safe_inp_row = arith.select(
                    load_a,
                    inp_row,
                    fx.Int32(arith.constant(0, type=T.i32)),
                )
                feat_base = fx.Index(safe_inp_row) * fx.Index(const_expr(C_IN))

                for c0 in range_constexpr(0, C_IN, 2):
                    a_val = feat_.load(
                        feat_base + fx.Index(const_expr(c0)) + fx.Index(mfma_k_lane)
                    )
                    a_val = arith.select(load_a, a_val, zero_f32)
                    b_off = (
                        fx.Index(const_expr(c0 * C_OUT_TILE))
                        + fx.Index(mfma_k_lane) * fx.Index(const_expr(C_OUT_TILE))
                        + fx.Index(mfma_col)
                    )
                    b_val = w_lds.load(b_off)
                    cur_acc = fx.memref_load_vec(acc_reg)
                    new_acc = rocdl.mfma_f32_32x32x2f32(
                        T.vec(16, T.f32), a_val, b_val, cur_acc, 0, 0, 0
                    )
                    fx.memref_store_vec(new_acc, acc_reg)

                gpu.barrier()
                scf.YieldOp([])

        final_acc = fx.memref_load_vec(acc_reg)
        for ri in range_constexpr(16):
            store_row = row_tile_base + c_row_vec_base + fx.Int32(const_expr(ri))
            store_ok = arith.cmpi(
                arith.CmpIPredicate.slt, store_row, num_act_out_val
            )
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
            grid=(num_tiles * N_C_OUT_TILES,),
            block=(BLOCK_THREADS,),
            stream=stream,
        )

    return launch_fn, BLOCK_M


def implicit_gemm_mfma_f32_32x32x2f32_forward(
    features: torch.Tensor,
    filters: torch.Tensor,
    indice_pairs: torch.Tensor,
    indice_pair_num: torch.Tensor,
    num_activate_out: int,
) -> Optional[torch.Tensor]:
    """Run the f32 32x32x2 MFMA implicit GEMM kernel."""
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
    if c_in % 2 != 0 or c_out % 32 != 0:
        return None

    block_m = 32
    num_tiles = (num_activate_out + block_m - 1) // block_m
    sorted_inp, sorted_out, sorted_kv, mask, pair_start, pair_end, inp_row_lut = (
        hip.build_implicit_gemm_mask(indice_pairs, indice_pair_num, num_activate_out, block_m)
    )
    if sorted_inp.shape[0] == 0:
        return torch.zeros(num_activate_out, c_out, dtype=torch.float32, device=device)

    key = ("mfma_f32_32x32x2f32", c_in, c_out, kv, dtype_str)
    if key not in MFMA_F32_32X32X2F32_COMPILED_KERNELS:
        try:
            launch_fn, block_m = _compile_implicit_gemm_mfma_f32_32x32x2f32(
                c_in, c_out, kv, dtype_str
            )
            MFMA_F32_32X32X2F32_COMPILED_KERNELS[key] = (launch_fn, block_m)
        except Exception as e:
            import traceback

            warnings.warn(
                "Failed to compile mfma_f32_32x32x2f32 implicit GEMM "
                f"kernel: {e}\n{traceback.format_exc()}"
            )
            return None
    else:
        launch_fn, block_m = MFMA_F32_32X32X2F32_COMPILED_KERNELS[key]

    weights_packed = _pack_weights(filters, 32)
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
