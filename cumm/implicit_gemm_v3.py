"""V3: Pair-scan implicit GEMM kernel.

Weight in LDS + per-thread pair scan (scf.ForOp over pair_start..pair_end).
First FlyDSL fused kernel — gather+dot+scatter in one kernel.
"""
import math
import os
from typing import Optional, Dict

import torch

from cumm.implicit_gemm_common import (
    _get_hip_module, _ensure_flydsl_path, _forward_common,
)

_V3_COMPILED_KERNELS: Dict = {}


def _compile_implicit_gemm_v3(c_in: int, c_out: int, kv: int, dtype_str: str):
    _ensure_flydsl_path()

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import gpu, arith, range_constexpr, const_expr, buffer_ops, rocdl, vector
    from flydsl.expr.typing import T
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import scf
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
    from flydsl.compiler.kernel_function import CompilationContext
    from kernels.tensor_shim import GTensor, STensor

    C_IN = c_in
    C_OUT = c_out
    KV = kv
    BLOCK_M = 64
    W_ELEMS = C_IN * C_OUT
    OUT_VEC = min(4, C_OUT)
    C_OUT_VECS = C_OUT // OUT_VEC

    if dtype_str == 'f32':
        DT_BYTES = 4
    elif dtype_str in ('f16', 'bf16'):
        DT_BYTES = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    allocator = SmemAllocator(None, arch="gfx942", global_sym_name="smem_ig_v3")
    smem_w_offset = allocator._align(allocator.ptr, 16)
    W_BYTES = W_ELEMS * DT_BYTES
    allocator.ptr = smem_w_offset + W_BYTES

    LDG_VEC = min(4, W_ELEMS)
    W_VEC_LOAD_PER_THREAD = math.ceil(W_ELEMS / (BLOCK_M * LDG_VEC))

    def _make_kernel(is_f32=True, use_f16=True):
        NEED_EXTF = not is_f32
        DT_TAG = 'f32' if is_f32 else ('f16' if use_f16 else 'bf16')

        @flyc.kernel(known_block_size=[BLOCK_M, 1, 1])
        def kernel(features: fx.Tensor, weights: fx.Tensor, output: fx.Tensor,
                   sorted_inp: fx.Tensor, sorted_out: fx.Tensor,
                   mask: fx.Tensor, pair_start: fx.Tensor, pair_end: fx.Tensor,
                   num_act_out_val: fx.Int32):
            dt = T.f32 if const_expr(DT_TAG == 'f32') else (T.f16 if const_expr(DT_TAG == 'f16') else T.bf16)
            tid = fx.Int32(gpu.thread_idx.x)
            bid = fx.Int32(gpu.block_idx.x)

            feat_ = GTensor(features, dtype=dt, shape=(-1,))
            w_ = GTensor(weights, dtype=dt, shape=(-1,))
            out_ = GTensor(output, dtype=T.f32, shape=(-1,))
            sinp_ = GTensor(sorted_inp, dtype=T.i32, shape=(-1,))
            sout_ = GTensor(sorted_out, dtype=T.i32, shape=(-1,))
            mask_ = GTensor(mask, dtype=T.i32, shape=(-1,))
            ps_ = GTensor(pair_start, dtype=T.i32, shape=(-1,))
            pe_ = GTensor(pair_end, dtype=T.i32, shape=(-1,))

            base_ptr = allocator.get_base()
            smem_w_ptr = SmemPtr(base_ptr, smem_w_offset, dt, shape=(W_ELEMS,))
            w_lds = STensor(smem_w_ptr, dt, shape=(W_ELEMS,))

            my_out_row = fx.Int32(bid) * fx.Int32(const_expr(BLOCK_M)) + tid
            out_row_base = fx.Index(my_out_row) * fx.Index(const_expr(C_OUT))

            for k in range_constexpr(KV):
                mask_idx = fx.Index(bid) * fx.Index(const_expr(KV)) + fx.Index(const_expr(k))
                is_active = arith.cmpi(arith.CmpIPredicate.ne,
                                       mask_.load(mask_idx),
                                       fx.Int32(arith.constant(0, type=T.i32)))
                kv_if = scf.IfOp(is_active, results_=[], has_else=False)
                with ir.InsertionPoint(kv_if.then_block):
                    w_base = fx.Index(const_expr(k * W_ELEMS))
                    for wi in range_constexpr(W_VEC_LOAD_PER_THREAD):
                        w_elem_idx = fx.Index(const_expr(wi * BLOCK_M * LDG_VEC)) + fx.Index(tid) * fx.Index(const_expr(LDG_VEC))
                        w_valid = arith.cmpi(arith.CmpIPredicate.ult, w_elem_idx, fx.Index(const_expr(W_ELEMS)))
                        w_if = scf.IfOp(w_valid, results_=[], has_else=False)
                        with ir.InsertionPoint(w_if.then_block):
                            w_vec = w_.vec_load((w_base + w_elem_idx,), const_expr(LDG_VEC))
                            w_lds.vec_store((w_elem_idx,), w_vec, const_expr(LDG_VEC))
                            scf.YieldOp([])
                    gpu.barrier()

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

                                for j in range_constexpr(C_OUT_VECS):
                                    out_off = out_row_base + fx.Index(const_expr(j * OUT_VEC))
                                    acc = out_.vec_load((out_off,), const_expr(OUT_VEC))

                                    for c in range_constexpr(C_IN):
                                        f_val = feat_.load(feat_base + fx.Index(const_expr(c)))
                                        if const_expr(NEED_EXTF):
                                            f_val = arith.extf(T.f32, f_val)
                                        f_bcast = vector.broadcast(
                                            T.vec(const_expr(OUT_VEC), T.f32), f_val)

                                        w_off = fx.Index(const_expr(c * C_OUT + j * OUT_VEC))
                                        w_vec = w_lds.vec_load((w_off,), const_expr(OUT_VEC))
                                        if const_expr(NEED_EXTF):
                                            w_f32_elems = []
                                            for ve in range_constexpr(OUT_VEC):
                                                e = vector.extract(w_vec,
                                                    static_position=[const_expr(ve)],
                                                    dynamic_position=[])
                                                w_f32_elems.append(arith.extf(T.f32, e))
                                            w_f32 = vector.from_elements(
                                                T.vec(const_expr(OUT_VEC), T.f32), w_f32_elems)
                                        else:
                                            w_f32 = w_vec

                                        acc = arith.addf(acc, arith.mulf(f_bcast, w_f32))

                                    out_.vec_store((out_off,), acc, const_expr(OUT_VEC))

                                scf.YieldOp([])
                            scf.YieldOp([])
                        scf.YieldOp([])
                    gpu.barrier()
                    scf.YieldOp([])

        return kernel

    if dtype_str == 'f32':
        implicit_gemm_kernel = _make_kernel(is_f32=True)
    elif dtype_str == 'f16':
        implicit_gemm_kernel = _make_kernel(is_f32=False, use_f16=True)
    else:
        implicit_gemm_kernel = _make_kernel(is_f32=False, use_f16=False)

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


def implicit_gemm_v3_forward(
    features: torch.Tensor,
    filters: torch.Tensor,
    indice_pairs: torch.Tensor,
    indice_pair_num: torch.Tensor,
    num_activate_out: int,
) -> Optional[torch.Tensor]:
    return _forward_common(
        features, filters, indice_pairs, indice_pair_num, num_activate_out,
        version_tag='v3',
        compiled_kernels=_V3_COMPILED_KERNELS,
        compile_fn=_compile_implicit_gemm_v3,
        needs_sorted_arrays=True,
    )
