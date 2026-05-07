"""Tile-owned scalar implicit GEMM family member.

This is a CUDA-style scalar tile kernel:
  - A/features global -> register, B/weight -> LDS, C -> register
  - block owns a BLOCK_M x C_OUT_TILE output tile
  - each thread owns one (output row, output-channel vector) inside the tile

It is intentionally still scalar FMA (no MFMA, no A LDS) and acts as the
long-lived scalar fallback for the implicit GEMM family.
"""
import math
import warnings
from typing import Dict, Optional

import torch

from cumm.implicit_gemm_common import (
    C_OUT_TILE_MAX,
    _dtype_to_str,
    _ensure_flydsl_path,
    _get_hip_module,
    _pack_weights,
)

SCALAR_TILE_COMPILED_KERNELS: Dict = {}


def _compile_implicit_gemm_scalar_tile(c_in: int, c_out: int, kv: int, dtype_str: str):
    _ensure_flydsl_path()

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import arith, const_expr, gpu, range_constexpr, vector
    from flydsl.expr.typing import T
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import scf
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
    from kernels.tensor_shim import GTensor, STensor

    C_IN = c_in
    C_OUT = c_out
    KV = kv
    BLOCK_M = 64

    C_OUT_TILE = min(C_OUT_TILE_MAX, C_OUT)
    N_C_OUT_TILES = C_OUT // C_OUT_TILE
    OUT_VEC = min(4, C_OUT_TILE)
    C_OUT_TILE_VECS = C_OUT_TILE // OUT_VEC
    BLOCK_THREADS = BLOCK_M * C_OUT_TILE_VECS

    W_TILE_ELEMS = C_IN * C_OUT_TILE
    if dtype_str == "f32":
        DT_BYTES = 4
    elif dtype_str in ("f16", "bf16"):
        DT_BYTES = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    W_TILE_BYTES = W_TILE_ELEMS * DT_BYTES
    allocator = SmemAllocator(None, arch="gfx942", global_sym_name="smem_ig_scalar_tile")
    smem_w_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = smem_w_offset + W_TILE_BYTES

    LDG_VEC = min(4, W_TILE_ELEMS)
    W_VEC_LOAD_PER_THREAD = math.ceil(W_TILE_ELEMS / (BLOCK_THREADS * LDG_VEC))

    def _make_kernel(is_f32=True, use_f16=True):
        NEED_EXTF = not is_f32
        DT_TAG = "f32" if is_f32 else ("f16" if use_f16 else "bf16")

        @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
        def kernel(features: fx.Tensor, weights_packed: fx.Tensor, output: fx.Tensor,
                   inp_row_lut: fx.Tensor, mask: fx.Tensor,
                   num_act_out_val: fx.Int32):
            dt = T.f32 if const_expr(DT_TAG == "f32") else (T.f16 if const_expr(DT_TAG == "f16") else T.bf16)
            tid = fx.Int32(gpu.thread_idx.x)
            bid = fx.Int32(gpu.block_idx.x)

            feat_ = GTensor(features, dtype=dt, shape=(-1,))
            wp_ = GTensor(weights_packed, dtype=dt, shape=(-1,))
            out_ = GTensor(output, dtype=T.f32, shape=(-1,))
            lut_ = GTensor(inp_row_lut, dtype=T.i32, shape=(-1,))
            mask_ = GTensor(mask, dtype=T.i32, shape=(-1,))

            base_ptr = allocator.get_base()
            smem_w_ptr = SmemPtr(base_ptr, smem_w_offset, dt, shape=(W_TILE_ELEMS,))
            w_lds = STensor(smem_w_ptr, dt, shape=(W_TILE_ELEMS,))

            # Tile ownership: C_OUT_TILE_VECS threads cooperate on one output row.
            row_lane = tid // fx.Int32(const_expr(C_OUT_TILE_VECS))
            col_vec = tid % fx.Int32(const_expr(C_OUT_TILE_VECS))
            my_out_row = bid * fx.Int32(const_expr(BLOCK_M)) + row_lane
            valid_thread = arith.cmpi(arith.CmpIPredicate.slt, my_out_row, num_act_out_val)
            out_row_base = fx.Index(my_out_row) * fx.Index(const_expr(C_OUT))

            zero_scalar = arith.constant(0.0, type=T.f32)
            zero_vec = vector.broadcast(T.vec(const_expr(OUT_VEC), T.f32), zero_scalar)

            acc_reg_ty = fx.MemRefType.get(
                T.f32, fx.LayoutType.get(const_expr(OUT_VEC), 1),
                fx.AddressSpace.Register,
            )
            acc_reg_lay = fx.make_layout(const_expr(OUT_VEC), 1)

            for ct in range_constexpr(N_C_OUT_TILES):
                c_out_offset = fx.Index(const_expr(ct * C_OUT_TILE))
                acc_reg = fx.memref_alloca(acc_reg_ty, acc_reg_lay)
                fx.memref_store_vec(zero_vec, acc_reg)

                for k in range_constexpr(KV):
                    mask_idx = fx.Index(bid) * fx.Index(const_expr(KV)) + fx.Index(const_expr(k))
                    is_active = arith.cmpi(
                        arith.CmpIPredicate.ne,
                        mask_.load(mask_idx),
                        fx.Int32(arith.constant(0, type=T.i32)),
                    )
                    kv_if = scf.IfOp(is_active, results_=[], has_else=False)
                    with ir.InsertionPoint(kv_if.then_block):
                        w_base = fx.Index(const_expr(k * N_C_OUT_TILES * W_TILE_ELEMS + ct * W_TILE_ELEMS))
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

                        thread_if = scf.IfOp(valid_thread, results_=[], has_else=False)
                        with ir.InsertionPoint(thread_if.then_block):
                            lut_idx = (
                                fx.Index(bid) * fx.Index(const_expr(KV * BLOCK_M))
                                + fx.Index(const_expr(k * BLOCK_M))
                                + fx.Index(row_lane)
                            )
                            inp_row = lut_.load(lut_idx)
                            has_pair = arith.cmpi(
                                arith.CmpIPredicate.sge,
                                inp_row,
                                fx.Int32(arith.constant(0, type=T.i32)),
                            )
                            pair_if = scf.IfOp(has_pair, results_=[], has_else=False)
                            with ir.InsertionPoint(pair_if.then_block):
                                feat_base = fx.Index(inp_row) * fx.Index(const_expr(C_IN))

                                for c in range_constexpr(C_IN):
                                    f_val = feat_.load(feat_base + fx.Index(const_expr(c)))
                                    if const_expr(NEED_EXTF):
                                        f_val = arith.extf(T.f32, f_val)
                                    f_bcast = vector.broadcast(T.vec(const_expr(OUT_VEC), T.f32), f_val)

                                    w_off = (
                                        fx.Index(const_expr(c * C_OUT_TILE))
                                        + fx.Index(col_vec) * fx.Index(const_expr(OUT_VEC))
                                    )
                                    w_vec = w_lds.vec_load((w_off,), const_expr(OUT_VEC))
                                    if const_expr(NEED_EXTF):
                                        w_f32_elems = []
                                        for ve in range_constexpr(OUT_VEC):
                                            e = vector.extract(
                                                w_vec,
                                                static_position=[const_expr(ve)],
                                                dynamic_position=[],
                                            )
                                            w_f32_elems.append(arith.extf(T.f32, e))
                                        w_f32 = vector.from_elements(
                                            T.vec(const_expr(OUT_VEC), T.f32), w_f32_elems
                                        )
                                    else:
                                        w_f32 = w_vec

                                    cur_acc = fx.memref_load_vec(acc_reg)
                                    new_acc = arith.addf(cur_acc, arith.mulf(f_bcast, w_f32))
                                    fx.memref_store_vec(new_acc, acc_reg)

                                scf.YieldOp([])
                            scf.YieldOp([])
                        gpu.barrier()
                        scf.YieldOp([])

                ep_if = scf.IfOp(valid_thread, results_=[], has_else=False)
                with ir.InsertionPoint(ep_if.then_block):
                    out_off = (
                        out_row_base
                        + c_out_offset
                        + fx.Index(col_vec) * fx.Index(const_expr(OUT_VEC))
                    )
                    final_acc = fx.memref_load_vec(acc_reg)
                    out_.vec_store((out_off,), final_acc, const_expr(OUT_VEC))
                    scf.YieldOp([])

        return kernel

    if dtype_str == "f32":
        kernel = _make_kernel(is_f32=True)
    elif dtype_str == "f16":
        kernel = _make_kernel(is_f32=False, use_f16=True)
    else:
        kernel = _make_kernel(is_f32=False, use_f16=False)

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
            features, weights_packed, output,
            inp_row_lut, mask,
            num_act_out,
        ).launch(
            grid=(num_tiles,),
            block=(BLOCK_THREADS,),
            stream=stream,
        )

    return launch_fn, BLOCK_M


def implicit_gemm_scalar_tile_forward(
    features: torch.Tensor,
    filters: torch.Tensor,
    indice_pairs: torch.Tensor,
    indice_pair_num: torch.Tensor,
    num_activate_out: int,
) -> Optional[torch.Tensor]:
    """Run the tile-owned scalar implicit GEMM fallback."""
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

    if c_in % 4 != 0 or c_out % 4 != 0:
        return None

    dtype_str = _dtype_to_str(dtype)
    if dtype_str is None:
        return None

    BLOCK_M = 64
    num_tiles = (num_activate_out + BLOCK_M - 1) // BLOCK_M

    sorted_inp, sorted_out, sorted_kv, mask, pair_start, pair_end, inp_row_lut = (
        hip.build_implicit_gemm_mask(indice_pairs, indice_pair_num, num_activate_out, BLOCK_M)
    )

    if sorted_inp.shape[0] == 0:
        return torch.zeros(num_activate_out, c_out, dtype=torch.float32, device=device)

    key = ("scalar_tile", c_in, c_out, kv, dtype_str)
    if key not in SCALAR_TILE_COMPILED_KERNELS:
        try:
            launch_fn, block_m = _compile_implicit_gemm_scalar_tile(c_in, c_out, kv, dtype_str)
            SCALAR_TILE_COMPILED_KERNELS[key] = (launch_fn, block_m)
        except Exception as e:
            import traceback

            warnings.warn(
                f"Failed to compile scalar_tile implicit GEMM kernel: {e}\n{traceback.format_exc()}"
            )
            return None
    else:
        launch_fn, block_m = SCALAR_TILE_COMPILED_KERNELS[key]

    c_out_tile = min(C_OUT_TILE_MAX, c_out)
    weights_packed = _pack_weights(filters, c_out_tile)

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

    if dtype != torch.float32:
        out_features = out_features.to(dtype)

    return out_features
