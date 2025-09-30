# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_chunk_state.py

# ruff: noqa: E501

import torch

from vllm.triton_utils import tl, triton

from .mamba_ssm import softplus


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_H': 2}),
        triton.Config({'BLOCK_SIZE_H': 4}),
        triton.Config({'BLOCK_SIZE_H': 8}),
        triton.Config({'BLOCK_SIZE_H': 16}),
        triton.Config({'BLOCK_SIZE_H': 32}),
        triton.Config({'BLOCK_SIZE_H': 64}),
    ],
    key=['chunk_size', 'nheads'],
)
@triton.jit
def _chunk_cumsum_fwd_kernel(
    # Pointers to matrices
    dt_ptr,
    A_ptr,
    dt_bias_ptr,
    dt_out_ptr,
    dA_cumsum_ptr,
    cu_chunk_seqlens_ptr,
    # Matrix dimension
    batch,
    seqlen,
    nheads,
    chunk_size,
    dt_min,
    dt_max,
    # Strides
    stride_dt_batch,
    stride_dt_seqlen,
    stride_dt_head,
    stride_A_head,
    stride_dt_bias_head,
    stride_dt_out_batch,
    stride_dt_out_chunk,
    stride_dt_out_head,
    stride_dt_out_csize,
    stride_dA_cs_batch,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dA_cs_csize,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_CHUNK: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)

    # if dt is long, may cause problems, so use 64 bit
    # https://github.com/triton-lang/triton/issues/1058
    pid_c = tl.program_id(axis=1).to(tl.int64)
    pid_h = tl.program_id(axis=2)

    chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
    chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)

    dt_ptr += pid_b * stride_dt_batch + chunk_seqlen_start * stride_dt_seqlen
    dt_out_ptr += pid_b * stride_dt_out_batch + pid_c * stride_dt_out_chunk
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    dt_ptrs = dt_ptr + (offs_h[:, None] * stride_dt_head +
                        offs_c[None, :] * stride_dt_seqlen)
    A_ptrs = A_ptr + offs_h * stride_A_head
    dt_out_ptrs = dt_out_ptr + (offs_h[:, None] * stride_dt_out_head +
                                offs_c[None, :] * stride_dt_out_csize)
    dA_cs_ptrs = dA_cumsum_ptr + (offs_h[:, None] * stride_dA_cs_head +
                                  offs_c[None, :] * stride_dA_cs_csize)
    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

    dt = tl.load(dt_ptrs,
                 mask=(offs_h[:, None] < nheads) &
                 (offs_c[None, :] < chunk_size_limit),
                 other=0.0).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + offs_h * stride_dt_bias_head,
                          mask=offs_h < nheads,
                          other=0.0).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, softplus(dt), dt)
    # As of Triton 2.2.0, tl.clamp is not available yet
    # dt = tl.clamp(dt, dt_min, dt_max)
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where(
        (offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt,
        0.0)
    tl.store(dt_out_ptrs,
             dt,
             mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    dA = dt * A[:, None]
    dA_cs = tl.cumsum(dA, axis=1)
    tl.store(dA_cs_ptrs,
             dA_cs,
             mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))


@triton.autotune(
    configs=[
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 64
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 32
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32
            },
            num_stages=5,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 32,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32
            },
            num_stages=5,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32
            },
            num_stages=4,
            num_warps=2),
    ],
    key=['hdim', 'dstate', 'chunk_size'],
)
@triton.jit
def _chunk_state_fwd_kernel(
    # Pointers to matrices
    x_ptr,
    b_ptr,
    states_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    cu_chunk_seqlens_ptr,
    seq_idx_ptr,
    # Matrix dimensions
    hdim,
    dstate,
    chunk_size,
    batch,
    seqlen,
    nheads_ngroups_ratio,
    # Strides
    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_x_hdim,
    stride_b_batch,
    stride_b_seqlen,
    stride_b_head,
    stride_b_dstate,
    stride_states_batch,
    stride_states_chunk,
    stride_states_head,
    stride_states_hdim,
    stride_states_dstate,
    stride_dt_batch,
    stride_dt_chunk,
    stride_dt_head,
    stride_dt_csize,
    stride_dA_cs_batch,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dA_cs_csize,
    stride_seq_idx_batch,
    stride_seq_idx_seqlen,
    # Meta-parameters
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1).to(tl.int64)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
    chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)
    b_ptr += pid_b * stride_b_batch + chunk_seqlen_start * stride_b_seqlen + (
        pid_h // nheads_ngroups_ratio) * stride_b_head
    x_ptr += pid_b * stride_x_batch + chunk_seqlen_start * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_hdim +
                      offs_k[None, :] * stride_x_seqlen)
    b_ptrs = b_ptr + (offs_n[None, :] * stride_b_dstate +
                      offs_k[:, None] * stride_b_seqlen)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr +
                         (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize

    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(x_ptrs,
                    mask=(offs_m[:, None] < hdim) &
                    (offs_k[None, :] < chunk_size_limit - k),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=(offs_k[:, None] < chunk_size_limit - k) &
                    (offs_n[None, :] < dstate),
                    other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs,
                          mask=offs_k < chunk_size_limit - k,
                          other=0.0).to(tl.float32)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k,
                       other=0.0).to(tl.float32)
        scale = tl.exp(dA_cs_last - dA_cs_k) * dt_k
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
    states = acc.to(states_ptr.dtype.element_ty)

    states_ptr += pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (offs_m[:, None] * stride_states_hdim +
                                offs_n[None, :] * stride_states_dstate)
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 64
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 32
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32
            },
            num_stages=5,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 32,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32
            },
            num_stages=5,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32
            },
            num_stages=4,
            num_warps=2),
    ],
    key=['hdim', 'dstate', 'chunk_size'],
)
@triton.jit
def _chunk_state_varlen_kernel(
    # Pointers to matrices
    x_ptr,
    b_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    chunk_states_ptr,
    cu_seqlens_ptr,
    states_ptr,
    initstates_ptr,
    # Matrix dimensions
    hdim,
    dstate,
    chunk_size,
    seqlen,
    nheads_ngroups_ratio,
    # Strides
    stride_x_seqlen,
    stride_x_head,
    stride_x_hdim,
    stride_b_seqlen,
    stride_b_head,
    stride_b_dstate,
    stride_dt_chunk,
    stride_dt_head,
    stride_dt_csize,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dA_cs_csize,
    stride_chunk_states_chunk,
    stride_chunk_states_head,
    stride_chunk_states_hdim,
    stride_chunk_states_dstate,
    stride_states_batch,
    stride_states_head,
    stride_states_hdim,
    stride_states_dstate,
    stride_init_states_batch,
    stride_init_states_head,
    stride_init_states_hdim,
    stride_init_states_dstate,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HAS_INITSTATES: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    end_idx = tl.load(cu_seqlens_ptr + pid_b + 1)
    pid_c = (end_idx - 1) // chunk_size
    b_ptr += pid_c * chunk_size * stride_b_seqlen + (
        pid_h // nheads_ngroups_ratio) * stride_b_head
    x_ptr += pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    chunk_states_ptr += pid_c * stride_chunk_states_chunk + pid_h * stride_chunk_states_head

    if HAS_INITSTATES:
        # if there are init states provided, we differentiate between states (which
        # are boundary conditions at a chunk boundary) and initstates (which are boundary
        # conditions when a new example in a cont batch starts)
        initstates_ptr += pid_h * stride_init_states_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_hdim +
                      offs_k[None, :] * stride_x_seqlen)
    b_ptrs = b_ptr + (offs_n[None, :] * stride_b_dstate +
                      offs_k[:, None] * stride_b_seqlen)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (end_idx - pid_c * chunk_size - 1) *
                         stride_dA_cs_csize).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize

    chunk_size_limit = end_idx - pid_c * chunk_size
    start_idx = tl.load(cu_seqlens_ptr + pid_b)
    start_idx_cur = tl.maximum(start_idx - pid_c * chunk_size, 0)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(x_ptrs,
                    mask=(offs_m[:, None] < hdim) &
                    (offs_k[None, :] < chunk_size_limit - k) &
                    (offs_k[None, :] >= start_idx_cur - k),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=(offs_k[:, None] < chunk_size_limit - k) &
                    (offs_n[None, :] < dstate) &
                    (offs_k[:, None] >= start_idx_cur - k),
                    other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs,
                          mask=offs_k < chunk_size_limit - k,
                          other=0.0).to(tl.float32)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k,
                       other=0.0).to(tl.float32)
        scale = tl.where(
            (offs_k >= start_idx_cur - k) & (offs_k < chunk_size_limit - k),
            tl.exp(dA_cs_last - dA_cs_k) * dt_k, 0.0)
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    # If the sequence starts after the last chunk idx, we don't need to add the contribution from the last chunk
    # If HAS_INITSTATES==True need to consider two possibilities
    # - if start_idx < pid_c * chunk_size, then we need to take the past_states_ptrs
    # - if state_idx >= pid * chunk_size, then we need to insert initstates
    if ((start_idx < pid_c * chunk_size)  # first chunk
            or (HAS_INITSTATES)):

        dA_cs_boundary = 0.0  # default

        if not HAS_INITSTATES:
            past_states_ptrs = chunk_states_ptr + (
                offs_m[:, None] * stride_chunk_states_hdim +
                offs_n[None, :] * stride_chunk_states_dstate)
        else:

            # - this seems repetitive, buts its to help the compiler
            if start_idx < pid_c * chunk_size:
                past_states_ptrs = chunk_states_ptr + (
                    offs_m[:, None] * stride_chunk_states_hdim +
                    offs_n[None, :] * stride_chunk_states_dstate)
            else:
                past_states_ptrs = initstates_ptr + (
                    pid_b * stride_init_states_batch +
                    offs_m[:, None] * stride_init_states_hdim +
                    offs_n[None, :] * stride_init_states_dstate)

                # need to adjust the boundary
                if start_idx > pid_c * chunk_size:
                    dA_cs_boundary = tl.load(dA_cumsum_ptr +
                                             (start_idx - pid_c * chunk_size -
                                              1) * stride_dA_cs_csize).to(
                                                  tl.float32)

        past_states = tl.load(past_states_ptrs,
                              mask=(offs_m[:, None] < hdim) &
                              (offs_n[None, :] < dstate),
                              other=0.0).to(tl.float32)

        scale = tl.exp(dA_cs_last - dA_cs_boundary)
        acc += past_states * scale

    states = acc.to(states_ptr.dtype.element_ty)

    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (offs_m[:, None] * stride_states_hdim +
                                offs_n[None, :] * stride_states_dstate)
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)


def _chunk_cumsum_fwd(dt,
                      A,
                      chunk_size,
                      cu_chunk_seqlens,
                      dt_bias=None,
                      dt_softplus=False,
                      dt_limit=(0.0, float("inf"))):
    batch, seqlen, nheads = dt.shape
    assert A.shape == (nheads, )
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, )
    nchunks = cu_chunk_seqlens.shape[0] - 1
    dt_out = torch.empty(batch,
                         nheads,
                         nchunks,
                         chunk_size,
                         device=dt.device,
                         dtype=torch.float32)
    dA_cumsum = torch.empty(batch,
                            nheads,
                            nchunks,
                            chunk_size,
                            device=dt.device,
                            dtype=torch.float32)
    grid_chunk_cs = lambda META: (batch, nchunks,
                                  triton.cdiv(nheads, META['BLOCK_SIZE_H']))
    with torch.cuda.device(dt.device.index):
        _chunk_cumsum_fwd_kernel[grid_chunk_cs](
            dt,
            A,
            dt_bias,
            dt_out,
            dA_cumsum,
            cu_chunk_seqlens,
            batch,
            seqlen,
            nheads,
            chunk_size,
            dt_limit[0],
            dt_limit[1],
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            A.stride(0),
            dt_bias.stride(0) if dt_bias is not None else 0,
            dt_out.stride(0),
            dt_out.stride(2),
            dt_out.stride(1),
            dt_out.stride(3),
            dA_cumsum.stride(0),
            dA_cumsum.stride(2),
            dA_cumsum.stride(1),
            dA_cumsum.stride(3),
            dt_softplus,
            HAS_DT_BIAS=dt_bias is not None,
            BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
        )
    return dA_cumsum, dt_out


def _chunk_state_fwd(B,
                     x,
                     dt,
                     dA_cumsum,
                     cu_chunk_seqlens,
                     seq_idx=None,
                     states=None,
                     states_in_fp32=True):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if states is not None:
        assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    else:
        states_dtype = torch.float32 if states_in_fp32 else B.dtype
        states = torch.empty((batch, nchunks, nheads, headdim, dstate),
                             device=x.device,
                             dtype=states_dtype)
    grid = lambda META: (
        triton.cdiv(headdim, META['BLOCK_SIZE_M']) * triton.cdiv(
            dstate, META['BLOCK_SIZE_N']), batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_state_fwd_kernel[grid](
            x,
            B,
            states,
            dt,
            dA_cumsum,
            cu_chunk_seqlens,
            seq_idx,
            headdim,
            dstate,
            chunk_size,
            batch,
            seqlen,
            nheads // ngroups,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            B.stride(-1),
            states.stride(0),
            states.stride(1),
            states.stride(2),
            states.stride(3),
            states.stride(4),
            dt.stride(0),
            dt.stride(2),
            dt.stride(1),
            dt.stride(3),
            dA_cumsum.stride(0),
            dA_cumsum.stride(2),
            dA_cumsum.stride(1),
            dA_cumsum.stride(3),
            *((seq_idx.stride(0),
               seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_SEQ_IDX=seq_idx is not None,
        )
    return states


def chunk_state_varlen(B,
                       x,
                       dt,
                       dA_cumsum,
                       cu_seqlens,
                       chunk_states,
                       initial_states=None):
    total_seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    batch = cu_seqlens.shape[0] - 1
    cu_seqlens = cu_seqlens.contiguous()
    assert nheads % ngroups == 0
    assert B.shape == (total_seqlen, ngroups, dstate)
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert chunk_states.shape == (nchunks, nheads, headdim, dstate)

    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)

    states = torch.empty(batch,
                         nheads,
                         headdim,
                         dstate,
                         dtype=chunk_states.dtype,
                         device=chunk_states.device)
    grid = lambda META: (triton.cdiv(headdim, META['BLOCK_SIZE_M']) * triton.
                         cdiv(dstate, META['BLOCK_SIZE_N']), batch, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_state_varlen_kernel[grid](
            x,
            B,
            dt,
            dA_cumsum,
            chunk_states,
            cu_seqlens,
            states,
            initial_states,
            headdim,
            dstate,
            chunk_size,
            total_seqlen,
            nheads // ngroups,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            dt.stride(1),
            dt.stride(0),
            dt.stride(2),
            dA_cumsum.stride(1),
            dA_cumsum.stride(0),
            dA_cumsum.stride(2),
            chunk_states.stride(0),
            chunk_states.stride(1),
            chunk_states.stride(2),
            chunk_states.stride(3),
            states.stride(0),
            states.stride(1),
            states.stride(2),
            states.stride(3),
            *((initial_states.stride(0), initial_states.stride(1),
               initial_states.stride(2),
               initial_states.stride(3)) if initial_states is not None else
              (0, 0, 0, 0)),
            HAS_INITSTATES=initial_states is not None)
    return states

@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 4,
            'BLOCK_SIZE_N': 8
            },
            num_stages=8,
            num_warps=16),
        triton.Config({
            'BLOCK_SIZE_M': 8,
            'BLOCK_SIZE_N': 8
            },
            num_stages=8,
            num_warps=16),
        triton.Config({
            'BLOCK_SIZE_M': 16,
            'BLOCK_SIZE_N': 8
            },
            num_stages=8,
            num_warps=16),
        triton.Config({
            'BLOCK_SIZE_M': 32,
            'BLOCK_SIZE_N': 8
            },
            num_stages=8,
            num_warps=16),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 8
            },
            num_stages=8,
            num_warps=16),
        triton.Config({
            'BLOCK_SIZE_M': 4,
            'BLOCK_SIZE_N': 16
            },
            num_stages=8,
            num_warps=16),
        triton.Config({
            'BLOCK_SIZE_M': 8,
            'BLOCK_SIZE_N': 16
            },
            num_stages=8,
            num_warps=16),
        triton.Config({
            'BLOCK_SIZE_M': 16,
            'BLOCK_SIZE_N': 16
            },
            num_stages=8,
            num_warps=16),
        triton.Config({
            'BLOCK_SIZE_M': 32,
            'BLOCK_SIZE_N': 16
            },
            num_stages=8,
            num_warps=16),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 16
            },
            num_stages=8,
            num_warps=16),
        triton.Config({
            'BLOCK_SIZE_M': 8,
            'BLOCK_SIZE_N': 32
            },
            num_stages=8,
            num_warps=16),
        triton.Config({
            'BLOCK_SIZE_M': 16,
            'BLOCK_SIZE_N': 32
            },
            num_stages=8,
            num_warps=16),
        triton.Config({
            'BLOCK_SIZE_M': 32,
            'BLOCK_SIZE_N': 32
            },
            num_stages=8,
            num_warps=16),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 32
            },
            num_stages=8,
            num_warps=16),
        triton.Config({
            'BLOCK_SIZE_M': 4,
            'BLOCK_SIZE_N': 64
            },
            num_stages=8,
            num_warps=16),
        triton.Config({
            'BLOCK_SIZE_M': 8,
            'BLOCK_SIZE_N': 64
            },
            num_stages=8,
            num_warps=16),
        triton.Config({
            'BLOCK_SIZE_M': 16,
            'BLOCK_SIZE_N': 64
            },
            num_stages=8,
            num_warps=16),
        triton.Config({
            'BLOCK_SIZE_M': 32,
            'BLOCK_SIZE_N': 64
            },
            num_stages=8,
            num_warps=16),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64
            },
            num_stages=8,
            num_warps=16),
    ],
    key=['dstate', 'headdim'],
)
@triton.jit
def _state_cache_fwd_kernel(
    # Pointers to matrices
    states_ptr,
    cache_state_ptr,
    state_indices_tensor_ptr,
    n_blocks_to_fill_ptr,
    current_first_idx_ptr,
    last_computed_token_block_offset_ptr,
    last_chunk_ptr,
    # Matrix dimensions
    chunk_size: tl.constexpr,
    nheads: tl.constexpr,
    nheads_pw2: tl.constexpr,
    headdim: tl.constexpr,
    headdim_pw2: tl.constexpr,
    dstate: tl.constexpr,
    dstate_pw2: tl.constexpr,
    nseq: tl.constexpr,
    # Strides
    state_indices_stride: tl.constexpr,
    chunk_stride: tl.constexpr,
    state_chunk_stride: tl.constexpr,
    state_nheads_stride: tl.constexpr,
    state_headdim_stride: tl.constexpr,
    state_dstate_stride: tl.constexpr,
    cache_state_cacheline_stride: tl.constexpr,
    cache_state_nheads_stride: tl.constexpr,
    cache_state_headdim_stride: tl.constexpr,
    cache_state_dstate_stride: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ):
    
    # single-sequence id
    idx_seq = tl.program_id(0) % nseq
    idx_block_to_fill = tl.program_id(0) // nseq
    # idx_seq = tl.program_id(0)
    # idx_block_to_fill = tl.program_id(1)
    
    # index for block to fill. If larger than the number of blocks to fill for the current sequence, return
    n_blocks_to_fill = tl.load(n_blocks_to_fill_ptr + idx_seq)
    
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    # elements along the number of heads
    idx_nheads = tl.arange(0, nheads_pw2)
    
    # elements along the head dimension
    # idx_headdim = tl.arange(0, headdim_pw2)
    idx_headdim = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    
    # elements along the state dimension
    # idx_dstate = tl.arange(0, dstate_pw2)
    idx_dstate = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    idx_last_chunk = tl.load(last_chunk_ptr + idx_seq).to(tl.int64)
    if idx_seq > 0:
        idx_last_chunk_prev = tl.load(last_chunk_ptr + (idx_seq - 1)).to(tl.int64)
        idx_last_chunk_p1 = idx_last_chunk_prev + 1
    else:
        idx_last_chunk_prev = tl.full((), 0, dtype=tl.int64)
        idx_last_chunk_p1 = tl.full((), 0, dtype=tl.int64)
    
    state_offset = tl.full((), 0, dtype=tl.int64)
    idx_cache_state = tl.full((), 0, dtype=tl.int64)
    
    if idx_block_to_fill > n_blocks_to_fill:
        return
    else:
        # Get the current block index for the sequence
        idx_block_cache = tl.load(current_first_idx_ptr + idx_seq) + idx_block_to_fill
        # Get the index from the state_indices_tensor
        idx_cache_state = tl.load(state_indices_tensor_ptr +
                                  idx_seq * state_indices_stride + 
                                  idx_block_cache
                                 ).to(tl.int64)
        
        if idx_block_to_fill == n_blocks_to_fill:
            # Take the state offset directly from the 
            state_offset = idx_last_chunk
        else:            
            # Get the last_computed_token_block_offset for the sequence
            idx_last_computed_token_block_offset = tl.load(last_computed_token_block_offset_ptr + idx_seq).to(tl.int64)
            
            chunk_offset = idx_last_chunk_p1 + chunk_stride - 1 - idx_last_computed_token_block_offset // chunk_size
            state_offset = chunk_offset + idx_block_to_fill * chunk_stride
    
    state_at_seq_block_ptr = states_ptr + \
                            state_offset * state_chunk_stride + \
                            (idx_nheads * state_nheads_stride)[:, None, None] + \
                            (idx_headdim * state_headdim_stride)[None, :, None] + \
                            (idx_dstate * state_dstate_stride)[None, None, :]
    mask = (idx_nheads < nheads)[:, None, None] & \
           (idx_headdim < headdim)[None, :, None] & \
           (idx_dstate < dstate)[None, None, :]
    state_at_seq_block = tl.load(state_at_seq_block_ptr, mask, 0.0)
    
    current_cache_state_ptr = cache_state_ptr + \
                              idx_cache_state * cache_state_cacheline_stride + \
                              (idx_nheads * cache_state_nheads_stride)[:, None, None] + \
                              (idx_headdim * cache_state_headdim_stride)[None, :, None] + \
                              (idx_dstate * cache_state_dstate_stride)[None, None, :]
    
    tl.debug_barrier()  #  NOTE: use this due to bug in Triton compiler
    tl.store(current_cache_state_ptr, state_at_seq_block, mask)
    
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 2}),
        triton.Config({'BLOCK_SIZE_N': 4}),
        triton.Config({'BLOCK_SIZE_N': 8}),
        triton.Config({'BLOCK_SIZE_N': 16}),
        triton.Config({'BLOCK_SIZE_N': 32}),
        triton.Config({'BLOCK_SIZE_N': 64}),
    ],
    key=['nheads', 'dstate', 'headdim'],
)
@triton.jit
def _init_state_fwd_kernel(
    # Pointers to matrices
    ssm_state_ptr,
    init_states_ptr,
    state_indices_ptr,
    last_state_idx_ptr,
    has_initial_states_ptr,
    # Matrix dimensions
    nheads: tl.constexpr,
    nheads_pw2: tl.constexpr,
    headdim: tl.constexpr,
    headdim_pw2: tl.constexpr,
    dstate: tl.constexpr,
    # Strides
    state_indices_stride: tl.constexpr,
    state_chunk_stride: tl.constexpr,
    state_nheads_stride: tl.constexpr,
    state_headdim_stride: tl.constexpr,
    state_dstate_stride: tl.constexpr,
    cache_state_cacheline_stride: tl.constexpr,
    cache_state_nheads_stride: tl.constexpr,
    cache_state_headdim_stride: tl.constexpr,
    cache_state_dstate_stride: tl.constexpr,
    # Meta-parameters
    IS_CACHE_ENABLED: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ):
    
    # single-sequence id
    idx_seq = tl.program_id(0)
    
    pid_n = tl.program_id(1)
    
    # elements along the number of heads
    idx_nheads = tl.arange(0, nheads_pw2)
    
    # elements along the head dimension
    idx_headdim = tl.arange(0, headdim_pw2)
    
    # elements along the state dimension
    idx_dstate = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    has_initial_states = tl.load(has_initial_states_ptr + idx_seq).to(tl.int32)
    
    ssm_state_offset = tl.load(state_indices_ptr + idx_seq).to(tl.int64)
    if IS_CACHE_ENABLED:
        idx_last_chunk = tl.load(last_state_idx_ptr + idx_seq).to(tl.int64)
        ssm_state_offset = tl.load(state_indices_ptr +
                                  idx_seq * state_indices_stride + 
                                  idx_last_chunk
                                 ).to(tl.int64)
             
    state_at_seq_block_ptr = ssm_state_ptr + \
                             ssm_state_offset * state_chunk_stride + \
                             (idx_nheads * state_nheads_stride)[:, None, None] + \
                             (idx_headdim * state_headdim_stride)[None, :, None] + \
                             (idx_dstate * state_dstate_stride)[None, None, :]
    mask_load = (has_initial_states > 0) & \
                (idx_nheads < nheads)[:, None, None] & \
                (idx_headdim < headdim)[None, :, None] & \
                (idx_dstate < dstate)[None, None, :]
    state_at_seq_block = tl.load(state_at_seq_block_ptr, mask_load, 0.0)
    
    mask_store = (idx_nheads < nheads)[:, None, None] & \
                 (idx_headdim < headdim)[None, :, None] & \
                 (idx_dstate < dstate)[None, None, :]
    current_init_states_ptr = init_states_ptr + \
                              idx_seq * cache_state_cacheline_stride + \
                              (idx_nheads * cache_state_nheads_stride)[:, None, None] + \
                              (idx_headdim * cache_state_headdim_stride)[None, :, None] + \
                              (idx_dstate * cache_state_dstate_stride)[None, None, :]
    
    tl.debug_barrier()  #  NOTE: use this due to bug in Triton compiler
    tl.store(current_init_states_ptr, state_at_seq_block, mask_store)

def _state_cache_fwd(states=None,
                     cache_state=None,
                     cu_seqlens=None,
                     cu_chunk_seqlens=None,
                     state_indices_tensor=None,
                     chunk_stride=None,
                     n_blocks_to_fill_tensor=None,
                     current_first_idx_tensor=None,
                     last_computed_token_block_offset_tensor=None,
                     last_chunk=None,
                     chunk_size=None):
    
    batch, nchunks, nheads, headdim, dstate = states.shape
    nseq = cu_seqlens.shape[0] - 1 # Actually is number of sequences in the "batch"
    n_blocks_to_fill_max = max(n_blocks_to_fill_tensor).item()
    
    assert nchunks == (cu_chunk_seqlens.shape[0] - 1)
    
    # grid = lambda META: (nseq * (n_blocks_to_fill_max + 1), # The +1 is for the last state that is always stored
    #                      triton.cdiv(dstate, META['BLOCK_SIZE_N'])
    #                     )
    grid = lambda META: (nseq * (n_blocks_to_fill_max + 1), # The +1 is for the last state that is always stored
                         triton.cdiv(headdim, META['BLOCK_SIZE_M']),
                         triton.cdiv(dstate, META['BLOCK_SIZE_N']),
                        )
    # grid = lambda META: (nseq * (n_blocks_to_fill_max + 1), # The +1 is for the last state that is always stored
    #                     )
    
    with torch.cuda.device(states.device.index):
        _state_cache_fwd_kernel[grid](
            states,
            cache_state,
            state_indices_tensor,
            n_blocks_to_fill_tensor,
            current_first_idx_tensor,
            last_computed_token_block_offset_tensor,
            last_chunk,
            chunk_size,
            nheads,
            triton.next_power_of_2(nheads),
            headdim,
            triton.next_power_of_2(headdim),
            dstate,
            triton.next_power_of_2(dstate),
            nseq,
            state_indices_tensor.stride(0),
            chunk_stride,
            states.stride(1),
            states.stride(2),
            states.stride(3),
            states.stride(4),
            cache_state.stride(0),
            cache_state.stride(1),
            cache_state.stride(2),
            cache_state.stride(3),
        )
        
def _init_state_fwd(ssm_state=None,
                    init_states=None,
                    cu_seqlens=None,
                    state_indices_tensor=None,
                    last_state_idx_tensor=None,
                    has_initial_states_tensor=None,
                    ):
    
    _, nheads, headdim, dstate = ssm_state.shape
    nseq = cu_seqlens.shape[0] - 1 # Actually is number of sequences in the "batch"
    
    grid = lambda META: (nseq, # The +1 is for the last state that is always stored
                         triton.cdiv(dstate, META['BLOCK_SIZE_N'])
    )
    
    with torch.cuda.device(init_states.device.index):
        _init_state_fwd_kernel[grid](
            ssm_state,
            init_states,
            state_indices_tensor,
            last_state_idx_tensor,
            has_initial_states_tensor,
            nheads,
            triton.next_power_of_2(nheads),
            headdim,
            triton.next_power_of_2(headdim),
            dstate,
            state_indices_tensor.stride(0),
            ssm_state.stride(0),
            ssm_state.stride(1),
            ssm_state.stride(2),
            ssm_state.stride(3),
            init_states.stride(0),
            init_states.stride(1),
            init_states.stride(2),
            init_states.stride(3),
            IS_CACHE_ENABLED=last_state_idx_tensor is not None,
        )