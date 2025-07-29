import math
import torch
import torch.nn.functional as F
import pytest

from einops import rearrange, repeat

from mamba_ssm.ops.triton.selective_state_update import selective_state_update as ssu_mamba
from mamba_ssm.ops.triton.selective_state_update import selective_state_update_ref
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined as ssd_mamba
from mamba_ssm.ops.triton.ssd_combined import ssd_chunk_scan_combined_ref as ssd_mamba_ref

from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update as ssu_vllm
from vllm.model_executor.layers.mamba.ops.ssd_combined import mamba_chunk_scan_combined as ssd_vllm

def test_selective_state_update_with_heads(dim, dstate, ngroups, has_z, tie_hdim, itype):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (5e-3, 3e-2)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 1e-1
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    headdim = 64
    nheads = dim // headdim
    state = torch.randn(batch_size, nheads, headdim, dstate, dtype=itype, device=device)
    x = torch.randn(batch_size, nheads, headdim, device=device, dtype=itype)
    if not tie_hdim:
        dt = torch.randn(batch_size, nheads, headdim, device=device, dtype=itype)
        dt_bias = torch.rand(nheads, headdim, device=device) - 4.0
        A = -torch.rand(nheads, headdim, dstate, device=device) - 1.0
        D = torch.randn(nheads, headdim, device=device)
    else:
        dt = repeat(torch.randn(batch_size, nheads, device=device, dtype=itype), "b h -> b h p", p=headdim)
        dt_bias = repeat(torch.rand(nheads, device=device) - 4.0, "h -> h p", p=headdim)
        A = repeat(-torch.rand(nheads, device=device) - 1.0, "h -> h p n", p=headdim, n=dstate)
        D = repeat(torch.randn(nheads, device=device), "h -> h p", p=headdim)
    B = torch.randn(batch_size, ngroups, dstate, device=device)
    C = torch.randn(batch_size, ngroups, dstate, device=device)
    if has_z:
        z = torch.randn_like(x)
    else:
        z = None
    state_ref = state.detach().clone()
    state_og = state.detach().clone()
    out = ssu_mamba(state, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)
    out_vllm = ssu_vllm(state_og, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)
    out_ref, state_ref_last = selective_state_update_ref(state_ref, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)
    # out_ref, state_ref_last = selective_state_update_ref(state_ref, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)

    print(f"Output max diff (mamba - vllm): {(out - out_vllm).abs().max().item()}")
    print(f"Output mean diff (mamba - vllm): {(out - out_vllm).abs().mean().item()}")
    print(f"Output max diff (ref - vllm): {(out_ref - out_vllm).abs().max().item()}")
    print(f"Output mean diff (ref - vllm): {(out_ref - out_vllm).abs().mean().item()}")
    assert torch.allclose(state, state_ref, rtol=rtol, atol=atol)
    assert torch.allclose(out, out_vllm, rtol=rtol, atol=atol)
    
# Test for mamba2 state update without conv-1D
def test_chunk_scan_combined():
    torch.manual_seed(42)
    itype = torch.float16
    
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (5e-3, 3e-2)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 1e-1

    ## Dimensions
    # Denoted (B, T, Q, D, P) in the paper
    # batch, seqlen, chunk_size, dim, headdim = 1, 16, 4, 8, 4
    batch, seqlen, chunk_size, dim, headdim = 3, 16, 4, 8, 4
    nheads = dim // headdim  # (H) in the paper
    ngroups = 1 # (G) in the paper
    dstate = 4  # (N) in the paper
    dtype = torch.float32
    device = "cuda"
    skip_chunks = 1
    n_chunks = int(seqlen / chunk_size)
    n_chunks_split = n_chunks - skip_chunks

    if batch > 1:
        print('WARNING: BS > 1 is currently being tested and may not work as expected!')

    x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
    dt = F.softplus(torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device) - 4).requires_grad_()
    A = (-torch.exp(torch.rand(nheads, dtype=torch.float32, device=device))).requires_grad_()
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
    D = torch.randn(nheads, dtype=dtype, device=device)

    def split_off_prefilled_parts(x, dt, A, B, C, D, seq_idx, skip_chunks = 0):
        return x[:, chunk_size*skip_chunks:, :],\
               dt[:, chunk_size*skip_chunks:, :],\
               A,\
               B[:, chunk_size*skip_chunks:, :],\
               C[:, chunk_size*skip_chunks:, :],\
               D,\
               seq_idx[:, chunk_size*skip_chunks:]

    def to_vllm_shapes(x, dt, B, C, seq_idx, states):
        return torch.concatenate(torch.split(x, 1), 1),\
               torch.concatenate(torch.split(dt, 1), 1),\
               torch.concatenate(torch.split(B, 1), 1),\
               torch.concatenate(torch.split(C, 1), 1),\
               torch.concatenate(torch.split(seq_idx, 1), 1),\
               torch.concatenate(torch.split(states, 1), 1)
    
    def from_vllm_shapes(out_vllm):
        return torch.concatenate(torch.split(out_vllm, (n_chunks - skip_chunks) * chunk_size, 1), 0)
               
               
    # This variable indicates for each batch to which sequence a token belongs to. I.e., for a single batch and a single sequence this would be [0, 0, 0, 0, ...]
    seq_idx = torch.tensor([[idx] * seqlen for idx in range(batch)]).to(device=device, dtype=torch.int32)
    # This variable indicates for a flattened tensor of tokens, what the chunk sizes are. They may be the same (somewhat default case), but they may also be different, i.e., across batch boundaries
    chunk_indices = torch.tensor([idx for idx in range(batch * n_chunks_split)]).to(device=device, dtype=torch.int32)
    # This variable indicates where the individual chunks start at
    chunk_offset = (torch.zeros(batch * (n_chunks - skip_chunks), device=device, dtype=torch.int32))

    # Split off some chunks
    x_split, dt_split, A_split, B_split, C_split, D_split, seq_idx_split = split_off_prefilled_parts(x, dt, A, B, C, D, seq_idx, skip_chunks = skip_chunks)

    # Compute outputs and states from the PyTorch reference implementation
    out_ref, states_passed_ref, states_ref = ssd_mamba_ref(x, dt, A, B, C, chunk_size, D=None)
    out_ref2, states_passed_ref2, states_ref2 = ssd_mamba_ref(x_split, dt_split, A, B_split, C_split, chunk_size, D=None)
    
    # Compute the output from the mamba version
    out = ssd_mamba(x, dt, A, B, C, chunk_size, D=None)
    
    # Utilization of just the state, leads to wrong results here
    out_split_state_passed = ssd_mamba(x_split, dt_split, A, B_split, C_split, chunk_size, D=None, initial_states=states_passed_ref[:, skip_chunks, :])
    out_split_no_init = ssd_mamba(x_split, dt_split, A, B_split, C_split, chunk_size, D=None)
    assert torch.allclose(out_ref, out, rtol=rtol, atol=atol), "Output between naive implementation and vLLM differed!"
    assert torch.allclose(out_ref2, out_split_no_init, rtol=rtol, atol=atol), "Output between naive implementation and vLLM differed!"
    assert not torch.allclose(out[:, skip_chunks*chunk_size:, :], out_split_no_init, rtol=rtol, atol=atol), "Prefix computing WITHOUT initial states are identical. This is unexpected"
    assert torch.allclose(out[:, skip_chunks*chunk_size:, :], out_split_state_passed, rtol=rtol, atol=atol), "Prefix computing WITH initial states failed with the mamba implementation failed!"
    
    # Compute the output from the vllm implementation
    out_vllm, states_passed_vllm = ssd_vllm(x, dt, A, B, C, chunk_size, D=None)
    assert torch.allclose(out, out_vllm, rtol=rtol, atol=atol), "vLLM and mamba results differ! This is unexpected!"
    assert torch.allclose(states_passed_ref, states_passed_vllm, rtol=rtol, atol=atol), "vLLM and mamba results differ! This is unexpected!"
    if batch > 1:
        x_split_vllm, dt_split_vllm, B_split_vllm, C_split_vllm, seq_idx_split_vllm, states_passed_ref_vllm = to_vllm_shapes(x_split, dt_split, B_split, C_split, seq_idx_split, states_passed_ref[:, skip_chunks, :])
        # x_split_vllm, seq_idx_split_vllm = create_vllm_input(x_split, dt_split, A, B_split, C_split, D=None, seq_idx=seq_idx_split)
        # cu_seqlens_vllm = torch.tensor([0, (n_chunks - skip_chunks) * chunk_size, 2 * ((n_chunks - skip_chunks) * chunk_size)], dtype=torch.int32, device=device)
        cu_seqlens_vllm = torch.tensor([el * (n_chunks - skip_chunks) * chunk_size for el in range(batch + 1, )], dtype=torch.int32, device=device)
        out_vllm_split_state_passed, _ = ssd_vllm(x_split_vllm,
                                               dt_split_vllm,
                                               A,
                                               B_split_vllm,
                                               C_split_vllm,
                                               chunk_size,
                                               D=None,
                                               initial_states=states_passed_ref[:, skip_chunks, :],
                                               chunk_indices=chunk_indices,
                                               chunk_offsets=chunk_offset,
                                               seq_idx=seq_idx_split_vllm,
                                               cu_seqlens=cu_seqlens_vllm,
                                               return_varlen_states=True)
        out_vllm_split_state_passed_mamba = from_vllm_shapes(out_vllm_split_state_passed)
        assert torch.allclose(out[:, skip_chunks*chunk_size:, :], out_vllm_split_state_passed_mamba, rtol=rtol, atol=atol), "Prefix computing WITH initial states failed with the vLLM implementation failed!"
    
    print('All checks passed!')
    
# TODO: Test for entire mamba2 including conv-1D
    
# test_selective_state_update_with_heads(dim=2048, dstate=16, ngroups=1, has_z=True, tie_hdim=True, itype=torch.float16)
test_chunk_scan_combined()