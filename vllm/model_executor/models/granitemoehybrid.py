# SPDX-License-Identifier: Apache-2.0
"""Inference-only Bamba model."""
# Added by the IBM Team, 2024
from typing import Iterable, Optional, Set, Tuple

import torch
from torch import nn
from transformers import GraniteMoeHybridConfig
from transformers import BambaConfig

from vllm.attention.layer import Attention
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear, MergedColumnParallelLinear,
                                               QKVParallelLinear, ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_mixer2 import (
    MambaMixer2, extra_groups_for_head_shards)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.mamba_cache import (MambaCacheManager,
                                                    MambaCacheParams)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.utils import LayerBlockType

from .interfaces import (HasInnerState, IsHybrid, SupportsLoRA, SupportsPP,
                         SupportsQuant, SupportsV0Only)
from .utils import (AutoWeightsLoader, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)


from .granitemoe import GraniteMoeAttention, GraniteMoeMoE
from .granitemoeshared import GraniteMoeSharedMLP


class BambaMLP(nn.Module):

    def __init__(
        self,
        config: GraniteMoeHybridConfig,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=bias,
            quant_config=quant_config,
        )
        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class GraniteMoeHybridMixerDecoderLayer(nn.Module):

    def __init__(self,
                 config: GraniteMoeHybridConfig,
                 layer_idx: int,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "") -> None:
        super().__init__()
        self.config = config

        #From GraniteMoeSharedDecoderLayer:
        self.hidden_size = config.hidden_size
        
        # Requires transformers > 4.32.0
        # rope_theta = getattr(config, "rope_theta", 10000)
        # self.self_attn = GraniteMoeAttention(
        #     hidden_size=self.hidden_size,
        #     num_heads=config.num_attention_heads,
        #     max_position=config.max_position_embeddings,
        #     num_kv_heads=config.num_key_value_heads,
        #     rope_theta=rope_theta,
        #     cache_config=cache_config,
        #     quant_config=quant_config,
        #     prefix=f"{prefix}.self_attn",
        #     attention_multiplier=config.attention_multiplier)
        
        self.self_attn_type = "mamba2" #TODO: For completeness, not sure if needed ;; Below name change was required to ensure the same naming for checkpoint loading
        self.self_attn = MambaMixer2(hidden_size= config.hidden_size,
                                ssm_state_size = config.mamba_d_state,
                                conv_kernel_size = config.mamba_d_conv,
                                intermediate_size = config.mamba_expand *\
                                                    config.hidden_size,
                                use_conv_bias = config.mamba_conv_bias,
                                use_bias = config.mamba_proj_bias,
                                n_groups=config.mamba_n_groups,
                                num_heads=config.mamba_n_heads,
                                head_dim=config.mamba_d_head,
                                rms_norm_eps=config.rms_norm_eps,
                                activation=config.hidden_act,
                                chunk_size=config.mamba_chunk_size,
                                quant_config=quant_config)

        self.block_sparse_moe = GraniteMoeMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.block_sparse_moe")
        self.shared_mlp = None if \
            getattr(config, 'shared_intermediate_size', 0) == 0 \
            else GraniteMoeSharedMLP(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_mlp"
            )
        # Done: Cast GraniteMoeSharedConfig(config) , pass prefix ?
        # self.shared_mlp = None if config.shared_intermediate_size == 0 else GraniteMoeSharedMLP(config, quant_config=quant_config)

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

        self.residual_multiplier = config.residual_multiplier


        # #self.feed_forward = BambaMLP(config, quant_config=quant_config)
        # self.input_layernorm = RMSNorm(config.hidden_size,
        #                                eps=config.rms_norm_eps)
        # self.pre_ff_layernorm = RMSNorm(config.hidden_size,
        #                                 eps=config.rms_norm_eps)
        

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        mamba_cache_params: MambaCacheParams,
        sequence_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attn(hidden_states, mamba_cache_params,
                                   sequence_idx)
        # # Fully Connected
        # hidden_states, residual = self.pre_ff_layernorm(
        #     hidden_states, residual)
        # hidden_states = self.feed_forward(hidden_states)
        # return hidden_states, residual

        hidden_states = residual + hidden_states * self.residual_multiplier
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.shared_mlp is None:
            hidden_states = self.block_sparse_moe(hidden_states)
        else:
            # create a copy since block_sparse_moe modifies in-place
            moe_hidden_states = hidden_states.clone()
            moe_hidden_states = self.block_sparse_moe(moe_hidden_states)
            hidden_states = moe_hidden_states + self.shared_mlp(hidden_states)
            del moe_hidden_states
        hidden_states = residual + hidden_states * self.residual_multiplier

        return hidden_states, residual
     
class DeepseekV2MLAAttention(nn.Module):
    """
    Main reference: DeepseekV2 paper, and FlashInfer Implementation
    (https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).
    
    For more info see MLACommonImpl in: vllm/attention/backends/mla/utils.py
    """
    def __init__(
        self,
        config: GraniteMoeHybridConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        rms_norm_eps = config.rms_norm_eps
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        qk_nope_head_dim = config.mla_
        
        # TODO: Granite 4.0 config:
        # query_compression_size: 384
        # key_value_compression_size: 96
        # Issues:
        #  1) qk - same dimension, but we have different
        #  2) uses PE, our model is noPE
        
        # qk_nope_head_dim: int,
        # qk_rope_head_dim: int,
        # v_head_dim: int,
        # q_lora_rank: Optional[int],
        # kv_lora_rank: int,
        # #rope_theta: float = 10000,
        # #rope_scaling: Optional[Dict[str, Any]] = None,
        # max_position_embeddings: int = 8192,
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        #self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(self.hidden_size,
                                             self.q_lora_rank,
                                             bias=False,
                                             quant_config=quant_config,
                                             prefix=f"{prefix}.q_a_proj")
            self.q_a_layernorm = RMSNorm(self.q_lora_rank,
                                         eps=rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(q_lora_rank,
                                                 self.num_heads *
                                                 self.qk_head_dim,
                                                 bias=False,
                                                 quant_config=quant_config,
                                                 prefix=f"{prefix}.q_b_proj")
        else:
            self.q_proj = ColumnParallelLinear(self.hidden_size,
                                               self.num_heads *
                                               self.qk_head_dim,
                                               bias=False,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.q_proj")

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa")
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
                                      eps=rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj")
        self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim,
                                        self.hidden_size,
                                        bias=False,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj")

        # if rope_scaling:
        #     rope_scaling["rope_type"] = 'deepseek_yarn'
        # self.rotary_emb = get_rope(qk_rope_head_dim,
        #                            rotary_dim=qk_rope_head_dim,
        #                            max_position=max_position_embeddings,
        #                            base=rope_theta,
        #                            rope_scaling=rope_scaling,
        #                            is_neox_style=False)
        # if rope_scaling:
        #     mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
        #     scaling_factor = rope_scaling["factor"]
        #     mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
        #     self.scaling = self.scaling * mscale * mscale

        # In the MLA backend, kv_cache includes both k_c and
        # pe (i.e. decoupled position embeddings). In particular,
        # the concat_and_cache_mla op requires
        #     k_c.size(1) + k_pe.size(1) == kv_cache.size(2)
        # i.e.
        #     kv_lora_rank + qk_rope_head_dim == head_size
        self.mla_attn = Attention(
            num_heads=self.num_local_heads,
            head_size=self.kv_lora_rank + self.qk_rope_head_dim,
            scale=self.scaling,
            num_kv_heads=1,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_mla=True,
            # MLA Args
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            rotary_emb=self.rotary_emb,
            q_proj=self.q_proj if self.q_lora_rank is None else self.q_b_proj,
            kv_b_proj=self.kv_b_proj,
            o_proj=self.o_proj,
        )

        self.prefix = prefix
        self.debug_layer_idx = int(self.prefix.split(".")[-2])

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            ckq = self.q_a_proj(hidden_states)[0]
            hidden_states_or_q_c = self.q_a_layernorm(ckq)
        else:
            hidden_states_or_q_c = hidden_states
        kv_c, k_pe = self.kv_a_proj_with_mqa(hidden_states)[0].split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c.contiguous())
        return self.mla_attn(hidden_states_or_q_c,
                             kv_c_normed,
                             k_pe,
                             output_shape=hidden_states.shape)


class GraniteMoeHybridAttention(nn.Module):

    def __init__(
        self,
        config: GraniteMoeHybridConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Copied from HF implementation:
        self.causal = True
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # TO DO add this bias later
        #self.add_bias = config.add_bias
        self.query_compression_size = config.mla_query_comp_size
        self.key_value_compression_size = config.mla_key_value_comp_size

        self.head_dim = self.hidden_size // self.num_heads 
        # self.position_embedding_type = config.position_embedding_type
        self.attention_multiplier = config.attention_multiplier
        ##self.layer_idx = layer_idx

        # TO DO- will bias be a flag in config?
        # self.position_embedding_type == "rope": not implemented so went to else

        #self.c_attn_down_projection = nn.Linear(self.hidden_size, self.query_compression_size + 2 * self.key_value_compression_size, bias=False)
        # TODO: Tensor optimizations
        self.c_attn_down_projection = ReplicatedLinear(self.hidden_size,
                                          self.query_compression_size + 2 * self.key_value_compression_size,
                                          bias=False,
                                          quant_config=quant_config)    
        
        #self.query_up_projection = nn.Linear(self.query_compression_size, self.hidden_size, bias=False)
        self.query_up_projection = ReplicatedLinear(self.query_compression_size,
                                          self.hidden_size,
                                          bias=False,
                                          quant_config=quant_config)
        
        #self.key_up_projection = nn.Linear(self.key_value_compression_size, self.hidden_size, bias=False)
        self.key_up_projection = ReplicatedLinear(self.key_value_compression_size,
                                          self.hidden_size,
                                          bias=False,
                                          quant_config=quant_config)
        
        #self.value_up_projection = nn.Linear(self.key_value_compression_size, self.hidden_size, bias=False)
        self.value_up_projection = ReplicatedLinear(self.key_value_compression_size,
                                          self.hidden_size,
                                          bias=False,
                                          quant_config=quant_config)
        
        #self.c_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.c_proj = ReplicatedLinear(self.hidden_size,
                                          self.hidden_size,
                                          bias=False,
                                          quant_config=quant_config)

        # TO confirm the softmax_dropout and dropout variable names
        self.softmax_dropout_p = config.mla_softmax_dropout
        self.softmax_dropout = nn.Identity() if config.mla_softmax_dropout == 0 else nn.Dropout(config.mla_softmax_dropout)
        self.dropout = nn.Identity() if config.mla_dropout == 0 else nn.Dropout(config.mla_dropout)

        self.scaling = self.attention_multiplier

        # self.hidden_size = hidden_size
        # tp_size = get_tensor_model_parallel_world_size()
        # self.total_num_heads = num_heads
        # assert self.total_num_heads % tp_size == 0
        # self.num_heads = self.total_num_heads // tp_size
        # self.total_num_kv_heads = num_kv_heads
        # if self.total_num_kv_heads >= tp_size:
        #     # Number of KV heads is greater than TP size, so we partition
        #     # the KV heads across multiple tensor parallel GPUs.
        #     assert self.total_num_kv_heads % tp_size == 0
        # else:
        #     # Number of KV heads is less than TP size, so we replicate
        #     # the KV heads across multiple tensor parallel GPUs.
        #     assert tp_size % self.total_num_kv_heads == 0
        # self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # self.head_dim = hidden_size // self.total_num_heads
        # self.q_size = self.num_heads * self.head_dim
        # self.kv_size = self.num_kv_heads * self.head_dim
        # self.scaling = (attention_multiplier if attention_multiplier
        #                 is not None else self.head_dim**-1)
        # self.rope_theta = rope_theta

        # self.qkv_proj = QKVParallelLinear(
        #     hidden_size,
        #     self.head_dim,
        #     self.total_num_heads,
        #     self.total_num_kv_heads,
        #     bias=False,
        #     quant_config=quant_config,
        #     prefix=f"{prefix}.qkv_proj",
        # )
        # self.o_proj = RowParallelLinear(
        #     self.total_num_heads * self.head_dim,
        #     hidden_size,
        #     bias=False,
        #     quant_config=quant_config,
        #     prefix=f"{prefix}.o_proj",
        # )
        # self.rotary_emb = get_rope(
        #     self.head_dim,
        #     rotary_dim=self.head_dim,
        #     max_position=max_position,
        #     base=int(self.rope_theta),
        #     is_neox_style=True,
        # )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              #num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # qkv, _ = self.qkv_proj(hidden_states)
        # q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # q, k = self.rotary_emb(positions, q, k)
        # attn_output = self.attn(q, k, v)
        # output, _ = self.o_proj(attn_output)
        # return output

        hidden_states = self.c_attn_down_projection(hidden_states)[0] #TODO: Why do we get a tuple here?
        query, key, value = hidden_states.split(
                (self.query_compression_size, self.key_value_compression_size, self.key_value_compression_size), dim=-1
            )
        query = self.query_up_projection(query)[0]
        key = self.key_up_projection(key)[0]
        value = self.value_up_projection(value)[0]
        # reference 
        # https://github.com/IBM/dolomite-engine/blob/main/dolomite_engine/hf_models/modeling_utils/sequence_mixer_blocks/multihead_latent_attention.py#L177
        
        # STW: no need to reshape, as vLLM's attn already knows the self.num_heads
        # batch_size, query_length = query.shape[:-1]
        # key_length = key.shape[1]
        # query = query.view(batch_size, query_length, self.num_heads, -1).transpose(1, 2)
        # key = key.view(batch_size, key_length, self.num_heads, -1).transpose(1, 2)
        # value = value.view(batch_size, key_length, self.num_heads, -1).transpose(1, 2)

        hidden_states = self.attn( #F.scaled_dot_product_attention(
            query,
            key,
            value,
            # STW: Settings passed to the constructor instead:
            # attn_mask=attention_mask,
            # dropout_p=self.softmax_dropout_p if self.training else 0,
            # is_causal=self.causal if attention_mask is None else False,
            # scale=self.attention_multiplier,
        )
        del query, key, value

        # batch_size = hidden_states.shape[0]
        # hidden_states = hidden_states.transpose(1, 2)
        # hidden_states = hidden_states.reshape(batch_size, -1, self.num_heads * self.head_dim)

        hidden_states = self.c_proj(hidden_states)[0]
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class GraniteMoeHybridDecoderLayer(nn.Module):

    def __init__(
        self,
        config: GraniteMoeHybridConfig,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = GraniteMoeHybridAttention(
            config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn")
        self.block_sparse_moe = GraniteMoeMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.block_sparse_moe")
        self.shared_mlp = None if \
            getattr(config, 'shared_intermediate_size', 0) == 0 \
            else GraniteMoeSharedMLP(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_mlp"
            )

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

        self.residual_multiplier = config.residual_multiplier

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        # Parameters below are used by Mamba2 mixer
        residual: Optional[torch.Tensor],
        mamba_cache_params: MambaCacheParams,
        sequence_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states = residual + hidden_states * self.residual_multiplier
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.shared_mlp is None:
            hidden_states = self.block_sparse_moe(hidden_states)
        else:
            # create a copy since block_sparse_moe modifies in-place
            moe_hidden_states = hidden_states.clone()
            moe_hidden_states = self.block_sparse_moe(moe_hidden_states)
            hidden_states = moe_hidden_states + self.shared_mlp(hidden_states)
            del moe_hidden_states
        hidden_states = residual + hidden_states * self.residual_multiplier

        return hidden_states, residual

class BambaAttentionDecoderLayer(nn.Module):

    def __init__(
        self,
        config: GraniteMoeHybridConfig,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        if hasattr(config, "partial_rotary_factor"):
            rotary_dim = self.head_dim * config.partial_rotary_factor
        elif hasattr(config, "attn_rotary_emb"):
            rotary_dim = config.attn_rotary_emb  # for backward compatibility
        else:
            rotary_dim = self.head_dim  # default

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position_embeddings,
            rope_scaling=rope_scaling,
            base=rope_theta,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),  # see impl of get_rope
        )

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim,
                                        config.hidden_size,
                                        bias=False,
                                        quant_config=quant_config)

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )

        self.feed_forward = BambaMLP(config, quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size,
                                        eps=config.rms_norm_eps)

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attention(
            positions=positions,
            hidden_states=hidden_states,
        )
        # Fully Connected
        hidden_states, residual = self.pre_ff_layernorm(
            hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    # "attention": BambaAttentionDecoderLayer,
    # "mamba": BambaMixerDecoderLayer,
    # "multihead_latent_attention": GraniteMoeHybridDecoderLayer,
    # "mamba2": BambaMixerDecoderLayer
    "multihead_latent_attention": GraniteMoeHybridDecoderLayer,
    "mamba2": GraniteMoeHybridMixerDecoderLayer,
}


class GraniteMoeHybridModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        lora_vocab = ((lora_config.lora_extra_vocab_size *
                       (lora_config.max_loras or 1)) if lora_config else 0)
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            layer_class = ALL_DECODER_LAYER_TYPES[
                config.layer_types[layer_idx]]
            return layer_class(
                config,
                layer_idx,
                cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers")
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        self.norm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mamba_cache_params: MambaCacheParams,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # pass a sequence index tensor, that is required for
        # proper continuous batching computation including
        # chunked prefill
        seq_idx = None
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata.num_prefills > 0:
            seq_idx = torch.zeros_like(input_ids, dtype=torch.int32)
            for i, (srt, end) in enumerate(
                    zip(
                        attn_metadata.query_start_loc,
                        attn_metadata.query_start_loc[1:],
                    )):
                seq_idx[srt:end] = i
            seq_idx.unsqueeze_(0)

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        residual = None
        num_attn = 0
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if isinstance(layer, GraniteMoeHybridDecoderLayer):
                num_attn += 1

            layer_mamba_cache_params = None
            if isinstance(layer, GraniteMoeHybridMixerDecoderLayer):
                layer_mamba_cache_params = mamba_cache_params.at_layer_idx(
                    i - num_attn)

            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                mamba_cache_params=layer_mamba_cache_params,
                sequence_idx=seq_idx,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        
        #loaded_params.add(p)

        def _load(n, p):
            param = params_dict[n]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, p)
            loaded_params.add(n)
        
        def _load_expert(n, p, name, shard_id, expert_id):
            param = params_dict[n]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, p, name, shard_id=shard_id, expert_id=expert_id)
            loaded_params.add(n)

        from ..layers.fused_moe import FusedMoE
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts)

        for n, p in weights:
            #print('HF', n)

            if "A_log" in n:
                n = n.replace("A_log", "A")

            #expert: p.size() # (62, 1024, 1536)
            # Logic from: https://github.com/vllm-project/vllm/blob/f49e5aff11c986ed4d45202b1716c5d74786efa9/vllm/model_executor/models/granitemoeshared.py#L215
            #<expert logic>
            # if n.endswith('.block_sparse_moe.input_linear.weight'):
            #     for e in range(p.size(0)):
            #         w1_name = n.replace(
            #             '.block_sparse_moe.input_linear.weight',
            #             f".block_sparse_moe.experts.{e}.w1.weight")
            #         w3_name = n.replace(
            #             '.block_sparse_moe.input_linear.weight',
            #             f".block_sparse_moe.experts.{e}.w3.weight")
            #         w1_param, w3_param = p[e].chunk(2, dim=0)
            #         _load(w1_name, w1_param)
            #         _load(w3_name, w3_param)                    
            # elif n.endswith('.block_sparse_moe.output_linear.weight'):
            #     for e in range(p.size(0)):
            #         w2_name = n.replace(
            #             '.block_sparse_moe.output_linear.weight',
            #             f".block_sparse_moe.experts.{e}.w2.weight")
            #         w2_param = p[e]
            #         _load(w2_name, w2_param)
            # elif n.endswith('.block_sparse_moe.router.layer.weight'):
            #     gate_name = n.replace('.block_sparse_moe.router.layer.weight',
            #                           ".block_sparse_moe.gate.weight")
            #     _load(gate_name, p)
            #expert: p.size() # (62, 1024, 1536)
            # Logic from: https://github.com/vllm-project/vllm/blob/f49e5aff11c986ed4d45202b1716c5d74786efa9/vllm/model_executor/models/granitemoeshared.py#L215
            #<expert logic>  Mapping different parameters layout: from HF (input_linear, output_linear, router) to vLLM (experts_w13({e}.w1, {e}.w2), experts_w3({e}.w3), gate)
            if n.endswith('.block_sparse_moe.input_linear.weight'):
                for e in range(p.size(0)):
                    w1_name = n.replace(
                        '.block_sparse_moe.input_linear.weight',
                        f".block_sparse_moe.experts.{e}.w1.weight")
                    w3_name = n.replace(
                        '.block_sparse_moe.input_linear.weight',
                        f".block_sparse_moe.experts.{e}.w3.weight")
                    w1_param, w3_param = p[e].chunk(2, dim=0)
                    _load_expert(n.replace('.input_linear.','.experts.w13_'), w1_param, w1_name, shard_id='w1', expert_id=e)
                    _load_expert(n.replace('.input_linear.','.experts.w13_'), w3_param, w3_name, shard_id='w3', expert_id=e)
            elif n.endswith('.block_sparse_moe.output_linear.weight'):
                for e in range(p.size(0)):
                    w2_name = n.replace(
                        '.block_sparse_moe.output_linear.weight',
                        f".block_sparse_moe.experts.{e}.w2.weight")
                    w2_param = p[e] 
                                #vLLM module address, checkpoint_data, vLLM submodule address (FusedMOE)
                    _load_expert(n.replace('.output_linear.', '.experts.w2_'), w2_param, w2_name, shard_id='w2', expert_id=e)            
            elif n.endswith('.block_sparse_moe.router.layer.weight'):
                gate_name = n.replace('.block_sparse_moe.router.layer.weight',
                                      ".block_sparse_moe.gate.weight")
                _load(gate_name, p)
            # </expert logic>
            else:
                _load(n,p)

            # if "rotary_emb.inv_freq" in name:
            #     continue

            # if "A_log" in name:
            #     name = name.replace("A_log", "A")

            # if ".self_attn." in name:
            #     name = name.replace(".self_attn", "")

            # for param_name, weight_name, shard_id in stacked_params_mapping:
            #     if weight_name not in name:
            #         continue

            #     name = name.replace(weight_name, param_name)
            #     # Skip loading extra bias for GPTQ models.
            #     if name.endswith(".bias") and name not in params_dict:
            #         continue
            #     # Skip layers on other devices.
            #     if is_pp_missing_parameter(name, self):
            #         continue
            #     param = params_dict[name]
            #     weight_loader = param.weight_loader
            #     weight_loader(param, loaded_weight, shard_id)
            #     break
            # else:
            #     # Skip loading extra bias for GPTQ models.
            #     if name.endswith(".bias") and name not in params_dict:
            #         continue
            #     if is_pp_missing_parameter(name, self):
            #         continue

            #     param = params_dict[name]
            #     weight_loader = getattr(param, "weight_loader",
            #                             default_weight_loader)
            #     weight_loader(param, loaded_weight)

        return loaded_params


class GraniteMoeHybridForCausalLM(nn.Module, HasInnerState, SupportsLoRA, SupportsPP,
                       IsHybrid, SupportsV0Only, SupportsQuant):
    
    # <LoRA specific attributes>
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["up_proj", "down_proj"]
    }
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings", #TODO: Removed lm_head
    }
    embedding_padding_modules = ["lm_head"] #TODO: Removed lm_head
    # </LoRA specific attributes>

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        lora_config = vllm_config.lora_config
        scheduler_config = vllm_config.scheduler_config
        assert not cache_config.enable_prefix_caching, \
            "GraniteMoeHybrid currently does not support prefix caching"

        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = GraniteMoeHybridModel(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

        # Granite MOE Shared:
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"))
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size,
                                                scale=1 /
                                                self.config.logits_scaling)
        # BAMBA:
        # self.lm_head = ParallelLMHead(
        #     self.unpadded_vocab_size,
        #     config.hidden_size,
        #     org_num_embeddings=config.vocab_size,
        #     padding_size=DEFAULT_VOCAB_PADDING_SIZE
        #     # We need bigger padding if using lora for kernel
        #     # compatibility
        #     if not lora_config else lora_config.lora_vocab_padding_size,
        # )
        # Used to track and store by the Mamba cache between steps.
        self.mamba_cache: Optional[MambaCacheManager] = None

        # self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
        #                                         config.vocab_size)
        self.sampler = get_sampler()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs):
        if self.mamba_cache is None:

            # STW: HF code from Friday returns 40, HF code from Monday returns 0. Should be 36.
            #num_mamba_layers = self.model_config.get_num_layers_by_block_type(
            #            self.vllm_config.parallel_config, LayerBlockType.mamba)
            # STW: This works properly:
            num_mamba_layers = len(list(filter(lambda x: x == "mamba2", self.model_config.hf_config.layer_types)))

            self.mamba_cache = MambaCacheManager(
                self.vllm_config, self.model_config.dtype, num_mamba_layers,
                *self._get_mamba_cache_shape())
        mamba_cache_params = self.mamba_cache.current_run_tensors(**kwargs)
        hidden_states = self.model(input_ids, positions, mamba_cache_params,
                                   intermediate_tensors, inputs_embeds)

        return hidden_states

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def _get_mamba_cache_shape(
            self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        world_size = get_tensor_model_parallel_world_size()
        hidden_size = self.config.hidden_size

        conv_state_shape, temporal_state_shape = None, None

        intermediate_size = self.config.mamba_expand * hidden_size

        # if n_groups is not divisible by world_size, need to extend the shards
        # to ensure all groups needed by a head is sharded along with it
        n_groups = (self.config.mamba_n_groups + extra_groups_for_head_shards(
            self.config.mamba_n_groups, world_size))

        # - heads and n_groups are TP-ed
        conv_dim = (intermediate_size +
                    2 * n_groups * self.config.mamba_d_state)
        conv_state_shape = (
            divide(conv_dim, world_size),
            self.config.mamba_d_conv - 1,
        )

        # These are not TP-ed as they depend on A, dt_bias, D
        # - they are typically small
        #   e.g., (h_heads, d_head, d_state) = (128, 64, 128)
        temporal_state_shape = (
            divide(self.config.mamba_n_heads, world_size),
            self.config.mamba_d_head,
            self.config.mamba_d_state,
        )
        return conv_state_shape, temporal_state_shape

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)