# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Bernini Wan renderer used by LightX2V DMD training.

The module layout and numerical order follow the renderer implementation in
the local Bernini repository.  The state-dict names intentionally remain
compatible with the released Bernini high/low expert checkpoints, while the
execution path is implemented locally and does not inherit Diffusers models.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors import safe_open
from safetensors.torch import save_file

from lightx2v_train.runtime.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    is_sequence_parallel_enabled,
)
from lightx2v_train.runtime.sequence_parallel import all_gather_sequence, all_to_all_4d, shrink_sequence

_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
_WEIGHTS_INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
_CUDA_ATTENTION_BACKEND: tuple[str, Any] | None = None


class _Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self):
        return dict(self)


@dataclass
class BerniniTransformerOutput:
    sample: torch.Tensor


def _pad_sequence(tensor: torch.Tensor, padded_length: int) -> torch.Tensor:
    padding = padded_length - tensor.shape[1]
    if padding < 0:
        raise ValueError(f"Cannot pad sequence length {tensor.shape[1]} to {padded_length}.")
    if padding == 0:
        return tensor
    return torch.cat([tensor, tensor.new_zeros(tensor.shape[0], padding, *tensor.shape[2:])], dim=1)


def _select_cuda_attention_backend() -> tuple[str, Any]:
    global _CUDA_ATTENTION_BACKEND
    if _CUDA_ATTENTION_BACKEND is not None:
        return _CUDA_ATTENTION_BACKEND
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9:
        try:
            from flash_attn_interface import flash_attn_varlen_func

            _CUDA_ATTENTION_BACKEND = ("fa3", flash_attn_varlen_func)
            return _CUDA_ATTENTION_BACKEND
        except Exception:
            pass
    try:
        from flash_attn import flash_attn_varlen_func

        _CUDA_ATTENTION_BACKEND = ("fa2", flash_attn_varlen_func)
    except Exception:
        _CUDA_ATTENTION_BACKEND = ("sdpa", None)
    return _CUDA_ATTENTION_BACKEND


def _single_segment_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Non-causal packed attention for one Bernini sample."""
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("Bernini attention expects packed [tokens, heads, head_dim] tensors.")
    flash_dtypes = {torch.float16, torch.bfloat16}
    if query.device.type == "cuda" and query.dtype in flash_dtypes and key.dtype == query.dtype == value.dtype:
        backend, attention_fn = _select_cuda_attention_backend()
        if backend != "sdpa":
            cu_q = torch.tensor([0, query.shape[0]], dtype=torch.int32, device=query.device)
            cu_k = torch.tensor([0, key.shape[0]], dtype=torch.int32, device=key.device)
            if backend == "fa3":
                output = attention_fn(
                    query.contiguous(),
                    key.contiguous(),
                    value.contiguous(),
                    cu_seqlens_q=cu_q,
                    cu_seqlens_k=cu_k,
                    max_seqlen_q=query.shape[0],
                    max_seqlen_k=key.shape[0],
                    causal=False,
                )
                return output[0] if isinstance(output, tuple) else output
            return attention_fn(
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
                cu_q,
                cu_k,
                query.shape[0],
                key.shape[0],
                causal=False,
            )

    query = query.transpose(0, 1).unsqueeze(0)
    key = key.transpose(0, 1).unsqueeze(0)
    value = value.transpose(0, 1).unsqueeze(0)
    output = F.scaled_dot_product_attention(query, key, value, is_causal=False)
    return output.squeeze(0).transpose(0, 1).contiguous()


def _apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    hidden_states_complex = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(-1, (-1, 2)))
    output = torch.view_as_real(hidden_states_complex * freqs).flatten(-2)
    return output.type_as(hidden_states)


def _rope_params(position: int | torch.Tensor, dim: int, theta: float = 10000.0) -> torch.Tensor:
    if dim % 2:
        raise ValueError(f"RoPE dimension must be even, got {dim}.")
    if isinstance(position, int):
        positions = torch.arange(position, dtype=torch.float64, device="cpu")
    else:
        positions = position.to(dtype=torch.float64)
    frequencies = 1.0 / torch.pow(
        torch.tensor(theta, dtype=torch.float64, device=positions.device),
        torch.arange(0, dim, 2, dtype=torch.float64, device=positions.device) / dim,
    )
    phases = torch.outer(positions, frequencies)
    return torch.polar(torch.ones_like(phases), phases)


class BerniniRMSNorm(nn.Module):
    """The FP32-variance RMSNorm used by the local Bernini renderer."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = torch.Size((dim,))
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().square().mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)
        return hidden_states * self.weight


class BerniniFP32LayerNorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(input_dtype)


def _timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> torch.Tensor:
    if timesteps.ndim != 1:
        raise ValueError(f"Timesteps must be one-dimensional, got shape={tuple(timesteps.shape)}.")
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)
    embedding = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    embedding = scale * torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)
    if flip_sin_to_cos:
        embedding = torch.cat([embedding[:, half_dim:], embedding[:, :half_dim]], dim=-1)
    if embedding_dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class BerniniTimesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return _timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )


class BerniniTimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class BerniniTextProjection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, hidden_size)
        self.act_1 = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act_1(self.linear_1(hidden_states)))


class BerniniTimeTextEmbedding(nn.Module):
    def __init__(self, dim: int, time_freq_dim: int, time_proj_dim: int, text_embed_dim: int):
        super().__init__()
        self.timesteps_proj = BerniniTimesteps(time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = BerniniTimestepEmbedding(time_freq_dim, dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = BerniniTextProjection(text_embed_dim, dim)
        self.image_embedder = None

    def forward(self, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor, encoder_hidden_states_image=None):
        if encoder_hidden_states_image is not None:
            raise ValueError("Bernini DMD does not use image added-KV embeddings.")
        timestep = self.timesteps_proj(timestep)
        weight_dtype = self.time_embedder.linear_1.weight.dtype
        if timestep.dtype != weight_dtype and weight_dtype != torch.int8:
            timestep = timestep.to(weight_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))
        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        return temb, timestep_proj, encoder_hidden_states, None


class BerniniGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(hidden_states), approximate="tanh")


class BerniniFeedForward(nn.Module):
    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        self.net = nn.ModuleList([BerniniGELU(dim, inner_dim), nn.Dropout(0.0), nn.Linear(inner_dim, dim)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class BerniniRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
        use_src_id_rotary_emb: bool = False,
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.use_src_id_rotary_emb = use_src_id_rotary_emb

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        self.freqs = torch.cat([_rope_params(max_seq_len, dim, theta) for dim in (t_dim, h_dim, w_dim)], dim=1)

    def forward(self, hidden_states: torch.Tensor, source_id: float | None = None) -> torch.Tensor:
        _, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        if num_frames % p_t or height % p_h or width % p_w:
            raise ValueError(
                f"Latent shape {(num_frames, height, width)} is not divisible by patch_size={self.patch_size}."
            )
        grid_t, grid_h, grid_w = num_frames // p_t, height // p_h, width // p_w
        if max(grid_t, grid_h, grid_w) > self.max_seq_len:
            raise ValueError(f"Bernini RoPE grid {(grid_t, grid_h, grid_w)} exceeds max_seq_len={self.max_seq_len}.")

        if self.freqs.device != hidden_states.device:
            self.freqs = self.freqs.to(hidden_states.device)
        sizes = (
            self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
            self.attention_head_dim // 6,
            self.attention_head_dim // 6,
        )
        freq_t, freq_h, freq_w = self.freqs.split(sizes, dim=1)
        freq_t = freq_t[:grid_t].view(grid_t, 1, 1, -1).expand(grid_t, grid_h, grid_w, -1)
        freq_h = freq_h[:grid_h].view(1, grid_h, 1, -1).expand(grid_t, grid_h, grid_w, -1)
        freq_w = freq_w[:grid_w].view(1, 1, grid_w, -1).expand(grid_t, grid_h, grid_w, -1)
        freqs = torch.cat([freq_t, freq_h, freq_w], dim=-1).reshape(1, 1, grid_t * grid_h * grid_w, -1)

        if self.use_src_id_rotary_emb:
            if source_id is None:
                raise ValueError("source_id is required when use_src_id_rotary_emb=True.")
            position = torch.tensor([float(source_id)], dtype=torch.float64, device=hidden_states.device)
            source_freqs = _rope_params(position, self.attention_head_dim, self.theta)
            freqs = freqs * source_freqs.view(1, 1, 1, -1)
        return freqs


class BerniniAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"Attention dim={dim} must be divisible by num_heads={num_heads}.")
        self.heads = num_heads
        self.dim_head = dim // num_heads
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Dropout(0.0)])
        self.norm_q = BerniniRMSNorm(dim, eps)
        self.norm_k = BerniniRMSNorm(dim, eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        rotary_emb: torch.Tensor | None = None,
        sequence_length: int | None = None,
        local_valid_length: int | None = None,
    ) -> torch.Tensor:
        is_cross_attention = encoder_hidden_states is not None
        key_value_states = encoder_hidden_states if is_cross_attention else hidden_states

        query = self.norm_q(self.to_q(hidden_states)).unflatten(2, (self.heads, self.dim_head))
        key = self.norm_k(self.to_k(key_value_states)).unflatten(2, (self.heads, self.dim_head))
        value = self.to_v(key_value_states).unflatten(2, (self.heads, self.dim_head))

        if not is_cross_attention:
            if rotary_emb is None or sequence_length is None:
                raise ValueError("Bernini self-attention requires RoPE and the unpadded sequence length.")
            query = all_to_all_4d(query, scatter_dim=2, gather_dim=1)
            key = all_to_all_4d(key, scatter_dim=2, gather_dim=1)
            value = all_to_all_4d(value, scatter_dim=2, gather_dim=1)
            padded_sequence_length = query.shape[1]
            query = query[:, :sequence_length]
            key = key[:, :sequence_length]
            value = value[:, :sequence_length]
            query = _apply_rotary_emb(query, rotary_emb)
            key = _apply_rotary_emb(key, rotary_emb)
            output = _single_segment_attention(query.squeeze(0), key.squeeze(0), value.squeeze(0)).unsqueeze(0)
            output = _pad_sequence(output, padded_sequence_length)
            output = all_to_all_4d(output, scatter_dim=1, gather_dim=2)
        else:
            if rotary_emb is not None:
                raise ValueError("Bernini cross-attention does not use VAE-token RoPE.")
            valid_length = hidden_states.shape[1] if local_valid_length is None else local_valid_length
            query = query[:, :valid_length]
            output = _single_segment_attention(query.squeeze(0), key.squeeze(0), value.squeeze(0)).unsqueeze(0)
            output = _pad_sequence(output, hidden_states.shape[1])

        output = output.flatten(2, 3).type_as(query)
        output = self.to_out[0](output)
        return self.to_out[1](output)


class BerniniTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attn_norm: bool,
        eps: float,
    ):
        super().__init__()
        self.norm1 = BerniniFP32LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.attn1 = BerniniAttention(dim, num_heads, eps)
        self.attn2 = BerniniAttention(dim, num_heads, eps)
        self.norm2 = BerniniFP32LayerNorm(dim, eps=eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.ffn = BerniniFeedForward(dim, ffn_dim)
        self.norm3 = BerniniFP32LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        sequence_length: int,
        local_valid_length: int,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table.float() + temb.float()
        ).chunk(6, dim=1)

        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attention_output = self.attn1(
            norm_hidden_states,
            rotary_emb=rotary_emb,
            sequence_length=sequence_length,
        )
        hidden_states = (hidden_states.float() + attention_output * gate_msa).type_as(hidden_states)

        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attention_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            local_valid_length=local_valid_length,
        )
        hidden_states = hidden_states + attention_output

        norm_hidden_states = (
            self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa
        ).type_as(hidden_states)
        feed_forward_output = self.ffn(norm_hidden_states)
        return (hidden_states.float() + feed_forward_output.float() * c_gate_msa).type_as(hidden_states)


class BerniniWanTransformer3DModel(nn.Module):
    """Local Bernini renderer with checkpoint-compatible module names."""

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: str | None = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: int | None = None,
        added_kv_proj_dim: int | None = None,
        rope_max_seq_len: int = 1024,
        use_src_id_rotary_emb: bool = False,
        **config_metadata,
    ):
        super().__init__()
        patch_size = tuple(int(value) for value in patch_size)
        if len(patch_size) != 3 or any(value <= 0 for value in patch_size):
            raise ValueError(f"patch_size must contain three positive integers, got {patch_size}.")
        if qk_norm != "rms_norm_across_heads":
            raise ValueError(f"Bernini requires qk_norm='rms_norm_across_heads', got {qk_norm!r}.")
        if image_dim is not None or added_kv_proj_dim is not None:
            raise ValueError("Bernini DMD supports source VAE tokens, not image added-KV attention.")
        if attention_head_dim % 2:
            raise ValueError(f"attention_head_dim must be even, got {attention_head_dim}.")

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels
        config_values = {
            "patch_size": patch_size,
            "num_attention_heads": num_attention_heads,
            "attention_head_dim": attention_head_dim,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "text_dim": text_dim,
            "freq_dim": freq_dim,
            "ffn_dim": ffn_dim,
            "num_layers": num_layers,
            "cross_attn_norm": cross_attn_norm,
            "qk_norm": qk_norm,
            "eps": eps,
            "image_dim": image_dim,
            "added_kv_proj_dim": added_kv_proj_dim,
            "rope_max_seq_len": rope_max_seq_len,
            "use_src_id_rotary_emb": use_src_id_rotary_emb,
        }
        config_values.update(config_metadata)
        config_values.setdefault("_class_name", "WanTransformer3DModel")
        self.config = _Config(config_values)

        self.rope = BerniniRotaryPosEmbed(
            attention_head_dim,
            patch_size,
            rope_max_seq_len,
            use_src_id_rotary_emb=use_src_id_rotary_emb,
        )
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.condition_embedder = BerniniTimeTextEmbedding(inner_dim, freq_dim, inner_dim * 6, text_dim)
        self.blocks = nn.ModuleList(
            [
                BerniniTransformerBlock(inner_dim, ffn_dim, num_attention_heads, cross_attn_norm, eps)
                for _ in range(num_layers)
            ]
        )
        self.norm_out = BerniniFP32LayerNorm(inner_dim, eps=eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)
        self.gradient_checkpointing = False

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def enable_lightx2v_sequence_parallel(self):
        sp_size = get_sequence_parallel_world_size()
        if self.config.num_attention_heads % sp_size:
            raise ValueError(
                f"Bernini num_attention_heads={self.config.num_attention_heads} is not divisible by sp_size={sp_size}."
            )
        return self

    @staticmethod
    def cuda_attention_backend() -> str:
        return _select_cuda_attention_backend()[0]

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def enable_liger_rms_norm(self, *, in_place: bool = True) -> int:
        try:
            from liger_kernel.transformers import LigerRMSNorm
        except ImportError as exc:
            raise ImportError(
                "Bernini rms_norm_backend=liger requires liger-kernel>=0.7.0. "
                "Install it with `pip install 'liger-kernel>=0.7.0'`, or set "
                "BERNINI_RMS_NORM_BACKEND=torch."
            ) from exc

        replaced = 0
        for parent in list(self.modules()):
            for name, child in list(parent.named_children()):
                if not isinstance(child, BerniniRMSNorm):
                    continue
                replacement = LigerRMSNorm(
                    hidden_size=child.dim[0],
                    eps=child.eps,
                    offset=0.0,
                    casting_mode="llama",
                    init_fn="ones",
                    in_place=in_place,
                    elementwise_affine=True,
                )
                replacement.weight = child.weight
                replacement.train(child.training)
                setattr(parent, name, replacement)
                replaced += 1
        return replaced

    def patch_vae_latent(self, hidden_states: torch.Tensor, source_id: float = 0.0):
        rotary_emb = self.rope(hidden_states, source_id=source_id)
        hidden_states = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        return hidden_states, rotary_emb

    def patch_vae_embedding(self, hidden_states: torch.Tensor):
        return self.patch_embedding(hidden_states).flatten(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        source_image_vae_patches: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
        source_image_rope_cos: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
        source_image_rope_sin: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> BerniniTransformerOutput | tuple[torch.Tensor]:
        if attention_kwargs:
            unsupported = {key: value for key, value in attention_kwargs.items() if key != "scale" or value != 1.0}
            if unsupported:
                raise ValueError(f"Unsupported Bernini attention kwargs: {unsupported}.")
        if encoder_hidden_states_image is not None:
            raise ValueError("Bernini DMD uses cached source VAE tokens, not encoder_hidden_states_image.")
        if hidden_states.ndim != 5:
            raise ValueError(f"Bernini latent must have shape [B,C,T,H,W], got {tuple(hidden_states.shape)}.")
        if hidden_states.shape[0] != 1:
            raise ValueError("Bernini DMD currently requires batch_size=1; use gradient accumulation for larger batches.")
        if timestep.ndim != 1 or timestep.shape[0] != 1:
            raise ValueError(f"Bernini DMD expects one timestep for its single sample, got {tuple(timestep.shape)}.")
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != 1:
            raise ValueError(
                f"Bernini context must have shape [1,sequence,{self.config.text_dim}], got {tuple(encoder_hidden_states.shape)}."
            )

        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        grid_t, grid_h, grid_w = num_frames // p_t, height // p_h, width // p_w
        target_hidden_states, target_rotary_emb = self.patch_vae_latent(hidden_states, source_id=0.0)
        target_rotary_emb = target_rotary_emb.transpose(1, 2)
        target_sequence_length = target_hidden_states.shape[1]

        source_values = (source_image_vae_patches, source_image_rope_cos, source_image_rope_sin)
        if all(value is None for value in source_values):
            source_sequence_length = 0
            combined_hidden_states = target_hidden_states
            rotary_emb = target_rotary_emb
        else:
            if any(value is None for value in source_values):
                raise ValueError("Bernini R2V requires source patches and both source RoPE cos/sin lists.")
            if any(not isinstance(value, (list, tuple)) for value in source_values):
                raise TypeError("Bernini R2V source patches and RoPE values must be lists or tuples.")
            if not (len(source_image_vae_patches) == len(source_image_rope_cos) == len(source_image_rope_sin) > 0):
                raise ValueError("Bernini R2V source patch and RoPE lists must have the same non-zero length.")

            source_hidden_states = []
            source_rotary_emb = []
            expected_patch_shape = (p_t, p_h, p_w)
            for image_index, (patches, rope_cos, rope_sin) in enumerate(
                zip(source_image_vae_patches, source_image_rope_cos, source_image_rope_sin)
            ):
                if not all(torch.is_tensor(value) for value in (patches, rope_cos, rope_sin)):
                    raise TypeError(f"Bernini source inputs[{image_index}] must all be tensors.")
                if patches.ndim != 6:
                    raise ValueError(
                        f"Bernini source patches[{image_index}] must have shape [B,N,C,pt,ph,pw], got {tuple(patches.shape)}."
                    )
                if patches.shape[0] != batch_size or patches.shape[2] != self.config.in_channels:
                    raise ValueError(f"Bernini source patches[{image_index}] have incompatible shape {tuple(patches.shape)}.")
                if tuple(patches.shape[-3:]) != expected_patch_shape:
                    raise ValueError(
                        f"Bernini source patches[{image_index}] use patch shape {tuple(patches.shape[-3:])}, expected {expected_patch_shape}."
                    )
                expected_rope_shape = (batch_size, patches.shape[1], 1, self.config.attention_head_dim)
                if tuple(rope_cos.shape) != expected_rope_shape or tuple(rope_sin.shape) != expected_rope_shape:
                    raise ValueError(
                        f"Bernini source RoPE[{image_index}] must have shape {expected_rope_shape}, "
                        f"got cos={tuple(rope_cos.shape)} sin={tuple(rope_sin.shape)}."
                    )

                num_tokens = patches.shape[1]
                patches = patches.to(device=hidden_states.device, dtype=hidden_states.dtype)
                embedded = self.patch_embedding(patches.flatten(0, 1))
                if tuple(embedded.shape[-3:]) != (1, 1, 1):
                    raise ValueError(f"Bernini source patches[{image_index}] must each produce exactly one token.")
                source_hidden_states.append(embedded.flatten(1).unflatten(0, (batch_size, num_tokens)))
                rope_cos = rope_cos.to(device=hidden_states.device, dtype=torch.float64)
                rope_sin = rope_sin.to(device=hidden_states.device, dtype=torch.float64)
                source_rotary_emb.append(torch.complex(rope_cos[..., 0::2], rope_sin[..., 0::2]))

            source_sequence_length = sum(value.shape[1] for value in source_hidden_states)
            combined_hidden_states = torch.cat([*source_hidden_states, target_hidden_states], dim=1)
            rotary_emb = torch.cat([*source_rotary_emb, target_rotary_emb], dim=1)

        sequence_length = combined_hidden_states.shape[1]
        local_valid_length = sequence_length
        if is_sequence_parallel_enabled():
            sp_size = get_sequence_parallel_world_size()
            padded_sequence_length = math.ceil(sequence_length / sp_size) * sp_size
            local_sequence_length = padded_sequence_length // sp_size
            local_start = get_sequence_parallel_rank() * local_sequence_length
            local_valid_length = max(0, min(local_sequence_length, sequence_length - local_start))
            combined_hidden_states = shrink_sequence(
                _pad_sequence(combined_hidden_states, padded_sequence_length),
                dim=1,
            )

        temb, timestep_proj, encoder_hidden_states, _ = self.condition_embedder(
            timestep,
            encoder_hidden_states,
            None,
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def block_forward(states, context, modulation, rope, current_block=block):
                    return current_block(
                        states,
                        context,
                        modulation,
                        rope,
                        sequence_length,
                        local_valid_length,
                    )

                combined_hidden_states = torch.utils.checkpoint.checkpoint(
                    block_forward,
                    combined_hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    use_reentrant=False,
                )
            else:
                combined_hidden_states = block(
                    combined_hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    sequence_length,
                    local_valid_length,
                )

        shift_table, scale_table = self.scale_shift_table.float().chunk(2, dim=1)
        shift = shift_table + temb.float().unsqueeze(1)
        scale = scale_table + temb.float().unsqueeze(1)
        combined_hidden_states = (
            self.norm_out(combined_hidden_states.float()) * (1 + scale) + shift
        ).type_as(combined_hidden_states)
        combined_hidden_states = self.proj_out(combined_hidden_states)

        if is_sequence_parallel_enabled():
            combined_hidden_states = all_gather_sequence(combined_hidden_states, dim=1)[:, :sequence_length]
        if source_sequence_length:
            combined_hidden_states = combined_hidden_states[
                :,
                source_sequence_length : source_sequence_length + target_sequence_length,
            ]

        output = combined_hidden_states.reshape(
            batch_size,
            grid_t,
            grid_h,
            grid_w,
            p_t,
            p_h,
            p_w,
            self.config.out_channels,
        )
        output = output.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = output.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        if not return_dict:
            return (output,)
        return BerniniTransformerOutput(sample=output)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        *,
        torch_dtype: torch.dtype | str | None = None,
        subfolder: str | None = None,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unsupported native Bernini from_pretrained arguments: {sorted(kwargs)}.")
        model_dir = Path(pretrained_model_name_or_path)
        if subfolder is not None:
            model_dir = model_dir / subfolder
        model_dir = model_dir.expanduser().resolve()
        config_path = model_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Bernini config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if not isinstance(config, dict):
            raise TypeError(f"Bernini config must be a JSON object, got {type(config)!r}.")

        with torch.device("meta"):
            model = cls(**config)
        expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
        file_to_keys = cls._validate_checkpoint_headers(model_dir, expected)
        target_dtype = cls._resolve_dtype(torch_dtype)

        loaded_keys = set()
        for filename, keys in file_to_keys.items():
            with safe_open(filename, framework="pt", device="cpu") as handle:
                shard = {}
                for key in keys:
                    tensor = handle.get_tensor(key)
                    if target_dtype is not None and tensor.is_floating_point():
                        tensor = tensor.to(target_dtype)
                    shard[key] = tensor
            model.load_state_dict(shard, strict=False, assign=True)
            loaded_keys.update(shard)
        if loaded_keys != set(expected):
            raise RuntimeError("Internal Bernini checkpoint loader error: loaded key set changed after header validation.")
        meta_parameters = [name for name, parameter in model.named_parameters() if parameter.is_meta]
        if meta_parameters:
            raise RuntimeError(f"Bernini checkpoint left meta parameters uninitialized: {meta_parameters[:8]}.")
        return model

    @staticmethod
    def _resolve_dtype(value: torch.dtype | str | None) -> torch.dtype | None:
        if value is None or isinstance(value, torch.dtype):
            return value
        normalized = str(value).removeprefix("torch.").lower()
        aliases = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
        }
        if normalized not in aliases:
            raise ValueError(f"Unsupported Bernini torch_dtype={value!r}.")
        return aliases[normalized]

    @classmethod
    def _validate_checkpoint_headers(cls, model_dir: Path, expected: dict[str, tuple[int, ...]]):
        index_path = model_dir / _WEIGHTS_INDEX_NAME
        single_path = model_dir / _WEIGHTS_NAME
        if index_path.is_file():
            with index_path.open("r", encoding="utf-8") as handle:
                index = json.load(handle)
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"Invalid Bernini safetensors index: {index_path}")
            file_to_declared_keys: dict[Path, set[str]] = {}
            for key, filename in weight_map.items():
                if not isinstance(key, str) or not isinstance(filename, str):
                    raise TypeError(f"Invalid Bernini weight_map entry in {index_path}: {key!r} -> {filename!r}")
                file_to_declared_keys.setdefault(model_dir / filename, set()).add(key)
        elif single_path.is_file():
            file_to_declared_keys = {single_path: set(expected)}
        else:
            raise FileNotFoundError(
                f"Bernini weights not found; expected {single_path.name} or {index_path.name} in {model_dir}."
            )

        declared_keys = set().union(*file_to_declared_keys.values())
        missing = sorted(set(expected) - declared_keys)
        unexpected = sorted(declared_keys - set(expected))
        if missing or unexpected:
            raise RuntimeError(
                f"Bernini checkpoint key mismatch: missing={missing[:8]} unexpected={unexpected[:8]} "
                f"(expected={len(expected)}, declared={len(declared_keys)})."
            )

        file_to_keys: dict[Path, list[str]] = {}
        seen = set()
        for filename, declared in file_to_declared_keys.items():
            if not filename.is_file():
                raise FileNotFoundError(f"Bernini checkpoint shard declared by index is missing: {filename}")
            with safe_open(filename, framework="pt", device="cpu") as handle:
                actual = set(handle.keys())
                if actual != declared:
                    raise RuntimeError(
                        f"Bernini shard/index mismatch for {filename.name}: "
                        f"missing={sorted(declared - actual)[:8]} unexpected={sorted(actual - declared)[:8]}."
                    )
                for key in sorted(actual):
                    if key in seen:
                        raise RuntimeError(f"Duplicate Bernini tensor key across shards: {key}")
                    shape = tuple(handle.get_slice(key).get_shape())
                    if shape != expected[key]:
                        raise RuntimeError(
                            f"Bernini tensor shape mismatch for {key}: checkpoint={shape}, expected={expected[key]}."
                        )
                    seen.add(key)
            file_to_keys[filename] = sorted(actual)
        return file_to_keys

    def save_config(self, save_directory: str | os.PathLike):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        config = self.config.to_dict()
        config["patch_size"] = list(config["patch_size"])
        with (save_directory / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        *,
        safe_serialization: bool = True,
        max_shard_size: str | int = "10GB",
        state_dict: dict[str, torch.Tensor] | None = None,
    ):
        if not safe_serialization:
            raise ValueError("Native Bernini checkpoints are saved as safetensors only.")
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self.save_config(save_directory)

        for stale_path in save_directory.glob("diffusion_pytorch_model*.safetensors"):
            stale_path.unlink()
        index_path = save_directory / _WEIGHTS_INDEX_NAME
        if index_path.exists():
            index_path.unlink()

        state_dict = self.state_dict() if state_dict is None else state_dict
        split = split_torch_state_dict_into_shards(
            state_dict,
            max_shard_size=max_shard_size,
            filename_pattern="diffusion_pytorch_model{suffix}.safetensors",
        )
        for filename, tensor_names in split.filename_to_tensors.items():
            shard = {
                name: state_dict[name].detach().cpu().contiguous()
                for name in tensor_names
            }
            save_file(shard, save_directory / filename, metadata={"format": "pt"})
        if split.is_sharded:
            payload = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
            with index_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")


__all__ = [
    "BerniniRMSNorm",
    "BerniniTransformerOutput",
    "BerniniWanTransformer3DModel",
]
