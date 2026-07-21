"""Trainable Bernini (non-R) high/low renderer expert adapter."""

import os

import torch
from loguru import logger
from peft import LoraConfig

from lightx2v_train.runtime.distributed import get_sequence_parallel_world_size
from lightx2v_train.utils.registry import MODEL_REGISTER
from lightx2v_train.utils.utils import get_running_dtype

from .native.bernini import BerniniRMSNorm, BerniniWanTransformer3DModel
from .wan_t2v import WanT2VModel


@MODEL_REGISTER("bernini_t2v")
@MODEL_REGISTER("bernini_t2v_a14b")
class BerniniT2VA14BModel(WanT2VModel):
    """Pure-T2V adapter for one split Bernini Wan2.2 renderer expert.

    MLLM, connector, T5, and VAE execution are intentionally excluded. The
    adapter consumes cached 4096-wide Bernini contexts and loads exactly one
    split Bernini renderer expert.
    """

    renderer_task = "t2v"

    _CONDITION_KEYS = (
        "cond_embeds_wtxt_wvit",
        "cond_embeds_wtxt_wovit",
        "cond_embeds_wotxt_wvit",
        "cond_embeds_wotxt_wovit",
    )
    _DEFAULT_CONDITION_KEY = "cond_embeds_wtxt_wvit"
    _SOURCE_CONDITION_KEYS = (
        "source_image_vae_patches",
        "source_image_rope_cos",
        "source_image_rope_sin",
    )
    _DEFAULT_LORA_TARGETS = (
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "ffn.net.0.proj",
        "ffn.net.2",
    )
    _EXPERT_ALIASES = {
        "high": "high",
        "high_noise": "high",
        "high_noise_model": "high",
        "low": "low",
        "low_noise": "low",
        "low_noise_model": "low",
    }

    def load_components(self, transformer_only=False, reference_model=None):
        model_config = self.config["model"]
        transformer_path = os.fspath(model_config["pretrained_model_name_or_path"])

        if model_config.get("load_vae", False):
            raise ValueError("Bernini DMD samples renderer latents directly; model.load_vae must be false.")
        if model_config.get("load_text_encoder", False):
            raise ValueError("Bernini DMD requires offline MLLM+T5 contexts; model.load_text_encoder must be false.")
        if model_config.get("load_mllm", False):
            raise ValueError("Bernini DMD requires offline MLLM contexts; model.load_mllm must be false.")
        if model_config.get("causal", False):
            raise ValueError("Bernini DMD trains bidirectional Wan2.2 experts; model.causal=true is unsupported.")

        self.expert = self._resolve_expert(model_config, transformer_path)
        self.transformer_path = transformer_path
        self.load_vae = False
        self.load_text_encoder = False
        self.load_transformer = bool(model_config.get("load_transformer", True))
        self.use_causal_transformer = False
        self.sample_posterior = False
        self.num_train_timesteps = int(self.config.get("scheduler", {}).get("num_train_timesteps", 1000))
        self.max_sequence_length = int(model_config.get("max_sequence_length", 512))
        self.transformer_param_dtype = get_running_dtype(model_config.get("transformer_param_dtype", "fp32"))
        self.rms_norm_backend = str(model_config.get("rms_norm_backend", "torch")).strip().lower()
        if self.rms_norm_backend not in {"torch", "liger"}:
            raise ValueError(f"Unsupported Bernini rms_norm_backend={self.rms_norm_backend!r}; expected 'torch' or 'liger'.")
        self.vae_stride = tuple(model_config.get("vae_stride", (4, 8, 8)))
        self.patch_size = tuple(model_config.get("patch_size", (1, 2, 2)))
        self.sp_size = get_sequence_parallel_world_size()
        self.attention_backend = "uninitialized"
        self.rms_norm_module_count = 0

        self.vae = None
        self.text_encoder = None
        self.text_pipeline = None
        self.transformer = self._load_transformer(transformer_path) if self.load_transformer else None
        if self.transformer is not None:
            self._configure_transformer()

        self.vae_scale_factor_temporal = self.vae_stride[0]
        self.vae_scale_factor_spatial = self.vae_stride[1]
        if reference_model is not None:
            self.vae_scale_factor_temporal = reference_model.vae_scale_factor_temporal
            self.vae_scale_factor_spatial = reference_model.vae_scale_factor_spatial

        logger.info(
            "[model] Bernini {} A14B expert={} transformer={} offline_conditioning=true "
            "attention_backend={} rms_norm_backend={} rms_norm_modules={}",
            self.renderer_task.upper(),
            self.expert,
            self.transformer_path,
            self.attention_backend,
            self.rms_norm_backend,
            self.rms_norm_module_count,
        )

    @classmethod
    def _resolve_expert(cls, model_config, transformer_path):
        configured = model_config.get("expert", model_config.get("training_target"))
        path_name = os.path.basename(os.path.normpath(transformer_path)).lower()
        inferred = cls._EXPERT_ALIASES.get(path_name)
        if configured is None:
            if inferred is None:
                raise ValueError(f"Bernini DMD requires model.expert='high' or 'low' when it cannot be inferred from the split expert directory name: {transformer_path!r}.")
            return inferred

        key = str(configured).strip().lower()
        if key not in cls._EXPERT_ALIASES:
            raise ValueError(f"Unsupported Bernini model.expert={configured!r}; expected 'high' or 'low'.")
        expert = cls._EXPERT_ALIASES[key]
        if inferred is not None and inferred != expert:
            raise ValueError(f"model.expert={configured!r} selects {expert!r}, but {transformer_path!r} looks like the {inferred!r} split expert.")
        return expert

    def _load_transformer(self, model_path):
        transformer = BerniniWanTransformer3DModel.from_pretrained(
            model_path,
            torch_dtype=self.transformer_param_dtype,
        )
        return transformer.to(self.device, dtype=self.transformer_param_dtype)

    def _configure_transformer(self):
        self.transformer.enable_lightx2v_sequence_parallel()
        self.attention_backend = self.transformer.cuda_attention_backend()
        if self.device.type == "cuda" and self.attention_backend == "sdpa":
            logger.warning(
                "Bernini CUDA attention fell back to PyTorch SDPA; install FlashAttention 2 or 3 "
                "before full-resolution DMD training for the expected memory use and throughput."
            )
        self.patch_size = tuple(self.transformer.config.patch_size)
        text_dim = int(self.transformer.config.text_dim)
        if text_dim != 4096:
            raise ValueError(f"Bernini offline contexts require transformer text_dim=4096, got {text_dim}.")
        if getattr(self.transformer.config, "added_kv_proj_dim", None) is not None:
            raise ValueError("Bernini DMD supports the pure-T2V renderer expert only, not an image added-KV model.")
        num_heads = int(self.transformer.config.num_attention_heads)
        if self.sp_size > 1 and num_heads % self.sp_size != 0:
            raise ValueError(f"Bernini num_attention_heads={num_heads} must be divisible by sp_size={self.sp_size}.")

        expected_rms_norms = 4 * len(self.transformer.blocks)
        if self.rms_norm_backend == "liger":
            self.rms_norm_module_count = self.transformer.enable_liger_rms_norm()
        else:
            self.rms_norm_module_count = sum(isinstance(module, BerniniRMSNorm) for module in self.transformer.modules())
        if self.rms_norm_module_count != expected_rms_norms:
            raise RuntimeError(
                f"Bernini expected {expected_rms_norms} attention Q/K RMSNorm modules, "
                f"found {self.rms_norm_module_count} for backend={self.rms_norm_backend!r}."
            )

    def add_lora(self, rank, alpha, target_modules=None):
        if target_modules is None:
            target_modules = list(self._DEFAULT_LORA_TARGETS)
        super().add_lora(rank, alpha, target_modules)

    def _lora_config_for_infer(self):
        training_config = self.config.get("training", {})
        inference_config = self.config.get("inference", {})
        lora_config = dict(training_config.get("student", {}).get("lora", {}))
        lora_config.update(training_config.get("lora", {}))
        lora_config.update(inference_config.get("lora_config", {}))
        rank = int(lora_config.get("rank", 128))
        return LoraConfig(
            r=rank,
            lora_alpha=int(lora_config.get("alpha", rank)),
            init_lora_weights="gaussian",
            target_modules=lora_config.get("target_modules", list(self._DEFAULT_LORA_TARGETS)),
        )

    def encode_condition(self, sample):
        conditioning = sample["conditioning"]
        cached = conditioning.get("positive")
        if cached is None:
            cached = conditioning if any(key in conditioning for key in self._CONDITION_KEYS) else None
        if cached is None:
            raise RuntimeError("Bernini DMD has no online condition encoder. Provide conditioning.positive with cached MLLM+T5 contexts.")
        return self.prepare_cached_condition(cached)

    def encode_prompt_condition(self, prompt):
        raise RuntimeError("Bernini DMD requires offline MLLM+T5 contexts; raw prompt encoding is unavailable.")

    def prepare_cached_condition(self, condition):
        if torch.is_tensor(condition):
            return {"prompt_embed": self._prepare_context_tensor(condition)}
        if not isinstance(condition, dict):
            raise TypeError(f"Bernini cached condition must be a tensor or mapping, got {type(condition)!r}.")

        prepared = {}
        if condition.get("prompt_embed") is not None:
            prepared["prompt_embed"] = self._prepare_context_tensor(condition["prompt_embed"])
        for key in self._CONDITION_KEYS:
            value = condition.get(key)
            if isinstance(value, dict):
                value = value.get("prompt_embed")
            if value is not None:
                prepared[key] = self._prepare_context_tensor(value)
        source_values = [condition.get(key) for key in self._SOURCE_CONDITION_KEYS]
        if any(value is not None for value in source_values):
            if any(value is None for value in source_values):
                missing = [key for key, value in zip(self._SOURCE_CONDITION_KEYS, source_values) if value is None]
                raise KeyError(f"Bernini R2V cached condition is missing: {', '.join(missing)}.")
            for key, value in zip(self._SOURCE_CONDITION_KEYS, source_values):
                prepared[key] = self._prepare_source_list(value, key)
        if not prepared:
            expected = ", ".join(self._CONDITION_KEYS)
            raise KeyError(f"Bernini cached condition must contain prompt_embed or one of: {expected}.")
        return prepared

    def _prepare_source_list(self, value, key):
        if not isinstance(value, (list, tuple)) or not value:
            raise TypeError(f"Bernini cached source field {key!r} must be a non-empty list of tensors.")
        dtype = self.running_dtype if key == "source_image_vae_patches" else torch.float64
        prepared = []
        for image_index, tensor in enumerate(value):
            if not torch.is_tensor(tensor):
                raise TypeError(f"Bernini cached source field {key!r}[{image_index}] must be a tensor, got {type(tensor)!r}.")
            if key != "source_image_vae_patches" and tensor.dtype != torch.float64:
                raise ValueError(
                    f"Bernini cached source field {key!r}[{image_index}] must be FP64; "
                    "rebuild the R2V condition cache with the current preprocessor."
                )
            prepared.append(tensor.to(device=self.device, dtype=dtype))
        return prepared

    def _prepare_context_tensor(self, context):
        if not torch.is_tensor(context):
            raise TypeError(f"Bernini context must be a tensor, got {type(context)!r}.")
        context = context.to(device=self.device, dtype=self.running_dtype)
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context.ndim != 3 or context.shape[-1] != 4096:
            raise ValueError(f"Bernini context must have shape [batch, sequence, 4096] or [sequence, 4096], got {tuple(context.shape)}.")
        return context

    def _condition_to_context_tensor(self, condition, batch_size):
        if torch.is_tensor(condition):
            context = condition
        elif isinstance(condition, dict):
            context = condition.get("prompt_embed")
            if context is None:
                context = condition.get(self._DEFAULT_CONDITION_KEY)
            if isinstance(context, dict):
                context = context.get("prompt_embed")
        else:
            context = None
        if context is None:
            raise KeyError("Bernini denoise expects a context tensor, prompt_embed, or cond_embeds_wtxt_wvit.")

        context = self._prepare_context_tensor(context)
        if context.shape[0] == 1 and batch_size > 1:
            context = context.expand(batch_size, -1, -1)
        elif context.shape[0] != batch_size:
            raise ValueError(f"Bernini context batch size {context.shape[0]} does not match latent batch size {batch_size}.")
        # Bernini's official use_truncate=False path permits a context longer
        # than 512 after concatenating T5 and MLLM features. Never slice here.
        return context

    def _condition_to_source_inputs(self, condition, batch_size):
        if not isinstance(condition, dict):
            return None, None, None
        values = [condition.get(key) for key in self._SOURCE_CONDITION_KEYS]
        if all(value is None for value in values):
            return None, None, None
        if any(value is None for value in values):
            missing = [key for key, value in zip(self._SOURCE_CONDITION_KEYS, values) if value is None]
            raise KeyError(f"Bernini R2V condition is missing: {', '.join(missing)}.")

        prepared = [self._prepare_source_list(value, key) for key, value in zip(self._SOURCE_CONDITION_KEYS, values)]
        patches_list, cos_list, sin_list = prepared
        if not (len(patches_list) == len(cos_list) == len(sin_list)):
            raise ValueError("Bernini R2V source patch and RoPE lists must have the same length.")

        in_channels = int(self.transformer.config.in_channels)
        patch_size = tuple(self.transformer.config.patch_size)
        head_dim = int(self.transformer.config.attention_head_dim)
        for image_index, (patches, rope_cos, rope_sin) in enumerate(zip(patches_list, cos_list, sin_list)):
            if patches.ndim == 5:
                patches = patches.unsqueeze(0)
                patches_list[image_index] = patches
            if rope_cos.ndim == 3:
                rope_cos = rope_cos.unsqueeze(0)
                cos_list[image_index] = rope_cos
            if rope_sin.ndim == 3:
                rope_sin = rope_sin.unsqueeze(0)
                sin_list[image_index] = rope_sin
            expected_patch_tail = (in_channels, *patch_size)
            if patches.ndim != 6 or patches.shape[0] != batch_size or tuple(patches.shape[2:]) != expected_patch_tail:
                raise ValueError(f"Bernini source patches[{image_index}] must have shape [B,N,{in_channels},{','.join(map(str, patch_size))}], got {tuple(patches.shape)}.")
            expected_rope_shape = (batch_size, patches.shape[1], 1, head_dim)
            if tuple(rope_cos.shape) != expected_rope_shape or tuple(rope_sin.shape) != expected_rope_shape:
                raise ValueError(f"Bernini source RoPE[{image_index}] must have shape {expected_rope_shape}, got cos={tuple(rope_cos.shape)} sin={tuple(rope_sin.shape)}.")
        return patches_list, cos_list, sin_list

    def denoise(self, denoiser_input, timestep_or_sigma, condition):
        hidden_states = denoiser_input.hidden_states.to(device=self.device, dtype=self.running_dtype)
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.unsqueeze(0)
        if hidden_states.ndim != 5:
            raise ValueError(f"Bernini latent must have shape [B,C,T,H,W], got {tuple(hidden_states.shape)}.")

        timestep = timestep_or_sigma.float().to(device=self.device) * self.num_train_timesteps
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.numel() == 1 and hidden_states.shape[0] > 1:
            timestep = timestep.expand(hidden_states.shape[0])
        elif timestep.ndim != 1 or timestep.shape[0] != hidden_states.shape[0]:
            raise ValueError(f"Bernini timestep shape {tuple(timestep.shape)} does not match latent batch {hidden_states.shape[0]}.")

        context = self._condition_to_context_tensor(condition, batch_size=hidden_states.shape[0])
        source_patches, source_rope_cos, source_rope_sin = self._condition_to_source_inputs(
            condition,
            batch_size=hidden_states.shape[0],
        )
        with self.transformer_forward_context():
            return self.transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=context,
                source_image_vae_patches=source_patches,
                source_image_rope_cos=source_rope_cos,
                source_image_rope_sin=source_rope_sin,
                return_dict=False,
            )[0]

    def _latent_channels(self):
        if self.transformer is not None:
            return int(self.transformer.config.in_channels)
        return int(self.config["model"].get("latent_channels", 16))


@MODEL_REGISTER("bernini_r2v")
@MODEL_REGISTER("bernini_r2v_a14b")
class BerniniR2VA14BModel(BerniniT2VA14BModel):
    """Full-Bernini R2V adapter with cached MLLM, T5, VAE-patch, and RoPE inputs."""

    renderer_task = "r2v"
