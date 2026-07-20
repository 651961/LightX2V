import os

import torch
from loguru import logger

from lightx2v_train.runtime.distributed import get_sequence_parallel_world_size
from lightx2v_train.utils.registry import MODEL_REGISTER
from lightx2v_train.utils.utils import get_running_dtype

from .native.wan.modules.t5 import T5EncoderModel
from .native.wan.modules.vae import WanVAE
from .wan_t2v import WanT2VModel


@MODEL_REGISTER("wan2_2_t2v_a14b")
class Wan2_2T2VA14BModel(WanT2VModel):
    """Trainable adapter for one Wan2.2 T2V A14B noise expert.

    Wan2.2 A14B stores the high- and low-noise DiTs in separate Diffusers
    directories while the VAE and T5 assets live at the model root.  A DMD
    role therefore selects exactly one expert with ``model.expert`` and points
    ``model.pretrained_model_name_or_path`` at that expert directory.  The
    shared assets are resolved from ``model.base_model_path`` or from explicit
    component paths.

    The denoiser itself is the regular Wan T2V ``WanModel`` inherited from
    :class:`WanT2VModel`.  Consequently LoRA injection, FSDP2 block sharding,
    gradient checkpointing, and the sequence-parallel attention path are
    shared with Wan2.1 instead of being reimplemented here.
    """

    _EXPERT_ALIASES = {
        "high": "high",
        "high_noise": "high",
        "high_noise_model": "high",
        "low": "low",
        "low_noise": "low",
        "low_noise_model": "low",
    }
    _EXPERT_DIR_NAMES = {
        "high": "high_noise_model",
        "low": "low_noise_model",
    }

    def load_components(self, transformer_only=False, reference_model=None):
        model_config = self.config["model"]
        transformer_path = os.fspath(model_config["pretrained_model_name_or_path"])

        self.expert = self._resolve_expert(model_config, transformer_path)
        self.expert_dir_name = self._EXPERT_DIR_NAMES[self.expert]
        self.transformer_path = transformer_path
        self.base_model_path = self._resolve_base_model_path(model_config, transformer_path)

        self.load_vae = bool(model_config.get("load_vae", True))
        self.load_text_encoder = bool(model_config.get("load_text_encoder", True))
        self.load_transformer = bool(model_config.get("load_transformer", True))
        if model_config.get("causal", False):
            raise ValueError("Wan2.2 T2V A14B DMD uses a bidirectional high/low expert; model.causal=true is not supported by this adapter.")
        self.use_causal_transformer = False

        self.sample_posterior = bool(model_config.get("sample_posterior", True))
        scheduler_config = self.config.get("scheduler", {})
        self.num_train_timesteps = int(scheduler_config.get("num_train_timesteps", 1000))
        self.max_sequence_length = int(model_config.get("max_sequence_length", 512))
        self.transformer_param_dtype = get_running_dtype(model_config.get("transformer_param_dtype", "fp32"))
        self.vae_dtype = get_running_dtype(model_config.get("vae_dtype", "fp32"))
        self.t5_dtype = get_running_dtype(model_config.get("t5_dtype", "bf16"))
        self.t5_cpu = bool(model_config.get("t5_cpu", False))
        self.vae_stride = tuple(model_config.get("vae_stride", (4, 8, 8)))
        self.patch_size = tuple(model_config.get("patch_size", (1, 2, 2)))
        self.sp_size = get_sequence_parallel_world_size()

        # These attributes are consumed by inherited Wan helpers.  A14B DMD is
        # bidirectional, so the cache-related values are intentionally inert.
        self.num_frame_per_chunk = int(model_config.get("num_frame_per_chunk", 1))
        self.local_attn_size = int(model_config.get("local_attn_size", -1))
        self.sink_size = int(model_config.get("sink_size", 0))
        self.defer_kv_cache_updates = False
        self.detach_kv_cache_updates = False
        self.independent_first_frame = False
        # Keep the inherited cached-latent path valid when DMD deliberately
        # skips loading the VAE.  Transformer-only fake/teacher roles share
        # this attribute from the student model below.
        self.vae = None
        self.text_encoder = None
        self.text_pipeline = None

        if transformer_only:
            if reference_model is not None:
                self._share_frozen_components(reference_model)
            self.transformer = self._load_transformer(self.transformer_path) if self.load_transformer else None
            if self.transformer is not None:
                self._configure_transformer()
            self._set_vae_scale_factors()
            logger.info("[model] Wan2.2 A14B expert={} transformer={}", self.expert, self.transformer_path)
            return

        self.transformer = None
        if self.load_transformer:
            self.transformer = self._load_transformer(self.transformer_path)
            self._configure_transformer()

        if self.load_vae:
            vae_checkpoint = self._resolve_component_path(
                model_config,
                keys=("vae_checkpoint", "vae_path"),
                default_name="Wan2.1_VAE.pth",
            )
            self.vae = WanVAE(vae_pth=vae_checkpoint, dtype=self.vae_dtype, device=self.device)
            self.vae.model.requires_grad_(False)

        if self.load_text_encoder:
            t5_checkpoint = self._resolve_component_path(
                model_config,
                keys=("t5_checkpoint", "text_encoder_checkpoint"),
                default_name="models_t5_umt5-xxl-enc-bf16.pth",
            )
            t5_tokenizer = self._resolve_component_path(
                model_config,
                keys=("t5_tokenizer", "tokenizer_path"),
                default_name=os.path.join("google", "umt5-xxl"),
            )
            self.text_encoder = T5EncoderModel(
                text_len=self.max_sequence_length,
                dtype=self.t5_dtype,
                device=torch.device("cpu"),
                checkpoint_path=t5_checkpoint,
                tokenizer_path=t5_tokenizer,
            )
            self.text_encoder.model.requires_grad_(False)
            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)

        self._set_vae_scale_factors()
        logger.info(
            "[model] Wan2.2 A14B expert={} transformer={} shared_components={}",
            self.expert,
            self.transformer_path,
            self.base_model_path,
        )

    @classmethod
    def _resolve_expert(cls, model_config, transformer_path):
        configured = model_config.get("expert", model_config.get("training_target"))
        path_name = os.path.basename(os.path.normpath(transformer_path))
        inferred = cls._EXPERT_ALIASES.get(path_name)

        if configured is None:
            if inferred is None:
                valid = ", ".join(sorted(cls._EXPERT_DIR_NAMES))
                raise ValueError(
                    "Wan2.2 A14B requires model.expert to select the trainable noise expert "
                    f"({valid}); it could not be inferred from transformer path {transformer_path!r}."
                )
            return inferred

        key = str(configured).strip().lower()
        if key not in cls._EXPERT_ALIASES:
            valid = ", ".join(sorted(cls._EXPERT_ALIASES))
            raise ValueError(f"Unsupported Wan2.2 A14B model.expert={configured!r}; expected one of: {valid}.")
        expert = cls._EXPERT_ALIASES[key]
        if inferred is not None and inferred != expert:
            raise ValueError(
                f"model.expert={configured!r} selects {expert!r}, but pretrained_model_name_or_path "
                f"points at the {inferred!r} expert directory: {transformer_path}"
            )
        return expert

    @classmethod
    def _resolve_base_model_path(cls, model_config, transformer_path):
        explicit = model_config.get("base_model_path", model_config.get("shared_components_path"))
        if explicit is not None:
            return os.fspath(explicit)

        path_name = os.path.basename(os.path.normpath(transformer_path))
        if path_name in cls._EXPERT_ALIASES:
            return os.path.dirname(os.path.normpath(transformer_path))
        return transformer_path

    def _resolve_component_path(self, model_config, keys, default_name):
        value = None
        for key in keys:
            if model_config.get(key) is not None:
                value = os.fspath(model_config[key])
                break
        if value is None:
            return os.path.join(self.base_model_path, default_name)
        if os.path.isabs(value):
            return value
        return os.path.join(self.base_model_path, value)

    def _share_frozen_components(self, reference_model):
        self.vae = reference_model.vae
        self.text_encoder = reference_model.text_encoder
        self.text_pipeline = reference_model.text_pipeline
        self.vae_stride = reference_model.vae_stride
        self.patch_size = reference_model.patch_size
        self.max_sequence_length = reference_model.max_sequence_length
        self.vae_scale_factor_temporal = reference_model.vae_scale_factor_temporal
        self.vae_scale_factor_spatial = reference_model.vae_scale_factor_spatial

    def _set_vae_scale_factors(self):
        self.vae_scale_factor_temporal = self.vae_stride[0]
        self.vae_scale_factor_spatial = self.vae_stride[1]
