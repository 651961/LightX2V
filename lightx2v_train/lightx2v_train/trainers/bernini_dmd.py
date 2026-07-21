"""T2V and R2V DMD training for full Bernini's high/low experts.

The MLLM planner, connector, and T5 encoder are deliberately absent from the
training process.  Their four renderer contexts are cached offline.  Student
rollout and fake-score training use the complete text+planned-ViT context,
while the frozen teacher reproduces Bernini's text and planned-ViT APG chain.
"""

import torch

from lightx2v_train.runtime.sequence_parallel import broadcast_sequence_parallel_value
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .wan22_dmd import Wan22A14BDmdTrainer


@TRAINER_REGISTER("bernini_dmd")
class BerniniDmdTrainer(Wan22A14BDmdTrainer):
    """Expert-aware Bernini DMD with fully offline planner conditioning."""

    trainer_name = "bernini_dmd"
    allowed_model_names = {"bernini_t2v", "bernini_t2v_a14b"}
    default_lora_target_modules = (
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "ffn.net.0.proj",
        "ffn.net.2",
    )

    _CONTEXT_KEYS = (
        "cond_embeds_wtxt_wvit",
        "cond_embeds_wtxt_wovit",
        "cond_embeds_wotxt_wvit",
        "cond_embeds_wotxt_wovit",
    )
    _FULL_CONTEXT = "cond_embeds_wtxt_wvit"
    _TEXT_ONLY_CONTEXT = "cond_embeds_wtxt_wovit"
    _BASE_CONTEXT = "cond_embeds_wotxt_wovit"

    def __init__(self, config):
        super().__init__(config)
        teacher_config = self.training_config.get("teacher", {})
        self.omega_txt = float(teacher_config.get("omega_txt", self.dmd_config.get("omega_txt", 4.0)))
        self.omega_tgt = float(teacher_config.get("omega_tgt", self.dmd_config.get("omega_tgt", 0.5)))
        if self.omega_txt < 0 or self.omega_tgt < 0:
            raise ValueError(f"Bernini teacher guidance requires non-negative omega_txt and omega_tgt, got omega_txt={self.omega_txt} and omega_tgt={self.omega_tgt}.")

    def _encode_conditions(self, sample):
        conditioning = sample["conditioning"]
        cached = conditioning.get("positive", conditioning)
        if not isinstance(cached, dict):
            raise TypeError("Bernini DMD requires an offline condition mapping containing all four cond_embeds_* tensors.")

        missing = [key for key in self._CONTEXT_KEYS if key not in cached]
        if missing:
            raise KeyError(f"Bernini DMD condition cache is missing: {', '.join(missing)}. Cache MLLM, connector, and T5 outputs before training.")

        contexts = {key: self._prepare_cached_context(cached[key], key) for key in self._CONTEXT_KEYS}
        batch_sizes = {condition["prompt_embed"].shape[0] for condition in contexts.values()}
        if len(batch_sizes) != 1:
            shapes = {key: tuple(value["prompt_embed"].shape) for key, value in contexts.items()}
            raise ValueError(f"Bernini cached contexts must have the same batch size, got {shapes}.")

        contexts = broadcast_sequence_parallel_value(contexts)
        return contexts[self._FULL_CONTEXT], contexts

    def _prepare_cached_context(self, value, key):
        if isinstance(value, dict):
            if "prompt_embed" not in value:
                raise KeyError(f"Bernini cached context {key!r} must be a tensor or contain prompt_embed.")
            value = value["prompt_embed"]
        if not torch.is_tensor(value):
            raise TypeError(f"Bernini cached context {key!r} must be a tensor, got {type(value)!r}.")

        prompt_embed = value.to(device=self.model.device, dtype=self.running_dtype)
        if prompt_embed.ndim == 2:
            prompt_embed = prompt_embed.unsqueeze(0)
        if prompt_embed.ndim != 3 or prompt_embed.shape[-1] != 4096:
            raise ValueError(f"Bernini cached context {key!r} must have shape [batch, sequence, 4096], got {tuple(prompt_embed.shape)}.")
        return {"prompt_embed": prompt_embed}

    @staticmethod
    def _apg_delta(delta, ref, parallel_scale=0.2, orthogonal_scale=1.0, eps=1e-8):
        """Project a guidance delta exactly as Bernini's renderer sampler does."""
        batch_size = delta.shape[0]
        delta_flat = delta.reshape(batch_size, -1)
        ref_flat = ref.reshape(batch_size, -1)
        ref_norm_sq = (ref_flat * ref_flat).sum(dim=1, keepdim=True).clamp_min(eps)
        projection = (delta_flat * ref_flat).sum(dim=1, keepdim=True) / ref_norm_sq
        delta_parallel = projection * ref_flat
        delta_orthogonal = delta_flat - delta_parallel
        return parallel_scale * delta_parallel.reshape_as(delta) + orthogonal_scale * delta_orthogonal.reshape_as(delta)

    def _predict_teacher_velocity(self, latents, sigma, condition, teacher_condition):
        if not isinstance(teacher_condition, dict):
            raise TypeError("Bernini teacher conditioning must contain the four cached cond_embeds_* contexts.")

        base = self._predict_velocity(self.teacher_model, latents, sigma, teacher_condition[self._BASE_CONTEXT])
        text_only = self._predict_velocity(
            self.teacher_model,
            latents,
            sigma,
            teacher_condition[self._TEXT_ONLY_CONTEXT],
        )
        full = self._predict_velocity(self.teacher_model, latents, sigma, teacher_condition[self._FULL_CONTEXT])

        text_delta = self._apg_delta(text_only - base, ref=text_only)
        target_delta = self._apg_delta(full - text_only, ref=full)
        return base + self.omega_txt * text_delta + self.omega_tgt * target_delta


@TRAINER_REGISTER("bernini_r2v_dmd")
class BerniniR2VDmdTrainer(BerniniDmdTrainer):
    """Bernini R2V DMD with cached reference-image VAE conditioning."""

    trainer_name = "bernini_r2v_dmd"
    allowed_model_names = {"bernini_r2v", "bernini_r2v_a14b"}

    _SOURCE_KEYS = (
        "source_image_vae_patches",
        "source_image_rope_cos",
        "source_image_rope_sin",
    )

    def __init__(self, config):
        super().__init__(config)
        teacher_config = self.training_config.get("teacher", {})
        self.omega_img = float(teacher_config.get("omega_img", 4.5))
        self.omega_txt = float(teacher_config.get("omega_txt", 4.0))
        self.omega_tgt = float(teacher_config.get("omega_tgt", 1.5))
        self.omega_scale = float(teacher_config.get("omega_scale", 0.8))
        if any(value < 0 for value in (self.omega_img, self.omega_txt, self.omega_tgt, self.omega_scale)):
            raise ValueError(
                "Bernini R2V teacher guidance requires non-negative omega_img, "
                f"omega_txt, omega_tgt, and omega_scale, got {self.omega_img}, "
                f"{self.omega_txt}, {self.omega_tgt}, and {self.omega_scale}."
            )

        # Full Bernini applies omega_scale exactly once when inference switches
        # from the high-noise expert to the low-noise expert.
        if self.expert == "low_noise":
            self.omega_img *= self.omega_scale
            self.omega_txt *= self.omega_scale
            self.omega_tgt *= self.omega_scale

    def _encode_conditions(self, sample):
        conditioning = sample["conditioning"]
        cached = conditioning.get("positive", conditioning)
        if not isinstance(cached, dict):
            raise TypeError("Bernini R2V DMD requires an offline condition mapping.")

        required = self._CONTEXT_KEYS + self._SOURCE_KEYS
        missing = [key for key in required if key not in cached]
        if missing:
            raise KeyError(f"Bernini R2V DMD condition cache is missing: {', '.join(missing)}. Cache MLLM/T5 contexts and reference-image VAE conditioning before training.")

        prepared = {key: self._prepare_cached_context(cached[key], key) for key in self._CONTEXT_KEYS}
        batch_sizes = {condition["prompt_embed"].shape[0] for condition in prepared.values()}
        if len(batch_sizes) != 1:
            shapes = {key: tuple(value["prompt_embed"].shape) for key, value in prepared.items()}
            raise ValueError(f"Bernini cached contexts must have the same batch size, got {shapes}.")
        batch_size = batch_sizes.pop()

        for key in self._SOURCE_KEYS:
            prepared[key] = self._prepare_cached_source(cached[key], key, batch_size)
        source_counts = {key: len(prepared[key]) for key in self._SOURCE_KEYS}
        if len(set(source_counts.values())) != 1:
            raise ValueError(f"Bernini cached source condition lists must have the same length, got {source_counts}.")

        prepared = broadcast_sequence_parallel_value(prepared)
        full_condition = self._condition_with_sources(prepared[self._FULL_CONTEXT], prepared)
        return full_condition, prepared

    def _prepare_cached_source(self, value, key, batch_size):
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"Bernini cached source condition {key!r} must be a list of per-image tensors.")
        if not value:
            raise ValueError(f"Bernini cached source condition {key!r} must contain at least one reference image.")
        return [self._prepare_cached_source_tensor(item, key, batch_size) for item in value]

    def _prepare_cached_source_tensor(self, value, key, batch_size):
        if not torch.is_tensor(value):
            raise TypeError(f"Bernini cached source condition {key!r} must contain tensors, got {type(value)!r}.")

        dtype = self.running_dtype if key == "source_image_vae_patches" else None
        value = value.to(device=self.model.device, dtype=dtype)
        expected_ndim = 6 if key == "source_image_vae_patches" else 4
        if value.ndim == expected_ndim - 1:
            value = value.unsqueeze(0)
        if value.ndim != expected_ndim:
            raise ValueError(f"Bernini cached source condition {key!r} must have {expected_ndim} dimensions including batch, got shape {tuple(value.shape)}.")
        if value.shape[0] != batch_size:
            raise ValueError(f"Bernini cached source condition {key!r} batch size {value.shape[0]} does not match context batch size {batch_size}.")
        return value

    def _condition_with_sources(self, context, cached):
        return {
            "prompt_embed": context["prompt_embed"],
            **{key: cached[key] for key in self._SOURCE_KEYS},
        }

    def _predict_teacher_velocity(self, latents, sigma, condition, teacher_condition):
        if not isinstance(teacher_condition, dict):
            raise TypeError("Bernini R2V teacher conditioning must contain cached contexts and source-image VAE conditioning.")

        base_condition = teacher_condition[self._BASE_CONTEXT]
        image_condition = self._condition_with_sources(base_condition, teacher_condition)
        text_condition = self._condition_with_sources(teacher_condition[self._TEXT_ONLY_CONTEXT], teacher_condition)
        full_condition = self._condition_with_sources(teacher_condition[self._FULL_CONTEXT], teacher_condition)

        base = self._predict_velocity(self.teacher_model, latents, sigma, base_condition)
        image = self._predict_velocity(self.teacher_model, latents, sigma, image_condition)
        text = self._predict_velocity(self.teacher_model, latents, sigma, text_condition)
        full = self._predict_velocity(self.teacher_model, latents, sigma, full_condition)

        image_delta = self._apg_delta(image - base, ref=image)
        text_delta = self._apg_delta(text - image, ref=text)
        target_delta = self._apg_delta(full - text, ref=full)
        return base + self.omega_img * image_delta + self.omega_txt * text_delta + self.omega_tgt * target_delta
