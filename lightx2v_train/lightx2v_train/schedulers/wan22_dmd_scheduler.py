"""Wan2.2 A14B high/low-expert scheduler math for DMD training.

Wan2.2-A14B divides the flow trajectory at a (raw) boundary timestep.  The
low-noise expert predicts ``x0`` below the boundary, while the high-noise
expert predicts the latent at the boundary.  In particular, samples used to
train the high fake score are *not* produced by ordinary rectified-flow
interpolation; they use the conditional Gaussian bridge from the Wan2.2 DMD
recipe.

This module keeps the public LightX2V convention: timesteps passed to model
code are normalized, already-warped sigmas in ``[0, 1]``.  Methods whose name
contains ``raw_timestep`` are the only methods which consume the unwarped
Wan timestep convention (0 is clean and ``num_train_timesteps`` is noise).
"""

from typing import Optional, Tuple

import torch

from .dmd_scheduler import DMDFlowMatchingScheduler


class Wan22DMDFlowMatchingScheduler(DMDFlowMatchingScheduler):
    """DMD scheduler implementing the Wan2.2-A14B boundary bridge.

    Args:
        config: The normal LightX2V training config.
        dmd_config: ``training.dmd``.  The relevant entries are
            ``boundary_step`` (default 500), ``training_target`` (or
            ``expert``), and ``timestep_shift`` (defaulting to the scheduler's
            static time-shift value).

    All sample/endpoint methods support a scalar sigma, one sigma per batch
    item (``[B]``), or one sigma per latent frame (``[B, F]`` for LightX2V
    latents laid out as ``[B, C, F, H, W]``).  Computation involving fp16 or
    bf16 tensors is promoted to fp32 and the result is cast back to the input
    tensor's dtype.
    """

    _HIGH_EXPERT_NAMES = frozenset(("high", "high_noise", "high_noise_model"))
    _LOW_EXPERT_NAMES = frozenset(("low", "low_noise", "low_noise_model"))

    def __init__(self, config, dmd_config=None):
        dmd_config = {} if dmd_config is None else dmd_config
        super().__init__(config, dmd_config)

        self.boundary_step = int(dmd_config.get("boundary_step", self.num_train_timesteps // 2))
        if not 0 < self.boundary_step < self.num_train_timesteps:
            raise ValueError(
                "training.dmd.boundary_step must be strictly between 0 and "
                f"num_train_timesteps={self.num_train_timesteps}, got {self.boundary_step}."
            )

        time_shift_settings = config.get("scheduler", {}).get("time_shift_settings", {}) or {}
        default_shift = time_shift_settings.get("time_shift_mu", 1.0) if time_shift_settings.get("do_time_shift", False) else 1.0
        self.timestep_shift = float(dmd_config.get("timestep_shift", default_shift))
        if self.timestep_shift <= 0:
            raise ValueError(f"training.dmd.timestep_shift must be positive, got {self.timestep_shift}.")

        target = dmd_config.get("training_target", dmd_config.get("expert", "high_noise"))
        self.training_target = self.normalize_expert_name(target)

        # Store this as a Python float so moving a scheduler between devices
        # never leaves a stale device-bound scalar behind.
        boundary_raw_sigma = self.boundary_step / self.num_train_timesteps
        self.boundary_sigma = float(self._linear_shift_value(boundary_raw_sigma))

        # Self-Forcing-Plus samples the inner 4%-96% of each expert interval.
        self.expert_min_ratio = float(dmd_config.get("expert_min_step_ratio", 0.04))
        self.expert_max_ratio = float(dmd_config.get("expert_max_step_ratio", 0.96))
        if not 0.0 <= self.expert_min_ratio < self.expert_max_ratio <= 1.0:
            raise ValueError(
                "Wan2.2 expert timestep ratios must satisfy "
                f"0 <= min < max <= 1, got {self.expert_min_ratio}, {self.expert_max_ratio}."
            )

    @classmethod
    def normalize_expert_name(cls, expert: str) -> str:
        expert = str(expert).lower()
        if expert in cls._HIGH_EXPERT_NAMES:
            return "high_noise"
        if expert in cls._LOW_EXPERT_NAMES:
            return "low_noise"
        raise ValueError(f"Unsupported Wan2.2 expert {expert!r}; expected 'high_noise' or 'low_noise'.")

    def _linear_shift_value(self, raw_sigma):
        return self.timestep_shift * raw_sigma / (1.0 + (self.timestep_shift - 1.0) * raw_sigma)

    def raw_timestep_to_sigma(self, raw_timestep, *, device=None, dtype=torch.float32):
        """Convert an unwarped Wan timestep to LightX2V's warped sigma."""

        raw_timestep = torch.as_tensor(raw_timestep, device=device, dtype=torch.float32)
        raw_sigma = raw_timestep / float(self.num_train_timesteps)
        sigma = self._linear_shift_value(raw_sigma)
        return sigma.to(dtype=dtype) if dtype is not None else sigma

    def sigma_to_raw_timestep(self, sigma, *, dtype=torch.float32):
        """Invert :meth:`raw_timestep_to_sigma` (useful for logging/windows)."""

        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=sigma.device if isinstance(sigma, torch.Tensor) else None)
        denominator = self.timestep_shift - (self.timestep_shift - 1.0) * sigma
        raw_sigma = sigma / denominator
        raw_timestep = raw_sigma * float(self.num_train_timesteps)
        return raw_timestep.to(dtype=dtype) if dtype is not None else raw_timestep

    def expert_raw_timestep_bounds(self, expert: Optional[str] = None) -> Tuple[int, int]:
        """Return the inclusive-low/exclusive-high integer sampling bounds."""

        expert = self.normalize_expert_name(expert or self.training_target)
        if expert == "high_noise":
            interval = self.num_train_timesteps - self.boundary_step
            lower = int(self.boundary_step + interval * self.expert_min_ratio)
            upper = int(self.boundary_step + interval * self.expert_max_ratio)
        else:
            interval = self.boundary_step
            lower = int(interval * self.expert_min_ratio)
            upper = int(interval * self.expert_max_ratio)
        return lower, upper

    def sample_expert_sigma(
        self,
        batch_size: int,
        *,
        expert: Optional[str] = None,
        device=None,
        dtype=None,
        generator=None,
    ):
        """Sample the expert-local score timestep exactly as the reference.

        Integer raw timesteps are sampled uniformly from ``[lower, upper)``
        and then warped once.  The returned tensor has shape ``[B]``.
        """

        lower, upper = self.expert_raw_timestep_bounds(expert)
        device = self.device if device is None else device
        raw_timestep = torch.randint(lower, upper, (int(batch_size),), device=device, generator=generator)
        return self.raw_timestep_to_sigma(raw_timestep, dtype=dtype or self.running_dtype)

    @staticmethod
    def _calculation_dtype(tensor: torch.Tensor):
        if tensor.dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return tensor.dtype

    @staticmethod
    def _expand_values(values, reference: torch.Tensor, *, dtype=None):
        """Broadcast scalar/[B]/[B,F] values over a LightX2V latent."""

        values = torch.as_tensor(values, device=reference.device, dtype=dtype)
        if values.ndim == 0:
            return values.reshape(*([1] * reference.ndim))
        if values.ndim == reference.ndim:
            return values
        if values.ndim == 1:
            if values.shape[0] not in (1, reference.shape[0]):
                raise ValueError(f"Sigma batch dimension {values.shape[0]} does not match latent batch {reference.shape[0]}.")
            return values.reshape(values.shape[0], *([1] * (reference.ndim - 1)))
        if values.ndim == 2 and reference.ndim >= 3:
            if values.shape[0] not in (1, reference.shape[0]) or values.shape[1] not in (1, reference.shape[2]):
                raise ValueError(
                    f"Per-frame sigma shape {tuple(values.shape)} is incompatible with LightX2V latent shape {tuple(reference.shape)}."
                )
            return values.reshape(values.shape[0], 1, values.shape[1], *([1] * (reference.ndim - 3)))
        raise ValueError(f"Cannot broadcast sigma shape {tuple(values.shape)} to latent shape {tuple(reference.shape)}.")

    def boundary_sigma_like(self, reference: torch.Tensor, sigma_bound=None, *, dtype=None):
        sigma_bound = self.boundary_sigma if sigma_bound is None else sigma_bound
        return self._expand_values(sigma_bound, reference, dtype=dtype)

    @staticmethod
    def calculate_alpha_beta_high(sigma, sigma_bound):
        """Conditional Gaussian bridge coefficients for the high expert.

        This is the Wan2.2 reference formula::

            alpha = (1 - sigma) / (1 - sigma_bound)
            beta  = sqrt(sigma**2 - (alpha * sigma_bound)**2)

        The radicand clamp only suppresses tiny negative round-off at the
        boundary; valid high-expert sigmas satisfy ``sigma >= sigma_bound``.
        """

        alpha = (1.0 - sigma) / (1.0 - sigma_bound)
        beta_squared = sigma.square() - (alpha * sigma_bound).square()
        beta = torch.sqrt(torch.clamp_min(beta_squared, 0.0))
        return alpha, beta

    @staticmethod
    def calculate_alpha_beta_low(sigma, sigma_bound):
        """Reference low-interval coefficients (kept for exact parity)."""

        beta = sigma / sigma_bound
        alpha = 1.0 - beta
        return alpha, beta

    def add_noise_high(self, boundary_latent, noise, sigmas, sigma_bound=None):
        """Corrupt a boundary latent with the Wan2.2 high Gaussian bridge."""

        output_dtype = boundary_latent.dtype
        calculation_dtype = self._calculation_dtype(boundary_latent)
        boundary_latent = boundary_latent.to(dtype=calculation_dtype)
        noise = noise.to(device=boundary_latent.device, dtype=calculation_dtype)
        sigma = self._expand_values(sigmas, boundary_latent, dtype=calculation_dtype)
        bound = self.boundary_sigma_like(boundary_latent, sigma_bound, dtype=calculation_dtype)
        alpha, beta = self.calculate_alpha_beta_high(sigma, bound)
        return (alpha * boundary_latent + beta * noise).to(dtype=output_dtype)

    def add_noise_low(self, clean_latent, noise, sigmas, sigma_bound=None):
        """Corrupt an x0 latent for the low expert.

        ``sigma_bound`` is accepted for a symmetric call signature, but the
        reference implementation uses ordinary rectified-flow interpolation
        in the low interval.
        """

        del sigma_bound
        output_dtype = clean_latent.dtype
        calculation_dtype = self._calculation_dtype(clean_latent)
        clean_latent = clean_latent.to(dtype=calculation_dtype)
        noise = noise.to(device=clean_latent.device, dtype=calculation_dtype)
        sigma = self._expand_values(sigmas, clean_latent, dtype=calculation_dtype)
        return ((1.0 - sigma) * clean_latent + sigma * noise).to(dtype=output_dtype)

    def add_noise_for_expert(self, endpoint, noise, sigmas, expert: Optional[str] = None):
        expert = self.normalize_expert_name(expert or self.training_target)
        if expert == "high_noise":
            return self.add_noise_high(endpoint, noise, sigmas)
        return self.add_noise_low(endpoint, noise, sigmas)

    def flow_to_high_endpoint(self, flow_pred, noisy_latent, sigmas, sigma_bound=None):
        """Convert Wan flow output to the predicted boundary latent.

        This conversion is used for the real/fake score predictions in the DMD
        gradient and for generator rollout: ``x_bound = x_t - (t-s) v``.
        """

        output_dtype = flow_pred.dtype
        calculation_dtype = self._calculation_dtype(flow_pred)
        flow = flow_pred.to(dtype=calculation_dtype)
        sample = noisy_latent.to(device=flow.device, dtype=calculation_dtype)
        sigma = self._expand_values(sigmas, sample, dtype=calculation_dtype)
        bound = self.boundary_sigma_like(sample, sigma_bound, dtype=calculation_dtype)
        return (sample - (sigma - bound) * flow).to(dtype=output_dtype)

    def high_endpoint_to_flow(self, boundary_latent, noisy_latent, sigmas, sigma_bound=None):
        """Inverse of :meth:`flow_to_high_endpoint`."""

        output_dtype = boundary_latent.dtype
        calculation_dtype = self._calculation_dtype(boundary_latent)
        endpoint = boundary_latent.to(dtype=calculation_dtype)
        sample = noisy_latent.to(device=endpoint.device, dtype=calculation_dtype)
        sigma = self._expand_values(sigmas, sample, dtype=calculation_dtype)
        bound = self.boundary_sigma_like(sample, sigma_bound, dtype=calculation_dtype)
        return ((sample - endpoint) / (sigma - bound)).to(dtype=output_dtype)

    def flow_to_low_endpoint(self, flow_pred, noisy_latent, sigmas, *, output_dtype=None):
        """Convert Wan flow output to x0: ``x0 = x_t - t v``."""

        output_dtype = flow_pred.dtype if output_dtype is None else output_dtype
        calculation_dtype = self._calculation_dtype(flow_pred)
        flow = flow_pred.to(dtype=calculation_dtype)
        sample = noisy_latent.to(device=flow.device, dtype=calculation_dtype)
        sigma = self._expand_values(sigmas, sample, dtype=calculation_dtype)
        return (sample - sigma * flow).to(dtype=output_dtype)

    def low_endpoint_to_flow(self, clean_latent, noisy_latent, sigmas):
        """Inverse of :meth:`flow_to_low_endpoint`."""

        output_dtype = clean_latent.dtype
        calculation_dtype = self._calculation_dtype(clean_latent)
        endpoint = clean_latent.to(dtype=calculation_dtype)
        sample = noisy_latent.to(device=endpoint.device, dtype=calculation_dtype)
        sigma = self._expand_values(sigmas, sample, dtype=calculation_dtype)
        return ((sample - endpoint) / sigma).to(dtype=output_dtype)

    def flow_to_endpoint(self, flow_pred, noisy_latent, sigmas, expert: Optional[str] = None):
        expert = self.normalize_expert_name(expert or self.training_target)
        if expert == "high_noise":
            return self.flow_to_high_endpoint(flow_pred, noisy_latent, sigmas)
        return self.flow_to_low_endpoint(flow_pred, noisy_latent, sigmas)

    def flow_to_high_bridge_endpoint(self, flow_pred, noisy_latent, sigmas, sigma_bound=None, *, output_dtype=None):
        """Recover the high endpoint for the fake-score denoising loss.

        Unlike :meth:`flow_to_high_endpoint`, this is the inverse associated
        with :meth:`add_noise_high`.  It exactly follows the expression in
        ``Self-Forcing-Plus``'s Wan2.2 ``critic_loss``.
        """

        output_dtype = flow_pred.dtype if output_dtype is None else output_dtype
        calculation_dtype = self._calculation_dtype(flow_pred)
        flow = flow_pred.to(dtype=calculation_dtype)
        sample = noisy_latent.to(device=flow.device, dtype=calculation_dtype)
        sigma = self._expand_values(sigmas, sample, dtype=calculation_dtype)
        bound = self.boundary_sigma_like(sample, sigma_bound, dtype=calculation_dtype)
        alpha, beta = self.calculate_alpha_beta_high(sigma, bound)
        beta_squared = beta.square()

        numerator = (1.0 - bound) * (sigma - beta_squared) * sample
        numerator = numerator - (1.0 - bound) * (1.0 - sigma) * beta_squared * flow
        denominator = (1.0 - sigma) * beta_squared + (1.0 - bound) * (sigma - beta_squared) * alpha
        return (numerator / denominator).to(dtype=output_dtype)

    def flow_to_training_endpoint(self, flow_pred, noisy_latent, sigmas, expert: Optional[str] = None):
        """Endpoint conversion used by the trainable fake-score loss."""

        expert = self.normalize_expert_name(expert or self.training_target)
        if expert == "high_noise":
            return self.flow_to_high_bridge_endpoint(
                flow_pred,
                noisy_latent,
                sigmas,
                output_dtype=torch.float32,
            )
        return self.flow_to_low_endpoint(
            flow_pred,
            noisy_latent,
            sigmas,
            output_dtype=torch.float32,
        )

    def flow_to_training_target(self, flow_pred, noisy_latent, sigmas, expert: Optional[str] = None):
        """Alias with the terminology used by the fake-score training loop."""

        return self.flow_to_training_endpoint(flow_pred, noisy_latent, sigmas, expert=expert)
