import torch.distributed as dist
from loguru import logger

from lightx2v_train.runtime.ddp import apply_ddp, ddp_enabled, set_ddp_gradient_sync
from lightx2v_train.runtime.distributed import is_distributed
from lightx2v_train.runtime.fsdp import apply_fsdp2, fsdp2_enabled


def synchronize_trainable_parameters(model):
    """Make freshly initialized trainable parameters identical on all ranks.

    FSDP shards parameters across its mesh while sequence parallelism keeps a
    replica of each shard.  Randomly initialized adapters must therefore be
    identical *before* sharding; synchronizing gradients afterwards cannot
    repair different LoRA bases.
    """

    if not is_distributed():
        return model

    num_parameters = 0
    for parameter in model.trainable_parameters():
        dist.broadcast(parameter.detach(), src=0)
        num_parameters += parameter.numel()
    logger.info("Synchronized {} freshly initialized trainable parameters from rank 0", num_parameters)
    return model


def apply_parallel(model, config):
    """Apply the configured distributed parallel strategy exactly once."""

    if not is_distributed():
        return model

    use_ddp = ddp_enabled(config)
    use_fsdp2 = fsdp2_enabled(config)
    if use_ddp and use_fsdp2:
        raise RuntimeError("DP(DDP) and FSDP2 cannot both be enabled for the same distributed job.")

    if use_ddp:
        return apply_ddp(model, config)

    if use_fsdp2:
        return apply_fsdp2(model, config)

    logger.warning("Distributed training is initialized, but neither DP(DDP) nor FSDP2 is enabled. The model will run without distributed wrapping.")
    return model


def set_parallel_gradient_sync(model, enabled):
    model.set_fsdp2_gradient_sync(enabled)
    set_ddp_gradient_sync(model.denoiser_module(), enabled)
