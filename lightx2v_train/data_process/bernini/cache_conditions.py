#!/usr/bin/env python3
"""Cache full-Bernini planner/T5 conditions for T2V or R2V DMD.

This always skips both 14B Wan experts.  T2V remains a planner-only path and
does not construct the VAE.  R2V additionally loads the frozen VAE while
preprocessing reference images, then stores the normalized source patches and
source-id RoPE needed by the renderer; training itself still does not load a
VAE.

Each output condition contains the exact four routes consumed by Bernini's
WAPG renderer::

    cond_embeds_wtxt_wvit
    cond_embeds_wtxt_wovit
    cond_embeds_wotxt_wvit
    cond_embeds_wotxt_wovit

The resulting directory is directly readable by LightX2V ``latent_dataset``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

CONDITION_KEYS = (
    "cond_embeds_wtxt_wvit",
    "cond_embeds_wtxt_wovit",
    "cond_embeds_wotxt_wvit",
    "cond_embeds_wotxt_wovit",
)
SOURCE_CONDITION_KEYS = (
    "source_image_vae_patches",
    "source_image_rope_cos",
    "source_image_rope_sin",
)
SAVE_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


@dataclass(frozen=True)
class PromptEntry:
    row_index: int
    prompt: str
    uid: str
    reference_images: tuple["ReferenceImage", ...] = ()


@dataclass(frozen=True)
class ReferenceImage:
    path: str
    size: int
    mtime_ns: int
    fingerprint: str

    def metadata(self):
        return {
            "path": self.path,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "fingerprint": self.fingerprint,
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="TXT, JSON, JSONL, or CSV prompt metadata")
    parser.add_argument("--output-dir", required=True, help="Output latent_dataset directory")
    parser.add_argument("--bernini-model", required=True, help="Full ByteDance/Bernini-Diffusers directory")
    parser.add_argument(
        "--bernini-repo",
        default=None,
        help="Path to the full Bernini source checkout when the bernini package is not installed",
    )
    parser.add_argument("--prompt-column", default="caption")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--prompt-index", type=int, default=0, help="Prompt field index for list-like JSON rows")
    parser.add_argument("--task", choices=("t2v", "r2v"), default="t2v")
    parser.add_argument(
        "--reference-images-column",
        default="ref_images",
        help="R2V reference-image list column; falls back to 'images'",
    )
    parser.add_argument(
        "--media-root",
        default=None,
        help="Resolve relative R2V reference paths against this directory (default: input metadata directory)",
    )
    parser.add_argument("--negative-prompt", default=None, help="Default: Bernini's official Wan2.2 negative prompt")
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Default: full Bernini's official task-specific system prompt; pass an empty string to disable it",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=842,
        help="Full-Bernini resize limit; used by the R2V reference VAE and recorded for T2V",
    )
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--vit-min-pixels", type=int, default=3136)
    parser.add_argument("--vit-max-pixels", type=int, default=50176)
    parser.add_argument("--planning-step", type=int, default=25)
    parser.add_argument("--vit-denoising-step", type=int, default=5)
    parser.add_argument("--vit-txt-cfg", type=float, default=1.2)
    parser.add_argument("--vit-img-cfg", type=float, default=1.0)
    parser.add_argument(
        "--interpolate-src-id",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Interpolate more than max-trained-src-id reference source IDs into the trained range",
    )
    parser.add_argument("--max-trained-src-id", type=int, default=5)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument(
        "--truncate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Truncate final contexts above max length (official full-Bernini inference default: false)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Fixed per-prompt planning seed")
    parser.add_argument("--save-dtype", choices=tuple(SAVE_DTYPES), default="bf16")
    parser.add_argument("--device", default="cuda", help="Planner/T5 compute device; cuda maps to LOCAL_RANK")
    parser.add_argument(
        "--keep-models-on-device",
        action="store_true",
        help="Keep both MLLM and T5 resident together; faster but requires substantially more VRAM",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate metadata and print the target template without loading models")
    return parser.parse_args()


def prompt_from_row(row, prompt_column: str, prompt_index: int):
    if isinstance(row, str):
        return row
    if isinstance(row, MappingLike):
        value = row.get(prompt_column)
        if value is None:
            for fallback in ("prompt", "text", "caption"):
                if row.get(fallback) is not None:
                    value = row[fallback]
                    break
        return value
    if isinstance(row, (list, tuple)):
        return row[prompt_index]
    return None


class MappingLike(dict):
    """Marker used only to make prompt_from_row's accepted mapping explicit."""


def normalize_row(row):
    if isinstance(row, dict) and not isinstance(row, MappingLike):
        return MappingLike(row)
    return row


def reference_paths_from_row(row, column: str):
    if not isinstance(row, MappingLike):
        return []
    value = row.get(column)
    if value is None and column != "ref_images":
        value = row.get("ref_images")
    if value is None:
        value = row.get("images")
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON reference-image list: {value!r}") from error
        else:
            value = [stripped]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Reference images must be a list or path string, got {type(value)!r}.")
    paths = []
    for item in value:
        if not isinstance(item, (str, os.PathLike)) or not os.fspath(item).strip():
            raise TypeError(f"Reference-image entries must be non-empty paths, got {item!r}.")
        paths.append(os.fspath(item).strip())
    return paths


def resolve_reference_image(value: str, media_root: Path) -> ReferenceImage:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = media_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Reference image not found: {path}")
    stat = path.stat()
    identity = f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}"
    return ReferenceImage(
        path=str(path),
        size=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
        fingerprint=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
    )


def read_entries(
    path: Path,
    prompt_column: str,
    id_column: str,
    prompt_index: int,
    task: str = "t2v",
    reference_images_column: str = "ref_images",
    media_root: Path | None = None,
) -> list[PromptEntry]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".list"}:
        with path.open("r", encoding="utf-8") as handle:
            rows = [line.strip() for line in handle if line.strip()]
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [normalize_row(row) for row in csv.DictReader(handle)]
    elif suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        rows.append(normalize_row(json.loads(line)))
                    except json.JSONDecodeError as error:
                        raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            nested = False
            for key in ("data", "samples", "prompts"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    nested = True
                    break
            if not nested and task == "r2v":
                payload = [payload]
        if not isinstance(payload, list):
            raise TypeError(f"JSON input must contain a list, got {type(payload)!r}.")
        rows = [normalize_row(row) for row in payload]
    else:
        raise ValueError(f"Unsupported input suffix {suffix!r}; use TXT, JSON, JSONL, or CSV.")

    media_root = path.parent if media_root is None else media_root
    media_root = media_root.expanduser().resolve()
    resolved_references = {}
    entries = []
    for row_index, row in enumerate(rows):
        prompt = prompt_from_row(row, prompt_column, prompt_index)
        if prompt is None or not str(prompt).strip():
            raise ValueError(f"Missing prompt in input row {row_index}: {row!r}")
        uid = row.get(id_column) if isinstance(row, MappingLike) else None
        uid = str(uid) if uid not in (None, "") else f"{row_index:08d}"
        references = []
        if task == "r2v":
            raw_references = reference_paths_from_row(row, reference_images_column)
            if not raw_references:
                raise ValueError(f"R2V input row {row_index} has no reference images in {reference_images_column!r} or 'images'.")
            for raw_path in raw_references:
                cache_key = str((media_root / raw_path).resolve()) if not Path(raw_path).expanduser().is_absolute() else str(Path(raw_path).expanduser().resolve())
                reference = resolved_references.get(cache_key)
                if reference is None:
                    reference = resolve_reference_image(raw_path, media_root)
                    resolved_references[cache_key] = reference
                references.append(reference)
        entries.append(
            PromptEntry(
                row_index=row_index,
                prompt=str(prompt).strip(),
                uid=uid,
                reference_images=tuple(references),
            )
        )
    return entries


def configure_import_path(repo: str | None):
    if repo is None:
        return
    repo_path = str(Path(repo).expanduser().resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def resolve_rank(args):
    rank = int(os.environ.get("RANK", "0")) if args.rank is None else args.rank
    world_size = int(os.environ.get("WORLD_SIZE", "1")) if args.world_size is None else args.world_size
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, world_size), got rank={rank}, world_size={world_size}.")
    device = args.device
    if device == "cuda":
        device = f"cuda:{local_rank}"
    return rank, world_size, torch.device(device)


def set_seed(seed: int, device: torch.device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def atomic_torch_save(path: Path, payload, overwrite: bool):
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return True


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def condition_schema(task: str) -> str:
    if task == "r2v":
        return "lightx2v.bernini.r2v_conditions"
    return "lightx2v.bernini.conditions"


def cache_signature(args, system_prompt: str, negative_prompt: str):
    signature = {
        "schema": condition_schema(args.task),
        "bernini_model": str(Path(args.bernini_model).expanduser().resolve()),
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "max_image_size": args.max_image_size,
        "vit_min_pixels": args.vit_min_pixels,
        "vit_max_pixels": args.vit_max_pixels,
        "seed": args.seed,
        "planning_step": args.planning_step,
        "vit_denoising_step": args.vit_denoising_step,
        "vit_txt_cfg": args.vit_txt_cfg,
        "vit_img_cfg": args.vit_img_cfg,
        "max_sequence_length": args.max_sequence_length,
        "truncate": args.truncate,
        "save_dtype": args.save_dtype,
        "system_prompt_sha256": text_hash(system_prompt),
        "negative_prompt_sha256": text_hash(negative_prompt),
    }
    if args.task == "r2v":
        signature.update(
            {
                "task": "r2v",
                "interpolate_src_id": args.interpolate_src_id,
                "max_trained_src_id": args.max_trained_src_id,
            }
        )
    return signature


def entry_reference_metadata(entry: PromptEntry):
    return [reference.metadata() for reference in entry.reference_images]


def validate_existing_condition(path: Path, entry: PromptEntry, signature: dict):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("prompt") != entry.prompt:
        raise ValueError(f"Existing cache {path} does not match prompt row {entry.row_index}; pass --overwrite.")
    cached_signature = payload.get("cache_meta")
    if not isinstance(cached_signature, dict):
        raise ValueError(f"Existing cache {path} has no cache_meta; pass --overwrite.")
    mismatched = {key: (cached_signature.get(key), expected) for key, expected in signature.items() if cached_signature.get(key) != expected}
    if mismatched:
        raise ValueError(f"Existing cache {path} was built with different settings: {mismatched}. Pass --overwrite.")
    expected_references = entry_reference_metadata(entry)
    cached_references = payload.get("reference_images", [])
    if cached_references != expected_references:
        raise ValueError(f"Existing cache {path} was built from different reference images. Pass --overwrite.")
    conditions = payload.get("conditions")
    if not isinstance(conditions, dict):
        raise ValueError(f"Existing cache {path} has no conditions mapping; pass --overwrite.")
    for key in CONDITION_KEYS:
        tensor = conditions.get(key)
        if not torch.is_tensor(tensor) or tensor.ndim != 2 or tensor.shape[-1] != 4096:
            shape = None if not torch.is_tensor(tensor) else tuple(tensor.shape)
            raise ValueError(f"Existing cache {path} has invalid {key} shape={shape}; pass --overwrite.")
    if entry.reference_images:
        source_values = {key: conditions.get(key) for key in SOURCE_CONDITION_KEYS}
        if any(not isinstance(value, list) for value in source_values.values()):
            raise ValueError(f"Existing R2V cache {path} has invalid source-image condition lists; pass --overwrite.")
        list_lengths = {key: len(value) for key, value in source_values.items()}
        expected_count = len(entry.reference_images)
        if set(list_lengths.values()) != {expected_count}:
            raise ValueError(f"Existing R2V cache {path} has source-image counts {list_lengths}, expected {expected_count}; pass --overwrite.")
        for index, (patches, rope_cos, rope_sin) in enumerate(
            zip(
                source_values["source_image_vae_patches"],
                source_values["source_image_rope_cos"],
                source_values["source_image_rope_sin"],
            )
        ):
            if not torch.is_tensor(patches) or patches.ndim != 5 or tuple(patches.shape[1:]) != (16, 1, 2, 2):
                shape = None if not torch.is_tensor(patches) else tuple(patches.shape)
                raise ValueError(f"Existing R2V cache {path} has invalid source patches {index} shape={shape}.")
            expected_rope_shape = (patches.shape[0], 1, 128)
            if (
                not torch.is_tensor(rope_cos)
                or not torch.is_tensor(rope_sin)
                or tuple(rope_cos.shape) != expected_rope_shape
                or tuple(rope_sin.shape) != expected_rope_shape
                or rope_cos.dtype != torch.float64
                or rope_sin.dtype != torch.float64
            ):
                cos_shape = None if not torch.is_tensor(rope_cos) else tuple(rope_cos.shape)
                sin_shape = None if not torch.is_tensor(rope_sin) else tuple(rope_sin.shape)
                raise ValueError(f"Existing R2V cache {path} has invalid source RoPE {index}: cos={cos_shape}, sin={sin_shape}.")


def load_planner_only(model_dir: Path, device: torch.device, load_vae: bool = False):
    from bernini.models import BerniniConfig, BerniniModel
    from bernini.pipeline import AutoencoderKLWan, BerniniPipeline, _localize_bernini_config
    from transformers import AutoProcessor, AutoTokenizer

    config = BerniniConfig.from_pretrained(model_dir)
    _localize_bernini_config(config, model_dir)
    if getattr(config, "model_type", None) != "bernini":
        raise ValueError(f"Expected a full Bernini model, got model_type={getattr(config, 'model_type', None)!r}.")

    # Keep GEN_Wanx22 present for the connector's construction-time contract,
    # but skip allocating/loading either 14B transformer and its scheduler.
    config.cotrain = False
    config.skip_transformer_1 = True
    config.skip_transformer_2 = True
    config.use_unipc = False
    model = BerniniModel.from_pretrained(
        model_dir,
        subfolder=config.bernini_ckpt_subfolder,
        config=config,
        low_cpu_mem_usage=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        config.t5_tokenizer_path,
        subfolder=config.t5_tokenizer_subfolder,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        config.processor_config_path,
        subfolder=config.processor_subfolder,
        padding_side="right",
        trust_remote_code=True,
    )
    vae = None
    if load_vae:
        vae = AutoencoderKLWan.from_pretrained(
            config.vae_model_path,
            subfolder=config.vae_subfolder,
            torch_dtype=torch.float32,
        )
        vae.eval()
        vae.requires_grad_(False)
    pipeline = BerniniPipeline(config, model, vae, tokenizer, processor, device)
    return pipeline


def build_target_template(pipeline, args):
    from bernini.data.utils.video_utils import smart_video_nframes
    from bernini.data_utils import FakeVideoReader

    if args.num_frames < 1 or (args.num_frames - 1) % 4 != 0:
        raise ValueError("--num-frames must be 4*n+1 for the Wan VAE geometry.")
    if args.height % 16 or args.width % 16:
        raise ValueError("--height and --width must be divisible by 16.")
    if args.fps < 8:
        raise ValueError("--fps must be at least 8 so full Bernini's vit_fps=fps//8 is non-zero.")

    reader = FakeVideoReader(args.num_frames, args.height, args.width, fps=args.fps)
    indices = smart_video_nframes(
        total_frames=reader.length,
        video_fps=reader.fps,
        fps=args.fps // 8,
        frame_factor=2,
        max_frames=args.num_frames,
        add_one=False,
    )
    processed = pipeline.vit_processor.video_processor(
        videos=reader.sample(indices),
        return_tensors="pt",
        size={"shortest_edge": args.vit_min_pixels, "longest_edge": args.vit_max_pixels},
    )
    grid = processed["video_grid_thw"].cpu()
    merge_size = pipeline.vit_processor.image_processor.merge_size
    token_counts = grid.prod(dim=-1) // (merge_size**2)
    hidden_size = int(pipeline.model.mllm.config.hidden_size)
    visual_embeds = [torch.zeros(int(count), hidden_size, dtype=torch.bfloat16) for count in token_counts]

    with open(pipeline.config.vae_config_path, "r", encoding="utf-8") as handle:
        vae_config = json.load(handle)
    z_dim = int(vae_config["z_dim"])
    latent_shape = (
        1 + (args.num_frames - 1) // 4,
        args.height // 8,
        args.width // 8,
    )
    # Bernini's formatter expects distribution parameters, not sampled/mode
    # latents.  Values do not affect planner conditions; only this shape does.
    dummy_distribution = torch.zeros(1, 2 * z_dim, *latent_shape, dtype=torch.bfloat16)
    del processed
    return {
        "video_embeds": [tensor_to_bytes(tensor) for tensor in visual_embeds],
        "video_grid_thw": grid.tolist(),
        "video_vae_latents": [tensor_to_bytes(dummy_distribution)],
        "vit_frame_count": len(indices),
        "visual_token_count": int(token_counts.sum()),
        "latent_shape": latent_shape,
    }


def move_planner(pipeline, device):
    pipeline.model.mllm.to(device=device, dtype=pipeline.weight_dtype)
    if pipeline.connector is not None:
        pipeline.connector.to(device=device, dtype=pipeline.weight_dtype)
    if getattr(pipeline.model, "vit_decoder", None) is not None:
        pipeline.model.vit_decoder.to(device=device, dtype=pipeline.weight_dtype)


def move_planner_to_cpu(pipeline):
    pipeline.model.mllm.to("cpu")
    if pipeline.connector is not None:
        pipeline.connector.to("cpu")
    if getattr(pipeline.model, "vit_decoder", None) is not None:
        pipeline.model.vit_decoder.to("cpu")


def reference_feature_signature(reference: ReferenceImage, args):
    return {
        "schema": "lightx2v.bernini.reference_features",
        "bernini_model": str(Path(args.bernini_model).expanduser().resolve()),
        "reference": reference.metadata(),
        "max_image_size": args.max_image_size,
        "vit_min_pixels": args.vit_min_pixels,
        "vit_max_pixels": args.vit_max_pixels,
    }


def cache_reference_features(pipeline, entries, args, device, cache_dir: Path):
    """Persist each unique R2V reference feature and return fingerprint -> path."""
    from bernini.data_utils import VAEVideoTransform, get_vae_features, get_vit_features

    unique_references = {}
    for entry in entries:
        for reference in entry.reference_images:
            unique_references.setdefault(reference.fingerprint, reference)
    if not unique_references:
        return {}
    if pipeline.vae is None:
        raise RuntimeError("R2V condition caching requires the frozen Bernini VAE.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_paths = {}
    pending = []
    for reference in unique_references.values():
        signature = reference_feature_signature(reference, args)
        cache_key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()
        path = cache_dir / f"reference_{cache_key}.pt"
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict) or payload.get("cache_meta") != signature:
                raise ValueError(f"Invalid reference feature cache {path}; remove it and rerun preprocessing.")
            required = ("image_embed", "image_grid_thw", "image_vae_latent")
            if any(key not in payload for key in required):
                raise ValueError(f"Incomplete reference feature cache {path}; remove it and rerun preprocessing.")
        else:
            pending.append((reference, signature, path))
        feature_paths[reference.fingerprint] = path

    if not pending:
        return feature_paths

    move_planner(pipeline, device)
    pipeline.vae.to(device=device, dtype=torch.float32)
    vae_transform = VAEVideoTransform(
        max_image_size=args.max_image_size,
        min_image_size=240,
        image_stride=16,
    )
    for reference, signature, path in pending:
        image_inputs = pipeline.vit_processor.image_processor(
            images=[reference.path],
            return_tensors="pt",
            min_pixels=args.vit_min_pixels,
            max_pixels=args.vit_max_pixels,
        )
        grid = image_inputs["image_grid_thw"]
        embeds = get_vit_features(
            pipeline.model.mllm,
            image_inputs["pixel_values"],
            grid,
        )
        if len(embeds) != 1 or grid.shape[0] != 1:
            raise RuntimeError(f"Expected one visual feature for {reference.path}, got {len(embeds)}.")
        vae_distribution = get_vae_features(pipeline.vae, vae_transform(reference.path))
        payload = {
            "image_embed": tensor_to_bytes(embeds[0].detach().cpu()),
            "image_grid_thw": grid[0].detach().cpu().tolist(),
            "image_vae_latent": vae_distribution,
            "cache_meta": signature,
        }
        atomic_torch_save(path, payload, overwrite=False)
        del image_inputs, embeds

    if not args.keep_models_on_device:
        pipeline.vae.to("cpu")
        move_planner_to_cpu(pipeline)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return feature_paths


def move_tensors(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensors(item, device) for item in value)
    return value


def source_renderer_conditions(transformed, reference_count: int, save_dtype: torch.dtype):
    """Split official packed VAE inputs back into per-reference renderer inputs."""
    inputs = transformed["inputs"]
    packed = inputs["input_vae_latents"]
    packed_rope = inputs["input_vae_rope"]
    shapes = inputs["input_vae_shape"]
    masks = inputs["vae_latents_mask"]
    if shapes.shape[0] != reference_count + 1:
        raise RuntimeError(f"Expected {reference_count} source images plus one target video, got VAE shapes {shapes.tolist()}.")

    source_patches = []
    source_rope_cos = []
    source_rope_sin = []
    offset = 0
    for index in range(reference_count):
        frames, height, width = (int(value) for value in shapes[index].tolist())
        if height % 2 or width % 2:
            raise ValueError(f"Reference latent shape must be patchable by 1x2x2, got {(frames, height, width)}.")
        sequence_length = frames * (height // 2) * (width // 2)
        patches = packed[offset : offset + sequence_length]
        rope = packed_rope[offset : offset + sequence_length]
        source_mask = masks[offset : offset + sequence_length]
        if patches.shape[0] != sequence_length or rope.shape[0] != sequence_length or source_mask.any():
            raise RuntimeError(f"Invalid packed source-image segment {index} for shape {(frames, height, width)}.")
        if patches.ndim != 5 or tuple(patches.shape[1:]) != (16, 1, 2, 2):
            raise RuntimeError(f"Unexpected Bernini source VAE patch shape: {tuple(patches.shape)}.")
        if not torch.is_complex(rope) or tuple(rope.shape[1:]) != (1, 64):
            raise RuntimeError(f"Unexpected Bernini complex source RoPE shape: {tuple(rope.shape)}.")
        source_patches.append(patches.detach().to(device="cpu", dtype=save_dtype).contiguous())
        source_rope_cos.append(rope.real.repeat_interleave(2, dim=-1).double().cpu().contiguous())
        source_rope_sin.append(rope.imag.repeat_interleave(2, dim=-1).double().cpu().contiguous())
        offset += sequence_length
    return {
        "source_image_vae_patches": source_patches,
        "source_image_rope_cos": source_rope_cos,
        "source_image_rope_sin": source_rope_sin,
    }


def planner_routes(pipeline, transformed, args):
    inputs = transformed["inputs"]
    uncond = transformed["uncond_inputs"]
    imgcond = transformed["imgcond_inputs"]

    def embeddings(branch):
        return pipeline.model.format_mllm_inputs_embeds(
            input_ids=branch["input_ids"],
            visual_embeds=branch["visual_embeds"],
            visual_input_mask=branch["visual_input_token_mask"],
            visual_output_mask=branch["visual_output_token_mask"],
        ).to(pipeline.weight_dtype)

    input_embeds = embeddings(inputs)
    uncond_embeds = embeddings(uncond)
    imgcond_embeds = embeddings(imgcond)

    def mask_outputs(branch_embeds, branch):
        return pipeline.model.post_process_input_embeds(
            branch_embeds.unsqueeze(0),
            branch["visual_output_token_mask"],
            tgt_vit_mask=None,
            inference=True,
        )["input_embeds"]

    return pipeline.sample_vit_embed(
        input_embeds=mask_outputs(input_embeds, inputs),
        attention_mask_4d=inputs["attention_mask_4d"].unsqueeze(0),
        position_ids=inputs["position_ids"].unsqueeze(0),
        visual_output_token_mask=inputs["visual_output_token_mask"],
        uncond_input_embeds=mask_outputs(uncond_embeds, uncond),
        uncond_position_ids=uncond["position_ids"].unsqueeze(0),
        uncond_attention_mask_4d=uncond["attention_mask_4d"].unsqueeze(0),
        uncond_visual_output_token_mask=uncond["visual_output_token_mask"],
        imgcond_input_embeds=mask_outputs(imgcond_embeds, imgcond),
        imgcond_position_ids=imgcond["position_ids"].unsqueeze(0),
        imgcond_attention_mask_4d=imgcond["attention_mask_4d"].unsqueeze(0),
        imgcond_visual_output_token_mask=imgcond["visual_output_token_mask"],
        planning_step=args.planning_step,
        vit_txt_cfg=args.vit_txt_cfg,
        vit_img_cfg=args.vit_img_cfg,
        vit_denoising_step=args.vit_denoising_step,
    )


def final_conditions(pipeline, routes, positive_text: str, negative_text: str, args, device):
    from bernini.pipeline import _get_t5_text_ids

    t5 = pipeline.model.t5_text_encoder
    t5.to(device)
    positive_ids, positive_mask = _get_t5_text_ids(positive_text, pipeline.t5_tokenizer)
    negative_ids, negative_mask = _get_t5_text_ids(negative_text, pipeline.t5_tokenizer)
    positive = pipeline.model.get_t5_text_embeddings_sample(positive_ids.to(device), positive_mask.to(device))
    negative = pipeline.model.get_t5_text_embeddings_sample(negative_ids.to(device), negative_mask.to(device))

    conditions = {
        "cond_embeds_wtxt_wvit": torch.cat([positive, routes["cond_embeds_wtxt_wvit"]], dim=1),
        "cond_embeds_wtxt_wovit": torch.cat([positive, routes["cond_embeds_wtxt_wovit"]], dim=1),
        "cond_embeds_wotxt_wvit": torch.cat([negative, routes["cond_embeds_wotxt_wvit"]], dim=1),
        "cond_embeds_wotxt_wovit": torch.cat([negative, routes["cond_embeds_wotxt_wovit"]], dim=1),
    }
    output_dtype = SAVE_DTYPES[args.save_dtype]
    for key, tensor in conditions.items():
        if tensor.shape[1] < args.max_sequence_length:
            padding = tensor.new_zeros(1, args.max_sequence_length - tensor.shape[1], tensor.shape[2])
            tensor = torch.cat([tensor, padding], dim=1)
        if args.truncate and tensor.shape[1] > args.max_sequence_length:
            tensor = tensor[:, : args.max_sequence_length]
        conditions[key] = tensor.squeeze(0).to(device="cpu", dtype=output_dtype).contiguous()
    return conditions


def cache_entry(
    pipeline,
    template,
    entry,
    system_prompt,
    negative_prompt,
    signature,
    reference_features,
    args,
    device,
):
    from bernini.data_utils import generate_unified_inputs
    from bernini.pipeline import _prompt_clean

    set_seed(args.seed, device)
    raw_prompt = _prompt_clean(entry.prompt)
    reference_paths = [reference.path for reference in entry.reference_images]
    sample = {
        "uid": entry.uid,
        "inputs": generate_unified_inputs(
            raw_prompt,
            input_image_paths=reference_paths,
            input_video_paths=[],
            has_video_input=False,
            output_t=args.num_frames,
            output_h=args.height,
            output_w=args.width,
        ),
        "video_embeds": template["video_embeds"],
        "video_grid_thw": template["video_grid_thw"],
        "video_vae_latents": template["video_vae_latents"],
    }
    if entry.reference_images:
        features = [torch.load(reference_features[reference.fingerprint], map_location="cpu", weights_only=False) for reference in entry.reference_images]
        sample.update(
            {
                "image_embeds": [feature["image_embed"] for feature in features],
                "image_grid_thw": [feature["image_grid_thw"] for feature in features],
                "image_vae_latents": [feature["image_vae_latent"] for feature in features],
            }
        )
    transformed = pipeline.transform_inputs(
        sample,
        args.num_frames,
        task_name=args.task,
        neg_prompt=negative_prompt,
    )
    source_conditions = {}
    if entry.reference_images:
        source_conditions = source_renderer_conditions(
            transformed,
            len(entry.reference_images),
            SAVE_DTYPES[args.save_dtype],
        )
    transformed = move_tensors(transformed, device)

    if not args.keep_models_on_device:
        move_planner(pipeline, device)
    routes = planner_routes(pipeline, transformed, args)
    if any(routes.get(key) is None for key in CONDITION_KEYS):
        missing = [key for key in CONDITION_KEYS if routes.get(key) is None]
        raise RuntimeError(f"Full Bernini planner did not produce all four condition routes: {missing}.")
    if not args.keep_models_on_device:
        move_planner_to_cpu(pipeline)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    t5_prompt = _prompt_clean(system_prompt + raw_prompt)
    conditions = final_conditions(pipeline, routes, t5_prompt, _prompt_clean(negative_prompt), args, device)
    conditions.update(source_conditions)
    if not args.keep_models_on_device:
        pipeline.model.t5_text_encoder.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    payload = {
        "prompt": entry.prompt,
        "clean_prompt": raw_prompt,
        "conditions": conditions,
        "cache_meta": signature,
    }
    if entry.reference_images:
        payload["reference_images"] = entry_reference_metadata(entry)
    return payload


def initialize_distributed(world_size: int):
    if world_size <= 1:
        return False
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is unavailable but world_size > 1.")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo")
    return True


def global_pending_count(local_pending: int, distributed: bool, device: torch.device) -> int:
    """Return the sum of pending entries so every rank takes the model-load branch."""
    if not distributed:
        return local_pending
    backend = str(torch.distributed.get_backend()).lower()
    collective_device = device if "nccl" in backend else torch.device("cpu")
    pending_tensor = torch.tensor(local_pending, dtype=torch.int64, device=collective_device)
    torch.distributed.all_reduce(pending_tensor, op=torch.distributed.ReduceOp.SUM)
    return int(pending_tensor.item())


def write_manifest(path: Path, rows):
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def merge_manifests(output_dir: Path, world_size: int):
    rows = []
    for rank in range(world_size):
        shard = output_dir / f"metadata_rank{rank:05d}.jsonl"
        with shard.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    rows.sort(key=lambda row: row["row_index"])
    write_manifest(output_dir / "metadata.jsonl", rows)


def main():
    args = parse_args()
    if args.max_trained_src_id < 1:
        raise ValueError("--max-trained-src-id must be positive.")
    configure_import_path(args.bernini_repo)
    rank, world_size, device = resolve_rank(args)
    input_path = Path(args.input).expanduser().resolve()
    media_root = input_path.parent if args.media_root is None else Path(args.media_root).expanduser().resolve()
    entries = read_entries(
        input_path,
        args.prompt_column,
        args.id_column,
        args.prompt_index,
        task=args.task,
        reference_images_column=args.reference_images_column,
        media_root=media_root,
    )
    if args.max_samples is not None:
        entries = entries[: args.max_samples]
    local_entries = [entry for entry in entries if entry.row_index % world_size == rank]
    print(f"entries={len(entries)} local_entries={len(local_entries)} rank={rank}/{world_size} device={device}", flush=True)

    if args.dry_run:
        reference_count = sum(len(entry.reference_images) for entry in entries)
        unique_references = len({reference.fingerprint for entry in entries for reference in entry.reference_images})
        print(
            f"task={args.task} target={args.num_frames}x{args.height}x{args.width} "
            f"planning={args.planning_step} vit_denoising={args.vit_denoising_step} "
            f"cfg={args.vit_txt_cfg}/{args.vit_img_cfg} references={reference_count} "
            f"unique_references={unique_references}",
            flush=True,
        )
        return

    distributed = initialize_distributed(world_size)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.prompt_enhancer import get_system_prompt_for_task

    negative_prompt = DEFAULT_NEG_PROMPT if args.negative_prompt is None else args.negative_prompt
    system_prompt = get_system_prompt_for_task(args.task) if args.system_prompt is None else args.system_prompt
    signature = cache_signature(args, system_prompt, negative_prompt)
    rows = []
    pending = []
    for entry in local_entries:
        relative = Path("conditions") / f"condition_{entry.row_index:08d}.pt"
        path = output_dir / relative
        if path.exists() and not args.overwrite:
            validate_existing_condition(path, entry, signature)
            row = {
                "id": entry.uid,
                "row_index": entry.row_index,
                "caption": entry.prompt,
                "condition_path": str(relative),
            }
            if entry.reference_images:
                row["reference_images"] = [reference.path for reference in entry.reference_images]
            rows.append(row)
            print(f"rank={rank} skipped valid cache row={entry.row_index}", flush=True)
        else:
            pending.append((entry, relative, path))

    # BerniniModel.from_pretrained contains distributed barriers.  Loading only
    # on ranks with local cache misses can therefore deadlock when cache hits
    # are uneven.  Make the decision collectively and let every rank build the
    # same planner/template whenever any rank has work left.
    global_pending = global_pending_count(len(pending), distributed, device)
    print(f"rank={rank} local_pending={len(pending)} global_pending={global_pending}", flush=True)

    pipeline = None
    template = None
    reference_features = {}
    if global_pending:
        if device.type != "cuda":
            raise ValueError("Full Bernini's visual-token sampler currently requires a CUDA device.")
        torch.cuda.set_device(device)
        pipeline = load_planner_only(
            Path(args.bernini_model).expanduser().resolve(),
            device,
            load_vae=args.task == "r2v",
        )
        pipeline.config.interpolate_src_id = args.interpolate_src_id
        pipeline.config.max_trained_src_id = args.max_trained_src_id
        template = build_target_template(pipeline, args)
        if args.keep_models_on_device:
            move_planner(pipeline, device)
            pipeline.model.t5_text_encoder.to(device)
        if args.task == "r2v" and pending:
            reference_features = cache_reference_features(
                pipeline,
                [entry for entry, _, _ in pending],
                args,
                device,
                output_dir / "reference_features",
            )

    for local_index, (entry, relative, path) in enumerate(pending):
        payload = cache_entry(
            pipeline,
            template,
            entry,
            system_prompt,
            negative_prompt,
            signature,
            reference_features,
            args,
            device,
        )
        atomic_torch_save(path, payload, args.overwrite)
        row = {
            "id": entry.uid,
            "row_index": entry.row_index,
            "caption": entry.prompt,
            "condition_path": str(relative),
        }
        if entry.reference_images:
            row["reference_images"] = [reference.path for reference in entry.reference_images]
        rows.append(row)
        print(f"rank={rank} cached {local_index + 1}/{len(pending)} row={entry.row_index}", flush=True)

    rows.sort(key=lambda row: row["row_index"])
    shard_manifest = output_dir / ("metadata.jsonl" if world_size == 1 else f"metadata_rank{rank:05d}.jsonl")
    write_manifest(shard_manifest, rows)
    if rank == 0 and template is not None:
        atomic_json(
            output_dir / "cache_config.json",
            {
                "schema": condition_schema(args.task),
                "task": args.task,
                "bernini_model": str(Path(args.bernini_model).expanduser().resolve()),
                "condition_keys": list(CONDITION_KEYS) + (list(SOURCE_CONDITION_KEYS) if args.task == "r2v" else []),
                "height": args.height,
                "width": args.width,
                "num_frames": args.num_frames,
                "max_image_size": args.max_image_size,
                "fps": args.fps,
                "seed": args.seed,
                "planning_step": args.planning_step,
                "vit_denoising_step": args.vit_denoising_step,
                "vit_txt_cfg": args.vit_txt_cfg,
                "vit_img_cfg": args.vit_img_cfg,
                "max_sequence_length": args.max_sequence_length,
                "truncate": args.truncate,
                "save_dtype": args.save_dtype,
                "vit_min_pixels": args.vit_min_pixels,
                "vit_max_pixels": args.vit_max_pixels,
                "system_prompt": system_prompt,
                "negative_prompt": negative_prompt,
                **(
                    {
                        "reference_images_column": args.reference_images_column,
                        "media_root": str(media_root),
                        "interpolate_src_id": args.interpolate_src_id,
                        "max_trained_src_id": args.max_trained_src_id,
                        "reference_feature_cache": "reference_features",
                    }
                    if args.task == "r2v"
                    else {}
                ),
                "target_template": {
                    "vit_frame_count": template["vit_frame_count"],
                    "visual_token_count": template["visual_token_count"],
                    "latent_shape": list(template["latent_shape"]),
                    "video_grid_thw": template["video_grid_thw"],
                },
            },
        )
    elif rank == 0 and not (output_dir / "cache_config.json").is_file():
        raise FileNotFoundError("All rank-0 condition files were reused, but cache_config.json is missing. Rebuild with --overwrite so directory-level provenance can be restored.")

    if distributed:
        torch.distributed.barrier()
        if rank == 0:
            merge_manifests(output_dir, world_size)
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    print(f"rank={rank} wrote {len(rows)} records to {shard_manifest}", flush=True)


if __name__ == "__main__":
    main()
