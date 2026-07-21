#!/usr/bin/env python3
"""Split the two Wan experts out of a full Bernini checkpoint.

The full Bernini model does not use the Bernini-R state-dict layout.  Its
expert prefixes are exactly::

    high: diff_dec.transformer.
    low:  diff_dec_low.transformer_2.

The output is two native Bernini expert directories that can be loaded by
LightX2V without constructing the MLLM, T5 encoder, VAE, or the other expert.
Released sharded safetensors checkpoints are streamed one output shard at a
time.  A single PyTorch checkpoint is accepted as a compatibility fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXPERTS = {
    "high": {
        "prefix": "diff_dec.transformer.",
        "directory": "high_noise_model",
        "config": "transformer_config.json",
    },
    "low": {
        "prefix": "diff_dec_low.transformer_2.",
        "directory": "low_noise_model",
        "config": "transformer_2_config.json",
    },
}
KNOWN_STATE_CONTAINERS = (
    "generator_ema",
    "model_ema",
    "ema",
    "generator",
    "model",
    "state_dict",
)
DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        help=("Bernini-Diffusers root, its bernini/ directory, a safetensors file/index, or a single PyTorch checkpoint"),
    )
    parser.add_argument("output_dir", help="Output root for high_noise_model/ and low_noise_model/")
    parser.add_argument(
        "--experts",
        choices=("both", "high", "low"),
        default="both",
        help="Experts to extract (default: both)",
    )
    parser.add_argument(
        "--config-root",
        default=None,
        help="Directory containing transformer_config.json and transformer_2_config.json",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help=("Outer state-dict namespace before the exact expert prefix. For example: generator_ema.  The default is auto-detection."),
    )
    parser.add_argument(
        "--prefer-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer a namespace whose name contains 'ema' when both EMA and online weights exist",
    )
    parser.add_argument(
        "--dtype",
        choices=("keep", "bf16", "fp16", "fp32"),
        default="keep",
        help="Optional output cast (default: preserve checkpoint dtype)",
    )
    parser.add_argument("--max-shard-size", default="4GB", help="Maximum output shard size, for example 4GB or 2048MB")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated expert directories")
    parser.add_argument("--dry-run", action="store_true", help="Print the selected namespace and tensor counts without writing")
    return parser.parse_args()


def parse_size(value: str) -> int:
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?I?B)?\s*", value, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Invalid size {value!r}; expected a value such as 4GB or 2048MiB.")
    number = float(match.group(1))
    unit = (match.group(2) or "B").upper()
    decimal = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}
    binary = {"KIB": 2**10, "MIB": 2**20, "GIB": 2**30, "TIB": 2**40}
    multiplier = decimal.get(unit, binary.get(unit))
    if multiplier is None:
        raise ValueError(f"Unsupported size unit in {value!r}.")
    size = int(number * multiplier)
    if size <= 0:
        raise ValueError("--max-shard-size must be positive.")
    return size


def has_safetensors(path: Path) -> bool:
    if path.is_file():
        return path.suffix == ".safetensors" or path.name.endswith(".safetensors.index.json")
    return any(path.glob("*.safetensors")) or any(path.glob("*.safetensors.index.json"))


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        return path
    candidates = (path / "bernini", path)
    for candidate in candidates:
        if candidate.is_dir() and has_safetensors(candidate):
            return candidate
    pytorch_files = [*path.glob("*.pt"), *path.glob("*.pth"), *path.glob("*.bin")]
    if len(pytorch_files) == 1:
        return pytorch_files[0]
    raise FileNotFoundError(f"No Bernini safetensors checkpoint found under {path}.")


def resolve_config_root(checkpoint_arg: Path, resolved_checkpoint: Path, explicit: str | None, experts) -> Path:
    if explicit is not None:
        candidates = [Path(explicit).expanduser().resolve()]
    else:
        original = checkpoint_arg.expanduser().resolve()
        base = resolved_checkpoint if resolved_checkpoint.is_dir() else resolved_checkpoint.parent
        candidates = [original, original.parent, base, base.parent]
    for candidate in candidates:
        if candidate.is_dir() and all((candidate / EXPERTS[expert]["config"]).is_file() for expert in experts):
            return candidate
    expected = ", ".join(EXPERTS[expert]["config"] for expert in experts)
    raise FileNotFoundError(f"Could not find {expected}. Pass --config-root explicitly.")


def safetensors_sources(checkpoint: Path) -> dict[str, Path]:
    if checkpoint.is_file() and checkpoint.name.endswith(".safetensors.index.json"):
        index_path = checkpoint
        checkpoint_dir = checkpoint.parent
    elif checkpoint.is_file() and checkpoint.suffix == ".safetensors":
        index_path = None
        checkpoint_dir = checkpoint.parent
    else:
        checkpoint_dir = checkpoint
        preferred = (
            "model.safetensors.index.json",
            "diffusion_pytorch_model.safetensors.index.json",
            "pytorch_model.safetensors.index.json",
        )
        index_path = next((checkpoint_dir / name for name in preferred if (checkpoint_dir / name).is_file()), None)
        if index_path is None:
            indexes = sorted(checkpoint_dir.glob("*.safetensors.index.json"))
            if len(indexes) > 1:
                raise RuntimeError(f"Multiple safetensors indexes found in {checkpoint_dir}; pass one index explicitly.")
            index_path = indexes[0] if indexes else None

    if index_path is not None:
        with index_path.open("r", encoding="utf-8") as handle:
            weight_map = json.load(handle).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid or empty weight_map in {index_path}.")
        return {key: checkpoint_dir / filename for key, filename in weight_map.items()}

    files = [checkpoint] if checkpoint.is_file() else sorted(checkpoint_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {checkpoint_dir}.")
    sources = {}
    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in sources:
                    raise RuntimeError(f"Duplicate tensor key {key!r} in {filename} and {sources[key]}.")
                sources[key] = filename
    return sources


def namespace_counts(keys) -> dict[str, dict[str, int]]:
    counts = defaultdict(lambda: defaultdict(int))
    for key in keys:
        for expert, info in EXPERTS.items():
            marker = info["prefix"]
            offset = key.find(marker)
            if offset >= 0 and key[offset + len(marker) :]:
                counts[key[:offset]][expert] += 1
    return {namespace: dict(values) for namespace, values in counts.items()}


def is_ema_namespace(value: str) -> bool:
    return any("ema" in part for part in value.lower().replace("-", "_").split("."))


def validate_full_bernini_config(config_root: Path):
    config_path = config_root / "config.json"
    if not config_path.is_file():
        print("WARNING: config-root has no config.json, so model_type cannot be verified. The exact full-Bernini prefixes will still be enforced; do not point this script at Bernini-R.")
        return
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    model_type = payload.get("model_type")
    if model_type != "bernini":
        raise ValueError(f"Expected full Bernini config with model_type='bernini', got {model_type!r} at {config_path}. Bernini-R/renderer checkpoints are intentionally unsupported by this script.")


def select_namespace(keys, experts, explicit: str | None, prefer_ema: bool):
    counts = namespace_counts(keys)
    if explicit is not None:
        namespace = explicit
        missing = [expert for expert in experts if counts.get(namespace, {}).get(expert, 0) == 0]
        if missing:
            raise KeyError(f"Namespace {namespace!r} has no tensors for experts {missing}; detected={counts}.")
        return namespace, counts

    candidates = [namespace for namespace, values in counts.items() if all(values.get(expert, 0) > 0 for expert in experts)]
    if not candidates:
        raise KeyError(f"No common namespace contains the exact full-Bernini prefixes for experts={experts}. Detected namespaces: {counts}")

    def score(namespace):
        values = counts[namespace]
        ema_match = is_ema_namespace(namespace)
        preferred = ema_match if prefer_ema else not ema_match
        return preferred, min(values[expert] for expert in experts), sum(values[expert] for expert in experts), -len(namespace)

    return max(candidates, key=score), counts


def nested_tensor_mappings(payload):
    """Yield named tensor mappings from common PyTorch checkpoint containers."""
    if not isinstance(payload, Mapping):
        return
    seen = set()

    def visit(name, value, depth):
        if not isinstance(value, Mapping) or id(value) in seen:
            return
        seen.add(id(value))
        tensors = {key: item for key, item in value.items() if isinstance(key, str) and torch.is_tensor(item)}
        if tensors:
            yield name, tensors
        if depth < 2:
            for key in KNOWN_STATE_CONTAINERS:
                child = value.get(key)
                if isinstance(child, Mapping):
                    child_name = f"{name}.{key}" if name else key
                    yield from visit(child_name, child, depth + 1)

    yield from visit("root", payload, 0)


def select_pytorch_mapping(payload, experts, namespace, prefer_ema):
    choices = []
    for container, mapping in nested_tensor_mappings(payload):
        try:
            selected_namespace, counts = select_namespace(mapping, experts, namespace, prefer_ema)
        except KeyError:
            continue
        container_is_ema = "ema" in container.lower()
        preferred = container_is_ema if prefer_ema else not container_is_ema
        coverage = sum(counts[selected_namespace][expert] for expert in experts)
        choices.append(((preferred, coverage), container, mapping, selected_namespace, counts))
    if not choices:
        raise KeyError("The PyTorch checkpoint has no tensor mapping with the exact full-Bernini expert prefixes.")
    _, container, mapping, selected_namespace, counts = max(choices, key=lambda item: item[0])
    return container, mapping, selected_namespace, counts


def tensor_nbytes(tensor: torch.Tensor, dtype: torch.dtype | None) -> int:
    element_size = torch.empty((), dtype=dtype or tensor.dtype).element_size()
    return tensor.numel() * element_size


def prepare_destination(path: Path, overwrite: bool):
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {path}. Pass --overwrite to replace it.")
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def save_expert(
    destination: Path,
    tensor_keys: list[str],
    load_tensor,
    config_path: Path,
    max_shard_size: int,
    output_dtype: torch.dtype | None,
    overwrite: bool,
):
    prepare_destination(destination, overwrite)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp.", dir=destination.parent))
    parts = []
    weight_map = {}
    total_size = 0
    current = {}
    current_size = 0

    def flush():
        nonlocal current, current_size
        if not current:
            return
        part_path = temp_dir / f"part-{len(parts) + 1:05d}.safetensors"
        save_file(current, part_path, metadata={"format": "pt"})
        parts.append((part_path, list(current)))
        current = {}
        current_size = 0

    try:
        for output_key in sorted(tensor_keys):
            tensor = load_tensor(output_key).detach().cpu()
            if output_dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(output_dtype)
            tensor = tensor.contiguous()
            size = tensor_nbytes(tensor, None)
            if current and current_size + size > max_shard_size:
                flush()
            current[output_key] = tensor
            current_size += size
            total_size += size
        flush()

        shard_count = len(parts)
        if shard_count == 0:
            raise RuntimeError(f"No tensors selected for {destination.name}.")
        if shard_count == 1:
            final_name = "diffusion_pytorch_model.safetensors"
            os.replace(parts[0][0], temp_dir / final_name)
            for key in parts[0][1]:
                weight_map[key] = final_name
        else:
            for index, (part_path, keys) in enumerate(parts, start=1):
                final_name = f"diffusion_pytorch_model-{index:05d}-of-{shard_count:05d}.safetensors"
                os.replace(part_path, temp_dir / final_name)
                for key in keys:
                    weight_map[key] = final_name
            index_payload = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
            with (temp_dir / "diffusion_pytorch_model.safetensors.index.json").open("w", encoding="utf-8") as handle:
                json.dump(index_payload, handle, indent=2, sort_keys=True)
                handle.write("\n")

        shutil.copy2(config_path, temp_dir / "config.json")
        os.replace(temp_dir, destination)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return {"tensor_count": len(tensor_keys), "total_size": total_size, "shard_count": shard_count}


def main():
    args = parse_args()
    requested_experts = ["high", "low"] if args.experts == "both" else [args.experts]
    checkpoint_arg = Path(args.checkpoint)
    checkpoint = resolve_checkpoint(checkpoint_arg)
    config_root = resolve_config_root(checkpoint_arg, checkpoint, args.config_root, requested_experts)
    validate_full_bernini_config(config_root)
    output_root = Path(args.output_dir).expanduser().resolve()
    output_dtype = None if args.dtype == "keep" else DTYPES[args.dtype]
    max_shard_size = parse_size(args.max_shard_size)

    is_safe = checkpoint.is_dir() or checkpoint.suffix == ".safetensors" or checkpoint.name.endswith(".safetensors.index.json")
    if is_safe:
        sources = safetensors_sources(checkpoint)
        namespace, counts = select_namespace(sources, requested_experts, args.namespace, args.prefer_ema)
        container = "safetensors"
        state_mapping = None
    else:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        container, state_mapping, namespace, counts = select_pytorch_mapping(payload, requested_experts, args.namespace, args.prefer_ema)
        sources = None

    print(f"checkpoint={checkpoint}")
    print(f"state_container={container} namespace={namespace!r} prefer_ema={args.prefer_ema}")
    for expert in requested_experts:
        print(f"{expert}: prefix={namespace + EXPERTS[expert]['prefix']} tensors={counts[namespace][expert]}")
    if args.dry_run:
        return

    manifest = {
        "source_checkpoint": str(checkpoint),
        "state_container": container,
        "namespace": namespace,
        "prefer_ema": args.prefer_ema,
        "dtype": args.dtype,
        "experts": {},
    }

    for expert in requested_experts:
        info = EXPERTS[expert]
        full_prefix = namespace + info["prefix"]
        if sources is not None:
            selected = {key[len(full_prefix) :]: (filename, key) for key, filename in sources.items() if key.startswith(full_prefix)}

            def load_tensor(output_key, selected=selected):
                filename, source_key = selected[output_key]
                with safe_open(filename, framework="pt", device="cpu") as handle:
                    return handle.get_tensor(source_key)

        else:
            selected = {key[len(full_prefix) :]: key for key in state_mapping if key.startswith(full_prefix)}

            def load_tensor(output_key, selected=selected, state_mapping=state_mapping):
                return state_mapping[selected[output_key]]

        if len(selected) != counts[namespace][expert]:
            raise RuntimeError(f"Namespace accounting mismatch for {expert}: selected={len(selected)} expected={counts[namespace][expert]}.")
        destination = output_root / info["directory"]
        result = save_expert(
            destination=destination,
            tensor_keys=list(selected),
            load_tensor=load_tensor,
            config_path=config_root / info["config"],
            max_shard_size=max_shard_size,
            output_dtype=output_dtype,
            overwrite=args.overwrite,
        )
        result.update({"source_prefix": full_prefix, "directory": info["directory"], "config": info["config"]})
        manifest["experts"][expert] = result
        print(f"wrote {expert} expert -> {destination} ({result['tensor_count']} tensors, {result['shard_count']} shards)")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "split_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
