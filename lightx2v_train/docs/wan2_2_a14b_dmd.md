# Wan2.2 T2V-A14B DMD training

This training path distills the Wan2.2 high- and low-noise experts as two
jobs. It uses the same DMD generator/teacher/fake-score structure as the Wan2.1
trainer, with the Wan2.2 boundary bridge added around it.

## Configurations

| Expert | LoRA | Full parameters |
| --- | --- | --- |
| high | `configs/train/dmd/wan2_2_t2v_a14b_high_dmd_lora.yaml` | `configs/train/dmd/wan2_2_t2v_a14b_high_dmd.yaml` |
| low | `configs/train/dmd/wan2_2_t2v_a14b_low_dmd_lora.yaml` | `configs/train/dmd/wan2_2_t2v_a14b_low_dmd.yaml` |

All four configurations use:

- `training.method: wan22_a14b_dmd`;
- boundary timestep 500 and timestep shift 5;
- high rollout steps `[1000, 750]`, standard CFG 5, and score timesteps
  sampled from the inner 4%-96% of the high interval;
- low rollout steps `[500, 250]`, standard CFG 4, and score timesteps
  sampled from the inner 4%-96% of the low interval;
- one student update per five fake-score updates;
- BF16 latent/forward/FSDP mixed precision with FP32 timestep math and FP32
  reductions.

The CFG values are in LightX2V's standard convention:

```text
unconditional + scale * (conditional - unconditional)
```

Thus high/low values 5/4 reproduce the effective guidance of the reference
Wan2.2 implementation, whose configuration values 4/3 are applied as
`conditional + scale * (conditional - unconditional)`.

## Model layout

Set `model.base_model_path` to the shared A14B root. The configs resolve each
DiT from its expert subdirectory:

```text
Wan2.2-T2V-A14B/
├── high_noise_model/
├── low_noise_model/
├── models_t5_umt5-xxl-enc-bf16.pth
├── google/umt5-xxl/
└── Wan2.1_VAE.pth
```

The supplied DMD configs disable the VAE because training starts from sampled
latent noise and periodic single-expert inference is disabled. They also
disable the T5 and read precomputed prompt embeddings from `latent_dataset`;
build that cache before training as described in [Prompt caching](#prompt-caching).

## Training order

Train high first. Low training is not independent: before every low rollout it
runs a frozen, already-distilled high expert from noise to the boundary.

For a high LoRA checkpoint, use the base high expert plus the adapter:

```yaml
training:
  dmd:
    high_prefix:
      pretrained_model_name_or_path: ${model.base_model_path}/high_noise_model
      lora_path: /path/to/high_lora/checkpoint-000005000
      lora:
        rank: 128
        alpha: 128
        target_modules: [q, k, v, o, ffn.0, ffn.2]
      denoising_step_list: [1000, 750]
```

For a full-parameter high checkpoint, point directly at the exported HF
transformer:

```yaml
training:
  dmd:
    high_prefix:
      pretrained_model_name_or_path: /path/to/high_full/checkpoint-000005000/student_consolidated/transformer
      denoising_step_list: [1000, 750]
```

The full high config sets `training.save_consolidated_student: true`, so that
directory is produced alongside the normal resumable FSDP distributed state.
A full HF export gathers the 14B student weights, so saving it every 50 steps as
configured is substantially more expensive than saving a LoRA checkpoint.
A raw `.pt` or `.safetensors` high checkpoint is also accepted:

```yaml
training:
  dmd:
    high_prefix:
      pretrained_model_name_or_path: ${model.base_model_path}/high_noise_model
      checkpoint_path: /path/to/distill_model.safetensors
      checkpoint_strict: true
      denoising_step_list: [1000, 750]
```

The high-prefix format is independent of the low student format: for example,
a high LoRA may be used while training low with full parameters. The frozen
high prefix loads in BF16 by default even when the trainable low expert keeps
FP32 master parameters; override `high_prefix.transformer_param_dtype` only if
another load dtype is required.

## FSDP2 and sequence parallelism

The sample configs default to an 8-GPU **shared** topology:

```text
PARALLEL_LAYOUT=shared, SP_SIZE=8, FSDP_SIZE=8, WORLD_SIZE=8
```

The same eight ranks shard both the 14B parameters and the video sequence.
This is useful for A14B DMD because FSDP can use all eight GPUs without giving
up SP8. The runtime uses separate NCCL process groups for the two collective
streams. It also changes FSDP's gradient reduction from average to sum: these
ranks hold sequence partitions of one sample, not eight data replicas. The
student, teacher, fake-score model, and low job's frozen high-prefix model all
use this layout. Both the PyTorch 2.7
`set_reduce_scatter_divide_factor` API and the renamed PyTorch 2.8
`set_gradient_divide_factor` API are supported.

Shared layout currently has the strict contract:

```text
SP_SIZE = FSDP_SIZE = WORLD_SIZE
```

Do not enable `distributed.dp` at the same time as FSDP2; the runtime rejects
that conflicting configuration.

The original orthogonal two-dimensional mesh remains available:

```text
PARALLEL_LAYOUT=orthogonal
SP_SIZE * FSDP_SIZE = WORLD_SIZE
```

For example, `SP_SIZE=2, FSDP_SIZE=4, WORLD_SIZE=8` processes four independent
prompts while each prompt is sequence-sharded over two ranks. A 32-GPU full
run can start with orthogonal SP4 x FSDP8. `SP_SIZE` must divide the Wan
attention head count and the padded video sequence length.

Full DMD holds two trainable optimizers plus a frozen teacher (and one more
frozen high model in the low job), so its memory requirement is substantially
higher than LoRA. In shared layout the effective prompt batch is
`data.train.batch_size * gradient_accumulation_iters`. In orthogonal layout it
is `data.train.batch_size * FSDP_SIZE * gradient_accumulation_iters`. With
batch size 1, shared SP8/FSDP8 uses accumulation 64 by default to reproduce the
reference total batch of 64. Override `WAN22_GRAD_ACCUM_ITERS` for a lighter
exploratory run.

## Commands to run

The sample recipes expose their paths through `WAN22_MODEL_PATH`,
`WAN22_PROMPT_CACHE_PATH`, and `WAN22_OUTPUT_DIR`, so the YAML files do not
need to be edited for a normal launch. Low LoRA additionally reads
`WAN22_HIGH_LORA_PATH`; low full training reads `WAN22_HIGH_FULL_PATH`.
`WAN22_MAX_TRAIN_ITERS`, `WAN22_GRAD_ACCUM_ITERS`,
`WAN22_SAVE_EVERY_ITERS`, and `WAN22_NUM_WORKERS` can override the corresponding
runtime values, which is useful for a short smoke test.

High LoRA on one 8-GPU node:

```bash
cd /path/to/LightX2V
WAN22_MODEL_PATH=/path/to/models/Wan2.2-T2V-A14B \
WAN22_PROMPT_CACHE_PATH=/path/to/cache/wan22_a14b_prompts \
WAN22_OUTPUT_DIR=/path/to/output/high_lora \
PARALLEL_LAYOUT=shared SP_SIZE=8 FSDP_SIZE=8 \
torchrun --standalone --nproc-per-node=8 \
  lightx2v_train/train.py \
  --config lightx2v_train/configs/train/dmd/wan2_2_t2v_a14b_high_dmd_lora.yaml
```

After selecting the high checkpoint in the low config, train low LoRA:

```bash
cd /path/to/LightX2V
WAN22_MODEL_PATH=/path/to/models/Wan2.2-T2V-A14B \
WAN22_PROMPT_CACHE_PATH=/path/to/cache/wan22_a14b_prompts \
WAN22_HIGH_LORA_PATH=/path/to/output/high_lora/checkpoint-000005000 \
WAN22_OUTPUT_DIR=/path/to/output/low_lora \
PARALLEL_LAYOUT=shared SP_SIZE=8 FSDP_SIZE=8 \
torchrun --standalone --nproc-per-node=8 \
  lightx2v_train/train.py \
  --config lightx2v_train/configs/train/dmd/wan2_2_t2v_a14b_low_dmd_lora.yaml
```

Use the corresponding config without the `_lora` suffix for full-parameter
training. A multi-node full example with four 8-GPU nodes is:

```bash
cd /path/to/LightX2V
WAN22_MODEL_PATH=/path/to/models/Wan2.2-T2V-A14B \
WAN22_PROMPT_CACHE_PATH=/path/to/cache/wan22_a14b_prompts \
WAN22_OUTPUT_DIR=/path/to/output/high_full \
PARALLEL_LAYOUT=orthogonal SP_SIZE=4 FSDP_SIZE=8 torchrun \
  --nnodes=4 --nproc-per-node=8 --node-rank=${NODE_RANK} \
  --master-addr=${MASTER_ADDR} --master-port=${MASTER_PORT} \
  lightx2v_train/train.py \
  --config lightx2v_train/configs/train/dmd/wan2_2_t2v_a14b_high_dmd.yaml
```

Resume uses the existing DMD checkpoint path. Set `resume.auto_resume: true`
to select the latest checkpoint under `training.output_dir`. New checkpoints
record the parallel layout, SP size, and FSDP size; resume rejects an
incompatible topology instead of silently loading differently sharded state.

## Prompt caching

The default recipes require offline T5 embeddings and never load the text
encoder during DMD training. Build the cache once before starting high
training; the high and low jobs share the same cache:

```bash
cd /path/to/LightX2V
python lightx2v_train/data_process/wan/build_latent_dataset.py \
  /path/to/prompts/vidprom_filtered_extended.txt \
  --output-dir /path/to/cache/wan22_a14b_prompts \
  --model-dir /path/to/models/Wan2.2-T2V-A14B \
  --cache-components prompt \
  --text-device cuda \
  --save-dtype bf16
```

The prompt-only preprocessing path does not load the VAE. It writes
`metadata.jsonl`, one positive condition per prompt, and
`negative_condition.pt` for teacher CFG. Point `WAN22_PROMPT_CACHE_PATH` at
this output directory; all four supplied configs already use `latent_dataset`
and set `model.load_text_encoder: false`.

## Checkpoints and inference

LoRA checkpoints contain `pytorch_lora_weights.safetensors`. Full FSDP2 jobs
save resumable distributed student/fake/optimizer state; configs with
`save_consolidated_student: true` additionally export the student as a normal
HF transformer. FSDP2 LoRA export gathers only adapter tensors, not the frozen
14B base weights.

The trainer currently saves the online student, not an EMA copy. It can load a
reference checkpoint containing `generator_ema` as a frozen high prefix, but
does not maintain EMA during this training job.

`inference.method` is intentionally `none`: a high or low expert alone cannot
produce a complete Wan2.2 A14B sample. Validate by loading both distilled
expert outputs into the regular Wan2.2 high/low inference pipeline.
