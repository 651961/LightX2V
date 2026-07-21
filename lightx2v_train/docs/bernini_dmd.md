# Full Bernini T2V/R2V-A14B DMD 训练

本文只针对 **full Bernini（MLLM semantic planner + Wan2.2 renderer）**，不针对
Bernini-R。训练阶段只加载提前拆出的一个 Wan renderer expert，并从磁盘读取
MLLM + T5 的最终四路条件；R2V 还会读取离线缓存的 reference VAE patches 和
source-ID RoPE。正式训练不会加载 MLLM、connector、T5 或 VAE。

> 当前 full Bernini 仓库只公开了推理代码，公开训练代码属于 Bernini-R，也没有
> 官方 Bernini DMD 轨迹。因此这里的 DMD rollout/schedule 是一个明确、可配置的
> 工程假设：暂时复用 LightX2V 现有 Wan2.2 A14B DMD 的 boundary 和每个 expert
> 两步 rollout。这里没有根据 Bernini 的 `0.875` 推理切换点自行发明四步蒸馏轨迹。

## 文件与配置

| 任务 | Expert | LoRA | 全参数 |
| --- | --- | --- | --- |
| T2V | high | `configs/train/dmd/bernini_t2v_a14b_high_dmd_lora.yaml` | `configs/train/dmd/bernini_t2v_a14b_high_dmd.yaml` |
| T2V | low | `configs/train/dmd/bernini_t2v_a14b_low_dmd_lora.yaml` | `configs/train/dmd/bernini_t2v_a14b_low_dmd.yaml` |
| R2V | high | `configs/train/dmd/bernini_r2v_a14b_high_dmd_lora.yaml` | `configs/train/dmd/bernini_r2v_a14b_high_dmd.yaml` |
| R2V | low | `configs/train/dmd/bernini_r2v_a14b_low_dmd_lora.yaml` | `configs/train/dmd/bernini_r2v_a14b_low_dmd.yaml` |

这些配置默认使用 Liger attention Q/K RMSNorm，训练环境需安装：

```bash
pip install 'liger-kernel>=0.7.0'
```

也可以在安装 LightX2V 时使用 `pip install -e '.[liger]'`。

上述配置默认均为：

- `model.name: bernini_t2v_a14b` 或 `bernini_r2v_a14b`；
- `training.method`：T2V 使用 `bernini_dmd`，R2V 使用 `bernini_r2v_dmd`；
- shared FSDP2-8 + sequence-parallel-8；
- batch size 1、gradient accumulation 64；
- 每 50 iteration 保存一次，最多保留 4 个 checkpoint；
- `inference.method: none`，训练中不执行单 expert 推理；
- `rms_norm_backend: liger`，替换每个已加载 model copy 中的 160 个
  attention Q/K RMSNorm，不替换 FP32LayerNorm；
- 81 帧、480×848，与 full Bernini 官方 T2V 脚本的生成几何一致；
- teacher 使用 full Bernini T2V 官方 WAPG 强度 `omega_txt=4.0`、
  `omega_tgt=0.5`。

R2V 对齐官方 `scripts/bernini/run_r2v.sh`：high expert 使用
`omega_img/txt/tgt=4.5/4.0/1.5`；low expert 切换后将三者都乘
`omega_scale=0.8`，有效值为 `3.6/3.2/1.2`。纯 R2V 没有 source video，
因此脚本传入的 `omega_vid` 不参与 `vae_txt_vit_wapg` 计算。

Liger 与 PyTorch backend 保存相同的 `norm_q.weight`/`norm_k.weight`，不改变
checkpoint 格式。可设置 `BERNINI_RMS_NORM_BACKEND=torch` 退回与 Bernini
原仓库一致的 FP32-variance RMSNorm 公式做 A/B。实际吞吐收益需要在目标 GPU
上测量，尤其是 RMSNorm weight 冻结的 LoRA 训练。

启动日志会同时打印 renderer 的 `attention_backend`。正式分辨率训练应为 `fa2`
或 H100 上的 `fa3`；若只能回退 PyTorch `sdpa`，代码会显式告警。

LoRA 的 student 和 fake-score model 均使用 rank/alpha 128，目标模块为
Bernini renderer checkpoint 的原始模块名：

```text
to_q, to_k, to_v, to_out.0, ffn.net.0.proj, ffn.net.2
```

## 1. 拆分 full Bernini 的 high/low expert

下载完整的 `ByteDance/Bernini-Diffusers` 后运行：

```bash
cd /path/to/LightX2V

python lightx2v_train/data_process/bernini/split_experts.py \
  /path/to/ByteDance/Bernini-Diffusers \
  /path/to/cache/bernini_experts \
  --prefer-ema \
  --max-shard-size 4GB
```

脚本严格使用 full Bernini 的前缀：

| Expert | 输入 checkpoint 前缀 | 配置来源 |
| --- | --- | --- |
| high | `diff_dec.transformer.` | `transformer_config.json` |
| low | `diff_dec_low.transformer_2.` | `transformer_2_config.json` |

特别是 low expert **不是** Bernini-R 使用的
`diff_dec.transformer_2.`。当 config root 存在 `config.json` 时，脚本还会强制
检查 `model_type == "bernini"`，明确拒绝 renderer/Bernini-R。如果显式
`--config-root` 没有根 `config.json`，脚本会告警，但仍严格要求上述两个 full
Bernini 前缀。

checkpoint 同时包含 online 和 EMA 权重时，默认优先选择外层名字包含 `ema` 的
共同 namespace，例如：

```text
generator_ema.diff_dec.transformer.*
generator_ema.diff_dec_low.transformer_2.*
```

使用 `--no-prefer-ema` 可优先 online 权重，使用 `--namespace <prefix>` 可显式
锁定外层 namespace。最终选择会写入 `split_manifest.json`，不要仅凭日志猜测
拆出的是哪一套权重。

输出为可由 LightX2V 原生 Bernini loader 严格加载的目录：

```text
bernini_experts/
├── split_manifest.json
├── high_noise_model/
│   ├── config.json
│   └── diffusion_pytorch_model[-xxxxx-of-xxxxx].safetensors
└── low_noise_model/
    ├── config.json
    └── diffusion_pytorch_model[-xxxxx-of-xxxxx].safetensors
```

脚本支持 released sharded safetensors、单个 safetensors/index，以及单文件
PyTorch checkpoint。可以先使用 `--dry-run` 检查 namespace 和 tensor 数量。

## 2. 离线生成条件

缓存脚本直接复用 full Bernini 原仓库的推理预处理、MLLM planner、connector、
T5、VAE 和 source-ID RoPE 路径；LightX2V 新增的只是把最终 renderer 输入离线
保存成 DMD 数据集格式。这不是 Bernini-R 的 `image_embeds`/`video_embeds` ViT
cache。

### T2V

单 GPU 示例：

```bash
cd /path/to/LightX2V

python lightx2v_train/data_process/bernini/cache_conditions.py \
  /path/to/prompts/train.txt \
  --output-dir /path/to/cache/bernini_t2v_conditions \
  --bernini-model /path/to/ByteDance/Bernini-Diffusers \
  --bernini-repo /path/to/Bernini \
  --device cuda \
  --save-dtype bf16
```

不传 `--task` 时保持原有 T2V planner-only 路径：

- 加载 Qwen2.5-VL MLLM、connector、visual-token decoder 和冻结 UMT5；
- **不构造或加载两个 14B DiT expert**；
- **不构造 VAE**，只给 Bernini formatter 提供一个不参与条件计算的 shape-only
  target distribution；
- 默认在 MLLM 和 T5 两阶段之间 CPU offload，以降低显存峰值；显存足够时可加
  `--keep-models-on-device` 提速。

默认值严格对齐 full Bernini 官方 `scripts/bernini/run_t2v.sh` 和 CLI：

| 参数 | 默认值 |
| --- | --- |
| 输出 | 81 帧、480×848、16 FPS |
| `max_image_size` | 842（写入 metadata；planner-only 不运行 VAE） |
| seed | 42，每条 prompt 固定重置 |
| planning step | 25 |
| ViT denoising step | 5 |
| ViT text/image CFG | 1.2 / 1.0 |
| system prompt | `get_system_prompt_for_task("t2v")` |
| negative prompt | Bernini `DEFAULT_NEG_PROMPT` |
| context | 不足 512 补零，默认不截断超过 512 的 context |

最后一项对应官方 `use_truncate=False`。如确实需要固定 512，可显式传
`--truncate`，但训练和最终推理必须使用同一处理方式。

每个 `conditions/condition_XXXXXXXX.pt` 保存：

```python
{
    "prompt": str,
    "clean_prompt": str,
    "conditions": {
        "cond_embeds_wtxt_wvit": Tensor[L1, 4096],
        "cond_embeds_wtxt_wovit": Tensor[L2, 4096],
        "cond_embeds_wotxt_wvit": Tensor[L3, 4096],
        "cond_embeds_wotxt_wovit": Tensor[L4, 4096],
    },
    "cache_meta": {...},
}
```

默认 dtype 为 BF16。`cache_meta` 和目录级 `cache_config.json` 记录模型路径、
几何、ViT resize、system/negative prompt、seed、planning 参数、CFG、截断规则和
dtype。已有 cache 在未传 `--overwrite` 时会先校验这些设置及四路 tensor，再
直接跳过 planner/T5 计算；设置不同会报错，避免混用不一致的 context。

输出目录可直接给 `latent_dataset`：

```text
bernini_t2v_conditions/
├── cache_config.json
├── metadata.jsonl
└── conditions/
    ├── condition_00000000.pt
    └── ...
```

使用 `torchrun` 可按 prompt index 分片到多张 GPU；各 rank 处理自己的 prompt，
结束后 rank 0 合并 `metadata_rankXXXXX.jsonl`。恢复不完整缓存时会先在所有 rank
汇总 cache miss：只要任一 rank 仍有待处理项，所有 rank 都参加 planner 的模型
初始化和模板构建（上游加载过程包含 distributed barrier），但仅有本地 miss 的
rank 执行条件生成与落盘：

```bash
torchrun --standalone --nproc-per-node=8 \
  lightx2v_train/data_process/bernini/cache_conditions.py \
  /path/to/prompts/train.jsonl \
  --output-dir /path/to/cache/bernini_t2v_conditions \
  --bernini-model /path/to/ByteDance/Bernini-Diffusers \
  --bernini-repo /path/to/Bernini
```

### R2V

R2V metadata 每行必须包含 prompt 和 reference 图片列表。默认读取 `ref_images`，
缺失时回退到 `images`；JSON/JSONL 中应为字符串数组，CSV 中可填写 JSON 数组。
相对路径默认相对 metadata 所在目录，也可用 `--media-root` 显式指定根目录。例如：

```json
{"id":"0001","prompt":"...","ref_images":["refs/a.png","refs/b.png"],"video":"videos/target.mp4"}
```

```bash
torchrun --standalone --nproc-per-node=8 \
  lightx2v_train/data_process/bernini/cache_conditions.py \
  /path/to/r2v/metadata.jsonl \
  --task r2v \
  --media-root /path/to/r2v \
  --reference-images-column ref_images \
  --output-dir /path/to/cache/bernini_r2v_conditions \
  --bernini-model /path/to/ByteDance/Bernini-Diffusers \
  --bernini-repo /path/to/Bernini
```

上面一个命令会一次性完成 MLLM、T5 和 reference VAE 的离线计算，不需要分别运行
三个预处理脚本：

- **MLLM**：Qwen2.5-VL、connector 和 visual-token decoder 根据 prompt 与
  reference 的 ViT 特征生成 Bernini 的四路 planner context；
- **T5**：冻结 UMT5 分别编码 `system prompt + prompt` 和 negative prompt，再与
  对应 planner context 拼接。context 不足 512 时补零，默认保留超过 512 的内容；
- **VAE**：只用冻结 Wan VAE 编码 reference 图片并生成 source patches 和
  source-ID RoPE；DMD 不读取或编码 metadata 中的目标视频。

整个预处理仍跳过两个 14B DiT expert。重复图片的 ViT/VAE 结果持久化在输出目录的
`reference_features/`。同一 rank 内的重复图片只编码一次，恢复或后续重跑会直接
复用；多 rank 首次同时遇到同一张尚未缓存的图片时最多会各自编码一次，但落盘采用
原子替换，不会产生半写文件。condition 文件在四路 context 外还按图片顺序保存：

```text
source_image_vae_patches[i] : [Ni, 16, 1, 2, 2]
source_image_rope_cos[i]    : [Ni, 1, 128], FP64
source_image_rope_sin[i]    : [Ni, 1, 128], FP64
```

R2V condition schema 为 `lightx2v.bernini.r2v_conditions`。已有 FP32 RoPE
cache 会被拒绝，需要重新运行上面的预处理命令，以保留 Bernini 原生 complex128
RoPE 的数值精度。

RoPE 已包含该图片的 source ID。1～5 张图依次使用 ID 1～5；超过 5 张时默认严格
按照 full Bernini 推理逻辑，将 ID 均匀插值到 `[1,5]`。图片次序属于条件语义，缓存
脚本不会排序。cache provenance 同时绑定 reference 的绝对路径、文件大小、mtime 和
fingerprint；图片发生变化后旧 condition 不会被静默复用。

metadata 中的目标 `video` 对 DMD 不是必需输入：目标 latent 由训练时的随机噪声和
student rollout 在线产生，几何由 YAML 的 81×480×848 决定，不需要缓存
`video_latent_path`。reference 数量可变，因此当前 R2V 配置固定 `batch_size: 1`；
梯度累积仍可保持 64。

## 3. 训练顺序

先训练 high，再将 high 的蒸馏结果作为 frozen prefix 训练 low。这些 YAML 均从
以下环境变量读取路径：

- `BERNINI_EXPERT_ROOT`：第 1 步的 expert 拆分目录；
- `BERNINI_CONDITION_CACHE`：T2V 第 2 步的四路 condition cache；
- `BERNINI_R2V_CONDITION_CACHE`：R2V 第 2 步的四路 context、reference VAE
  patches 与 source-ID RoPE cache；
- `BERNINI_OUTPUT_DIR`：当前任务输出目录；
- `BERNINI_HIGH_LORA_PATH`：low-LoRA 使用的已蒸馏 high LoRA checkpoint；
- `BERNINI_HIGH_FULL_PATH`：low-full 使用的已蒸馏 high native Bernini
  transformer。

High LoRA：

```bash
BERNINI_EXPERT_ROOT=/path/to/cache/bernini_experts \
BERNINI_CONDITION_CACHE=/path/to/cache/bernini_t2v_conditions \
BERNINI_OUTPUT_DIR=/path/to/output/bernini_high_lora \
PARALLEL_LAYOUT=shared SP_SIZE=8 FSDP_SIZE=8 \
torchrun --standalone --nproc-per-node=8 \
  lightx2v_train/train.py \
  --config lightx2v_train/configs/train/dmd/bernini_t2v_a14b_high_dmd_lora.yaml
```

随后训练 Low LoRA：

```bash
BERNINI_EXPERT_ROOT=/path/to/cache/bernini_experts \
BERNINI_CONDITION_CACHE=/path/to/cache/bernini_t2v_conditions \
BERNINI_HIGH_LORA_PATH=/path/to/output/bernini_high_lora/checkpoint-000005000 \
BERNINI_OUTPUT_DIR=/path/to/output/bernini_low_lora \
PARALLEL_LAYOUT=shared SP_SIZE=8 FSDP_SIZE=8 \
torchrun --standalone --nproc-per-node=8 \
  lightx2v_train/train.py \
  --config lightx2v_train/configs/train/dmd/bernini_t2v_a14b_low_dmd_lora.yaml
```

R2V High LoRA 使用对应的 R2V condition cache：

```bash
BERNINI_EXPERT_ROOT=/path/to/cache/bernini_experts \
BERNINI_R2V_CONDITION_CACHE=/path/to/cache/bernini_r2v_conditions \
BERNINI_OUTPUT_DIR=/path/to/output/bernini_r2v_high_lora \
PARALLEL_LAYOUT=shared SP_SIZE=8 FSDP_SIZE=8 \
torchrun --standalone --nproc-per-node=8 \
  lightx2v_train/train.py \
  --config lightx2v_train/configs/train/dmd/bernini_r2v_a14b_high_dmd_lora.yaml
```

随后训练 R2V Low LoRA；其 frozen high prefix 必须指向上一步的 R2V high
checkpoint：

```bash
BERNINI_EXPERT_ROOT=/path/to/cache/bernini_experts \
BERNINI_R2V_CONDITION_CACHE=/path/to/cache/bernini_r2v_conditions \
BERNINI_HIGH_LORA_PATH=/path/to/output/bernini_r2v_high_lora/checkpoint-000005000 \
BERNINI_OUTPUT_DIR=/path/to/output/bernini_r2v_low_lora \
PARALLEL_LAYOUT=shared SP_SIZE=8 FSDP_SIZE=8 \
torchrun --standalone --nproc-per-node=8 \
  lightx2v_train/train.py \
  --config lightx2v_train/configs/train/dmd/bernini_r2v_a14b_low_dmd_lora.yaml
```

T2V/R2V 全参数训练均使用去掉 `_lora` 后缀的对应配置。High-full 配置启用了
`save_consolidated_student`，供 low-full 的 `BERNINI_HIGH_FULL_PATH` 指向：

```text
checkpoint-XXXXXXXXX/student_consolidated/transformer
```

shared 布局要求 `SP_SIZE=FSDP_SIZE=WORLD_SIZE`。默认 accumulation 64 对应有效
prompt batch 64；显存不足时可以先通过 `BERNINI_GRAD_ACCUM_ITERS` 做小规模验证，
但正式实验需要重新确认总 batch 和学习率。

## 4. 当前 schedule 假设

上述配置当前使用：

```text
boundary_step = 500
high rollout = [1000, 750]
low rollout  = [500, 250]
expert score sampling = 每个区间内部的 4%～96%
```

这些值来自当前 LightX2V Wan2.2 A14B DMD 实现，不是 Bernini 官方发布的 DMD
超参数。Bernini 推理配置中的 `switch_dit_boundary=0.875` 不能直接推出一个正确的
四步蒸馏轨迹，所以本实现没有把它硬编码成未经验证的 `[1000,...,875,...]`
序列。后续若官方训练参数可用，应同时更新：

- `boundary_step`；
- high/low `denoising_step_list`；
- low job 的 `high_prefix.denoising_step_list`；
- expert score timestep 区间和 shift。

## 5. 权重与验证

- LoRA checkpoint：`pytorch_lora_weights.safetensors`，只含 adapter 权重。
- Full FSDP checkpoint：包含可恢复的 distributed student/fake/optimizer state；
  `save_consolidated_student: true` 额外导出可由原生 Bernini loader 重载的
  safetensors transformer。
- 当前 trainer 保存 online student，不在训练中维护 EMA。expert 拆分脚本可以优先
  读取原始 full Bernini checkpoint 中已有的 EMA namespace，但这与“DMD 训练时
  新维护 EMA”是两件不同的事。

训练中不做单 expert 推理，因为 high 或 low 单独都不是完整生成 pipeline。验证时
应把蒸馏后的 high 和 low 一起装回 full Bernini/Wan2.2 renderer 流程，并使用与
cache 一致的 system prompt、negative prompt、planner seed、WAPG 和 81×480×848
几何。
