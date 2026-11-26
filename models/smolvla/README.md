# SmolVLA (450M) for RLinf

[English](README.md) | [简体中文](README.zh-CN.md)

A self-contained, inference-only RLinf-Hub adapter for the official
[`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) checkpoint.
The adapter calls the upstream LeRobot implementation; it does not copy model code
or modify RLinf.

## Supported contract

- RLinf `0.3.0` with the `embodied` extra.
- LeRobot `0.4.4` with the `smolvla` extra and Transformers `4.57.1`.
- Validated with CUDA 12.8 builds of Torch `2.10.0` and TorchVision `0.25.0`.
- Checkpoint revision `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`.
- The adapter removes only the checkpoint's `type` registry tag before LeRobot
  0.4.4 parses the config; model fields and weights are unchanged.
- Three `observation.images.camera*` inputs, a 6D state, and a 6D action chunk of
  length 50.
- FP32 inference; `precision=fp16` uses autocast with FP32 weights. The same
  switch selects BF16 autocast on BF16-capable hardware.

PPO/GRPO logprobs, actor replay, and training are not implemented. The adapter
raises `NotImplementedError` for those paths instead of returning fabricated
statistics.

## Package layout

```text
models/smolvla/
├── README.md
├── README.zh-CN.md
├── requirements.txt
├── configs/
│   └── smolvla.yaml
├── smolvla/
│   ├── __init__.py
│   └── adapter.py
└── tests/
    └── test_smolvla.py
```

## Install

Install RLinf `0.3.0` with its official embodied environment first, using the
RLinf Docker image or from an RLinf checkout:

```bash
bash requirements/install.sh embodied --env dummy
```

Then layer the model-specific dependencies in that environment:

```bash
python -m pip install -r models/smolvla/requirements.txt
huggingface-cli download lerobot/smolvla_base \
  --revision 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6
```

`requirements.txt` contains only dependencies beyond RLinf and records the tested
RLinf version in its header. LeRobot `0.7.0` is intentionally not used: its
SmolVLA extra requires Transformers 5.x, while RLinf `0.3.0[embodied]` caps
Transformers at `4.57.6`. The lock also uses `datasets==4.8.0` because
`datasets 5.2` requires Hugging Face Hub 1.x, which conflicts with Transformers
4.57's `<1.0` requirement. `torchcodec==0.10.0` satisfies LeRobot 0.4.4's
`<0.11.0` cap.

## Register with RLinf

Run commands from the Hub root:

```bash
export PYTHONPATH="$PWD/models/smolvla${PYTHONPATH:+:$PYTHONPATH}"
export RLINF_EXT_MODULE=smolvla
python -c "from smolvla import register; register(); print('smolvla registered')"
```

`RLINF_EXT_MODULE` performs registration in every Ray worker. An external driver
must also register before calling `validate_cfg()`:

```python
from smolvla import register

register()
cfg = validate_cfg(cfg)
```

## Hydra model fragment

`models/smolvla/configs/smolvla.yaml` is a model-group fragment. Add the config
directory to Hydra's search path and include it for both actor and rollout when
building a complete RLinf recipe:

```yaml
hydra:
  searchpath:
    - file://${oc.env:SMOLVLA_CONFIG_DIR}
```

```bash
export SMOLVLA_CONFIG_DIR="$PWD/models/smolvla/configs"
```

```yaml
defaults:
  - smolvla@actor.model
  - smolvla@rollout.model
```

The fragment uses `num_action_chunks: 50` because the adapter returns the full
`predict_action_chunk()` result. It does not call `select_action()` or maintain a
second action queue.

## Observation and normalization contract

| RLinf input | Shape | LeRobot target |
|---|---|---|
| `main_images` | `[B,H,W,C]` or `[B,1,H,W,C]` | `observation.images.camera1` |
| `extra_view_images` | `[B,2,H,W,C]` or two `[B,H,W,C]` tensors | `observation.images.camera2`, `camera3` |
| `states` | `[B,6]` | `observation.state` |
| `task_descriptions` | `list[str]` of length `B` | `task` |

Integer images are divided by 255; floating images are treated as already being in
`[0,1]`. HWC inputs are converted to CHW. The upstream processor performs tokenization,
image normalization, and optional state/action normalization. Its automatic
add-batch step is removed because RLinf observations are already batched.

The adapter rebuilds processors from the checkpoint config instead of loading the
base checkpoint's stale saved pipeline. The base checkpoint contains only
ambiguous legacy per-robot action statistics, so `stats_path: null` uses unit
mean/std statistics for a shape-only smoke test. For meaningful physical action
values, point `stats_path` at a JSON file using the current feature keys:

```json
{
  "observation.state": {"mean": [0, 0, 0, 0, 0, 0], "std": [1, 1, 1, 1, 1, 1]},
  "action": {"mean": [0, 0, 0, 0, 0, 0], "std": [1, 1, 1, 1, 1, 1]}
}
```

## Verify

CPU contract tests do not download weights:

```bash
pytest -q models/smolvla/tests/test_smolvla.py -k 'not gpu'
```

Real-checkpoint GPU smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 SMOLVLA_GPU_SMOKE=1 \
  pytest -q --log-cli-level=INFO models/smolvla/tests/test_smolvla.py -k 'gpu and fp32'

CUDA_VISIBLE_DEVICES=0 SMOLVLA_GPU_SMOKE=1 \
  pytest -q --log-cli-level=INFO models/smolvla/tests/test_smolvla.py -k 'gpu and fp16'

CUDA_VISIBLE_DEVICES=0 SMOLVLA_GPU_SMOKE=1 \
  pytest -q --log-cli-level=INFO models/smolvla/tests/test_smolvla.py -k 'gpu and bf16'
```

Expected GPU output is one finite CPU FP32 tensor with shape `[1, 50, 6]` per
precision, plus checkpoint revision, latency, GPU name, and peak CUDA memory in the
log.

## Evidence

Local validation used Windows 11, Python 3.12.16, Torch 2.10.0+cu128 (CUDA 12.8),
and an RTX 5070 Ti (sm_120) with NVIDIA driver 621.14 and a 300 W power limit.
The CPU suite passed 8 tests, including a real LeRobot normalize/unnormalize
round trip. The GPU suite passed 3 real-checkpoint tests (all 11 tests green),
and FP16/BF16 matrix multiplications were verified on sm_120. The checkpoint
contains 906,712,520 bytes and has SHA256
`7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`
(revision `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`); loading it took 45.4 s in
the audit run.

### Latency: 90 measured samples per precision

Batch size is 1. All precisions reuse the same loaded model, observations, and
noise. Each precision runs three independent rounds; every round performs two
warmup calls followed by 30 measured `predict_action_batch` calls, so each
precision contributes six warmup calls and 90 measured calls. The measured
boundary is the full adapter call, including preprocessing, tokenization,
flow-matching inference, and postprocessing. `torch.cuda.synchronize()` wraps
every measured call, and `empty_cache()` is never called while timing. The GPU
idled at P8 before the run (25.35 W, 1,200 MHz SM, 405 MHz memory) and settled at
P1 afterwards (67.14 W, 2,887 MHz SM, 13,801 MHz memory, 14% utilization).
Values are milliseconds over all 90 samples (`n=90`).

| Precision | P50 | P95 | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 525.33 | 6,329.80 | 1,553.76 | 4,019.25 | 481.26 | 25,322.57 |
| FP16 autocast | 657.75 | 680.02 | 722.40 | 433.60 | 616.39 | 4,220.54 |
| BF16 autocast | 648.54 | 668.99 | 648.24 | 15.85 | 582.13 | 686.48 |

Per-round P50 (rounds 1/2/3): FP32 507.83 / 535.38 / 541.96 ms; FP16 660.62 /
659.36 / 656.69 ms; BF16 650.05 / 648.23 / 646.17 ms. The medians are stable
across rounds. FP32 and FP16 produced long-tail outliers (maximum 25,322 ms and
4,221 ms) on this shared desktop; they are retained rather than trimmed, so the
FP32 mean, P95, and standard deviation are much larger than its median. These
are preliminary end-to-end numbers from a shared desktop, not a controlled
performance benchmark.

First forward after each mode switch took 3,759.8 ms (FP32), 782.4 ms (FP16),
and 740.9 ms (BF16). Model load (45.4 s), warmups, and these first calls are
excluded from the table above.

### Weight storage vs compute vs output dtype

The precision switch only enables CUDA autocast; it does not quantize or
re-store weights.

| Stage | FP32 mode | FP16 mode | BF16 mode |
|---|---|---|---|
| Checkpoint tensor storage (on disk) | mixed: 474 BF16 tensors (446,772,624 elements) + 26 F32 tensors (3,273,552 elements) | same checkpoint | same checkpoint |
| Loaded parameters in the adapter | 500 `torch.float32` tensors (450,046,176 elements) + 2 FP32 buffers (64 elements) | same FP32 master weights | same FP32 master weights |
| Autocast state during inference | disabled (`actual_autocast_dtype=None`) | enabled, `torch.float16` | enabled, `torch.bfloat16` |
| Probe `nn.Linear` output dtype | `torch.float32` | `torch.float16` | `torch.bfloat16` |
| Adapter output | CPU `torch.float32` `[B,50,6]` | CPU `torch.float32` `[B,50,6]` | CPU `torch.float32` `[B,50,6]` |

The audit read the safetensors header directly and counted 500 parameter tensors
plus 2 buffers on the loaded model. All three modes returned finite outputs of
shape `[1, 50, 6]`.

### Earlier measurements (retained, different protocol)

These were measured before the 90-sample audit with a shorter protocol: one
session, two warmups, and ten measured calls per precision, plus separate
cold-start calls in fresh processes. They are kept as historical records and are
not overwritten by the table above; the sample counts and session differ, so the
two sets are not directly comparable.

| Precision | Action shape | Finite | Cold first call | Steady P50 (n=10) | Steady mean (n=10) | Peak CUDA (steady) |
|---|---|---|---:|---:|---:|---:|
| FP32 | `[1,50,6]` | yes | 16,610 ms | 525.8 ms | 523.9 ms | 1,802 MiB |
| FP16 autocast | `[1,50,6]` | yes | 16,369 ms | 638.8 ms | 637.3 ms | 1,971 MiB |
| BF16 autocast | `[1,50,6]` | yes | 16,627 ms | 633.3 ms | 633.7 ms | 1,971 MiB |

Raw logs and per-call latency samples are retained in a separate local evidence
bundle for the PR description. They are intentionally not shipped inside
`models/smolvla/`.

A complete distributed RLinf evaluation is still pending on Linux. The GDUT SSH
entry points were rate-limited during this run; the local evidence validates
registration, contracts, normalization, strict checkpoint loading, and all three
inference precisions, not a benchmark success rate.
