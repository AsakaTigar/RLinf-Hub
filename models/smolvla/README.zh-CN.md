# 面向 RLinf 的 SmolVLA (450M)

[English](README.md) | [简体中文](README.zh-CN.md)

一个自包含、仅用于推理的 RLinf-Hub 适配器，面向官方
[`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) checkpoint。
适配器直接调用上游 LeRobot 实现，不复制模型代码，也不修改 RLinf。

## 支持的契约

- RLinf `0.3.0`，启用 `embodied` extra。
- LeRobot `0.4.4`，启用 `smolvla` extra，搭配 Transformers `4.57.1`。
- 已在 CUDA 12.8 构建的 Torch `2.10.0` 与 TorchVision `0.25.0` 上验证。
- Checkpoint revision 为 `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`。
- 适配器仅移除 checkpoint 中的 `type` 注册标记，供 LeRobot 0.4.4 解析配置；
  模型字段与权重均不作改动。
- 三路 `observation.images.camera*` 输入、6 维 state、长度为 50 的 6 维
  action chunk。
- 支持 FP32 推理；`precision=fp16` 在 FP32 权重上启用 autocast。同一开关在
  支持 BF16 的硬件上会选择 BF16 autocast。

PPO/GRPO logprob、actor replay 与训练均未实现。适配器对这些路径抛出
`NotImplementedError`，而不是返回伪造的统计量。

## 目录结构

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

## 安装

先按官方方式安装 RLinf `0.3.0` 的 embodied 环境，可使用 RLinf Docker 镜像，
或在 RLinf 代码仓库中执行：

```bash
bash requirements/install.sh embodied --env dummy
```

然后在同一环境中叠加模型侧依赖：

```bash
python -m pip install -r models/smolvla/requirements.txt
huggingface-cli download lerobot/smolvla_base \
  --revision 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6
```

`requirements.txt` 只包含 RLinf 之外的依赖，并在文件头记录验证过的 RLinf
版本。此处刻意不使用 LeRobot `0.7.0`：它的 SmolVLA extra 需要 Transformers
5.x，而 RLinf `0.3.0[embodied]` 将 Transformers 限制在 `4.57.6`。锁定文件
使用 `datasets==4.8.0`，因为 `datasets 5.2` 需要 Hugging Face Hub 1.x，与
Transformers 4.57 的 `<1.0` 要求冲突。`torchcodec==0.10.0` 满足 LeRobot
0.4.4 的 `<0.11.0` 上限。

在极简 Linux 镜像中，`torchcodec` 还需要系统 FFmpeg 共享库。如果导入 LeRobot
时报缺少 `libavcodec`/`libavformat`，请安装 FFmpeg（例如
`apt-get install -y ffmpeg`）。

## 向 RLinf 注册

在 Hub 根目录执行：

```bash
export PYTHONPATH="$PWD/models/smolvla${PYTHONPATH:+:$PYTHONPATH}"
export RLINF_EXT_MODULE=smolvla
python -c "from smolvla import register; register(); print('smolvla registered')"
```

`RLINF_EXT_MODULE` 会在每个 Ray worker 中完成注册。外部驱动脚本在调用
`validate_cfg()` 之前也必须先注册：

```python
from smolvla import register

register()
cfg = validate_cfg(cfg)
```

## Hydra 模型片段

`models/smolvla/configs/smolvla.yaml` 是模型组片段。构建完整 RLinf recipe 时，
把配置目录加入 Hydra 的 search path，并在 actor 与 rollout 两侧都 include：

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

该片段使用 `num_action_chunks: 50`，因为适配器返回完整的
`predict_action_chunk()` 结果。它不调用 `select_action()`，也不维护第二条
action 队列。

## 观测与归一化契约

| RLinf 输入 | Shape | LeRobot 目标 |
|---|---|---|
| `main_images` | `[B,H,W,C]` 或 `[B,1,H,W,C]` | `observation.images.camera1` |
| `extra_view_images` | `[B,2,H,W,C]` 或两个 `[B,H,W,C]` 张量 | `observation.images.camera2`、`camera3` |
| `states` | `[B,6]` | `observation.state` |
| `task_descriptions` | 长度为 `B` 的 `list[str]` | `task` |

整数图像会除以 255；浮点图像被视为已位于 `[0,1]`。HWC 输入会转换为 CHW。
上游 processor 负责 tokenization、图像归一化，以及可选的 state/action
归一化。其自动 add-batch 步骤被移除，因为 RLinf 的观测已经带 batch 维。

适配器根据 checkpoint 配置重建 processor，而不是加载 base checkpoint 中
过期的已保存 pipeline。base checkpoint 只包含含义不明确的旧版逐机器人
action 统计量，因此 `stats_path: null` 会使用单位 mean/std，仅用于 shape
冒烟测试。若需要具有物理意义的 action 值，请把 `stats_path` 指向使用当前
feature key 的 JSON 文件：

```json
{
  "observation.state": {"mean": [0, 0, 0, 0, 0, 0], "std": [1, 1, 1, 1, 1, 1]},
  "action": {"mean": [0, 0, 0, 0, 0, 0], "std": [1, 1, 1, 1, 1, 1]}
}
```

## 验证

CPU 契约测试不会下载权重：

```bash
pytest -q models/smolvla/tests/test_smolvla.py -k 'not gpu'
```

真实 checkpoint 的 GPU 冒烟测试：

```bash
CUDA_VISIBLE_DEVICES=0 SMOLVLA_GPU_SMOKE=1 \
  pytest -q --log-cli-level=INFO models/smolvla/tests/test_smolvla.py -k 'gpu and fp32'

CUDA_VISIBLE_DEVICES=0 SMOLVLA_GPU_SMOKE=1 \
  pytest -q --log-cli-level=INFO models/smolvla/tests/test_smolvla.py -k 'gpu and fp16'

CUDA_VISIBLE_DEVICES=0 SMOLVLA_GPU_SMOKE=1 \
  pytest -q --log-cli-level=INFO models/smolvla/tests/test_smolvla.py -k 'gpu and bf16'
```

每种精度预期的 GPU 输出是一个有限的 CPU FP32 张量，shape 为 `[1, 50, 6]`；
日志同时包含 checkpoint revision、延迟、GPU 名称与峰值 CUDA 显存。

## 验证证据

### Linux A100 主验证

验证环境为 Ubuntu 26.04.1 LTS、2× NVIDIA A100-SXM4-80GB（驱动 595.99.02，
功耗上限 400 W）、Python 3.12.16、PyTorch 2.10.0+cu128（CUDA 12.8）、
Ray 2.60.1、RLinf 0.3.0、LeRobot 0.4.4、Transformers 4.57.1 与
torchcodec 0.10.0。Checkpoint 大小为 906,712,520 字节，SHA256 为
`7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`
（revision `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`）；加载耗时 32.0 秒。

全量测试 **11/11 通过，耗时 112.77 秒**（8 项 CPU 契约测试 + 3 项真实
checkpoint GPU 测试），其中包含真实的 LeRobot normalize/unnormalize 往返。
每个 GPU 测试都返回有限的 CPU FP32 `[1, 50, 6]` 张量。

Ray 分布式验证：本地 Ray 集群启动 2 个 worker，每个 worker 独占一张 A100，
仅通过 `RLINF_EXT_MODULE=smolvla` 完成自动注册，没有手动 import。两个 worker
都严格加载官方 checkpoint，执行 BF16 autocast 前向，返回有限的
`[1, 50, 6]` 动作（287.34 ms 与 283.43 ms）。驱动进程以 `RAY_SMOKE_OK`
和退出码 0 结束。该验证覆盖分布式注册与推理，不包含任务级 benchmark rollout。

### 延迟：每种精度 90 个正式样本（A100）

Batch size 为 1。每种精度独立执行三轮；每轮先做两次 warmup，再执行 30 次
正式 `predict_action_batch` 调用，因此每种精度共 6 次 warmup 与 90 次正式
调用。计时边界是完整的适配器调用，包含预处理、tokenization、flow-matching
推理与后处理。每次正式调用都用 `torch.cuda.synchronize()` 包裹，计时期间
从不调用 `empty_cache()`。测试前 GPU 处于 P0（34.53 W），测试后仍为 P0
（50.11 W、利用率 32%）。下表中数值为全部 90 个样本的毫秒数（`n=90`）。

| 精度 | 首次 forward | P50 | P95 | 均值 | 标准差 | 最小值 | 最大值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 946.82 | 251.80 | 261.82 | 253.49 | 5.85 | 248.71 | 297.88 |
| FP16 autocast | 3,995.21 | 275.47 | 285.15 | 276.39 | 4.61 | 270.67 | 295.17 |
| BF16 autocast | 294.11 | 269.32 | 278.06 | 270.54 | 3.91 | 265.68 | 284.41 |

三种模式都返回有限的 CPU FP32 `[1, 50, 6]` 动作。每次切换精度后的首次调用
单独列出，不计入稳态统计。

### 权重存储、计算与输出 dtype

精度开关只启用 CUDA autocast；它不会量化或重新存储权重。Linux 审计复现了
与 Windows 完全一致的 dtype 直方图。

| 阶段 | FP32 模式 | FP16 模式 | BF16 模式 |
|---|---|---|---|
| Checkpoint 张量存储（磁盘上） | 混合：474 个 BF16 张量（446,772,624 个元素）+ 26 个 F32 张量（3,273,552 个元素） | 同一 checkpoint | 同一 checkpoint |
| 适配器加载后的参数 | 500 个 `torch.float32` 张量（450,046,176 个元素）+ 2 个 FP32 buffer（64 个元素） | 同样的 FP32 主权重 | 同样的 FP32 主权重 |
| 推理时 autocast 状态 | 关闭（`actual_autocast_dtype=None`） | 开启，`torch.float16` | 开启，`torch.bfloat16` |
| 探针 `nn.Linear` 输出 dtype | `torch.float32` | `torch.float16` | `torch.bfloat16` |
| 适配器输出 | CPU `torch.float32` `[B,50,6]` | CPU `torch.float32` `[B,50,6]` | CPU `torch.float32` `[B,50,6]` |

审计直接读取 safetensors header，并在加载后的模型上统计到 500 个参数张量与
2 个 buffer。三种模式都返回有限的 `[1, 50, 6]` 输出。

### Windows RTX 5070 Ti 跨平台补充验证

Windows 11 环境使用 Python 3.12.16、Torch 2.10.0+cu128、RTX 5070 Ti
（sm_120），驱动 621.14，功耗上限 300 W。同样的 11 项测试全部通过，并在
sm_120 上验证了 FP16/BF16 矩阵乘法；该轮模型加载耗时 45.4 秒。Checkpoint
的 SHA256 与 revision 与 Linux 一致。

Batch size 为 1；每种精度同样执行三轮、每轮两次 warmup 加 30 次正式调用
（90 个样本）。测试前 GPU 处于 P8（25.35 W、SM 1,200 MHz、显存 405 MHz），
测试后稳定在 P1（67.14 W、SM 2,887 MHz、显存 13,801 MHz、利用率 14%）。

| 精度 | P50 | P95 | 均值 | 标准差 | 最小值 | 最大值 |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 525.33 | 6,329.80 | 1,553.76 | 4,019.25 | 481.26 | 25,322.57 |
| FP16 autocast | 657.75 | 680.02 | 722.40 | 433.60 | 616.39 | 4,220.54 |
| BF16 autocast | 648.54 | 668.99 | 648.24 | 15.85 | 582.13 | 686.48 |

分轮 P50（第 1/2/3 轮）：FP32 为 507.83 / 535.38 / 541.96 ms；FP16 为
660.62 / 659.36 / 656.69 ms；BF16 为 650.05 / 648.23 / 646.17 ms。各轮中位数
稳定。在这台共享桌面上，FP32 与 FP16 出现了长尾离群点（最大值分别为
25,322 ms 与 4,221 ms）；这些样本被保留而非剔除，因此 FP32 的均值、P95 与
标准差远高于其中位数。这些是共享桌面上的初步端到端数字，不是受控性能基准。

每次切换精度后的首次 forward 分别耗时 3,759.8 ms（FP32）、782.4 ms（FP16）、
740.9 ms（BF16）。模型加载（45.4 秒）、warmup 与这些首次调用均不计入上表。

### 早期 Windows 测量（保留，协议不同）

以下数据在 90 样本审计之前测得，协议更短：单次会话、两次 warmup、每种精度
十次正式调用，另在独立进程中测量冷启动。它们作为历史记录保留，不会被上表
覆盖；样本数量与会话均不同，两组数据不可直接比较。

| 精度 | Action shape | 有限 | 冷启动首次调用 | 稳态 P50（n=10） | 稳态均值（n=10） | 稳态峰值 CUDA |
|---|---|---|---:|---:|---:|---:|
| FP32 | `[1,50,6]` | 是 | 16,610 ms | 525.8 ms | 523.9 ms | 1,802 MiB |
| FP16 autocast | `[1,50,6]` | 是 | 16,369 ms | 638.8 ms | 637.3 ms | 1,971 MiB |
| BF16 autocast | `[1,50,6]` | 是 | 16,627 ms | 633.3 ms | 633.7 ms | 1,971 MiB |

原始日志与逐次延迟样本保存在独立的本地证据包中，供 PR 描述使用；它们刻意
不放入 `models/smolvla/`。

本适配器仅用于推理。Linux 验证覆盖契约测试与双 worker Ray 分布式注册/推理，
不包含任务级 benchmark rollout 或成功率。训练、actor replay 与 PPO/GRPO
logprob 仍未实现，相关路径会抛出 `NotImplementedError`。
