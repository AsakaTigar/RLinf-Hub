# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from rlinf.config import SupportedModel
from rlinf.models import get_model
from smolvla import register
from smolvla.adapter import SmolVLAForRLActionPrediction, build_smolvla
from torch import nn

logger = logging.getLogger(__name__)


def _load_config() -> OmegaConf:
    path = Path(__file__).resolve().parents[1] / "configs" / "smolvla.yaml"
    cfg = OmegaConf.load(path)
    cfg.load_to_device = False
    return cfg


def _fake_model_config() -> SimpleNamespace:
    from lerobot.configs.types import FeatureType, PolicyFeature

    return SimpleNamespace(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (6,)),
            "observation.images.camera1": PolicyFeature(
                FeatureType.VISUAL, (3, 256, 256)
            ),
            "observation.images.camera2": PolicyFeature(
                FeatureType.VISUAL, (3, 256, 256)
            ),
            "observation.images.camera3": PolicyFeature(
                FeatureType.VISUAL, (3, 256, 256)
            ),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
        chunk_size=50,
        max_action_dim=32,
        device="cuda",
        dtype=torch.bfloat16,
        load_vlm_weights=True,
        compile_model=True,
        rtc_config=object(),
    )


class _FakeLeRobotPolicy(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))
        self.config = config
        self.batches = []
        self.noises = []

    def predict_action_chunk(
        self, batch: dict, noise: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        self.batches.append(batch)
        self.noises.append(noise)
        batch_size = batch["observation.state"].shape[0]
        base = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6)
        return base.expand(batch_size, 50, 6).clone()


class _FakePreprocessor:
    def __init__(self, state: dict, add_batch_step: object) -> None:
        self.steps = [add_batch_step, SimpleNamespace(name="normalize")]
        self.state = state

    def __call__(self, batch: dict) -> dict:
        self.state["preprocessor_input"] = batch
        output = dict(batch)
        output["observation.state"] = batch["observation.state"] * 2.0
        return output


@pytest.fixture
def vendor_stubs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    import draccus
    import huggingface_hub
    from lerobot.policies import factory
    from lerobot.policies.smolvla import modeling_smolvla
    from lerobot.processor.batch_processor import AddBatchDimensionProcessorStep

    class FakeAddBatchStep(AddBatchDimensionProcessorStep):
        def __init__(self) -> None:
            pass

    state = {"config": _fake_model_config(), "factory_args": None}
    policy = _FakeLeRobotPolicy(state["config"])
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"type": "smolvla"}), encoding="utf-8")
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *args, **kwargs: str(config_path)
    )

    def from_pretrained(*args, **kwargs):
        return policy

    def make_pre_post_processors(*args, **kwargs):
        state["factory_args"] = {"args": args, "kwargs": kwargs}
        state["preprocessor_input"] = None
        state["postprocessor_input"] = None

        def postprocess(actions):
            state["postprocessor_input"] = actions
            return actions / 2.0

        return _FakePreprocessor(state, FakeAddBatchStep()), postprocess

    def parse_config(config_cls, path, *args, **kwargs):
        state["parsed_config_data"] = json.loads(Path(path).read_text(encoding="utf-8"))
        return state["config"]

    monkeypatch.setattr(draccus, "parse", parse_config)
    monkeypatch.setattr(
        modeling_smolvla.SmolVLAPolicy, "from_pretrained", from_pretrained
    )
    monkeypatch.setattr(factory, "make_pre_post_processors", make_pre_post_processors)
    state["policy"] = policy
    return state


def _env_obs(batch_size: int = 2) -> dict:
    return {
        "main_images": torch.full((batch_size, 8, 9, 3), 0.5),
        "extra_view_images": torch.full(
            (batch_size, 2, 8, 9, 3), 128, dtype=torch.uint8
        ),
        "states": torch.arange(batch_size * 6, dtype=torch.float32).reshape(
            batch_size, 6
        ),
        "task_descriptions": ["pick the block"] * batch_size,
    }


def test_extension_loads_in_fresh_process() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["RLINF_EXT_MODULE"] = "smolvla"
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "from rlinf.config import SupportedModel; "
        "from rlinf.scheduler.cluster import load_user_extension_module; "
        "load_user_extension_module(); "
        "print(SupportedModel.get('smolvla').value)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "smolvla"


def test_register_is_idempotent() -> None:
    register()
    register()
    assert SupportedModel.get("smolvla").value == "smolvla"


def test_builder_uses_upstream_and_unit_stats(vendor_stubs: dict) -> None:
    cfg = _load_config()
    adapter = build_smolvla(cfg, torch.float32)

    assert adapter.policy is vendor_stubs["policy"]
    assert "type" not in vendor_stubs["parsed_config_data"]
    assert adapter.policy.anchor.dtype == torch.float32
    assert adapter.compute_dtype == torch.float32
    assert len(adapter.preprocessor.steps) == 1
    assert vendor_stubs["config"].device == "cpu"
    assert vendor_stubs["config"].dtype == torch.float32
    assert vendor_stubs["config"].load_vlm_weights is False
    assert vendor_stubs["config"].compile_model is False
    assert vendor_stubs["config"].rtc_config is None

    factory_args = vendor_stubs["factory_args"]
    assert factory_args["kwargs"]["pretrained_path"] is None
    dataset_stats = factory_args["kwargs"]["dataset_stats"]
    assert torch.equal(dataset_stats["observation.state"]["mean"], torch.zeros(6))
    assert torch.equal(dataset_stats["observation.state"]["std"], torch.ones(6))
    assert torch.equal(dataset_stats["action"]["mean"], torch.zeros(6))
    assert torch.equal(dataset_stats["action"]["std"], torch.ones(6))
    assert factory_args["kwargs"]["preprocessor_overrides"] == {
        "device_processor": {"device": "cpu", "float_dtype": None}
    }


def test_get_model_resolves_precision(vendor_stubs: dict) -> None:
    register()
    cfg = _load_config()
    cfg.precision = "fp16"
    model = get_model(cfg)
    assert isinstance(model, SmolVLAForRLActionPrediction)
    assert model.policy.anchor.dtype == torch.float32
    assert model.compute_dtype == torch.float16


def test_predict_action_maps_and_postprocesses(vendor_stubs: dict) -> None:
    adapter = build_smolvla(_load_config(), torch.float32)
    actions, result = adapter.predict_action_batch(_env_obs(), mode="eval")

    preprocessor_input = vendor_stubs["preprocessor_input"]
    assert preprocessor_input["observation.images.camera1"].shape == (2, 3, 8, 9)
    assert torch.all(preprocessor_input["observation.images.camera1"] == 0.5)
    assert torch.allclose(
        preprocessor_input["observation.images.camera2"],
        torch.full((2, 3, 8, 9), 128 / 255.0),
    )
    assert preprocessor_input["task"] == ["pick the block", "pick the block"]

    policy_batch = vendor_stubs["policy"].batches[-1]
    assert torch.equal(
        policy_batch["observation.state"],
        torch.arange(12, dtype=torch.float32).reshape(2, 6) * 2.0,
    )
    expected = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6) / 2.0
    expected = expected.expand(2, 50, 6)
    assert actions.shape == (2, 50, 6)
    assert actions.dtype == torch.float32
    assert actions.device.type == "cpu"
    assert actions.is_contiguous()
    assert torch.equal(actions, expected)
    assert result["prev_logprobs"] is None
    assert result["prev_values"] is None
    assert result["forward_inputs"]["action"].shape == (2, 300)


def test_explicit_stats_path_is_forwarded(vendor_stubs: dict, tmp_path: Path) -> None:
    stats = {
        "observation.state": {
            "mean": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "std": [2.0] * 6,
        },
        "action": {"mean": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0], "std": [3.0] * 6},
    }
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(stats), encoding="utf-8")
    cfg = _load_config()
    cfg.stats_path = str(path)
    build_smolvla(cfg, torch.float32)
    dataset_stats = vendor_stubs["factory_args"]["kwargs"]["dataset_stats"]
    assert torch.equal(
        dataset_stats["observation.state"]["mean"],
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    assert torch.equal(dataset_stats["action"]["std"], torch.tensor([3.0] * 6))


def test_training_paths_are_rejected(vendor_stubs: dict) -> None:
    adapter = build_smolvla(_load_config(), torch.float32)
    with pytest.raises(NotImplementedError, match="logprobs or values"):
        adapter.predict_action_batch(_env_obs(), calculate_logprobs=True)
    with pytest.raises(NotImplementedError, match="deterministic evaluation"):
        adapter.default_forward(forward_inputs={})


def test_lerobot_normalization_round_trip() -> None:
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor.normalize_processor import (
        NormalizerProcessorStep,
        UnnormalizerProcessorStep,
    )

    features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
        "action": PolicyFeature(FeatureType.ACTION, (2,)),
    }
    norm_map = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }
    stats = {
        "observation.state": {
            "mean": torch.tensor([1.0, 2.0]),
            "std": torch.tensor([2.0, 3.0]),
        },
        "action": {
            "mean": torch.tensor([3.0, 4.0]),
            "std": torch.tensor([5.0, 6.0]),
        },
    }
    normalizer = NormalizerProcessorStep(
        features=features, norm_map=norm_map, stats=stats
    )
    unnormalizer = UnnormalizerProcessorStep(
        features=features, norm_map=norm_map, stats=stats
    )
    state = torch.tensor([[3.0, 5.0], [7.0, 9.0]])
    action = torch.tensor([[8.0, 10.0], [13.0, 16.0]])
    transition = {"observation": {"observation.state": state}, "action": action}

    normalized = normalizer(transition)
    expected_state = (state - stats["observation.state"]["mean"]) / (
        stats["observation.state"]["std"] + 1e-8
    )
    expected_action = (action - stats["action"]["mean"]) / (
        stats["action"]["std"] + 1e-8
    )
    torch.testing.assert_close(
        normalized["observation"]["observation.state"], expected_state
    )
    torch.testing.assert_close(normalized["action"], expected_action)

    restored = unnormalizer(normalized)
    torch.testing.assert_close(
        restored["observation"]["observation.state"], state, rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(restored["action"], action, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("precision", "dtype"),
    [
        ("fp32", torch.float32),
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
    ],
)
def test_gpu_checkpoint_smoke(precision: str, dtype: torch.dtype) -> None:
    if os.environ.get("SMOLVLA_GPU_SMOKE") != "1":
        pytest.skip("set SMOLVLA_GPU_SMOKE=1 to run the real checkpoint")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real checkpoint smoke test")

    cfg = _load_config()
    cfg.precision = precision
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    adapter = build_smolvla(cfg, dtype).to("cuda").eval()
    torch.manual_seed(0)
    env_obs = {
        "main_images": torch.rand(1, 32, 32, 3),
        "extra_view_images": torch.rand(1, 2, 32, 32, 3),
        "states": torch.zeros(1, 6),
        "task_descriptions": ["pick up the block"],
    }
    noise = torch.randn(1, cfg.num_action_chunks, adapter.policy.config.max_action_dim)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    actions, _ = adapter.predict_action_batch(env_obs, mode="eval", noise=noise)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert actions.shape == (1, 50, 6)
    assert actions.dtype == torch.float32
    assert actions.device.type == "cpu"
    assert torch.isfinite(actions).all()
    logger.info(
        json.dumps(
            {
                "checkpoint": cfg.model_path,
                "revision": cfg.model_revision,
                "precision": precision,
                "action_shape": list(actions.shape),
                "finite": True,
                "latency_ms": elapsed_ms,
                "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                "gpu": torch.cuda.get_device_name(0),
            }
        )
    )
