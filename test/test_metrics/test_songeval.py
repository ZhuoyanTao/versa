import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from versa.definition import MetricCategory, MetricRegistry, MetricType
from versa.utterance_metrics import songeval


class DummySongEvalModel:
    def __call__(self, hidden):
        return torch.tensor([[3.12345, 4.0, 2.5, 1.25, 5.0]], dtype=torch.float32)


class DummyMuQ:
    def __init__(self):
        self.audio_shape = None

    def __call__(self, audio, output_hidden_states=True):
        assert output_hidden_states
        self.audio_shape = tuple(audio.shape)
        hidden_states = [torch.zeros((1, 2, 3), device=audio.device) for _ in range(7)]
        return {"hidden_states": hidden_states}


def test_songeval_metric_returns_stable_prefixed_keys():
    model_dict = {
        "device": "cpu",
        "model": DummySongEvalModel(),
        "muq": DummyMuQ(),
    }

    scores = songeval.songeval_metric(
        model_dict,
        np.ones(16000, dtype=np.float32),
        fs=16000,
    )

    assert scores == {
        "songeval_coherence": 3.1235,
        "songeval_musicality": 4.0,
        "songeval_memorability": 2.5,
        "songeval_clarity": 1.25,
        "songeval_naturalness": 5.0,
    }


def test_songeval_metric_validates_predictions(monkeypatch):
    monkeypatch.setattr(
        songeval,
        "songeval_model_setup",
        lambda **kwargs: {
            "device": "cpu",
            "model": DummySongEvalModel(),
            "muq": DummyMuQ(),
        },
    )

    metric = songeval.SongEvalMetric()

    with pytest.raises(ValueError, match="Predicted signal"):
        metric.compute(None, metadata={"sample_rate": 16000})


@pytest.mark.parametrize(
    "audio",
    [
        np.ones((16000, 2), dtype=np.float32),
        np.ones((2, 16000), dtype=np.float32),
    ],
)
def test_songeval_metric_downmixes_stereo(audio):
    muq = DummyMuQ()
    model_dict = {
        "device": "cpu",
        "model": DummySongEvalModel(),
        "muq": muq,
    }

    songeval.songeval_metric(model_dict, audio, fs=16000)

    assert muq.audio_shape == (1, 24000)


@pytest.mark.parametrize(
    ("audio", "sample_rate", "message"),
    [
        (np.array([], dtype=np.float32), 24000, "non-empty"),
        (np.array([0.0, np.nan], dtype=np.float32), 24000, "NaN"),
        (np.ones((2, 3, 4), dtype=np.float32), 24000, "mono audio"),
        (np.ones(100, dtype=np.float32), 0, "positive integer"),
        (np.ones(100, dtype=np.float32), 24000.0, "positive integer"),
    ],
)
def test_songeval_metric_rejects_invalid_audio(audio, sample_rate, message):
    model_dict = {
        "device": "cpu",
        "model": DummySongEvalModel(),
        "muq": DummyMuQ(),
    }

    with pytest.raises(ValueError, match=message):
        songeval.songeval_metric(model_dict, audio, fs=sample_rate)


def test_songeval_metric_validates_predictor_output_shape():
    class InvalidModel:
        def __call__(self, hidden):
            return torch.ones((1, 4), dtype=torch.float32)

    model_dict = {
        "device": "cpu",
        "model": InvalidModel(),
        "muq": DummyMuQ(),
    }

    with pytest.raises(RuntimeError, match="expected 5 scores"):
        songeval.songeval_metric(
            model_dict,
            np.ones(24000, dtype=np.float32),
            fs=24000,
        )


def test_songeval_metric_forwards_local_and_offline_configuration(monkeypatch):
    calls = {}

    def fake_setup(**kwargs):
        calls.update(kwargs)
        return {
            "device": "cpu",
            "model": DummySongEvalModel(),
            "muq": DummyMuQ(),
        }

    monkeypatch.setattr(songeval, "songeval_model_setup", fake_setup)

    songeval.SongEvalMetric(
        {
            "cache_dir": "cache-root",
            "model_dir": "local-songeval",
            "muq_model": "local-muq",
            "offline": True,
            "use_gpu": True,
        }
    )

    assert calls == {
        "cache_dir": "cache-root",
        "model_dir": "local-songeval",
        "muq_model": "local-muq",
        "offline": True,
        "use_gpu": True,
    }


def test_songeval_offline_mode_requires_local_assets(tmp_path):
    with pytest.raises(FileNotFoundError, match="assets are not installed"):
        songeval._resolve_songeval_dir(tmp_path, offline=True)


def test_songeval_downloads_and_pins_missing_assets(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, check):
        assert check
        calls.append(command)
        if command[1] == "clone":
            model_dir = tmp_path / "SongEval"
            (model_dir / "ckpt").mkdir(parents=True)
            (model_dir / "model.py").touch()
            (model_dir / "config.yaml").touch()
            (model_dir / "ckpt" / "model.safetensors").touch()

    monkeypatch.setattr(songeval.subprocess, "run", fake_run)

    assert songeval._resolve_songeval_dir(tmp_path) == tmp_path / "SongEval"
    assert calls == [
        [
            "git",
            "clone",
            songeval.SONGEVAL_REPOSITORY,
            str(tmp_path / "SongEval"),
        ],
        [
            "git",
            "-C",
            str(tmp_path / "SongEval"),
            "checkout",
            "--detach",
            songeval.SONGEVAL_REVISION,
        ],
    ]


def test_songeval_resolves_complete_local_assets(tmp_path):
    model_dir = tmp_path / "custom-songeval"
    (model_dir / "ckpt").mkdir(parents=True)
    (model_dir / "model.py").touch()
    (model_dir / "config.yaml").touch()
    (model_dir / "ckpt" / "model.safetensors").touch()

    assert songeval._resolve_songeval_dir(tmp_path, model_dir) == model_dir


def test_songeval_setup_loads_local_assets_without_generic_model_import(
    monkeypatch, tmp_path
):
    existing_generic_model = sys.modules.get("model")
    model_dir = tmp_path / "SongEval"
    (model_dir / "ckpt").mkdir(parents=True)
    (model_dir / "model.py").write_text(
        "import torch\n"
        "class Generator(torch.nn.Module):\n"
        "    def __init__(self, num_classes):\n"
        "        super().__init__()\n"
        "        self.bias = torch.nn.Parameter(torch.zeros(num_classes))\n"
        "    def forward(self, hidden):\n"
        "        return self.bias.unsqueeze(0)\n",
        encoding="utf-8",
    )
    (model_dir / "config.yaml").write_text("placeholder", encoding="utf-8")
    (model_dir / "ckpt" / "model.safetensors").touch()
    muq_dir = tmp_path / "muq"
    muq_dir.mkdir()

    class FakeOmegaConf:
        @staticmethod
        def load(path):
            return SimpleNamespace(
                generator={"_target_": "model.Generator", "num_classes": 5}
            )

        @staticmethod
        def to_container(config, resolve=True):
            assert resolve
            return dict(config)

    class FakeMuQ:
        requested_path = None
        requested_kwargs = None

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            cls.requested_path = path
            cls.requested_kwargs = kwargs
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(songeval, "OmegaConf", FakeOmegaConf)
    monkeypatch.setattr(songeval, "MuQ", FakeMuQ)
    monkeypatch.setattr(
        songeval,
        "load_file",
        lambda path, device: {"bias": torch.zeros(5)},
    )

    model_dict = songeval.songeval_model_setup(
        cache_dir=tmp_path,
        model_dir=model_dir,
        muq_model=muq_dir,
        offline=True,
    )

    assert model_dict["device"] == "cpu"
    assert type(model_dict["model"]).__module__.startswith("_versa_songeval_model_")
    assert FakeMuQ.requested_path == str(muq_dir)
    assert FakeMuQ.requested_kwargs == {
        "cache_dir": tmp_path / "huggingface",
        "local_files_only": True,
    }
    assert sys.modules.get("model") is existing_generic_model


def test_songeval_registration_metadata():
    registry = MetricRegistry()

    songeval.register_songeval_metric(registry)
    metadata = registry.get_metadata("song_eval")

    assert metadata.name == "songeval"
    assert metadata.category == MetricCategory.INDEPENDENT
    assert metadata.metric_type == MetricType.DICT
    assert not metadata.requires_reference
    assert metadata.gpu_compatible
    assert not metadata.auto_install
    assert "einops" in metadata.dependencies
    assert "hydra" not in metadata.dependencies
    assert registry.get_metric("SongEval") is songeval.SongEvalMetric


def test_songeval_setup_reports_missing_optional_dependencies(monkeypatch):
    monkeypatch.setattr(songeval, "MuQ", None)

    with pytest.raises(ImportError, match="SongEval requires optional dependencies"):
        songeval.songeval_model_setup()


@pytest.mark.real_model
def test_songeval_real_model_inference():
    if os.environ.get("VERSA_RUN_REAL_MODEL_TESTS") != "1":
        pytest.skip("Set VERSA_RUN_REAL_MODEL_TESTS=1 to run real SongEval inference")

    sample_rate = 24000
    time = np.linspace(0, 2.0, sample_rate * 2, endpoint=False, dtype=np.float32)
    audio = (0.1 * np.sin(2 * np.pi * 440.0 * time)).astype(np.float32)

    model = songeval.songeval_model_setup(use_gpu=False)
    scores = songeval.songeval_metric(model, audio, sample_rate)

    assert set(scores) == {key for key, _ in songeval.SONGEVAL_OUTPUTS}
    assert all(np.isfinite(value) for value in scores.values())
