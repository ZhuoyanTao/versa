"""Entrypoint contracts exercised through the real scorer with tiny fake metrics."""

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from test.audio_utils import generate_fixed_wav
from versa import scorer_shared
from versa.bin import scorer, scorer_chunk, scoring
from versa.definition import (
    BaseMetric,
    MetricCategory,
    MetricMetadata,
    MetricRegistry,
    MetricType,
)


@pytest.fixture
def scoring_case(tmp_path, monkeypatch):
    for key in (
        "VERSA_CACHE_DIR",
        "VERSA_HF_CACHE_DIR",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "TORCH_HOME",
        "NEMO_CACHE_DIR",
        "XDG_CACHE_HOME",
    ):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    calls = {"utterance": [], "corpus": [], "closed": [], "released": []}

    class UtteranceMetric(BaseMetric):
        def _setup(self):
            self.io = self.config.get("io")
            self.cache_dir = self.config.get("cache_dir")

        def get_metadata(self):
            return MetricMetadata(
                "test_utterance",
                MetricCategory.INDEPENDENT,
                MetricType.FLOAT,
                False,
                False,
                False,
                False,
                [],
                "test",
            )

        def compute(self, predictions, references=None, metadata=None):
            calls["utterance"].append((references is not None, metadata["text"]))
            return {"test_score": 0.5}

    class CorpusMetric(UtteranceMetric):
        def get_metadata(self):
            return MetricMetadata(
                "test_corpus",
                MetricCategory.DISTRIBUTIONAL,
                MetricType.FLOAT,
                False,
                False,
                False,
                False,
                [],
                "test",
            )

        def compute(self, predictions, references=None, metadata=None):
            calls["corpus"].append((predictions, references, self.config, metadata))
            return {"corpus_score": 0.25}

    registry = MetricRegistry()
    for cls in (UtteranceMetric, CorpusMetric):
        registry.register(cls, cls().get_metadata())
    real_scorer = scorer_shared.VersaScorer
    monkeypatch.setattr(scorer_shared, "VersaScorer", lambda: real_scorer(registry))
    monkeypatch.setattr(scorer_chunk, "VersaScorer", lambda: real_scorer(registry))
    original_close = scorer_shared.ScoreProcessor.close

    def close(processor):
        original_close(processor)
        calls["closed"].append(
            processor.file_handle is None or processor.file_handle.closed
        )

    monkeypatch.setattr(scorer_shared.ScoreProcessor, "close", close)
    monkeypatch.setattr(
        scorer_shared,
        "_release_metric_resources",
        lambda: calls["released"].append(True),
    )
    for folder in ("pred", "gt"):
        (tmp_path / folder).mkdir()
        generate_fixed_wav(tmp_path / folder / "utt.wav")
    (tmp_path / "text").write_text("utt.wav hello world\n")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump([{"name": "test_utterance"}, {"name": "test_corpus"}])
    )
    argv = [
        "versa-score",
        "--pred",
        str(tmp_path / "pred"),
        "--gt",
        str(tmp_path / "gt"),
        "--text",
        str(tmp_path / "text"),
        "--io",
        "dir",
        "--score_config",
        str(config),
        "--output_file",
        str(tmp_path / "scores.jsonl"),
        "--cache_folder",
        str(tmp_path / "cache"),
    ]
    return tmp_path, argv, calls


@pytest.mark.parametrize("mode", ["ordinary", "metric", "chunk_cli", "chunks"])
@pytest.mark.parametrize("no_match", [False, True])
def test_entrypoint_outputs_inputs_and_resume(
    scoring_case, monkeypatch, mode, no_match
):
    root, argv, calls = scoring_case
    entrypoint = scorer_chunk if mode in ("chunk_cli", "chunks") else scorer
    if mode == "metric":
        argv += ["--scoring_mode", "metric"]
    if mode == "chunks":
        argv += [
            "--enable_chunking",
            "--chunk_duration",
            "0.5",
            "--hop_duration",
            "0.5",
        ]
    if no_match:
        argv += ["--no_match"]
    monkeypatch.setattr(sys, "argv", argv)
    entrypoint.main()
    output = root / "scores.jsonl"
    before = output.read_text()
    rows = [json.loads(line) for line in before.splitlines()]
    keys = (
        ["utt.wav@0.000-0.500", "utt.wav@0.500-1.000"]
        if mode == "chunks"
        else ["utt.wav"]
    )
    assert rows == [{"key": key, "test_score": 0.5} for key in keys]
    assert calls["utterance"] == [(not no_match, "hello world")] * len(keys)
    assert yaml.safe_load(Path(str(output) + ".corpus").read_text()) == {
        "corpus_score": 0.25
    }
    pred, gt, config, metadata = calls["corpus"][0]
    if mode == "chunks":
        assert pred == str(output) + ".chunks/pred"
        assert gt is None
    elif mode == "chunk_cli":
        assert pred == str(root / "pred")
        assert gt == (None if no_match else str(root / "gt"))
    else:
        assert pred == {"utt.wav": str(root / "pred/utt.wav")}
        assert gt == (None if no_match else {"utt.wav": str(root / "gt/utt.wav")})
    assert metadata["text_info"] == dict.fromkeys(keys, "hello world")
    if entrypoint is scorer_chunk:
        assert config["cache_dir"] == str(root / "cache/test_corpus")
        assert config["io"] == "dir"
    else:
        assert config["cache_dir"] == str(root / "cache/test_corpus")
        assert "io" not in config
    monkeypatch.setattr(sys, "argv", argv + ["--resume"])
    entrypoint.main()
    assert output.read_text() == before
    # Metric mode intentionally recomputes while merging persisted rows.
    assert len(calls["utterance"]) == len(keys) * (2 if mode == "metric" else 1)
    assert calls["closed"] and all(calls["closed"])
    assert bool(calls["released"]) == (mode == "metric")


@pytest.mark.parametrize("entrypoint", [scorer, scorer_chunk])
def test_explicit_corpus_config_wins(scoring_case, monkeypatch, entrypoint):
    root, argv, calls = scoring_case
    (root / "config.yaml").write_text(
        yaml.safe_dump(
            [{"name": "test_corpus", "io": "soundfile", "cache_dir": "explicit"}]
        )
    )
    monkeypatch.setattr(sys, "argv", argv)
    entrypoint.main()
    assert calls["utterance"] == []
    assert calls["corpus"][0][2]["cache_dir"] == "explicit"
    assert calls["corpus"][0][2]["io"] == "soundfile"


@pytest.mark.parametrize(
    "entrypoint,error", [(scorer, SystemExit), (scorer_chunk, SystemExit)]
)
def test_empty_config_contract(scoring_case, monkeypatch, entrypoint, error):
    root, argv, _ = scoring_case
    (root / "config.yaml").write_text("[]\n")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(error) as exc:
        entrypoint.main()
    assert exc.value.code == 2


def test_cuda_validation(scoring_case, monkeypatch):
    _, argv, _ = scoring_case
    monkeypatch.setattr(sys, "argv", argv + ["--use_gpu"])
    monkeypatch.setattr(scoring.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CUDA device"):
        scorer.main()
