import os

import numpy as np
import pytest

from versa.utterance_metrics.speaker import (
    is_transformers_available,
    resolve_speaker_backend,
)

RUN_REAL_MODEL_TESTS = os.environ.get("VERSA_RUN_REAL_MODEL_TESTS") == "1"


def test_resolve_backend_default_tag_is_espnet():
    assert resolve_speaker_backend(model_tag="default") == "espnet"


def test_resolve_backend_espnet_prefix_is_espnet():
    assert resolve_speaker_backend(model_tag="espnet/voxcelebs12_rawnet3") == "espnet"


def test_resolve_backend_hf_tag_is_huggingface():
    assert resolve_speaker_backend(model_tag="microsoft/wavlm-base-sv") == "huggingface"


def test_resolve_backend_local_espnet_files_win():
    assert (
        resolve_speaker_backend(
            model_tag="microsoft/wavlm-base-sv",
            model_path="/path/model.pth",
            model_config="/path/config.yaml",
        )
        == "espnet"
    )


def test_resolve_backend_explicit_override():
    assert (
        resolve_speaker_backend(model_tag="default", backend="huggingface")
        == "huggingface"
    )
    assert resolve_speaker_backend(model_tag="some/tag", backend="espnet") == "espnet"


def test_resolve_backend_invalid_backend_raises():
    with pytest.raises(ValueError, match="backend"):
        resolve_speaker_backend(model_tag="default", backend="wavlm2000")


def _fixed_audio(freq, duration=1.0, sample_rate=16000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    return (envelope * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.mark.real_model
@pytest.mark.skipif(
    not RUN_REAL_MODEL_TESTS,
    reason="Set VERSA_RUN_REAL_MODEL_TESTS=1 to run real model-backed checks",
)
@pytest.mark.skipif(
    not is_transformers_available(), reason="Transformers not available"
)
def test_hf_speaker_model_embedding_shape():
    from versa.utterance_metrics.speaker import hf_speaker_model_setup

    model = hf_speaker_model_setup(model_tag="microsoft/wavlm-base-sv", use_gpu=False)
    embedding = model(_fixed_audio(150))
    assert embedding.dim() == 2
    assert embedding.shape[0] == 1
    assert embedding.shape[1] > 0


@pytest.mark.real_model
@pytest.mark.skipif(
    not RUN_REAL_MODEL_TESTS,
    reason="Set VERSA_RUN_REAL_MODEL_TESTS=1 to run real model-backed checks",
)
@pytest.mark.skipif(
    not is_transformers_available(), reason="Transformers not available"
)
def test_hf_speaker_metric_identical_signals():
    from versa.utterance_metrics.speaker import hf_speaker_model_setup, speaker_metric

    model = hf_speaker_model_setup(model_tag="microsoft/wavlm-base-sv", use_gpu=False)
    audio = _fixed_audio(150)
    result = speaker_metric(model, audio, audio, 16000)
    assert "spk_similarity" in result
    assert result["spk_similarity"] == pytest.approx(1.0, abs=1e-4)


@pytest.mark.real_model
@pytest.mark.skipif(
    not RUN_REAL_MODEL_TESTS,
    reason="Set VERSA_RUN_REAL_MODEL_TESTS=1 to run real model-backed checks",
)
@pytest.mark.skipif(
    not is_transformers_available(), reason="Transformers not available"
)
def test_hf_speaker_metric_different_signals():
    from versa.utterance_metrics.speaker import hf_speaker_model_setup, speaker_metric

    model = hf_speaker_model_setup(model_tag="microsoft/wavlm-base-sv", use_gpu=False)
    same = speaker_metric(model, _fixed_audio(150), _fixed_audio(150), 16000)
    diff = speaker_metric(model, _fixed_audio(150), _fixed_audio(420), 16000)
    assert diff["spk_similarity"] < same["spk_similarity"]
    assert -1.0 <= diff["spk_similarity"] <= 1.0


@pytest.mark.real_model
@pytest.mark.skipif(
    not RUN_REAL_MODEL_TESTS,
    reason="Set VERSA_RUN_REAL_MODEL_TESTS=1 to run real model-backed checks",
)
@pytest.mark.skipif(
    not is_transformers_available(), reason="Transformers not available"
)
def test_speaker_metric_class_wavlm_backend():
    from versa.utterance_metrics.speaker import SpeakerMetric

    metric = SpeakerMetric({"model_tag": "microsoft/wavlm-base-sv", "use_gpu": False})
    assert metric.backend == "huggingface"

    audio = _fixed_audio(150)
    result = metric.compute(audio, audio, metadata={"sample_rate": 16000})
    assert result["spk_similarity"] == pytest.approx(1.0, abs=1e-4)


@pytest.mark.real_model
@pytest.mark.skipif(
    not RUN_REAL_MODEL_TESTS,
    reason="Set VERSA_RUN_REAL_MODEL_TESTS=1 to run real model-backed checks",
)
@pytest.mark.skipif(
    not is_transformers_available(), reason="Transformers not available"
)
def test_speaker_metric_class_requires_both_signals():
    from versa.utterance_metrics.speaker import SpeakerMetric

    metric = SpeakerMetric({"model_tag": "microsoft/wavlm-base-sv", "use_gpu": False})
    with pytest.raises(ValueError, match="Predicted signal"):
        metric.compute(None, _fixed_audio(150), metadata={"sample_rate": 16000})
    with pytest.raises(ValueError, match="Reference signal"):
        metric.compute(_fixed_audio(150), None, metadata={"sample_rate": 16000})


def test_speaker_metadata_mentions_both_backends():
    from versa.utterance_metrics.speaker import _speaker_metadata

    metadata = _speaker_metadata()
    assert "transformers" in metadata.dependencies
    assert "espnet2" in metadata.dependencies


def test_cache_namespace_speaker_espnet_default():
    from versa.scorer_shared import configure_metric_cache_dirs

    configs = configure_metric_cache_dirs(
        [{"name": "speaker", "model_tag": "default"}], cache_folder="/tmp/vc"
    )
    assert configs[0]["cache_dir"].endswith("espnet_model_zoo")


def test_cache_namespace_speaker_hf_tag():
    from versa.scorer_shared import configure_metric_cache_dirs

    configs = configure_metric_cache_dirs(
        [{"name": "speaker", "model_tag": "microsoft/wavlm-base-sv"}],
        cache_folder="/tmp/vc",
    )
    assert configs[0]["cache_dir"].endswith("huggingface")


def test_cache_namespace_speaker_alias_hf_tag():
    from versa.scorer_shared import configure_metric_cache_dirs

    configs = configure_metric_cache_dirs(
        [{"name": "spk_similarity", "model_tag": "microsoft/wavlm-base-sv"}],
        cache_folder="/tmp/vc",
    )
    assert configs[0]["cache_dir"].endswith("huggingface")


def test_cache_namespace_explicit_cache_dir_wins():
    """Explicit cache_dir wins at the config-plumbing level.

    Note: at model-load time, get_hf_cache_dir gives the VERSA_HF_CACHE_DIR
    environment variable precedence over this per-metric cache_dir.
    """
    from versa.scorer_shared import configure_metric_cache_dirs

    configs = configure_metric_cache_dirs(
        [
            {
                "name": "speaker",
                "model_tag": "microsoft/wavlm-base-sv",
                "cache_dir": "/custom/cache",
            }
        ],
        cache_folder="/tmp/vc",
    )
    assert configs[0]["cache_dir"] == "/custom/cache"


def test_cache_namespace_backend_override():
    from versa.scorer_shared import configure_metric_cache_dirs

    configs = configure_metric_cache_dirs(
        [{"name": "speaker", "model_tag": "default", "backend": "huggingface"}],
        cache_folder="/tmp/vc",
    )
    assert configs[0]["cache_dir"].endswith("huggingface")


def test_resolve_backend_none_tag_is_espnet():
    assert resolve_speaker_backend(model_tag=None) == "espnet"


def test_resolve_backend_non_string_tag_does_not_crash():
    assert resolve_speaker_backend(model_tag=123) == "huggingface"


def test_effective_dependencies_wavlm_excludes_espnet():
    from versa.config_validation import _effective_dependencies
    from versa.utterance_metrics.speaker import _speaker_metadata

    deps = _effective_dependencies(
        "speaker", _speaker_metadata(), {"model_tag": "microsoft/wavlm-base-sv"}
    )
    assert "espnet2" not in deps
    assert "transformers" in deps


def test_effective_dependencies_default_excludes_transformers():
    from versa.config_validation import _effective_dependencies
    from versa.utterance_metrics.speaker import _speaker_metadata

    deps = _effective_dependencies(
        "speaker", _speaker_metadata(), {"model_tag": "default"}
    )
    assert "transformers" not in deps
    assert "espnet2" in deps
