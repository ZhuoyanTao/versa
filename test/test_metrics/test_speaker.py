import numpy as np
import pytest

from versa.utterance_metrics.speaker import resolve_speaker_backend


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


from versa.utterance_metrics.speaker import is_transformers_available


def _fixed_audio(freq, duration=1.0, sample_rate=16000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    return (envelope * np.sin(2 * np.pi * freq * t)).astype(np.float32)


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
