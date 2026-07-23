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
