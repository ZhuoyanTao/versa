#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import numpy as np
import torch

from versa.audio_utils import resample_audio
from versa.huggingface_cache import configure_huggingface_cache, get_hf_cache_dir

try:
    from espnet2.bin.spk_inference import Speech2Embedding
except ImportError:
    Speech2Embedding = None

try:
    from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoFeatureExtractor = None
    AutoModelForAudioXVector = None
    TRANSFORMERS_AVAILABLE = False

from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType

logger = logging.getLogger(__name__)


def is_transformers_available():
    """Check whether transformers is importable for the HuggingFace backend."""
    return TRANSFORMERS_AVAILABLE


ESPNET_DEFAULT_SPEAKER_TAG = "espnet/voxcelebs12_rawnet3"
SPEAKER_BACKENDS = ("espnet", "huggingface")


def resolve_speaker_backend(
    model_tag="default", backend=None, model_path=None, model_config=None
):
    """Resolve which speaker-model backend a configuration refers to.

    Args:
        model_tag: Model tag from the config ("default", an ESPnet hub tag
            such as "espnet/voxcelebs12_rawnet3", or any other HuggingFace
            repo id such as "microsoft/wavlm-base-sv").
        backend: Optional explicit backend override ("espnet" or "huggingface").
        model_path: Optional local ESPnet model checkpoint path.
        model_config: Optional local ESPnet train config path.

    Returns:
        "espnet" or "huggingface".
    """
    if backend is not None:
        if backend not in SPEAKER_BACKENDS:
            raise ValueError(
                "Unknown speaker backend '{}'. Supported backends: {}".format(
                    backend, SPEAKER_BACKENDS
                )
            )
        return backend
    if model_path is not None and model_config is not None:
        return "espnet"
    if model_tag == "default" or model_tag.startswith("espnet/"):
        return "espnet"
    return "huggingface"


def speaker_model_setup(
    model_tag="default",
    model_path=None,
    model_config=None,
    use_gpu=False,
    cache_dir=None,
):
    if Speech2Embedding is None:
        raise ImportError("speaker requires espnet. Please install espnet and retry")

    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    if model_path is not None and model_config is not None:
        model = Speech2Embedding(
            model_file=model_path, train_config=model_config, device=device
        )
    else:
        if model_tag == "default":
            model_tag = ESPNET_DEFAULT_SPEAKER_TAG
        if cache_dir is None:
            model = Speech2Embedding.from_pretrained(model_tag=model_tag, device=device)
        else:
            try:
                from espnet_model_zoo.downloader import ModelDownloader
            except ImportError:
                raise ImportError(
                    "speaker requires espnet_model_zoo. Please install it and retry"
                )
            model_kwargs = ModelDownloader(cachedir=cache_dir).download_and_unpack(
                model_tag
            )
            model = Speech2Embedding(device=device, **model_kwargs)
    return model


class HFSpeakerModel:
    """Callable wrapper around a HuggingFace x-vector speaker model.

    Mirrors the call signature of espnet's Speech2Embedding so that
    speaker_metric can consume either backend interchangeably.
    """

    def __init__(self, model, feature_extractor, device):
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device

    def __call__(self, speech):
        inputs = self.feature_extractor(
            speech, sampling_rate=16000, return_tensors="pt"
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.embeddings


def hf_speaker_model_setup(
    model_tag="microsoft/wavlm-base-sv", use_gpu=False, cache_dir=None
):
    """Load a HuggingFace x-vector speaker model (e.g. WavLM-base-sv).

    Works with any AutoModelForAudioXVector checkpoint, including
    microsoft/wavlm-base-sv, microsoft/unispeech-sat-base-sv, and other
    x-vector fine-tuned speech encoders on the HuggingFace hub.
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "HuggingFace speaker models require transformers. "
            "Please install it with `pip install transformers` "
            "(or `pip install versa[ml]`) and retry."
        )
    if use_gpu and not torch.cuda.is_available():
        logger.warning("use_gpu requested but CUDA is unavailable; using CPU.")
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    resolved_cache_dir = get_hf_cache_dir(cache_dir)
    configure_huggingface_cache(resolved_cache_dir)
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_tag, cache_dir=resolved_cache_dir
    )
    model = (
        AutoModelForAudioXVector.from_pretrained(
            model_tag, cache_dir=resolved_cache_dir
        )
        .to(device)
        .eval()
    )
    return HFSpeakerModel(model, feature_extractor, device)


def speaker_metric(model, pred_x, gt_x, fs):
    # NOTE(jiatong): only work for 16000 Hz
    if fs < 16000:
        logger.warning(
            "Speaker similarity with sampling rates below 16 kHz may be unreliable "
            "for speaker embedding models such as the default RawNet3 model."
        )

    if fs != 16000:
        gt_x = resample_audio(gt_x, fs, 16000)
        pred_x = resample_audio(pred_x, fs, 16000)

    embedding_gen = model(pred_x).squeeze(0).cpu().numpy()
    embedding_gt = model(gt_x).squeeze(0).cpu().numpy()
    similarity = np.dot(embedding_gen, embedding_gt) / (
        np.linalg.norm(embedding_gen) * np.linalg.norm(embedding_gt)
    )
    return {"spk_similarity": similarity}


class SpeakerMetric(BaseMetric):
    """Speaker embedding cosine similarity."""

    def _setup(self):
        self.model_tag = self.config.get("model_tag", "default")
        self.model_path = self.config.get("model_path")
        self.model_config = self.config.get("model_config")
        self.use_gpu = self.config.get("use_gpu", False)
        self.cache_dir = self.config.get("cache_dir", "versa_cache/espnet_model_zoo")
        self.model = speaker_model_setup(
            model_tag=self.model_tag,
            model_path=self.model_path,
            model_config=self.model_config,
            use_gpu=self.use_gpu,
            cache_dir=self.cache_dir,
        )

    def compute(self, predictions, references=None, metadata=None):
        if predictions is None:
            raise ValueError("Predicted signal must be provided")
        if references is None:
            raise ValueError("Reference signal must be provided")

        fs = metadata.get("sample_rate", 16000) if metadata else 16000
        return speaker_metric(
            self.model, np.asarray(predictions), np.asarray(references), fs
        )

    def get_metadata(self):
        return _speaker_metadata()


def _speaker_metadata():
    return MetricMetadata(
        name="speaker",
        category=MetricCategory.NON_MATCH,
        metric_type=MetricType.FLOAT,
        requires_reference=True,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["espnet2", "librosa", "numpy"],
        description="Speaker embedding cosine similarity",
        paper_reference="https://arxiv.org/abs/2401.17230",
        implementation_source="https://github.com/espnet/espnet",
    )


def register_speaker_metric(registry):
    """Register speaker similarity with the registry."""
    registry.register(
        SpeakerMetric,
        _speaker_metadata(),
        aliases=["spk_similarity", "speaker_similarity"],
    )


if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    metric = SpeakerMetric()
    print("metrics: {}".format(metric.compute(a, b, metadata={"sample_rate": 16000})))
