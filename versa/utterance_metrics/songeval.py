import importlib.util
import hashlib
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from versa.audio_utils import resample_audio
from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType
from versa.huggingface_cache import get_hf_cache_dir

try:
    from muq import MuQ
except ImportError:
    MuQ = None

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

try:
    from safetensors.torch import load_file
except ImportError:
    load_file = None


logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 24000
DEFAULT_MUQ_MODEL = "OpenMuQ/MuQ-large-msd-iter"
SONGEVAL_REPOSITORY = "https://github.com/ASLP-lab/SongEval.git"
SONGEVAL_REVISION = "848fb2ff3a2a9d64dcb20d46f07238c28abd7add"
SONGEVAL_OUTPUTS = (
    ("songeval_coherence", "Coherence"),
    ("songeval_musicality", "Musicality"),
    ("songeval_memorability", "Memorability"),
    ("songeval_clarity", "Clarity"),
    ("songeval_naturalness", "Naturalness"),
)


def _require_songeval_dependencies():
    missing = []
    if MuQ is None:
        missing.append("muq")
    if OmegaConf is None:
        missing.append("omegaconf")
    if load_file is None:
        missing.append("safetensors")

    if missing:
        raise ImportError(
            "SongEval requires optional dependencies: {}. "
            "Install them with `tools/install_songeval.sh`.".format(", ".join(missing))
        )


def _required_songeval_files(songeval_dir):
    return (
        songeval_dir / "model.py",
        songeval_dir / "config.yaml",
        songeval_dir / "ckpt" / "model.safetensors",
    )


def _resolve_songeval_dir(cache_dir, model_dir=None, offline=False):
    songeval_dir = Path(model_dir) if model_dir else Path(cache_dir) / "SongEval"
    missing_files = [
        str(path)
        for path in _required_songeval_files(songeval_dir)
        if not path.is_file()
    ]
    if (
        missing_files
        and model_dir is None
        and not offline
        and not songeval_dir.exists()
    ):
        songeval_dir.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading SongEval into %s", songeval_dir)
        subprocess.run(
            ["git", "clone", SONGEVAL_REPOSITORY, str(songeval_dir)], check=True
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(songeval_dir),
                "checkout",
                "--detach",
                SONGEVAL_REVISION,
            ],
            check=True,
        )
        missing_files = [
            str(path)
            for path in _required_songeval_files(songeval_dir)
            if not path.is_file()
        ]

    if missing_files:
        raise FileNotFoundError(
            "SongEval assets are not installed or are incomplete. Missing: {}. "
            "Run `tools/install_songeval.sh`, or set `model_dir` to a complete "
            "local SongEval checkout.".format(", ".join(missing_files))
        )
    return songeval_dir


def _load_generator_class(songeval_dir):
    """Load upstream's Generator without exposing its generic `model` module name."""
    module_path = (songeval_dir / "model.py").resolve()
    module_digest = hashlib.sha256(str(module_path).encode("utf-8")).hexdigest()[:12]
    module_name = f"_versa_songeval_model_{module_digest}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Unable to import SongEval model code from {module_path}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

    try:
        return module.Generator
    except AttributeError as error:
        raise ImportError(
            f"SongEval model module {module_path} does not define Generator"
        ) from error


def songeval_model_setup(
    cache_dir="versa_cache",
    use_gpu=False,
    model_dir=None,
    muq_model=DEFAULT_MUQ_MODEL,
    offline=False,
):
    """Set up SongEval classifier and MuQ encoder."""
    _require_songeval_dependencies()
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    if use_gpu and device == "cpu":
        logger.warning(
            "SongEval requested GPU execution, but CUDA is unavailable; using CPU"
        )

    songeval_dir = _resolve_songeval_dir(cache_dir, model_dir, offline=offline)
    logger.info("Using SongEval assets in %s", songeval_dir)

    model_path = songeval_dir / "ckpt" / "model.safetensors"
    config_path = songeval_dir / "config.yaml"

    with torch.no_grad():
        train_config = OmegaConf.load(config_path)
        generator_config = OmegaConf.to_container(train_config.generator, resolve=True)
        target = generator_config.pop("_target_", None)
        if target != "model.Generator":
            raise ValueError(
                "Unsupported SongEval generator target {!r}; expected "
                "'model.Generator' from pinned revision {}".format(
                    target, SONGEVAL_REVISION
                )
            )
        generator_class = _load_generator_class(songeval_dir)
        model = generator_class(**generator_config)
        state_dict = load_file(model_path, device="cpu")
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device).eval()

    if offline and not Path(muq_model).is_dir():
        raise FileNotFoundError(
            "SongEval offline mode requires `muq_model` to point to a local "
            f"MuQ model directory; got {muq_model!r}"
        )
    muq_cache_dir = Path(get_hf_cache_dir(Path(cache_dir) / "huggingface"))
    muq_model = MuQ.from_pretrained(
        str(muq_model),
        cache_dir=muq_cache_dir,
        local_files_only=offline,
    )
    muq_model = muq_model.to(device).eval()

    model_dict = {"model": model, "muq": muq_model, "device": device}
    return model_dict


def songeval_metric(model_dict, pred, fs):
    """
    pred: np.ndarray, original waveform
    fs: original sampling rate
    return: dict, metric results for five SongEval dimensions
    """
    device = model_dict["device"]
    model = model_dict["model"]
    muq_model = model_dict["muq"]

    if not isinstance(fs, (int, np.integer)) or fs <= 0:
        raise ValueError(f"Sample rate must be a positive integer; got {fs!r}")

    pred = np.asarray(pred)
    if pred.ndim == 2:
        if pred.shape[1] <= 8:
            pred = pred.mean(axis=1)
        elif pred.shape[0] <= 8:
            pred = pred.mean(axis=0)
        else:
            raise ValueError(
                "SongEval received ambiguous 2-D audio; expected samples x channels "
                "or channels x samples with at most 8 channels"
            )
    if pred.ndim != 1:
        raise ValueError(f"SongEval expects mono audio; got shape {pred.shape}")
    if pred.size == 0:
        raise ValueError("SongEval requires non-empty audio")
    if not np.isfinite(pred).all():
        raise ValueError("SongEval audio contains NaN or infinite values")

    pred = resample_audio(pred, int(fs), TARGET_SAMPLE_RATE)

    audio = torch.as_tensor(pred, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        output = muq_model(audio, output_hidden_states=True)
        try:
            hidden = output["hidden_states"][6]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                "MuQ output does not contain the expected hidden state at layer 6"
            ) from error
        scores_g = model(hidden).squeeze(0)

    if scores_g.ndim != 1 or scores_g.numel() != len(SONGEVAL_OUTPUTS):
        raise RuntimeError(
            "SongEval predictor returned shape {}; expected {} scores".format(
                tuple(scores_g.shape), len(SONGEVAL_OUTPUTS)
            )
        )

    values = {}
    for index, (output_key, _) in enumerate(SONGEVAL_OUTPUTS):
        values[output_key] = round(scores_g[index].item(), 4)

    return values


class SongEvalMetric(BaseMetric):
    """SongEval song aesthetics predictor."""

    def _setup(self):
        self.cache_dir = self.config.get("cache_dir", "versa_cache")
        self.use_gpu = self.config.get("use_gpu", False)
        self.model_dir = self.config.get("model_dir")
        self.muq_model = self.config.get("muq_model", DEFAULT_MUQ_MODEL)
        self.offline = self.config.get("offline", False)
        self.model_dict = songeval_model_setup(
            cache_dir=self.cache_dir,
            use_gpu=self.use_gpu,
            model_dir=self.model_dir,
            muq_model=self.muq_model,
            offline=self.offline,
        )

    def compute(self, predictions, references=None, metadata=None):
        if predictions is None:
            raise ValueError("Predicted signal must be provided")

        fs = metadata.get("sample_rate", TARGET_SAMPLE_RATE) if metadata else 24000
        return songeval_metric(self.model_dict, np.asarray(predictions), fs)

    def get_metadata(self):
        return _songeval_metadata()


def _songeval_metadata():
    return MetricMetadata(
        name="songeval",
        category=MetricCategory.INDEPENDENT,
        metric_type=MetricType.DICT,
        requires_reference=False,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=[
            "einops",
            "muq",
            "numpy",
            "omegaconf",
            "safetensors",
            "torch",
        ],
        description="SongEval song aesthetics scores for generated songs",
        paper_reference="https://arxiv.org/abs/2505.10793",
        implementation_source="https://github.com/ASLP-lab/SongEval",
    )


def register_songeval_metric(registry):
    """Register SongEval with the registry."""
    registry.register(
        SongEvalMetric,
        _songeval_metadata(),
        aliases=["SongEval", "song_eval", "songeval_metric"],
    )


songeval_setup = songeval_model_setup


if __name__ == "__main__":
    a = np.random.rand(24000).astype(np.float32)
    model = songeval_model_setup(use_gpu=True)
    print("metrics:", songeval_metric(model, a, 24000))
