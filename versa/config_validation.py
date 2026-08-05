#!/usr/bin/env python3

# Copyright 2026 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Validation helpers for score configuration files."""

import ast
import importlib.util
import inspect
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Set

from versa.definition import MetricRegistry
from versa.metric_discovery import create_metric_discovery_registry

_BASE_CONFIG_KEYS = {"name", "use_gpu"}
_GPU_REQUIRED_PREFIXES = ("qwen2_audio_", "qwen_omni_")
_GPU_REQUIRED_METRICS = {"audiobox_aesthetics"}


def _effective_dependencies(metric_name, metadata, config):
    """Return the dependencies actually required by this metric config.

    The speaker metric bundles both backends in its metadata; only the
    backend selected by the config is required at runtime.
    """
    name = str(metric_name).lower()
    if name in {"speaker", "spk_similarity", "speaker_similarity"}:
        from versa.utterance_metrics.speaker import resolve_speaker_backend

        try:
            backend = resolve_speaker_backend(
                model_tag=config.get("model_tag", "default"),
                backend=config.get("backend"),
                model_path=config.get("model_path"),
                model_config=config.get("model_config"),
            )
        except ValueError:
            return list(metadata.dependencies)
        excluded = "espnet2" if backend == "huggingface" else "transformers"
        return [dep for dep in metadata.dependencies if dep != excluded]
    return list(metadata.dependencies)


@dataclass(frozen=True)
class ScoreConfigValidationError:
    """One score configuration validation issue."""

    index: Optional[int]
    metric_name: Optional[str]
    message: str

    def format(self) -> str:
        location = "score_config"
        if self.index is not None:
            location += f"[{self.index}]"
        if self.metric_name:
            location += f" ({self.metric_name})"
        return f"{location}: {self.message}"


class ScoreConfigValidationException(ValueError):
    """Raised when a score configuration cannot be used safely."""

    def __init__(self, errors: Iterable[ScoreConfigValidationError]):
        self.errors = list(errors)
        message = "Invalid score configuration:\n" + "\n".join(
            f"- {error.format()}" for error in self.errors
        )
        super().__init__(message)


def validate_score_config(
    score_config: Any,
    *,
    registry: Optional[MetricRegistry] = None,
    use_gt: bool = True,
    use_gt_text: bool = False,
    use_gpu: bool = False,
) -> None:
    """Validate a YAML score config before metric setup or scoring starts."""

    errors: List[ScoreConfigValidationError] = []
    if not isinstance(score_config, list) or not score_config:
        raise ScoreConfigValidationException(
            [
                ScoreConfigValidationError(
                    None,
                    None,
                    "expected a non-empty YAML list of metric mappings",
                )
            ]
        )

    registry = registry or create_metric_discovery_registry()
    discovery_registry: Optional[MetricRegistry] = None

    for index, config in enumerate(score_config):
        if not isinstance(config, dict):
            errors.append(
                ScoreConfigValidationError(
                    index,
                    None,
                    "expected a mapping with at least a 'name' key",
                )
            )
            continue

        metric_name = config.get("name")
        if not isinstance(metric_name, str) or not metric_name:
            errors.append(
                ScoreConfigValidationError(
                    index,
                    None,
                    "missing required metric name; add 'name: <metric>'",
                )
            )
            continue

        metadata = registry.get_metadata(metric_name)
        metric_class = registry.get_metric(metric_name)
        if metadata is None:
            discovery_registry = (
                discovery_registry or create_metric_discovery_registry()
            )
            metadata = discovery_registry.get_metadata(metric_name)
            metric_class = discovery_registry.get_metric(metric_name)
        if metadata is None or metric_class is None:
            errors.append(
                ScoreConfigValidationError(
                    index,
                    metric_name,
                    _unknown_metric_message(registry, metric_name),
                )
            )
            continue

        if metadata.requires_reference and not use_gt:
            errors.append(
                ScoreConfigValidationError(
                    index,
                    metric_name,
                    "requires reference audio; provide --gt or remove this metric",
                )
            )

        if metadata.requires_text and not use_gt_text:
            errors.append(
                ScoreConfigValidationError(
                    index,
                    metric_name,
                    "requires reference text; provide --text or remove this metric",
                )
            )

        if _requires_gpu_flag(metadata.name) and not use_gpu:
            errors.append(
                ScoreConfigValidationError(
                    index,
                    metric_name,
                    "requires GPU execution; rerun with --use_gpu or remove this metric",
                )
            )

        missing_dependencies = [
            dependency
            for dependency in _effective_dependencies(metric_name, metadata, config)
            if not _dependency_available(dependency)
        ]
        if missing_dependencies:
            errors.append(
                ScoreConfigValidationError(
                    index,
                    metric_name,
                    "missing optional dependencies: "
                    + ", ".join(missing_dependencies)
                    + ". Install them or remove this metric from the config",
                )
            )

        allowed_keys = (
            None
            if metric_class.__name__ == "_DiscoveredMetric"
            else _allowed_config_keys(metric_class)
        )
        if allowed_keys:
            unknown_keys = sorted(set(config) - allowed_keys)
            if unknown_keys:
                errors.append(
                    ScoreConfigValidationError(
                        index,
                        metric_name,
                        "unknown parameter(s): "
                        + ", ".join(unknown_keys)
                        + ". Allowed parameters: "
                        + ", ".join(sorted(allowed_keys)),
                    )
                )

    if errors:
        raise ScoreConfigValidationException(errors)


def _unknown_metric_message(registry: MetricRegistry, metric_name: str) -> str:
    message = f"unknown metric name '{metric_name}'"
    suggestions = _suggest_metric_names(registry, metric_name)
    if suggestions:
        message += "; did you mean " + ", ".join(suggestions) + "?"
    return message


def _suggest_metric_names(registry: MetricRegistry, metric_name: str) -> List[str]:
    query = metric_name.lower()
    names = set(registry.list_metrics()) | set(registry.list_aliases())
    return sorted(name for name in names if query in name.lower())[:5]


def _requires_gpu_flag(metric_name: str) -> bool:
    return metric_name in _GPU_REQUIRED_METRICS or metric_name.startswith(
        _GPU_REQUIRED_PREFIXES
    )


@lru_cache(maxsize=None)
def _dependency_available(dependency: str) -> bool:
    return importlib.util.find_spec(dependency) is not None


@lru_cache(maxsize=None)
def _allowed_config_keys(metric_class: type) -> Optional[frozenset]:
    keys: Set[str] = set(_BASE_CONFIG_KEYS)
    found_source = False

    for cls in inspect.getmro(metric_class):
        if cls is object:
            continue
        try:
            source = inspect.getsource(cls)
        except (OSError, TypeError):
            continue

        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        found_source = True
        for node in ast.walk(tree):
            key = _self_config_get_key(node) or _self_config_subscript_key(node)
            if key:
                keys.add(key)

    return frozenset(keys) if found_source else None


def _self_config_get_key(node: ast.AST) -> Optional[str]:
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "get":
        return None
    if not _is_self_config(node.func.value):
        return None
    if not node.args:
        return None
    first_arg = node.args[0]
    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
        return first_arg.value
    return None


def _self_config_subscript_key(node: ast.AST) -> Optional[str]:
    if not isinstance(node, ast.Subscript) or not _is_self_config(node.value):
        return None
    slice_node = node.slice
    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
        return slice_node.value
    return None


def _is_self_config(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "config"
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )
