from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("shellshot.models")

KNOWN_METADATA: dict[str, dict[str, str]] = {
    "mobilenet": {
        "name": "MobileNetV2",
        "architecture": "Lightweight convolutional neural network",
        "description": "A compact transfer-learning architecture designed for efficient image classification.",
    },
    "mobilenetv2": {
        "name": "MobileNetV2",
        "architecture": "Lightweight convolutional neural network",
        "description": "A compact transfer-learning architecture designed for efficient image classification.",
    },
    "resnet": {
        "name": "ResNet50",
        "architecture": "Residual convolutional neural network",
        "description": "A deeper residual network that learns robust visual features through skip connections.",
    },
    "resnet50": {
        "name": "ResNet50",
        "architecture": "Residual convolutional neural network",
        "description": "A deeper residual network that learns robust visual features through skip connections.",
    },
    "efficientnet": {
        "name": "EfficientNet",
        "architecture": "Compound-scaled convolutional neural network",
        "description": "A balanced architecture that scales depth, width, and resolution for strong efficiency.",
    },
    "efficientnetv2s": {
        "name": "EfficientNetV2S",
        "architecture": "Compound-scaled convolutional neural network",
        "description": "A balanced architecture that scales depth, width, and resolution for strong efficiency.",
    },
}

PREPROCESSORS: dict[str, Any] = {}


def _get_preprocessors() -> dict[str, Any]:
    global PREPROCESSORS
    if not PREPROCESSORS:
        try:
            import tensorflow as tf

            PREPROCESSORS = {
                "mobilenet": tf.keras.applications.mobilenet_v2.preprocess_input,
                "mobilenetv2": tf.keras.applications.mobilenet_v2.preprocess_input,
                "resnet": tf.keras.applications.resnet50.preprocess_input,
                "resnet50": tf.keras.applications.resnet50.preprocess_input,
                "efficientnet": tf.keras.applications.efficientnet_v2.preprocess_input,
                "efficientnetv2s": tf.keras.applications.efficientnet_v2.preprocess_input,
            }
        except ImportError:
            pass
    return PREPROCESSORS


@dataclass
class ModelMetadata:
    model_id: str
    name: str
    architecture: str
    description: str
    filename: str


@dataclass
class LoadedModel:
    model: Any
    model_id: str
    metadata: ModelMetadata


def format_model_name(raw_stem: str) -> str:
    """Format a raw filename stem into a readable display name."""
    clean = re.sub(r"[_\-]+", " ", raw_stem).strip()
    return " ".join(word.capitalize() for word in clean.split())


class ModelService:
    """Dynamically discovers and loads all available Keras models from the model directory."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self._models: dict[str, LoadedModel] = {}
        self._metadata_map: dict[str, ModelMetadata] = {}
        self._aliases: dict[str, str] = {}
        self._errors: dict[str, str] = {}

    def _resolve_id(self, model_id: str) -> str:
        """Resolve aliases like 'mobilenet' -> 'mobilenetv2' if needed."""
        return self._aliases.get(model_id, model_id)

    def load_all(self) -> None:
        """Scan model_dir and load each detected model file exactly once."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_paths = sorted(
            list(self.model_dir.glob("*.keras")) + list(self.model_dir.glob("*.h5"))
        )

        self._metadata_map.clear()
        self._aliases.clear()

        if not model_paths:
            for std_id in ["mobilenet", "resnet", "efficientnet"]:
                meta = KNOWN_METADATA.get(std_id, {})
                self._metadata_map[std_id] = ModelMetadata(
                    model_id=std_id,
                    name=meta.get("name", std_id.title()),
                    architecture=meta.get("architecture", "Convolutional Neural Network"),
                    description=meta.get("description", "Image classification model."),
                    filename=f"{std_id}.keras",
                )
            return

        import tensorflow as tf

        preproc_map = _get_preprocessors()

        for model_path in model_paths:
            stem_lower = model_path.stem.lower()
            model_id = re.sub(r"[^a-z0-9_\-]", "", stem_lower)

            if "mobilenet" in stem_lower:
                self._aliases["mobilenet"] = model_id
            elif "resnet" in stem_lower:
                self._aliases["resnet"] = model_id
            elif "efficientnet" in stem_lower:
                self._aliases["efficientnet"] = model_id

            known = KNOWN_METADATA.get(model_id) or KNOWN_METADATA.get(stem_lower)
            if known:
                meta = ModelMetadata(
                    model_id=model_id,
                    name=known["name"],
                    architecture=known["architecture"],
                    description=known["description"],
                    filename=model_path.name,
                )
            else:
                formatted_name = format_model_name(model_path.stem)
                meta = ModelMetadata(
                    model_id=model_id,
                    name=formatted_name,
                    architecture="Custom Vision Classifier",
                    description=f"Custom neural network loaded from {model_path.name}.",
                    filename=model_path.name,
                )

            self._metadata_map[model_id] = meta

            if model_id in self._models:
                continue

            try:
                logger.info("Loading %s from %s...", meta.name, model_path)
                custom_objs = {}
                if model_id in preproc_map:
                    custom_objs["preprocess_input"] = preproc_map[model_id]
                elif stem_lower in preproc_map:
                    custom_objs["preprocess_input"] = preproc_map[stem_lower]

                try:
                    loaded = tf.keras.models.load_model(model_path, custom_objects=custom_objs)
                except Exception:
                    loaded = tf.keras.models.load_model(model_path)

                output_shape = loaded.output_shape
                if isinstance(output_shape, list) or not output_shape or output_shape[-1] != 4:
                    raise ValueError("The model must return 4 class outputs.")

                self._models[model_id] = LoadedModel(model=loaded, model_id=model_id, metadata=meta)
                logger.info("%s (%s) loaded successfully.", meta.name, model_id)
            except Exception as exc:
                self._errors[model_id] = f"Could not load {model_path.name}: {exc}"
                logger.exception("Unable to load %s (%s): %s", meta.name, model_id, exc)

    def is_available(self, model_id: str) -> bool:
        real_id = self._resolve_id(model_id)
        return real_id in self._models

    def is_known(self, model_id: str) -> bool:
        real_id = self._resolve_id(model_id)
        return real_id in self._metadata_map

    def get(self, model_id: str) -> LoadedModel | None:
        real_id = self._resolve_id(model_id)
        return self._models.get(real_id)

    def get_error(self, model_id: str) -> str | None:
        real_id = self._resolve_id(model_id)
        return self._errors.get(real_id)

    def list_models(self) -> list[dict[str, object]]:
        self.load_all()
        return [
            {
                "id": meta.model_id,
                "name": meta.name,
                "architecture": meta.architecture,
                "description": meta.description,
                "available": self.is_available(meta.model_id),
            }
            for meta in self._metadata_map.values()
        ]