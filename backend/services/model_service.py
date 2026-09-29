from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("shellshot.models")


MODEL_DEFINITIONS = {
    "mobilenet": {
        "name": "MobileNetV2",
        "architecture": "Lightweight convolutional neural network",
        "description": "A compact transfer-learning architecture designed for efficient image classification.",
        "filename": "mobilenetv2.keras",
    },
    "resnet": {
        "name": "ResNet50",
        "architecture": "Residual convolutional neural network",
        "description": "A deeper residual network that learns robust visual features through skip connections.",
        "filename": "resnet50.keras",
    },
    "efficientnet": {
        "name": "EfficientNet",
        "architecture": "Compound-scaled convolutional neural network",
        "description": "A balanced architecture that scales depth, width, and resolution for strong efficiency.",
        "filename": "efficientnet.keras",
    },
}


@dataclass
class LoadedModel:
    model: Any
    model_id: str


class ModelService:
    """Loads each available Keras model once and keeps it in memory."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self._models: dict[str, LoadedModel] = {}
        self._errors: dict[str, str] = {}

    def load_all(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        for model_id, definition in MODEL_DEFINITIONS.items():
            model_path = self.model_dir / definition["filename"]
            if not model_path.exists():
                logger.warning("%s model not found at %s", definition["name"], model_path)
                continue

            try:
                import tensorflow as tf

                logger.info("Loading %s...", definition["name"])
                loaded = tf.keras.models.load_model(model_path)
                self._models[model_id] = LoadedModel(model=loaded, model_id=model_id)
                logger.info("%s loaded successfully.", definition["name"])
            except Exception as exc:
                self._errors[model_id] = "The model file could not be loaded."
                logger.exception("Unable to load %s: %s", definition["name"], exc)

    def is_available(self, model_id: str) -> bool:
        return model_id in self._models

    def get(self, model_id: str) -> LoadedModel | None:
        return self._models.get(model_id)

    def get_error(self, model_id: str) -> str | None:
        return self._errors.get(model_id)

    def list_models(self) -> list[dict[str, object]]:
        return [
            {
                "id": model_id,
                "name": definition["name"],
                "architecture": definition["architecture"],
                "description": definition["description"],
                "available": self.is_available(model_id),
            }
            for model_id, definition in MODEL_DEFINITIONS.items()
        ]