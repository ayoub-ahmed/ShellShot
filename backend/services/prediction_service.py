from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

CLASS_NAMES = ["Green Turtle", "Leatherback Turtle", "Loggerhead Turtle", "Unknown"]


def resize_with_pad(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Resize an image without stretching it, padding the shorter side."""
    image = image.convert("RGB")
    target_width, target_height = target_size
    scale = min(target_width / image.width, target_height / image.height)
    resized_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", target_size, (0, 0, 0))
    left = (target_width - resized.width) // 2
    top = (target_height - resized.height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def prepare_image(file_bytes: bytes, target_size: tuple[int, int]) -> np.ndarray:
    image = Image.open(BytesIO(file_bytes))
    padded = resize_with_pad(image, target_size)
    return np.expand_dims(np.asarray(padded, dtype=np.float32), axis=0)


def model_input_size(model: Any) -> tuple[int, int]:
    input_shape = model.input_shape
    if isinstance(input_shape, list) or len(input_shape) != 4:
        raise ValueError("The model must accept one image input.")

    height, width = input_shape[1], input_shape[2]
    if not isinstance(height, int) or not isinstance(width, int):
        raise ValueError("The model must define a fixed image input size.")
    return width, height


def _probabilities(raw_output: Any) -> np.ndarray:
    values = np.asarray(raw_output)
    if values.ndim > 1:
        values = values[0]
    values = values.astype(np.float64)
    if values.size != len(CLASS_NAMES):
        raise ValueError("The model output does not contain four class probabilities.")

    if not np.isfinite(values).all():
        raise ValueError("The model returned invalid probability values.")
    if np.any(values < 0) or not np.isclose(values.sum(), 1.0, atol=1e-3):
        shifted = values - np.max(values)
        exponentiated = np.exp(shifted)
        values = exponentiated / exponentiated.sum()
    return values / values.sum()


def predict(model: Any, file_bytes: bytes) -> tuple[str, float, dict[str, float]]:
    image = prepare_image(file_bytes, model_input_size(model))
    raw_output = model.predict(image, verbose=0)
    probabilities = _probabilities(raw_output)
    predicted_index = int(np.argmax(probabilities))
    return (
        CLASS_NAMES[predicted_index],
        float(probabilities[predicted_index]),
        {name: float(probabilities[index]) for index, name in enumerate(CLASS_NAMES)},
    )