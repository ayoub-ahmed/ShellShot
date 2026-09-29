from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.services.model_service import MODEL_DEFINITIONS, ModelService
from backend.services.prediction_service import predict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("shellshot.api")
model_service = ModelService(settings.model_dir)

SUPPORTED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def error_response(code: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": {"code": code, "message": message}},
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_service.load_all()
    yield


api = FastAPI(
    title="ShellShot API",
    version="1.0.0",
    description="TensorFlow-backed sea turtle image classification API.",
    lifespan=lifespan,
    root_path="/api",
)
api.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@api.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@api.get("/models")
async def list_models() -> dict[str, list[dict[str, object]]]:
    return {"models": model_service.list_models()}


@api.get("/evaluation")
async def evaluation() -> dict[str, object]:
    evaluation_path = Path(__file__).parent / "evaluation.json"
    if not evaluation_path.exists():
        return {
            "available": False,
            "message": "Evaluation results will appear here after model evaluation data is connected.",
        }

    try:
        data = json.loads(evaluation_path.read_text(encoding="utf-8"))
        return {"available": True, "message": "Evaluation data loaded.", "metrics": data}
    except (OSError, json.JSONDecodeError):
        return {
            "available": False,
            "message": "Evaluation data is configured but could not be read.",
        }


@api.post("/predict")
async def predict_image(
    file: Annotated[UploadFile, File(...)],
    model: Annotated[str, Query(...)],
) -> JSONResponse:
    if model not in MODEL_DEFINITIONS:
        return error_response(
            "INVALID_MODEL",
            "Select one of the supported models: MobileNetV2, ResNet50, or EfficientNet.",
            400,
        )

    if not model_service.is_available(model):
        return error_response(
            "MODEL_UNAVAILABLE",
            "The selected model is not currently available. Add its .keras file to backend/models/.",
            503,
        )

    suffix = Path(file.filename or "").suffix.lower()
    if file.content_type not in SUPPORTED_MIME_TYPES or suffix not in SUPPORTED_EXTENSIONS:
        return error_response(
            "INVALID_FILE_TYPE",
            "Upload a JPG, JPEG, PNG, or WEBP image.",
            400,
        )

    content = await file.read()
    if len(content) > settings.max_file_size_bytes:
        return error_response(
            "FILE_TOO_LARGE",
            f"Images must be {settings.max_file_size_mb} MB or smaller.",
            413,
        )

    try:
        prediction, confidence, probabilities = predict(model_service.get(model).model, content)  # type: ignore[union-attr]
        status = "unknown" if prediction == "Unknown" else (
            "low_confidence" if confidence < settings.confidence_threshold else "detected"
        )
        return JSONResponse(
            content={
                "success": True,
                "model": MODEL_DEFINITIONS[model]["name"],
                "modelId": model,
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": probabilities,
                "threshold": settings.confidence_threshold,
                "status": status,
            }
        )
    except Exception:
        logger.exception("Prediction failed for model %s", model)
        return error_response(
            "PREDICTION_FAILED",
            "The image could not be analyzed. Try another supported image.",
            422,
        )


@api.exception_handler(Exception)
async def unhandled_exception(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled API error: %s", exc)
    return error_response("INTERNAL_ERROR", "The server could not complete that request.", 500)


# The shared proxy forwards /api/* without stripping the prefix. Mounting the
# documented FastAPI app keeps /docs and /openapi.json available at /api/docs
# and /api/openapi.json while keeping route definitions clean.
app = FastAPI(docs_url=None, openapi_url=None)
app.mount("/api", api)