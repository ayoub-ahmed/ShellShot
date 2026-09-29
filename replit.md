# ShellShot

ShellShot is a TensorFlow-backed sea turtle image classification app for the Samsung Innovation Campus AI Capstone Project.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the FastAPI service through the managed API workflow
- `pnpm --filter @workspace/shellshot run dev` — run the React/Vite frontend through the managed web workflow
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `python -m uvicorn backend.main:app --reload --port 8000` — run FastAPI directly from the repository root
- Python dependencies are listed in `backend/requirements.txt`

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Python FastAPI + Uvicorn
- ML: TensorFlow/Keras, NumPy, Pillow
- Frontend API types: OpenAPI + Orval
- API codegen: Orval (from OpenAPI spec)
- Frontend: React, Vite, TypeScript, Tailwind CSS, shadcn/ui, Wouter

## Where things live

- `backend/main.py` — FastAPI app, validation, endpoints, and startup lifecycle
- `backend/services/model_service.py` — one-time model loading and availability state
- `backend/services/prediction_service.py` — RGB conversion, resize-with-pad, and model inference
- `backend/models/` — place `mobilenetv2.keras`, `resnet50.keras`, and `efficientnet.keras` here
- `lib/api-spec/openapi.yaml` — source of truth for the frontend/backend API contract
- `artifacts/shellshot/` — React/Vite application

## Architecture decisions

- TensorFlow preprocessing stays inside saved Keras models when present; the backend only converts RGB, resize-pads to 224x224, and batches the image.
- Missing or unloadable model files are reported as `MODEL_UNAVAILABLE`; the API stays up so the UI can demonstrate the full unavailable state.
- Uploaded images are read in memory and never stored permanently.
- Evaluation values are intentionally empty until real evaluation data is supplied in `backend/evaluation.json`.

## Product

- Upload JPG, JPEG, PNG, and WEBP images up to 10 MB.
- Select MobileNetV2, ResNet50, or EfficientNet and receive the real model prediction, confidence, all class probabilities, and explicit Unknown/Low Confidence status.
- Review model availability, evaluation readiness, project context, and technical architecture.

## User preferences

- Do not invent predictions, accuracy values, or evaluation results.
- Keep the backend Python/FastAPI/TensorFlow based; do not replace it with Node.js.

## Gotchas

- Add model files to `backend/models/` before expecting a successful prediction.
- After changing `lib/api-spec/openapi.yaml`, run `pnpm --filter @workspace/api-spec run codegen`.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
