# ShellShot

ShellShot is a full-stack AI image classification application for the Samsung Innovation Campus AI Capstone Project. It classifies uploaded images into Green Turtle, Leatherback Turtle, Loggerhead Turtle, or Unknown using trained TensorFlow/Keras models.

The app is intentionally honest about model state: it does not fabricate predictions, confidence values, or evaluation metrics. If a model file is not present or cannot load, the API remains available and the UI explains that the model is unavailable.

## Technology

- Frontend: React, Vite, TypeScript, Tailwind CSS, shadcn/ui, Wouter, Lucide React
- Backend: Python, FastAPI, Uvicorn
- Machine learning: TensorFlow/Keras, NumPy, Pillow
- API contract: OpenAPI with generated React Query and Zod helpers

## Folder structure

```text
.
├── artifacts/shellshot/       # React/Vite frontend
├── backend/
│   ├── main.py                # FastAPI application and routes
│   ├── config.py              # Environment-backed settings
│   ├── models/                # Place trained .keras files here
│   ├── services/
│   │   ├── model_service.py   # Startup model loading
│   │   └── prediction_service.py
│   └── requirements.txt
├── lib/api-spec/openapi.yaml  # API source of truth
└── README.md
```

## Model files

Copy trained models into `backend/models/` using these exact names:

```text
backend/models/mobilenetv2.keras
backend/models/resnet50.keras
backend/models/efficientnet.keras
```

Each available model is loaded once during FastAPI startup. One failed model does not prevent other models from loading.

The current MobileNetV2 and ResNet50 models include preprocessing inside the saved Keras model. The backend therefore only:

1. Converts the upload to RGB.
2. Resizes with padding to 224x224.
3. Adds the batch dimension.
4. Passes the image to the loaded model.

No application-level MobileNetV2 or ResNet preprocessing is applied a second time.

## Backend setup

### Linux/macOS

```bash
python3 -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

### Windows

```powershell
py -3 -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

When run directly, the API is available at `/api/healthz`, `/api/docs`, and `/api/openapi.json`. Through the managed project preview, the same paths are used.

## Frontend setup

Run this from the repository root in a second terminal while the backend is running:

### Linux/macOS

```bash
PORT=5173 BASE_PATH=/ API_PROXY_TARGET=http://127.0.0.1:8000 \
  pnpm --filter @workspace/shellshot run dev
```

### Windows PowerShell

```bash
$env:PORT = "5173"
$env:BASE_PATH = "/"
$env:API_PROXY_TARGET = "http://127.0.0.1:8000"
pnpm --filter @workspace/shellshot run dev
```

Open `http://localhost:5173`. The Vite development proxy forwards browser requests from `/api/*` to the FastAPI server on port 8000.

From the Replit workspace root, the managed workflow is:

```bash
pnpm --filter @workspace/shellshot run dev
```

The managed workflow supplies its own port and path settings and does not need `API_PROXY_TARGET`.

## Environment variables

The backend has sensible defaults for local development. To customize them, export these variables before starting FastAPI:

```dotenv
FRONTEND_URL=http://localhost:5173
CONFIDENCE_THRESHOLD=0.60
MAX_FILE_SIZE_MB=10
MODEL_DIR=./backend/models
```

The application does not load `.env` files automatically, so use your shell's environment-variable syntax or a process manager to apply custom values.

`Unknown` remains a real model class. A prediction is marked `low_confidence` when its confidence is below `CONFIDENCE_THRESHOLD`, and `unknown` when the model selects the Unknown class.

## API endpoints

### `GET /api/healthz`

Returns `{ "status": "ok" }`.

### `GET /api/models`

Returns the supported model metadata and availability determined from the files that loaded at startup.

### `POST /api/predict?model=mobilenet`

Accepts a multipart form upload with a `file` field. Supported model IDs are `mobilenet`, `resnet`, and `efficientnet`. Supported image types are JPG, JPEG, PNG, and WEBP, up to 10 MB.

Successful responses include:

- `prediction`
- `confidence` from 0 to 1
- `probabilities` for all four classes
- `model` and `modelId`
- `threshold`
- `status`: `detected`, `unknown`, or `low_confidence`

### `GET /api/evaluation`

Returns an explicit unavailable message until real evaluation data is connected. To provide data, add a JSON object at `backend/evaluation.json` with real results only.

### Swagger

- `/api/docs`
- `/api/openapi.json`

## Error handling

The API returns structured errors with codes such as `INVALID_FILE_TYPE`, `FILE_TOO_LARGE`, `MODEL_UNAVAILABLE`, `PREDICTION_FAILED`, `INVALID_MODEL`, and `INTERNAL_ERROR`. Stack traces are logged server-side but are not returned to users.

## Regenerating API types

After changing `lib/api-spec/openapi.yaml`, run:

```bash
pnpm --filter @workspace/api-spec run codegen
```

## Troubleshooting

- If all models show unavailable, confirm the filenames and folder are exactly correct.
- If a model is present but unavailable, inspect the FastAPI logs for a TensorFlow/Keras loading error.
- If the UI cannot reach the API, confirm both managed workflows are running and that the API service is using the `/api` path.
- Do not add preprocessing outside the saved model unless the model was trained without an internal preprocessing layer.