---
name: ShellShot runtime
description: FastAPI backend routing and model behavior that future work should preserve.
---

ShellShot's Python backend lives at the repository root while its managed API workflow starts in the API artifact directory, so the workflow must change to the repository root before launching Uvicorn.

**Why:** The first managed launch could not import the root-level backend package until the working-directory difference was accounted for.

**How to apply:** Keep the API artifact workflow's repository-root handoff when changing backend startup or deployment settings.