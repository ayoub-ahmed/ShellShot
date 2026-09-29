from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    frontend_url: str
    confidence_threshold: float
    max_file_size_mb: int
    model_dir: Path

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


def load_settings() -> Settings:
    model_dir = Path(os.getenv("MODEL_DIR", "./backend/models"))
    if not model_dir.is_absolute():
        model_dir = Path.cwd() / model_dir

    return Settings(
        frontend_url=os.getenv("FRONTEND_URL", "http://localhost:5173"),
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.60")),
        max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "10")),
        model_dir=model_dir,
    )


settings = load_settings()