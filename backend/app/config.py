from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Automatic Fine-Tune Framework"
    default_model_id: str = "unsloth/Llama-3.2-3B-bnb-4bit"
    max_seq_length: int = 2048
    data_dir: Path = PROJECT_ROOT / "data"
    runs_dir: Path = PROJECT_ROOT / "data" / "runs"
    hf_token: str | None = None
    cors_origins: list[str] = [
        "http://localhost:3003",
        "http://127.0.0.1:3003",
    ]

    @field_validator("data_dir", "runs_dir", mode="before")
    @classmethod
    def resolve_paths(cls, value):
        if isinstance(value, str):
            path = Path(value)
            return path if path.is_absolute() else PROJECT_ROOT / path
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
