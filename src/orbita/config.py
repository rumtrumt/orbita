"""Конфигурация приложения. Все параметры задаются переменными окружения ORBITA_* или .env."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from platformdirs import user_data_path
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class NluMode(StrEnum):
    RULES = "rules"
    LLM = "llm"
    HYBRID = "hybrid"


class TtsProvider(StrEnum):
    PIPER = "piper"
    EDGE = "edge"
    OFF = "off"


def _default_data_dir() -> Path:
    # Каталог вне проекта: CTranslate2 на Windows не открывает модели по путям с кириллицей.
    return user_data_path("orbita", appauthor=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ORBITA_", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    data_dir: Path = Field(default_factory=_default_data_dir)
    database_url: str | None = None
    default_timezone: str = "Europe/Moscow"

    nlu_mode: NluMode = NluMode.HYBRID
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "qwen3:4b"
    llm_api_key: SecretStr = SecretStr("ollama")
    llm_timeout_seconds: float = 25.0

    whisper_model: str = "large-v3-turbo"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"
    whisper_beam_size: int = 5
    max_audio_bytes: int = 15 * 1024 * 1024

    tts_provider: TtsProvider = TtsProvider.PIPER
    piper_voice: str = "ru_RU-irina-medium"
    edge_voice: str = "ru-RU-SvetlanaNeural"

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    semantic_search: bool = True

    meeting_default_minutes: int = 60
    meeting_remind_before_minutes: int = 15
    reminder_poll_seconds: float = 15.0

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def resolved_database_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'orbita.db').as_posix()}"

    @property
    def timezone(self) -> ZoneInfo:
        return ZoneInfo(self.default_timezone)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
