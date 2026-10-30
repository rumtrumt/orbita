"""Синтез речи. Основной движок — локальный Piper, запасной — голос браузера."""

from __future__ import annotations

import asyncio
import io
import shutil
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import structlog

from orbita.domain.errors import VoiceUnavailableError

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Speech:
    audio: bytes
    media_type: str


class Synthesizer(Protocol):
    def synthesize(self, text: str) -> Speech:
        """Озвучить текст, уже подготовленный функцией to_speech."""
        ...


class PiperSynthesizer:
    """Нейросетевой голос на ONNX Runtime: работает офлайн и на CPU быстрее реального времени."""

    def __init__(self, voice: str, models_dir: Path) -> None:
        self._voice_name = voice
        self._models_dir = models_dir
        self._voice: Any = None
        self._lock = threading.Lock()

    def _download(self) -> Path:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        locale, speaker, quality = self._voice_name.split("-")
        language = locale.split("_")[0]
        prefix = f"{language}/{locale}/{speaker}/{quality}/{self._voice_name}"
        model = hf_hub_download("rhasspy/piper-voices", f"{prefix}.onnx")
        config = hf_hub_download("rhasspy/piper-voices", f"{prefix}.onnx.json")
        target = self._models_dir / "piper"
        target.mkdir(parents=True, exist_ok=True)
        local = target / f"{self._voice_name}.onnx"
        if not local.exists():
            shutil.copy(model, local)
            shutil.copy(config, target / f"{self._voice_name}.onnx.json")
        return local

    def _espeak_data(self) -> Path:
        """Копия данных espeak-ng: библиотека не открывает пути с кириллицей на Windows."""
        import piper  # noqa: PLC0415

        source = Path(piper.__file__).parent / "espeak-ng-data"
        target = self._models_dir / "espeak-ng-data"
        if not target.exists():
            shutil.copytree(source, target)
        return target

    def _load(self) -> Any:
        with self._lock:
            if self._voice is None:
                from piper import PiperVoice  # noqa: PLC0415

                try:
                    self._voice = PiperVoice.load(
                        self._download(), espeak_data_dir=self._espeak_data()
                    )
                except Exception as error:
                    raise VoiceUnavailableError(f"Piper не загрузился: {error}") from error
                log.info("tts.ready", voice=self._voice_name)
            return self._voice

    def warm_up(self) -> None:
        self._load()

    def synthesize(self, text: str) -> Speech:
        voice = self._load()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            voice.synthesize_wav(text, wav)
        return Speech(buffer.getvalue(), "audio/wav")


class EdgeSynthesizer:
    """Облачные голоса Microsoft Edge: качественнее, но нужен интернет."""

    def __init__(self, voice: str) -> None:
        self._voice = voice

    def synthesize(self, text: str) -> Speech:
        import edge_tts  # noqa: PLC0415

        async def collect() -> bytes:
            chunks = bytearray()
            async for chunk in edge_tts.Communicate(text, self._voice).stream():
                if chunk["type"] == "audio":
                    chunks.extend(chunk["data"])
            return bytes(chunks)

        try:
            return Speech(asyncio.run(collect()), "audio/mpeg")
        except Exception as error:
            raise VoiceUnavailableError(f"Edge TTS недоступен: {error}") from error
