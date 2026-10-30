"""Распознавание речи через Whisper (реализация faster-whisper на CTranslate2)."""

from __future__ import annotations

import io
import os
import re
import threading
import time
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Protocol

import av
import numpy as np
import structlog

from orbita.domain.errors import SpeechNotRecognizedError, VoiceUnavailableError
from orbita.voice.transcript import polish

log = structlog.get_logger(__name__)

# Подсказка смещает распознавание к лексике планировщика и к записи чисел цифрами.
PROMPT = (
    "Напомни завтра в 10 утра позвонить Ивану. Поставь встречу с командой в пятницу в 15:00. "
    "Что у меня на следующей неделе? Найди задачу про договор. Удали созвон с заказчиком. "
    "Перенеси планёрку на час позже. Отмени последнее действие. Отметь отчёт выполненным."
)
# На тишине Whisper дописывает фразы из титров, на которых обучался.
HALLUCINATIONS = re.compile(
    r"субтитр|продолжение следует|спасибо за просмотр|подписывайтесь|ставьте лайк|dimatorzok"
    r"|amara\.org|редактор субтитров|корректор",
    re.IGNORECASE,
)
MAX_NO_SPEECH = 0.75
SAMPLE_RATE = 16000


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    duration: float
    seconds_spent: float


class SpeechRecognizer(Protocol):
    def transcribe(self, audio: bytes) -> Transcript:
        """Распознать аудио в любом контейнере, который понимает FFmpeg."""
        ...


def _expose_cuda_libraries() -> None:
    """Сделать видимыми DLL из пакетов nvidia-*: CTranslate2 ищет их через PATH."""
    spec = find_spec("nvidia")
    for root in (spec.submodule_search_locations or []) if spec else []:
        for folder in Path(root).glob("*/bin"):
            os.environ["PATH"] = f"{folder}{os.pathsep}{os.environ['PATH']}"


def _decode(audio: bytes) -> np.ndarray:
    """WebM/Opus из браузера в 16 кГц моно: формат, который принимает Whisper."""
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    chunks: list[np.ndarray] = []
    with av.open(io.BytesIO(audio), mode="r") as container:
        for frame in container.decode(audio=0):
            chunks.extend(chunk.to_ndarray() for chunk in resampler.resample(frame))
        chunks.extend(chunk.to_ndarray() for chunk in resampler.resample(None))
    if not chunks:
        raise SpeechNotRecognizedError("Пустая запись")
    return np.concatenate(chunks, axis=1).reshape(-1).astype(np.float32) / 32768.0


class WhisperRecognizer:
    def __init__(self, model: str, device: str, compute_type: str, beam_size: int) -> None:
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._model: Any = None
        self._lock = threading.Lock()
        self.active_device: str | None = None

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                self._model = self._load_with_fallback()
            return self._model

    def _load_with_fallback(self) -> Any:
        from faster_whisper import WhisperModel  # noqa: PLC0415 — тяжёлый импорт

        _expose_cuda_libraries()
        if self._device == "auto":
            plans = [("cuda", "int8_float16"), ("cpu", "int8")]
        else:
            plans = [(self._device, self._compute_type)]
        last_error: Exception | None = None
        for device, compute in plans:
            compute_type = compute if self._compute_type == "auto" else self._compute_type
            try:
                started = time.perf_counter()
                model = WhisperModel(self._model_name, device=device, compute_type=compute_type)
                self._probe(model)
            except Exception as error:
                log.warning("stt.load_failed", device=device, error=str(error))
                last_error = error
                continue
            self.active_device = device
            log.info("stt.ready", device=device, seconds=round(time.perf_counter() - started, 1))
            return model
        raise VoiceUnavailableError(f"Whisper не загрузился: {last_error}")

    @staticmethod
    def _probe(model: Any) -> None:
        """Прогон на тишине: отсутствие CUDA-библиотек проявляется только при инференсе."""
        segments, _ = model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language="ru")
        list(segments)

    def warm_up(self) -> None:
        self._load()

    def transcribe(self, audio: bytes) -> Transcript:
        model = self._load()
        started = time.perf_counter()
        try:
            segments, info = model.transcribe(
                _decode(audio),
                language="ru",
                beam_size=self._beam_size,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                initial_prompt=PROMPT,
                condition_on_previous_text=False,
            )
            parts = [
                segment.text.strip()
                for segment in segments
                if segment.no_speech_prob < MAX_NO_SPEECH
                and not HALLUCINATIONS.search(segment.text)
            ]
        except SpeechNotRecognizedError:
            raise
        except Exception as error:
            log.warning("stt.decode_failed", error=str(error))
            raise SpeechNotRecognizedError("Не удалось прочитать аудио") from error
        text = polish(" ".join(parts).strip())
        if not text:
            raise SpeechNotRecognizedError("Речь не распознана")
        return Transcript(text, float(info.duration), time.perf_counter() - started)
