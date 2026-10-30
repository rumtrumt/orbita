"""Распознавание через OpenAI-совместимый API, например Groq.

Нужно для серверов без GPU и с малым объёмом памяти, где локальная модель не помещается.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import structlog

from orbita.domain.errors import SpeechNotRecognizedError, VoiceUnavailableError
from orbita.voice.stt import HALLUCINATIONS, MAX_NO_SPEECH, PROMPT, Transcript

log = structlog.get_logger(__name__)


def _filename(audio: bytes) -> str:
    """API определяет формат по расширению, а браузеры записывают в разных контейнерах."""
    if audio.startswith(b"\x1a\x45\xdf\xa3"):
        return "speech.webm"
    if audio.startswith(b"RIFF"):
        return "speech.wav"
    if audio.startswith(b"OggS"):
        return "speech.ogg"
    if audio[4:8] == b"ftyp":
        return "speech.mp4"
    return "speech.webm"


def _text_from(body: dict[str, Any]) -> str:
    segments = body.get("segments")
    if segments is None:
        return str(body.get("text", "")).strip()
    parts = [
        str(segment.get("text", "")).strip()
        for segment in segments
        if float(segment.get("no_speech_prob", 0.0)) < MAX_NO_SPEECH
        and not HALLUCINATIONS.search(str(segment.get("text", "")))
    ]
    return " ".join(parts).strip()


class RemoteRecognizer:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/audio/transcriptions"
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._http = httpx.Client(timeout=timeout, transport=transport)

    def transcribe(self, audio: bytes) -> Transcript:
        started = time.perf_counter()
        try:
            response = self._http.post(
                self._url,
                headers=self._headers,
                files={"file": (_filename(audio), audio)},
                data={
                    "model": self._model,
                    "language": "ru",
                    "prompt": PROMPT,
                    "response_format": "verbose_json",
                    "temperature": "0",
                },
            )
        except httpx.HTTPError as error:
            raise VoiceUnavailableError(f"Сервис распознавания недоступен: {error}") from error
        if response.status_code == 400:
            # Так API отвечает на пустую или слишком короткую запись.
            log.warning("stt.rejected", body=response.text[:200])
            raise SpeechNotRecognizedError("Запись не принята")
        if response.is_error:
            raise VoiceUnavailableError(f"Сервис распознавания: {response.status_code}")
        body = response.json()
        text = _text_from(body)
        if not text:
            raise SpeechNotRecognizedError("Речь не распознана")
        return Transcript(text, float(body.get("duration", 0.0)), time.perf_counter() - started)
