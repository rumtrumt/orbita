"""Декодер аудио. Заглушка распознавателя в API-тестах его не затрагивает."""

import io
import wave

import numpy as np
import pytest

from orbita.domain.errors import SpeechNotRecognizedError
from orbita.voice.stt import SAMPLE_RATE, WhisperRecognizer, _decode


def wav_bytes(seconds: float = 1.0, rate: int = 22050) -> bytes:
    samples = (np.sin(np.arange(int(rate * seconds)) / 20) * 8000).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(rate)
        target.writeframes(samples.tobytes())
    return buffer.getvalue()


def test_decode_resamples_to_whisper_format():
    audio = _decode(wav_bytes(seconds=1.0))
    assert audio.dtype == np.float32
    assert abs(len(audio) - SAMPLE_RATE) < SAMPLE_RATE * 0.02
    assert 0.1 < np.abs(audio).max() <= 1.0


class SilentModel:
    def transcribe(self, *args, **kwargs):
        raise AssertionError("Модель не должна получать битое аудио")


def test_broken_audio_is_reported_as_unrecognized():
    recognizer = WhisperRecognizer("x", "cpu", "int8", 1)
    recognizer._model = SilentModel()
    with pytest.raises(SpeechNotRecognizedError):
        recognizer.transcribe(b"not audio at all")
