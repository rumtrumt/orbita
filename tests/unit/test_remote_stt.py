"""Облачное распознавание: разбор ответов и отказов API."""

import httpx
import pytest

from orbita.domain.errors import SpeechNotRecognizedError, VoiceUnavailableError
from orbita.voice.remote_stt import RemoteRecognizer


def recognizer(handler):
    return RemoteRecognizer(
        "https://stt.test/v1", "whisper", "key", 5, transport=httpx.MockTransport(handler)
    )


def segment(text, no_speech=0.0):
    return {"text": text, "no_speech_prob": no_speech}


def test_returns_text_and_sends_expected_request():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = request.read()
        return httpx.Response(200, json={"duration": 2, "segments": [segment(" Привет ")]})

    result = recognizer(handler).transcribe(b"\x1a\x45\xdf\xa3audio")
    assert result.text == "Привет"
    assert result.duration == 2
    assert seen["url"] == "https://stt.test/v1/audio/transcriptions"
    assert seen["auth"] == "Bearer key"
    assert b"speech.webm" in seen["body"]


def test_silence_and_hallucinations_are_filtered():
    body = {"segments": [segment("Продолжение следует"), segment("шум", no_speech=0.9)]}
    with pytest.raises(SpeechNotRecognizedError):
        recognizer(lambda _: httpx.Response(200, json=body)).transcribe(b"RIFFaudio")


def test_rejected_recording_is_reported_as_unrecognized():
    with pytest.raises(SpeechNotRecognizedError):
        recognizer(lambda _: httpx.Response(400, text="too short")).transcribe(b"x")


def test_service_failure_is_unavailable():
    with pytest.raises(VoiceUnavailableError):
        recognizer(lambda _: httpx.Response(429, text="limit")).transcribe(b"x")


def test_network_failure_is_unavailable():
    def handler(request):
        raise httpx.ConnectError("нет сети")

    with pytest.raises(VoiceUnavailableError):
        recognizer(handler).transcribe(b"x")
