"""Живой прогон голосового конвейера: настоящие Piper, Whisper, сервер и Chromium.

Фразы синтезируются Piper, подаются в браузер как сигнал виртуального микрофона и проходят весь
путь: MediaRecorder → /api/voice → Whisper → диалог → ответ → синтез. Запускается вручную на
машине с моделями, потому что тянет гигабайты весов и работает минутами.

    python tests/e2e/voice_roundtrip.py http://127.0.0.1:8000
"""

from __future__ import annotations

import io
import sys
import tempfile
import time
import wave
from pathlib import Path

from playwright.sync_api import sync_playwright

from orbita.config import get_settings
from orbita.voice.speech_text import to_speech
from orbita.voice.tts import PiperSynthesizer

PHRASES = [
    "Поставь встречу с Иваном в понедельник",
    "В пятнадцать ноль ноль",
    "Напомни мне завтра в десять утра позвонить маме",
    "Что у меня в понедельник?",
]


def with_padding(audio: bytes) -> bytes:
    """Добавить тишину: виртуальный микрофон зациклен, а запись должна остановиться сама."""
    with wave.open(io.BytesIO(audio)) as source:
        params = source.getparams()
        frames = source.readframes(params.nframes)
    silence = b"\0" * (params.framerate * params.sampwidth * 5)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setparams(params)
        target.writeframes(silence[: params.framerate] + frames + silence)
    return buffer.getvalue()


def main(base: str) -> int:
    settings = get_settings()
    synthesizer = PiperSynthesizer(settings.piper_voice, settings.models_dir)
    failures = 0
    token = ""
    with sync_playwright() as playwright, tempfile.TemporaryDirectory() as folder:
        for index, phrase in enumerate(PHRASES):
            sample = Path(folder) / f"{index}.wav"
            sample.write_bytes(with_padding(synthesizer.synthesize(to_speech(phrase)).audio))
            browser = playwright.chromium.launch(
                args=[
                    "--no-proxy-server",
                    "--use-fake-ui-for-media-stream",
                    "--use-fake-device-for-media-stream",
                    f"--use-file-for-fake-audio-capture={sample.as_posix()}",
                    "--autoplay-policy=no-user-gesture-required",
                ]
            )
            context = browser.new_context(permissions=["microphone"])
            if index:
                context.add_init_script(f"localStorage.setItem('orbita.token', '{token}')")
            page = context.new_page()
            page.goto(base)
            if index == 0:
                page.fill("#gate-name", f"Голос {int(time.time())}")
                page.fill("#gate-pin", "1234")
                page.click(".gate__submit")
            page.wait_for_selector("#composer-input")
            page.wait_for_function("document.body.dataset.mode === 'idle'")
            token = page.evaluate("localStorage.getItem('orbita.token')")
            before = page.locator(".msg--assistant").count()
            started = time.perf_counter()
            page.click("#mic")
            page.wait_for_function(
                f"document.querySelectorAll('.msg--assistant').length > {before}", timeout=60000
            )
            page.wait_for_timeout(300)
            heard = page.locator(".msg--user").last.inner_text()
            reply = page.locator(".msg--assistant").last.inner_text()
            print(f"фраза:  {phrase}\nуслышано: {heard}\nответ: {reply}")
            print(f"время: {time.perf_counter() - started:.1f} c\n")
            failures += not reply
            browser.close()
    return failures


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"))
