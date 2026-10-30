"""Сквозной замер речи: фраза → синтез → Whisper → правила → сверка с золотым примером.

    python evals/run_voice_eval.py [--voices ru_RU-irina-medium ru_RU-ruslan-medium] [--errors]

Показывает, сколько команд теряется именно на распознавании. Голос синтетический, поэтому оценка
оптимистична по сравнению с живой речью в шумной комнате.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parents[1]))

from tests.golden.checks import CASES, mismatches

from orbita.config import Settings
from orbita.nlu.commands import NluContext
from orbita.nlu.rules import RuleInterpreter
from orbita.voice.speech_text import to_speech
from orbita.voice.stt import WhisperRecognizer
from orbita.voice.tts import EdgeSynthesizer, PiperSynthesizer

MSK = ZoneInfo("Europe/Moscow")
NOW = datetime(2026, 10, 30, 14, 0, tzinfo=MSK)


def normalized(text: str) -> str:
    return re.sub(r"[^\w:]+", " ", text.lower().replace("ё", "е")).strip()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--voices",
        nargs="+",
        default=[
            "ru_RU-irina-medium",
            "ru_RU-ruslan-medium",
            "ru_RU-denis-medium",
        ],
    )
    parser.add_argument("--errors", action="store_true")
    args = parser.parse_args()

    settings = Settings(_env_file=None)
    recognizer = WhisperRecognizer(
        settings.whisper_model,
        settings.whisper_device,
        settings.whisper_compute_type,
        settings.whisper_beam_size,
    )
    rules = RuleInterpreter()
    context = NluContext(now=NOW, timezone=MSK)
    print("| Голос | Фраз | Команда верна после распознавания | Текст совпал | Время Whisper |")
    print("|---|---:|---:|---:|---:|")
    for voice in args.voices:
        synthesizer = (
            EdgeSynthesizer(voice)
            if voice.startswith("ru-RU")
            else PiperSynthesizer(voice, settings.models_dir)
        )
        correct = exact = 0
        spent = 0.0
        for case in CASES:
            transcript = recognizer.transcribe(
                synthesizer.synthesize(to_speech(case["text"])).audio
            )
            spent += transcript.seconds_spent
            # Буква ё не влияет на смысл, а в заголовке Whisper её часто не ставит.
            expected = {
                k: v.replace("ё", "е") if isinstance(v, str) else v for k, v in case.items()
            }
            command = rules.interpret(transcript.text, context)
            command.title = command.title.replace("ё", "е") if command.title else None
            command.new_title = None
            expected.pop("new_title", None)
            problems = mismatches(command, expected)
            correct += not problems
            exact += normalized(transcript.text) == normalized(case["text"])
            if problems and args.errors:
                print(f"  ✗ {case['text']} → {transcript.text}: {'; '.join(problems)}")
        total = len(CASES)
        print(
            f"| {voice} | {total} | {correct / total:.0%} | {exact / total:.0%} "
            f"| {spent / total:.2f} c |"
        )
    print(f"\nВремя замера: {time.strftime('%H:%M')}")


if __name__ == "__main__":
    main()
