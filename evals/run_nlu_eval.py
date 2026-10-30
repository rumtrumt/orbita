"""Сравнение интерпретаторов на золотом и сложном наборах.

    python evals/run_nlu_eval.py [--mode rules llm hybrid] [--errors]

Для llm и hybrid нужна запущенная модель из настроек.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

from tests.golden.checks import CASES, mismatches

from orbita.config import NluMode, Settings
from orbita.container import build_interpreter
from orbita.nlu.commands import NluContext

MSK = ZoneInfo("Europe/Moscow")
NOW = datetime(2026, 10, 30, 14, 0, tzinfo=MSK)
HARD = yaml.safe_load((Path(__file__).parent / "nlu_hard.yaml").read_text("utf-8"))
HOLDOUT = yaml.safe_load((Path(__file__).parent / "nlu_holdout.yaml").read_text("utf-8"))
HOLDOUT_2 = yaml.safe_load((Path(__file__).parent / "nlu_holdout_2.yaml").read_text("utf-8"))


def evaluate(interpreter, cases, show_errors: bool) -> tuple[int, int, float]:
    passed_action = passed_full = 0
    spent = 0.0
    for case in cases:
        started = time.perf_counter()
        command = interpreter.interpret(case["text"], NluContext(now=NOW, timezone=MSK))
        spent += time.perf_counter() - started
        problems = mismatches(command, case)
        passed_action += command.action.value == case["action"]
        passed_full += not problems
        if problems and show_errors:
            print(f"  ✗ {case['text']}: {'; '.join(problems)}")
    return passed_action, passed_full, spent / len(cases)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", nargs="+", default=["rules", "llm", "hybrid"])
    parser.add_argument("--errors", action="store_true")
    args = parser.parse_args()

    print("| Режим | Набор | Реплик | Намерение | Полностью | Среднее время |")
    print("|---|---|---:|---:|---:|---:|")
    for mode in args.mode:
        interpreter = build_interpreter(Settings(nlu_mode=NluMode(mode), _env_file=None))
        for name, cases in (
            ("золотой", CASES),
            ("сложный", HARD),
            ("настроечный", HOLDOUT),
            ("контрольный", HOLDOUT_2),
        ):
            action, full, seconds = evaluate(interpreter, cases, args.errors)
            total = len(cases)
            print(
                f"| {mode} | {name} | {total} | {action / total:.0%} | {full / total:.0%} "
                f"| {seconds * 1000:.0f} мс |"
            )


if __name__ == "__main__":
    main()
