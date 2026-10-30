"""Сверка команды с золотым примером. Общая для тестов правил и оценки LLM."""

from __future__ import annotations

from datetime import date, time
from pathlib import Path
from typing import Any

import yaml

from orbita.nlu.commands import Command

CASES: list[dict[str, Any]] = yaml.safe_load(
    (Path(__file__).parent / "nlu.yaml").read_text("utf-8")
)


def mismatches(command: Command, case: dict[str, Any]) -> list[str]:
    """Список расхождений; пустой означает полное совпадение."""
    problems: list[str] = []

    def expect(field: str, actual: Any, wanted: Any) -> None:
        if actual != wanted:
            problems.append(f"{field}: {actual!r} != {wanted!r}")

    expect("action", command.action.value, case["action"])
    if "title" in case:
        expect("title", command.title, case["title"])
    if "query" in case and case["query"] not in (command.query or ""):
        problems.append(f"query: {command.query!r} не содержит {case['query']!r}")
    if "kind" in case:
        expect("kind", command.kind.value if command.kind else None, case["kind"])
    if "day" in case:
        day = case["day"]
        expect("day", command.when.day, day if isinstance(day, date) else date.fromisoformat(day))
    if "at" in case:
        expect("at", command.when.at, time.fromisoformat(case["at"]))
    if "deadline" in case:
        expect("deadline", command.when.deadline, case["deadline"])
    if "recurrence" in case:
        expect("recurrence", command.when.recurrence, case["recurrence"])
    if "period" in case:
        period = command.when.period
        expect("period", [period.start, period.end] if period else None, case["period"])
    if "past" in case:
        expect("past", command.past, case["past"])
    if "only_overdue" in case:
        expect("only_overdue", command.only_overdue, case["only_overdue"])
    if "include_done" in case:
        expect("include_done", command.include_done, case["include_done"])
    if "shift" in case:
        expect("shift", command.when.shift_minutes, case["shift"])
    if "new_title" in case:
        expect("new_title", command.new_title, case["new_title"])
    reference = command.reference
    if "ordinal" in case:
        expect("ordinal", reference.ordinal if reference else None, case["ordinal"])
    if "ref" in case:
        if reference is None:
            problems.append("reference отсутствует")
        elif case["ref"] == "pronoun":
            expect("pronoun", reference.pronoun, True)
        else:
            expect("ref.query", reference.query, case["ref"])
    return problems
