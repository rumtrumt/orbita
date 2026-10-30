"""Золотой набор реплик для интерпретатора на правилах."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from orbita.nlu.commands import NluContext
from orbita.nlu.rules import RuleInterpreter
from tests.golden.checks import CASES, mismatches

MSK = ZoneInfo("Europe/Moscow")
NOW = datetime(2026, 10, 30, 14, 0, tzinfo=MSK)


@pytest.mark.parametrize("case", CASES, ids=[c["text"][:40] for c in CASES])
def test_golden(case):
    command = RuleInterpreter().interpret(case["text"], NluContext(now=NOW, timezone=MSK))
    assert mismatches(command, case) == []
