"""HTTP-контракт: авторизация, изоляция пользователей, голос, календарь."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from orbita.api.app import create_app
from orbita.clock import FixedClock
from orbita.config import NluMode, Settings, TtsProvider
from orbita.container import Container
from orbita.domain.errors import SpeechNotRecognizedError
from orbita.events import EventBus
from orbita.nlu.rules import RuleInterpreter
from orbita.services.reminders import fire_due_reminders
from orbita.voice.stt import Transcript
from orbita.voice.tts import Speech
from tests.conftest import NOW


class FakeRecognizer:
    text: str | None = "Поставь встречу с Иваном завтра в 15:00"

    def transcribe(self, audio: bytes) -> Transcript:
        if self.text is None:
            raise SpeechNotRecognizedError("тишина")
        return Transcript(self.text, 2.0, 0.1)


class FakeSynthesizer:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def synthesize(self, text: str) -> Speech:
        self.spoken.append(text)
        return Speech(b"RIFF", "audio/wav")


@pytest.fixture
def container(database, tmp_path) -> Container:
    settings = Settings(
        data_dir=tmp_path,
        nlu_mode=NluMode.RULES,
        tts_provider=TtsProvider.OFF,
        semantic_search=False,
        llm_api_key=SecretStr("x"),
        _env_file=None,
    )
    return Container(
        settings=settings,
        clock=FixedClock(NOW),
        database=database,
        interpreter=RuleInterpreter(),
        embedder=None,
        reranker=None,
        recognizer=FakeRecognizer(),
        synthesizer=FakeSynthesizer(),
        bus=EventBus(),
    )


@pytest.fixture
def client(container):
    with TestClient(create_app(container)) as test_client:
        yield test_client


def login(client, name="Анна", pin="1234") -> dict[str, str]:
    response = client.post("/api/session", json={"name": name, "pin": pin})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['token']}"}


def test_requires_auth(client):
    assert client.post("/api/chat", json={"text": "привет"}).status_code == 401


def test_wrong_pin_rejected(client):
    login(client)
    response = client.post("/api/session", json={"name": "анна", "pin": "9999"})
    assert response.status_code == 403


def test_chat_and_agenda(client):
    headers = login(client)
    turn = client.post(
        "/api/chat", json={"text": "Добавь встречу с командой завтра в 15:00"}, headers=headers
    ).json()
    assert turn["action"] == "created_meeting"
    assert turn["items"][0]["end"] == "16:00"
    assert "пятнадцать" in turn["speech"]
    agenda = client.get(
        "/api/agenda", params={"start": "2026-10-31", "end": "2026-10-31"}, headers=headers
    ).json()
    assert [o["title"] for o in agenda["occurrences"]] == ["Встреча с командой"]


def test_users_are_isolated(client):
    anna, boris = login(client), login(client, "Борис")
    client.post("/api/chat", json={"text": "Завтра купить молоко"}, headers=anna)
    reply = client.post("/api/chat", json={"text": "Что у меня завтра?"}, headers=boris).json()
    assert "молоко" not in reply["reply"]
    assert client.get("/api/messages", headers=boris).json()[0]["text"] == "Что у меня завтра?"


def test_item_of_other_user_is_hidden(client):
    anna, boris = login(client), login(client, "Борис")
    turn = client.post("/api/chat", json={"text": "Завтра купить молоко"}, headers=anna).json()
    item_id = turn["items"][0]["id"]
    assert client.delete(f"/api/items/{item_id}", headers=boris).status_code == 404


def test_voice_pipeline_streams_stages(client):
    headers = login(client)
    response = client.post(
        "/api/voice", files={"audio": ("a.webm", b"x", "audio/webm")}, headers=headers
    )
    stages = [json.loads(line) for line in response.text.splitlines()]
    assert [s["stage"] for s in stages] == ["transcribing", "transcript", "thinking", "done"]
    assert stages[-1]["turn"]["action"] == "created_meeting"


def test_silence_is_reported(client, container):
    container.recognizer.text = None
    headers = login(client)
    response = client.post("/api/voice", files={"audio": ("a.webm", b"x")}, headers=headers)
    assert '"no_speech"' in response.text


def test_tts_uses_spoken_form(client, container):
    headers = login(client)
    assert client.post("/api/tts", json={"text": "в 15:00"}, headers=headers).status_code == 200
    assert container.synthesizer.spoken == ["в пятнадцать ноль ноль"]


def test_calendar_feed(client):
    headers = login(client)
    client.post("/api/chat", json={"text": "По понедельникам в 10 планерка"}, headers=headers)
    path = client.get("/api/me", headers=headers).json()["calendar"]
    body = client.get(path).text
    assert "BEGIN:VCALENDAR" in body
    assert "RRULE:FREQ=WEEKLY;BYDAY=MO" in body
    assert "SUMMARY:Планёрка" in body


def test_reminder_fires_once(client, container):
    headers = login(client)
    client.post("/api/chat", json={"text": "Напомни через час выпить воды"}, headers=headers)
    assert fire_due_reminders(container) == 0
    container.clock.moment = NOW + timedelta(hours=1, minutes=1)
    assert fire_due_reminders(container) == 1
    assert fire_due_reminders(container) == 0
    last = client.get("/api/messages", headers=headers).json()[-1]
    assert last["text"] == "Напоминаю: Выпить воды."


def test_recurring_reminder_rearms(client, container):
    headers = login(client)
    client.post(
        "/api/chat", json={"text": "Напоминай каждый день в 15:00 пить воду"}, headers=headers
    )
    container.clock.moment = NOW + timedelta(hours=2)
    assert fire_due_reminders(container) == 1
    container.clock.moment = NOW + timedelta(days=1, hours=2)
    assert fire_due_reminders(container) == 1


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["nlu_mode"] == "rules"
