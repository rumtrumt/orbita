"""HTTP-слой. Только транспорт: разбор запроса, авторизация, сериализация."""

import asyncio
import hashlib
import hmac
import json
import secrets
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated, Any

import httpx
import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from orbita import __version__
from orbita.container import Container
from orbita.domain.errors import (
    ItemNotFoundError,
    SpeechNotRecognizedError,
    VoiceUnavailableError,
)
from orbita.domain.models import User
from orbita.services.assistant import Assistant
from orbita.services.reminders import reminder_loop
from orbita.storage.repositories import MessageRepository, UserRepository
from orbita.voice.speech_text import to_speech

log = structlog.get_logger(__name__)

_STATIC = Path(__file__).parents[1] / "web" / "static"
_SCRYPT = {"n": 2**14, "r": 8, "p": 1}
_HEARTBEAT_SECONDS = 20.0


class SessionRequest(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    pin: str = Field(min_length=4, max_length=32)


class ChatRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


def _hash_pin(pin: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(pin.encode(), salt=salt, **_SCRYPT)
    return f"{salt.hex()}${digest.hex()}"


def _pin_matches(pin: str, stored: str) -> bool:
    salt, _, _ = stored.partition("$")
    return hmac.compare_digest(_hash_pin(pin, bytes.fromhex(salt)), stored)


def create_app(container: Container) -> FastAPI:
    assistant = Assistant(container)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        container.bus.bind(asyncio.get_running_loop())
        task = asyncio.create_task(reminder_loop(container))
        # Прогрев в фоне: первая голосовая команда не должна ждать загрузку моделей.
        warmers = [
            warm
            for engine in (container.recognizer, container.synthesizer, container.embedder)
            if (warm := getattr(engine, "warm_up", None))
        ]
        warm_tasks = [asyncio.create_task(asyncio.to_thread(_safely, warm)) for warm in warmers]
        yield
        task.cancel()
        for warm_task in warm_tasks:
            warm_task.cancel()

    app = FastAPI(title="Орбита", version=__version__, lifespan=lifespan)

    def current_user(authorization: Annotated[str | None, Header()] = None) -> User:
        token = (authorization or "").removeprefix("Bearer ").strip()
        return _user_by_token(token)

    def _user_by_token(token: str) -> User:
        with container.database.session() as session:
            user = UserRepository(session).by_token(token) if token else None
        if user is None:
            raise HTTPException(401, "Нужно войти")
        return user

    Current = Annotated[User, Depends(current_user)]

    @app.post("/api/session")
    def open_session(body: SessionRequest) -> dict[str, Any]:
        name = body.name.strip()
        token = secrets.token_urlsafe(32)
        with container.database.session() as session:
            users = UserRepository(session)
            user = users.by_name(name)
            if user is None:
                user = users.add(
                    User(
                        name=name,
                        timezone=container.settings.default_timezone,
                        calendar_token=secrets.token_urlsafe(24),
                    ),
                    token,
                    container.clock.now(container.settings.timezone),
                    _hash_pin(body.pin),
                )
            else:
                assert user.id is not None
                if not _pin_matches(body.pin, users.pin_hash(user.id)):
                    raise HTTPException(403, "Неверный PIN")
                users.rotate_token(user.id, token)
        return {"token": token, "user": _user_payload(user)}

    @app.get("/api/me")
    def me(user: Current) -> dict[str, Any]:
        return _user_payload(user)

    @app.post("/api/chat")
    def chat(body: ChatRequest, user: Current) -> dict[str, Any]:
        return assistant.turn(user, body.text.strip())

    @app.post("/api/voice")
    async def voice(audio: UploadFile, user: Current) -> StreamingResponse:
        data = await audio.read()
        if len(data) > container.settings.max_audio_bytes:
            raise HTTPException(413, "Слишком длинная запись")
        return StreamingResponse(_voice_stages(data, user), media_type="application/x-ndjson")

    def _voice_stages(data: bytes, user: User) -> Iterator[str]:
        """Поэтапный ответ: интерфейс показывает, что происходит, а не один долгий спиннер."""

        def line(**fields: Any) -> str:
            return json.dumps(fields, ensure_ascii=False) + "\n"

        yield line(stage="transcribing")
        try:
            transcript = container.recognizer.transcribe(data)
        except SpeechNotRecognizedError:
            yield line(stage="no_speech", message="Не расслышала. Повторите, пожалуйста.")
            return
        except VoiceUnavailableError as error:
            log.error("voice.unavailable", error=str(error))
            yield line(stage="error", message="Распознавание сейчас недоступно, напишите текстом.")
            return
        yield line(
            stage="transcript",
            text=transcript.text,
            seconds=round(transcript.seconds_spent, 2),
        )
        yield line(stage="thinking")
        yield line(stage="done", turn=assistant.turn(user, transcript.text, via="voice"))

    @app.post("/api/tts")
    def tts(body: SpeechRequest, user: Current) -> Response:
        if container.synthesizer is None:
            return Response(status_code=204)
        try:
            speech = container.synthesizer.synthesize(to_speech(body.text))
        except VoiceUnavailableError as error:
            log.warning("tts.unavailable", error=str(error))
            # 204 — знак интерфейсу озвучить ответ голосом браузера.
            return Response(status_code=204)
        return Response(speech.audio, media_type=speech.media_type)

    @app.get("/api/agenda")
    def agenda(user: Current, start: date | None = None, end: date | None = None) -> dict[str, Any]:
        first = start or container.clock.now(container.settings.timezone).date()
        return assistant.agenda(user, first, end or first + timedelta(days=6))

    @app.post("/api/items/{item_id}/complete")
    def complete_item(item_id: int, user: Current) -> dict[str, bool]:
        with assistant.scope(user) as scope:
            assert user.id is not None
            scope.items.complete(_item(scope, user.id, item_id))
        return {"ok": True}

    @app.delete("/api/items/{item_id}")
    def delete_item(item_id: int, user: Current) -> dict[str, bool]:
        with assistant.scope(user) as scope:
            assert user.id is not None
            scope.items.delete(_item(scope, user.id, item_id))
        return {"ok": True}

    def _item(scope: Any, user_id: int, item_id: int) -> Any:
        try:
            return scope.items.get(user_id, item_id)
        except ItemNotFoundError as error:
            raise HTTPException(404, "Запись не найдена") from error

    @app.get("/api/messages")
    def messages(user: Current) -> list[dict[str, Any]]:
        assert user.id is not None
        with container.database.session() as session:
            rows = MessageRepository(session).recent(user.id)
            return [{"role": row.role, "text": row.text, "payload": row.payload} for row in rows]

    @app.get("/api/events")
    async def events(request: Request, token: str = Query()) -> StreamingResponse:
        user = await asyncio.to_thread(_user_by_token, token)
        assert user.id is not None
        return StreamingResponse(
            _event_stream(request, user.id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def _event_stream(request: Request, user_id: int) -> AsyncIterator[str]:
        queue = container.bus.subscribe(user_id)
        try:
            yield ": connected\n\n"
            while not await request.is_disconnected():
                try:
                    event = await asyncio.wait_for(queue.get(), _HEARTBEAT_SECONDS)
                except TimeoutError:
                    yield ": ping\n\n"
                    continue
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            container.bus.unsubscribe(user_id, queue)

    @app.get("/api/calendar/{token}.ics")
    def calendar(token: str) -> PlainTextResponse:
        with container.database.session() as session:
            user = UserRepository(session).by_calendar_token(token)
        if user is None:
            raise HTTPException(404, "Календарь не найден")
        return PlainTextResponse(assistant.calendar(user), media_type="text/calendar")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        settings = container.settings
        return {
            "status": "ok",
            "version": __version__,
            "nlu_mode": settings.nlu_mode.value,
            "llm": _llm_alive(settings.llm_base_url)
            if settings.nlu_mode.value != "rules"
            else None,
            "stt_device": getattr(container.recognizer, "active_device", None),
            "tts": settings.tts_provider.value,
            "semantic_search": container.embedder is not None,
        }

    app.mount("/", StaticFiles(directory=_STATIC, html=True), name="web")
    return app


def _safely(warm: Any) -> None:
    try:
        warm()
    except Exception as error:
        log.warning("warmup.failed", error=str(error))


def _llm_alive(base_url: str) -> bool:
    try:
        with httpx.Client(trust_env=False, timeout=1.5) as client:
            return client.get(f"{base_url.rstrip('/')}/models").status_code < 500
    except httpx.HTTPError:
        return False


def _user_payload(user: User) -> dict[str, Any]:
    return {
        "name": user.name,
        "timezone": user.timezone,
        "calendar": f"/api/calendar/{user.calendar_token}.ics",
    }
