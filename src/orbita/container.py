"""Сборка зависимостей приложения. Единственное место, знающее о конкретных реализациях."""

from __future__ import annotations

from dataclasses import dataclass

from orbita.clock import Clock, SystemClock
from orbita.config import NluMode, Settings, TtsProvider
from orbita.events import EventBus
from orbita.llm.client import LlmClient
from orbita.llm.rerank import LlmReranker
from orbita.nlu.commands import Interpreter
from orbita.nlu.hybrid import HybridInterpreter, LlmFirstInterpreter
from orbita.nlu.llm import LlmInterpreter
from orbita.nlu.rules import RuleInterpreter
from orbita.services.embeddings import Embedder, FastEmbedder
from orbita.services.search import Reranker
from orbita.storage.db import Database, make_engine, upgrade_database
from orbita.voice.stt import SpeechRecognizer, WhisperRecognizer
from orbita.voice.tts import EdgeSynthesizer, PiperSynthesizer, Synthesizer


@dataclass(slots=True)
class Container:
    settings: Settings
    clock: Clock
    database: Database
    interpreter: Interpreter
    embedder: Embedder | None
    reranker: Reranker | None
    recognizer: SpeechRecognizer
    synthesizer: Synthesizer | None
    bus: EventBus


def build_llm_client(settings: Settings) -> LlmClient:
    return LlmClient(
        settings.llm_base_url,
        settings.llm_model,
        settings.llm_api_key.get_secret_value(),
        settings.llm_timeout_seconds,
    )


def build_interpreter(settings: Settings) -> Interpreter:
    rules = RuleInterpreter()
    if settings.nlu_mode is NluMode.RULES:
        return rules
    llm = LlmInterpreter(build_llm_client(settings))
    if settings.nlu_mode is NluMode.LLM:
        return LlmFirstInterpreter(rules, llm)
    return HybridInterpreter(rules, llm)


def build_synthesizer(settings: Settings) -> Synthesizer | None:
    if settings.tts_provider is TtsProvider.PIPER:
        return PiperSynthesizer(settings.piper_voice, settings.models_dir)
    if settings.tts_provider is TtsProvider.EDGE:
        return EdgeSynthesizer(settings.edge_voice)
    return None


def build_container(settings: Settings) -> Container:
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    url = settings.resolved_database_url
    upgrade_database(url)
    embedder = (
        FastEmbedder(settings.embedding_model, settings.models_dir / "embeddings")
        if settings.semantic_search
        else None
    )
    return Container(
        settings=settings,
        clock=SystemClock(),
        database=Database(make_engine(url)),
        interpreter=build_interpreter(settings),
        embedder=embedder,
        reranker=(
            LlmReranker(build_llm_client(settings))
            if settings.nlu_mode is not NluMode.RULES
            else None
        ),
        recognizer=WhisperRecognizer(
            settings.whisper_model,
            settings.whisper_device,
            settings.whisper_compute_type,
            settings.whisper_beam_size,
        ),
        synthesizer=build_synthesizer(settings),
        bus=EventBus(),
    )
