"""Ошибки предметной области."""

from __future__ import annotations


class OrbitaError(Exception):
    """Базовая ошибка приложения."""


class ItemNotFoundError(OrbitaError):
    pass


class AuthError(OrbitaError):
    pass


class SpeechNotRecognizedError(OrbitaError):
    """В аудио нет разборчивой речи."""


class VoiceUnavailableError(OrbitaError):
    """Голосовой движок не загрузился."""
