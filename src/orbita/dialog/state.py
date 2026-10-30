"""Состояние диалога и его сериализация в JSON."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, time
from typing import Any

from orbita.domain.models import ItemKind
from orbita.services.items import ItemDraft


@dataclass(slots=True)
class DialogState:
    # Записи, о которых шла речь последней: на них указывают «её», «вторую», «эту задачу».
    focus: list[int] = field(default_factory=list)
    # Параметры последнего списка, чтобы понять «а на завтра?».
    last_list_kind: str | None = None
    has_last_list: bool = False
    pending: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> DialogState:
        return cls(**data) if data else cls()


def draft_to_json(draft: ItemDraft) -> dict[str, Any]:
    data = asdict(draft)
    data["kind"] = draft.kind.value
    data["day"] = draft.day.isoformat() if draft.day else None
    data["at"] = draft.at.isoformat() if draft.at else None
    return data


def draft_from_json(data: dict[str, Any]) -> ItemDraft:
    values = dict(data)
    values["kind"] = ItemKind(values["kind"])
    values["day"] = date.fromisoformat(values["day"]) if values["day"] else None
    values["at"] = time.fromisoformat(values["at"]) if values["at"] else None
    return ItemDraft(**values)
