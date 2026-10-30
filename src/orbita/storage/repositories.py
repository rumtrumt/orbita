"""Репозитории: единственное место, где встречаются ORM и доменные сущности."""

from __future__ import annotations

import hashlib
from datetime import date, datetime, time
from typing import Any

from sqlalchemy import or_, select, update
from sqlalchemy.orm import Session

from orbita.domain.errors import ItemNotFoundError
from orbita.domain.models import Item, ItemKind, ItemStatus, User
from orbita.storage.orm import DialogRow, EmbeddingRow, ItemRow, MessageRow, RevisionRow, UserRow

_ITEM_FIELDS = (
    "user_id", "title", "day", "at", "duration_minutes", "deadline", "recurrence", "remind_at",
    "remind_before_minutes", "notes", "source_text", "created_at", "updated_at", "completed_at",
    "deleted_at",
)  # fmt: skip


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _to_item(row: ItemRow) -> Item:
    values = {name: getattr(row, name) for name in _ITEM_FIELDS}
    return Item(id=row.id, kind=ItemKind(row.kind), status=ItemStatus(row.status), **values)


def _apply(row: ItemRow, item: Item) -> None:
    for name in _ITEM_FIELDS:
        setattr(row, name, getattr(item, name))
    row.kind = item.kind.value
    row.status = item.status.value


def item_snapshot(item: Item) -> dict[str, Any]:
    def encode(value: Any) -> Any:
        return value.isoformat() if isinstance(value, date | time | datetime) else value

    data = {name: encode(getattr(item, name)) for name in _ITEM_FIELDS}
    return data | {"kind": item.kind.value, "status": item.status.value}


def _restore_snapshot(row: ItemRow, snapshot: dict[str, Any]) -> None:
    converters = {
        "day": date.fromisoformat,
        "at": time.fromisoformat,
        "remind_at": datetime.fromisoformat,
        "created_at": datetime.fromisoformat,
        "updated_at": datetime.fromisoformat,
        "completed_at": datetime.fromisoformat,
        "deleted_at": datetime.fromisoformat,
    }
    for name, value in snapshot.items():
        convert = converters.get(name)
        setattr(row, name, convert(value) if convert and value is not None else value)


class ItemRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, item: Item) -> Item:
        row = ItemRow()
        _apply(row, item)
        self._session.add(row)
        self._session.flush()
        item.id = row.id
        return item

    def save(self, item: Item) -> Item:
        row = self._session.get(ItemRow, item.id)
        if row is None:
            raise ItemNotFoundError(str(item.id))
        _apply(row, item)
        self._session.flush()
        return item

    def get(self, user_id: int, item_id: int, *, with_deleted: bool = False) -> Item:
        row = self._session.get(ItemRow, item_id)
        if row is None or row.user_id != user_id or (row.deleted_at and not with_deleted):
            raise ItemNotFoundError(str(item_id))
        return _to_item(row)

    def get_many(self, user_id: int, item_ids: list[int]) -> list[Item]:
        rows = self._session.scalars(
            select(ItemRow).where(
                ItemRow.user_id == user_id, ItemRow.id.in_(item_ids), ItemRow.deleted_at.is_(None)
            )
        )
        by_id = {row.id: _to_item(row) for row in rows}
        return [by_id[i] for i in item_ids if i in by_id]

    def live(self, user_id: int) -> list[Item]:
        """Все неудалённые записи пользователя."""
        rows = self._session.scalars(
            select(ItemRow).where(ItemRow.user_id == user_id, ItemRow.deleted_at.is_(None))
        )
        return [_to_item(row) for row in rows]

    def in_window(self, user_id: int, start: date, end: date) -> list[Item]:
        """Записи, которые могут попасть в окно: разовые по дате и все серии, начатые до конца."""
        rows = self._session.scalars(
            select(ItemRow).where(
                ItemRow.user_id == user_id,
                ItemRow.deleted_at.is_(None),
                ItemRow.day.is_not(None),
                or_(
                    ItemRow.day.between(start, end),
                    (ItemRow.recurrence.is_not(None)) & (ItemRow.day <= end),
                ),
            )
        )
        return [_to_item(row) for row in rows]

    def overdue(self, user_id: int, today: date) -> list[Item]:
        rows = self._session.scalars(
            select(ItemRow).where(
                ItemRow.user_id == user_id,
                ItemRow.deleted_at.is_(None),
                ItemRow.status == ItemStatus.PENDING.value,
                ItemRow.kind != ItemKind.MEETING.value,
                ItemRow.day < today,
                ItemRow.recurrence.is_(None),
            )
        )
        return [_to_item(row) for row in rows]

    def undated(self, user_id: int) -> list[Item]:
        rows = self._session.scalars(
            select(ItemRow).where(
                ItemRow.user_id == user_id,
                ItemRow.deleted_at.is_(None),
                ItemRow.status == ItemStatus.PENDING.value,
                ItemRow.day.is_(None),
            )
        )
        return [_to_item(row) for row in rows]

    def due_reminders(self, moment: datetime) -> list[Item]:
        rows = self._session.scalars(
            select(ItemRow).where(
                ItemRow.remind_at.is_not(None),
                ItemRow.remind_at <= moment,
                ItemRow.deleted_at.is_(None),
                ItemRow.status == ItemStatus.PENDING.value,
            )
        )
        return [_to_item(row) for row in rows]

    def record_revision(
        self, item: Item, action: str, before: dict[str, Any] | None, moment: datetime
    ) -> None:
        assert item.id is not None
        self._session.add(
            RevisionRow(
                user_id=item.user_id,
                item_id=item.id,
                action=action,
                before=before,
                created_at=moment,
            )
        )
        self._session.flush()

    def undo_last(self, user_id: int, moment: datetime) -> tuple[Item, str] | None:
        """Откатить последнее изменение. Возвращает запись в новом состоянии и тип действия."""
        revision = self._session.scalar(
            select(RevisionRow)
            .where(RevisionRow.user_id == user_id, RevisionRow.undone.is_(False))
            .order_by(RevisionRow.id.desc())
            .limit(1)
        )
        if revision is None:
            return None
        row = self._session.get(ItemRow, revision.item_id)
        if row is None:
            return None
        if revision.before is None:
            row.deleted_at = moment  # отмена создания
        else:
            _restore_snapshot(row, revision.before)
        revision.undone = True
        self._session.flush()
        return _to_item(row), revision.action


class EmbeddingRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert(self, item_id: int, model: str, text: str, vector: bytes) -> None:
        row = self._session.get(EmbeddingRow, item_id)
        if row is None:
            self._session.add(EmbeddingRow(item_id=item_id, model=model, text=text, vector=vector))
        else:
            row.model, row.text, row.vector = model, text, vector
        self._session.flush()

    def for_user(self, user_id: int, model: str) -> dict[int, tuple[str, bytes]]:
        rows = self._session.execute(
            select(EmbeddingRow.item_id, EmbeddingRow.text, EmbeddingRow.vector)
            .join(ItemRow, ItemRow.id == EmbeddingRow.item_id)
            .where(ItemRow.user_id == user_id, EmbeddingRow.model == model)
        )
        return {item_id: (text, vector) for item_id, text, vector in rows}


class UserRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    @staticmethod
    def _to_user(row: UserRow) -> User:
        return User(
            id=row.id,
            name=row.name,
            timezone=row.timezone,
            calendar_token=row.calendar_token,
            created_at=row.created_at,
        )

    def by_token(self, token: str) -> User | None:
        row = self._session.scalar(select(UserRow).where(UserRow.token_hash == hash_token(token)))
        return self._to_user(row) if row else None

    def by_name(self, name: str) -> User | None:
        row = self._session.scalar(select(UserRow).where(UserRow.name_key == name.casefold()))
        return self._to_user(row) if row else None

    def by_calendar_token(self, token: str) -> User | None:
        row = self._session.scalar(select(UserRow).where(UserRow.calendar_token == token))
        return self._to_user(row) if row else None

    def rotate_token(self, user_id: int, token: str) -> None:
        self._session.execute(
            update(UserRow).where(UserRow.id == user_id).values(token_hash=hash_token(token))
        )

    def pin_hash(self, user_id: int) -> str:
        row = self._session.get(UserRow, user_id)
        return row.pin_hash if row else ""

    def add(self, user: User, token: str, moment: datetime, pin_hash: str = "") -> User:
        row = UserRow(
            name=user.name,
            name_key=user.name.casefold(),
            token_hash=hash_token(token),
            pin_hash=pin_hash,
            timezone=user.timezone,
            calendar_token=user.calendar_token,
            created_at=moment,
        )
        self._session.add(row)
        self._session.flush()
        return self._to_user(row)

    def get(self, user_id: int) -> User | None:
        row = self._session.get(UserRow, user_id)
        return self._to_user(row) if row else None


class DialogRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def load(self, user_id: int) -> dict[str, Any] | None:
        row = self._session.get(DialogRow, user_id)
        return row.state if row else None

    def save(self, user_id: int, state: dict[str, Any], moment: datetime) -> None:
        row = self._session.get(DialogRow, user_id)
        if row is None:
            self._session.add(DialogRow(user_id=user_id, state=state, updated_at=moment))
        else:
            row.state, row.updated_at = state, moment
        self._session.flush()


class MessageRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(
        self, user_id: int, role: str, text: str, payload: dict[str, Any] | None, moment: datetime
    ) -> None:
        self._session.add(
            MessageRow(user_id=user_id, role=role, text=text, payload=payload, created_at=moment)
        )
        self._session.flush()

    def recent(self, user_id: int, limit: int = 60) -> list[MessageRow]:
        rows = self._session.scalars(
            select(MessageRow)
            .where(MessageRow.user_id == user_id)
            .order_by(MessageRow.id.desc())
            .limit(limit)
        )
        return list(reversed(list(rows)))
