"""Сквозные диалоги поверх настоящей БД. «Сейчас» — пятница, 30.10.2026, 14:00."""

from datetime import date, datetime, time, timedelta

from orbita.domain.models import ItemKind, ItemStatus
from tests.conftest import MSK, NOW


class TestCreate:
    def test_reminder_is_saved_with_all_fields(self, assistant):
        reply = assistant.say("Напомни мне завтра в 10 утра позвонить Ивану")
        item = assistant.items()[0]
        assert item.kind is ItemKind.REMINDER
        assert item.title == "Позвонить Ивану"
        assert (item.day, item.at) == (date(2026, 10, 31), time(10))
        assert item.created_at is not None
        assert item.status is ItemStatus.PENDING
        assert "31 октября" in reply.reply and "Позвонить Ивану" in reply.reply

    def test_task_with_date_only(self, assistant):
        assistant.say("В пятницу нужно отправить отчет")
        item = assistant.items()[0]
        assert (item.kind, item.day, item.at) == (ItemKind.TASK, date(2026, 11, 6), None)

    def test_deadline(self, assistant):
        assistant.say("Поставь задачу купить билеты до конца недели")
        item = assistant.items()[0]
        assert item.deadline and item.day == date(2026, 11, 1)

    def test_vague_period_becomes_deadline_for_task(self, assistant):
        assistant.say("На следующей неделе сходить к врачу")
        item = assistant.items()[0]
        assert item.deadline and item.day == date(2026, 11, 8)

    def test_meeting(self, assistant):
        assistant.say("Добавь встречу с командой завтра в 15:00")
        item = assistant.items()[0]
        assert item.kind is ItemKind.MEETING
        assert item.duration_minutes == 60
        assert item.remind_at == datetime(2026, 10, 31, 14, 45, tzinfo=MSK)

    def test_persists_between_turns(self, assistant):
        assistant.say("Добавь встречу с командой завтра в 15:00")
        reply = assistant.say("Что у меня завтра?")
        assert "Встреча с командой" in reply.reply


class TestClarification:
    def test_meeting_without_time_asks_and_continues(self, assistant):
        reply = assistant.say("Поставь встречу с Иваном завтра")
        assert reply.reply == "На какое время поставить встречу?"
        assert reply.awaiting
        assert assistant.items() == []
        done = assistant.say("в 15")
        item = assistant.items()[0]
        assert (item.title, item.day, item.at) == (
            "Встреча с Иваном",
            date(2026, 10, 31),
            time(15),
        )
        assert not done.awaiting

    def test_time_answer_keeps_named_day(self, assistant):
        assistant.say("Поставь встречу с Иваном в понедельник")
        assistant.say("в 8 вечера")
        item = assistant.items()[0]
        assert (item.day, item.at) == (date(2026, 11, 2), time(20))

    def test_meeting_without_day(self, assistant):
        reply = assistant.say("Поставь встречу с Иваном")
        assert "день" in reply.reply
        assert assistant.say("завтра").reply == "На какое время поставить встречу?"
        assistant.say("в 12")
        assert assistant.items()[0].at == time(12)

    def test_reminder_without_time(self, assistant):
        reply = assistant.say("Напомни завтра позвонить маме")
        assert reply.reply == "Во сколько напомнить?"
        assistant.say("не важно")
        assert assistant.items()[0].at == time(9)

    def test_empty_reminder_asks_subject(self, assistant):
        assert assistant.say("Напомни завтра в 10").reply == "О чём напомнить?"
        assistant.say("позвонить врачу")
        assert assistant.titles() == ["Позвонить врачу"]

    def test_cancel_clarification(self, assistant):
        assistant.say("Поставь встречу с Иваном завтра")
        assert "отменила" in assistant.say("отмена").reply
        assert assistant.items() == []

    def test_new_command_replaces_question(self, assistant):
        assistant.say("Поставь встречу с Иваном завтра")
        reply = assistant.say("Что у меня сегодня?")
        assert reply.action == "list"
        assert assistant.items() == []

    def test_unclear_answer_is_asked_again_then_dropped(self, assistant):
        assistant.say("Поставь встречу с Иваном завтра")
        assert "На какое время" in assistant.say("хм").reply
        assert assistant.say("хм").action == "unknown"

    def test_past_time_is_questioned(self, assistant):
        reply = assistant.say("Поставь встречу сегодня в 10 утра")
        assert "прошло" in reply.reply
        assistant.say("да")
        assert assistant.items()[0].day == date(2026, 10, 31)

    def test_new_command_is_not_taken_for_answer_to_past_question(self, assistant):
        assistant.say("Поставь встречу сегодня в 10 утра")
        reply = assistant.say("Завтра купить билеты")
        assert reply.action == "created_task"
        assert assistant.titles() == ["Купить билеты"]

    def test_new_command_is_not_taken_for_answer_to_conflict(self, assistant):
        assistant.say("Встреча с командой завтра в 15:00")
        assistant.say("Созвон с заказчиком завтра в 15:30")
        reply = assistant.say("Завтра купить билеты")
        assert reply.action == "created_task"
        assert assistant.titles() == ["Встреча с командой", "Купить билеты"]

    def test_past_date_can_be_replaced(self, assistant):
        assistant.say("Напомни сегодня в 9 утра выпить воды")
        assistant.say("в 18:00")
        assert (assistant.items()[0].day, assistant.items()[0].at) == (
            date(2026, 10, 30),
            time(18),
        )


class TestList:
    def seed(self, assistant):
        assistant.say("Добавь встречу с командой сегодня в 15:00")
        assistant.say("Напомни сегодня в 18 полить цветы")
        assistant.say("Завтра купить молоко")
        assistant.say("Встреча с дизайнером в четверг в 11")

    def test_same_intent_different_wording(self, assistant):
        self.seed(assistant)
        for phrase in ("Что у меня сегодня?", "Какие планы на сегодня?"):
            reply = assistant.say(phrase)
            assert reply.action == "list"
            assert "Встреча с командой" in reply.reply and "Полить цветы" in reply.reply

    def test_filter_meetings(self, assistant):
        self.seed(assistant)
        reply = assistant.say("Покажи мои встречи на сегодня")
        assert "Встреча с командой" in reply.reply
        assert "цветы" not in reply.reply

    def test_tasks_tomorrow(self, assistant):
        self.seed(assistant)
        assert "Купить молоко" in assistant.say("Какие задачи у меня на завтра?").reply

    def test_week(self, assistant):
        self.seed(assistant)
        reply = assistant.say("Что запланировано на следующую неделю?")
        assert "Встреча с дизайнером" in reply.reply

    def test_empty_period(self, assistant):
        assert "планов нет" in assistant.say("Что у меня послезавтра?").reply

    def test_no_meetings_answer(self, assistant):
        assistant.say("Завтра купить молоко")
        assert "встреч нет" in assistant.say("Есть ли у меня встречи завтра?").reply

    def test_elliptical_follow_up(self, assistant):
        self.seed(assistant)
        assistant.say("Что у меня сегодня?")
        assert "Купить молоко" in assistant.say("а на завтра?").reply

    def test_overdue_shown_today(self, assistant, clock):
        assistant.say("Завтра купить молоко")
        clock.moment = NOW + timedelta(days=3)
        reply = assistant.say("Что у меня сегодня?")
        assert "просрочено" in reply.reply and "Купить молоко" in reply.reply

    def test_recurring_appears_on_each_day(self, assistant):
        assistant.say("Каждый день в 8 утра делать зарядку")
        assert "Делать зарядку" in assistant.say("Что у меня послезавтра?").reply
        assert "7 записей" in assistant.say("Что на следующей неделе?").reply


class TestSearch:
    def test_morphology(self, assistant):
        assistant.say("Завтра сделать презентацию для клиента")
        reply = assistant.say("Найди задачу про презентацию")
        assert "Сделать презентацию для клиента" in reply.reply

    def test_when_was_meeting(self, assistant, clock):
        assistant.say("Встреча с Иваном во вторник в 12")
        clock.moment = NOW + timedelta(days=7)
        reply = assistant.say("Когда у меня была встреча с Иваном?")
        assert reply.reply == "Встреча с Иваном была во вторник, 3 ноября, в 12:00."

    def test_contract(self, assistant):
        assistant.say("Завтра подписать договор с подрядчиком")
        reply = assistant.say("Я когда-то ставил задачу по договору, найди ее")
        assert "Подписать договор" in reply.reply

    def test_typo_from_recognition(self, assistant):
        assistant.say("Завтра отправить презентацию")
        assert "Отправить презентацию" in assistant.say("Найди задачу про призентацию").reply

    def test_nothing_found(self, assistant):
        assistant.say("Завтра купить молоко")
        assert "Не нашла" in assistant.say("Найди задачу про слона").reply

    def test_found_item_becomes_context(self, assistant):
        assistant.say("Завтра подписать договор")
        assistant.say("Найди задачу про договор")
        assistant.say("Удали её")
        assert assistant.items() == []


class TestContext:
    def test_move_last_created(self, assistant):
        assistant.say("Встреча с Иваном в понедельник в 10")
        reply = assistant.say("А перенеси ее на завтра")
        assert assistant.items()[0].day == date(2026, 10, 31)
        assert assistant.items()[0].at == time(10)
        assert "Перенесла" in reply.reply

    def test_move_time_keeps_day(self, assistant):
        assistant.say("Встреча с Иваном в понедельник в 10")
        assistant.say("перенеси её на 16:00")
        item = assistant.items()[0]
        assert (item.day, item.at) == (date(2026, 11, 2), time(16))

    def test_shift(self, assistant):
        assistant.say("Встреча с Иваном в понедельник в 10")
        assistant.say("сдвинь её на час позже")
        assert assistant.items()[0].at == time(11)

    def test_correction(self, assistant):
        assistant.say("Встреча с Иваном в понедельник в 10")
        assistant.say("нет, лучше в 12")
        assert assistant.items()[0].at == time(12)

    def test_move_to_next_week_keeps_weekday(self, assistant):
        assistant.say("Встреча с Иваном во вторник в 10")
        assistant.say("перенеси её на следующую неделю")
        assert assistant.items()[0].day == date(2026, 11, 3)

    def test_ordinal_from_list(self, assistant):
        assistant.say("Завтра купить молоко")
        assistant.say("Завтра купить хлеб")
        assistant.say("Что у меня завтра?")
        assistant.say("Удали вторую")
        assert assistant.titles() == ["Купить молоко"]

    def test_move_without_target_day_asks(self, assistant):
        assistant.say("Встреча с Иваном в понедельник в 10")
        assert assistant.say("перенеси её").reply == "На когда перенести?"
        assistant.say("на среду")
        assert assistant.items()[0].day == date(2026, 11, 4)

    def test_rename(self, assistant):
        assistant.say("Завтра сделать отчёт")
        assistant.say("Переименуй задачу про отчёт в Годовой отчёт")
        assert assistant.titles() == ["Годовой отчёт"]


class TestAmbiguity:
    def test_choice_between_matches(self, assistant):
        assistant.say("Завтра написать отчёт")
        assistant.say("В понедельник проверить отчёт")
        reply = assistant.say("Удали задачу про отчёт")
        assert reply.awaiting and "1)" in reply.reply
        assistant.say("вторую")
        assert len(assistant.items()) == 1

    def test_choice_by_words(self, assistant):
        assistant.say("Завтра написать отчёт")
        assistant.say("В понедельник проверить отчёт")
        assistant.say("Удали задачу про отчёт")
        assistant.say("ту где написать")
        assert assistant.titles() == ["Проверить отчёт"]

    def test_no_context_for_pronoun(self, assistant):
        reply = assistant.say("Удали её")
        assert "какую запись" in reply.reply


class TestChanges:
    def test_complete(self, assistant):
        assistant.say("Завтра отправить отчёт")
        reply = assistant.say("Отметь задачу про отчёт выполненной")
        item = assistant.items()[0]
        assert item.status is ItemStatus.DONE and item.completed_at
        assert "Отметила" in reply.reply

    def test_completed_task_leaves_agenda(self, assistant):
        assistant.say("Завтра отправить отчёт")
        assistant.say("Готово")
        assert "планов нет" in assistant.say("Что у меня завтра?").reply

    def test_recurring_task_moves_forward(self, assistant):
        assistant.say("Каждый понедельник отправлять отчёт")
        assistant.say("Готово")
        item = assistant.items()[0]
        assert item.status is ItemStatus.PENDING and item.day == date(2026, 11, 9)

    def test_delete_and_undo(self, assistant):
        assistant.say("Завтра отправить отчёт")
        assistant.say("Удали задачу про отчёт")
        assert assistant.items() == []
        assert "Вернула" in assistant.say("Отмени").reply
        assert assistant.titles() == ["Отправить отчёт"]

    def test_undo_move(self, assistant):
        assistant.say("Встреча с Иваном в понедельник в 10")
        assistant.say("перенеси её на среду")
        assistant.say("Отмени")
        assert assistant.items()[0].day == date(2026, 11, 2)

    def test_undo_create(self, assistant):
        assistant.say("Завтра купить молоко")
        assistant.say("Отмени")
        assert assistant.items() == []

    def test_undo_twice_walks_history(self, assistant):
        assistant.say("Встреча с Иваном в понедельник в 10")
        assistant.say("перенеси её на среду")
        assistant.say("перенеси её на четверг")
        assistant.say("Отмени")
        assistant.say("Отмени")
        assert assistant.items()[0].day == date(2026, 11, 2)

    def test_bulk_delete_requires_confirmation(self, assistant):
        assistant.say("Завтра купить молоко")
        assistant.say("Завтра купить хлеб")
        reply = assistant.say("Удали все задачи на завтра")
        assert reply.awaiting and len(assistant.items()) == 2
        assistant.say("да")
        assert assistant.items() == []


class TestConflicts:
    def seed(self, assistant):
        assistant.say("Встреча с командой завтра в 15:00")

    def test_conflict_detected_with_suggestion(self, assistant):
        self.seed(assistant)
        reply = assistant.say("Созвон с заказчиком завтра в 15:30")
        assert reply.action == "conflict"
        assert "Встреча с командой" in reply.reply and "16:00" in reply.reply
        assert len(assistant.items()) == 1

    def test_accept_suggestion(self, assistant):
        self.seed(assistant)
        assistant.say("Созвон с заказчиком завтра в 15:30")
        assistant.say("да")
        created = next(i for i in assistant.items() if "Созвон" in i.title)
        assert created.at == time(16)

    def test_keep_overlap(self, assistant):
        self.seed(assistant)
        assistant.say("Созвон с заказчиком завтра в 15:30")
        assistant.say("оставь как есть")
        assert next(i for i in assistant.items() if "Созвон" in i.title).at == time(15, 30)

    def test_answer_with_other_time(self, assistant):
        self.seed(assistant)
        assistant.say("Созвон с заказчиком завтра в 15:30")
        assistant.say("давай в 18")
        assert next(i for i in assistant.items() if "Созвон" in i.title).at == time(18)

    def test_adjacent_meetings_do_not_conflict(self, assistant):
        self.seed(assistant)
        assert assistant.say("Созвон с заказчиком завтра в 16:00").action == "created_meeting"

    def test_conflict_when_moving(self, assistant):
        self.seed(assistant)
        assistant.say("Созвон с заказчиком завтра в 17:00")
        reply = assistant.say("перенеси её на 15:00")
        assert reply.action == "conflict"
        assistant.say("да")
        moved = next(i for i in assistant.items() if "Созвон" in i.title)
        assert moved.at == time(16)

    def test_moving_does_not_conflict_with_itself(self, assistant):
        self.seed(assistant)
        assert assistant.say("сдвинь её на 30 минут позже").action == "updated"

    def test_recurring_meeting_conflicts(self, assistant):
        assistant.say("По понедельникам в 10 планерка")
        assert assistant.say("Встреча с Иваном в понедельник в 10").action == "conflict"
