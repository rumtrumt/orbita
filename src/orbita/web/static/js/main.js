import * as api from "./api.js";
import { Dial } from "./dial.js";
import { Orb } from "./orb.js";
import { Player, Recorder } from "./voice.js";

const $ = (id) => document.getElementById(id);
const WEEKDAYS = ["воскресенье", "понедельник", "вторник", "среда", "четверг", "пятница", "суббота"];
const WEEKDAYS_SHORT = ["вс", "пн", "вт", "ср", "чт", "пт", "сб"];
const MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"];
const KINDS = { task: "задача", reminder: "напоминание", meeting: "встреча" };
const LABELS = {
  idle: "Нажмите и говорите",
  listening: "Слушаю",
  transcribing: "Распознаю",
  thinking: "Думаю",
  speaking: "Говорю",
};
const STATIC_SUGGESTIONS = ["Что у меня сегодня?", "Поставь встречу завтра в 15:00", "Напомни через час позвонить"];
const REFRESH_MS = 60000;

const state = { selected: null, today: null, sound: true, busy: false, spaceHeld: false };
const recorder = new Recorder();
const player = new Player();
let orb;
let dial;

const isoDate = (date) =>
  `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
const parseDate = (iso) => new Date(`${iso}T12:00:00`);
const addDays = (iso, days) => {
  const date = parseDate(iso);
  date.setDate(date.getDate() + days);
  return isoDate(date);
};

function preference(key, fallback) {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

function remember(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Настройка просто не сохранится.
  }
}

function setMode(mode, source) {
  document.body.dataset.mode = mode;
  $("stage-label").textContent = LABELS[mode];
  $("mic").setAttribute("aria-label", mode === "listening" ? "Остановить запись" : "Начать запись");
  const orbMode = mode === "transcribing" || mode === "thinking" ? "working" : mode;
  orb?.setMode(orbMode, source);
}

// Разговор ---------------------------------------------------------------------------------

function addMessage(role, text, options = {}) {
  const node = document.createElement("div");
  node.className = `msg msg--${role}`;
  if (options.voice) node.classList.add("is-voice");
  if (options.question) node.classList.add("msg--question");
  if (options.reminder) node.classList.add("msg--reminder");
  if (options.pending) node.classList.add("msg--pending");
  node.append(document.createTextNode(text));
  if (options.items?.length) node.append(renderCards(options.items));
  $("log").append(node);
  $("log").scrollTop = $("log").scrollHeight;
  return node;
}

function renderCards(items) {
  const list = document.createElement("div");
  list.className = "msg__cards";
  items.slice(0, 6).forEach((item) => {
    const card = document.createElement("div");
    card.className = "card";
    const title = document.createElement("span");
    title.textContent = item.title;
    const kind = document.createElement("span");
    kind.className = "card__kind";
    kind.textContent = KINDS[item.kind];
    const when = document.createElement("span");
    when.className = "card__when";
    when.textContent = item.when;
    card.append(title, kind, when);
    list.append(card);
  });
  return list;
}

function renderChips(suggestions) {
  $("chips").replaceChildren(
    ...suggestions.map((text) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = text;
      button.addEventListener("click", () => sendText(text));
      return button;
    }),
  );
}

async function speak(turn) {
  if (!state.sound) return;
  player.stop();
  try {
    const blob = await api.speech(turn.speech || turn.reply);
    setMode("speaking", () => player.level());
    if (blob) await player.play(blob);
    else await player.speakWithBrowser(turn.speech || turn.reply);
  } catch (error) {
    console.warn("Озвучивание не удалось", error);
  }
}

async function finishTurn(turn, { spoken }) {
  addMessage("assistant", turn.reply, { items: turn.items, question: turn.awaiting });
  renderChips(turn.suggestions.length ? turn.suggestions : []);
  if (turn.changed) await refresh();
  if (spoken) await speak(turn);
  setMode("idle");
  state.busy = false;
  // На уточняющий вопрос голосом отвечают голосом: микрофон открывается сам.
  if (spoken && turn.awaiting && state.sound) startListening();
}

async function sendText(text) {
  if (state.busy || !text.trim()) return;
  player.stop();
  state.busy = true;
  addMessage("user", text);
  setMode("thinking");
  try {
    await finishTurn(await api.chat(text), { spoken: false });
  } catch (error) {
    failTurn(error);
  }
}

function failTurn(error) {
  addMessage("assistant", error.message || "Что-то пошло не так. Попробуйте ещё раз.");
  setMode("idle");
  state.busy = false;
}

async function sendVoice(blob) {
  state.busy = true;
  setMode("transcribing");
  const placeholder = addMessage("user", "…", { voice: true, pending: true });
  try {
    await api.voice(blob, async (event) => {
      if (event.stage === "transcript") {
        placeholder.textContent = event.text;
        placeholder.classList.remove("msg--pending");
      } else if (event.stage === "thinking") {
        setMode("thinking");
      } else if (event.stage === "no_speech" || event.stage === "error") {
        placeholder.remove();
        addMessage("assistant", event.message);
        setMode("idle");
        state.busy = false;
      } else if (event.stage === "done") {
        await finishTurn(event.turn, { spoken: true });
      }
    });
  } catch (error) {
    placeholder.remove();
    failTurn(error);
  }
}

async function startListening() {
  if (state.busy || recorder.active) return;
  player.stop();
  player.connect();
  try {
    await recorder.start(
      (level) => setMode("listening", level),
      (blob, heardSpeech) => {
        setMode("idle");
        if (heardSpeech) sendVoice(blob);
      },
    );
  } catch (error) {
    console.warn(error);
    setMode("idle");
    addMessage("assistant", "Нет доступа к микрофону. Разрешите его в браузере или напишите текстом.");
  }
}

function toggleListening() {
  if (document.body.dataset.mode === "speaking") {
    player.stop();
    return;
  }
  if (recorder.active) recorder.stop();
  else startListening();
}

// Расписание -------------------------------------------------------------------------------

function renderEntry(entry, withDay) {
  const row = document.createElement("li");
  row.className = `entry--${entry.kind}${entry.status === "done" ? " entry--done" : ""}`;
  const time = document.createElement("span");
  time.className = "entry__time";
  time.textContent = withDay ? "" : entry.at || "день";
  const title = document.createElement("span");
  title.className = "entry__title";
  title.textContent = entry.title;
  const actions = document.createElement("span");
  actions.className = "entry__actions";
  const action = (label, text, handler) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = text;
    button.title = label;
    button.setAttribute("aria-label", `${label}: ${entry.title}`);
    button.addEventListener("click", async () => {
      await handler(entry.id);
      await refresh();
    });
    return button;
  };
  if (entry.kind !== "meeting") actions.append(action("Выполнено", "✓", api.complete));
  actions.append(action("Удалить", "✕", api.remove));
  row.append(time, title, actions);
  row.addEventListener("pointerenter", () => dial.highlight(entry.id, true));
  row.addEventListener("pointerleave", () => dial.highlight(entry.id, false));
  row.dataset.id = entry.id;
  return row;
}

function renderList(id, entries, empty, withDay = false) {
  const list = $(id);
  list.replaceChildren(...entries.map((entry) => renderEntry(entry, withDay)));
  if (!entries.length && empty) {
    const row = document.createElement("li");
    row.className = "agenda__empty";
    row.textContent = empty;
    list.append(row);
  }
}

async function refresh() {
  const today = isoDate(new Date());
  state.selected ??= today;
  const weekStart = addDays(today, -((parseDate(today).getDay() + 6) % 7));
  const week = Array.from({ length: 7 }, (_, index) => addDays(weekStart, index));
  const data = await api.agenda(week[0], addDays(week[0], 6));
  state.today = data.today;
  if (!week.includes(state.selected)) {
    const extra = await api.agenda(state.selected, state.selected);
    data.occurrences.push(...extra.occurrences);
  }
  const selected = data.occurrences.filter((entry) => entry.day === state.selected);

  const date = parseDate(state.selected);
  $("day-weekday").textContent = WEEKDAYS[date.getDay()];
  $("day-date").textContent = `${date.getDate()} ${MONTHS[date.getMonth()]}`;
  renderWeek(week, new Set(data.occurrences.map((entry) => entry.day)));
  renderList("agenda", selected, "Ничего не запланировано");
  renderList("overdue", data.overdue, null, true);
  renderList("undated", data.undated, null, true);
  $("overdue-group").hidden = !data.overdue.length;
  $("undated-group").hidden = !data.undated.length;
  dial.renderEvents(selected);
  dial.renderNow(state.selected === data.today ? data.now : null);
}

function renderWeek(days, busy) {
  $("week").replaceChildren(
    ...days.map((day) => {
      const date = parseDate(day);
      const button = document.createElement("button");
      button.type = "button";
      button.innerHTML = `<span>${WEEKDAYS_SHORT[date.getDay()]}</span><b>${date.getDate()}</b><i></i>`;
      button.classList.toggle("has-items", busy.has(day));
      if (day === state.selected) button.setAttribute("aria-current", "date");
      button.addEventListener("click", () => {
        state.selected = day;
        refresh();
      });
      return button;
    }),
  );
}

// Запуск -----------------------------------------------------------------------------------

function toast(text) {
  const node = document.createElement("div");
  node.className = "toast";
  node.textContent = text;
  $("toasts").append(node);
  setTimeout(() => node.remove(), 8000);
}

function listenToServer() {
  const events = new EventSource(api.eventsUrl());
  events.onmessage = async (message) => {
    const event = JSON.parse(message.data);
    if (event.type === "changed") refresh();
    if (event.type === "reminder") {
      toast(event.text);
      addMessage("assistant", event.text, { reminder: true });
      if (Notification?.permission === "granted") new Notification("Орбита", { body: event.text });
      if (!state.busy && !recorder.active) {
        await speak({ reply: event.text, speech: event.speech });
        setMode("idle");
      }
      refresh();
    }
  };
}

function bindControls() {
  $("mic").addEventListener("click", toggleListening);
  $("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    const input = $("composer-input");
    const text = input.value;
    input.value = "";
    sendText(text);
  });
  $("sound-toggle").addEventListener("click", (event) => {
    state.sound = !state.sound;
    remember("orbita.sound", state.sound ? "on" : "off");
    event.currentTarget.setAttribute("aria-pressed", String(state.sound));
    if (!state.sound) player.stop();
  });
  $("theme-toggle").addEventListener("click", () => {
    const dark = getComputedStyle(document.documentElement).colorScheme === "dark";
    const theme = dark ? "light" : "dark";
    document.documentElement.dataset.theme = theme;
    remember("orbita.theme", theme);
    orb.readColors();
  });
  $("logout").addEventListener("click", () => {
    api.logout();
    location.reload();
  });

  // Пробел работает как рация: удерживаете и говорите.
  addEventListener("keydown", (event) => {
    if (event.code !== "Space" || event.repeat || event.target.matches("input, button, a")) return;
    event.preventDefault();
    state.spaceHeld = true;
    startListening();
  });
  addEventListener("keyup", (event) => {
    if (event.code === "Space" && state.spaceHeld) {
      state.spaceHeld = false;
      recorder.stop();
    }
  });
}

async function enter(user) {
  $("gate").hidden = true;
  $("app").hidden = false;
  orb = new Orb($("orb"), matchMedia("(prefers-reduced-motion: reduce)").matches);
  dial = new Dial($("dial"), (entry) => {
    $("dial-hint").textContent = entry ? `${entry.at} ${entry.title}` : "";
  });
  $("logout").textContent = user.name;
  $("calendar-link").href = user.calendar;
  state.sound = preference("orbita.sound", "on") === "on";
  $("sound-toggle").setAttribute("aria-pressed", String(state.sound));
  bindControls();
  setMode("idle");
  const history = await api.messages();
  history.forEach((message) =>
    addMessage(message.role, message.text, {
      items: message.payload?.items,
      voice: message.payload?.via === "voice",
      reminder: message.payload?.reminder,
      question: message.payload?.awaiting,
    }),
  );
  if (!history.length) {
    addMessage("assistant", `${user.name}, здравствуйте. Скажите или напишите, что запланировать.`);
  }
  const last = history.at(-1)?.payload;
  renderChips(last?.awaiting ? last.suggestions : STATIC_SUGGESTIONS);
  await refresh();
  setInterval(refresh, REFRESH_MS);
  listenToServer();
}

function showGate() {
  $("gate").hidden = false;
  $("gate-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    $("gate-error").textContent = "";
    try {
      await enter(await api.login($("gate-name").value.trim(), $("gate-pin").value));
      if ("Notification" in window && Notification.permission === "default") {
        Notification.requestPermission();
      }
    } catch (error) {
      $("gate-error").textContent = error.message;
    }
  });
}

const savedTheme = preference("orbita.theme", null);
if (savedTheme) document.documentElement.dataset.theme = savedTheme;

if (api.hasToken()) {
  api.me().then(enter).catch(showGate);
} else {
  showGate();
}
