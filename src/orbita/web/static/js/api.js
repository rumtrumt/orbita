// Обращения к серверу. Токен живёт только здесь.
const TOKEN_KEY = "orbita.token";

function storage(action, value) {
  try {
    if (action === "get") return localStorage.getItem(TOKEN_KEY);
    if (action === "set") localStorage.setItem(TOKEN_KEY, value);
    if (action === "clear") localStorage.removeItem(TOKEN_KEY);
  } catch {
    // Приватный режим: сессия просто не переживёт перезагрузку.
  }
  return null;
}

let token = storage("get");

export const hasToken = () => Boolean(token);
export const eventsUrl = () => `/api/events?token=${encodeURIComponent(token)}`;

export function logout() {
  token = null;
  storage("clear");
}

async function call(path, options = {}) {
  const headers = { ...options.headers };
  if (token) headers.Authorization = `Bearer ${token}`;
  const response = await fetch(path, { ...options, headers });
  if (response.status === 401) {
    logout();
    location.reload();
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || "Сервер недоступен");
  }
  return response;
}

const json = (body) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export async function login(name, pin) {
  const body = await (await call("/api/session", json({ name, pin }))).json();
  token = body.token;
  storage("set", token);
  return body.user;
}

export const me = async () => (await call("/api/me")).json();
export const messages = async () => (await call("/api/messages")).json();
export const chat = async (text) => (await call("/api/chat", json({ text }))).json();
export const agenda = async (start, end) =>
  (await call(`/api/agenda?start=${start}&end=${end}`)).json();
export const complete = (id) => call(`/api/items/${id}/complete`, { method: "POST" });
export const remove = (id) => call(`/api/items/${id}`, { method: "DELETE" });

export async function speech(text) {
  const response = await call("/api/tts", json({ text }));
  return response.status === 204 ? null : response.blob();
}

// Сервер отдаёт NDJSON: по строке на каждый этап обработки голоса.
export async function voice(blob, onStage) {
  const form = new FormData();
  form.append("audio", blob, "speech.webm");
  const response = await call("/api/voice", { method: "POST", body: form });
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    const lines = buffer.split("\n");
    buffer = lines.pop();
    lines.filter(Boolean).forEach((line) => onStage(JSON.parse(line)));
  }
}
