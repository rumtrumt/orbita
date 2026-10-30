// Запись с микрофона с автоостановкой по тишине и воспроизведение ответов.
const SILENCE_MS = 1100;
const SPEECH_LEVEL = 0.05;
const NO_SPEECH_TIMEOUT_MS = 6000;
const MAX_RECORDING_MS = 30000;

const levelReader = (analyser) => {
  const samples = new Uint8Array(analyser.fftSize);
  return () => {
    analyser.getByteTimeDomainData(samples);
    let sum = 0;
    for (const sample of samples) sum += ((sample - 128) / 128) ** 2;
    return Math.min(Math.sqrt(sum / samples.length) * 3.2, 1);
  };
};

export class Recorder {
  constructor() {
    this.recorder = null;
    this.context = null;
  }

  get active() {
    return this.recorder?.state === "recording";
  }

  async start(onLevelSource, onFinish) {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    this.context = new AudioContext();
    const analyser = this.context.createAnalyser();
    analyser.fftSize = 1024;
    this.context.createMediaStreamSource(stream).connect(analyser);
    const level = levelReader(analyser);
    onLevelSource(level);

    const chunks = [];
    let heardSpeech = false;
    this.recorder = new MediaRecorder(stream);
    this.recorder.ondataavailable = (event) => chunks.push(event.data);
    this.recorder.onstop = () => {
      stream.getTracks().forEach((track) => track.stop());
      this.context.close();
      onFinish(new Blob(chunks, { type: this.recorder.mimeType }), heardSpeech);
    };
    this.recorder.start();

    const started = performance.now();
    let lastVoice = started;
    const watch = () => {
      if (!this.active) return;
      const now = performance.now();
      if (level() > SPEECH_LEVEL) {
        heardSpeech = true;
        lastVoice = now;
      }
      const silent = heardSpeech && now - lastVoice > SILENCE_MS;
      const nobody = !heardSpeech && now - started > NO_SPEECH_TIMEOUT_MS;
      if (silent || nobody || now - started > MAX_RECORDING_MS) this.stop();
      else setTimeout(watch, 60);
    };
    watch();
  }

  stop() {
    if (this.active) this.recorder.stop();
  }
}

export class Player {
  constructor() {
    this.audio = new Audio();
    this.context = null;
    this.level = () => 0;
  }

  // Анализатор подключается один раз: повторный createMediaElementSource бросает ошибку.
  connect() {
    if (this.context) return;
    this.context = new AudioContext();
    const analyser = this.context.createAnalyser();
    analyser.fftSize = 1024;
    const source = this.context.createMediaElementSource(this.audio);
    source.connect(analyser);
    analyser.connect(this.context.destination);
    this.level = levelReader(analyser);
  }

  async play(blob) {
    this.connect();
    await this.context.resume();
    this.audio.src = URL.createObjectURL(blob);
    await new Promise((resolve, reject) => {
      this.audio.onended = resolve;
      this.audio.onpause = resolve;
      this.audio.onerror = reject;
      this.audio.play().catch(reject);
    });
    URL.revokeObjectURL(this.audio.src);
  }

  // Запасной голос: если серверный синтез недоступен, говорит браузер.
  speakWithBrowser(text) {
    return new Promise((resolve) => {
      if (!("speechSynthesis" in window)) {
        resolve();
        return;
      }
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "ru-RU";
      utterance.onend = resolve;
      utterance.onerror = resolve;
      speechSynthesis.speak(utterance);
    });
  }

  stop() {
    this.audio.pause();
    if ("speechSynthesis" in window) speechSynthesis.cancel();
  }
}
