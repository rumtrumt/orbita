// Голосовая «сфера»: концентрические кольца, которые деформирует звук, как запись сейсмографа.
const RINGS = 26;
const POINTS = 144;
const TAU = Math.PI * 2;

// Плавный псевдошум: сумма синусов дешевле шума Перлина и не даёт видимого шага.
const wave = (angle, ring, time) =>
  Math.sin(angle * 3 + time * 1.1 + ring * 0.35) * 0.5 +
  Math.sin(angle * 5 - time * 0.7 + ring * 0.2) * 0.3 +
  Math.sin(angle * 9 + time * 1.9 - ring * 0.5) * 0.2;

export class Orb {
  constructor(canvas, reducedMotion) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.mode = "idle";
    this.level = 0;
    this.smooth = 0;
    this.time = 0;
    this.source = () => 0;
    this.reducedMotion = reducedMotion;
    this.last = performance.now();
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
    this.readColors();
    requestAnimationFrame((now) => this.frame(now));
  }

  resize() {
    const ratio = Math.min(devicePixelRatio || 1, 2);
    const { width, height } = this.canvas.getBoundingClientRect();
    this.canvas.width = width * ratio;
    this.canvas.height = height * ratio;
    this.size = Math.min(width, height);
    this.ratio = ratio;
  }

  readColors() {
    const style = getComputedStyle(document.documentElement);
    this.foreground = style.getPropertyValue("--fg").trim();
    this.accent = style.getPropertyValue("--accent").trim();
  }

  setMode(mode, source) {
    this.mode = mode;
    this.source = source || (() => 0);
  }

  frame(now) {
    const delta = Math.min((now - this.last) / 1000, 0.05);
    this.last = now;
    const speed = { idle: 0.35, listening: 1.1, working: 2.6, speaking: 1.3 }[this.mode];
    this.time += delta * (this.reducedMotion ? 0 : speed);
    const target = this.source();
    this.smooth += (target - this.smooth) * (target > this.smooth ? 0.45 : 0.1);
    this.draw();
    requestAnimationFrame((next) => this.frame(next));
  }

  draw() {
    const { context: c, canvas, size, ratio } = this;
    c.setTransform(ratio, 0, 0, ratio, 0, 0);
    c.clearRect(0, 0, canvas.width, canvas.height);
    c.translate(canvas.width / ratio / 2, canvas.height / ratio / 2);
    const inner = size * 0.03;
    const outer = size * 0.3;
    const active = this.mode === "listening" || this.mode === "speaking";
    const energy = this.mode === "working" ? 0.32 : active ? 0.1 + this.smooth : 0.07;
    c.strokeStyle = active || this.mode === "working" ? this.accent : this.foreground;

    for (let ring = 0; ring < RINGS; ring += 1) {
      const progress = ring / (RINGS - 1);
      const base = inner + (outer - inner) * progress;
      // Середина диска подвижнее краёв: контур сферы остаётся узнаваемым.
      const swing = energy * size * 0.05 * Math.sin(progress * Math.PI);
      c.globalAlpha = 0.14 + 0.5 * (1 - progress) ** 1.5;
      c.lineWidth = 0.9;
      c.beginPath();
      for (let point = 0; point <= POINTS; point += 1) {
        const angle = (point / POINTS) * TAU;
        const radius = base + wave(angle, ring, this.time) * swing;
        const x = Math.cos(angle) * radius;
        const y = Math.sin(angle) * radius;
        if (point === 0) c.moveTo(x, y);
        else c.lineTo(x, y);
      }
      c.stroke();
    }
    c.globalAlpha = 1;
  }
}
