// Циферблат суток: 24 часа по кругу, полночь сверху. Встречи — дуги, задачи и напоминания — точки.
const NS = "http://www.w3.org/2000/svg";
const RING = 420;
const DOT_RING = 378;
const TAU = Math.PI * 2;

function el(name, attributes = {}, parent) {
  const node = document.createElementNS(NS, name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
  parent?.appendChild(node);
  return node;
}

const toMinutes = (clock) => {
  const [hours, minutes] = clock.split(":").map(Number);
  return hours * 60 + minutes;
};

function polar(minutes, radius) {
  const angle = (minutes / 1440) * TAU - Math.PI / 2;
  return [Math.cos(angle) * radius, Math.sin(angle) * radius];
}

function arcPath(startMinutes, endMinutes, radius) {
  const [x1, y1] = polar(startMinutes, radius);
  const [x2, y2] = polar(endMinutes, radius);
  const large = endMinutes - startMinutes > 720 ? 1 : 0;
  return `M${x1} ${y1}A${radius} ${radius} 0 ${large} 1 ${x2} ${y2}`;
}

export class Dial {
  constructor(svg, onHover) {
    this.svg = svg;
    this.onHover = onHover;
    this.nodes = new Map();
    this.drawFace();
    this.events = el("g", {}, svg);
    this.hand = el("g", {}, svg);
  }

  drawFace() {
    el("circle", { class: "track", r: RING }, this.svg);
    for (let hour = 0; hour < 24; hour += 1) {
      const major = hour % 6 === 0;
      const [x1, y1] = polar(hour * 60, RING + 10);
      const [x2, y2] = polar(hour * 60, RING + (major ? 26 : 18));
      el("line", {
        class: `tick${major ? " tick--major" : ""}`, x1, y1, x2, y2,
      }, this.svg);
      if (hour % 3 === 0) {
        const [x, y] = polar(hour * 60, RING + 50);
        el("text", { x, y }, this.svg).textContent = String(hour).padStart(2, "0");
      }
    }
    this.passed = el("path", { class: "passed" }, this.svg);
  }

  renderEvents(occurrences) {
    this.events.replaceChildren();
    this.nodes.clear();
    occurrences.filter((entry) => entry.at).forEach((entry) => {
      const start = toMinutes(entry.at);
      let node;
      if (entry.end) {
        let end = toMinutes(entry.end);
        if (end <= start) end = 1439;
        const done = entry.status === "done" ? " arc--done" : "";
        node = el("path", { class: `arc${done}`, d: arcPath(start, end, RING) }, this.events);
      } else {
        const [cx, cy] = polar(start, DOT_RING);
        node = el("circle", { class: `dot dot--${entry.kind}`, cx, cy, r: 10 }, this.events);
      }
      node.addEventListener("pointerenter", () => this.onHover(entry));
      node.addEventListener("pointerleave", () => this.onHover(null));
      this.nodes.set(entry.id, node);
    });
  }

  highlight(id, on) {
    this.nodes.get(id)?.classList.toggle("is-hot", on);
  }

  // Стрелка показывается только на сегодняшнем циферблате.
  renderNow(clock) {
    this.hand.replaceChildren();
    if (!clock) {
      this.passed.removeAttribute("d");
      return;
    }
    const minutes = toMinutes(clock);
    const [x1, y1] = polar(minutes, RING - 24);
    const [x2, y2] = polar(minutes, RING + 26);
    el("line", { class: "now", x1, y1, x2, y2 }, this.hand);
    const [cx, cy] = polar(minutes, RING);
    el("circle", { class: "now-dot", cx, cy, r: 7 }, this.hand);
    this.passed.setAttribute("d", arcPath(0.01, Math.max(minutes, 1), RING - 8));
  }
}
