export function finiteExtent(values: (number | null)[]): [number, number] | null {
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (v === null || !Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (min === Infinity) return null;
  return [min, max];
}

export function niceTicks(min: number, max: number, count = 5): number[] {
  if (min === max) {
    const pad = Math.abs(min) || 1;
    min -= pad;
    max += pad;
  }
  const span = max - min;
  const rawStep = span / Math.max(1, count - 1);
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const norm = rawStep / mag;
  const stepMul = norm >= 5 ? 5 : norm >= 2 ? 2 : 1;
  const step = stepMul * mag;
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let t = start; t <= max + step * 1e-6; t += step) {
    ticks.push(Number(t.toFixed(10)));
  }
  return ticks;
}

export function formatTick(value: number): string {
  const abs = Math.abs(value);
  if (abs !== 0 && (abs >= 10000 || abs < 0.001)) {
    return value.toExponential(1);
  }
  if (Number.isInteger(value)) return String(value);
  return value.toFixed(abs >= 100 ? 0 : abs >= 10 ? 1 : 2);
}
