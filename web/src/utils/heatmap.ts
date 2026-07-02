import type { MetricRow, MetricValue } from "../types";

/** Columns where a lower value is better (green), mirroring RdYlGn_r. */
export const LOWER_IS_BETTER = new Set(["MAE", "RMSE", "MASE", "sMAPE"]);
/** Columns where a higher value is better (green), mirroring RdYlGn. */
export const HIGHER_IS_BETTER = new Set(["coverage", "skill_vs_naive_%"]);

export interface Range {
  min: number;
  max: number;
}

function toNumber(value: MetricValue): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function columnRange(rows: MetricRow[], column: string): Range | null {
  const values: number[] = [];
  for (const row of rows) {
    const n = toNumber(row[column]);
    if (n !== null) values.push(n);
  }
  if (values.length === 0) return null;
  return { min: Math.min(...values), max: Math.max(...values) };
}

// RdYlGn stops: red -> yellow -> green.
const RED = [215, 48, 39];
const YELLOW = [255, 255, 191];
const GREEN = [26, 152, 80];

function lerp(a: number[], b: number[], t: number): number[] {
  return a.map((v, i) => Math.round(v + (b[i] - v) * t));
}

/** Interpolated Red-Yellow-Green color for t in [0, 1] (0 = red, 1 = green). */
function ryg(t: number): number[] {
  const clamped = Math.max(0, Math.min(1, t));
  return clamped < 0.5
    ? lerp(RED, YELLOW, clamped / 0.5)
    : lerp(YELLOW, GREEN, (clamped - 0.5) / 0.5);
}

export function heatColor(
  column: string,
  value: MetricValue,
  range: Range | null,
): string | undefined {
  const n = toNumber(value);
  if (n === null || range === null) return undefined;

  const lower = LOWER_IS_BETTER.has(column);
  const higher = HIGHER_IS_BETTER.has(column);
  if (!lower && !higher) return undefined;

  const span = range.max - range.min;
  const norm = span === 0 ? 0.5 : (n - range.min) / span;
  const t = lower ? 1 - norm : norm;
  const [r, g, b] = ryg(t);
  return `rgba(${r}, ${g}, ${b}, 0.26)`;
}
