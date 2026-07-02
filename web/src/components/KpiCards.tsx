import type { MetricRow, MetricValue } from "../types";

function num(value: MetricValue): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function fixed(value: MetricValue, digits: number): string {
  const n = num(value);
  return n === null ? "\u2014" : n.toFixed(digits);
}

function percent(value: MetricValue): string {
  const n = num(value);
  return n === null ? "\u2014" : `${(n * 100).toFixed(1)}%`;
}

interface Props {
  best: MetricRow;
  columns: string[];
}

export function KpiCards({ best, columns }: Props) {
  const has = (c: string) => columns.includes(c);
  const cards = [
    { label: "Best Model", value: best.model },
    { label: "MAE", value: has("MAE") ? fixed(best.MAE, 3) : "\u2014" },
    { label: "Coverage", value: has("coverage") ? percent(best.coverage) : "\u2014" },
    { label: "MASE", value: has("MASE") ? fixed(best.MASE, 3) : "\u2014" },
  ];

  return (
    <div className="kpis">
      {cards.map((c) => (
        <div className="metric-card" key={c.label}>
          <div className="metric-label">{c.label}</div>
          <div className="metric-value">{c.value}</div>
        </div>
      ))}
    </div>
  );
}
