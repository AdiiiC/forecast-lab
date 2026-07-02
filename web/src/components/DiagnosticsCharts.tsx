import type { ReliabilityRow } from "../types";
import { finiteExtent, formatTick, niceTicks } from "../utils/chart";
import { InfoState } from "./States";

interface Props {
  rows: ReliabilityRow[];
}

const SW = 560;
const SH = 300;
const SM = { top: 16, right: 18, bottom: 40, left: 52 };

const TOL = 0.03;

function calibColor(empirical: number, nominal: number): string {
  if (empirical < nominal - TOL) return "#e5564e";
  if (empirical > nominal + TOL) return "#c9a227";
  return "#4bb06a";
}

function SharpnessCoverage({ rows }: { rows: ReliabilityRow[] }) {
  const pts = rows.filter(
    (r) => r.empirical !== null && r.mean_width !== null,
  ) as (ReliabilityRow & { empirical: number; mean_width: number })[];

  if (pts.length === 0) return <InfoState message="No interval models to compare." />;

  const nominal = pts.find((p) => p.nominal !== null)?.nominal ?? 0.9;
  const wExt = finiteExtent(pts.map((p) => p.mean_width)) ?? [0, 1];
  const wTicks = niceTicks(Math.min(0, wExt[0]), wExt[1], 5);
  const yLo = Math.min(0, wTicks[0]);
  const yHi = Math.max(wExt[1], wTicks[wTicks.length - 1]);
  const ySpan = yHi - yLo || 1;

  const plotW = SW - SM.left - SM.right;
  const plotH = SH - SM.top - SM.bottom;
  const xAt = (cov: number) => SM.left + cov * plotW;
  const yAt = (w: number) => SM.top + (1 - (w - yLo) / ySpan) * plotH;
  const xTicks = [0, 0.2, 0.4, 0.6, 0.8, 1];

  return (
    <div className="chart-card">
      <div className="chart-head">
        <span className="chart-name">sharpness vs coverage</span>
        <span className="chart-sub">lower-right is better</span>
      </div>
      <svg viewBox={`0 0 ${SW} ${SH}`} width="100%" role="img" aria-label="sharpness versus coverage">
        {wTicks.map((t) => (
          <g key={`w${t}`}>
            <line x1={SM.left} x2={SW - SM.right} y1={yAt(t)} y2={yAt(t)} stroke="#2a313b" />
            <text x={SM.left - 8} y={yAt(t) + 3} className="ax-label" textAnchor="end">
              {formatTick(t)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={`x${t}`} x={xAt(t)} y={SH - 20} className="ax-label" textAnchor="middle">
            {t.toFixed(1)}
          </text>
        ))}

        <line
          x1={xAt(nominal)}
          x2={xAt(nominal)}
          y1={SM.top}
          y2={SH - SM.bottom}
          stroke="#b6c73f"
          strokeDasharray="4 4"
          strokeWidth={1}
        />
        <text x={xAt(nominal)} y={SM.top + 2} className="ax-note" textAnchor="middle">
          target {nominal}
        </text>

        {pts.map((p) => (
          <g key={p.model}>
            <circle cx={xAt(p.empirical)} cy={yAt(p.mean_width)} r={5}
              fill={calibColor(p.empirical, nominal)} stroke="#0f1216" strokeWidth={1} />
            <text x={xAt(p.empirical) + 8} y={yAt(p.mean_width) + 3} className="ax-note">
              {p.model}
            </text>
          </g>
        ))}

        <text x={SM.left + plotW / 2} y={SH - 4} className="ax-title" textAnchor="middle">
          empirical coverage
        </text>
      </svg>
    </div>
  );
}

function Calibration({ rows }: { rows: ReliabilityRow[] }) {
  const pts = rows.filter((r) => r.empirical !== null) as (ReliabilityRow & {
    empirical: number;
  })[];
  if (pts.length === 0) return <InfoState message="No coverage data available." />;

  const nominal = pts.find((p) => p.nominal !== null)?.nominal ?? 0.9;
  const rowH = 30;
  const CW = 560;
  const left = 120;
  const right = 18;
  const top = 34;
  const barW = CW - left - right;
  const height = top + pts.length * rowH + 16;
  const xAt = (cov: number) => left + Math.max(0, Math.min(1, cov)) * barW;

  return (
    <div className="chart-card">
      <div className="chart-head">
        <span className="chart-name">calibration — empirical coverage</span>
        <span className="chart-sub">bar = achieved · line = {nominal} target</span>
      </div>
      <svg viewBox={`0 0 ${CW} ${height}`} width="100%" role="img" aria-label="calibration coverage by model">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={xAt(t)} x2={xAt(t)} y1={top - 6} y2={height - 8} stroke="#2a313b" />
            <text x={xAt(t)} y={top - 12} className="ax-label" textAnchor="middle">
              {t.toFixed(2)}
            </text>
          </g>
        ))}

        {pts.map((p, i) => {
          const y = top + i * rowH;
          const color = calibColor(p.empirical, nominal);
          return (
            <g key={p.model}>
              <text x={left - 10} y={y + rowH / 2 + 3} className="ax-note" textAnchor="end">
                {p.model}
              </text>
              <rect x={left} y={y + 5} width={xAt(p.empirical) - left} height={rowH - 14}
                fill={color} opacity={0.85} rx={2} />
              <text x={xAt(p.empirical) + 6} y={y + rowH / 2 + 3} className="ax-note">
                {p.empirical.toFixed(3)}
              </text>
            </g>
          );
        })}

        <line x1={xAt(nominal)} x2={xAt(nominal)} y1={top - 2} y2={height - 8}
          stroke="#b6c73f" strokeDasharray="4 4" strokeWidth={1.2} />
      </svg>
    </div>
  );
}

export function DiagnosticsCharts({ rows }: Props) {
  return (
    <div className="chart-stack">
      <SharpnessCoverage rows={rows} />
      <Calibration rows={rows} />
    </div>
  );
}
