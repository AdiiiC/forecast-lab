import type { MetricRow, MetricValue } from "../types";
import { finiteExtent, formatTick, niceTicks } from "../utils/chart";
import { InfoState } from "./States";

// ── helpers ───────────────────────────────────────────────────────────────────

function num(v: MetricValue): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

// ── Champion card ─────────────────────────────────────────────────────────────

interface ChampionProps {
  best: MetricRow;
  columns: string[];
}

export function ChampionCard({ best, columns }: ChampionProps) {
  const has = (c: string) => columns.includes(c);
  const mae = has("MAE") ? num(best.MAE) : null;
  const skill = has("skill_vs_naive_%") ? num(best["skill_vs_naive_%"]) : null;
  const sig = best.sig ?? "";
  const sigBadge =
    sig === "***" || sig === "**" || sig === "*"
      ? { label: "statistically significant", cls: "badge-green" }
      : { label: "marginal edge", cls: "badge-amber" };

  return (
    <div className="champion-card">
      <div className="champion-crown" aria-hidden>
        ★
      </div>
      <div className="champion-eyebrow">Best Performing Model</div>
      <div className="champion-name">{best.model}</div>
      {mae !== null && (
        <div className="champion-sub">{mae.toFixed(3)} units average error</div>
      )}
      {skill !== null && (
        <div className="champion-sub">
          {skill >= 0 ? "+" : ""}
          {skill.toFixed(1)}% better than guessing
        </div>
      )}
      <span className={`badge ${sigBadge.cls}`}>{sigBadge.label}</span>
    </div>
  );
}

// ── Horizontal MAE bar chart ───────────────────────────────────────────────────

interface BarProps {
  rows: MetricRow[];
  best: string;
}

const BW = 640;
const BH_ROW = 38;
const BM = { top: 14, right: 60, bottom: 28, left: 110 };

export function MaeBarChart({ rows, best }: BarProps) {
  const sorted = [...rows]
    .filter((r) => num(r.MAE) !== null)
    .sort((a, b) => (num(a.MAE) ?? 0) - (num(b.MAE) ?? 0));

  if (sorted.length === 0) return <InfoState message="No MAE data." />;

  const vals = sorted.map((r) => num(r.MAE) as number);
  const ext = finiteExtent(vals)!;
  const ticks = niceTicks(0, ext[1] * 1.12, 5);
  const xMax = ticks[ticks.length - 1];
  const plotW = BW - BM.left - BM.right;
  const plotH = sorted.length * BH_ROW;
  const totalH = BM.top + plotH + BM.bottom;

  const xAt = (v: number) => BM.left + (v / xMax) * plotW;
  const yAt = (i: number) => BM.top + i * BH_ROW + BH_ROW / 2;

  return (
    <div className="chart-card">
      <div className="chart-head">
        <span className="chart-name">Average prediction error by model</span>
        <span className="chart-sub">shorter bar = more accurate</span>
      </div>
      <svg viewBox={`0 0 ${BW} ${totalH}`} width="100%" role="img" aria-label="MAE bar chart">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={xAt(t)} x2={xAt(t)} y1={BM.top} y2={BM.top + plotH} stroke="#2a313b" />
            <text x={xAt(t)} y={BM.top + plotH + 16} className="ax-label" textAnchor="middle">
              {formatTick(t)}
            </text>
          </g>
        ))}

        {sorted.map((r, i) => {
          const v = num(r.MAE) as number;
          const isWinner = r.model === best;
          const barW = (v / xMax) * plotW;
          const cy = yAt(i);
          return (
            <g key={r.model}>
              <rect
                x={BM.left}
                y={cy - BH_ROW * 0.28}
                width={barW}
                height={BH_ROW * 0.56}
                fill={isWinner ? "#b6c73f" : "#2a313b"}
                rx={2}
              />
              <text
                x={BM.left - 8}
                y={cy + 4}
                className="ax-note"
                textAnchor="end"
                fill={isWinner ? "#b6c73f" : "#e6ebf0"}
              >
                {r.model}
              </text>
              <text x={xAt(v) + 6} y={cy + 4} className="ax-note">
                {v.toFixed(3)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ── Accuracy vs Coverage scatter ─────────────────────────────────────────────

interface ScatterProps {
  rows: MetricRow[];
  best: string;
}

const SW = 640;
const SH = 300;
const SM = { top: 20, right: 20, bottom: 40, left: 52 };

export function AccCovScatter({ rows, best }: ScatterProps) {
  const pts = rows.filter(
    (r) => num(r.MAE) !== null && num(r.coverage) !== null,
  ) as MetricRow[];

  if (pts.length < 2) return null;

  const maeExt = finiteExtent(pts.map((r) => num(r.MAE)!))!;
  const xTicks = niceTicks(maeExt[0] * 0.95, maeExt[1] * 1.05, 5);
  const xLo = xTicks[0];
  const xHi = xTicks[xTicks.length - 1];

  const plotW = SW - SM.left - SM.right;
  const plotH = SH - SM.top - SM.bottom;
  const xAt = (v: number) => SM.left + ((v - xLo) / (xHi - xLo)) * plotW;
  const yAt = (v: number) => SM.top + (1 - v) * plotH;

  const yTicks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div className="chart-card">
      <div className="chart-head">
        <span className="chart-name">Accuracy vs confidence calibration</span>
        <span className="chart-sub">aim for top-left · shaded = 85–95% target zone</span>
      </div>
      <svg viewBox={`0 0 ${SW} ${SH}`} width="100%" role="img" aria-label="accuracy vs coverage scatter">
        {/* grid */}
        {xTicks.map((t) => (
          <g key={`x${t}`}>
            <line x1={xAt(t)} x2={xAt(t)} y1={SM.top} y2={SH - SM.bottom} stroke="#2a313b" />
            <text x={xAt(t)} y={SH - 8} className="ax-label" textAnchor="middle">
              {formatTick(t)}
            </text>
          </g>
        ))}
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={SM.left} x2={SW - SM.right} y1={yAt(t)} y2={yAt(t)} stroke="#2a313b" />
            <text x={SM.left - 6} y={yAt(t) + 4} className="ax-label" textAnchor="end">
              {(t * 100).toFixed(0)}%
            </text>
          </g>
        ))}

        {/* target zone 85–95% */}
        <rect
          x={SM.left}
          width={plotW}
          y={yAt(0.95)}
          height={yAt(0.85) - yAt(0.95)}
          fill="rgba(182,199,63,0.08)"
        />
        <text x={SW - SM.right - 4} y={yAt(0.9) + 3} className="ax-note" textAnchor="end" fill="#b6c73f">
          target zone
        </text>

        {/* points */}
        {pts.map((r) => {
          const x = xAt(num(r.MAE)!);
          const y = yAt(num(r.coverage)!);
          const isWinner = r.model === best;
          return (
            <g key={r.model}>
              <circle
                cx={x}
                cy={y}
                r={isWinner ? 7 : 5}
                fill={isWinner ? "#b6c73f" : "#8b96a4"}
                stroke="#0f1216"
                strokeWidth={1}
              />
              <text
                x={x + (isWinner ? 10 : 7)}
                y={y + 4}
                className="ax-note"
                fill={isWinner ? "#b6c73f" : "#e6ebf0"}
              >
                {r.model}
              </text>
            </g>
          );
        })}

        <text x={SM.left + plotW / 2} y={SH - 4} className="ax-title" textAnchor="middle">
          average error — lower means more accurate →
        </text>
      </svg>
    </div>
  );
}

// ── Overview tab ──────────────────────────────────────────────────────────────

interface OverviewProps {
  rows: MetricRow[];
  columns: string[];
}

export function OverviewTab({ rows, columns }: OverviewProps) {
  if (rows.length === 0) return null;
  const best = rows[0];

  return (
    <div className="overview-grid">
      <ChampionCard best={best} columns={columns} />
      <div className="chart-stack">
        <MaeBarChart rows={rows} best={best.model} />
        <AccCovScatter rows={rows} best={best.model} />
      </div>
    </div>
  );
}
