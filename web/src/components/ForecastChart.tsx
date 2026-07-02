import type { ForecastSeries } from "../types";
import { finiteExtent, formatTick, niceTicks } from "../utils/chart";

interface Props {
  name: string;
  series: ForecastSeries;
  alpha: number | null;
}

const W = 720;
const H = 260;
const M = { top: 14, right: 16, bottom: 30, left: 46 };

function pathFrom(
  values: (number | null)[],
  xAt: (i: number) => number,
  yAt: (v: number) => number,
): string {
  let d = "";
  let pen = false;
  values.forEach((v, i) => {
    if (v === null || !Number.isFinite(v)) {
      pen = false;
      return;
    }
    d += `${pen ? "L" : "M"}${xAt(i).toFixed(1)},${yAt(v).toFixed(1)} `;
    pen = true;
  });
  return d.trim();
}

/** Native SVG forecast chart: actual + forecast lines with a shaded PI band. */
export function ForecastChart({ name, series, alpha }: Props) {
  const { actual, forecast, lo, hi } = series;
  const n = Math.max(actual.length, forecast.length);

  const pool: (number | null)[] = [...actual, ...forecast];
  if (lo) pool.push(...lo);
  if (hi) pool.push(...hi);
  const ext = finiteExtent(pool);

  if (n === 0 || !ext) {
    return (
      <div className="chart-card">
        <div className="chart-head">
          <span className="chart-name">{name}</span>
        </div>
        <div className="chart-empty">No numeric series for this model.</div>
      </div>
    );
  }

  const [dMin, dMax] = ext;
  const ticks = niceTicks(dMin, dMax, 5);
  const yLo = Math.min(dMin, ticks[0]);
  const yHi = Math.max(dMax, ticks[ticks.length - 1]);
  const ySpan = yHi - yLo || 1;

  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;
  const xAt = (i: number) =>
    M.left + (n > 1 ? (i / (n - 1)) * plotW : plotW / 2);
  const yAt = (v: number) => M.top + (1 - (v - yLo) / ySpan) * plotH;

  const hasBand = Boolean(lo && hi);
  let bandPath = "";
  if (lo && hi) {
    const top: string[] = [];
    const bottom: string[] = [];
    for (let i = 0; i < n; i++) {
      const h = hi[i];
      const l = lo[i];
      if (h === null || l === null || !Number.isFinite(h) || !Number.isFinite(l))
        continue;
      top.push(`${xAt(i).toFixed(1)},${yAt(h).toFixed(1)}`);
      bottom.unshift(`${xAt(i).toFixed(1)},${yAt(l).toFixed(1)}`);
    }
    if (top.length) bandPath = `M${top.join("L")}L${bottom.join("L")}Z`;
  }

  const xTicks = n > 1 ? [0, Math.floor((n - 1) / 2), n - 1] : [0];
  const pi = alpha !== null ? Math.round((1 - alpha) * 100) : null;

  return (
    <div className="chart-card">
      <div className="chart-head">
        <span className="chart-name">{name}</span>
        <span className="chart-legend">
          <i className="lg lg-actual" /> actual
          <i className="lg lg-forecast" /> forecast
          {hasBand && (
            <>
              <i className="lg lg-band" /> {pi !== null ? `${pi}% PI` : "PI"}
            </>
          )}
        </span>
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        role="img"
        aria-label={`${name} forecast versus actual`}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line
              x1={M.left}
              x2={W - M.right}
              y1={yAt(t)}
              y2={yAt(t)}
              stroke="#2a313b"
              strokeWidth={1}
            />
            <text x={M.left - 8} y={yAt(t) + 3} className="ax-label" textAnchor="end">
              {formatTick(t)}
            </text>
          </g>
        ))}

        {xTicks.map((i) => (
          <text key={i} x={xAt(i)} y={H - 10} className="ax-label" textAnchor="middle">
            {i}
          </text>
        ))}
        <text x={M.left + plotW / 2} y={H - 10} className="ax-title" textAnchor="middle" opacity={0}>
          horizon step
        </text>

        {bandPath && <path d={bandPath} fill="rgba(182,199,63,0.16)" stroke="none" />}

        <path
          d={pathFrom(actual, xAt, yAt)}
          fill="none"
          stroke="#e6ebf0"
          strokeWidth={1.8}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        <path
          d={pathFrom(forecast, xAt, yAt)}
          fill="none"
          stroke="#b6c73f"
          strokeWidth={1.8}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      </svg>
    </div>
  );
}
