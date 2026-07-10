import { useEffect, useRef, useState } from "react";
import { ApiError, fetchCompare } from "../api";
import type { CompareResponse } from "../types";
import { ErrorState, InfoState, Loading } from "./States";

const BETTER_LOWER = new Set([
  "MAE", "RMSE", "sMAPE", "MASE", "Winkler", "PI_width",
  "NV_cost(3:1)", "CRPS", "energy", "QLoss", "DM_p_vs_naive",
]);

function deltaColor(metric: string, delta: number): string {
  if (delta === 0) return "";
  const lowerIsBetter = BETTER_LOWER.has(metric);
  const improved = lowerIsBetter ? delta < 0 : delta > 0;
  return improved ? "var(--pos)" : "var(--neg)";
}

function fmt(v: number | string | null | undefined): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isFinite(v) ? v.toFixed(4) : "—";
  return String(v);
}

interface Props {
  runA: string;
  runs: string[];
}

export function CompareTab({ runA, runs }: Props) {
  const [runB, setRunB] = useState<string>("");
  const [data, setData] = useState<CompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const prevKey = useRef("");

  const otherRuns = runs.filter((r) => r !== runA);

  useEffect(() => {
    if (!runB) return;
    const key = `${runA}|${runB}`;
    if (key === prevKey.current) return;
    prevKey.current = key;

    setLoading(true);
    setError(null);
    setData(null);
    fetchCompare(runA, runB)
      .then(setData)
      .catch((e) =>
        setError(e instanceof ApiError ? e.message : "Could not load comparison."),
      )
      .finally(() => setLoading(false));
  }, [runA, runB]);

  // reset when primary run changes
  useEffect(() => {
    setRunB("");
    setData(null);
    setError(null);
    prevKey.current = "";
  }, [runA]);

  return (
    <div className="compare-wrap">
      <p className="caption" style={{ marginBottom: "1rem" }}>
        Compare any two experiments side-by-side. Green delta = improvement, red = regression.
      </p>

      <div className="compare-selector">
        <label className="field-label">Compare&nbsp;<strong>{runA}</strong>&nbsp;against</label>
        <select
          className="run-select"
          value={runB}
          onChange={(e) => setRunB(e.target.value)}
        >
          <option value="">— pick a second experiment —</option>
          {otherRuns.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
      </div>

      {loading && <Loading label="Loading comparison…" />}
      {error && <ErrorState message={error} />}
      {!loading && !error && !runB && (
        <InfoState message="Select a second experiment above to see the comparison." />
      )}

      {data && (
        <>
          <p className="caption" style={{ marginTop: "1rem" }}>
            {data.shared_models.length} shared models · {data.shared_metrics.length} shared metrics
          </p>
          <div className="table-wrap">
            <table className="leaderboard compare-table">
              <thead>
                <tr>
                  <th className="model">model</th>
                  {data.shared_metrics.map((m) => (
                    <th key={m} colSpan={3} style={{ textAlign: "center" }}>
                      {m}
                    </th>
                  ))}
                </tr>
                <tr>
                  <th />
                  {data.shared_metrics.map((m) => (
                    <>
                      <th key={`${m}-a`} className="sub-col">{data.run_a}</th>
                      <th key={`${m}-b`} className="sub-col">{data.run_b}</th>
                      <th key={`${m}-d`} className="sub-col">Δ</th>
                    </>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.diff.map((row) => (
                  <tr key={row.model}>
                    <td className="model">{row.model}</td>
                    {data.shared_metrics.map((m) => {
                      const va = row[`${m}_${data.run_a}`] as number | null;
                      const vb = row[`${m}_${data.run_b}`] as number | null;
                      const delta = row[`${m}_delta`] as number | null;
                      const color = delta != null ? deltaColor(m, delta) : "";
                      return (
                        <>
                          <td key={`${m}-a`}>{fmt(va)}</td>
                          <td key={`${m}-b`}>{fmt(vb)}</td>
                          <td
                            key={`${m}-d`}
                            style={{ color, fontWeight: color ? 600 : undefined }}
                          >
                            {delta != null
                              ? `${delta >= 0 ? "+" : ""}${delta.toFixed(4)}`
                              : "—"}
                          </td>
                        </>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
