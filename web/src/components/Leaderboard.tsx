import { useMemo } from "react";
import type { MetricRow, MetricValue } from "../types";
import { columnRange, heatColor, type Range } from "../utils/heatmap";

function formatCell(value: MetricValue): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  return String(value);
}

function buildCsv(columns: string[], rows: MetricRow[]): string {
  const header = ["model", ...columns];
  const lines = [header.join(",")];
  for (const row of rows) {
    const cells = header.map((col) => {
      const v = row[col];
      if (v === null || v === undefined) return "";
      const s = String(v);
      return s.includes(",") ? `"${s}"` : s;
    });
    lines.push(cells.join(","));
  }
  return lines.join("\n");
}

interface Props {
  columns: string[];
  rows: MetricRow[];
  run: string;
}

export function Leaderboard({ columns, rows, run }: Props) {
  const ranges = useMemo(() => {
    const map: Record<string, Range | null> = {};
    for (const col of columns) map[col] = columnRange(rows, col);
    return map;
  }, [columns, rows]);

  function download() {
    const csv = buildCsv(columns, rows);
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${run}-metrics.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <>
      <div className="table-wrap">
        <table className="leaderboard">
          <thead>
            <tr>
              <th className="model">model</th>
              {columns.map((c) => (
                <th key={c}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.model}>
                <td className="model">{row.model}</td>
                {columns.map((c) => (
                  <td
                    key={c}
                    style={{ background: heatColor(c, row[c], ranges[c]) }}
                  >
                    {formatCell(row[c])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="toolbar">
        <button className="btn" onClick={download}>
          Download metrics.csv
        </button>
      </div>
    </>
  );
}
