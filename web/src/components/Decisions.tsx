import type { DecisionValue } from "../types";
import { LineChart } from "./LineChart";
import { InfoState } from "./States";

function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function scalar(value: DecisionValue): string {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(3);
  }
  return String(value);
}

interface Props {
  decisions: Record<string, DecisionValue>;
}

export function Decisions({ decisions }: Props) {
  const entries = Object.entries(decisions);
  if (entries.length === 0) {
    return (
      <InfoState message="No decisions.json found for this run. Add a decisions block to your config to generate them." />
    );
  }

  return (
    <>
      {entries.map(([key, value]) => (
        <div className="decision" key={key}>
          <div className="decision-key">{key}</div>
          {Array.isArray(value) ? (
            value.length === 0 ? (
              <InfoState message="No values recorded for this artifact." />
            ) : (
              <>
                <div className="stat-row">
                  <div className="stat">
                    <div className="stat-label">Mean</div>
                    <div className="stat-value">{mean(value).toFixed(3)}</div>
                  </div>
                  <div className="stat">
                    <div className="stat-label">Min</div>
                    <div className="stat-value">
                      {Math.min(...value).toFixed(3)}
                    </div>
                  </div>
                  <div className="stat">
                    <div className="stat-label">Max</div>
                    <div className="stat-value">
                      {Math.max(...value).toFixed(3)}
                    </div>
                  </div>
                </div>
                <LineChart values={value} />
              </>
            )
          ) : (
            <div className="scalar">{scalar(value)}</div>
          )}
        </div>
      ))}
    </>
  );
}
