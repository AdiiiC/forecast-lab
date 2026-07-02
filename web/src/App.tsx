import { useEffect, useState } from "react";
import {
  ApiError,
  diagnosticUrl,
  fetchDecisions,
  fetchDiagnostics,
  fetchForecasts,
  fetchMetrics,
  fetchPlots,
  fetchReliability,
  fetchRuns,
  plotUrl,
} from "./api";
import type {
  DecisionValue,
  ForecastsResponse,
  ImageItem,
  MetricsResponse,
  ReliabilityRow,
} from "./types";
import { Sidebar } from "./components/Sidebar";
import { KpiCards } from "./components/KpiCards";
import { Tabs } from "./components/Tabs";
import { Leaderboard } from "./components/Leaderboard";
import { ImageGrid } from "./components/ImageGrid";
import { ForecastChart } from "./components/ForecastChart";
import { DiagnosticsCharts } from "./components/DiagnosticsCharts";
import { Decisions } from "./components/Decisions";
import { ErrorState, InfoState, Loading } from "./components/States";

const TABS = ["Leaderboard", "Forecasts", "Diagnostics", "Decisions"];

function messageOf(err: unknown, fallback: string): string {
  return err instanceof ApiError ? err.message : fallback;
}

export function App() {
  const [runs, setRuns] = useState<string[]>([]);
  const [runsLoading, setRunsLoading] = useState(true);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [metricsError, setMetricsError] = useState<string | null>(null);

  const [plots, setPlots] = useState<ImageItem[]>([]);
  const [plotsError, setPlotsError] = useState<string | null>(null);
  const [forecasts, setForecasts] = useState<ForecastsResponse | null>(null);
  const [diagnostics, setDiagnostics] = useState<ImageItem[]>([]);
  const [diagnosticsError, setDiagnosticsError] = useState<string | null>(null);
  const [reliability, setReliability] = useState<ReliabilityRow[]>([]);
  const [decisions, setDecisions] = useState<Record<string, DecisionValue>>({});
  const [decisionsError, setDecisionsError] = useState<string | null>(null);

  const [active, setActive] = useState(0);

  useEffect(() => {
    fetchRuns()
      .then(({ runs: list }) => {
        setRuns(list);
        setSelected(list.length ? list[list.length - 1] : null);
      })
      .catch((err) =>
        setRunsError(messageOf(err, "Could not load the list of runs.")),
      )
      .finally(() => setRunsLoading(false));
  }, []);

  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    setActive(0);

    setMetrics(null);
    setMetricsError(null);
    setMetricsLoading(true);
    fetchMetrics(selected)
      .then((m) => !cancelled && setMetrics(m))
      .catch(
        (err) =>
          !cancelled &&
          setMetricsError(messageOf(err, "Could not load metrics.")),
      )
      .finally(() => !cancelled && setMetricsLoading(false));

    setPlotsError(null);
    setForecasts(null);
    fetchForecasts(selected)
      .then((r) => !cancelled && setForecasts(r))
      .catch(() => !cancelled && setForecasts({ exists: false, alpha: null, series: {} }));
    fetchPlots(selected)
      .then((r) => !cancelled && setPlots(r.plots))
      .catch((err) => {
        if (cancelled) return;
        setPlots([]);
        setPlotsError(messageOf(err, "Could not load forecast plots."));
      });

    setDiagnosticsError(null);
    setReliability([]);
    fetchReliability(selected)
      .then((r) => !cancelled && setReliability(r.rows))
      .catch(() => !cancelled && setReliability([]));
    fetchDiagnostics(selected)
      .then((r) => !cancelled && setDiagnostics(r.diagnostics))
      .catch((err) => {
        if (cancelled) return;
        setDiagnostics([]);
        setDiagnosticsError(messageOf(err, "Could not load diagnostics."));
      });

    setDecisionsError(null);
    fetchDecisions(selected)
      .then((r) => !cancelled && setDecisions(r.decisions))
      .catch((err) => {
        if (cancelled) return;
        setDecisions({});
        setDecisionsError(messageOf(err, "Could not load decisions."));
      });

    return () => {
      cancelled = true;
    };
  }, [selected]);

  const rows = metrics?.rows ?? [];
  const columns = metrics?.columns ?? [];

  return (
    <div className="shell">
      <Sidebar runs={runs} selected={selected} onSelect={setSelected} />

      <main className="main">
        {runsLoading && <Loading label="Loading runs…" />}
        {runsError && <ErrorState message={runsError} />}
        {!runsLoading && !runsError && runs.length === 0 && (
          <InfoState message="No completed runs found in runs/." />
        )}

        {selected && (
          <>
            <div className="header">
              <span className="hmark" aria-hidden />
              <h1>Forecast Lab</h1>
              <span className="badge">run: {selected}</span>
            </div>
            <p className="caption">
              Backtest results · {rows.length} models · runs/{selected}
            </p>

            {metricsLoading && <Loading label="Loading metrics…" />}
            {metricsError && <ErrorState message={metricsError} />}
            {!metricsLoading && !metricsError && rows.length === 0 && (
              <InfoState message={`Run ${selected} has no metric rows to display.`} />
            )}

            {!metricsLoading && !metricsError && rows.length > 0 && (
              <>
                <KpiCards best={rows[0]} columns={columns} />
                <Tabs tabs={TABS} active={active} onChange={setActive} />

                {active === 0 && (
                  <>
                    <h2 className="section-header">Model Leaderboard</h2>
                    <Leaderboard columns={columns} rows={rows} run={selected} />
                  </>
                )}

                {active === 1 && (
                  <>
                    <h2 className="section-header">Per-model Forecast Plots</h2>
                    {forecasts?.exists &&
                    Object.keys(forecasts.series).length > 0 ? (
                      <div className="chart-stack">
                        {Object.entries(forecasts.series).map(([name, s]) => (
                          <ForecastChart
                            key={name}
                            name={name}
                            series={s}
                            alpha={forecasts.alpha}
                          />
                        ))}
                      </div>
                    ) : plotsError ? (
                      <ErrorState message={plotsError} />
                    ) : (
                      <ImageGrid
                        items={plots}
                        urlFor={(file) => plotUrl(selected, file)}
                        emptyMessage="No forecast data found. Run the backtest first."
                      />
                    )}
                  </>
                )}

                {active === 2 && (
                  <>
                    <h2 className="section-header">Calibration Diagnostics</h2>
                    {reliability.length > 0 ? (
                      <DiagnosticsCharts rows={reliability} />
                    ) : diagnosticsError ? (
                      <ErrorState message={diagnosticsError} />
                    ) : (
                      <ImageGrid
                        items={diagnostics}
                        urlFor={(file) => diagnosticUrl(selected, file)}
                        emptyMessage="No diagnostic data found."
                      />
                    )}
                  </>
                )}

                {active === 3 && (
                  <>
                    <h2 className="section-header">Decision Artifacts</h2>
                    {decisionsError ? (
                      <ErrorState message={decisionsError} />
                    ) : (
                      <Decisions decisions={decisions} />
                    )}
                  </>
                )}
              </>
            )}
          </>
        )}
      </main>
    </div>
  );
}
