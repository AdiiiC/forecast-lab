import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, fetchJobs, pollJob, triggerJob } from "../api";
import type { JobResponse, JobStatus, JobSummary } from "../types";

const STATUS_COLOR: Record<JobStatus, string> = {
  running: "#e8a94d",
  success: "#4bb06a",
  failed: "#e5564e",
};

function StatusBadge({ status }: { status: JobStatus }) {
  return (
    <span
      className="badge"
      style={{
        background: `${STATUS_COLOR[status]}22`,
        color: STATUS_COLOR[status],
        border: `1px solid ${STATUS_COLOR[status]}55`,
      }}
    >
      {status}
    </span>
  );
}

interface JobCardProps {
  job: JobResponse;
}

function JobCard({ job }: JobCardProps) {
  return (
    <div className="job-card">
      <div className="job-card-head">
        <code className="job-id">{job.job_id.slice(0, 8)}…</code>
        <StatusBadge status={job.status} />
        <span className="job-config">{job.config}</span>
        {job.elapsed_seconds != null && (
          <span className="job-elapsed">{job.elapsed_seconds}s</span>
        )}
      </div>
      {job.log_tail.length > 0 && (
        <pre className="job-log">{job.log_tail.join("\n")}</pre>
      )}
    </div>
  );
}

export function JobRunner() {
  const [config, setConfig] = useState("configs/energy_v2.yaml");
  const [track, setTrack] = useState(false);
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);

  // Active job being polled
  const [activeJob, setActiveJob] = useState<JobResponse | null>(null);
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  // Historical jobs list
  const [history, setHistory] = useState<JobSummary[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);

  // Load history on mount
  useEffect(() => {
    fetchJobs()
      .then((r) => setHistory(r.jobs as JobSummary[]))
      .catch(() => {/* no-op — API may not be running */})
      .finally(() => setHistoryLoading(false));
  }, []);

  const stopPolling = useCallback(() => {
    if (pollTimer.current) {
      clearInterval(pollTimer.current);
      pollTimer.current = null;
    }
  }, []);

  // Poll active job every 2s while running
  useEffect(() => {
    if (!activeJob || activeJob.status !== "running") {
      stopPolling();
      return;
    }
    pollTimer.current = setInterval(() => {
      pollJob(activeJob.job_id)
        .then((updated) => {
          setActiveJob(updated);
          if (updated.status !== "running") {
            stopPolling();
            // Refresh history
            fetchJobs().then((r) => setHistory(r.jobs as JobSummary[])).catch(() => {});
          }
        })
        .catch(() => {/* keep existing state */});
    }, 2000);
    return stopPolling;
  }, [activeJob, stopPolling]);

  async function handleLaunch() {
    setLaunching(true);
    setLaunchError(null);
    try {
      const resp = await triggerJob(config.trim(), track);
      // Start polling immediately
      const initial: JobResponse = {
        job_id: resp.job_id,
        status: "running",
        config: config.trim(),
        elapsed_seconds: 0,
        exit_code: null,
        log_tail: [],
      };
      setActiveJob(initial);
    } catch (e) {
      setLaunchError(e instanceof ApiError ? e.message : "Failed to launch job.");
    } finally {
      setLaunching(false);
    }
  }

  return (
    <div className="job-runner">
      <p className="caption" style={{ marginBottom: "1rem" }}>
        Trigger a backtest run from the browser. The job runs server-side; this panel
        streams the log tail and shows status in real time.
      </p>

      {/* Launch form */}
      <div className="job-form">
        <div className="field-group">
          <label className="field-label" htmlFor="config-input">
            Config file path
          </label>
          <input
            id="config-input"
            className="field-input"
            value={config}
            onChange={(e) => setConfig(e.target.value)}
            placeholder="configs/energy_v2.yaml"
          />
        </div>

        <div className="field-group field-inline">
          <label className="field-label checkbox-label">
            <input
              type="checkbox"
              checked={track}
              onChange={(e) => setTrack(e.target.checked)}
            />
            Log to MLflow
          </label>
        </div>

        <button
          className="btn btn-primary"
          onClick={handleLaunch}
          disabled={launching || !config.trim()}
        >
          {launching ? "Launching…" : "Run backtest"}
        </button>

        {launchError && (
          <p className="error-inline">{launchError}</p>
        )}
      </div>

      {/* Active job */}
      {activeJob && (
        <div style={{ marginTop: "1.5rem" }}>
          <h3 className="section-header">Active job</h3>
          <JobCard job={activeJob} />
        </div>
      )}

      {/* History */}
      <div style={{ marginTop: "1.5rem" }}>
        <h3 className="section-header">Job history</h3>
        {historyLoading && (
          <p className="caption">Loading…</p>
        )}
        {!historyLoading && history.length === 0 && (
          <p className="caption">No jobs yet.</p>
        )}
        {history.map((j) => (
          <div key={j.job_id} className="job-history-row">
            <code className="job-id">{j.job_id.slice(0, 8)}…</code>
            <StatusBadge status={j.status as JobStatus} />
            <span className="job-config">{j.config}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
