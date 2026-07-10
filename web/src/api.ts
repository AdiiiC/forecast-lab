import type {
  CompareResponse,
  DecisionsResponse,
  ForecastsResponse,
  ImageItem,
  JobResponse,
  MetricsResponse,
  ReliabilityResponse,
} from "./types";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function getJson<T>(url: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch {
    throw new ApiError("Cannot reach the Forecast Lab API. Is it running?", 0);
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status}).`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* keep default detail */
    }
    throw new ApiError(detail, res.status);
  }
  return (await res.json()) as T;
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError("Cannot reach the Forecast Lab API. Is it running?", 0);
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status}).`;
    try {
      const b = await res.json();
      if (b?.detail) detail = String(b.detail);
    } catch {
      /* keep default detail */
    }
    throw new ApiError(detail, res.status);
  }
  return (await res.json()) as T;
}

export function fetchRuns(): Promise<{ runs: string[] }> {
  return getJson("/api/runs");
}

export function fetchMetrics(run: string): Promise<MetricsResponse> {
  return getJson(`/api/runs/${encodeURIComponent(run)}/metrics`);
}

export function fetchPlots(run: string): Promise<{ plots: ImageItem[] }> {
  return getJson(`/api/runs/${encodeURIComponent(run)}/plots`);
}

export function fetchDiagnostics(
  run: string,
): Promise<{ diagnostics: ImageItem[] }> {
  return getJson(`/api/runs/${encodeURIComponent(run)}/diagnostics`);
}

export function fetchDecisions(run: string): Promise<DecisionsResponse> {
  return getJson(`/api/runs/${encodeURIComponent(run)}/decisions`);
}

export function fetchReliability(run: string): Promise<ReliabilityResponse> {
  return getJson(`/api/runs/${encodeURIComponent(run)}/reliability`);
}

export function fetchForecasts(run: string): Promise<ForecastsResponse> {
  return getJson(`/api/runs/${encodeURIComponent(run)}/forecasts`);
}

export function fetchCompare(
  runA: string,
  runB: string,
): Promise<CompareResponse> {
  return getJson(
    `/api/runs/${encodeURIComponent(runA)}/compare/${encodeURIComponent(runB)}`,
  );
}

export function triggerJob(config: string, track = false): Promise<{ job_id: string; status: string }> {
  return postJson("/api/jobs", { config, track });
}

export function pollJob(jobId: string): Promise<JobResponse> {
  return getJson(`/api/jobs/${encodeURIComponent(jobId)}`);
}

export function fetchJobs(): Promise<{ jobs: { job_id: string; status: string; config: string }[] }> {
  return getJson("/api/jobs");
}

export function plotUrl(run: string, file: string): string {
  return `/api/runs/${encodeURIComponent(run)}/plots/${encodeURIComponent(file)}`;
}

export function diagnosticUrl(run: string, file: string): string {
  return `/api/runs/${encodeURIComponent(run)}/diagnostics/${encodeURIComponent(file)}`;
}