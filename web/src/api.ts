import type {
  DecisionsResponse,
  ForecastsResponse,
  ImageItem,
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

export function plotUrl(run: string, file: string): string {
  return `/api/runs/${encodeURIComponent(run)}/plots/${encodeURIComponent(file)}`;
}

export function diagnosticUrl(run: string, file: string): string {
  return `/api/runs/${encodeURIComponent(run)}/diagnostics/${encodeURIComponent(file)}`;
}
