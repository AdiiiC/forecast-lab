export type MetricValue = string | number | null;

export interface MetricRow {
  model: string;
  [column: string]: MetricValue;
}

export interface MetricsResponse {
  columns: string[];
  rows: MetricRow[];
}

export interface ImageItem {
  name: string;
  file: string;
}

export type DecisionValue = number | number[] | string | boolean;

export interface DecisionsResponse {
  exists: boolean;
  decisions: Record<string, DecisionValue>;
}

export interface ReliabilityRow {
  model: string;
  nominal: number | null;
  empirical: number | null;
  mean_width: number | null;
}

export interface ReliabilityResponse {
  exists: boolean;
  rows: ReliabilityRow[];
}

export interface ForecastSeries {
  actual: (number | null)[];
  forecast: (number | null)[];
  lo: (number | null)[] | null;
  hi: (number | null)[] | null;
}

export interface ForecastsResponse {
  exists: boolean;
  alpha: number | null;
  series: Record<string, ForecastSeries>;
}

// ── Run comparison ─────────────────────────────────────────────────────────

export interface CompareDiffRow {
  model: string;
  [key: string]: number | string | null;
}

export interface CompareResponse {
  run_a: string;
  run_b: string;
  shared_models: string[];
  shared_metrics: string[];
  diff: CompareDiffRow[];
}

// ── Async jobs ─────────────────────────────────────────────────────────────

export type JobStatus = "running" | "success" | "failed";

export interface JobResponse {
  job_id: string;
  status: JobStatus;
  config: string;
  elapsed_seconds: number | null;
  exit_code: number | null;
  log_tail: string[];
}

export interface JobSummary {
  job_id: string;
  status: JobStatus;
  config: string;
}
