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
