interface Props {
  values: number[];
  height?: number;
}

/** Minimal inline SVG line chart for a numeric series. */
export function LineChart({ values, height = 120 }: Props) {
  if (values.length === 0) return null;

  const width = 640;
  const pad = 6;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const stepX = values.length > 1 ? (width - pad * 2) / (values.length - 1) : 0;

  const points = values
    .map((v, i) => {
      const x = pad + i * stepX;
      const y = height - pad - ((v - min) / span) * (height - pad * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width="100%"
      height={height}
      preserveAspectRatio="none"
      role="img"
      aria-label="series line chart"
    >
      {values.length > 1 ? (
        <polyline
          points={points}
          fill="none"
          stroke="#b6c73f"
          strokeWidth={1.8}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      ) : (
        <circle cx={width / 2} cy={height / 2} r={3} fill="#b6c73f" />
      )}
    </svg>
  );
}
