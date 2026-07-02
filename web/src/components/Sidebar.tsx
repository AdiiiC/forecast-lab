interface Props {
  runs: string[];
  selected: string | null;
  onSelect: (run: string) => void;
}

export function Sidebar({ runs, selected, onSelect }: Props) {
  return (
    <aside className="sidebar">
      <div className="mark">
        <span className="sq" aria-hidden />
        <span className="txt">Forecast Lab</span>
      </div>
      <hr className="rule" />

      <div>
        <label className="field-label" htmlFor="run-select">
          Select run
        </label>
        <select
          id="run-select"
          className="select"
          value={selected ?? ""}
          onChange={(e) => onSelect(e.target.value)}
          disabled={runs.length === 0}
        >
          {runs.length === 0 && <option value="">No runs available</option>}
          {runs.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
      </div>

      {selected && (
        <div className="side-path">Path: runs/{selected}</div>
      )}

      <hr className="rule" />
      <div>
        <div className="side-heading">About</div>
        <p className="side-about">
          Walk-forward backtest explorer with probabilistic metrics, calibration
          diagnostics, and decision artifacts.
        </p>
      </div>
    </aside>
  );
}
