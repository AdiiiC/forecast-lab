from __future__ import annotations
import argparse
import random
from pathlib import Path
import numpy as np
import yaml
from rich.console import Console
from rich.table import Table
from .data_hier import synthetic_retail
from .models import build
from .backtest_hier import hier_backtest, hier_report
from .tracking import run as tracking_run, repro_hash


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser("forecast-lab-hier")
    ap.add_argument("--config", required=True)
    ap.add_argument("--track", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg.get("seed", 0))
    console = Console()
    console.rule(f"[bold]forecast-lab hierarchical[/]  run={repro_hash(cfg)}")

    h = synthetic_retail(**cfg["dataset"].get("synthetic", {}))
    console.print(f"hierarchy: total nodes={h.n_total}  bottom={h.n_bottom}  "
                  f"levels={h.levels}")

    season = cfg["dataset"]["season_length"]
    base_spec = cfg["base_model"]
    def factory():
        return build(base_spec, season_length=season)

    bt = cfg["backtest"]

    with tracking_run("forecast-lab-hier", run_name=repro_hash(cfg),
                      cfg=cfg, enabled=args.track) as mlrun:
        all_reports = {}
        for method in cfg["reconciliation"]:
            console.print(f"[green]→[/] reconciliation = [cyan]{method}[/]")
            res = hier_backtest(factory, h,
                                horizon=bt["horizon"], n_folds=bt["n_folds"],
                                min_train_size=bt["min_train_size"],
                                stride=bt["stride"], mode=bt["mode"],
                                alpha=cfg["intervals"]["alpha"],
                                reconciliation=method)
            df = hier_report(res, h)
            all_reports[method] = df
            for _, row in df.iterrows():
                mlrun.log_metrics({
                    f"{method}.{row['level']}.MAE_base":  row["MAE_base"],
                    f"{method}.{row['level']}.MAE_recon": row["MAE_recon"],
                    f"{method}.{row['level']}.delta_%":   row["delta_%"],
                })

        # Pretty print
        for method, df in all_reports.items():
            t = Table(title=f"Reconciliation: {method}")
            for c in df.columns:
                t.add_column(c, justify="right" if c != "level" else "left")
            for _, row in df.iterrows():
                t.add_row(*[f"{v:.3f}" if isinstance(v, float) else str(v)
                            for v in row.values])
            console.print(t)

        out = Path(cfg["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        for method, df in all_reports.items():
            df.to_csv(out / f"recon_{method}.csv", index=False)
        console.print(f"\n[dim]artifacts → {out}[/]")


if __name__ == "__main__":
    main()