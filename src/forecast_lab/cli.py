from __future__ import annotations
import argparse
import json
import pickle
import random
from pathlib import Path
import numpy as np
import yaml
from rich.console import Console

from .data import load_series
from .models import build
from .conformal import ConformalWrapper
from .backtest import backtest
from .report import aggregate, print_table, plot_folds
from .tracking import run as tracking_run, repro_hash
from .tuning import tune
from .calibration import plot_diagnostics


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser("forecast-lab")
    ap.add_argument("--config", required=True)
    ap.add_argument("--tune", action="store_true",
                    help="Run Optuna HPO for models whose spec includes `search`.")
    ap.add_argument("--track", action="store_true", help="Log to MLflow.")
    ap.add_argument("--experiment", default="forecast-lab")
    ap.add_argument("--no-cache", action="store_true",
                    help="Ignore cached fold results and re-run all models.")
    ap.add_argument("--only", nargs="+", metavar="MODEL",
                    help="Run only these model names (others load from cache if available).")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg.get("seed", 0))

    from .device import adapt_config
    profile = adapt_config(cfg)

    console = Console()
    console.rule(f"[bold]forecast-lab[/]  run={repro_hash(cfg)}")
    _gpu = f"{profile.gpu_name} {profile.gpu_memory_gb:g}GB" if profile.gpu_name else "none"
    console.print(
        f"[cyan]host[/] device={profile.device}  cpu={profile.cpu_count}  "
        f"ram={profile.total_ram_gb:g}GB  gpu={_gpu}"
    )

    # ─── Data ──────────────────────────────────────────────────────────────
    y, cov = load_series(cfg["dataset"])
    season = cfg["dataset"]["season_length"]
    console.print(f"series n={len(y):,}  freq={y.index.freqstr}  "
                  f"range=[{y.index.min()} → {y.index.max()}]")
    if cov.has_any:
        console.print(f"covariates: known_future={list(cov.known_future.columns)}  "
                      f"observed={list(cov.observed.columns)}")

    # ─── Optional preprocessing (Milestone 5) ──────────────────────────────
    pp_cfg = cfg.get("preprocess", {})
    if pp_cfg.get("enabled", False):
        from .preprocessing import preprocess
        y, flags, pp_report = preprocess(
            y, season=season,
            do_kalman=pp_cfg.get("kalman", True),
            outlier_method=pp_cfg.get("outliers", "stl"),
            find_changepoints=pp_cfg.get("changepoints", True),
        )
        console.print(f"[yellow]preprocess[/] missing→{pp_report.n_missing_filled}  "
                      f"outliers→{pp_report.n_outliers}  "
                      f"changepoints→{len(pp_report.changepoints)}")

    bt = cfg["backtest"]
    iv = cfg["intervals"]
    par = cfg.get("parallel", {})
    results: dict = {}

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / ".cache"
    cache_dir.mkdir(exist_ok=True)

    with tracking_run(args.experiment, run_name=repro_hash(cfg),
                      cfg=cfg, enabled=args.track) as mlrun:

        # ─── Model loop ────────────────────────────────────────────────────
        for spec in cfg["models"]:
            search = spec.pop("search", None) if isinstance(spec, dict) else None

            # Optional Optuna HPO over walk-forward
            if args.tune and search:
                console.print(f"[yellow]HPO[/] {spec['name']} "
                              f"({search.get('n_trials', 25)} trials)")
                spec = tune(spec, y, season, bt, iv["alpha"], search["space"],
                            n_trials=search.get("n_trials", 25),
                            seed=cfg.get("seed", 0), mlflow_run=mlrun)

            model = build(spec, season_length=season)

            # Auto-wrap point-only models with split conformal if requested
            if iv.get("conformal") and not model.produces_intervals:
                model = ConformalWrapper(model,
                                         calibration_size=iv["calibration_size"],
                                         horizon=bt["horizon"])

            cache_file = cache_dir / f"{model.name}.pkl"
            only_filter = args.only
            skip_run = (
                not args.no_cache
                and (only_filter is None or model.name not in only_filter)
                and cache_file.exists()
            )
            if skip_run:
                with open(cache_file, "rb") as fh:
                    results[model.name] = pickle.load(fh)
                console.print(f"[bold green]→[/] [cyan]{model.name}[/] [dim](loaded from cache)[/]")
                continue

            console.print(f"[bold green]→[/] running [cyan]{model.name}[/]")

            # Parallel walk-forward if enabled (Milestone 6), else serial
            if par.get("enabled", False):
                from .parallel import parallel_backtest
                results[model.name] = parallel_backtest(
                    model, y, cov=cov,
                    horizon=bt["horizon"], n_folds=bt["n_folds"],
                    min_train_size=bt["min_train_size"], stride=bt["stride"],
                    mode=bt["mode"], alpha=iv["alpha"],
                    backend=par.get("backend", "process"),
                    n_workers=par.get("n_workers"),
                    desc=model.name,
                )
            else:
                results[model.name] = backtest(
                    model, y, cov=cov,
                    horizon=bt["horizon"], n_folds=bt["n_folds"],
                    min_train_size=bt["min_train_size"], stride=bt["stride"],
                    mode=bt["mode"], alpha=iv["alpha"], desc=model.name,
                )

            # Checkpoint this model's folds so a re-run can skip it
            with open(cache_file, "wb") as fh:
                pickle.dump(results[model.name], fh)

        # ─── Aggregate + report ────────────────────────────────────────────
        df = aggregate(results, season=season, alpha=iv["alpha"])
        print_table(df, alpha=iv["alpha"])

        df.to_csv(out / "metrics.csv")
        plot_folds(results, out / "plots", alpha=iv["alpha"])
        plot_diagnostics(results, out / "diagnostics")

        # MLflow metrics + artifacts
        for name, row in df.iterrows():
            mlrun.log_metrics({f"{name}.{k}": v for k, v in row.items()
                               if isinstance(v, (int, float)) and v == v})
        mlrun.log_artifact(out / "metrics.csv")

        # ─── Decision-rule artifacts (Milestone 5) ─────────────────────────
        dec_cfg = cfg.get("decisions")
        if dec_cfg:
            from .decision import newsvendor_order, safety_stock, dispatch_threshold
            from .models.base import Forecast

            best = df.index[0]
            last = results[best][-1]
            proxy = Forecast(mean=last.y_pred, lo=last.lo, hi=last.hi,
                             samples=last.samples, quantiles=last.quantiles,
                             q_levels=last.q_levels)
            decisions: dict = {}
            if "newsvendor" in dec_cfg:
                decisions["newsvendor_order"] = newsvendor_order(
                    proxy, **dec_cfg["newsvendor"]).tolist()
            if "safety_stock" in dec_cfg:
                decisions["safety_stock"] = safety_stock(
                    proxy, **dec_cfg["safety_stock"])
            if "dispatch" in dec_cfg:
                plan = dispatch_threshold(proxy, **dec_cfg["dispatch"])
                decisions["dispatch_triggered_steps"] = plan.triggered_steps.tolist()
                decisions["dispatch_expected_overage"] = plan.expected_overage.tolist()

            (out / "decisions.json").write_text(
                json.dumps(decisions, indent=2, default=str))
            mlrun.log_artifact(out / "decisions.json")
            console.print(f"[bold]decisions[/] for best model "
                          f"[cyan]{best}[/] → {list(decisions.keys())}")

        # ─── Verdict line ──────────────────────────────────────────────────
        base = "seasonal_naive"
        if base in df.index:
            winners = df[(df["MAE"] < df.loc[base, "MAE"]) &
                         (df.get("DM_p_vs_naive", 1.0) < 0.05)].index.tolist()
            if winners:
                console.print(f"\n[bold]Beat seasonal-naive at p<0.05:[/] "
                              f"{', '.join(winners)}")
            else:
                console.print("\n[bold red]Nothing beats seasonal-naive at p<0.05. "
                              "That's the honest answer.[/]")
        console.print(f"\n[dim]artifacts → {out}[/]")


if __name__ == "__main__":
    main()