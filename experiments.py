"""Systematic experiment search, robustness check and final test evaluation.

    python experiments.py            # runs everything; finished runs are cached and skipped

Phases (all settings in config.yaml):
  1. search   - greedy, one change at a time, 3 seeds each, scored on the 2014-2015 validation set
  2. cv       - blocked K-fold over years for the top configurations (robustness, picks the winner)
  3. ablation - drop one feature group at a time from the winner
  4. final    - retrain the winner on 2006-2015, evaluate ONCE on 2016

Every single training run is appended to runs/experiments.jsonl (the complete experiment log).
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import weather_lib as wl

RUNS = Path("runs")
LOG = RUNS / "experiments.jsonl"


def key_of(params, seed, fold):
    s = json.dumps(params, sort_keys=True) + f"|{seed}|{fold}"
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def load_cache():
    if not LOG.exists():
        return {}
    return {r["key"]: r for r in map(json.loads, LOG.read_text().splitlines())}


CACHE = {}


def run(ds, params, seed, fold, phase, label, train_m, val_m):
    k = key_of(params, seed, fold)
    if k in CACHE:
        return CACHE[k]
    t0 = time.time()
    res = wl.train_eval(ds, params, seed, train_m, val_m)
    rec = {"key": k, "phase": phase, "label": label, "seed": seed, "fold": fold,
           "params": params, **res, "seconds": round(time.time() - t0, 1)}
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    CACHE[k] = rec
    print(f"  {phase:8s} {label:45s} seed {seed:5d} fold {fold}: val MAE {res['val_mae']:.3f}  "
          f"({res['epochs_run']} ep, {rec['seconds']}s)", flush=True)
    return rec


def score(ds, cfg, params, phase, label):
    tr, va = ds.mask("train"), ds.mask("val")
    maes = [run(ds, params, s, "val", phase, label, tr, va)["val_mae"] for s in cfg["seeds"]]
    return float(np.mean(maes)), float(np.std(maes, ddof=1))


def describe(variant):
    return ", ".join(f"{k}={v}" for k, v in variant.items())


def main():
    global CACHE
    cfg = wl.load_config()
    RUNS.mkdir(exist_ok=True)
    CACHE = load_cache()
    ds = wl.Dataset(cfg, verbose=True)

    # ---------------- 1. greedy search
    print("\n=== 1. search (validation 2014-2015)")
    best = dict(cfg["baseline"])
    best_mean, best_std = score(ds, cfg, best, "search", "baseline")
    steps = [{"stage": "baseline", "change": "Class 2 baseline", "mean": best_mean, "std": best_std, "kept": True, "params": best}]
    for stage in cfg["search"]:
        stage_best = None
        for v in stage["variants"]:
            p = {**best, **v}
            if p == best:
                continue
            m, s = score(ds, cfg, p, "search", f"{stage['name']}: {describe(v)}")
            steps.append({"stage": stage["name"], "change": describe(v), "mean": m, "std": s, "kept": False, "params": p})
            if m < best_mean and (stage_best is None or m < stage_best[0]):
                stage_best = (m, s, p, len(steps) - 1)
        if stage_best:
            best_mean, best_std, best, i = stage_best
            steps[i]["kept"] = True
        print(f"  -> after stage '{stage['name']}': {best_mean:.3f} ± {best_std:.3f}")
    pd.DataFrame(steps).drop(columns="params").round(4).to_csv(RUNS / "search_steps.csv", index=False)

    # ---------------- 2. blocked K-fold over years for the top-k distinct configurations
    print("\n=== 2. blocked K-fold over years")
    tried = {}   # every configuration scored in the search above -> mean validation MAE
    for st in steps:
        tried[json.dumps(st["params"], sort_keys=True)] = st["mean"]
    ranked = sorted(tried.items(), key=lambda kv: kv[1])
    top = [json.loads(k) for k, _ in ranked[: cfg["cv"]["top_k"]]]
    if best not in top:
        top.append(best)
    cv_rows = []
    for rank, p in enumerate(top):
        maes, epochs = [], []
        for years in cfg["cv"]["folds"]:
            va = ds.mask("years", years)
            tr = ds.mask("trainval") & ~va
            for s in cfg["seeds"]:
                r = run(ds, p, s, f"{years[0]}-{years[-1]}", "cv", f"top{rank + 1}", tr, va)
                maes.append(r["val_mae"]); epochs.append(r["best_epoch"])
        cv_rows.append({"config": rank + 1, "params": json.dumps({k: v for k, v in p.items() if cfg["baseline"].get(k) != v}),
                        "cv_mean": np.mean(maes), "cv_std": np.std(maes, ddof=1),
                        "median_best_epoch": int(np.median(epochs)), "full": p})
        print(f"  top{rank + 1}: CV MAE {np.mean(maes):.3f} ± {np.std(maes, ddof=1):.3f}")
    cv_df = pd.DataFrame(cv_rows)
    cv_df.drop(columns="full").round(4).to_csv(RUNS / "cv.csv", index=False)
    winner = cv_df.loc[cv_df["cv_mean"].idxmin()]
    final_p, final_epochs = winner["full"], int(winner["median_best_epoch"])
    (RUNS / "final_config.json").write_text(json.dumps({"params": final_p, "epochs": final_epochs}, indent=2))
    print("  winner:", winner["params"], "| epochs for final training:", final_epochs)

    # ---------------- 3. feature ablation (on the K-fold winner)
    print("\n=== 3. ablation")
    win_mean, win_std = score(ds, cfg, final_p, "ablation", "winner, nothing dropped")
    abl = [{"dropped": "(none)", "mean": win_mean, "std": win_std}]
    for g in cfg["ablation"]:
        m, s = score(ds, cfg, {**final_p, "drop_groups": [g]}, "ablation", f"drop {g}")
        abl.append({"dropped": g, "mean": m, "std": s})
    abl = pd.DataFrame(abl)
    abl["change_vs_full"] = abl["mean"] - win_mean
    abl.round(4).to_csv(RUNS / "ablation.csv", index=False)

    # ---------------- 4. final: retrain on 2006-2015, evaluate once on 2016
    if not cfg["run_test"]:
        print("\nrun_test is false - stopping before the test set.")
        return
    print("\n=== 4. final model, test set 2016")
    tv, te = ds.mask("trainval"), ds.mask("test")
    out = RUNS / "final"; out.mkdir(exist_ok=True)
    preds, maes = [], []
    for s in cfg["seeds"]:
        _, model, prep, _ = wl.train_eval(ds, final_p, s, tv, epochs=final_epochs,
                                          log_csv=out / f"history_seed_{s}.csv", return_model=True)
        model.save(out / f"model_seed_{s}.keras")
        pred = wl.predict_celsius(model, prep, te)
        preds.append(pred); maes.append(float(np.mean(np.abs(pred - ds.y[te]))))
        print(f"  seed {s}: test MAE {maes[-1]:.3f}")
    ens = np.mean(preds, axis=0)
    y = ds.y[te]
    dates = ds.dates[te]
    clim = (pd.Series(ds.y[tv], index=ds.dates[tv].dayofyear).groupby(level=0).mean()
            .reindex(range(1, 367)).interpolate().rolling(15, center=True, min_periods=1).mean())
    summary = {
        "test_mae_mean": float(np.mean(maes)), "test_mae_std": float(np.std(maes, ddof=1)),
        "test_mae_per_seed": maes,
        "test_mae_seed_ensemble": float(np.mean(np.abs(ens - y))),
        "test_rmse_seed_ensemble": float(np.sqrt(np.mean((ens - y) ** 2))),
        "test_persistence_mae": float(np.mean(np.abs(ds.last_mean[te] - y))),
        "test_climatology_mae": float(np.mean(np.abs(clim.loc[dates.dayofyear].to_numpy() - y))),
        "epochs": final_epochs, "params": final_p,
    }
    (out / "test_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame({"date": dates, "actual": y, "ensemble": ens, "persistence": ds.last_mean[te],
                  **{f"seed_{s}": p for s, p in zip(cfg["seeds"], preds)}}).to_csv(out / "test_predictions.csv", index=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "params"}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
