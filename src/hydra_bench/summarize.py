"""Aggregation of evaluation results (Tables 2 and 3 of the paper, plus extras).

Writes to the results directory:
  summary_by_target.csv            success rate per criterion and final (Table 2)
  summary_failure_breakdown.csv    all-OK / exactly one NG / two or more NG
  summary_by_category.csv          final success rate per context category (Table 3)
  summary_generator_x_target.csv   final success rate per (generator, target)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_contexts

log = logging.getLogger("hydra_bench")
CRITERIA = list("abcdef")
RATE_COLS = CRITERIA + ["abcok", "final"]


def load_results(results_dir: str | Path) -> pd.DataFrame:
    d = Path(results_dir)
    files = sorted(p for p in d.glob("*.csv") if not p.name.startswith("summary_"))
    if not files:
        raise FileNotFoundError(f"No result CSV files in {d}")
    df = pd.concat([pd.read_csv(p, encoding="utf-8-sig", keep_default_na=False) for p in files], ignore_index=True)
    if "eval_status" in df.columns:
        valid = df["eval_status"] == "ok"
    else:  # files produced by the original notebooks
        valid = (df["eval"].astype(str).str.strip() != "") & (df["eval"] != "str(message)")
    n_invalid = int((~valid).sum())
    if n_invalid:
        log.info("Excluding %d rows without a valid evaluation", n_invalid)
    return df[valid].reset_index(drop=True)


def _ok(df: pd.DataFrame, col: str) -> pd.Series:
    return (df[col].astype(str) == "OK").astype(float)


def _bootstrap_ci(x: np.ndarray, n_boot: int, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(df: pd.DataFrame, contexts=None, exclude: list[str] | None = None, bootstrap: int = 0) -> dict:
    df = df.copy()
    if exclude:
        df = df[~df["target_model"].isin(exclude)]
    for c in ("abcok", "final"):
        if c not in df.columns or (df[c] == "").any():
            cols = list("abc") if c == "abcok" else CRITERIA
            df[c] = np.where(df[cols].eq("OK").all(axis=1), "OK", "NG")
    n_ng = sum((df[k] != "OK").astype(int) for k in CRITERIA)

    # Table 2 ------------------------------------------------------------------
    rows = []
    for tgt, g in df.groupby("target_model", sort=False):
        row = {"target_model": tgt, "n": len(g), **{c: _ok(g, c).mean() * 100 for c in RATE_COLS}}
        if bootstrap:
            lo, hi = _bootstrap_ci(_ok(g, "final").to_numpy(), bootstrap)
            row["final_ci_low"], row["final_ci_high"] = lo * 100, hi * 100
        rows.append(row)
    by_target = pd.DataFrame(rows).sort_values("final", ascending=False)
    avg = {"target_model": "Average", "n": len(df), **{c: _ok(df, c).mean() * 100 for c in RATE_COLS}}
    by_target = pd.concat([by_target, pd.DataFrame([avg])], ignore_index=True)

    # failure breakdown --------------------------------------------------------------
    def breakdown(mask_df, counts):
        return {"n": len(mask_df), "all_ok": (counts == 0).mean() * 100,
                "one_ng": (counts == 1).mean() * 100, "multi_ng": (counts >= 2).mean() * 100}

    fb = [{"target_model": t, **breakdown(g, n_ng[g.index])} for t, g in df.groupby("target_model", sort=False)]
    fb.append({"target_model": "All", **breakdown(df, n_ng)})
    failure = pd.DataFrame(fb)

    # Table 3 ------------------------------------------------------------------
    by_category = None
    if contexts is not None and (df["category"] == "").any():
        mapping = {c["keyword"]: c["category"] for c in load_contexts(contexts)}
        df.loc[df["category"] == "", "category"] = df["genre"].map(mapping).fillna("")
    if (df["category"] != "").any():
        by_category = (
            df.assign(**{c: _ok(df, c) * 100 for c in RATE_COLS})
            .groupby("category")[RATE_COLS].mean()
            .assign(n=df.groupby("category").size())
            .reset_index()
        )

    # generator x target -----------------------------------------------------------
    gxt = None
    if "model" in df.columns and (df["model"] != "").any():
        gxt = df.assign(final_ok=_ok(df, "final") * 100).pivot_table(
            index="model", columns="target_model", values="final_ok", aggfunc="mean"
        )
        gxt["All targets"] = df.assign(final_ok=_ok(df, "final") * 100).groupby("model")["final_ok"].mean()

    return {"by_target": by_target, "failure": failure, "by_category": by_category, "generator_x_target": gxt}


def run_summarize(results_dir: str | Path, contexts=None, exclude: list[str] | None = None,
                  bootstrap: int = 0, output_dir: str | Path | None = None) -> dict:
    df = load_results(results_dir)
    res = summarize(df, contexts=contexts, exclude=exclude, bootstrap=bootstrap)
    out = Path(output_dir or results_dir)
    out.mkdir(parents=True, exist_ok=True)
    res["by_target"].to_csv(out / "summary_by_target.csv", index=False, encoding="utf-8-sig")
    res["failure"].to_csv(out / "summary_failure_breakdown.csv", index=False, encoding="utf-8-sig")
    if res["by_category"] is not None:
        res["by_category"].to_csv(out / "summary_by_category.csv", index=False, encoding="utf-8-sig")
    if res["generator_x_target"] is not None:
        res["generator_x_target"].to_csv(out / "summary_generator_x_target.csv", encoding="utf-8-sig")

    with pd.option_context("display.float_format", "{:.1f}".format, "display.width", 200,
                           "display.max_columns", 50):
        print("\n== Success rates (%) by target model ==")
        print(res["by_target"].to_string(index=False))
        print("\n== Failure breakdown (%) ==")
        print(res["failure"].to_string(index=False))
        if res["by_category"] is not None:
            print("\n== Final success rate (%) by category ==")
            print(res["by_category"][["category", "n", "final"]].to_string(index=False))
    print(f"\nSummary files written to {out}")
    return res
