"""Hypothesis/verification formulation by target models and judging by an
evaluator model (Sections 3.2 and 4).

Output layout (under ``output_dir``)::

    answers/<target>.csv                    target outputs (evaluator independent)
    evaluations/<evaluator>/<target>.csv    judgments per target
    evaluations/<evaluator>/summary_*.csv   aggregated tables
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from .config import ModelConfig, load_contexts, slugify
from .prompts import load_templates, render
from .providers import build_provider
from .runner import JsonlCache, call_with_retry, effective_concurrency, run_tasks

log = logging.getLogger("hydra_bench")

CRITERIA = list("abcdef")
ANSWER_COLUMNS = [
    "id", "genre", "category", "model", "situation", "target_model",
    "answer", "hypothesis", "verification", "target_status", "target_error",
]
EVAL_COLUMNS = ANSWER_COLUMNS + ["evaluator", "eval", *CRITERIA, "abcok", "final", "eval_status", "eval_error"]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_situations(situations_dir: str | Path, contexts=None) -> pd.DataFrame:
    """Load all situation CSVs (``failures_*.csv`` are ignored)."""
    d = Path(situations_dir)
    files = sorted(p for p in d.glob("*.csv") if not p.name.startswith("failures_"))
    if not files:
        raise FileNotFoundError(f"No situation CSV files found in {d}")
    frames = []
    for p in files:
        df = pd.read_csv(p, encoding="utf-8-sig", keep_default_na=False)
        if "situation" not in df.columns:
            raise ValueError(f"{p} has no 'situation' column")
        for col in ("genre", "category", "model"):
            if col not in df.columns:
                df[col] = ""
        if "id" not in df.columns or (df["id"] == "").any():
            df["id"] = [f"{p.stem}__{i:04d}" for i in range(len(df))]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if df["id"].duplicated().any():
        dup = df.loc[df["id"].duplicated(), "id"].tolist()[:5]
        raise ValueError(f"Duplicate situation ids across files, e.g. {dup}")
    if contexts is not None:
        mapping = {c["keyword"]: c["category"] for c in load_contexts(contexts)}
        missing = df["category"] == ""
        df.loc[missing, "category"] = df.loc[missing, "genre"].map(mapping).fillna("")
    log.info("Loaded %d situations from %d file(s) in %s", len(df), len(files), d)
    return df


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def parse_hypothesis(answer: str) -> tuple[str, str]:
    clean = answer.replace("*", "")
    h = re.search(r"hypothesis\s*[:：]\s*(.+?)(?=\n\s*verification\s*[:：]|\Z)", clean, re.S | re.I)
    v = re.search(r"verification\s*[:：]\s*(.+)", clean, re.S | re.I)
    return (h.group(1).strip() if h else "", v.group(1).strip() if v else "")


def parse_judgment(text: str) -> dict[str, str]:
    """Extract OK/NG per criterion with the same regex as in the experiments.

    Criteria that cannot be parsed are left empty and therefore count as
    not-OK in the final judgment (identical to the reported aggregation).
    """
    out = {}
    for k in CRITERIA:
        m = re.search(rf"{k}\s*[:：]\s*(OK|NG)", text, re.I)
        out[k] = m.group(1) if m else ""
    out["abcok"] = "OK" if all(out[k] == "OK" for k in "abc") else "NG"
    out["final"] = "OK" if all(out[k] == "OK" for k in CRITERIA) else "NG"
    return out


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def run_targets(situations: pd.DataFrame, targets: list[ModelConfig], out_dir: Path, templates, max_retries, concurrency):
    ans_dir = out_dir / "answers"
    ans_dir.mkdir(parents=True, exist_ok=True)
    rows = situations.to_dict("records")
    for tgt in targets:
        provider = build_provider(tgt)
        cache = JsonlCache(ans_dir / ".cache" / f"{tgt.slug}.jsonl")

        def work(row: dict, provider=provider, tgt=tgt) -> dict:
            rec = {c: row.get(c, "") for c in ("id", "genre", "category", "model", "situation")}
            rec["target_model"] = tgt.label
            try:
                ans = call_with_retry(provider, render(templates["target"], situation=row["situation"]), max_retries)
                h, v = parse_hypothesis(ans)
                rec.update(status="ok", answer=ans, hypothesis=h, verification=v, error="")
            except Exception as e:  # noqa: BLE001
                log.warning("Target %s failed on %s: %s", tgt.label, row["id"], e)
                rec.update(status="failed", answer="", hypothesis="", verification="", error=f"{type(e).__name__}: {e}")
            return rec

        conc = effective_concurrency(provider, concurrency, tgt.concurrency)
        run_tasks(rows, work, cache, conc, desc=f"target [{tgt.label}]")
        _write_answers(rows, cache, ans_dir / f"{tgt.slug}.csv")


def _write_answers(rows, cache: JsonlCache, path: Path):
    out = []
    for row in rows:
        rec = cache.records.get(row["id"])
        if rec is None:
            continue
        out.append({**rec, "target_status": rec["status"], "target_error": rec.get("error", "")})
    pd.DataFrame(out, columns=ANSWER_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")
    n_fail = sum(r["target_status"] != "ok" for r in out)
    log.info("Wrote %s (%d rows, %d failed)", path, len(out), n_fail)


def run_judge(answer_files: list[Path], evaluator: ModelConfig, out_dir: Path, templates, max_retries, concurrency):
    eval_dir = out_dir / "evaluations" / evaluator.slug
    eval_dir.mkdir(parents=True, exist_ok=True)
    provider = build_provider(evaluator)
    conc = effective_concurrency(provider, concurrency, evaluator.concurrency)
    written = []
    for path in answer_files:
        answers = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
        rows = answers.to_dict("records")
        cache = JsonlCache(eval_dir / ".cache" / f"{path.stem}.jsonl")

        def work(row: dict) -> dict:
            rec = {"id": row["id"], "evaluator": evaluator.label}
            if row.get("target_status", "ok") != "ok" or not str(row.get("answer", "")).strip():
                rec.update(status="skipped", eval="", error="no target answer")
                return rec
            prompt = render(templates["evaluation"], situation=row["situation"], answer=row["answer"])
            try:
                rec.update(status="ok", eval=call_with_retry(provider, prompt, max_retries), error="")
            except Exception as e:  # noqa: BLE001
                log.warning("Evaluator %s failed on %s: %s", evaluator.label, row["id"], e)
                rec.update(status="failed", eval="", error=f"{type(e).__name__}: {e}")
            return rec

        target_label = rows[0]["target_model"] if rows else path.stem
        run_tasks(rows, work, cache, conc, desc=f"judge [{evaluator.label}] {target_label}")

        out = []
        for row in rows:
            rec = cache.records.get(row["id"])
            if rec is None:
                continue
            merged = {c: row.get(c, "") for c in ANSWER_COLUMNS}
            merged.update(evaluator=evaluator.label, eval=rec["eval"], eval_status=rec["status"], eval_error=rec.get("error", ""))
            if rec["status"] == "ok":
                merged.update(parse_judgment(rec["eval"]))
            else:
                merged.update({k: "" for k in CRITERIA}, abcok="", final="")
            out.append(merged)
        dest = eval_dir / path.name
        pd.DataFrame(out, columns=EVAL_COLUMNS).to_csv(dest, index=False, encoding="utf-8-sig")
        written.append(dest)
        n_bad = sum(r["eval_status"] != "ok" for r in out)
        log.info("Wrote %s (%d rows, %d not evaluated)", dest, len(out), n_bad)
    return eval_dir, written


def run_evaluate(cfg: dict, output_dir: str | None = None, skip_target: bool = False, skip_judge: bool = False,
                 only_targets: list[str] | None = None, limit: int | None = None) -> Path | None:
    out_dir = Path(output_dir or cfg.get("output_dir", "outputs/results"))
    templates = load_templates(cfg.get("prompts"))
    max_retries = int(cfg.get("max_retries", 3))
    concurrency = int(cfg.get("concurrency", 1))
    targets = [ModelConfig.from_dict(t) for t in cfg.get("targets", [])]
    if only_targets:
        targets = [t for t in targets if t.label in only_targets or t.slug in only_targets]
        if not targets:
            raise ValueError(f"None of {only_targets} found among configured targets")

    if not skip_target:
        if not targets:
            raise ValueError("No targets configured (use --skip-target to judge existing answers only)")
        situations = load_situations(cfg["situations_dir"], cfg.get("contexts"))
        if limit:
            situations = situations.head(limit)
        run_targets(situations, targets, out_dir, templates, max_retries, concurrency)

    if skip_judge:
        return None
    if "evaluator" not in cfg:
        raise ValueError("No evaluator configured")
    evaluator = ModelConfig.from_dict(cfg["evaluator"])
    ans_dir = out_dir / "answers"
    if targets:
        answer_files = [ans_dir / f"{t.slug}.csv" for t in targets]
        missing = [p for p in answer_files if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Answer files not found: {missing}")
    else:  # judge every answer file present
        answer_files = sorted(ans_dir.glob("*.csv"))
        if not answer_files:
            raise FileNotFoundError(f"No answer files in {ans_dir}")
    eval_dir, _ = run_judge(answer_files, evaluator, out_dir, templates, max_retries, concurrency)

    from .summarize import run_summarize

    run_summarize(eval_dir, contexts=cfg.get("contexts"))
    return eval_dir


__all__ = ["run_evaluate", "load_situations", "parse_judgment", "parse_hypothesis", "slugify"]
