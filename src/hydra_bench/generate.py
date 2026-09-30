"""Situation generation (Section 3.1).

For each (generator model, context keyword, repetition):
  1. generate an original situation in the given context,
  2. generate two mutually exclusive origins (origin1 = trap O1, origin2 = O2),
  3. modify the situation so that origin2 holds, adding a misleading detail
     that makes origin1 look plausible -> modified situation S.
All three steps use the same generator model, as in the paper.

The raw output of step 3 is post-processed by ``clean_situation`` (headings
such as "**Situation:**" and trailing material such as answer options or
restated origins are removed). This reproduces exactly the situations used
in the paper; the raw text is kept in the ``situation_raw`` column.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import __version__
from .config import ModelConfig, load_contexts, slugify
from .prompts import load_templates, render
from .providers import build_provider
from .runner import JsonlCache, call_with_retry, effective_concurrency, run_tasks

log = logging.getLogger("hydra_bench")

SITUATION_COLUMNS = [
    "id", "genre", "category", "model", "original_situation",
    "origins", "origin1", "origin2", "situation", "situation_raw", "generator_id", "created_at", "hydra_version",
]

_LEADING_LABEL = re.compile(r"^\s*\*{0,2}\s*(situation|question|scenario)\s*\*{0,2}\s*[:：]\s*\*{0,2}\s*", re.I)


def clean_situation(text: str) -> str:
    """Extract the situation paragraph from the raw step-3 output.

    The prompt asks for a single paragraph, but some models add a title or a
    label ("**Situation:**") before it, or answer options, restated origins or
    explanations of the misleading detail after it. We split the output into
    blank-line separated blocks, drop a leading "Situation:/Question:/Scenario:"
    label from each block, and keep the longest block. Applied to the raw
    outputs of the paper's generation run, this reproduces all 269 situations
    shown to the target models exactly.
    """
    blocks = [b for b in re.split(r"\n[ \t]*\n", text) if b.strip()]
    if not blocks:
        return text.strip()
    blocks = [_LEADING_LABEL.sub("", b).strip("\n") for b in blocks]
    return max(blocks, key=len)


def parse_origins(text: str) -> tuple[str, str]:
    """Extract origin1/origin2 from the raw step-2 output (informational only;
    the raw text is what is passed to step 3, as in the experiments)."""
    clean = text.replace("*", "")
    m1 = re.search(r"origin\s*1\s*[:：]\s*(.+?)(?=\n\s*origin\s*2\s*[:：]|\Z)", clean, re.S | re.I)
    m2 = re.search(r"origin\s*2\s*[:：]\s*(.+)", clean, re.S | re.I)
    return (m1.group(1).strip() if m1 else "", m2.group(1).strip() if m2 else "")


def run_generate(cfg: dict, output_dir: str | None = None, limit: int | None = None) -> list[Path]:
    contexts = load_contexts(cfg.get("contexts", "ddc45"))
    if limit:
        contexts = contexts[:limit]
    n_rep = int(cfg.get("n_per_context", 1))
    out_dir = Path(output_dir or cfg.get("output_dir", "outputs/situations"))
    out_dir.mkdir(parents=True, exist_ok=True)
    templates = load_templates(cfg.get("prompts"))
    max_retries = int(cfg.get("max_retries", 3))
    concurrency = int(cfg.get("concurrency", 1))
    do_clean = bool(cfg.get("clean_situations", True))

    generators = [ModelConfig.from_dict(g) for g in cfg["generators"]]
    written = []
    for gen in generators:
        provider = build_provider(gen)
        cache = JsonlCache(out_dir / ".cache" / f"situations_{gen.slug}.jsonl")

        tasks = []
        for ctx in contexts:
            for rep in range(n_rep):
                tasks.append({
                    "id": f"{gen.slug}__{slugify(ctx['keyword'])}__{rep:02d}",
                    "genre": ctx["keyword"],
                    "category": ctx["category"],
                })

        def work(task: dict, provider=provider, gen=gen) -> dict:
            rec = {**task, "model": gen.label, "generator_id": gen.model, "hydra_version": __version__}
            try:
                situ = call_with_retry(provider, render(templates["situation"], context=task["genre"]), max_retries)
                ori = call_with_retry(provider, render(templates["origins"], original_situation=situ), max_retries)
                q = call_with_retry(
                    provider, render(templates["modify"], original_situation=situ, origins=ori), max_retries
                )
                o1, o2 = parse_origins(ori)
                rec.update(
                    status="ok", original_situation=situ, origins=ori, origin1=o1, origin2=o2,
                    situation=clean_situation(q) if do_clean else q, situation_raw=q,
                    created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                )
            except Exception as e:  # noqa: BLE001
                log.warning("Generation failed for %s: %s", task["id"], e)
                rec.update(status="failed", error=f"{type(e).__name__}: {e}")
            return rec

        conc = effective_concurrency(provider, concurrency, gen.concurrency)
        run_tasks(tasks, work, cache, conc, desc=f"generate [{gen.label}]")

        order = {t["id"]: i for i, t in enumerate(tasks)}
        recs = [cache.records[t["id"]] for t in tasks if t["id"] in cache.records]
        ok = sorted([r for r in recs if r.get("status") == "ok"], key=lambda r: order[r["id"]])
        failed = [r for r in recs if r.get("status") != "ok"]

        path = out_dir / f"situations_{gen.slug}.csv"
        pd.DataFrame(ok, columns=SITUATION_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")
        written.append(path)
        log.info("%s: %d situations written to %s", gen.label, len(ok), path)
        fail_path = out_dir / f"failures_{gen.slug}.csv"
        if failed:
            pd.DataFrame(failed)[["id", "genre", "model", "error"]].to_csv(fail_path, index=False, encoding="utf-8-sig")
            log.warning("%s: %d failures recorded in %s (rerun to retry)", gen.label, len(failed), fail_path)
        elif fail_path.exists():
            fail_path.unlink()
    return written
