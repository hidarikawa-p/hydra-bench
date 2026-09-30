"""Shared utilities: retrying calls, resumable JSONL caches, parallel execution."""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

from tqdm import tqdm

from .providers import EmptyResponseError, Provider

log = logging.getLogger("hydra_bench")


def call_with_retry(provider: Provider, prompt: str, max_retries: int = 3, backoff: float = 2.0) -> str:
    """Call ``provider.complete`` with exponential backoff on API errors.

    Empty responses (refusals, content filters) are *not* retried: the
    experiments treated them as failures without retrying.
    """
    attempt = 0
    while True:
        try:
            return provider.complete(prompt)
        except EmptyResponseError:
            raise
        except Exception as e:  # noqa: BLE001 - SDK-specific exception types
            attempt += 1
            if attempt > max_retries:
                raise
            wait = backoff**attempt
            log.warning("%s: %s (retry %d/%d in %.0fs)", provider.cfg.label, e, attempt, max_retries, wait)
            time.sleep(wait)


class JsonlCache:
    """Append-only JSONL file keyed by ``id``; enables resuming interrupted runs."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.records: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        self.records[rec["id"]] = rec  # later lines override earlier ones

    def done(self, key: str) -> bool:
        rec = self.records.get(key)
        return rec is not None and rec.get("status") == "ok"

    def add(self, rec: dict) -> None:
        with self._lock:
            self.records[rec["id"]] = rec
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_tasks(
    tasks: Iterable[dict],
    fn: Callable[[dict], dict],
    cache: JsonlCache,
    concurrency: int = 1,
    desc: str = "",
) -> None:
    """Run ``fn`` over tasks that are not yet completed in ``cache``."""
    todo = [t for t in tasks if not cache.done(t["id"])]
    if not todo:
        log.info("%s: nothing to do (all cached)", desc)
        return
    bar = tqdm(total=len(todo), desc=desc)
    if concurrency <= 1:
        for t in todo:
            cache.add(fn(t))
            bar.update(1)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [ex.submit(fn, t) for t in todo]
            for fut in as_completed(futures):
                cache.add(fut.result())
                bar.update(1)
    bar.close()


def effective_concurrency(provider: Provider, top_level: int, model_level: int | None) -> int:
    c = model_level if model_level is not None else top_level
    if not provider.thread_safe:
        return 1
    return max(1, int(c))
