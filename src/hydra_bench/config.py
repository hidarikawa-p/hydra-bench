"""Configuration handling.

Configuration files are JSON. Keys starting with ``_`` (e.g. ``"_comment"``)
are ignored everywhere, so they can be used for notes.
API keys are never read from configuration files; they are taken from
environment variables (see README).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

PROVIDERS = ("anthropic", "openai", "google", "bedrock", "hf_local", "hf_remote", "dummy")

DEFAULT_MAX_TOKENS = 16384


def _strip_comments(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return _strip_comments(json.load(f))


def slugify(text: str) -> str:
    """File-system friendly identifier derived from a label."""
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", text).strip("-").lower()
    return slug or "model"


@dataclass
class ModelConfig:
    """Specification of one model (generator, target or evaluator).

    Attributes
    ----------
    label:     display name used in output files and tables (e.g. "GPT-5.5").
    provider:  one of ``PROVIDERS``.
    model:     model identifier for the provider (API model ID, Bedrock model
               ID / inference profile, or Hugging Face repo ID / local path).
    params:    extra keyword arguments passed as-is to the provider API call.
    max_tokens: output token limit. Used where the API requires it
               (Anthropic ``max_tokens``, local HF ``max_new_tokens``); for
               other providers it is only sent if ``send_max_tokens`` is true.
    """

    label: str
    provider: str
    model: str
    params: dict = field(default_factory=dict)
    max_tokens: int = DEFAULT_MAX_TOKENS
    send_max_tokens: bool = False
    # provider-specific options
    api: str | None = None  # openai: "responses" (default) or "chat"
    base_url: str | None = None  # openai (chat) / OpenAI-compatible servers
    api_key_env: str | None = None  # override the environment variable name
    region: str | None = None  # bedrock
    inference_provider: str | None = None  # hf_remote (e.g. "together", "auto")
    model_kwargs: dict = field(default_factory=dict)  # hf_local from_pretrained kwargs
    concurrency: int | None = None  # overrides the top-level concurrency

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        d = _strip_comments(d)
        allowed = set(cls.__dataclass_fields__)
        unknown = set(d) - allowed
        if unknown:
            raise ValueError(f"Unknown model config keys {sorted(unknown)}; allowed: {sorted(allowed)}")
        for key in ("label", "provider", "model"):
            if key not in d:
                raise ValueError(f"Model config is missing required key '{key}': {d}")
        if d["provider"] not in PROVIDERS:
            raise ValueError(f"Unknown provider '{d['provider']}'. Choose from {PROVIDERS}.")
        return cls(**d)

    @property
    def slug(self) -> str:
        return slugify(self.label)


def load_contexts(spec: Any) -> list[dict]:
    """Load context keywords.

    ``spec`` may be
      * ``"ddc45"`` - the 45 keywords used in the paper (bundled),
      * a path to a JSON file containing a list, or
      * an inline list.
    List items may be strings or ``{"keyword": ..., "category": ...}`` objects.
    """
    if spec is None or spec == "ddc45":
        text = resources.files("hydra_bench.data").joinpath("contexts_ddc45.json").read_text(encoding="utf-8")
        items = json.loads(text)
    elif isinstance(spec, str):
        items = load_json(spec)
    else:
        items = spec
    if isinstance(items, dict) and "contexts" in items:
        items = items["contexts"]
    out = []
    for item in items:
        if isinstance(item, str):
            out.append({"keyword": item, "category": ""})
        else:
            out.append({"keyword": item["keyword"], "category": item.get("category", "")})
    return out
