"""Model providers.

Every provider exposes ``complete(prompt: str) -> str``: a single-turn user
message in, the model's text answer out. SDKs are imported lazily so that only
the extras you actually use need to be installed.
"""

from __future__ import annotations

import os

from ..config import ModelConfig


class EmptyResponseError(RuntimeError):
    """The model returned no text (e.g. refusal or content filter)."""


class Provider:
    #: whether calls can safely be issued from several threads
    thread_safe = True

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def complete(self, prompt: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    # helpers -----------------------------------------------------------------
    def _api_key(self, *default_envs: str) -> str | None:
        names = [self.cfg.api_key_env] if self.cfg.api_key_env else list(default_envs)
        for name in names:
            value = os.environ.get(name)
            if value:
                return value
        return None


_CACHE: dict[tuple, Provider] = {}


def build_provider(cfg: ModelConfig) -> Provider:
    """Instantiate (or reuse) the provider for ``cfg``."""
    key = (cfg.provider, cfg.model, cfg.label)
    if key in _CACHE:
        return _CACHE[key]
    p = cfg.provider
    if p == "anthropic":
        from .anthropic_provider import AnthropicProvider as cls
    elif p == "openai":
        from .openai_provider import OpenAIProvider as cls
    elif p == "google":
        from .google_provider import GoogleProvider as cls
    elif p == "bedrock":
        from .bedrock_provider import BedrockProvider as cls
    elif p == "hf_local":
        from .hf_local_provider import HFLocalProvider as cls
    elif p == "hf_remote":
        from .hf_remote_provider import HFRemoteProvider as cls
    elif p == "dummy":
        from .dummy_provider import DummyProvider as cls
    else:  # pragma: no cover - validated in ModelConfig
        raise ValueError(f"Unknown provider {p}")
    provider = cls(cfg)
    _CACHE[key] = provider
    return provider


def require(module: str, extra: str):
    """Import ``module`` or raise a helpful error naming the pip extra."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError(
            f"Package '{module}' is required for this provider. "
            f'Install it with: pip install "hydra-bench[{extra}]"'
        ) from e
