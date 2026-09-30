"""Anthropic Messages API (env: ANTHROPIC_API_KEY)."""

from __future__ import annotations

from . import EmptyResponseError, Provider, require


class AnthropicProvider(Provider):
    def __init__(self, cfg):
        super().__init__(cfg)
        anthropic = require("anthropic", "anthropic")
        self.client = anthropic.Anthropic(api_key=self._api_key("ANTHROPIC_API_KEY"))

    def complete(self, prompt: str) -> str:
        message = self.client.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **self.cfg.params,
        )
        # As in the experiments, the answer is the last text block
        # (earlier blocks may be thinking blocks when thinking is enabled).
        texts = [b.text for b in message.content if getattr(b, "type", None) == "text"]
        if not texts:
            raise EmptyResponseError(f"No text content (stop_reason={message.stop_reason})")
        return texts[-1]
