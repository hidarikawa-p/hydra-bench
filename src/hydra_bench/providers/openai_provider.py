"""OpenAI (env: OPENAI_API_KEY).

``api: "responses"`` (default) uses ``client.responses.create``, as in the
paper. ``api: "chat"`` uses Chat Completions and, together with ``base_url``,
also covers OpenAI-compatible servers (vLLM, Ollama, LM Studio, ...).
"""

from __future__ import annotations

from . import EmptyResponseError, Provider, require


class OpenAIProvider(Provider):
    def __init__(self, cfg):
        super().__init__(cfg)
        openai = require("openai", "openai")
        self.api = (cfg.api or "responses").lower()
        if self.api not in ("responses", "chat"):
            raise ValueError("openai 'api' must be 'responses' or 'chat'")
        api_key = self._api_key("OPENAI_API_KEY")
        if api_key is None and cfg.base_url:
            api_key = "EMPTY"  # local OpenAI-compatible servers usually ignore the key
        self.client = openai.OpenAI(api_key=api_key, base_url=cfg.base_url)

    def complete(self, prompt: str) -> str:
        params = dict(self.cfg.params)
        if self.api == "responses":
            if self.cfg.send_max_tokens:
                params.setdefault("max_output_tokens", self.cfg.max_tokens)
            response = self.client.responses.create(model=self.cfg.model, input=prompt, **params)
            text = response.output_text
        else:
            if self.cfg.send_max_tokens:
                params.setdefault("max_completion_tokens", self.cfg.max_tokens)
            response = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[{"role": "user", "content": prompt}],
                **params,
            )
            text = response.choices[0].message.content if response.choices else None
        if not text:
            raise EmptyResponseError("Empty response")
        return text
