"""Google Gemini API via google-genai (env: GEMINI_API_KEY or GOOGLE_API_KEY).

Extra ``params`` are passed as ``config`` (GenerateContentConfig fields),
e.g. ``{"temperature": 0.0}``.
"""

from __future__ import annotations

from . import EmptyResponseError, Provider, require


class GoogleProvider(Provider):
    def __init__(self, cfg):
        super().__init__(cfg)
        genai = require("google.genai", "google")
        self.client = genai.Client(api_key=self._api_key("GEMINI_API_KEY", "GOOGLE_API_KEY"))

    def complete(self, prompt: str) -> str:
        config = dict(self.cfg.params)
        if self.cfg.send_max_tokens:
            config.setdefault("max_output_tokens", self.cfg.max_tokens)
        kwargs = {"config": config} if config else {}
        response = self.client.models.generate_content(model=self.cfg.model, contents=prompt, **kwargs)
        text = None
        try:
            text = response.text  # concatenation of non-thought text parts
        except Exception:  # pragma: no cover - SDK specific
            text = None
        if not text:
            try:
                text = response.candidates[0].content.parts[0].text
            except Exception:
                text = None
        if not text:
            raise EmptyResponseError("Empty response")
        return text
