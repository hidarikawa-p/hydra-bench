"""Hugging Face remote inference via huggingface_hub (env: HF_TOKEN).

``model`` is a Hub repo ID or an endpoint URL. ``inference_provider`` selects
an Inference Provider (e.g. "auto", "together", "hf-inference").
"""

from __future__ import annotations

from . import EmptyResponseError, Provider, require


class HFRemoteProvider(Provider):
    def __init__(self, cfg):
        super().__init__(cfg)
        hub = require("huggingface_hub", "hf-remote")
        kwargs = {"token": self._api_key("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
        if cfg.inference_provider:
            kwargs["provider"] = cfg.inference_provider
        self.client = hub.InferenceClient(model=cfg.model, **kwargs)

    def complete(self, prompt: str) -> str:
        params = dict(self.cfg.params)
        if self.cfg.send_max_tokens:
            params.setdefault("max_tokens", self.cfg.max_tokens)
        response = self.client.chat_completion(messages=[{"role": "user", "content": prompt}], **params)
        text = response.choices[0].message.content if response.choices else None
        if not text:
            raise EmptyResponseError("Empty response")
        return text
