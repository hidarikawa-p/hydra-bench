"""Amazon Bedrock Converse API (standard AWS credentials; ``region`` optional).

Extra ``params`` are passed to ``converse`` (e.g. ``{"inferenceConfig":
{"maxTokens": 8192}}``).
"""

from __future__ import annotations

from . import EmptyResponseError, Provider, require


class BedrockProvider(Provider):
    def __init__(self, cfg):
        super().__init__(cfg)
        boto3 = require("boto3", "bedrock")
        kwargs = {"region_name": cfg.region} if cfg.region else {}
        self.client = boto3.client("bedrock-runtime", **kwargs)

    def complete(self, prompt: str) -> str:
        params = dict(self.cfg.params)
        if self.cfg.send_max_tokens:
            params.setdefault("inferenceConfig", {}).setdefault("maxTokens", self.cfg.max_tokens)
        response = self.client.converse(
            modelId=self.cfg.model,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            **params,
        )
        blocks = response.get("output", {}).get("message", {}).get("content", [])
        # Reasoning models (e.g. DeepSeek-R1) return a reasoningContent block
        # before the answer; use the text blocks only.
        texts = [b["text"] for b in blocks if "text" in b]
        if not texts:
            raise EmptyResponseError(f"No text content (stopReason={response.get('stopReason')})")
        return "".join(texts)
