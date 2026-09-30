"""Local Hugging Face model via transformers.

``model`` is a Hub repo ID or a local directory. ``model_kwargs`` are passed to
``from_pretrained`` (defaults: ``device_map="auto"``, ``torch_dtype="auto"``).
``params`` are passed to ``generate`` (e.g. ``{"do_sample": false}``);
``max_tokens`` is used as ``max_new_tokens``. The model's chat template is
applied to the single user message. Calls are serialized (no concurrency).
"""

from __future__ import annotations

import threading

from . import EmptyResponseError, Provider, require


class HFLocalProvider(Provider):
    thread_safe = False

    def __init__(self, cfg):
        super().__init__(cfg)
        transformers = require("transformers", "hf-local")
        kwargs = {"device_map": "auto", "torch_dtype": "auto"}
        kwargs.update(cfg.model_kwargs)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(cfg.model, **kwargs)
        self._lock = threading.Lock()

    def complete(self, prompt: str) -> str:
        with self._lock:
            inputs = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.model.device)
            params = {"max_new_tokens": self.cfg.max_tokens}
            params.update(self.cfg.params)
            output = self.model.generate(**inputs, **params)
            new_tokens = output[0][inputs["input_ids"].shape[-1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not text:
            raise EmptyResponseError("Empty response")
        return text
