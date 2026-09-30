"""Offline dummy provider for tests and dry runs (no API calls).

Returns deterministic, well-formed outputs for each HYDRA step, so that the
whole pipeline can be exercised without credentials. ``params`` may contain
``{"verdict": "NG"}`` to make the evaluator answer NG for criterion (e).
"""

from __future__ import annotations

import hashlib

from . import Provider


class DummyProvider(Provider):
    def complete(self, prompt: str) -> str:
        h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
        if prompt.startswith("Provide an example of a realistic"):
            return f"[dummy situation {h}] A shop that is always open was closed today."
        if prompt.startswith("Provide two possible origins"):
            return f"origin1: [dummy origin A {h}]\norigin2: [dummy origin B {h}]"
        if prompt.startswith("Create a question that asks for the origin"):
            return f"[dummy modified situation {h}] The shop was closed; a sign mentioned a delivery."
        if prompt.startswith("Provide the single most plausible hypothesis"):
            return f"hypothesis: [dummy hypothesis {h}]\nverification: [dummy verification {h}]"
        if prompt.startswith("Evaluate the hypothesis"):
            e = self.cfg.params.get("verdict", "OK") if int(h, 16) % 2 else "OK"
            return f"a: OK\nb: OK\nc: OK\nd: OK\ne: {e}\nf: OK"
        return f"[dummy output {h}]"
