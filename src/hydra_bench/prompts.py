"""Prompt templates used in HYDRA.

The default templates reproduce, character for character, the prompts used in
the experiments reported in the paper (see Appendix A). Placeholders are
substituted with plain string replacement (not ``str.format``), so templates may
safely contain braces.

Placeholders
------------
{context}             context keyword (situation generation, step 1)
{original_situation}  output of step 1
{origins}             raw output of step 2 ("origin1: ...\\norigin2: ...")
{situation}           modified situation S (output of step 3)
{answer}              raw hypothesis/verification output of the target model
"""

from __future__ import annotations

from pathlib import Path

# Step 1: original situation ------------------------------------------------
SITUATION = (
    "Provide an example of a realistic and likely scenario that slightly deviates "
    "from the usual in the context of {context}. "
    "For instance, you might find the road wet when leaving your home in the afternoon, "
    "even though the weather was clear in the morning, "
    "or you might find a café closed during its normal business hours. "
    "Make sure the cause of the situation cannot be easily determined through simple observation, "
    "such as by looking around. Also, avoid subjective examples involving personal feelings or thoughts; "
    "choose an objective and observable situation instead. Describe the situation in a few sentences, "
    "including some background details."
)

# Step 2: two mutually exclusive origins -------------------------------------
# NOTE: the missing space in "announced.Describe" is intentional; it matches
# the prompt used in the reported experiments.
ORIGINS = (
    "Provide two possible origins of the following situation. "
    "Do not assume that the described facts in the situation are incorrect, such as due to a misunderstanding. "
    "Keep both origins realistically plausible and mutually exclusive in principle. "
    "Avoid any origin that, if true, would clearly reveal itself as the cause. "
    'For example, if the situation is "the train did not arrive at the usual time without any announcement," '
    'the origin should not be "the schedule was moved forward only for today," '
    "since such a change would normally be announced."
    "Describe the result in one sentence each, using the following format:\n"
    "origin1:\n"
    "origin2:\n"
    "\n"
    "situation: {original_situation}"
)

# Step 3: modified situation S -----------------------------------------------
MODIFY = (
    "Create a question that asks for the origin of a situation by adding to, modifying, "
    "or reorganizing the given description. "
    "Ensure that origin2 is the correct answer and that origin1 is incorrect. "
    "At the same time, include a misleading detail that makes origin1 appear plausible, "
    "so the question is not too easy. "
    "Present the situation as a single paragraph, without including any instructions or options such as "
    '"Answer the origin of this situation."\n'
    "\n"
    "situation: {original_situation}\n"
    "\n"
    "{origins}"
)

# Target model: hypothesis and verification method -------------------------
TARGET = (
    "Provide the single most plausible hypothesis for why the following situation occurs, "
    "along with a method of verification. "
    "Describe each in a single sentence, using the following format:\n"
    "hypothesis:\n"
    "verification:\n"
    "\n"
    "{situation}"
)

# Evaluator -------------------------------------------------------------------
EVALUATION = (
    "Evaluate the hypothesis and the method used to verify it based on the following criteria:\n"
    "(a) The hypothesis reasonably explains why the situation occurs.\n"
    "(b) The hypothesis is realistically plausible.\n"
    "(c) The hypothesis is not almost always correct as a general rule, "
    "nor is it almost always correct when the situation occurs.\n"
    "(d) The result of the verification method significantly increases the likelihood "
    "of the hypothesis being correct.\n"
    "(e) It is realistically feasible to obtain the result of the verification method.\n"
    "(f) The verification method does not yield nearly the same result every time the situation occurs.\n"
    'Describe each result using only "OK" or "NG," following the format below:\n'
    "a: OK or NG\n"
    "b: OK or NG\n"
    "...\n"
    "f: OK or NG\n"
    "\n"
    "situation: {situation}\n"
    "{answer}"
)

DEFAULTS = {
    "situation": SITUATION,
    "origins": ORIGINS,
    "modify": MODIFY,
    "target": TARGET,
    "evaluation": EVALUATION,
}


def render(template: str, **values: str) -> str:
    """Substitute ``{name}`` placeholders by plain string replacement."""
    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", value)
    return out


def load_templates(overrides: dict | None = None) -> dict[str, str]:
    """Return the prompt templates, optionally overridden by text files.

    ``overrides`` maps a template name (``situation``, ``origins``, ``modify``,
    ``target``, ``evaluation``) to a path of a UTF-8 text file.
    """
    templates = dict(DEFAULTS)
    for name, path in (overrides or {}).items():
        if name.startswith("_"):
            continue
        if name not in DEFAULTS:
            raise ValueError(f"Unknown prompt name '{name}'. Valid names: {sorted(DEFAULTS)}")
        templates[name] = Path(path).read_text(encoding="utf-8")
    return templates
