# HYDRA: HYpothesis-Driven Reasoning Assessment

Code and data for the paper **"HYDRA: Evaluating Logical Thinking Abilities of LLMs through Formulating Hypotheses and Verification Methods"** (Findings of AACL-IJCNLP 2026).

HYDRA evaluates how well large language models (LLMs) formulate a hypothesis that explains a realistic, slightly unusual situation, together with a method to verify it. All stages are performed by LLMs:

1. **Situation generation** – a generator model writes an unusual situation for a context keyword, proposes two mutually exclusive origins (O1, O2), and rewrites the situation so that O2 holds while a misleading detail makes O1 look plausible (modified situation *S*). O1 and O2 are never shown to the target model.
2. **Hypothesis and verification formulation** – the target model proposes the single most plausible hypothesis *H* and a verification method *V*.
3. **Evaluation** – an evaluator model judges *(H, V)* on six criteria (a)–(f); a pair succeeds only if all six are OK.

Any model from **Anthropic, OpenAI, Google (Gemini API), Amazon Bedrock, and Hugging Face (local or remote)** can be used as generator, target, or evaluator. OpenAI-compatible servers (vLLM, Ollama, ...) are supported as well.

## Installation

```bash
git clone https://github.com/hidarikawa-p/hydra-bench.git
cd hydra-bench
pip install -e ".[api]"          # Anthropic, OpenAI, Google, Bedrock, HF remote
pip install -e ".[hf-local]"     # additionally, local Hugging Face models (transformers + torch)
```

Individual extras: `anthropic`, `openai`, `google`, `bedrock`, `hf-local`, `hf-remote`, `all`, `dev`. Python ≥ 3.10.

API keys are read from environment variables only (never from config files):

| Provider | Environment variables |
|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` |
| `openai` | `OPENAI_API_KEY` |
| `google` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| `bedrock` | standard AWS credentials (`AWS_PROFILE`, `AWS_ACCESS_KEY_ID`, ...) |
| `hf_remote` / `hf_local` | `HF_TOKEN` (for gated models) |

A different variable name can be set per model with `"api_key_env"`.

## Quick start (offline, no API keys)

```bash
hydra-bench generate -c configs/dummy_generate.json --limit 5
hydra-bench evaluate -c configs/dummy_evaluate.json
```

The `dummy` provider returns fixed, well-formed outputs so that you can check the pipeline end to end.

## Usage

### 1. Generate situations

```bash
hydra-bench generate -c configs/paper_generate.json
```

For every generator and every context keyword, three calls are made (original situation → origins → modified situation). Some models add a title, answer options, or even the correct origin around the requested paragraph; only the situation paragraph is kept (the raw output is saved in `situation_raw`). Output: `outputs/situations/situations_<generator>.csv`. Failed generations (e.g. content filters) are listed in `failures_<generator>.csv` and excluded.

### 2. Evaluate target models

```bash
hydra-bench evaluate -c configs/paper_evaluate.json
```

This runs every target on all situations in `situations_dir` and then judges the answers with the evaluator:

```
outputs/results/
├── answers/<target>.csv                    # hypotheses and verification methods
└── evaluations/<evaluator>/
    ├── <target>.csv                        # judgments (a)-(f), abcok, final
    └── summary_*.csv                       # aggregated tables
```

Useful options:

- `--skip-target` – judge existing answers only, e.g. with a different evaluator (Appendix C: `configs/paper_evaluate_gpt55_judge.json`).
- `--skip-judge` – generate answers only.
- `--targets "GPT-5.5" "Claude Opus 4.7"` – run a subset of the configured targets.
- `--limit N` – use only the first N situations (for testing).
- `-s DIR` / `-o DIR` – override `situations_dir` / `output_dir`.

To evaluate on the released dataset, point `situations_dir` to `data/situations` (the default in the paper configs).

### 3. Summarize

`evaluate` summarizes automatically; you can also run:

```bash
hydra-bench summarize outputs/results/evaluations/claude-opus-4.7 \
    --exclude "GPT-5.4 Nano" "Gemma 3 IT 4B" --bootstrap 1000
```

| File | Content |
|---|---|
| `summary_by_target.csv` | success rate (%) per criterion, `abcok` (a–c all OK), and final, per target plus pooled average (Table 2); optional bootstrap 95% CI |
| `summary_failure_breakdown.csv` | share of all-OK / exactly one NG / two or more NG |
| `summary_by_category.csv` | final success rate per Dewey Decimal category (Table 3) |
| `summary_generator_x_target.csv` | final success rate per (generator, target) pair |

### Resuming

Every call is cached in `.cache/*.jsonl` next to the outputs. If a run is interrupted, rerun the same command: finished items are skipped and failed items are retried.

## Configuration

Configs are JSON; keys starting with `_` are ignored (use them for comments).

**Generation config**

| Key | Description |
|---|---|
| `contexts` | `"ddc45"` (the 45 keywords of the paper, bundled), a path to a JSON list, or an inline list. Items are strings or `{"keyword": ..., "category": ...}`. |
| `n_per_context` | situations per (keyword, generator); default 1 |
| `clean_situations` | keep only the situation paragraph of the step-3 output (default `true`, as in the paper); the raw text is stored in `situation_raw` |
| `generators` | list of model specs |
| `output_dir`, `max_retries` (default 3), `concurrency` (default 1), `prompts` | see below |

**Evaluation config**

| Key | Description |
|---|---|
| `situations_dir` | directory with situation CSVs (needs at least a `situation` column) |
| `targets` | list of model specs |
| `evaluator` | one model spec |
| `contexts` | used to fill in categories if the situation files lack them |
| `output_dir`, `max_retries`, `concurrency`, `prompts` | as above |

**Model spec**

| Key | Providers | Description |
|---|---|---|
| `label` | all | display name used in files and tables (required) |
| `provider` | all | `anthropic`, `openai`, `google`, `bedrock`, `hf_local`, `hf_remote`, `dummy` (required) |
| `model` | all | model ID, Bedrock model ID / inference profile, HF repo ID or local path (required) |
| `params` | all | extra arguments passed to the API call as-is (e.g. `temperature`, `thinking`, `reasoning`, Bedrock `inferenceConfig`, HF `generate` kwargs) |
| `max_tokens` | all | output token limit, default 16384. Always sent to Anthropic (`max_tokens`) and local HF (`max_new_tokens`); sent to other APIs only if `send_max_tokens` is true |
| `api` | openai | `"responses"` (default, `client.responses.create`) or `"chat"` (Chat Completions) |
| `base_url` | openai | OpenAI-compatible endpoint (use with `"api": "chat"`) |
| `region` | bedrock | AWS region |
| `inference_provider` | hf_remote | Hugging Face Inference Provider (e.g. `"auto"`) |
| `model_kwargs` | hf_local | `from_pretrained` kwargs (defaults: `device_map="auto"`, `torch_dtype="auto"`) |
| `concurrency` | all | per-model override of parallel requests (local HF always runs sequentially) |
| `api_key_env` | API providers | name of the environment variable holding the key |

See `configs/example_all_providers.json` for an example of every provider.

**Custom prompts** – `"prompts": {"target": "my_prompts/target.txt"}` replaces a template with the content of a text file. Names: `situation`, `origins`, `modify`, `target`, `evaluation`. Placeholders: `{context}`, `{original_situation}`, `{origins}`, `{situation}`, `{answer}`.

## Reproducing the paper

- The released situations (`data/situations`) are exactly the texts shown to the target models. `clean_situation` reproduces all of them from the raw generator outputs.
- The default prompts are identical, character for character, to those used in the experiments (Appendix A); `tests/test_prompts.py` checks this.
- All generation parameters were left at the API defaults, except `max_tokens=16384` for Anthropic models. OpenAI models were called through the Responses API.
- The evaluator was Claude Opus 4.7 **without** thinking mode.
- Judgments are parsed with the regular expression `<criterion>\s*[:：]\s*(OK|NG)` (case-insensitive). A criterion that cannot be parsed counts as not OK in the final judgment. Items for which the evaluator returned no text are excluded from the rates.
- Model IDs in `configs/paper_*.json` are those available at the time of the experiments; providers may rename or retire them.

Because the situations cannot be regenerated identically, the situations used in the paper are released in `data/situations` (see [`DATA_CARD.md`](DATA_CARD.md)). Target answers and evaluator judgments are released in `data/results`; for example, Table 2 is reproduced by

```bash
hydra-bench summarize data/results/evaluations/claude-opus-4.7 --exclude "GPT-5.4 Nano" "Gemma 3 IT 4B" -o /tmp/table2
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Citation

```bibtex
To Be Provided
```

## License

Code: MIT (see `LICENSE`). Data in `data/`: CC BY 4.0 (see `DATA_CARD.md` for the terms of the models used to generate it).
