# Multi-Provider LLM Scheduler

A unified client for calling commercial LLM APIs across multiple providers,
designed for the persona-change-consistence experiments.

## Features

- **Unified interface** for OpenAI, Anthropic, Google Gemini, DeepSeek, Qwen,
  GLM (Zhipu), Moonshot, OpenRouter, Doubao (Volcano Engine), and Xiaomi MiMo.
- **Multiple keys per provider** with automatic rotation and load balancing.
- **Rate limiting** (RPM + concurrency caps) computed per-key.
- **Automatic retries** with exponential backoff and jitter.
- **Per-call traces** written to JSONL — full reproducibility and resume.
- **Cost accounting** based on per-model pricing in the config.
- **Thread-pool concurrency** for batch jobs.

## Setup

1. Copy the example config and fill in your real API keys:

   ```bash
   cp configs/api_keys.example.yaml configs/api_keys.yaml
   $EDITOR configs/api_keys.yaml
   ```

   The real file is gitignored — your keys never leave the local machine.

2. Install the optional SDKs you need:

   ```bash
   pip install pyyaml openai           # always required
   pip install anthropic               # for Claude
   pip install google-genai            # for Gemini
   ```

   Or all at once: `pip install -e ".[all-llm]"`.

## Configuration

`configs/api_keys.yaml` has three sections:

```yaml
providers:
  openai:
    type: openai_compat                # or: anthropic, google
    base_url: https://api.openai.com/v1
    keys:
      - id: openai-1                   # used in trace logs
        key: sk-...
        rpm: 5000                      # requests-per-minute (0 = unlimited)
        tpm: 800000                    # tokens-per-minute (informational)
        concurrency: 8                 # max in-flight per key
        enabled: true
    models:
      gpt-4o:
        input_per_1m: 2.50             # USD per 1M input tokens
        output_per_1m: 10.00

models:
  gpt-4o: openai:gpt-4o                # alias -> provider:model_id

defaults:
  retry:
    max_attempts: 5
    initial_backoff: 1.0
    max_backoff: 60.0
    backoff_multiplier: 2.0
    jitter: 0.2
  timeout: 120
  trace_dir: llm_traces
```

### Multiple keys per provider

Just add more entries under `keys:` — the scheduler will pick the least-loaded
eligible key for every request.

### Env-var key references

You can store the actual secret in an environment variable:

```yaml
keys:
  - id: openai-1
    key: ${OPENAI_API_KEY}      # or: env:OPENAI_API_KEY
```

### Adding a new provider

Most Chinese vendors expose an OpenAI-compatible endpoint. Add a block under
`providers:`, set `type: openai_compat`, point `base_url` at the vendor's API,
fill in keys and model prices, and add a route under `models:`.

For Anthropic and Google, use `type: anthropic` / `type: google` — those use
the native SDKs.

## Usage

### Single call

```python
from persona_change_consistence.llm import LLMClient

client = LLMClient(experiment="multimodel_replication_v1")

result = client.chat(
    model="claude-sonnet-4",
    messages=[
        {"role": "system", "content": "You are role-playing as ..."},
        {"role": "user", "content": "Rate yourself on BFI item 1 ..."},
    ],
    temperature=0.0,
    max_tokens=512,
    call_id="persona42_event3_post",   # stable id enables resume
)

print(result.content)
print(f"used {result.input_tokens}+{result.output_tokens} tokens, "
      f"${result.cost_usd:.4f}, latency={result.latency_sec:.2f}s")
```

### Parallel batch with resume

```python
jobs = [
    {
        "model": "gpt-4o-mini",
        "messages": [...],
        "temperature": 0.0,
        "call_id": f"persona{p}_event{e}_post",
    }
    for p in range(100)
    for e in range(11)
]

results = client.batch_chat(jobs, max_workers=8, resume=True)
client.print_summary()
```

If a run is interrupted, restart the script — calls with `call_id`s already
present (and successful) in the trace file are skipped automatically.

### Inspecting the trace

```python
from persona_change_consistence.llm import iter_trace, load_completed_ids

done = load_completed_ids("llm_traces/multimodel_replication_v1.jsonl")
print(len(done), "calls succeeded")

for record in iter_trace("llm_traces/multimodel_replication_v1.jsonl"):
    print(record["call_id"], record["model_alias"], record["total_cost_usd"])
```

## File layout

```
configs/
├── api_keys.example.yaml      # committed template
└── api_keys.yaml              # local, gitignored

src/persona_change_consistence/llm/
├── __init__.py                # public API
├── config.py                  # YAML loader & dataclasses
├── client.py                  # LLMClient + LLMResult
├── scheduler.py               # KeyPool + Scheduler (retries, RPM, concurrency)
├── tracker.py                 # JSONL trace + cost accounting + resume
└── providers/
    ├── base.py                # Provider abstract base, request/response types
    ├── openai_compat.py       # OpenAI / DeepSeek / Qwen / GLM / Moonshot / etc.
    ├── anthropic.py           # Claude
    └── google.py              # Gemini
```

## What gets logged

Every call (success or failure) appends one JSON line to
`<trace_dir>/<experiment>.jsonl`. Each record contains:

- `call_id`, `timestamp`, `experiment`
- Resolved `model_alias`, `provider`, `provider_model`, `key_id`
- `attempts`, `success`, `error`
- `input_tokens`, `output_tokens`, costs (in USD per category and total)
- `latency_sec`
- Full `request` (messages + parameters) and `response` (content + finish reason)

These traces are gitignored by default (`*.llm_trace.jsonl`, `llm_traces/`).
