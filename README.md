# BFI-Adapt

BFI-Adapt is an evaluation toolkit for measuring whether event-induced personality shifts in LLM agents align with human-expected Big Five change directions.

The repository provides the experiment runner, multi-provider LLM client, and analysis scripts used to compute pair-level reliability, directional consistency, and BFI-Adapt scores. It intentionally does not include raw model outputs, paper source, or generated paper figures.

## Repository Contents

- `scripts/full_experiment_v2.py`: single-model experiment runner for descriptive personas.
- `scripts/full_experiment_v2_multi.py`: curated multi-model runner using the shared LLM client.
- `scripts/kappa_dcr_analysis.py`: pair-level kappa, DCR, directional correctness, and BFI-Adapt scoring.
- `scripts/rq_stats.py`: supporting diagnostics for existence, magnitude, demographic shape, and individual shape.
- `scripts/build_existence_table.py`: helper for rendering existence-summary LaTeX tables from aggregate CSVs.
- `src/persona_change_consistence/llm/`: provider-agnostic chat-completions client and scheduler.
- `configs/api_keys.example.yaml`: API configuration template with placeholders only.
- `BFI.json`: BFI-44 item metadata and reverse-keying specification.

## Installation

```bash
git clone https://github.com/sci-m-wang/BFI-Adapt.git
cd BFI-Adapt
uv sync
```

Without `uv`:

```bash
python -m pip install -e .
```

## Configure API Access

```bash
cp configs/api_keys.example.yaml configs/api_keys.yaml
$EDITOR configs/api_keys.yaml
```

`configs/api_keys.yaml` is ignored by Git. You can also set:

```bash
export PERSONA_LLM_CONFIG=/path/to/api_keys.yaml
```

## Run Experiments

Small pilot:

```bash
python scripts/full_experiment_v2.py --pilot --method single --model gpt-4.1-mini
```

Full single-model run:

```bash
python scripts/full_experiment_v2.py --method single --model gpt-4.1-mini
```

Curated multi-model run:

```bash
python scripts/full_experiment_v2_multi.py \
  --config configs/api_keys.yaml \
  --output-dir results/v2_multi_run
```

## Compute BFI-Adapt Scores

The scoring script consumes one or more result directories containing per-model `full_results.json` files.

```bash
python scripts/kappa_dcr_analysis.py \
  --results-dirs results/v2_multi_run \
  --output-dir outputs/kappa_dcr
```

Supporting diagnostics:

```bash
python scripts/rq_stats.py \
  --results-dirs results/v2_multi_run \
  --kappa-csv outputs/kappa_dcr/per_model.csv \
  --out-dir outputs/rq_stats
```

## Data Policy

Raw LLM outputs may contain long generated reflections and provider traces, so they are not committed to this repository. If released, raw outputs should be published separately as a dataset artifact.

Do not commit API keys, `.env` files, raw `results/`, or LLM trace files.

## License

MIT License.
