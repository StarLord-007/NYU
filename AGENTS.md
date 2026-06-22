# AGENTS.md

## Cursor Cloud specific instructions

This repo is a **batch Python ML / data-science project** (NYU — microgravity
ignition classifier). There is **no web app, API server, database, or
long-running service** — everything runs as one-shot CLI scripts against the
committed `Microgravity_Database*.csv` / `database*.csv` datasets. There is no
lint config and no automated test suite.

### Environment / running
- Use `python3` (there is **no `python` shim** on PATH).
- Core Python deps are listed in `requirements.txt` (installed by the update
  script). They are enough for all modeling/eval/audit/presentation scripts; no
  API keys or external services are needed for this path.
- Scripts that import from `xgb_ignition_model_2` (i.e. `make_presentation.py`,
  `print_thresholds_v3.py`) must be run **from the repo root**.
- `Fable/` scripts are run from repo root and reference the dataset via
  `--data Microgravity_Database_Latest.csv` (see `Fable/README.md` for the full
  command list). `python3 Fable/fable_eval.py --quick` is the fastest end-to-end
  smoke test (~80s); `fable_train.py` writes a model to `Fable/model_outputs/`.
- Root scripts (`xgb_ignition_model_2.py`, etc.) write artifacts to
  `artifacts_v2/` / `artifacts_v3/`.

### Optional ETL stage (only if rebuilding the dataset from PDFs)
- The `First-Try/` PDF→data pipeline needs the heavy/optional deps in
  `requirements-etl.txt` (`torch`, `transformers`, `docling`, `PyMuPDF`, etc.)
  **and** an LLM API key (`GROQ_API_KEY` / `OPENAI_API_KEY` for a
  Groq/OpenAI-compatible endpoint). These are intentionally NOT installed by the
  default update script. The final CSVs are already committed, so this stage is
  not required for modeling.
