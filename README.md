# Majority voting and run-to-run consistency of an open-weight LLM on oncology exam questions

Lab 7 (AI Tools for Research, IIIT Allahabad) implementation study by Vibhum Sharma.
**Status: work in progress (protocol drafted, environment test pending).** Full documentation is
written at the end of the project (Prompt 17); commands marked **[TEST REQUIRED]** have not yet been run.

## Question
Does the majority answer of 10 sampled runs (temperature 0.7) of Qwen2.5-7B-Instruct answer oncology
clinical-vignette multiple-choice questions more accurately than its single greedy answer, and how
consistent are its answers across runs? Protocol: [`configs/protocol_v1.yaml`](configs/protocol_v1.yaml).

## Data
MedQA (USMLE, 4-option; Jin et al., arXiv:2009.13081), Hugging Face copy
`GBaker/MedQA-USMLE-4-options-hf`. **No MedQA item text is stored in this repository** (the data
licence is unclear): only item IDs, screening codes and rules. Source and checksums:
[`data/medqa_source.md`](data/medqa_source.md).
- Item screening: [`data/screening/SCREENING_RULES.md`](data/screening/SCREENING_RULES.md),
  agreement report [`data/screening/SCREENING_AGREEMENT.md`](data/screening/SCREENING_AGREEMENT.md)
  (items were screened by AI under written rules; no clinician checked them).
- Splits: [`data/splits/`](data/splits/) — 25 debug items, 633 test items, SHA-256 in the protocol.

## Layout
```
configs/     protocol (single source of truth for settings)
scripts/     data preparation (CPU)
src/         experiment code (written at Prompt 09)
notebooks/   Kaggle notebooks (GPU)
tests/       CPU tests
data/        IDs, codes, rules (raw/ and work/ are local only, never committed)
results/     derived tables and figures (raw/ is local only)
paper/       ACM acmart manuscript
```

## CPU setup (Windows or Linux) — tested on Windows, Python 3.12.7
```
pip install -r requirements.txt
python tests/test_cpu_env.py          # expected: 5/5 checks passed
```

## Rebuild the item list from MedQA — [TEST REQUIRED on a fresh clone]
```
# download the 3 parquet files listed in data/medqa_source.md into data/raw/medqa_4opt_hf/
python -X utf8 scripts/count_oncology_items.py data/raw/medqa_4opt_hf data/work
python -X utf8 scripts/validate_items.py data/raw/medqa_4opt_hf data/screening codes_adjudicated.csv
python -X utf8 scripts/make_splits.py data/screening codes_adjudicated.csv data/splits
```

## GPU (Kaggle, T4 x2) — [TEST REQUIRED]
Import `notebooks/00_environment_test.ipynb` into Kaggle (GPU T4 x2, Internet on) and run it.

## Licence
Code: to be chosen (Prompt 17). MedQA itself is not redistributed here.
