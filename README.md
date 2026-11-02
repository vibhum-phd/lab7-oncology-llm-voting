# Does asking again help? Majority voting and run-to-run consistency of a small open LLM on oncology exam questions

Lab 7 implementation study (AI Tools for Research, IIIT Allahabad) by **Vibhum Sharma**.
Release **`v1.1`** (paper draft plus reproduction tooling; `v1.0-paper-draft` is the paper as first revised). Paper: [`paper/main.pdf`](paper/main.pdf) (ACM `acmart`, 10 pages).

## Problem
Studies that compare large language models (LLMs) with oncology tumor boards usually ask a model once per case, but
an LLM can answer the same question differently when asked again. Using oncology questions from the MedQA
licensing-exam dataset as a controlled testbed, this study asks, for one small open-weight model
(Qwen2.5-7B-Instruct):

- **RQ1 (confirmatory).** Is the majority answer of 10 sampled runs (temperature 0.7) more accurate than the single
  greedy answer (temperature 0)?
- **RQ2.** How consistent are the answers across runs?
- **RQ3.** Does agreement between runs flag answers that are more likely to be correct?

The protocol, including the decision rules, was frozen in [`configs/protocol_v1.yaml`](configs/protocol_v1.yaml) at
commit `7890add`, before any model output on test items existed. Exploratory additions are in
[`configs/protocol_v2.yaml`](configs/protocol_v2.yaml).

## Key results (633 test items; from [`results/results.csv`](results/results.csv))
| | Value |
|---|---|
| Greedy accuracy | 67.1% (95% CI 63.5–70.8) |
| Majority of 10 sampled runs | 70.6% (67.0–74.2) |
| Difference (confirmatory) | +3.5 points (0.9 to 6.0), exact McNemar p = 0.012 |
| of which voting itself (post hoc) | +2.6 points (1.1 to 4.0) |
| Output tokens, vote vs greedy | 10.0× |
| Unanimous items (10/10 runs) | 49.4% |
| AUROC, run agreement vs correct vote | 0.737 (0.695–0.780) |

These are exam questions with one keyed answer. They are a proxy for, not a test of, tumor-board decisions; nothing
here measures clinical usefulness or safety. See the paper's Limitations section.

## Repository layout
```
configs/        protocol_v1.yaml (frozen) and protocol_v2.yaml (exploratory amendment): every setting
data/           medqa_source.md (where to get MedQA, checksums); screening/ (rules, codes, agreement);
                splits/ (item IDs, SHA-256 manifest). data/raw/ and data/work/ are local only (git-ignored)
src/board_vote/ data loading, prompts, parsing and voting, generation (vLLM or mock), evaluation, plots, probe
scripts/        data preparation, generation driver, evaluation, audit, tables, paper numbers (see below)
notebooks/      Kaggle notebooks 00–03 (GPU)
tests/          CPU tests (no GPU needed)
results/        derived results: results.csv, item_level.csv (letters only), tables/, figures/, audit,
                post hoc analyses; env/ holds per-run pip freezes and summaries. Raw model outputs are local only
paper/          main.tex, sections/, refs.bib, generated/ (numbers and tables written by script), figures/
```

## Data and licences
- **MedQA** (Jin et al., 2021), English 4-option USMLE items, from the Hugging Face copy
  `GBaker/MedQA-USMLE-4-options-hf`, commit `17af9355ef89fba60de966eabaeba797c695f86e` (dataset card tag `cc-by-sa-4.0`;
  the original repository states no data licence). **No question text is stored here.** Download it with
  `scripts/fetch_medqa.py`, which checks SHA-256 sums ([`data/medqa_source.md`](data/medqa_source.md)).
- Item IDs and keyed answer letters are released for reproducibility; they are derived from MedQA (see
  [`LICENSE-docs.md`](LICENSE-docs.md)).
- **Raw model outputs are not released**: they restate question text. All statistics can be recomputed from the
  released parsed answers (`results/item_level.csv`).
- No patient data, no paid APIs, no secrets: the Kaggle notebooks clone this public repository without a token, and
  the model and dataset are public and not gated.

## Setup
### CPU (Windows or Linux): data preparation, tests, evaluation, paper numbers
Tested on Windows 11 with Python 3.12.7 in a fresh virtual environment (see "Fresh-clone test" below).
```bash
git clone https://github.com/vibhum-phd/lab7-oncology-llm-voting.git
cd lab7-oncology-llm-voting
git checkout v1.1
python -m venv .venv
.venv\Scripts\activate            # Windows; on Linux: source .venv/bin/activate
pip install -r requirements.txt
python -X utf8 scripts/fetch_medqa.py data/raw/medqa_4opt_hf
```
Run every script from the repository root. `-X utf8` avoids Windows console encoding errors.

### GPU (Kaggle, 2 × T4 16 GB, internet on)
Generation needs vLLM on Linux GPUs. It was run only on Kaggle's free tier. Each notebook installs
`vllm==0.32.0`, removes Kaggle's clashing `torchaudio`/`torchao`, clones this repository at the commit set in its first
cell, downloads MedQA with checksums, runs the CPU tests, and then generates. Pinned versions:
[`requirements-gpu.txt`](requirements-gpu.txt); full environment of each run: `results/env/*/pip_freeze.txt`.

| Notebook | What it does | Kaggle run used for the paper |
|---|---|---|
| `00_environment_test.ipynb` | GPU, clone, data checksums, FP16 smoke test (transformers) | `lab7v2-00-environment-test` |
| `01_vllm_debug_run.ipynb` | vLLM on the 25 debug items | `lab7v2-01-vllm-debug-run` (v1 failed, v2 ok) |
| `02_run_experiment.ipynb` | protocol v1 arms: greedy, 10 sampled runs, 50-item greedy rerun; then evaluation. `SPLIT = "debug"` or `"test"` | `lab7v2-02-run-debug`, `lab7v2-03-run-test` |
| `03_run_extras.ipynb` | protocol v2 arms: full greedy rerun, T = 0.3 and 1.0, contamination probe | `lab7v2-04-run-extras` |

**How the runs were executed (Kaggle CLI 2.2.5).** For each run, the notebook was copied into its own folder, its
`COMMIT` cell was set to an exact commit hash, and a `kernel-metadata.json` was added next to it:
```json
{"id": "<your-kaggle-username>/lab7v2-03-run-test", "title": "lab7v2-03-run-test",
 "code_file": "02_run_experiment.ipynb", "language": "python", "kernel_type": "notebook",
 "is_private": true, "enable_gpu": true, "enable_internet": true,
 "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
```
```bash
kaggle kernels push -p <folder> --accelerator NvidiaTeslaT4 -t 14400
kaggle kernels status <your-kaggle-username>/lab7v2-03-run-test
kaggle kernels output <your-kaggle-username>/lab7v2-03-run-test -p results/env/kaggle_03_test_v1
```
Importing the notebook through the Kaggle web interface (Settings: GPU T4 × 2, Internet on) should work the same
way. **[TEST REQUIRED]**: that path was not used.

**Measured GPU time** (Kaggle session time from the run logs; generation time from `run_summary*.json`):

| Run | Session | Generation |
|---|---|---|
| 00 environment test | 0.12 h | — |
| 01 vLLM debug run, v1 (failed on a torchaudio clash) and v2 | 0.12 h + 0.18 h | — |
| 02 debug run of the full pipeline | 0.20 h | — |
| 03 test run (greedy + 10 sampled + 50 rerun, 633 items) | 0.76 h | 0.57 h |
| 04 exploratory run (rerun, T = 0.3, T = 1.0, probe) | 1.11 h | 0.95 h |
| **Total** | **2.49 h** | |

## Reproducing the study
### Level 1: recompute every statistic from the released parsed answers (CPU, about 1 minute; tested)
```bash
python -X utf8 scripts/recompute_from_item_level.py
```
Rebuilds `results/results.csv` from `results/item_level.csv` with the same code and seeds. Expected:
`50 result rows compared (rerun rows skipped); 0 differences`. The two greedy-rerun rows need raw outputs.

### Level 2: tests, item selection and paper (CPU + MedQA; tested)
```bash
for t in tests/test_*.py; do python -I -X utf8 $t; done       # bash; 7 files, all "N/N checks passed"
# PowerShell: Get-ChildItem tests\test_*.py | ForEach-Object { python -I -X utf8 $_.FullName }
python -X utf8 scripts/count_oncology_items.py data/raw/medqa_4opt_hf data/work
python -X utf8 scripts/make_splits.py data/screening codes_adjudicated.csv data/work/splits_check
python -X utf8 scripts/make_tables.py --results results
python -X utf8 scripts/make_paper_assets.py
```
- `count_oncology_items.py` rebuilds the 1,026-item keyword pool.
- `make_splits.py` rebuilds the split from the released codes. Its `debug_ids.txt` and `test_ids.txt` must equal
  `data/splits/`, with the SHA-256 values in `SPLIT_MANIFEST.json`.
- The screening codes themselves came from AI coders and cannot be regenerated by a script. The rules and both coders'
  codes are in `data/screening/`.
- To build the paper, run `pdflatex main`, `bibtex main` and `pdflatex main` twice in `paper/`. It needs a TeX
  distribution with `acmart`; it was built with TinyTeX (TeX Live 2026). Uploading `paper/` to Overleaf with pdfLaTeX
  should also work. **[TEST REQUIRED]** for Overleaf.

### Level 3: regenerate model outputs (Kaggle GPU, about 2 GPU-hours; executed by the author, not by a second person)
1. Run `02_run_experiment.ipynb` with `SPLIT = "test"` and `COMMIT` set to `7890add` (the frozen protocol) or to
   `v1.1` (same protocol and code).
2. Run `03_run_extras.ipynb` with the same commit.
3. Download the outputs. Put the `.jsonl` files of both runs in `results/raw/test/`, then:

```bash
python -X utf8 scripts/run_evaluation.py --split test --raw results/raw/test --out results
python -X utf8 scripts/run_extras_evaluation.py --split test --raw results/raw/test --out results
python -X utf8 scripts/audit_outputs.py --split test --raw results/raw/test --out results --env results/env/kaggle_03_test_v1 results/env/kaggle_04_extras_v1
python -X utf8 scripts/error_patterns.py --split test --raw results/raw/test --results results
python -X utf8 scripts/posthoc_review.py --split test --raw results/raw/test --results results
```
Sampled outputs are not bit-reproducible on GPUs, and greedy changed its letter on 1–4% of items between runs. New
outputs therefore give close, not identical, numbers. The released numbers come from the saved outputs of the runs
listed above.

## Minimum reproduction checklist for a second student
- [ ] Clone the repository and check out `v1.1`.
- [ ] Create a Python 3.12 virtual environment and `pip install -r requirements.txt`.
- [ ] `python -X utf8 scripts/fetch_medqa.py data/raw/medqa_4opt_hf`. Expect three `OK` lines (checksums match). On
      Windows, a notice from `huggingface_hub` about symlinks and Developer Mode is harmless.
- [ ] Run all `tests/test_*.py`. Expect every file to report "N/N checks passed". In `test_parse_vote.py`, part B prints
      `[SKIP]` because it needs a local-only file.
- [ ] `python -X utf8 scripts/recompute_from_item_level.py`. Expect `0 differences`.
- [ ] Rebuild the split (Level 2). `debug_ids.txt` and `test_ids.txt` should be identical to `data/splits/`.
- [ ] `python -X utf8 scripts/make_tables.py --results results` and `python -X utf8 scripts/make_paper_assets.py`.
      Then `git diff --stat` should show no change. On Windows, `git status` may still list files whose only
      difference is line endings.
- [ ] Optional: build the paper and compare it with `paper/main.pdf`.
- [ ] Optional, GPU: Level 3 on your own Kaggle account (needs your own `kaggle.json` token; never commit it).

## Limitations
- One 7B model, one prompt, one option order and one set of sampling seeds.
- Oncology relevance was screened by AI only (two instances of the same model; no clinician).
- Most test items come from MedQA's training split, so contamination cannot be ruled out.
- The model's answers show a shared position preference: wrong answers rarely pick option A.
- Several cited studies were read only as abstracts.
- Full list: Section 7 of the paper.

## Credits
- Data: MedQA (Jin et al., 2021, *Applied Sciences* 11(14):6421, doi:10.3390/app11146421), via
  `GBaker/MedQA-USMLE-4-options-hf`.
- Model: Qwen2.5-7B-Instruct (Qwen Team), revision `a09a35458c702b33eeacc393d103063234e8bc28`.
- Inference engine: vLLM (Kwon et al., SOSP 2023).
- Compute: Kaggle free tier.
- **AI assistance:** this study was carried out with extensive help from Claude Code (Anthropic, Claude Opus 5.5).
  It wrote the code, ran the GPU jobs through the Kaggle CLI, screened items, analysed the results and wrote the
  paper. The author chose the research gap, design and baseline and approved every protocol decision. See the
  paper's AI-use disclosure.

## Licence
Code: MIT ([`LICENSE`](LICENSE)). Paper, figures, tables, derived results and screening rules/codes: CC BY 4.0
([`LICENSE-docs.md`](LICENSE-docs.md)). MedQA is not redistributed.

## Citation
See [`CITATION.cff`](CITATION.cff). Sharma, V. (2026). *Does Asking Again Help? Majority Voting, Run-to-Run
Consistency and Agreement of a Small Open Language Model on Oncology Exam Questions.* Code and data release
`v1.1`, https://github.com/vibhum-phd/lab7-oncology-llm-voting.
