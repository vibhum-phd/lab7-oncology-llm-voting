# Screening rules for MedQA oncology items (written 2026-11-01, before screening)

Pool: 1,026 MedQA 4-option items with an oncology term in the vignette (relative's history excluded)
or in the correct answer (`scripts/count_oncology_items.py`, rules R1 or R4). Each item gets one code.

Screener: Claude Code (AI), reading `screening_view.txt` (question sentence, correct answer, up to two
oncology-term contexts from the vignette). Check (as revised on 2026-11-01): the planned human check was
replaced by an independent blind AI re-screen of a random 20% sample from full text, because no
clinically trained checker was available; disagreements are adjudicated from full text. Agreement is
reported as % and Cohen's kappa in `SCREENING_AGREEMENT.md`.

## Codes

**M: oncology management (eligible; planned subgroup).** The question asks what to do for a patient's
known or suspected neoplasm, or for a direct complication of the neoplasm or its treatment: treatment
choice, drug, surgery or radiotherapy, next step in management or work-up of a suspected neoplasm,
monitoring during cancer therapy, supportive or palliative care for the cancer.

**O: oncology, other (eligible).** The concept tested is a neoplasm in this patient, or the correct
answer is a neoplasm: diagnosis, pathology, mechanism or genetics, risk factors, prognosis, cancer
screening, paraneoplastic syndromes, adverse effects or mechanisms of anticancer therapy.

**X: exclude.**
- The oncology term is incidental: past history, comorbidity, or one of several options that is not
  the answer, and the answer does not depend on the neoplasm.
- A drug named in the oncology list is used for a non-cancer purpose (e.g. methotrexate for
  rheumatoid arthritis or ectopic pregnancy, cyclophosphamide for lupus).
- The patient has cancer but the question tests something else (e.g. ethics, communication, an
  unrelated infection), unless the answer depends on the cancer or its treatment.
- Benign lesions not usually managed by oncology (uterine leiomyoma, lipoma, nevus, cyst, abscess).

**U: unsure.** Cannot be decided from the screening view; decided afterwards from the full item text
(`ai_unsure_decisions.csv`).

## Clarifications settled during screening (added 2026-11-01, after the first pass)
These came up while coding and were applied to the whole pool; they were also given to the
independent second screener.
- Statistics or epidemiology calculations, study-design and bias questions: **X**, even when the
  scenario is about cancer.
- Ethics, communication and consent questions: **X**, unless the answer is a cancer-care action
  (e.g. hospice eligibility, palliative pain control: **M**).
- Surgical anatomy or embryology after cancer surgery (which nerve was injured, which ligament is
  cut), and generic post-operative complications (hypocalcaemia after thyroidectomy, post-operative
  DVT): **X**.
- Infections and other complications caused by cancer treatment (neutropenic fever, tumour lysis,
  chemotherapy toxicity): oncology — **M** if the question asks what to do, **O** otherwise.
- Work-up of a suspected neoplasm (biopsy, imaging, colposcopy, FNA): **M**.
- Cancer screening and risk-factor questions: **O**, unless they are statistics questions.
- Cardiac myxoma counts as a benign lesion not managed by oncology: **X**.

## Scope of "neoplasm"
Malignant and premalignant neoplasms (including CIN, Barrett's dysplasia, MGUS, myelodysplasia), and
benign tumours usually handled by oncology or a tumour board (CNS tumours, neuroendocrine and endocrine
tumours such as pheochromocytoma, insulinoma, pituitary adenoma).
