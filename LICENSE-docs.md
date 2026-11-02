# Licence for documents and data produced in this project

| Material | Licence |
|---|---|
| Code (`src/`, `scripts/`, `tests/`, `notebooks/`, `configs/`) | MIT — see [`LICENSE`](LICENSE) |
| Paper (`paper/`), figures, tables and derived results (`results/`), screening rules and codes (`data/screening/`) | Creative Commons Attribution 4.0 International (CC BY 4.0), © 2026 Vibhum Sharma — <https://creativecommons.org/licenses/by/4.0/> |

**Third-party material is not covered by these licences.**

- **MedQA question text and options are not included in this repository.** They come from MedQA (Jin et al., 2021),
  as copied to Hugging Face in `GBaker/MedQA-USMLE-4-options-hf`, whose dataset card carries the tag `cc-by-sa-4.0`.
  The original MedQA repository does not state a data licence (see [`data/medqa_source.md`](data/medqa_source.md)).
  Download the data from its source.
- The files `data/splits/*`, `data/screening/*codes*.csv` and `results/item_level.csv` contain MedQA item identifiers,
  and `results/item_level.csv` also contains the keyed answer letter of each item. They are derived from MedQA and are
  provided for reproducibility. Reusers should check the dataset's terms.
- Model outputs are not included; the model (Qwen2.5-7B-Instruct) is not redistributed and is used under its own licence.
