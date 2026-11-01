"""Download the three MedQA parquet files from Hugging Face and verify their SHA-256 (see data/medqa_source.md).

Usage (from repo/): python scripts/fetch_medqa.py data/raw/medqa_4opt_hf
Needs internet and huggingface_hub. Refuses to keep a file whose checksum differs.
"""
import hashlib
import shutil
import sys
from pathlib import Path

REPO, REVISION = "GBaker/MedQA-USMLE-4-options-hf", "refs/convert/parquet"
SHA256 = {
    "train": "616c13662bb281c57c299c998759e6206bba47365cd37ecb4d52e66712ec34ec",
    "validation": "841a5ffcddd415c9e28c0fd7f412e9ce4ece51c0e705aed7b0bfa324c916be95",
    "test": "debf8e08bce918806287597194c585648dd2b040fea34405f04a72a0dc6e270e",
}


def main(out_dir: str) -> int:
    from huggingface_hub import hf_hub_download
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, want in SHA256.items():
        dest = out / f"{split}.parquet"
        if not dest.exists():
            src = hf_hub_download(REPO, f"default/{split}/0000.parquet", repo_type="dataset", revision=REVISION)
            shutil.copyfile(src, dest)
        got = hashlib.sha256(dest.read_bytes()).hexdigest()
        if got != want:
            dest.unlink()
            print(f"{split}: checksum MISMATCH ({got}); file removed")
            return 1
        print(f"{split}: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
