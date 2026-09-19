"""Paths for INCLUDE / ISL work. Does not write into backend/training."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app import config  # noqa: E402  — read-only import

REPO = Path(__file__).resolve().parents[2]
DATA = config.ROOT / "data" / "include"
FEAT = DATA / "features"
TMP = DATA / "tmp"
META = DATA / "meta"
MANIFEST = DATA / "extract_manifest.jsonl"
INDEX = DATA / "subset_index.parquet"
LABELS = Path(__file__).resolve().parent / "labels_ins50.json"
CK_ASE = config.MODELS_DIR / "encoder_ase.pt"
CK_INS = config.MODELS_DIR / "encoder_ins.pt"
OUTPUTS = config.OUTPUTS_DIR / "ins"

MIN_FREE_GB = 1.5
REST_SIGN = "__REST__"
LANG = "ins"
HF_BASE = "https://huggingface.co/datasets/ai4bharat/INCLUDE/resolve/main"
ZENODO_API = "https://zenodo.org/api/records/4010759"


def ensure_dirs():
    for p in (DATA, FEAT, TMP, META, OUTPUTS, config.MODELS_DIR):
        p.mkdir(parents=True, exist_ok=True)


def free_gb(path: Path = DATA) -> float:
    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free / 1e9


def abort_if_low_disk():
    g = free_gb()
    if g < MIN_FREE_GB:
        raise SystemExit(f"disk {g:.2f} GB free < {MIN_FREE_GB} GB — abort")
    return g
