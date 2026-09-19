"""Paths for LSC50 / CSN work. Does not write into backend/training."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app import config  # noqa: E402  — read-only import

REPO = Path(__file__).resolve().parents[2]
DATA = config.ROOT / "data" / "lsc50"
FEAT = DATA / "features"
TMP = DATA / "tmp"
META = DATA / "meta"
MANIFEST = DATA / "extract_manifest.jsonl"
INDEX = DATA / "subset_index.parquet"
LABELS = Path(__file__).resolve().parent / "labels_csn50.json"
CK_ASE = config.MODELS_DIR / "encoder_ase.pt"
CK_CSN = config.MODELS_DIR / "encoder_csn.pt"
OUTPUTS = config.OUTPUTS_DIR / "csn"

MIN_FREE_GB = 1.5
# LANDMARKS.zip is ~2.77 GB; require this only before the download itself.
MIN_FREE_GB_DOWNLOAD = 4.0
REST_SIGN = "__REST__"
LANG = "csn"

FIGSHARE_ARTICLE = 27383016
FIGSHARE_API = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE}"
LANDMARKS_FILE_ID = 50117670
LANDMARKS_URL = f"https://ndownloader.figshare.com/files/{LANDMARKS_FILE_ID}"
LANDMARKS_ZIP_NAME = "LANDMARKS.zip"


def ensure_dirs():
    for p in (DATA, FEAT, TMP, META, OUTPUTS, config.MODELS_DIR):
        p.mkdir(parents=True, exist_ok=True)


def free_gb(path: Path = DATA) -> float:
    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free / 1e9


def abort_if_low_disk(minimum: float = MIN_FREE_GB):
    g = free_gb()
    if g < minimum:
        raise SystemExit(f"disk {g:.2f} GB free < {minimum} GB — abort")
    return g
