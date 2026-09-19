"""
v003 — idle / rest check on INCLUDE features (no webcam required).

Scores still-looking rest windows vs a batch of real signs. Does not start the
FastAPI app.

    python backend/scriptsISL/v003_idle_ins.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app.model import SignClassifier
from backend.trainingISL.dataset import load_rows
from backend.trainingISL.paths import CK_INS, FEAT, LANG, REST_SIGN


def main():
    if not CK_INS.exists():
        sys.exit(f"no {CK_INS}")
    ck = torch.load(CK_INS, map_location="cpu", weights_only=False)
    labels = ck["labels"]
    model = SignClassifier({LANG: len(labels)})
    model.load_state_dict(ck["model"])
    model.eval()
    rows = load_rows()
    rest = rows[rows.sign == REST_SIGN]
    signs = rows[rows.sign != REST_SIGN]
    if rest.empty:
        sys.exit("no rest features — re-run download_include so padding rest is saved")

    def pred(sid):
        w = np.load(FEAT / f"{int(sid)}.npy")
        x = torch.from_numpy(w[None].astype(np.float32))
        with torch.no_grad():
            p = torch.softmax(model(x, LANG)[0], -1)
        i = int(p.argmax())
        return labels[i], float(p[i])

    n_ok = 0
    for sid in rest.sequence_id.head(80):
        name, p = pred(sid)
        n_ok += int(name == REST_SIGN)
        print(f"  rest clip {int(sid)} -> {name} {p:.2f}")
    print(f"rest classified as rest: {n_ok}/{min(80, len(rest))}")

    n_sign_as_rest = 0
    k = min(80, len(signs))
    for sid in signs.sequence_id.head(k):
        name, p = pred(sid)
        n_sign_as_rest += int(name == REST_SIGN)
    print(f"real signs classified as rest: {n_sign_as_rest}/{k}")


if __name__ == "__main__":
    main()
