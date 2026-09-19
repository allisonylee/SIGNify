"""
v002 — evaluate encoder_csn.pt on held-out LSC50 volunteers. Does not touch app/.

    python backend/scriptsCSN/v002_evaluate_csn.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app.model import SignClassifier
from backend.trainingCSN.dataset import IsolatedSignDataset, load_rows, signer_independent_split
from backend.trainingCSN.paths import CK_CSN, LANG, OUTPUTS


def main():
    if not CK_CSN.exists():
        sys.exit(f"no {CK_CSN} — run train_head first")
    ck = torch.load(CK_CSN, map_location="cpu", weights_only=False)
    labels = ck["labels"]
    held = ck.get("held_out_signers") or []
    print(f"checkpoint {CK_CSN.name}  classes {len(labels)}  recorded val {ck.get('val_acc')}")
    print(f"held-out volunteers {held}")

    rows = load_rows()
    label_map = {s: i for i, s in enumerate(labels)}
    rows = rows[rows.sign.isin(label_map)].reset_index(drop=True)
    _, va, held2 = signer_independent_split(rows, max(len(held), 1), 0)
    print(f"split check held={held2} (seed 0); using checkpoint held-out if present")
    if held:
        va = rows[rows.participant_id.isin(held)].reset_index(drop=True)
    rest = va[va.sign == "__REST__"]
    signs = va[va.sign != "__REST__"]
    print(f"val sequences {len(va)}  signs {len(signs)}  rest {len(rest)}")
    print("NOTE: rest is easy and plentiful; signs-only top-1 is the number to quote")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        dev = torch.device("mps")
    model = SignClassifier({LANG: len(labels)})
    model.load_state_dict(ck["model"])
    model.to(dev).eval()

    dl = DataLoader(IsolatedSignDataset(va, label_map, train=False), batch_size=64)
    top1 = top5 = n = 0
    per_ok, per_tot = Counter(), Counter()
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    with torch.no_grad():
        for x, y in dl:
            lg = model(x.to(dev), LANG).cpu()
            k5 = lg.topk(min(5, len(labels)), dim=-1).indices
            pred = k5[:, 0]
            top1 += int((pred == y).sum())
            top5 += int((k5 == y[:, None]).any(-1).sum())
            n += len(y)
            for t, p in zip(y.tolist(), pred.tolist()):
                per_tot[t] += 1
                cm[t, p] += 1
                if t == p:
                    per_ok[t] += 1
    if n == 0:
        sys.exit("empty val set")
    accs = np.array([per_ok[i] / per_tot[i] if per_tot[i] else 0.0
                     for i in range(len(labels))])
    print(f"\nTOP-1 (signer-independent): {top1/n:.4f}  ({top1}/{n})")
    print(f"TOP-5: {top5/n:.4f}   chance {1/len(labels):.4f}")
    names = labels
    order = np.argsort(accs)
    named = [(names[i], accs[i], per_tot[i]) for i in order if per_tot[i]]
    print("worst:", ", ".join(f"{a}={b:.2f}(n={c})" for a, b, c in named[:8]))
    print("best:", ", ".join(f"{a}={b:.2f}(n={c})" for a, b, c in named[-8:]))
    rest_i = labels.index("__REST__") if "__REST__" in labels else None
    if rest_i is not None:
        print(f"rest predicted as rest: {cm[rest_i, rest_i]}/{per_tot[rest_i]}")
        print(f"non-rest predicted as rest: {cm[:, rest_i].sum() - cm[rest_i, rest_i]}")
        sign_n = n - per_tot[rest_i]
        sign_ok = top1 - cm[rest_i, rest_i]
        if sign_n:
            print(f"SIGNS-ONLY top-1: {sign_ok/sign_n:.4f}  ({sign_ok}/{sign_n})")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS / "eval_csn.json"
    out.write_text(json.dumps({
        "top1": top1 / n, "top5": top5 / n, "n": n, "labels": labels,
        "per_class": {labels[i]: accs[i] for i in range(len(labels))},
        "confusion": cm.tolist(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(OUTPUTS / "confusion_csn.npy", cm)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
