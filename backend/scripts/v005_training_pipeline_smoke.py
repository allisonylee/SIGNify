"""
v005 -- exercise the whole Stage 2 pipeline on SYNTHETIC features.

The real GISLR download is blocked on Kaggle competition-rules acceptance, so
this generates fake .npy features with the exact shape and layout the real
pipeline produces, then runs augmentation, the signer-independent split, and a
short training run. Proves the plumbing before the data arrives.

Writes into a scratch dir and deletes it. Touches nothing real.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app import config
from backend.app.landmarks import SLICE_LEFT_HAND, SLICE_RIGHT_HAND
from backend.training import dataset as D

ok = []
def check(name, cond, detail=""):
    ok.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}   {detail}")

N_SIGNS, N_SIGNERS, PER = 8, 6, 12
T = config.WINDOW_FRAMES
scratch = Path(tempfile.mkdtemp(prefix="gislr_smoke_"))
D.FEAT = scratch / "features"; D.FEAT.mkdir(parents=True)

print("\n[1] synthesise features with the real shape/layout")
rng = np.random.default_rng(0)
rows = []
sid = 0
for s in range(N_SIGNS):
    base = rng.normal(0, 1, (53, 2)).astype(np.float32)
    for p in range(N_SIGNERS):
        for _ in range(PER):
            sid += 1
            t = np.linspace(0, 1, T)[:, None, None].astype(np.float32)
            w = np.zeros((T, 53, 3), np.float32)
            w[:, :, :2] = base[None] * (1 + 0.35 * t) + rng.normal(0, .05, (T, 53, 2))
            w[:, :, 2] = 1.0
            np.save(D.FEAT / f"{sid}.npy", w)
            rows.append({"sequence_id": sid, "sign": f"SIGN{s}",
                         "participant_id": 1000 + p})
idx = pd.DataFrame(rows)
check("features written", len(list(D.FEAT.glob('*.npy'))) == len(idx), f"{len(idx)} files")
check("shape is (T,53,3)", np.load(D.FEAT / "1.npy").shape == (T, 53, 3))

print("\n[2] augmentation preserves shape and mask")
w = np.load(D.FEAT / "1.npy")
for nm, fn in [("horizontal_flip", D.horizontal_flip),
               ("time_warp", lambda a: D.time_warp(a, 1.2)),
               ("affine", lambda a: D.affine(a, 8.0, 1.05, np.float32([.01, -.01]))),
               ("drop_hand", lambda a: D.drop_hand(a, "left"))]:
    o = fn(w)
    binary = set(np.unique(o[..., 2]).tolist()) <= {0.0, 1.0}
    check(f"{nm}: shape+mask", o.shape == w.shape and binary and np.isfinite(o).all())

print("\n[3] horizontal_flip actually SWAPS the hands")
f = D.horizontal_flip(w)
swapped = np.allclose(f[:, SLICE_LEFT_HAND, 1], w[:, SLICE_RIGHT_HAND, 1])
negated = np.allclose(f[:, SLICE_LEFT_HAND, 0], -w[:, SLICE_RIGHT_HAND, 0])
check("left<->right swapped (y matches)", swapped)
check("x negated", negated)

print("\n[4] drop_hand zeroes coords AND mask")
d = D.drop_hand(w, "left")
check("left hand mask cleared", d[:, SLICE_LEFT_HAND, 2].sum() == 0)
check("right hand untouched", np.allclose(d[:, SLICE_RIGHT_HAND], w[:, SLICE_RIGHT_HAND]))

print("\n[5] signer-independent split leaks no signer")
tr, va, held = D.signer_independent_split(idx, n_val_signers=2, seed=0)
overlap = set(tr.participant_id) & set(va.participant_id)
check("zero signer overlap", not overlap, f"held out {held}, overlap={overlap or 'none'}")
check("both splits non-empty", len(tr) > 0 and len(va) > 0, f"{len(tr)}/{len(va)}")

print("\n[6] DataLoader yields correct tensors")
lm = {s: i for i, s in enumerate(sorted(idx.sign.unique()))}
from torch.utils.data import DataLoader
dl = DataLoader(D.GISLRDataset(tr, lm, True, 0), batch_size=16, shuffle=True)
xb, yb = next(iter(dl))
check("batch shape", tuple(xb.shape) == (16, T, 53, 3), str(tuple(xb.shape)))
check("labels in range", int(yb.min()) >= 0 and int(yb.max()) < N_SIGNS)

print("\n[7] short training run actually learns")
from backend.app.model import SignClassifier
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(0)
m = SignClassifier({"ase": N_SIGNS}).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
tr_dl = DataLoader(D.GISLRDataset(tr, lm, True, 0), batch_size=32, shuffle=True, drop_last=True)
va_dl = DataLoader(D.GISLRDataset(va, lm, False), batch_size=32)

def acc(dl_):
    m.eval(); c = n = 0
    with torch.no_grad():
        for x, y in dl_:
            c += int((m(x.to(dev), "ase").argmax(-1) == y.to(dev)).sum()); n += len(y)
    return c / n

a0 = acc(va_dl)
for ep in range(8):
    m.train()
    for x, y in tr_dl:
        loss = crit(m(x.to(dev), "ase"), y.to(dev))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
a1 = acc(va_dl)
chance = 1.0 / N_SIGNS
print(f"      chance {chance:.3f} | before {a0:.3f} | after 8 epochs {a1:.3f}")
check("learns above chance", a1 > chance * 1.5, f"{a1:.3f} > {chance*1.5:.3f}")
check("improved from init", a1 > a0, f"{a0:.3f} -> {a1:.3f}")

shutil.rmtree(scratch, ignore_errors=True)
print(f"\n{'='*60}\n  {sum(ok)}/{len(ok)} checks passed  (scratch cleaned)\n{'='*60}")
sys.exit(0 if all(ok) else 1)
