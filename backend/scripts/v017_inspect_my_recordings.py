"""
v017 -- sanity-check your own recorded signs before training on them.

Recording 20 words is ~20 minutes. Checking one word takes seconds, so check
first. Compares YOUR samples against the GISLR samples of the same word and
reports what the current model already predicts for them.

    python backend/scripts/v017_inspect_my_recordings.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app import config, features
from backend.app.landmarks import SLICE_LEFT_HAND, SLICE_POSE, SLICE_RIGHT_HAND
from backend.app.model import SignClassifier

DATA = config.ROOT / "data" / "gislr"
MINE, INDEX = DATA / "mine", DATA / "mine_index.parquet"

if not INDEX.exists():
    sys.exit("no recordings yet -- run v016_record_signs.py first")
mine = pd.read_parquet(INDEX)
mine = mine[mine.sequence_id.isin({int(p.stem) for p in MINE.glob("*.npy")})]
if mine.empty:
    sys.exit("index exists but no .npy files found")

print(f"{len(mine)} samples, {mine.sign.nunique()} word(s), "
      f"signer(s) {sorted(mine.participant_id.unique())}\n")

ck = torch.load(config.MODELS_DIR / "encoder_ase.pt", map_location="cpu",
                weights_only=False)
labels = ck["labels"]
m = SignClassifier({"ase": len(labels)}); m.load_state_dict(ck["model"]); m.eval()

gis_idx = pd.read_parquet(DATA / "subset_index.parquet")
pack = np.load(DATA / "features_all.npy", mmap_mode="r")
gis_ids = np.load(DATA / "features_ids.npy")
pos = {int(v): i for i, v in enumerate(gis_ids)}

ok = []
def check(n, c, d=""):
    ok.append(bool(c)); print(f"  [{'PASS' if c else 'FAIL'}] {n}   {d}")

for word in sorted(mine.sign.unique()):
    sub = mine[mine.sign == word]
    arrs = np.stack([np.load(MINE / f"{int(s)}.npy") for s in sub.sequence_id])
    print(f"=== {word}  ({len(arrs)} of your samples) ===")

    print("  [shape / sanity]")
    check("shape (N,32,53,3)", arrs.shape[1:] == (config.WINDOW_FRAMES, 53, 3),
          str(arrs.shape))
    check("no NaN/Inf", np.isfinite(arrs).all())
    check("mask is binary", set(np.unique(arrs[..., 2]).tolist()) <= {0.0, 1.0})
    check("samples differ from each other",
          float(np.abs(arrs[:, None] - arrs[None]).mean()) > 1e-6)

    lh = arrs[:, :, SLICE_LEFT_HAND, 2].mean()
    rh = arrs[:, :, SLICE_RIGHT_HAND, 2].mean()
    po = arrs[:, :, SLICE_POSE, 2].mean()
    print(f"  [detection]  pose {po:.0%}   left hand {lh:.0%}   right hand {rh:.0%}")
    check("pose almost always present", po > 0.9, f"{po:.0%}")
    check("at least one hand usually present", max(lh, rh) > 0.4,
          f"max {max(lh,rh):.0%}")

    # compare with GISLR for the same word
    g = gis_idx[gis_idx.sign == word]
    if len(g):
        grows = [pos[int(s)] for s in g.sequence_id.head(200) if int(s) in pos]
        garr = np.asarray(pack[sorted(grows)])
        gl = garr[:, :, SLICE_LEFT_HAND, 2].mean()
        gr = garr[:, :, SLICE_RIGHT_HAND, 2].mean()
        print(f"  [GISLR same word, n={len(garr)}]  left {gl:.0%}  right {gr:.0%}")
        mine_xy = arrs[arrs[..., 2] > 0.5][:, :2]
        gis_xy = garr[garr[..., 2] > 0.5][:, :2]
        print(f"  [coord spread] yours  x {mine_xy[:,0].std():.2f}  y {mine_xy[:,1].std():.2f}")
        print(f"                 GISLR  x {gis_xy[:,0].std():.2f}  y {gis_xy[:,1].std():.2f}")
        check("your spread is within 2x of GISLR's",
              0.5 < (mine_xy.std() / max(gis_xy.std(), 1e-6)) < 2.0,
              f"ratio {mine_xy.std()/max(gis_xy.std(),1e-6):.2f}")

    print("  [what the CURRENT model predicts for your samples]")
    with torch.no_grad():
        p = m(torch.from_numpy(arrs.astype(np.float32)), "ase").softmax(-1)
    top = p.argmax(-1).tolist()
    names = [labels[i] for i in top]
    correct = sum(n == word for n in names)
    conf = p.max(-1).values.mean().item()
    tgt = labels.index(word) if word in labels else None
    print(f"    predicted: {dict(pd.Series(names).value_counts())}")
    print(f"    correct {correct}/{len(names)}   mean confidence {conf:.2f}")
    if tgt is not None:
        print(f"    mean prob assigned to '{word}': {p[:, tgt].mean():.3f}")
        rank = (p > p[:, tgt:tgt+1]).sum(-1).float().mean().item() + 1
        print(f"    mean rank of '{word}': {rank:.1f} of {len(labels)}")
    print()

print(f"{'='*62}\n  {sum(ok)}/{len(ok)} structural checks passed\n{'='*62}")
print("\nReading the result:")
print("  structural checks fail  -> recording/segmentation problem, fix before")
print("                             recording more")
print("  checks pass, model wrong but the right word ranks high")
print("                          -> fine-tuning should work well")
print("  checks pass, right word ranks near random")
print("                          -> your signing differs a lot from GISLR;")
print("                             expect to need more samples per word")
