"""
Fine-tune on your own recorded signs -- FROZEN ENCODER, head only. Offline.

Approach A: the encoder learned hand shapes and motion primitives from 94,477
GISLR sequences and stays fixed; only the classification head adapts. ~326k
trainable params instead of 2.3M, so a few hundred of your samples cannot
overfit it into uselessness, and the encoder cannot be damaged.

    python -m backend.training.finetune --dry-run      # inspect, train nothing
    python -m backend.training.finetune --epochs 30

CLASS IMBALANCE IS THE REAL PROBLEM HERE. Ten of your samples against ~380
GISLR samples for the same word is 2.6% -- gradient noise. `--mine-repeat`
oversamples yours so they actually influence the head.

TWO SEPARATE NUMBERS ARE REPORTED, and both matter:
  GISLR held-out signers  -- did we BREAK what already worked?
  your held-out samples   -- did we FIX what did not?
Optimising one while wrecking the other is the failure mode.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app import config
from backend.app.model import SignClassifier
from backend.training.dataset import (
    GISLRDataset, MINE_SIGNER_BASE, load_rest_rows, load_rows,
    mine_signer_id, signer_independent_split,
)
from backend.training import dataset as D

DATA = config.ROOT / "data" / "gislr"
MINE, MINE_INDEX = DATA / "mine", DATA / "mine_index.parquet"
CKPT = config.MODELS_DIR / "encoder_ase.pt"


def load_mine() -> pd.DataFrame:
    if not MINE_INDEX.exists():
        return pd.DataFrame(columns=["sequence_id", "sign", "participant_id"])
    df = pd.read_parquet(MINE_INDEX)
    have = {int(p.stem) for p in MINE.glob("*.npy")}
    return df[df.sequence_id.isin(have)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mine-repeat", type=int, default=20,
                    help="oversample factor for your samples")
    ap.add_argument("--holdout-signer", type=int, default=None,
                    help="hold out this signer of YOURS for honest evaluation")
    ap.add_argument("--holdout-frac", type=float, default=0.25,
                    help="if only one signer, hold out this fraction instead")
    ap.add_argument("--gislr-cap", type=int, default=0,
                    help="cap GISLR samples per class (0 = all)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    mine = load_mine()
    if mine.empty:
        sys.exit("no recordings -- run scripts/v016_record_signs.py first")
    gis = load_rows()
    rest = load_rest_rows()

    print(f"YOUR data   : {len(mine)} samples, {mine.sign.nunique()} words, "
          f"signers {sorted(mine.participant_id.unique())}")
    for w, c in mine.sign.value_counts().items():
        print(f"    {w:<16} {c:>3}")
    print(f"GISLR       : {len(gis):,} samples")
    print(f"rest        : {len(rest):,} windows")

    # --- hold out some of YOUR data, honestly if possible -------------------
    signers = sorted(mine.participant_id.unique())
    if args.holdout_signer is not None:
        # accept either "1" or the namespaced 700001
        if args.holdout_signer < MINE_SIGNER_BASE:
            args.holdout_signer = mine_signer_id(args.holdout_signer)
    if args.holdout_signer is not None and args.holdout_signer in signers:
        mine_va = mine[mine.participant_id == args.holdout_signer]
        mine_tr = mine[mine.participant_id != args.holdout_signer]
        how = f"signer {args.holdout_signer} held out (signer-independent)"
    elif len(signers) > 1:
        hold = signers[-1]
        mine_va = mine[mine.participant_id == hold]
        mine_tr = mine[mine.participant_id != hold]
        how = f"signer {hold} held out (signer-independent)"
    else:
        shuf = mine.sample(frac=1.0, random_state=args.seed)
        k = max(1, int(len(shuf) * args.holdout_frac))
        mine_va, mine_tr = shuf.iloc[:k], shuf.iloc[k:]
        how = (f"random {k}/{len(mine)} held out -- ONLY ONE SIGNER, so this "
               f"number is OPTIMISTIC")
    print(f"\nholdout     : {how}")
    print(f"              train {len(mine_tr)}  /  val {len(mine_va)}")

    gis_tr, gis_va, held = signer_independent_split(gis, 4, args.seed)
    if args.gislr_cap:
        gis_tr = (gis_tr.sample(frac=1.0, random_state=args.seed)
                        .groupby("sign", sort=False).head(args.gislr_cap))
    if len(rest):
        r = rest.sample(frac=1.0, random_state=args.seed)
        k = int(len(r) * 0.2)
        gis_va = pd.concat([gis_va, r.iloc[:k]], ignore_index=True)
        gis_tr = pd.concat([gis_tr, r.iloc[k:]], ignore_index=True)

    train = pd.concat([gis_tr] + [mine_tr] * args.mine_repeat, ignore_index=True)
    share = len(mine_tr) * args.mine_repeat / max(len(train), 1)
    print(f"GISLR holdout: signers {held}, {len(gis_va):,} samples")
    print(f"train set   : {len(train):,} rows "
          f"(yours repeated {args.mine_repeat}x = {share:.1%} of it)")

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    labels = list(ck["labels"])
    n_orig = len(labels)

    model = SignClassifier({"ase": n_orig})
    model.load_state_dict(ck["model"]); model.to(dev)

    # WORDS THE MODEL HAS NEVER SEEN
    # Grow the head instead of refusing. expand_head copies the trained weights
    # so every existing class is bit-identical at initialisation; only the new
    # rows are fresh. A brand-new head would discard all 250 GISLR signs.
    new_words = sorted(set(mine.sign) - set(labels))
    if new_words:
        print(f"\nNEW WORDS not in the model's {n_orig} classes: {new_words}")
        for w in new_words:
            n = int((mine.sign == w).sum())
            note = "" if n >= 8 else "   <- thin; more reps recommended"
            print(f"    {w:<16} {n:>3} samples (YOURS ONLY, no GISLR){note}")
        labels += new_words
        model.expand_head("ase", len(new_words))
        model.to(dev)
        print(f"  head expanded {n_orig} -> {len(labels)} classes")

    lm = {s: i for i, s in enumerate(labels)}
    model.freeze_encoder(True)                      # <-- APPROACH A
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"\nfrozen encoder: training {n_tr:,} of {n_all:,} params "
          f"({n_tr/n_all:.1%})")

    if args.dry_run:
        print("\n--dry-run: nothing trained, no file written")
        return

    def loader(rows, train_mode):
        return DataLoader(GISLRDataset(rows, lm, train_mode, args.seed),
                          batch_size=args.batch, shuffle=train_mode,
                          drop_last=train_mode)
    tr_dl, gis_dl, mine_dl = loader(train, True), loader(gis_va, False), loader(mine_va, False)

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    if new_words:
        print(f"  note: new classes appear ONLY in your data, so the GISLR\n"
              f"        holdout number cannot measure them -- only 'yours' can.")

    def acc(dl):
        model.eval(); c = n = 0
        with torch.no_grad():
            for x, y in dl:
                c += int((model(x.to(dev), "ase").argmax(-1) == y.to(dev)).sum())
                n += len(y)
        return c / max(n, 1)

    base_g, base_m = acc(gis_dl), acc(mine_dl)
    print(f"\nBEFORE  GISLR holdout {base_g:.4f}   yours {base_m:.4f}\n")

    out = Path(args.out) if args.out else config.MODELS_DIR / "encoder_ase_finetuned.pt"
    best, hist = -1.0, []
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.perf_counter()
        for x, y in tr_dl:
            loss = crit(model(x.to(dev), "ase"), y.to(dev))
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(trainable, 5.0); opt.step()
        sched.step()
        g, mm = acc(gis_dl), acc(mine_dl)
        # Score on BOTH: helping yours while wrecking GISLR is not a win.
        score = mm - max(0.0, base_g - g) * 2.0
        hist.append({"epoch": ep, "gislr": g, "mine": mm, "score": score})
        flag = ""
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "labels": labels,
                        "lang": "ase", "val_acc": g, "mine_acc": mm,
                        "held_out_signers": held,
                        "finetuned_on": sorted(mine.sign.unique()),
                        "new_words": new_words}, out)
            flag = "  <- saved"
        print(f"  ep {ep:3d}/{args.epochs}  GISLR {g:.4f} "
              f"({g-base_g:+.4f})   yours {mm:.4f} ({mm-base_m:+.4f}){flag}",
              flush=True)

    rep = config.OUTPUTS_DIR / "finetune_history.json"
    rep.write_text(json.dumps({"args": vars(args), "base_gislr": base_g,
                               "base_mine": base_m, "history": hist}, indent=2))
    print(f"\nBASELINE untouched at {config.MODELS_DIR/'encoder_ase.BASELINE.pt'}")
    print(f"fine-tuned weights -> {out}")
    print(f"history -> {rep}")
    print("\nTo use it:  cp %s %s" % (out, CKPT))


if __name__ == "__main__":
    main()
