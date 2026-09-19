"""
Train the shared encoder on GISLR ASL. Offline; does not ship.

    python -m backend.training.train --epochs 40

Reports SIGNER-INDEPENDENT accuracy only -- see dataset.py for why.
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
from backend.app import config                                    # noqa: E402
from backend.app.model import SignClassifier                      # noqa: E402
from backend.training.dataset import (                            # noqa: E402
    GISLRDataset, load_rows, signer_independent_split,
)


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, lang, crit, opt, dev, train: bool):
    model.train(train)
    tot = correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        with torch.set_grad_enabled(train):
            logits = model(x, lang)
            loss = crit(logits, y)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        loss_sum += loss.detach().item() * len(y)
        correct += int((logits.argmax(-1) == y).sum())
        tot += len(y)
    return loss_sum / max(tot, 1), correct / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-signers", type=int, default=4)
    ap.add_argument("--lang", default="ase")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--with-rest", action="store_true",
                    help="include recorded __REST__ windows as a class")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = device()

    print("loading features ...", flush=True)
    rows = load_rows()
    if args.with_rest:
        from backend.training.dataset import load_rest_rows
        rest = load_rest_rows()
        if rest.empty:
            sys.exit("--with-rest but no rest windows; "
                     "run scripts/v010_record_rest.py first")
        print(f"including {len(rest):,} recorded __REST__ windows")
        rows = pd.concat([rows, rest], ignore_index=True)
    if rows.empty:
        sys.exit("no features on disk -- run backend.training.download first")
    labels = sorted(rows.sign.unique())
    label_map = {s: i for i, s in enumerate(labels)}
    tr_rows, va_rows, held = signer_independent_split(rows, args.val_signers, args.seed)

    print(f"device            {dev}", flush=True)
    print(f"sequences         {len(rows):,}  ({len(labels)} signs)")
    print(f"train / val       {len(tr_rows):,} / {len(va_rows):,}")
    print(f"HELD-OUT SIGNERS  {held}   <- signer-independent split")
    if va_rows.empty:
        sys.exit("validation split is empty -- need >1 signer")

    tr = DataLoader(GISLRDataset(tr_rows, label_map, True, args.seed),
                    batch_size=args.batch, shuffle=True, drop_last=True)
    va = DataLoader(GISLRDataset(va_rows, label_map, False),
                    batch_size=args.batch)

    model = SignClassifier({args.lang: len(labels)}).to(dev)
    print(f"params            {model.n_params()['total']:,}\n", flush=True)

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    out = Path(args.out) if args.out else config.MODELS_DIR / f"encoder_{args.lang}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    best, hist = 0.0, []

    for ep in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        trl, tra = run_epoch(model, tr, args.lang, crit, opt, dev, True)
        val, vaa = run_epoch(model, va, args.lang, crit, opt, dev, False)
        sched.step()
        hist.append({"epoch": ep, "train_loss": trl, "train_acc": tra,
                     "val_loss": val, "val_acc": vaa})
        flag = ""
        if vaa > best:
            best = vaa
            torch.save({"model": model.state_dict(), "labels": labels,
                        "lang": args.lang, "val_acc": vaa,
                        "held_out_signers": held}, out)
            flag = "  <- saved"
        print(f"  ep {ep:3d}/{args.epochs}  train {trl:.3f}/{tra:.3f}   "
              f"val {val:.3f}/{vaa:.3f}   {time.perf_counter()-t0:5.1f}s{flag}",
              flush=True)

    rep = config.OUTPUTS_DIR / f"train_{args.lang}_history.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(
        {"args": vars(args), "held_out_signers": held, "labels": labels,
         "best_signer_independent_acc": best, "history": hist}, indent=2))
    print(f"\nBEST SIGNER-INDEPENDENT ACC: {best:.4f}  ({best*100:.1f}%)")
    print(f"weights  {out}\nhistory  {rep}")


if __name__ == "__main__":
    main()
