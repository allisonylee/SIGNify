"""
Train only the LSC head on a frozen GISLR encoder.

Reads encoder_ase.pt (does not modify it). Writes encoder_csn.pt.

    python -m backend.trainingCSN.train_head --epochs 30
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
from backend.app.model import SignClassifier  # noqa: E402 — architecture only
from backend.trainingCSN.dataset import (  # noqa: E402
    IsolatedSignDataset, load_label_list, load_rows, signer_independent_split,
)
from backend.trainingCSN.paths import CK_ASE, CK_CSN, LANG, OUTPUTS, REST_SIGN  # noqa: E402


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, crit, opt, dev, train: bool):
    model.encoder.eval()
    model.heads[LANG].train(train)
    tot = correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        with torch.set_grad_enabled(train):
            logits = model(x, LANG)
            loss = crit(logits, y)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.heads[LANG].parameters(), 5.0)
            opt.step()
        loss_sum += loss.detach().item() * len(y)
        correct += int((logits.argmax(-1) == y).sum())
        tot += len(y)
    return loss_sum / max(tot, 1), correct / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-signers", type=int, default=1,
                    help="LSC50 has 5 volunteers; hold out 1 by default")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    if not CK_ASE.exists():
        sys.exit(f"missing ASL encoder {CK_ASE} — Stage 2 checkpoint required")

    rows = load_rows()
    if rows.empty:
        sys.exit("no LSC50 features — run backend.trainingCSN.download_lsc50")

    labels = load_label_list()
    present = set(rows.sign.unique())
    labels = [s for s in labels if s in present]
    if REST_SIGN not in labels and REST_SIGN in present:
        labels.append(REST_SIGN)

    rest = rows[rows.sign == REST_SIGN]
    signs = rows[rows.sign != REST_SIGN]
    cap = max(len(signs) // 4, 50)
    if len(rest) > cap:
        rest = rest.sample(n=cap, random_state=args.seed)
        rows = pd.concat([signs, rest], ignore_index=True)
        print(f"rest capped        {len(rest)} (was majority class)", flush=True)
    label_map = {s: i for i, s in enumerate(labels)}

    tr_rows, va_rows, held = signer_independent_split(rows, args.val_signers, args.seed)
    if va_rows.empty:
        sys.exit("validation split empty — need >1 participant_id")

    print(f"device            {device()}", flush=True)
    print(f"sequences         {len(rows):,}  ({len(labels)} classes)", flush=True)
    print(f"train / val       {len(tr_rows):,} / {len(va_rows):,}", flush=True)
    print(f"HELD-OUT SIGNERS  {held}   <- LSC50 volunteer ids", flush=True)

    drop_last = len(tr_rows) >= args.batch * 2
    tr = DataLoader(IsolatedSignDataset(tr_rows, label_map, True, args.seed),
                    batch_size=min(args.batch, max(len(tr_rows), 1)),
                    shuffle=True, drop_last=drop_last)
    va = DataLoader(IsolatedSignDataset(va_rows, label_map, False),
                    batch_size=min(args.batch, max(len(va_rows), 1)))

    ase = torch.load(CK_ASE, map_location="cpu", weights_only=False)
    n_ase = len(ase["labels"])
    donor = SignClassifier({"ase": n_ase})
    donor.load_state_dict(ase["model"])

    model = SignClassifier({LANG: len(labels)})
    model.encoder.load_state_dict(donor.encoder.state_dict())
    model.freeze_encoder(True)
    for p in model.encoder.parameters():
        p.requires_grad = False
    dev = device()
    model.to(dev)
    print(f"params trainable  {sum(p.numel() for p in model.heads[LANG].parameters()):,}",
          flush=True)

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.heads[LANG].parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best, hist = 0.0, []
    CK_CSN.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        trl, tra = run_epoch(model, tr, crit, opt, dev, True)
        val, vaa = run_epoch(model, va, crit, opt, dev, False)
        sched.step()
        hist.append({"epoch": ep, "train_loss": trl, "train_acc": tra,
                     "val_loss": val, "val_acc": vaa})
        flag = ""
        if vaa >= best:
            best = vaa
            torch.save({
                "model": model.state_dict(),
                "labels": labels,
                "lang": LANG,
                "val_acc": vaa,
                "held_out_signers": held,
                "encoder_from": str(CK_ASE.name),
                "note": "LSC head on frozen GISLR encoder. Not loaded by app/.",
            }, CK_CSN)
            flag = "  <- saved"
        print(f"  ep {ep:3d}/{args.epochs}  train {trl:.3f}/{tra:.3f}   "
              f"val {val:.3f}/{vaa:.3f}   {time.perf_counter()-t0:5.1f}s{flag}",
              flush=True)

    rep = OUTPUTS / "train_csn_history.json"
    rep.write_text(json.dumps({
        "args": vars(args), "held_out_signers": held, "labels": labels,
        "best_signer_independent_acc": best, "history": hist,
        "n_train": int(len(tr_rows)), "n_val": int(len(va_rows)),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nBEST SIGNER-INDEPENDENT ACC: {best:.4f}  ({best*100:.1f}%)")
    print(f"weights  {CK_CSN}\nhistory  {rep}")
    print("NOT wired into backend/app — load encoder_csn.pt later via add_head.")


if __name__ == "__main__":
    main()
