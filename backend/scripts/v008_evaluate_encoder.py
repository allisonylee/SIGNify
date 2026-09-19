"""
v008 -- evaluate the trained encoder on held-out signers.

"Best val acc" printed during training is not proof: it could be a fluke epoch,
the checkpoint could be broken, or the accuracy could be concentrated in a few
classes. Reload the saved weights and look at the numbers.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app import config
from backend.app.model import SignClassifier
from backend.training.dataset import GISLRDataset, load_rows, signer_independent_split

CK = config.MODELS_DIR / "encoder_ase.pt"
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

ck = torch.load(CK, map_location="cpu", weights_only=False)
labels = ck["labels"]
held = ck["held_out_signers"]
print(f"checkpoint      {CK.name}  ({CK.stat().st_size/1e6:.1f} MB)")
print(f"labels          {len(labels)}")
print(f"held-out signers {held}")
print(f"val_acc recorded {ck['val_acc']:.4f}\n")

model = SignClassifier({"ase": len(labels)})
model.load_state_dict(ck["model"])
model.to(dev).eval()

rows = load_rows()
label_map = {s: i for i, s in enumerate(labels)}
_, va, held2 = signer_independent_split(rows, len(held), 0)
assert sorted(held2) == sorted(held), f"split mismatch {held2} vs {held}"
print(f"val sequences   {len(va):,}  from signers {sorted(va.participant_id.unique())}")

dl = DataLoader(GISLRDataset(va, label_map, train=False), batch_size=128)
top1 = top5 = n = 0
per_class = Counter(); per_class_tot = Counter()
t0 = time.perf_counter()
with torch.no_grad():
    for x, y in dl:
        lg = model(x.to(dev), "ase").cpu()
        k5 = lg.topk(5, dim=-1).indices
        pred = k5[:, 0]
        top1 += int((pred == y).sum())
        top5 += int((k5 == y[:, None]).any(-1).sum())
        n += len(y)
        for t, p in zip(y.tolist(), pred.tolist()):
            per_class_tot[t] += 1
            if t == p:
                per_class[t] += 1
el = time.perf_counter() - t0

print(f"\n{'='*58}")
print(f"  TOP-1 (signer-independent) : {top1/n:.4f}   ({top1:,}/{n:,})")
print(f"  TOP-5                      : {top5/n:.4f}")
print(f"  chance                     : {1/len(labels):.4f}   -> {top1/n*len(labels):.0f}x")
print(f"{'='*58}")
print(f"  recorded during training   : {ck['val_acc']:.4f}  "
      f"{'MATCHES' if abs(top1/n - ck['val_acc']) < 0.02 else 'MISMATCH'}")

accs = np.array([per_class[i]/per_class_tot[i] for i in sorted(per_class_tot)])
names = [labels[i] for i in sorted(per_class_tot)]
print(f"\nper-class accuracy: mean {accs.mean():.3f}  median {np.median(accs):.3f}")
print(f"  classes at 0%: {(accs==0).sum()}/{len(accs)}   above 80%: {(accs>0.8).sum()}")
order = np.argsort(accs)
print("\n  worst 8:", ", ".join(f"{names[i]}={accs[i]:.2f}" for i in order[:8]))
print("  best  8:", ", ".join(f"{names[i]}={accs[i]:.2f}" for i in order[-8:]))

print(f"\ninference throughput: {n/el:,.0f} seq/s on {dev} (batched)")
x1 = torch.randn(1, 32, 53, 3, device=dev)
with torch.no_grad():
    for _ in range(5): model(x1, "ase")
    t = []
    for _ in range(30):
        s = time.perf_counter(); model(x1, "ase")
        if dev.type == "mps": torch.mps.synchronize()
        t.append((time.perf_counter()-s)*1000)
print(f"single-sequence latency: {np.median(t):.1f} ms  <- the live-pipeline number")
