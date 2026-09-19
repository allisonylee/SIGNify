"""
v001 — Step 0 preflight for ISL. Read-only against shipping trees.

    python backend/scriptsISL/v001_preflight.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app import config
from backend.app.landmarks import N_CHANNELS, N_LANDMARKS
from backend.trainingISL.paths import CK_ASE, DATA, FEAT, LABELS, abort_if_low_disk, ensure_dirs, free_gb


def main():
    print("step 0 preflight (ISL isolation)")
    print(f"  encoder_ase.pt     {CK_ASE.exists()}  {CK_ASE}")
    if CK_ASE.exists():
        import torch
        ck = torch.load(CK_ASE, map_location="cpu", weights_only=False)
        print(f"    ase labels {len(ck['labels'])}  val_acc {ck.get('val_acc')}")
        print(f"    held-out ASL signers {ck.get('held_out_signers')}")
    print(f"  tensor spec        ({config.WINDOW_FRAMES}, {N_LANDMARKS}, {N_CHANNELS})")
    print(f"  window seconds     {config.WINDOW_SECONDS} (live); train resamples full clip span=1.0")
    ensure_dirs()
    print(f"  free disk          {free_gb():.2f} GB")
    abort_if_low_disk()
    print(f"  include data dir   {DATA}")
    print(f"  features           {len(list(FEAT.glob('*.npy'))) if FEAT.exists() else 0}")
    print(f"  labels file        {LABELS.exists()}  {LABELS}")
    print("  forbidden writes: backend/app, backend/scripts, backend/training, frontend")
    print("  allowed writes:   backend/trainingISL, backend/scriptsISL, backend/models, backend/data/include")
    if not CK_ASE.exists():
        sys.exit("BLOCKED: no encoder_ase.pt")
    print("preflight OK")


if __name__ == "__main__":
    main()
