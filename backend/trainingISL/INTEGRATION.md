# Later integration (do not do this in the ISL isolation pass)

`encoder_ins.pt` is a `SignClassifier({"ins": N})` state dict plus `labels`.
`encoder_ase.pt` is left unchanged.

To wire later (edits would be in `backend/app`, which isolation forbids now):

1. Load both checkpoints.
2. `model = SignClassifier({"ase": n_ase, "ins": n_ins})`.
3. Copy `encoder.*` from either file (they should match; prefer `encoder_ase.pt`).
4. Copy `heads.ase.*` from `encoder_ase.pt` and `heads.ins.*` from `encoder_ins.pt`.
5. Pass `cfg["sign_language"]` into `model(window, lang)`.
6. Map `__REST__` to no emit.

Until those edits exist, use `backend/scriptsISL/v004_live_ins.py` only.
