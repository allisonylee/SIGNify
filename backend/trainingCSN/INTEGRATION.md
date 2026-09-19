# Later integration (do not do this in the CSN isolation pass)

`encoder_csn.pt` is a `SignClassifier({"csn": N})` state dict plus `labels`.
`encoder_ase.pt` is left unchanged.

To wire later (edits would be in `backend/app`, which isolation forbids now):

1. Load both checkpoints.
2. `model = SignClassifier({"ase": n_ase, "csn": n_csn})`.
3. Copy `encoder.*` from either file (they should match; prefer `encoder_ase.pt`).
4. Copy `heads.ase.*` from `encoder_ase.pt` and `heads.csn.*` from `encoder_csn.pt`.
5. Pass `cfg["sign_language"]` into `model(window, lang)` with `sign_language: "csn"`.
6. Map `__REST__` to no emit.

Until those edits exist, use `backend/scriptsCSN/v004_live_csn.py` only.
