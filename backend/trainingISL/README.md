# trainingISL / scriptsISL — Indian Sign Language (INCLUDE-50)

Isolated from the shipping backend. **Do not edit** `backend/app`,
`backend/scripts`, `backend/training`, or `frontend`.

## What this produces

`backend/models/encoder_ins.pt` — frozen GISLR encoder + new `ins` head.
`app/` does **not** load this file. How to wire it later is in
`INTEGRATION.md` in this folder.

## Commands

```
python backend/scriptsISL/v001_preflight.py
python -m backend.trainingISL.download_include --calibrate 20
python -m backend.trainingISL.download_include
python -m backend.trainingISL.train_head --epochs 30
python backend/scriptsISL/v002_evaluate_ins.py
python backend/scriptsISL/v003_idle_ins.py
python backend/scriptsISL/v004_live_ins.py
```

## Honesty

- Isolated INCLUDE-50 recognition, not ISL translation.
- `participant_id` is an MVI-number proxy (7 bins), not named signers.
- INCLUDE is Chennai / St. Louis School for the Deaf, not pan-Indian.
- License: CC BY 4.0 (Zenodo 4010759).
