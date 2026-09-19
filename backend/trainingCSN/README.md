# trainingCSN / scriptsCSN — Colombian Sign Language (LSC50)

Isolated from the shipping backend. **Do not edit** `backend/app`,
`backend/scripts`, `backend/training`, or `frontend`.

## What this produces

`backend/models/encoder_csn.pt` — frozen GISLR encoder + new `csn` head.
`app/` does **not** load this file. How to wire it later is in
`INTEGRATION.md` in this folder.

## Data

[LSC50](https://doi.org/10.1038/s41597-024-04172-5) (CC BY 4.0 on Figshare;
paper also notes CC BY-NC-ND for the article itself). 50 isolated LSC signs,
5 volunteers, pose + hand MediaPipe CSVs. We do **not** download the 6 GB
video archive or the IMU zip.

Spanish glosses are the class names (Table 2 of the data descriptor).

## Commands

```
python backend/scriptsCSN/v001_preflight.py
python -m backend.trainingCSN.download_lsc50 --calibrate 20
python -m backend.trainingCSN.download_lsc50
python -m backend.trainingCSN.train_head --epochs 30
python backend/scriptsCSN/v002_evaluate_csn.py
python backend/scriptsCSN/v003_idle_csn.py
python backend/scriptsCSN/v004_live_csn.py
```

If `LANDMARKS.zip` is already on disk:

```
python -m backend.trainingCSN.download_lsc50 --zip-dir PATH\TO\DIR
```

## Honesty

- Isolated LSC50 recognition, not LSC translation.
- 5 volunteers (3 native, 2 non-native). Signer-independent split holds out
  volunteer 4. **Quote signs-only:** 32.5% top-1 on 200 held-out sign clips
  (50 classes, chance 2%). Mixed val with rest is higher because rest is easy.
- LSC50 landmark zip has **1,000** pose+hand pairs (50×5×4), not the 4,000
  video files. Face CSVs and IMU are unused.
- Landmarks were extracted by the dataset authors with MediaPipe Holistic,
  then mapped to our 53-landmark tensor. Live inference still uses pinned
  `mediapipe==0.10.35` Tasks Holistic.
- Clip padding is almost always still (dataset protocol). Rest windows are
  capped at 1/4 of sign count so `__REST__` cannot dominate training.
