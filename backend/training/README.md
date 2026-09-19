# training/ -- offline, does not ship

Dataset preparation and model training. Produces weights in `../models/`,
which `../app/` loads at runtime.

    download.py    Kaggle GISLR fetch; streaming download-extract-delete for
                   video datasets (bounded temp dir, resume manifest).
    extract.py     Video -> MediaPipe landmarks -> .npy. Parallel across cores.
    dataset.py     Parquet/npy loading, signer-independent splits, augmentation.
    train.py       Training loop.
    evaluate.py    Signer-independent accuracy, confusion matrix.

**Imports `app.features`** for normalisation and resampling. Do not reimplement
them here -- see the note in ../README.md.
