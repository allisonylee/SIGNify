# Backend

## Layout rule

**Code is separated by whether it ships in the final app.**

| folder | ships? | what goes here |
|---|---|---|
| `app/` | **YES** | Everything the running app needs: FastAPI server, WS endpoint, ingest adapters, feature extraction, model definition, inference, LLM and TTS clients. |
| `training/` | no | Offline work: dataset download, landmark extraction, the training loop, evaluation. Produces weights that `app/` loads. |
| `scripts/` | no | One-off verification, probes, benchmarks, and the test harness. Disposable. Named `vNNN_description.py`. |
| `data/` | no | Datasets and metadata. Gitignored where large. |
| `models/` | no (gitignored) | Downloaded MediaPipe bundles + our trained weights. |
| `outputs/` | no (gitignored) | Generated reports, plots, logs. |

Anything written purely to check something belongs in `scripts/`, never in `app/`.

## The one deliberate exception

`training/` **imports feature code from `app/`** rather than duplicating it.

Feature extraction — which landmarks we keep, how we normalise, how we resample
to 32 frames — must be byte-identical between training and inference. If those
two drift apart you get train/serve skew, which does not throw, does not show up
in validation, and only appears as "the model is mysteriously bad live."

So `app/features.py` is the single source of truth and training imports it.
Never copy-paste a normalisation constant between the two.

## Environment

Python 3.14 venv at the repo root. `pip install -r requirements.txt`.

**Do not upgrade mediapipe past 0.10.35** — see the note in `requirements.txt`
and `outputs/stage0_api_matrix.txt`.
