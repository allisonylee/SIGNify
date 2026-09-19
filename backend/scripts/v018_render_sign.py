"""
v018 -- render a GISLR sample as a video. NOT SHIPPED.

GISLR contains NO video, only MediaPipe landmarks. This draws the skeleton back
out so you can see how the corpus signers actually perform a sign -- useful for
matching their style, since our recordings measured ~2x their motion magnitude.

    python backend/scripts/v018_render_sign.py hello
    python backend/scripts/v018_render_sign.py flower --n 3 --compare

--compare also renders YOUR recording of the same word beside it.
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app import config

DATA = config.ROOT / "data" / "gislr"
OUT = config.OUTPUTS_DIR
ARCHIVE = Path.home() / "Downloads" / "asl-signs.zip"
W = H = 480
FPS = 15

HAND_EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),
              (10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),
              (17,18),(18,19),(19,20)]
POSE_EDGES = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24)]

# Our 53-landmark feature layout: 21 left hand, 21 right hand, 11 upper pose.
# POSE_UPPER_NAMES order: NOSE MOUTH_L MOUTH_R L_SH R_SH L_EL R_EL L_WR R_WR L_HIP R_HIP
FEAT_POSE_EDGES = [(3,4),(3,5),(5,7),(4,6),(6,8),(3,9),(4,10),(9,10),(0,1),(0,2)]


def draw_feature_frame(w, size=(W, H), title=""):
    """
    Draw one (53,3) NORMALISED frame -- i.e. exactly what the model sees.

    Coordinates are body-relative: the shoulder midpoint is the origin and the
    unit is one shoulder width, so shoulders sit at x = +/-0.5. Map that back
    to pixels with a fixed scale so clips are directly comparable regardless of
    how far the signer sat from the camera.
    """
    img = np.full((size[1], size[0], 3), 22, np.uint8)
    cx, cy, sc = size[0] // 2, int(size[1] * 0.42), size[0] * 0.30
    det = w[:, 2] > 0.5
    def px(i):
        return (int(cx + w[i, 0] * sc), int(cy + w[i, 1] * sc))

    P = 42                                   # pose block offset
    for a, b in FEAT_POSE_EDGES:
        if det[P + a] and det[P + b]:
            cv2.line(img, px(P + a), px(P + b), (90, 150, 90), 2)
    for i in range(P, 53):
        if det[i]:
            cv2.circle(img, px(i), 3, (120, 220, 120), -1)
    for off, col in ((0, (240, 160, 60)), (21, (80, 80, 240))):
        if not det[off:off + 21].any():
            continue
        for a, b in HAND_EDGES:
            if det[off + a] and det[off + b]:
                cv2.line(img, px(off + a), px(off + b), col, 2)
        for i in range(off, off + 21):
            if det[i]:
                cv2.circle(img, px(i), 3, (255, 255, 255), -1)
    if title:
        cv2.putText(img, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (220, 220, 220), 1)
    return img


def render_mine(sign, n, fps):
    """Render OUR recordings of `sign` from the stored feature arrays."""
    idx_path = DATA / "mine_index.parquet"
    if not idx_path.exists():
        sys.exit("no recordings found")
    d = pd.read_parquet(idx_path)
    have = {int(f.stem) for f in (DATA / "mine").glob("*.npy")}
    d = d[(d.sign == sign) & (d.sequence_id.isin(have))]
    if d.empty:
        sys.exit(f"you have no recordings of '{sign}' "
                 f"(you have: {sorted(pd.read_parquet(idx_path).sign.unique())})")
    picks = d.head(n)
    arrs = [(int(r.participant_id), np.load(DATA / "mine" / f"{int(r.sequence_id)}.npy"))
            for r in picks.itertuples()]
    out = OUT / f"mine_{sign}.mp4"
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (W * len(arrs), H))
    T = arrs[0][1].shape[0]
    print(f"your '{sign}': {len(arrs)} clips, {T} frames each (resampled)")
    for i in range(T):
        tiles = [draw_feature_frame(a[i], title=f"YOURS {sign}  signer {pid-700000}"
                                                f"  f{i+1}/{T}")
                 for pid, a in arrs]
        vw.write(np.hstack(tiles))
    vw.release()
    print(f"wrote {out}")
    print(f"open it:  open {out}")


def draw_frame(g, size=(W, H), title=""):
    img = np.full((size[1], size[0], 3), 22, np.uint8)
    def pts(kind):
        b = g[g.type == kind].sort_values("landmark_index")
        return b[["x", "y"]].to_numpy(np.float32)
    def to_px(p):
        return (int(np.clip(p[0], 0, 1) * size[0]), int(np.clip(p[1], 0, 1) * size[1]))

    pose = pts("pose")
    if len(pose) >= 25 and not np.isnan(pose[:25]).all():
        for a, b in POSE_EDGES:
            if a < len(pose) and b < len(pose) and not (np.isnan(pose[a]).any()
                                                        or np.isnan(pose[b]).any()):
                cv2.line(img, to_px(pose[a]), to_px(pose[b]), (90, 150, 90), 2)
        for p in pose[:25]:
            if not np.isnan(p).any():
                cv2.circle(img, to_px(p), 3, (120, 220, 120), -1)
    for kind, col in (("left_hand", (240, 160, 60)), ("right_hand", (80, 80, 240))):
        h = pts(kind)
        if len(h) == 21 and not np.isnan(h).all():
            for a, b in HAND_EDGES:
                if not (np.isnan(h[a]).any() or np.isnan(h[b]).any()):
                    cv2.line(img, to_px(h[a]), to_px(h[b]), col, 2)
            for p in h:
                if not np.isnan(p).any():
                    cv2.circle(img, to_px(p), 3, (255, 255, 255), -1)
    if title:
        cv2.putText(img, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (220, 220, 220), 1)
    return img


def gislr_sequences(sign, n):
    idx = pd.read_csv(DATA / "train.csv")
    sub = idx[idx.sign == sign]
    if sub.empty:
        sys.exit(f"'{sign}' is not one of the 250 GISLR signs")
    if not ARCHIVE.exists():
        sys.exit(f"archive not found: {ARCHIVE}")
    picks = sub.sample(min(n, len(sub)), random_state=0)
    out = []
    with zipfile.ZipFile(ARCHIVE) as z:
        for r in picks.itertuples():
            df = pd.read_parquet(io.BytesIO(z.read(r.path)),
                                 columns=["frame", "type", "landmark_index", "x", "y"])
            out.append((int(r.participant_id), df))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sign")
    ap.add_argument("--n", type=int, default=2, help="how many GISLR samples")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--as-features", action="store_true",
                    help="draw GISLR through the same 53-landmark pipeline as "
                         "--mine, so the two are directly comparable")
    ap.add_argument("--mine", action="store_true",
                    help="render YOUR recordings instead of GISLR's")
    a = ap.parse_args()

    if a.mine:
        render_mine(a.sign, a.n, a.fps)
        return

    if a.as_features:
        # Draw GISLR through the SAME 53-landmark pipeline as our recordings,
        # so the two videos are directly comparable. Without this, GISLR looks
        # richer purely because the raw parquet had more points available --
        # a rendering artefact, not something the model sees.
        import numpy as _np
        gi = pd.read_parquet(DATA / "subset_index.parquet")
        sub = gi[gi.sign == a.sign]
        if sub.empty:
            sys.exit(f"'{a.sign}' not in the converted GISLR features")
        pack = _np.load(DATA / "features_all.npy", mmap_mode="r")
        ids = _np.load(DATA / "features_ids.npy")
        pos = {int(v): i for i, v in enumerate(ids)}
        picks = [(int(r.participant_id), pos[int(r.sequence_id)])
                 for r in sub.itertuples() if int(r.sequence_id) in pos][:a.n]
        arrs = [(pid, _np.asarray(pack[row])) for pid, row in picks]
        out = OUT / f"gislr_{a.sign}_features.mp4"
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             a.fps, (W * len(arrs), H))
        T = arrs[0][1].shape[0]
        print(f"GISLR '{a.sign}' as FEATURES: {len(arrs)} clips, {T} frames")
        for i in range(T):
            vw.write(np.hstack([
                draw_feature_frame(arr[i], title=f"GISLR {a.sign}  signer {pid}"
                                                 f"  f{i+1}/{T}")
                for pid, arr in arrs]))
        vw.release()
        print(f"wrote {out}\nopen it:  open {out}")
        return

    seqs = gislr_sequences(a.sign, a.n)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"gislr_{a.sign}.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                         a.fps, (W * len(seqs), H))
    n_frames = max(s[1].frame.nunique() for s in seqs)
    print(f"'{a.sign}': {len(seqs)} samples, "
          f"{[s[1].frame.nunique() for s in seqs]} frames")

    grouped = [(pid, {f: g for f, g in df.groupby("frame")},
                sorted(df.frame.unique())) for pid, df in seqs]
    for i in range(n_frames):
        tiles = []
        for pid, by_f, frames in grouped:
            j = min(i, len(frames) - 1)
            tiles.append(draw_frame(by_f[frames[j]],
                                    title=f"{a.sign}  signer {pid}  f{j+1}/{len(frames)}"))
        vw.write(np.hstack(tiles))
    vw.release()
    print(f"wrote {path}  ({n_frames} frames at {a.fps} fps = "
          f"{n_frames/a.fps:.1f}s)")
    print(f"open it:  open {path}")


if __name__ == "__main__":
    main()
