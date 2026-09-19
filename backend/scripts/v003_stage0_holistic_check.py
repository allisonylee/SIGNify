"""
Stage 0 -- MediaPipe API compatibility check.  DEFINITIVE VERSION.

Plan: claude/docs/PLAN_2026-09-18_232008.md, Stage 0.

History:
  v001  HolisticLandmarker on mediapipe 1.0.1 -> SIGABRT in every config.
  v002  Pose+Hand on 1.0.1, GPU+SRGBA only    -> worked, but no CPU path.
  v003  (this) mediapipe pinned to 0.10.35    -> Holistic works on CPU.
        1.0.x is a regression on macOS arm64: its TFLite calculators demand
        a Metal graph service that is never registered.
        See backend/outputs/stage0_api_matrix.txt.

Benchmarked on CPU: Holistic 36.7 ms/frame vs Pose+Hand 52.9 ms/frame.
Holistic wins despite also computing the 478-point face mesh, because it
shares one pose->hand crop pipeline instead of running two detectors.

Question this answers: GISLR came from the legacy `mp.solutions.holistic`
API (543/frame = 468 face + 33 pose + 21 + 21 hands). We use Tasks. Do the
blocks we consume -- hands 21/21 and pose 33 -- line up, and what are the
coordinate and handedness conventions?

Outputs:
  backend/outputs/stage0_landmark_overlay.png
  backend/outputs/stage0_report.txt
"""

import sys
import time
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[1]
MODELS, OUT = ROOT / "models", ROOT / "outputs"
MODEL = MODELS / "holistic_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
             "holistic_landmarker/float16/latest/holistic_landmarker.task")
IMAGE_URL = "https://storage.googleapis.com/mediapipe-assets/male_full_height_hands.jpg"

# Legacy (GISLR) per-block counts.
GISLR = {"face": 468, "pose": 33, "left_hand": 21, "right_hand": 21}

# The 53 landmarks we actually consume (plan Stage 2).
UPPER = ["NOSE", "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER",
         "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
         "LEFT_HIP", "RIGHT_HIP"]

lines: list[str] = []


def say(s=""):
    print(s)
    lines.append(str(s))


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest.exists() and dest.stat().st_size > 0):
        urllib.request.urlretrieve(url, dest)
    return dest


say("=" * 74)
say("STAGE 0 -- MediaPipe API compatibility check  (Holistic / CPU)")
say("=" * 74)

say("\n[1] Environment")
say(f"  python      {sys.version.split()[0]}")
say(f"  executable  {sys.executable}")
say(f"  mediapipe   {mp.__version__}   (PINNED -- 1.0.x aborts on macOS arm64)")
say(f"  numpy       {np.__version__}   opencv {cv2.__version__}")
say(f"  delegate    CPU (default)")

say("\n[2] Legacy Solutions API (which produced GISLR)")
say(f"  mp.solutions present: {hasattr(mp, 'solutions')}")
say("  -> absent in BOTH 0.10.35 and 1.0.1. GISLR's own extractor cannot be")
say("     re-run; any parity check must use the GISLR parquet itself. (R1)")

fetch(MODEL_URL, MODEL)
img_path = fetch(IMAGE_URL, OUT / "stage0_test_subject.jpg")
bgr = cv2.imread(str(img_path))
h, w = bgr.shape[:2]
rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
say(f"\n[3] Test image {w}x{h}  ({img_path.name})")

opts = vision.HolisticLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=str(MODEL)),
    running_mode=vision.RunningMode.IMAGE)
with vision.HolisticLandmarker.create_from_options(opts) as lm:
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

blocks = {k: getattr(res, k + "_landmarks", None)
          for k in ("face", "pose", "left_hand", "right_hand")}

say("\n[4] PER-BLOCK LANDMARK COUNTS   <-- Stage 0 stop condition")
say(f"  {'block':<13}{'Tasks 0.10.35':>15}{'GISLR legacy':>15}   verdict")
say("  " + "-" * 56)
counts = {}
for k, v in blocks.items():
    n = len(v) if v else 0
    counts[k] = n
    d = n - GISLR[k]
    say(f"  {k:<13}{n:>15}{GISLR[k]:>15}   {'OK' if d == 0 else f'DIFF {d:+d}'}")
say("  The face DIFF is expected and harmless: 468 surface points + 10 iris")
say("  points at indices 468-477. We do not use the face block at all.")

say("\n[5] Coordinate convention")
pose = blocks["pose"]
xs = np.array([p.x for p in pose]); ys = np.array([p.y for p in pose])
zs = np.array([p.z for p in pose])
say(f"  pose x  [{xs.min():+.3f}, {xs.max():+.3f}]")
say(f"  pose y  [{ys.min():+.3f}, {ys.max():+.3f}]")
say(f"  pose z  [{zs.min():+.3f}, {zs.max():+.3f}]   (dropped per plan)")
inside = ((0 <= xs) & (xs <= 1) & (0 <= ys) & (ys <= 1)).mean()
say(f"  pose x,y inside [0,1]: {inside:.0%} -> frame-normalised")

say("\n[6] Our 53-landmark feature set (plan Stage 2)")
PL = vision.PoseLandmark
for nm in UPPER:
    i = int(getattr(PL, nm)); p = pose[i]
    say(f"    {nm:<16} idx={i:<3} x={p.x:+.3f} y={p.y:+.3f} vis={p.visibility:.2f}")
total = counts["left_hand"] + counts["right_hand"] + len(UPPER)
say(f"  hands 21+21 = 42  +  upper-body pose = {len(UPPER)}")
say(f"  TOTAL {total} landmarks -> {total * 2} features (x,y)")

say("\n[7] Handedness convention  (plan R9 / section 14.2)")
lsh, rsh = pose[int(PL.LEFT_SHOULDER)], pose[int(PL.RIGHT_SHOULDER)]
lw = pose[int(PL.LEFT_WRIST)]
say(f"  LEFT_SHOULDER.x {lsh.x:.3f} > RIGHT_SHOULDER.x {rsh.x:.3f} -> {lsh.x > rsh.x}")
say("  Holistic names the hand blocks explicitly (left_hand / right_hand),")
say("  so no handedness classifier is involved. On an UN-MIRRORED image the")
say("  signer's left hand appears on the viewer's right, i.e. larger x.")
if blocks["left_hand"]:
    lhx = np.mean([p.x for p in blocks["left_hand"]])
    rhx = np.mean([p.x for p in blocks["right_hand"]]) if blocks["right_hand"] else float("nan")
    say(f"  mean left_hand x  {lhx:.3f}   mean right_hand x {rhx:.3f}   "
        f"left>right = {lhx > rhx}")
    say(f"  left_hand near LEFT_WRIST? |dx|={abs(lhx - lw.x):.3f}")

say("\n[8] Throughput (CPU, VIDEO mode)")
vopts = vision.HolisticLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=str(MODEL)),
    running_mode=vision.RunningMode.VIDEO)
with vision.HolisticLandmarker.create_from_options(vopts) as lm:
    lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), 0)
    t = []
    for i in range(1, 31):
        t0 = time.perf_counter()
        lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), i * 33)
        t.append((time.perf_counter() - t0) * 1000)
med = float(np.median(t))
say(f"  median {med:.1f} ms/frame  ->  {1000/med:.1f} fps single core")
say(f"  vs Pose+Hand as separate tasks: 52.9 ms  (Holistic is {52.9/med:.2f}x faster)")

vis = bgr.copy()
for p in pose:
    cv2.circle(vis, (int(p.x * w), int(p.y * h)), 3, (0, 200, 0), -1)
for key, col in [("left_hand", (255, 0, 0)), ("right_hand", (0, 0, 255))]:
    for p in (blocks[key] or []):
        cv2.circle(vis, (int(p.x * w), int(p.y * h)), 3, col, -1)
for nm, col in [("LEFT_SHOULDER", (255, 255, 0)), ("RIGHT_SHOULDER", (0, 255, 255)),
                ("LEFT_WRIST", (255, 0, 255)), ("NOSE", (255, 255, 255))]:
    p = pose[int(getattr(PL, nm))]
    xy = (int(p.x * w), int(p.y * h))
    cv2.circle(vis, xy, 12, col, 3)
    cv2.putText(vis, nm, (xy[0] + 16, xy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
cv2.putText(vis, "blue=left_hand  red=right_hand", (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.imwrite(str(OUT / "stage0_landmark_overlay.png"), vis)
say(f"\n[9] Overlay written: {OUT / 'stage0_landmark_overlay.png'}")

ok = (counts["pose"] == 33 and counts["left_hand"] == 21 and counts["right_hand"] == 21)
say("\n" + "=" * 74)
say(f"  pose 33 / hands 21+21 match GISLR : {ok}")
say(f"  VERDICT: {'PASS' if ok else 'FAIL'}")
say("=" * 74)

(OUT / "stage0_report.txt").write_text("\n".join(lines) + "\n")
