"""
Stage 0 -- MediaPipe API compatibility check.

Plan: claude/docs/PLAN_2026-09-18_232008.md, Stage 0.

Question this answers: GISLR landmarks were produced by the LEGACY
`mp.solutions.holistic` API (MediaPipe 0.9.0.1, 543 landmarks/frame =
468 face + 33 pose + 21 + 21 hands). We will run the Tasks API
`HolisticLandmarker`. Do the blocks we care about -- hands (21/21) and pose
(33) -- line up, and what is the coordinate convention?

We deliberately do NOT use face landmarks (plan Stage 2), so the known
468-vs-478 face discrepancy should be irrelevant. This script verifies that.

Outputs:
  backend/outputs/stage0_landmark_overlay.png   visual index check
  backend/outputs/stage0_report.txt             the numbers
"""

import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "holistic_landmarker.task"
OUT = ROOT / "outputs"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)
# full-height subject with both hands visible -- exercises pose AND both hands
IMAGE_URL = "https://storage.googleapis.com/mediapipe-assets/male_full_height_hands.jpg"

# What the GISLR (legacy) layout had, per block.
GISLR_EXPECTED = {"face": 468, "pose": 33, "left_hand": 21, "right_hand": 21}

lines: list[str] = []


def say(s=""):
    print(s)
    lines.append(str(s))


def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        say(f"  cached  {dest.name}  ({dest.stat().st_size:,} B)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    say(f"  fetching {dest.name} ...")
    urllib.request.urlretrieve(url, dest)
    say(f"  got     {dest.name}  ({dest.stat().st_size:,} B)")
    return dest


say("=" * 72)
say("STAGE 0 -- MediaPipe API compatibility check")
say("=" * 72)

say("\n[1] Environment")
say(f"  python     {sys.version.split()[0]}  ({sys.executable})")
say(f"  mediapipe  {mp.__version__}")
say(f"  numpy      {np.__version__}")
say(f"  opencv     {cv2.__version__}")

say("\n[2] Legacy API availability")
legacy = hasattr(mp, "solutions")
say(f"  mp.solutions present: {legacy}")
if not legacy:
    say("  -> FINDING: the legacy Solutions API (which produced GISLR) is GONE")
    say("     in this version. We cannot reproduce GISLR's exact extractor.")
    say("     Comparison must be against the GISLR parquet itself.")

say("\n[3] Assets")
fetch(MODEL_URL, MODEL)
img_path = OUT / "stage0_test_subject.jpg"
fetch(IMAGE_URL, img_path)

say("\n[4] Run HolisticLandmarker (Tasks API, RunningMode.IMAGE)")
BaseOptions = mp.tasks.BaseOptions
V = mp.tasks.vision
opts = V.HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL)),
    running_mode=V.RunningMode.IMAGE,
)
bgr = cv2.imread(str(img_path))
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
h, w = bgr.shape[:2]
say(f"  test image: {w}x{h}")

with V.HolisticLandmarker.create_from_options(opts) as lm:
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

say("\n[5] Result object surface")
fields = [f for f in dir(res) if not f.startswith("_")]
say(f"  {fields}")

blocks = {
    "face": getattr(res, "face_landmarks", None),
    "pose": getattr(res, "pose_landmarks", None),
    "left_hand": getattr(res, "left_hand_landmarks", None),
    "right_hand": getattr(res, "right_hand_landmarks", None),
}

say("\n[6] PER-BLOCK LANDMARK COUNTS  <-- the Stage 0 stop condition")
say(f"  {'block':<12} {'Tasks 1.0.1':>12} {'GISLR legacy':>14}   match")
say("  " + "-" * 50)
counts = {}
for name, val in blocks.items():
    n = 0 if not val else len(val)
    counts[name] = n
    exp = GISLR_EXPECTED[name]
    mark = "OK" if n == exp else f"DIFF ({n - exp:+d})"
    say(f"  {name:<12} {n:>12} {exp:>14}   {mark}")

say("\n[7] Coordinate convention")
pose = blocks["pose"]
if pose:
    xs = np.array([p.x for p in pose])
    ys = np.array([p.y for p in pose])
    zs = np.array([p.z for p in pose])
    say(f"  pose x range  [{xs.min():+.3f}, {xs.max():+.3f}]")
    say(f"  pose y range  [{ys.min():+.3f}, {ys.max():+.3f}]")
    say(f"  pose z range  [{zs.min():+.3f}, {zs.max():+.3f}]")
    inside = ((0 <= xs) & (xs <= 1) & (0 <= ys) & (ys <= 1)).mean()
    say(f"  fraction of pose x,y inside [0,1]: {inside:.1%}")
    say("  -> normalised to image frame" if inside > 0.8 else "  -> NOT frame-normalised")

say("\n[8] Index semantics -- are named pose indices where we expect?")
PL = V.PoseLandmark
for nm in ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_WRIST", "RIGHT_WRIST"]:
    idx = int(getattr(PL, nm))
    p = pose[idx]
    say(f"  {nm:<16} idx={idx:<3} x={p.x:+.3f} y={p.y:+.3f}")

lsh, rsh = pose[int(PL.LEFT_SHOULDER)], pose[int(PL.RIGHT_SHOULDER)]
nose = pose[int(PL.NOSE)]
say(f"\n  shoulder width (normalised units): {abs(lsh.x - rsh.x):.4f}")
say(f"  nose above shoulders?              {nose.y < min(lsh.y, rsh.y)}")
say("  NOTE: MediaPipe LEFT_* = the SUBJECT's left, which appears on the")
say("        RIGHT side of an un-mirrored image. Expect LEFT_SHOULDER.x >")
say(f"        RIGHT_SHOULDER.x -> actual: {lsh.x:.3f} > {rsh.x:.3f} = {lsh.x > rsh.x}")

say("\n[9] Our 53-landmark feature set (plan Stage 2)")
UPPER = ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
         "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP"]
say(f"  hands            {counts['left_hand']} + {counts['right_hand']} = "
    f"{counts['left_hand'] + counts['right_hand']}")
say(f"  upper-body pose  {len(UPPER)}  {UPPER}")
total = counts["left_hand"] + counts["right_hand"] + len(UPPER)
say(f"  TOTAL            {total} landmarks -> {total * 2} features (x,y)")
say(f"  face block used: NO  (so the {counts['face']}-vs-468 gap is irrelevant)")

# ---- overlay for visual index confirmation ----
vis = bgr.copy()
for name, val, col in [("pose", blocks["pose"], (0, 255, 0)),
                       ("left_hand", blocks["left_hand"], (255, 0, 0)),
                       ("right_hand", blocks["right_hand"], (0, 0, 255))]:
    if not val:
        continue
    for p in val:
        cv2.circle(vis, (int(p.x * w), int(p.y * h)), 4, col, -1)
for nm, col in [("LEFT_SHOULDER", (255, 255, 0)), ("RIGHT_SHOULDER", (0, 255, 255)),
                ("LEFT_WRIST", (255, 0, 255)), ("NOSE", (255, 255, 255))]:
    p = pose[int(getattr(PL, nm))]
    xy = (int(p.x * w), int(p.y * h))
    cv2.circle(vis, xy, 11, col, 3)
    cv2.putText(vis, nm, (xy[0] + 14, xy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
overlay_path = OUT / "stage0_landmark_overlay.png"
cv2.imwrite(str(overlay_path), vis)
say(f"\n[10] Overlay written: {overlay_path}")

say("\n" + "=" * 72)
hands_ok = counts["left_hand"] == 21 and counts["right_hand"] == 21
pose_ok = counts["pose"] == 33
say(f"HANDS 21/21 match GISLR : {hands_ok}")
say(f"POSE  33    matches GISLR: {pose_ok}")
say(f"VERDICT: {'PASS -- blocks we use are identical' if (hands_ok and pose_ok) else 'FAIL -- investigate'}")
say("=" * 72)

(OUT / "stage0_report.txt").write_text("\n".join(lines) + "\n")
