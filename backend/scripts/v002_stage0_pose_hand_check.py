"""
Stage 0 -- MediaPipe API compatibility check (Pose + Hand path).

Plan: claude/docs/PLAN_2026-09-18_232008.md, Stage 0.

Supersedes v001, which used HolisticLandmarker. v001 established that
HolisticLandmarker ABORTS on this machine in every delegate/format combination
(see backend/outputs/stage0_api_matrix.txt). PoseLandmarker + HandLandmarker
work, but ONLY with delegate=GPU and ImageFormat.SRGBA.

Question this answers: GISLR landmarks came from the legacy
`mp.solutions.holistic` API (543/frame = 468 face + 33 pose + 21 + 21 hands).
We use the Tasks API. Do the blocks we actually consume -- hands (21/21) and
pose (33) -- line up, and what are the coordinate and handedness conventions?

We do NOT use face landmarks (plan Stage 2), so the 468-vs-478 face gap is
irrelevant by construction.

Outputs:
  backend/outputs/stage0_landmark_overlay.png
  backend/outputs/stage0_report.txt
"""

import sys
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS, OUT = ROOT / "models", ROOT / "outputs"
V, BO = mp.tasks.vision, mp.tasks.BaseOptions

# The ONLY configuration that works here. Do not change without re-running v001.
DELEGATE = BO.Delegate.GPU
IMG_FORMAT = mp.ImageFormat.SRGBA

URLS = {
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task",
}
IMAGE_URL = "https://storage.googleapis.com/mediapipe-assets/male_full_height_hands.jpg"

# GISLR (legacy solutions API) per-block counts.
GISLR = {"pose": 33, "hand": 21}

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
say("STAGE 0 -- MediaPipe API compatibility check  (Pose + Hand)")
say("=" * 74)

say("\n[1] Environment")
say(f"  python     {sys.version.split()[0]}")
say(f"  executable {sys.executable}")
say(f"  mediapipe  {mp.__version__}")
say(f"  numpy      {np.__version__}   opencv {cv2.__version__}")
say(f"  delegate   {DELEGATE.name}    image format {IMG_FORMAT.name}")

say("\n[2] Legacy Solutions API (which produced GISLR)")
say(f"  mp.solutions present: {hasattr(mp, 'solutions')}")
say("  -> REMOVED in mediapipe 1.0.x. We cannot re-run GISLR's own extractor;")
say("     any comparison must be against the GISLR parquet itself.")

pose_model = fetch(URLS["pose"], MODELS / "pose_landmarker_lite.task")
hand_model = fetch(URLS["hand"], MODELS / "hand_landmarker.task")
img_path = fetch(IMAGE_URL, OUT / "stage0_test_subject.jpg")

bgr = cv2.imread(str(img_path))
h, w = bgr.shape[:2]
rgba = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA))
mp_img = mp.Image(image_format=IMG_FORMAT, data=rgba)
say(f"\n[3] Test image {w}x{h}  ({img_path.name})")

with V.PoseLandmarker.create_from_options(V.PoseLandmarkerOptions(
        base_options=BO(model_asset_path=str(pose_model), delegate=DELEGATE),
        running_mode=V.RunningMode.IMAGE)) as d:
    pres = d.detect(mp_img)

with V.HandLandmarker.create_from_options(V.HandLandmarkerOptions(
        base_options=BO(model_asset_path=str(hand_model), delegate=DELEGATE),
        running_mode=V.RunningMode.IMAGE, num_hands=2)) as d:
    hres = d.detect(mp_img)

n_people = len(pres.pose_landmarks)
n_hands = len(hres.hand_landmarks)
pose = pres.pose_landmarks[0]

say("\n[4] PER-BLOCK LANDMARK COUNTS   <-- Stage 0 stop condition")
say(f"  {'block':<14}{'Tasks 1.0.1':>13}{'GISLR legacy':>15}   verdict")
say("  " + "-" * 54)
say(f"  {'people':<14}{n_people:>13}{'-':>15}")
say(f"  {'pose':<14}{len(pose):>13}{GISLR['pose']:>15}   "
    f"{'OK' if len(pose) == GISLR['pose'] else 'DIFF'}")
for i, hl in enumerate(hres.hand_landmarks):
    lbl = hres.handedness[i][0].category_name
    say(f"  {'hand[' + str(i) + '] ' + lbl:<14}{len(hl):>13}{GISLR['hand']:>15}   "
        f"{'OK' if len(hl) == GISLR['hand'] else 'DIFF'}")
say(f"  hands detected: {n_hands}/2")

say("\n[5] Coordinate convention")
xs = np.array([p.x for p in pose]); ys = np.array([p.y for p in pose])
zs = np.array([p.z for p in pose])
say(f"  pose x  [{xs.min():+.3f}, {xs.max():+.3f}]")
say(f"  pose y  [{ys.min():+.3f}, {ys.max():+.3f}]")
say(f"  pose z  [{zs.min():+.3f}, {zs.max():+.3f}]   (dropped per plan Stage 2)")
inside = ((0 <= xs) & (xs <= 1) & (0 <= ys) & (ys <= 1)).mean()
say(f"  fraction of pose x,y in [0,1]: {inside:.0%} -> "
    f"{'frame-normalised' if inside > 0.8 else 'NOT frame-normalised'}")

say("\n[6] Index semantics -- named pose landmarks")
PL = V.PoseLandmark
for nm in ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW",
           "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP"]:
    i = int(getattr(PL, nm)); p = pose[i]
    say(f"  {nm:<16} idx={i:<3} x={p.x:+.3f} y={p.y:+.3f}")

lsh, rsh, nose = pose[int(PL.LEFT_SHOULDER)], pose[int(PL.RIGHT_SHOULDER)], pose[int(PL.NOSE)]
say("\n[7] Sanity + handedness convention  (R9 in the plan)")
say(f"  shoulder width (norm units)  {abs(lsh.x - rsh.x):.4f}")
say(f"  nose above shoulders         {nose.y < min(lsh.y, rsh.y)}")
say(f"  LEFT_SHOULDER.x > RIGHT_SHOULDER.x  ->  {lsh.x:.3f} > {rsh.x:.3f} = {lsh.x > rsh.x}")
say("  Un-mirrored image: the SUBJECT's left appears on the VIEWER's right,")
say("  so LEFT_*.x should exceed RIGHT_*.x. This is the wire convention in")
say("  plan section 14.2 -- clients send UN-MIRRORED data.")

say("\n[8] Our 53-landmark feature set (plan Stage 2)")
# 11 upper-body pose points. MOUTH_* give head orientation and a crude
# mouthing proxy without touching the face block (see plan Stage 2).
UPPER = ["NOSE", "MOUTH_LEFT", "MOUTH_RIGHT",
         "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
         "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP"]
total = 21 + 21 + len(UPPER)
say(f"  hands  21 + 21           = 42")
say(f"  upper-body pose          = {len(UPPER)}")
for nm in UPPER:
    i = int(getattr(PL, nm)); pt = pose[i]
    say(f"      {nm:<16} idx={i:<3} x={pt.x:+.3f} y={pt.y:+.3f} vis={pt.visibility:.2f}")
say(f"  TOTAL {total} landmarks -> {total * 2} features (x,y)")
say("  face block: NOT USED -> the 468-vs-478 discrepancy cannot bite us.")

vis = bgr.copy()
for p in pose:
    cv2.circle(vis, (int(p.x * w), int(p.y * h)), 3, (0, 200, 0), -1)
for i, hl in enumerate(hres.hand_landmarks):
    col = (255, 0, 0) if hres.handedness[i][0].category_name == "Left" else (0, 0, 255)
    for p in hl:
        cv2.circle(vis, (int(p.x * w), int(p.y * h)), 3, col, -1)
for nm, col in [("LEFT_SHOULDER", (255, 255, 0)), ("RIGHT_SHOULDER", (0, 255, 255)),
                ("LEFT_WRIST", (255, 0, 255)), ("NOSE", (255, 255, 255))]:
    p = pose[int(getattr(PL, nm))]
    xy = (int(p.x * w), int(p.y * h))
    cv2.circle(vis, xy, 12, col, 3)
    cv2.putText(vis, nm, (xy[0] + 16, xy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
cv2.imwrite(str(OUT / "stage0_landmark_overlay.png"), vis)
say(f"\n[9] Overlay written: {OUT / 'stage0_landmark_overlay.png'}")

pose_ok = len(pose) == GISLR["pose"]
hands_ok = n_hands == 2 and all(len(x) == GISLR["hand"] for x in hres.hand_landmarks)
say("\n" + "=" * 74)
say(f"  pose 33 matches GISLR   : {pose_ok}")
say(f"  both hands 21 match     : {hands_ok}")
say(f"  VERDICT: {'PASS' if pose_ok and hands_ok else 'FAIL'}")
say("=" * 74)

(OUT / "stage0_report.txt").write_text("\n".join(lines) + "\n")
