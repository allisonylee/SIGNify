# scripts/ -- verification and probes, NOT shipped

One-off scripts that check something, measure something, or stand in for a
component that does not exist yet. Disposable by design: kept for the audit
trail, not maintained.

Naming: `vNNN_short_description.py`, numbers never reused.

    v001_stage0_mediapipe_check.py   Stage 0, attempt 1. HolisticLandmarker on
                                     mediapipe 1.0.1 -> SIGABRT everywhere.
                                     Superseded; kept as evidence.
    v002_stage0_pose_hand_check.py   Stage 0, attempt 2. Pose+Hand on 1.0.1,
                                     GPU+SRGBA. Worked, but no CPU path.
                                     Superseded.
    v003_stage0_holistic_check.py    Stage 0 DEFINITIVE. Holistic on 0.10.35
                                     CPU. PASS. Re-run this on any new machine.

Planned:

    harness_client.py                Python stand-in for the browser: cv2
                                     webcam -> MediaPipe -> WS -> backend ->
                                     audio. Lets the backend be tested with no
                                     frontend. A test tool, so it lives here.
