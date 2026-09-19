# app/ -- ships in the final application

Only code the running app needs. If it exists purely to check or measure
something, it belongs in `../scripts/`.

Planned modules (see claude/docs/PLAN_2026-09-18_232008.md):

    main.py        FastAPI app, WS endpoint, health check. Serves NO html.
    ingest.py      LandmarkFrame + mode A (browser landmarks) / mode B
                   (JPEG frames -> MediaPipe here) adapters.
    landmarks.py   MediaPipe HolisticLandmarker wrapper; the 53-landmark spec.
    features.py    Normalisation + 32-frame timestamp resampling.
                   *** ALSO IMPORTED BY training/ -- single source of truth ***
    model.py       Shared encoder + per-language heads.
    inference.py   Ring buffer, motion gating, temporal smoothing, rest class.
    llm.py         Gemini Flash-Lite, thinking disabled.
    tts.py         ElevenLabs Flash v2.5 multi-context websocket.
    config.py      Settings, API keys from env.
