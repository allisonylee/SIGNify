"""
v022 -- pre-bake the pitch x age voice grid with ElevenLabs Voice Design.

RUN THIS OFFLINE, BEFORE THE DEMO. Never at request time.

MEASURED: one design call takes ~7 s and a create call adds more. Doing that
when a slider moves would stall the app for seconds; doing it at launch would
mean the app cannot speak until it finished. So every combination is designed
ONCE here, promoted to a real voice, and written to a lookup table that the
server reads at startup. At runtime, moving a slider is a dict lookup -- the
translation and speech paths never touch this API.

Voice Design is two steps (docs: api-reference/text-to-voice/design):
  1. POST /v1/text-to-voice/design  -> previews, each a temporary
     generated_voice_id. Takes PROSE, not structured age/pitch fields.
  2. POST /v1/text-to-voice         -> promotes one to a permanent voice_id.
Step 2 consumes a voice slot in the account, which is why this is resumable and
saves after every single cell: hitting a plan limit half way must not lose the
voices already paid for.

    .venv/bin/python backend/scripts/v022_design_voice_grid.py              # dry run
    .venv/bin/python backend/scripts/v022_design_voice_grid.py --apply --only 3,3
    .venv/bin/python backend/scripts/v022_design_voice_grid.py --apply
"""
import argparse, json, pathlib, sys, time, urllib.error, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "backend" / "app" / "voice_grid.json"
API = "https://api.elevenlabs.io"

# The sliders are 1..5. These are the words that reach ElevenLabs; the grid is
# only as good as they are, so they are written to vary ONE quality at a time.
PITCH = {
    1: "a very deep, low-pitched",
    2: "a deep, low-pitched",
    3: "a mid-pitched, natural",
    4: "a bright, high-pitched",
    5: "a very bright, very high-pitched",
}
# NO MINORS. "child" was the original bottom of this scale and ElevenLabs
# refused it outright -- HTTP 403 blocked_generation, "doesn't follow our safety
# guidelines". That is the correct call on their part: a synthetic child voice
# is a real abuse vector, so the scale starts at a young adult rather than being
# reworded to slip past the filter.
AGE = {
    1: "young adult in their late teens or early twenties",
    2: "adult in their mid twenties",
    3: "adult in their thirties",
    4: "middle-aged adult in their fifties",
    5: "elderly person in their seventies",
}
# 100-1000 chars is the documented range, and fixed text keeps previews
# comparable between cells instead of varying with auto-generated copy.
PREVIEW_TEXT = (
    "Hello, I am happy to see you. This voice will read sign language out loud, "
    "so it should sound clear, natural and easy to follow at a conversational "
    "pace, without sounding rushed or flat."
)


def describe(pitch: int, age: int) -> str:
    who = AGE[age]
    article = "an" if who[0] in "aeiou" else "a"      # "a adult" reads as noise
    return (f"{PITCH[pitch].capitalize()} voice belonging to {article} {who}. "
            f"Warm, clear and friendly, speaking at a natural conversational "
            f"pace with neutral pronunciation.")


def read_key() -> str:
    import os
    if k := os.environ.get("ELEVENLABS_API_KEY"):
        return k
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        name, _, value = line.strip().partition("=")
        if name == "ELEVENLABS_API_KEY":
            return value.strip().strip("\"'")
    sys.exit("no ELEVENLABS_API_KEY in the environment or .env")


def call(key, path, body, timeout=120):
    req = urllib.request.Request(
        API + path, data=json.dumps(body).encode(), method="POST",
        headers={"xi-api-key": key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def load_grid():
    if OUT.exists():
        return json.loads(OUT.read_text())
    return {"model": "eleven_flash_v2_5", "cells": {}}


def save_grid(g):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(g, indent=2, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually design and CREATE voices (consumes account "
                         "voice slots). Without it, nothing is sent.")
    ap.add_argument("--grid", type=int, default=5, help="N x N, sliders are 1..N")
    ap.add_argument("--only", default="", metavar="P,A",
                    help="just one cell, e.g. 3,3")
    ap.add_argument("--redo", action="store_true",
                    help="regenerate cells that already exist")
    args = ap.parse_args()

    key = read_key()
    grid = load_grid()
    cells = grid["cells"]

    wanted = ([tuple(int(x) for x in args.only.split(","))] if args.only
              else [(p, a) for p in range(1, args.grid + 1)
                    for a in range(1, args.grid + 1)])
    todo = [c for c in wanted if args.redo or f"{c[0]},{c[1]}" not in cells]

    print(f"  table      : {OUT}")
    print(f"  already in : {len(cells)} cells")
    print(f"  requested  : {len(wanted)}   to generate: {len(todo)}")
    if not args.apply:
        print("\n  DRY RUN -- nothing sent. Prompts that would be used:\n")
        for p, a in todo[:6]:
            print(f"    [{p},{a}] {describe(p, a)}")
        if len(todo) > 6:
            print(f"    ... and {len(todo) - 6} more")
        print(f"\n  ~{len(todo) * 9 / 60:.1f} min at ~9 s/cell, and {len(todo)} "
              f"voice slots in your ElevenLabs account.")
        print("  Re-run with --apply to do it.")
        return 0

    made = failed = 0
    for i, (p, a) in enumerate(todo, 1):
        kk = f"{p},{a}"
        prompt = describe(p, a)
        t0 = time.perf_counter()
        try:
            d = call(key, "/v1/text-to-voice/design",
                     {"voice_description": prompt, "text": PREVIEW_TEXT})
            previews = d.get("previews") or []
            if not previews:
                raise RuntimeError("design returned no previews")
            gvid = previews[0]["generated_voice_id"]
            t_design = time.perf_counter() - t0

            v = call(key, "/v1/text-to-voice",
                     {"voice_name": f"signagram-p{p}-a{a}",
                      "voice_description": prompt,
                      "generated_voice_id": gvid})
            vid = v.get("voice_id") or (v.get("voice") or {}).get("voice_id")
            if not vid:
                raise RuntimeError(f"create returned no voice_id: {str(v)[:200]}")
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:300]
            print(f"  [{i}/{len(todo)}] {kk}  HTTP {e.code}: {detail}")
            failed += 1
            # A blocked PROMPT is about that one cell -- skip it and keep
            # going. A permission or slot problem affects every remaining cell,
            # so stopping there saves 3 minutes of guaranteed failures.
            low = detail.lower()
            if "blocked_generation" in low or "safety guidelines" in low:
                print(f"           ElevenLabs refused this description; "
                      f"leaving [{p},{a}] unbaked and continuing.")
                continue
            if e.code in (401, 403) or "limit" in low:
                print("\n  STOPPING: permission or voice-slot limit, not a "
                      "transient error.")
                print(f"  {made} voices were created and saved; re-run to resume.")
                break
            continue
        except Exception as e:                               # noqa: BLE001
            print(f"  [{i}/{len(todo)}] {kk}  {type(e).__name__}: {e}")
            failed += 1
            continue

        cells[kk] = {"voice_id": vid, "name": f"signagram-p{p}-a{a}",
                     "pitch": p, "age": a, "description": prompt}
        save_grid(grid)                 # after EVERY cell, never at the end
        made += 1
        print(f"  [{i}/{len(todo)}] {kk} -> {vid}  "
              f"(design {t_design:.1f}s, total {time.perf_counter()-t0:.1f}s)")

    print(f"\n  created {made}, failed {failed}, table now has {len(cells)} cells")
    print(f"  written to {OUT}")
    return 0 if made or not todo else 1


if __name__ == "__main__":
    sys.exit(main())
