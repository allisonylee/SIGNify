/**
 * Sign to text, live. The browser half of the pipeline the Python harness
 * (backend/scripts/harness_client.py) has been driving all along.
 *
 * This is a deliberate PORT of one exact harness invocation, the one that was
 * measured to work:
 *
 *   harness_client.py --source webcam --mode A --seconds 90 \
 *     --vocab hello,me,happy,see,you --expect 5
 *
 * so the capture parameters below are the harness's defaults, not new choices.
 * Changing them means the demo is no longer the thing that was tested.
 *
 * ONE DEVIATION, and it is unavoidable. The harness runs MediaPipe in Python
 * (`--mode A`) and sends landmarks. A browser cannot: it would need a separate
 * JS build of MediaPipe, whose landmarks would differ from the ones the model
 * was trained and fine-tuned on. So this sends JPEG frames instead (mode B) and
 * the backend runs the SAME HolisticExtractor. Identical landmark code; the
 * only difference is JPEG quantisation at quality 70.
 *
 * MIRRORING (plan section 14.2): the wire format is UN-MIRRORED. getUserMedia
 * is already un-mirrored, so the canvas is drawn straight and only the on-screen
 * <video> is flipped by CSS -- looking at yourself in a mirror is what people
 * expect, and sending that would silently swap the signer's hands.
 *
 * This file does not touch index.html, which belongs to someone else. It reads
 * the existing `.stage` container and `.toggle` button instead.
 */
import { speakerLanguage, readVoice } from "./languages.js";

// ---- the demo contract, fixed -------------------------------------------
// Restricting 255 classes to 5 turns a 255-way decision into a 5-way one, and
// per-sign accuracy compounds across a sentence. `expect` then ends the
// utterance on COUNT rather than on a silence timeout, so the LLM fires the
// instant the fifth sign lands. Both are what the tested command used.
const DEMO_VOCAB = ["hello", "me", "happy", "see", "you"];
const DEMO_EXPECT = DEMO_VOCAB.length;
const SIGN_LANGUAGE = "ase";              // ASL. The only model this page uses.

// ---- harness defaults, copied ---------------------------------------------
const BACKEND = "ws://127.0.0.1:8000/ws"; // backend owns 8000; this page is 8080
const WIDTH = 640;                        // harness --width
const JPEG_QUALITY = 0.7;                 // harness --jpeg-quality 70
const TARGET_FPS = 25;                    // harness asks 30, camera delivers ~25

const stage = document.querySelector(".stage");
const committed = document.getElementById("committed");
const placeholder = document.getElementById("placeholder");
const speakButton = document.querySelector(".toggle");

let session = null;

// ---- UI --------------------------------------------------------------------

const ui = {};

function buildStage() {
  stage.textContent = "";
  stage.classList.add("sign-stage");
  ui.video = document.createElement("video");
  ui.video.autoplay = true;
  ui.video.playsInline = true;
  ui.video.muted = true;                  // or autoplay is blocked
  ui.video.className = "sign-video";
  // Empty in normal operation. Kept only so a camera failure has somewhere to
  // show itself; the CSS gives it no height while empty.
  ui.status = document.createElement("p");
  ui.status.className = "sign-status";
  ui.status.setAttribute("role", "status");
  ui.glosses = document.createElement("p");
  ui.glosses.className = "sign-glosses";
  stage.append(ui.video, ui.glosses, ui.status);
}

function setStatus(text) {
  ui.status.textContent = text;
}

/** Progress as the signs land: "hello me happy · 3/5". */
function showGlosses(list) {
  ui.glosses.textContent = list.length
    ? `${list.join(" ")}  ·  ${list.length}/${DEMO_EXPECT}`
    : "";
}

/**
 * Sentences ACCUMULATE. Each utterance is a separate thing the signer said, so
 * replacing the previous one would erase the conversation as it happens.
 */
function appendSentence(text) {
  const trimmed = (text ?? "").trim();
  if (!trimmed) return;
  committed.textContent = committed.textContent
    ? `${committed.textContent} ${trimmed}`
    : trimmed;
  placeholder.hidden = true;
}

function speakWanted() {
  return speakButton?.getAttribute("aria-pressed") === "true";
}

// ---- audio -----------------------------------------------------------------

/**
 * The backend returns whole-utterance PCM (s16le, 16 kHz, base64). Web Audio
 * wants float32 in [-1, 1], and clips are played strictly in order so a second
 * utterance cannot start on top of the first.
 */
const audio = { ctx: null, tail: Promise.resolve() };

/**
 * MUST be called from a real user gesture. Chrome starts an AudioContext
 * created outside one in the "suspended" state, and a suspended context plays
 * nothing -- silently, with no error anywhere. The audio arrives in a WebSocket
 * message callback, which is NOT a gesture, so the context is opened here, on
 * the Speak click, instead of lazily on first playback.
 */
function armAudio() {
  audio.ctx ??= new AudioContext();
  if (audio.ctx.state === "suspended") {
    audio.ctx.resume().catch(() => {});
  }
}

function playPcm(base64, sampleRate) {
  audio.ctx ??= new AudioContext();
  const bin = atob(base64);
  const pcm = new Int16Array(bin.length / 2);
  for (let i = 0; i < pcm.length; i += 1) {
    pcm[i] = (bin.charCodeAt(i * 2 + 1) << 8) | bin.charCodeAt(i * 2); // little endian
  }
  const buffer = audio.ctx.createBuffer(1, pcm.length, sampleRate);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < pcm.length; i += 1) {
    channel[i] = pcm[i] / 32768;
  }
  audio.tail = audio.tail.then(() => new Promise((resolve) => {
    const src = audio.ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(audio.ctx.destination);
    src.onended = resolve;
    src.start();
  })).catch(() => {});
}

// ---- wire ------------------------------------------------------------------

function configMessage() {
  return {
    type: "config",
    sign_language: SIGN_LANGUAGE,
    // Signing is read by the SPEAKING user, so the sentence belongs in THEIR
    // language, whatever signed language produced it.
    output_language: speakerLanguage().code,
    mode: speakWanted() ? "speech" : "text",
    // Slider positions, not audio parameters. The server maps the pair to a
    // voice ElevenLabs designed offline -- a dict lookup, no API call, so this
    // adds nothing to speech latency.
    voice: readVoice(),
    vocab: DEMO_VOCAB,
    expect: DEMO_EXPECT,
  };
}

/**
 * Push the current settings to an already-open socket. Toggling Speak must not
 * tear the camera down and build it back up -- `mode` is the only thing that
 * changed, and a restart would drop any signs mid-utterance.
 */
function sendConfig() {
  if (session?.socket?.readyState === WebSocket.OPEN) {
    session.socket.send(JSON.stringify(configMessage()));
  }
}

function handleMessage(raw, owner) {
  if (session !== owner) return;
  let m;
  try { m = JSON.parse(raw); } catch { return; }

  switch (m.type) {
    case "partial":
      showGlosses(m.gloss ?? []);
      break;
    case "text":
      appendSentence(m.text);
      showGlosses([]);
      break;
    case "audio":
      if (m.chunk) playPcm(m.chunk, m.sample_rate ?? 16000);
      break;
    case "rejected":
      // Without this, "not recognised" and "recognised then discarded" look
      // identical from the outside, which is the hardest kind of bug to debug
      // live on stage.
      console.debug("[sign] rejected:", m.reason, m.top);
      break;
    case "timing":
      console.debug("[sign] timing:", m);
      break;
    case "error":
      console.warn("[sign]", m.detail);
      setStatus(m.detail);
      break;
    default:
      break;
  }
}

// ---- capture ---------------------------------------------------------------

async function start() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("This browser cannot reach the camera. Serve the page over http://localhost.");
    return;
  }

  const pending = { stream: null, socket: null, timer: 0, canvas: null, ctx: null };

  try {
    setStatus("Waiting for camera access…");
    pending.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    ui.video.srcObject = pending.stream;
    await ui.video.play();

    setStatus("Connecting…");
    pending.socket = new WebSocket(BACKEND);
    await new Promise((resolve, reject) => {
      pending.socket.addEventListener("open", resolve, { once: true });
      pending.socket.addEventListener("error",
        () => reject(new Error("could not reach the backend on 127.0.0.1:8000")),
        { once: true });
    });

    // Downscale to the harness's working width. 720p is far more than MediaPipe
    // needs and costs real time on every frame.
    const vw = ui.video.videoWidth || 1280;
    const vh = ui.video.videoHeight || 720;
    const scale = Math.min(1, WIDTH / vw);
    pending.canvas = document.createElement("canvas");
    pending.canvas.width = Math.round(vw * scale);
    pending.canvas.height = Math.round(vh * scale);
    pending.ctx = pending.canvas.getContext("2d", { alpha: false });

    session = pending;
    pending.socket.addEventListener("message", (e) => handleMessage(e.data, pending));
    pending.socket.addEventListener("close", () => {
      if (session === pending) stop("Backend connection closed.");
    });

    pending.socket.send(JSON.stringify(configMessage()));

    const t0 = performance.now();
    let seq = 0;
    pending.timer = setInterval(() => {
      if (session !== pending || pending.socket.readyState !== WebSocket.OPEN) return;
      // BACKPRESSURE. MediaPipe costs ~19 ms a frame server-side; if the camera
      // ever outruns it, queueing frames adds latency to every later one and
      // the glosses drift behind the signing. Skip instead.
      if (pending.socket.bufferedAmount > 1_000_000) return;

      // Un-mirrored, deliberately. See the header note.
      pending.ctx.drawImage(ui.video, 0, 0, pending.canvas.width, pending.canvas.height);
      const dataUrl = pending.canvas.toDataURL("image/jpeg", JPEG_QUALITY);
      pending.socket.send(JSON.stringify({
        type: "frame",
        seq: seq++,
        t: (performance.now() - t0) / 1000,
        jpeg: dataUrl.slice(dataUrl.indexOf(",") + 1),
      }));
    }, Math.round(1000 / TARGET_FPS));

    stage.classList.add("live");
    setStatus("");
  } catch (error) {
    discard(pending);
    setStatus(error instanceof DOMException && error.name === "NotAllowedError"
      ? "Camera access was blocked."
      : error.message);
    throw error;
  }
}

function discard(r) {
  if (r.timer) clearInterval(r.timer);
  r.stream?.getTracks().forEach((t) => t.stop());
  if (r.socket && r.socket.readyState <= WebSocket.OPEN) r.socket.close();
  if (ui.video) ui.video.srcObject = null;
}

function stop(reason) {
  const closing = session;
  session = null;
  if (closing) discard(closing);
  stage.classList.remove("live");
  showGlosses([]);
  setStatus(reason ?? "Camera off");
}

// ---- wiring ----------------------------------------------------------------

buildStage();
// index.html's placeholder ("The translation will appear here...") already says
// the only thing that needs saying, so it is left alone.

// index.html's own inline handler already flips aria-pressed, so this listener
// READS the new state rather than owning it.
//
// Speak is audio ONLY. It does not touch the camera, which is always running:
// turning it off still recognises signs and still shows the sentence, it just
// does not speak. That maps to the backend's `mode`, so with it off the server
// skips the TTS round trip entirely rather than synthesising audio nobody plays.
speakButton?.addEventListener("click", () => {
  // The click is the ONLY user gesture this page gets now that the camera
  // starts by itself, and Chrome will not let an AudioContext run without one.
  // Arming here is what makes the very first spoken sentence audible.
  if (speakWanted()) armAudio();
  sendConfig();
});

// The camera comes up on its own -- no click, per the Sign tab's design. If the
// browser refuses, offer a retry, because Speak no longer doubles as a start
// button and there would otherwise be no way back.
start().catch(() => {
  setStatus("Camera unavailable. Click here to try again.");
  stage.addEventListener("click", function retry() {
    stage.removeEventListener("click", retry);
    start().catch(() => setStatus("Camera still unavailable."));
  });
});

window.addEventListener("pagehide", () => {
  if (session) { discard(session); session = null; }
});
