import { signerLanguage, speakerLanguage } from "./languages.js";
import { createTranslator, SAME_LANGUAGE, UNSUPPORTED } from "./translate.js";

const REALTIME_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime";
const TOKEN_ENDPOINT = "/api/scribe-token";
const MODEL_ID = "scribe_v2_realtime";
const SAMPLE_RATE = 16000;
const PARTIAL_DEBOUNCE_MS = 350;

// Anything the server sends under one of these types ends the session.
const FATAL_MESSAGES = new Set([
  "error",
  "auth_error",
  "quota_exceeded",
  "unaccepted_terms",
  "rate_limited",
  "queue_overflow",
  "resource_exhausted",
  "session_time_limit_exceeded",
  "input_error",
  "invalid_request",
  "chunk_size_exceeded",
  "transcriber_error",
]);

const listenButton = document.getElementById("listen");
const statusLabel = document.getElementById("status");
const noteLabel = document.getElementById("note");
const committedLabel = document.getElementById("committed");
const partialLabel = document.getElementById("partial");
const placeholder = document.getElementById("placeholder");
const visualizer = document.getElementById("viz");

let session = null;
let level = 0;

// Partial transcripts arrive about ten times a second. Translating every one
// would be wasteful and visually jittery, so they are debounced, and each
// translation carries a sequence number so a slow one cannot overwrite newer
// text after it finally resolves.
let pendingPartial = null;
let partialTimer = 0;
let partialSequence = 0;

// Committed transcripts have to appear in the order they were spoken, which
// concurrent translation does not guarantee on its own.
let committedChain = Promise.resolve();

function setStatus(state, message) {
  statusLabel.dataset.state = state;
  statusLabel.textContent = message;
}

function setNote(message) {
  noteLabel.textContent = message;
  noteLabel.hidden = !message;
}

function setPressed(pressed) {
  listenButton.setAttribute("aria-pressed", String(pressed));
  listenButton.textContent = pressed ? "Stop" : "Listen";
}

function renderTranscript() {
  placeholder.hidden = Boolean(committedLabel.textContent || partialLabel.textContent);
}

function appendCommitted(text) {
  const trimmed = text.trim();
  if (!trimmed) {
    return;
  }
  committedLabel.textContent = committedLabel.textContent
    ? `${committedLabel.textContent} ${trimmed}`
    : trimmed;
}

function setLevel(rms) {
  // Peak-with-decay reads better than raw RMS: bars jump on syllables and fall
  // back smoothly instead of flickering.
  const scaled = Math.min(1, Math.sqrt(rms) * 2.2);
  level = Math.max(scaled, level * 0.72);
  visualizer.style.setProperty("--level", level.toFixed(3));
}

function describeLanguages({ speaker, signer }) {
  return speaker.code === signer.code
    ? speaker.label
    : `${speaker.label} \u2192 ${signer.label}`;
}

function describeTranslator(translator, { speaker, signer }) {
  if (!translator.passthrough || translator.reason === SAME_LANGUAGE) {
    return "";
  }
  if (translator.reason === UNSUPPORTED) {
    return `This browser has no built-in translator, so the transcript stays in `
      + `${speaker.label} rather than ${signer.label}. Chrome and Edge translate on-device.`;
  }
  return `Could not start the ${speaker.label} to ${signer.label} translator `
    + `(${translator.error?.message ?? "unknown error"}), so the transcript stays in ${speaker.label}.`;
}

function toBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

/**
 * Where serve.py is, when this page was NOT served by it.
 *
 * `/api/scribe-token` is a relative path, so it only resolves when serve.py
 * served the page. Opening the pages from VS Code Live Preview (or any other
 * static server) instead makes that a 404 -- which has now cost two debugging
 * sessions. On localhost we therefore retry against serve.py's own port rather
 * than reporting a 404 that looks like a broken backend.
 *
 * Deliberately localhost-only: in production the page is served by a host that
 * proxies /api to the real backend, and a hardcoded dev port must never be
 * tried there.
 */
const LOCAL_TOKEN_FALLBACK = "http://127.0.0.1:8080/api/scribe-token";

function isLocal() {
  return ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);
}

async function requestToken(url) {
  const response = await fetch(url, { method: "POST" });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const err = new Error(payload.error ?? `token request failed (${response.status})`);
    err.status = response.status;
    throw err;
  }
  return payload.token;
}

async function fetchToken() {
  try {
    return await requestToken(TOKEN_ENDPOINT);
  } catch (error) {
    const servedElsewhere = error.status === 404 || error.status === undefined;
    if (!isLocal() || !servedElsewhere
        || location.origin === new URL(LOCAL_TOKEN_FALLBACK).origin) {
      throw error;
    }
    console.warn("[speech] no /api/scribe-token on this origin "
                 + `(${location.origin}); falling back to serve.py. `
                 + "Open the page from http://127.0.0.1:8080 to avoid this.");
    return requestToken(LOCAL_TOKEN_FALLBACK);
  }
}

function buildSocketUrl(token, languageCode) {
  const url = new URL(REALTIME_URL);
  url.searchParams.set("model_id", MODEL_ID);
  url.searchParams.set("token", token);
  url.searchParams.set("audio_format", `pcm_${SAMPLE_RATE}`);
  url.searchParams.set("commit_strategy", "vad");
  // What the speaker is saying, not what the signer reads -- translation to the
  // signer's language happens after transcription.
  url.searchParams.set("language_code", languageCode);
  return url.toString();
}

async function translateSafely(text, owner) {
  try {
    return await owner.translator.translate(text);
  } catch (error) {
    console.warn("[translate]", error);
    return text; // Text in the wrong language beats no text at all.
  }
}

function cancelPartial() {
  if (partialTimer) {
    clearTimeout(partialTimer);
    partialTimer = 0;
  }
  pendingPartial = null;
  partialSequence += 1; // Invalidates anything already in flight.
}

function queuePartial(text) {
  const owner = session;

  if (owner.translator.passthrough) {
    partialSequence += 1;
    partialLabel.textContent = text;
    renderTranscript();
    return;
  }

  pendingPartial = text;
  if (partialTimer) {
    return;
  }

  partialTimer = setTimeout(async () => {
    partialTimer = 0;
    const next = pendingPartial;
    pendingPartial = null;
    if (next === null || session !== owner) {
      return;
    }

    const sequence = ++partialSequence;
    const translated = await translateSafely(next, owner);
    if (sequence !== partialSequence || session !== owner) {
      return;
    }
    partialLabel.textContent = translated;
    renderTranscript();
  }, PARTIAL_DEBOUNCE_MS);
}

function queueCommitted(text) {
  const owner = session;
  committedChain = committedChain.then(async () => {
    if (session !== owner) {
      return;
    }
    const translated = await translateSafely(text, owner);
    if (session !== owner) {
      return;
    }
    appendCommitted(translated);
    partialLabel.textContent = "";
    renderTranscript();
  });
}

function handleMessage(raw) {
  if (!session) {
    return; // Stopped already; nothing left to render into.
  }

  let message;
  try {
    message = JSON.parse(raw);
  } catch {
    return;
  }

  const type = message.message_type;

  if (type === "session_started") {
    setStatus("live", `Listening \u00b7 ${describeLanguages(session.languages)}`);
    return;
  }

  if (type === "partial_transcript") {
    queuePartial(message.text ?? "");
    return;
  }

  if (type === "committed_transcript") {
    cancelPartial();
    queueCommitted(message.text ?? "");
    return;
  }

  if (type === "warning") {
    console.warn("[scribe]", message.warning);
    return;
  }

  if (FATAL_MESSAGES.has(type)) {
    stop(`${type.replace(/_/g, " ")}: ${message.error ?? "unknown error"}`);
  }
}

async function start() {
  // The speaker's language is what we listen for; the signer's is what we show.
  const languages = { speaker: speakerLanguage(), signer: signerLanguage() };

  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("error", "This browser cannot reach the microphone. Serve the page over http://localhost.");
    return;
  }

  listenButton.disabled = true;
  setNote("");

  const pending = {
    stream: null,
    context: null,
    socket: null,
    chunker: null,
    sink: null,
    translator: null,
    translatorSetup: null,
    languages,
  };

  try {
    // Started before anything is awaited, because downloading a language pack
    // needs fresh user activation and the microphone prompt can spend it. It is
    // awaited further down, so its setup overlaps the mic and socket work
    // instead of delaying them.
    pending.translatorSetup = createTranslator(
      languages.speaker.code,
      languages.signer.code,
      (loaded) => setStatus("connecting", `Downloading translator \u00b7 ${Math.round(loaded * 100)}%`),
    );

    setStatus("connecting", "Waiting for microphone access\u2026");
    pending.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    setStatus("connecting", `Connecting to ElevenLabs (${languages.speaker.label})\u2026`);
    const token = await fetchToken();

    pending.context = new AudioContext({ sampleRate: SAMPLE_RATE });
    await pending.context.audioWorklet.addModule("./pcm-worklet.js");

    pending.socket = new WebSocket(buildSocketUrl(token, languages.speaker.code));
    await new Promise((resolve, reject) => {
      pending.socket.addEventListener("open", resolve, { once: true });
      pending.socket.addEventListener("error", () => reject(new Error("could not reach ElevenLabs")), { once: true });
    });

    pending.translator = await pending.translatorSetup;
    setNote(describeTranslator(pending.translator, languages));

    // Translator setup is bounded but not instant, and the socket may not have
    // survived it.
    if (pending.socket.readyState !== WebSocket.OPEN) {
      throw new Error("ElevenLabs closed the connection during setup");
    }

    const source = pending.context.createMediaStreamSource(pending.stream);
    pending.chunker = new AudioWorkletNode(pending.context, "pcm-chunker", { numberOfOutputs: 1 });
    pending.sink = pending.context.createGain();
    pending.sink.gain.value = 0;

    pending.chunker.port.onmessage = ({ data }) => {
      setLevel(data.level);
      if (pending.socket.readyState === WebSocket.OPEN) {
        pending.socket.send(JSON.stringify({
          message_type: "input_audio_chunk",
          audio_base_64: toBase64(data.pcm),
          commit: false,
          sample_rate: SAMPLE_RATE,
        }));
      }
    };

    session = pending;

    // Listeners attach only once `session` exists, because the message handler
    // reaches for it, and `session_started` can arrive while setup is still
    // finishing.
    pending.socket.addEventListener("message", (event) => handleMessage(event.data));
    pending.socket.addEventListener("close", () => {
      if (session) {
        stop("Connection to ElevenLabs closed.");
      }
    });

    // Connecting the graph is what starts audio flowing, so it happens last. A
    // worklet is only pulled if its output reaches the destination, hence the
    // silent gain node rather than echoing the mic back.
    source.connect(pending.chunker);
    pending.chunker.connect(pending.sink);
    pending.sink.connect(pending.context.destination);

    visualizer.classList.add("live");
    setPressed(true);
    setStatus("live", `Listening \u00b7 ${describeLanguages(languages)}`);
  } catch (error) {
    discard(pending);
    const reason = error instanceof DOMException && error.name === "NotAllowedError"
      ? "Microphone access was blocked."
      : error.message;
    setStatus("error", reason);
  } finally {
    listenButton.disabled = false;
  }
}

function discard(resources) {
  if (resources.chunker) {
    resources.chunker.port.onmessage = null;
    resources.chunker.disconnect();
  }
  resources.sink?.disconnect();
  resources.stream?.getTracks().forEach((track) => track.stop());
  resources.context?.close().catch(() => {});
  if (resources.socket && resources.socket.readyState <= WebSocket.OPEN) {
    resources.socket.close();
  }
  if (resources.translator) {
    resources.translator.destroy();
  } else {
    // Setup was still running, so wait for it only to clean it up.
    resources.translatorSetup?.then((late) => late.destroy()).catch(() => {});
  }
}

function stop(reason) {
  const closing = session;
  session = null;
  cancelPartial();

  if (closing) {
    discard(closing);
  }

  // Whatever the model had in flight is the best text we will get for those
  // words, and it is already translated, so keep it rather than dropping it.
  appendCommitted(partialLabel.textContent);
  partialLabel.textContent = "";
  renderTranscript();

  visualizer.classList.remove("live");
  visualizer.style.removeProperty("--level");
  level = 0;
  setPressed(false);
  setStatus(reason ? "error" : "idle", reason ?? "Not listening");
}

listenButton.addEventListener("click", () => {
  if (session) {
    stop();
  } else {
    start();
  }
});

window.addEventListener("pagehide", () => {
  if (session) {
    discard(session);
    session = null;
  }
});

renderTranscript();
