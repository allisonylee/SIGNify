/**
 * Page chrome only: persist settings and render gloss/text.
 * Webcam, MediaPipe, and the WebSocket client are Allison's modules.
 *
 * Partner hook:
 *   SignagramUI.getConfig()           -> config payload for the WS
 *   SignagramUI.setSigning(true)
 *   SignagramUI.setPartial(["MOTHER"], 0.81)
 *   SignagramUI.setText("Hello.", "en")
 */
const STORAGE_KEY = "signagram.config";
const API_BASE_KEY = "signagram.api";

const PRESETS = {
  calm: { stability: 0.75, style: 0.0 },
  natural: { stability: 0.5, style: 0.0 },
  expressive: { stability: 0.3, style: 0.3 },
};

const DEFAULTS = {
  sign_language: "ase",
  output_language: "en",
  voice_id: "",
  preset: "natural",
  speed: 1,
  similarity_boost: 0.8,
  use_speaker_boost: false,
  mirror_preview: true,
};

function loadConfig() {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}") };
  } catch {
    return { ...DEFAULTS };
  }
}

function saveConfig(config) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
}

function apiBase() {
  return localStorage.getItem(API_BASE_KEY) || "http://127.0.0.1:8000";
}

function voicePayload(config) {
  const preset = PRESETS[config.preset] || PRESETS.natural;
  return {
    voice_id: config.voice_id || undefined,
    stability: preset.stability,
    similarity_boost: Number(config.similarity_boost),
    style: preset.style,
    speed: Number(config.speed),
    use_speaker_boost: Boolean(config.use_speaker_boost),
  };
}

function fillSelect(select, value) {
  if (!select) return;
  if (value && ![...select.options].some((option) => option.value === value)) {
    const extra = document.createElement("option");
    extra.value = value;
    extra.textContent = value;
    select.append(extra);
  }
  select.value = value;
}

function bindLivePage(config) {
  document.querySelectorAll("[data-sign-language]").forEach((el) => {
    fillSelect(el, config.sign_language);
    el.addEventListener("change", () => {
      const next = { ...loadConfig(), sign_language: el.value };
      saveConfig(next);
    });
  });

  document.querySelectorAll("[data-output-language]").forEach((el) => {
    fillSelect(el, config.output_language);
    el.addEventListener("change", () => {
      const next = { ...loadConfig(), output_language: el.value };
      saveConfig(next);
    });
  });

  const stage = document.querySelector(".stage");
  if (stage && config.mirror_preview) {
    stage.classList.add("mirrored");
  }
}

async function bindSettingsPage(config) {
  const form = document.getElementById("settings-form");
  if (!form) return;

  form.sign_language.value = config.sign_language;
  form.output_language.value = config.output_language;
  form.preset.value = config.preset;
  form.speed.value = config.speed;
  form.similarity_boost.value = config.similarity_boost;
  form.use_speaker_boost.checked = config.use_speaker_boost;
  form.mirror_preview.checked = config.mirror_preview;
  updateSpeedLabel(config.speed);

  form.addEventListener("input", (event) => {
    if (event.target.name === "speed") {
      updateSpeedLabel(form.speed.value);
    }
  });

  form.addEventListener("change", () => {
    saveConfig({
      sign_language: form.sign_language.value,
      output_language: form.output_language.value,
      voice_id: form.voice_id.value,
      preset: form.preset.value,
      speed: Number(form.speed.value),
      similarity_boost: Number(form.similarity_boost.value),
      use_speaker_boost: form.use_speaker_boost.checked,
      mirror_preview: form.mirror_preview.checked,
    });
    updateSpeedLabel(form.speed.value);
  });

  await loadVoices(form, config);
}

async function loadVoices(form, config) {
  const voiceSelect = form.voice_id;
  const status = document.getElementById("voices-status");
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 1200);

  try {
    const response = await fetch(`${apiBase()}/voices`, { signal: controller.signal });
    if (!response.ok) throw new Error(String(response.status));
    const voices = await response.json();
    voiceSelect.replaceChildren();
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = "Default voice";
    voiceSelect.append(empty);
    for (const voice of voices) {
      const option = document.createElement("option");
      option.value = voice.voice_id;
      const labels = voice.labels || {};
      option.textContent = [voice.name, labels.age, labels.gender, labels.accent]
        .filter(Boolean)
        .join(" · ");
      voiceSelect.append(option);
    }
    fillSelect(voiceSelect, config.voice_id);
    if (status) status.textContent = "Loaded from GET /voices. This page never sees the API key.";
  } catch {
    fillSelect(voiceSelect, config.voice_id);
    if (status) {
      status.textContent = "GET /voices is not up yet, so the list below is a placeholder.";
    }
  } finally {
    window.clearTimeout(timer);
  }
}

function updateSpeedLabel(value) {
  const label = document.getElementById("speed-value");
  if (label) label.textContent = Number(value).toFixed(2);
}

function setSigning(signing) {
  const el = document.getElementById("signing-state");
  if (!el) return;
  el.dataset.state = signing ? "signing" : "idle";
  el.textContent = signing ? "Signing" : "Idle";
}

function setPartial(gloss, conf) {
  const el = document.getElementById("partial-gloss");
  if (!el) return;
  const words = Array.isArray(gloss) ? gloss.join(" · ") : String(gloss || "");
  el.textContent = words || "Waiting for a sign…";
  el.dataset.conf = conf == null ? "" : String(conf);
}

function setText(text) {
  const el = document.getElementById("fluent-text");
  if (!el) return;
  el.textContent = text || "";
}

window.SignagramUI = {
  getConfig() {
    const config = loadConfig();
    return {
      type: "config",
      sign_language: config.sign_language,
      output_language: config.output_language,
      mode: document.body.dataset.mode,
      voice: voicePayload(config),
    };
  },
  setSigning,
  setPartial,
  setText,
};

const config = loadConfig();
bindLivePage(config);
bindSettingsPage(config);
