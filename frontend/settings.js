import {
  readSignedLanguage,
  writeSignedLanguage,
  readSpokenLanguage,
  writeSpokenLanguage,
  readVoice,
  writeVoice,
  VOICE_PITCH_LABELS,
  VOICE_AGE_LABELS,
} from "./languages.js";

const signedSelect = document.getElementById("signedLanguage");
const spokenSelect = document.getElementById("spokenLanguage");
const pitchInput = document.getElementById("voicePitch");
const ageInput = document.getElementById("voiceAge");
const voiceNote = document.getElementById("voiceNote");

signedSelect.value = readSignedLanguage();
spokenSelect.value = readSpokenLanguage();

signedSelect.addEventListener("change", () => writeSignedLanguage(signedSelect.value));
spokenSelect.addEventListener("change", () => writeSpokenLanguage(spokenSelect.value));

// ---- voice -----------------------------------------------------------------
// The two sliders pick a voice that ElevenLabs designed AHEAD OF TIME. Nothing
// is generated here: designing a voice takes about seven seconds and permanently
// consumes an account slot, so doing it on input would freeze this page on every
// drag. backend/scripts/v022_design_voice_grid.py bakes the grid; the server
// turns the pair into a voice_id with a dict lookup.

const voice = readVoice();
pitchInput.value = String(voice.pitch);
ageInput.value = String(voice.age);

function describe() {
  const p = Number(pitchInput.value);
  const a = Number(ageInput.value);
  return `${VOICE_PITCH_LABELS[p]}, ${VOICE_AGE_LABELS[a]}`;
}

function update() {
  writeVoice({ pitch: Number(pitchInput.value), age: Number(ageInput.value) });
  voiceNote.textContent = describe();
}

// `input` not `change`, so the label tracks the thumb while it is being dragged.
pitchInput.addEventListener("input", update);
ageInput.addEventListener("input", update);
voiceNote.textContent = describe();
