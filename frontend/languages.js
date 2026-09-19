/**
 * Two users, two independent settings.
 *
 *   - The signer picks a signed language, which implies the language they READ.
 *     ASL pairs with English, ISL with Hindi, CSN with Colombian Spanish.
 *   - The speaker picks the spoken/text language, which is what they SPEAK.
 *
 * So the speech page listens in the speaker's language and displays the
 * signer's, translating between the two, and the sign page produces text in the
 * speaker's language. Neither setting derives the other -- ASL alongside Spanish
 * is a normal pairing, not a mistake.
 */

export const DEFAULT_SIGNED_LANGUAGE = "ASL";
export const DEFAULT_SPOKEN_LANGUAGE = "English";

export const SIGNED_LANGUAGES = {
  ASL: { label: "English", code: "en" },
  ISL: { label: "Hindi", code: "hi" },
  CSN: { label: "Spanish", code: "es" },
};

export const SPOKEN_LANGUAGES = {
  English: { label: "English", code: "en" },
  Hindi: { label: "Hindi", code: "hi" },
  Spanish: { label: "Spanish", code: "es" },
};

const SIGNED_KEY = "signagram.signedLanguage";
const SPOKEN_KEY = "signagram.spokenLanguage";
const VOICE_KEY = "signagram.voice";

/**
 * Voice is two slider positions, 1..5 each, NOT audio parameters. ElevenLabs
 * has no pitch knob -- pitch comes from which voice you use -- so the pair is
 * sent to the server, which maps it to a voice designed offline. These labels
 * exist so the UI can say what the numbers mean; they are not sent anywhere.
 */
export const VOICE_PITCH_LABELS = {
  1: "very deep", 2: "deep", 3: "mid-pitched", 4: "bright", 5: "very bright",
};
export const VOICE_AGE_LABELS = {
  1: "young adult", 2: "mid twenties", 3: "thirties",
  4: "middle-aged", 5: "elderly",
};
export const DEFAULT_VOICE = { pitch: 3, age: 3 };

function clampStep(v) {
  const n = Math.round(Number(v));
  return Number.isFinite(n) ? Math.min(5, Math.max(1, n)) : 3;
}

export function readVoice() {
  try {
    const raw = localStorage.getItem(VOICE_KEY);
    if (raw) {
      const v = JSON.parse(raw);
      return { pitch: clampStep(v.pitch), age: clampStep(v.age) };
    }
  } catch {
    // Private mode, or somebody hand-edited the value. The default is fine.
  }
  return { ...DEFAULT_VOICE };
}

export function writeVoice({ pitch, age }) {
  write(VOICE_KEY, JSON.stringify({ pitch: clampStep(pitch), age: clampStep(age) }));
}

function read(key, allowed, fallback) {
  try {
    const stored = localStorage.getItem(key);
    if (stored && stored in allowed) {
      return stored;
    }
  } catch {
    // Private-mode browsers throw on localStorage; the default is fine.
  }
  return fallback;
}

function write(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Non-fatal: the choice just will not survive a reload.
  }
}

export function readSignedLanguage() {
  return read(SIGNED_KEY, SIGNED_LANGUAGES, DEFAULT_SIGNED_LANGUAGE);
}

export function writeSignedLanguage(value) {
  write(SIGNED_KEY, value);
}

export function readSpokenLanguage() {
  return read(SPOKEN_KEY, SPOKEN_LANGUAGES, DEFAULT_SPOKEN_LANGUAGE);
}

export function writeSpokenLanguage(value) {
  write(SPOKEN_KEY, value);
}

/** The language the signing user reads. */
export function signerLanguage() {
  return SIGNED_LANGUAGES[readSignedLanguage()];
}

/** The language the speaking user speaks and reads. */
export function speakerLanguage() {
  return SPOKEN_LANGUAGES[readSpokenLanguage()];
}
