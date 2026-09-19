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
