/**
 * Transcript translation on Chrome's built-in Translator API: on-device, no API
 * key, no network round trip per phrase.
 *
 * ElevenLabs cannot do this leg. Realtime Scribe has no translation parameter --
 * `language_code` only tells it what to expect on the way in -- and their only
 * translation product is the Dubbing API, which is file-based and takes minutes.
 *
 * Four situations collapse into a passthrough translator rather than an error, so
 * callers always get the same interface and only consult `reason` when they want
 * to explain themselves in the UI: both sides already share a language, the
 * browser has no Translator API, setup failed, or setup stalled.
 *
 * That last case is not hypothetical. Some Chromium embeddings expose
 * `Translator` while the model service behind it is absent, and then
 * `availability()` returns a promise that simply never settles -- so every wait
 * here is bounded. Transcription must never be held up by translation setup.
 */

export const SAME_LANGUAGE = "same-language";
export const UNSUPPORTED = "unsupported";
export const FAILED = "failed";

const AVAILABILITY_TIMEOUT_MS = 8000;
const SETUP_IDLE_TIMEOUT_MS = 15000;

function passthrough(reason, error = null) {
  return {
    passthrough: true,
    reason,
    error,
    async translate(text) {
      return text;
    },
    destroy() {},
  };
}

/**
 * Rejects once `promise` has gone `idleMs` without settling or reporting
 * activity. Downloads can legitimately run long, so progress events push the
 * deadline back; a silent stall still ends the wait.
 */
function withIdleTimeout(promise, idleMs, label, activity) {
  return new Promise((resolve, reject) => {
    const timer = setInterval(() => {
      if (Date.now() - activity.at >= idleMs) {
        clearInterval(timer);
        reject(new Error(`${label} stalled for ${Math.round(idleMs / 1000)}s`));
      }
    }, 500);

    const settle = (callback) => (value) => {
      clearInterval(timer);
      callback(value);
    };
    promise.then(settle(resolve), settle(reject));
  });
}

export function translationSupported() {
  return typeof Translator !== "undefined";
}

/**
 * Call this while user activation is still fresh. Downloading a language pack
 * requires it, and awaiting anything slow first -- the microphone permission
 * prompt especially -- can spend it.
 *
 * `onProgress` receives a 0..1 fraction, and only fires when a download happens.
 */
export async function createTranslator(sourceLanguage, targetLanguage, onProgress) {
  if (sourceLanguage === targetLanguage) {
    return passthrough(SAME_LANGUAGE);
  }
  if (!translationSupported()) {
    return passthrough(UNSUPPORTED);
  }

  const pair = { sourceLanguage, targetLanguage };
  let abandoned = false;

  try {
    const availability = await withIdleTimeout(
      Translator.availability(pair),
      AVAILABILITY_TIMEOUT_MS,
      "availability check",
      { at: Date.now() },
    );
    if (availability === "unavailable") {
      return passthrough(UNSUPPORTED);
    }

    const activity = { at: Date.now() };
    const creation = Translator.create({
      ...pair,
      monitor(monitor) {
        monitor.addEventListener("downloadprogress", (event) => {
          activity.at = Date.now();
          onProgress?.(event.loaded);
        });
      },
    });

    // If the wait below gives up, a late arrival would otherwise leak.
    creation.then((late) => abandoned && late.destroy()).catch(() => {});

    const translator = await withIdleTimeout(
      creation, SETUP_IDLE_TIMEOUT_MS, "translator setup", activity);

    return {
      passthrough: false,
      reason: "",
      error: null,
      translate: (text) => translator.translate(text),
      destroy: () => translator.destroy(),
    };
  } catch (error) {
    abandoned = true;
    return passthrough(FAILED, error);
  }
}
