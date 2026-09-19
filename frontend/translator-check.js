/**
 * Diagnostic for Chrome's built-in Translator API. Not part of the app -- it
 * exists because `Translator.availability()` cannot answer the only question
 * that matters.
 *
 * Chrome reports every pair as `downloadable` until a site creates a translator
 * for it, deliberately, to avoid leaking which packs a user has installed. So a
 * pair that will never work looks identical to one that will, and `create()` is
 * the only real test. This page runs that test across every direction the app
 * can ask for and shows what actually happens.
 */
import { SIGNED_LANGUAGES, SPOKEN_LANGUAGES } from "./languages.js";

const AVAILABILITY_TIMEOUT_MS = 20000;
const CREATE_TIMEOUT_MS = 120000;

const SAMPLES = {
  en: "Hello, my name is Isabella and this is a live transcript.",
  es: "Hola, me llamo Isabella y esta es una transcripción en vivo.",
  hi: "नमस्ते, मेरा नाम इसाबेला है और यह एक लाइव प्रतिलेख है।",
};

const LABELS = { en: "English", es: "Spanish", hi: "Hindi" };

const runButton = document.getElementById("run");
const rowsBody = document.getElementById("rows");
const envLabel = document.getElementById("env");

/**
 * Every direction the app can ask for: the speaker's language in, the signer's
 * language out. Derived from the real settings so this cannot drift from them.
 */
function requiredPairs() {
  const sources = new Set(Object.values(SPOKEN_LANGUAGES).map((l) => l.code));
  const targets = new Set(Object.values(SIGNED_LANGUAGES).map((l) => l.code));
  const pairs = [];

  for (const source of sources) {
    for (const target of targets) {
      if (source !== target) {
        pairs.push({ source, target });
      }
    }
  }
  return pairs;
}

function withTimeout(promise, ms, label) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} never settled (${ms / 1000}s)`)), ms);
    const settle = (callback) => (value) => {
      clearTimeout(timer);
      callback(value);
    };
    promise.then(settle(resolve), settle(reject));
  });
}

function describeEnvironment() {
  const present = typeof Translator !== "undefined";
  envLabel.innerHTML = `<code>Translator</code> API: <code>${present ? "present" : "MISSING"}</code>`
    + ` &middot; secure context: <code>${window.isSecureContext}</code>`
    + ` &middot; ${navigator.userAgent}`;
  return present;
}

function addRow({ source, target }) {
  const row = document.createElement("tr");
  row.innerHTML = `<td class="pair">${LABELS[source]} &rarr; ${LABELS[target]}`
    + ` <span style="opacity:.6">(${source}&rarr;${target})</span></td>`
    + `<td data-state="waiting">waiting</td>`
    + `<td data-state="waiting">waiting</td>`
    + `<td data-state="waiting">waiting</td>`;
  rowsBody.append(row);

  const [, availability, creation, translation] = row.children;
  return { availability, creation, translation };
}

function set(cell, state, text) {
  cell.dataset.state = state;
  cell.textContent = text;
}

async function run() {
  runButton.disabled = true;
  rowsBody.replaceChildren();

  if (!describeEnvironment()) {
    const row = document.createElement("tr");
    row.innerHTML = `<td colspan="4" data-state="bad">This browser has no Translator API at all.</td>`;
    rowsBody.append(row);
    runButton.disabled = false;
    return;
  }

  const pairs = requiredPairs();
  const cells = pairs.map(addRow);

  // Both calls for every pair are fired here, synchronously, before anything is
  // awaited: pack downloads need user activation, and awaiting pairs one at a
  // time would spend the click's activation before the later pairs ran.
  const attempts = pairs.map(({ source, target }, index) => {
    const options = { sourceLanguage: source, targetLanguage: target };
    return {
      availability: Translator.availability(options),
      creation: Translator.create({
        ...options,
        monitor(monitor) {
          monitor.addEventListener("downloadprogress", (event) => {
            set(cells[index].creation, "waiting", `downloading ${Math.round(event.loaded * 100)}%`);
          });
        },
      }),
    };
  });

  await Promise.all(attempts.map(async ({ availability, creation }, index) => {
    const cell = cells[index];
    const pair = pairs[index];

    try {
      set(cell.availability, "waiting", await withTimeout(availability, AVAILABILITY_TIMEOUT_MS, "availability"));
    } catch (error) {
      set(cell.availability, "bad", error.message);
    }

    let translator = null;
    try {
      translator = await withTimeout(creation, CREATE_TIMEOUT_MS, "create");
      set(cell.creation, "good", "created");
    } catch (error) {
      set(cell.creation, "bad", `${error.name ?? "Error"}: ${error.message}`);
      set(cell.translation, "bad", "\u2014");
      return;
    }

    try {
      set(cell.translation, "good", await translator.translate(SAMPLES[pair.source]));
    } catch (error) {
      set(cell.translation, "bad", `${error.name ?? "Error"}: ${error.message}`);
    } finally {
      translator.destroy();
    }
  }));

  runButton.disabled = false;
}

describeEnvironment();
runButton.addEventListener("click", run);
