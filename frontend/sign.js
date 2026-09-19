/**
 * Language plumbing for the sign page. Recognition is not wired up yet -- the
 * camera stage is still a mockup -- but the output language is settled: signing
 * is read by the SPEAKING user, so the text belongs in their spoken/text
 * language, independent of which signed language is being recognised.
 *
 * Whoever wires up recognition should send `target.code` as the output language.
 */
import { speakerLanguage } from "./languages.js";

const target = speakerLanguage();
const placeholder = document.getElementById("placeholder");

placeholder.textContent = `The translation will appear here in ${target.label}...`;
