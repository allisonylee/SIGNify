/**
 * v001 -- does the Settings pair actually route the way the Speech tab needs?
 *
 * NOT SHIPPED. Pure logic check on languages.js: the speech page must LISTEN in
 * the spoken/text language and DISPLAY the signed language's reading language.
 * Getting these backwards is invisible when both are English, which is the
 * default -- so it has to be checked with a mismatched pair.
 *
 *   node frontend/checks/v001_language_routing_check.mjs
 */
const store = new Map();
globalThis.localStorage = {
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
};

const L = await import("../languages.js");

const failed = [];
const check = (name, got, want) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) failed.push(name);
  console.log(`  [${ok ? "PASS" : "FAIL"}] ${name}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
};

console.log("\n1. defaults (both English -- the case that hides mistakes)");
check("   listen code", L.speakerLanguage().code, "en");
check("   display code", L.signerLanguage().code, "en");

console.log("\n2. the README's worked example: signed ASL, spoken Spanish");
L.writeSignedLanguage("ASL");
L.writeSpokenLanguage("Spanish");
check("   Scribe language_code (what we LISTEN for)", L.speakerLanguage().code, "es");
check("   transcript language (what we DISPLAY)", L.signerLanguage().code, "en");
check("   translator direction", [L.speakerLanguage().code, L.signerLanguage().code], ["es", "en"]);

console.log("\n3. reversed: signed ISL, spoken English -> en->hi");
L.writeSignedLanguage("ISL");
L.writeSpokenLanguage("English");
check("   translator direction", [L.speakerLanguage().code, L.signerLanguage().code], ["en", "hi"]);

console.log("\n4. every Settings option resolves (no undefined reaching Scribe)");
for (const k of Object.keys(L.SIGNED_LANGUAGES)) {
  L.writeSignedLanguage(k);
  check(`   signed ${k}`, typeof L.signerLanguage().code, "string");
}
for (const k of Object.keys(L.SPOKEN_LANGUAGES)) {
  L.writeSpokenLanguage(k);
  check(`   spoken ${k}`, typeof L.speakerLanguage().code, "string");
}

console.log("\n5. a junk stored value falls back instead of sending undefined");
store.set("signagram.signedLanguage", "KLINGON");
check("   bad signed value", L.readSignedLanguage(), L.DEFAULT_SIGNED_LANGUAGE);

console.log(`\n${failed.length ? "FAILURES: " + failed.join(", ") : "ALL PASS"}`);
process.exit(failed.length ? 1 : 0);
