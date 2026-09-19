import {
  readSignedLanguage,
  writeSignedLanguage,
  readSpokenLanguage,
  writeSpokenLanguage,
} from "./languages.js";

const signedSelect = document.getElementById("signedLanguage");
const spokenSelect = document.getElementById("spokenLanguage");

signedSelect.value = readSignedLanguage();
spokenSelect.value = readSpokenLanguage();

signedSelect.addEventListener("change", () => writeSignedLanguage(signedSelect.value));
spokenSelect.addEventListener("change", () => writeSpokenLanguage(spokenSelect.value));
