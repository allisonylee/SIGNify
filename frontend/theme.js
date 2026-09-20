/**
 * Manual theme, applied before first paint so a saved choice cannot flash
 * the wrong scheme. Loaded as a classic script in <head> on every page.
 *
 * Three choices: "system" (default), "light", "dark". The resolved scheme
 * is written to <html data-theme>, which is what style.css keys off.
 */
(function () {
  const KEY = "signify-theme";
  const root = document.documentElement;

  function systemTheme() {
    return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function stored() {
    try {
      const value = localStorage.getItem(KEY);
      if (value === "light" || value === "dark" || value === "system") return value;
    } catch {
      /* private mode, quota, or storage blocked */
    }
    return "system";
  }

  function resolved(choice) {
    return choice === "system" ? systemTheme() : choice;
  }

  function apply(choice) {
    const theme = resolved(choice);
    root.dataset.theme = theme;
    root.style.colorScheme = theme;
    const meta = document.querySelector('meta[name="color-scheme"]');
    if (meta) meta.content = theme;
  }

  function syncControl(choice) {
    document.querySelectorAll("[data-theme-choice]").forEach((button) => {
      const on = button.dataset.themeChoice === choice;
      button.setAttribute("aria-checked", String(on));
    });
  }

  function setChoice(choice) {
    try {
      localStorage.setItem(KEY, choice);
    } catch {
      /* persist is best-effort; the page still flips */
    }
    apply(choice);
    syncControl(choice);
  }

  apply(stored());

  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (stored() === "system") apply("system");
  });

  function bind() {
    const group = document.getElementById("appearance");
    if (!group) return;
    syncControl(stored());
    group.addEventListener("click", (event) => {
      const button = event.target.closest("[data-theme-choice]");
      if (!button) return;
      setChoice(button.dataset.themeChoice);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bind);
  } else {
    bind();
  }

  /* Animate later flips, not the first paint. */
  requestAnimationFrame(() => {
    root.dataset.themeReady = "";
  });
})();
