# Frontend

Static pages -- no build step, no `package.json`. Plain HTML, CSS, and ES modules.

| Page | What it does |
| --- | --- |
| `index.html` | Sign to text (still a mockup) |
| `speech.html` | Speech to text, live, via ElevenLabs Scribe v2 Realtime |
| `settings.html` | Signed language choice, which drives everything else |

## Running it

```
python frontend/serve.py
```

Then open <http://127.0.0.1:8080/speech.html>.

**Open it from `serve.py`, not from VS Code Live Preview or any other static
server.** `speech.js` posts to the relative path `/api/scribe-token`, which only
`serve.py` implements, so a static server answers it with a 404 and the Listen
button reports `token request failed (404)`. Port 8080, not 8000: the
sign-language backend (`backend/app/main.py`) owns 8000, and two servers
fighting over one port is what caused that 404 in the first place. Override with
`PORT=8081 python frontend/serve.py` if 8080 is taken.

Opening the files directly off disk will not work: `getUserMedia` requires a
secure context, and `file://` is not one, so the microphone is unavailable.
`http://127.0.0.1` counts as secure.

## The ElevenLabs key

Put it in the project-root `.env` (already gitignored):

```
ELEVENLABS_API_KEY=sk_...
```

`serve.py` prints `elevenlabs key : found` on startup when it picked one up. An
`ELEVENLABS_API_KEY` environment variable takes precedence over the file.

The key never reaches the browser. `serve.py` exposes `POST /api/scribe-token`,
which trades the key for a single-use token from ElevenLabs; the page passes
that token in the WebSocket query string. Tokens expire after about 15 minutes,
and a fresh one is fetched every time you press Listen.

## How the speech page works

Press **Listen** and the page opens a WebSocket directly to
`wss://api.elevenlabs.io/v1/speech-to-text/realtime`. Audio does not pass
through `serve.py`.

1. `getUserMedia` grabs the mic with echo cancellation and noise suppression on.
2. An `AudioContext` at 16 kHz feeds `pcm-worklet.js`, which converts float
   samples to 16-bit PCM and emits a 100 ms chunk at a time. No resampling is
   needed because the context is already at the rate the API wants.
3. Each chunk goes out base64-encoded as an `input_audio_chunk` message.
4. `partial_transcript` messages render greyed out -- the model may still revise
   those words. `committed_transcript` messages append to the settled text in
   black. The commit strategy is `vad`, so the server decides when a speech
   segment has ended.

Each chunk also carries a level reading, which drives the bars in the mic panel
instead of the idle CSS animation.

## Two users, two languages

There are two people in front of this app, and each page is written for the
*other* one:

- The **signer** picks a signed language in Settings, which implies the language
  they read: ASL means English, ISL means Hindi, CSN means Colombian Spanish.
- The **speaker** picks the spoken/text language, which is what they say out loud.

So the **speech page** is what the signer reads, and the **sign page** is what the
speaker reads. Neither setting derives the other, and a mismatched pair like ASL
alongside Spanish is a normal configuration, not a mistake.

Say the signed language is ASL and the spoken/text language is Spanish:

| Page | Input | Output |
| --- | --- | --- |
| Speech | listens for **Spanish** | shows **English** (for the ASL signer) |
| Sign | ASL signing | shows **Spanish** (for the speaker) |

Both settings and both mappings live in `languages.js`, and both persist in
`localStorage`.

The sign page currently only has the language plumbing -- recognition is still a
mockup, so `sign.js` just labels the output panel.

## Translation

Whenever the two languages differ, the speech page transcribes in the speaker's
language and then translates into the signer's:

```
Spanish speech -> Scribe (language_code=es) -> Spanish text -> translate es->en -> English
```

**ElevenLabs cannot do this leg.** Realtime Scribe has no translation parameter --
`language_code` only tells it what to expect on the way in -- and their only
translation product is the Dubbing API, which is file-based and takes minutes.

So `translate.js` uses Chrome's built-in Translator API instead: on-device, no API
key, no network round trip per phrase. The first use of a language pair downloads
a model, and progress shows in the status line.

Three cases fall back to passing text through untranslated. Matching languages
need no translation at all. A browser without the Translator API (anything
non-Chromium) keeps transcribing and says so in a note under the header rather
than pretending to translate. And if setting up a translator fails, the same note
explains why.

### Diagnosing translation failures

Open <http://127.0.0.1:8080/translator-check.html> with DevTools open.

It probes every direction the app can ask for and shows whether `create()` really
works, because `availability()` cannot tell you. Chrome reports *every* pair as
`downloadable` until a site creates a translator for it -- it hides real pack
availability for privacy -- so a pair that will never work looks identical to one
that will.

When a pair is refused, Chrome throws a deliberately generic `NotSupportedError`
("Unable to create translator for the given source and target language") but logs
the specific reason as a **console warning**. That warning is the only place the
real cause appears, which is why the console matters here.

Committed transcripts are translated in order. Partial transcripts arrive about
ten times a second, so they are debounced and sequence-numbered -- a slow
translation cannot overwrite newer text. When no translation is needed, partials
skip the debounce and update immediately.
