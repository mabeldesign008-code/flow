# Architecture

## Pipeline

```
                    ┌── app context (UIA)  ~5 ms, in parallel
                    │   process, window title, control type
                    ▼
always-on 16 kHz stream ──► 500 ms pre-roll ring
                              │
                              ├──► streaming socket (live partials)
                              │
                              │
        hotkey (Ctrl+Win) ────┤  mic is ALREADY live
                              ▼
                     energy VAD + RMS levelling      ~3 ms
                              ▼
                     AssemblyAI                      ~0.5 s tail
                     streaming, else batch (~2 s)
                     + keyterms (user dictionary)
                              ▼
                     Groq llama-3.1-8b-instant       ~200 ms
                     + low-confidence words
                     + per-app profile instruction
                              ▼
                     ┌─────  GUARD  ─────┐
                     │ negation flipped? │──► reject, use raw
                     │ number changed?   │
                     │ dict term gone?   │
                     │ >45% rewritten?   │
                     └─────────┬─────────┘
                               ▼
                  wait for modifier release
                               ▼
                     inject at cursor
                               ▼
                     auto-learn repeated terms
```

No fallback engines anywhere. A silent downgrade to a weaker model is worse
than an honest error: the user cannot tell their text got worse, so they
ship it.

## Modules

| Path | Role |
|---|---|
| `main.py` | Orchestration, settings window, tray, hotkeys |
| `audio/capture.py` | Always-on stream + pre-roll ring buffer |
| `audio/process.py` | VAD, high-pass, RMS levelling |
| `stt/assemblyai_client.py` | Transcription |
| `stt/dictionary.py` | User dictionary → keyterms |
| `stt/streaming.py` | Live partials over WebSocket |
| `context/app_context.py` | Foreground app via UI Automation |
| `context/profiles.py` | Per-app formatting rules |
| `context/learner.py` | Auto-learn dictionary terms |
| `refine/refiner.py` | Groq cleanup |
| `refine/guard.py` | **Hallucination rejection** |
| `ui/overlay.py` | Floating pill |
| `ui/theme.py` | Design tokens |
| `injector.py` | Text insertion |

---

## Why the mic is always on

**Audit 1.7.** The old code did this:

```python
winsound.Beep(1000, 100)   # blocks 100 ms
self.recorder.start()      # PortAudio open: another 50-300 ms
```

The mic went live 150–400 ms *after* the keypress. Users start talking
immediately, so **every utterance lost its onset** — "delete" → "lete",
"can't" → "ant".

Now one 16 kHz mono stream opens at launch and never closes, continuously
filling a 500 ms ring buffer. On hotkey we snapshot the ring and prepend
it. The onset is captured before the user's finger leaves the key. The beep
is fired on a thread.

The callback does exactly one thing — copy into the ring. The old one took
a lock, sorted a 50-element list, computed a median and called
`logger.warning()`, any of which can block and cause an xrun, which drops
frames, which raises WER.

## Why peak normalisation is gone

**Audit 1.8.** `audio * (0.9 / peak)` meant a keyboard click rescaled the
whole utterance around itself, and a quiet clip got boosted 20 dB along
with its noise floor.

Now: RMS-target −24 dBFS, computed over **voiced frames only**, gain capped
at +12 dB, with a limiter. Measured — a 0.99 transient click changes the
applied gain by <60% instead of dominating it entirely.

## Why the silence gate is gone

**Audit 1.9.** The old threshold was `max(3 × noise_floor, 0.01)`. An RMS
floor of 0.01 is ≈ **−40 dBFS**, which is loud for a laptop mic at arm's
length and far above whispered speech. That is why valid recordings came
back "No speech detected".

Now: noise floor from the quietest 20% of frames (robust to a clip that
opens on speech), threshold at `max(floor × 2.5, peak × 0.08)`, 250 ms
padding either side so plosive onsets and trailing fricatives survive.

Verified detection down to `amp=0.008` — a faint whisper the old gate
discarded outright.

**When VAD finds nothing, the whole clip is sent to the ASR anyway.** Our
VAD being wrong is far likelier than a real model finding nothing, and a
wrong guess would silently delete what the user said.

---

## The guard

The most important 150 lines in the codebase (`refine/guard.py`).

An LLM given only a 1-best transcript and told to "fix misheard words" will
convert *recognition* errors (visible, obviously wrong) into *fluent* errors
(invisible, plausibly wrong). The old defence was a prompt rule —
`"6. NEVER flip negations"`. A prompt rule is a hope.

| Check | Rationale |
|---|---|
| Negation count changed | `"can't ship"` → `"can ship"` inverts meaning |
| Number changed / dropped | `"$50"` → `"$15"` is catastrophic and invisible |
| Dictionary term dropped | The LLM "corrected" your product name away |
| >45% of words rewritten | Standard mode should correct, not reinvent |
| Much longer than input | Content was invented |

Two subtleties live testing forced:

- **Short text skips the ratio check.** A 3-word phrase hits 50% from one
  legitimate change.
- **`unable`/`impossible` count as negations**, so `"cannot ship"` →
  `"unable to ship"` stays balanced (allowed) while → `"can ship"` does not.

**It caught a real hallucination.** Given `"i cant make the meeting at three
thirty please reschedule"`, llama-3.1-8b returned:

> *"I'm afraid I won't be able to make the meeting at 3:30. Could we reschedule?"*

Politeness never spoken. Rejected → prompt hardened → 0/7 false rejections
afterwards.

## Evidence-based refinement

The refiner gets more than a string:

```
KNOWN TERMS: Kubernetes, WhisprFlow, Groq
UNCERTAIN: cuban, artists
TEXT:
deploy to cuban artists cluster
```

`UNCERTAIN` comes from AssemblyAI's per-word confidences. Correction becomes
*selection among candidates* — which LLMs do reliably — not free-form
guessing, which they don't.

`llama-3.1-8b-instant`, not 70b: at temperature 0 the small model handles
punctuation, casing and filler just as well at a fraction of the latency.

---

## The overlay

**Audit 1.19.** The old one rendered at 15 fps *forever*, even collapsed to
a 2 px sliver: a full `Image.new` at 2× supersample, a LANCZOS resize, a new
`ImageTk.PhotoImage` and `canvas.delete("all")` — fifteen times a second, on
the Tk main thread. `_draw_success` allocated a font every frame.

Now a dirty-flag loop: it renders at 60 fps while something is moving and
**stops entirely** when settled, polling at 8 Hz. The rest state is one
cached bitmap. Fonts load once.

`Spring.update()` also took a hardcoded `dt=0.016` in `_draw_waves` but real
`dt` in `update_position`, so animation speed differed between the 15 fps and
30 fps paths. All springs now integrate real `dt`, clamped at 50 ms so a
stalled main thread can't launch them.

### States

| State | Appearance |
|---|---|
| Idle | 96×8 px pill, 9 dots, no redraws |
| Hover | Expands to 232×44, shows the hotkey hint |
| Recording | 13-bar waveform, breathing glow, cancel + stop |
| Processing | Three cascading dots |
| Success | Green check + preview, springs in with overshoot |
| Error | Red cross + short reason, click to retry |

The waveform has centre-weighted response (centre bars react most, edges
trail) and an idle breathe so it never looks frozen in a silent room.

---

## Injection

**Audit 1.4.** Push-to-talk stops on key *release*, but only one of Ctrl/Win
has come up at that moment. If Win is still down the OS sees **Ctrl+Win+V**
(clipboard history), not paste. `wait_for_modifier_release()` polls
`GetAsyncKeyState` before sending anything.

**Audit 1.5.** The app ran two global keyboard hooks — the `keyboard`
package in the injector and `pynput` in main. Two low-level hooks intercept
each other's synthetic events. Now pynput only; the `keyboard` dependency
is gone.

Short text (≤120 chars) is typed directly as unicode — no clipboard
round-trip, works in fields that block paste. Longer text uses clipboard
with save/restore.

**Undo** sends `Ctrl+Z` rather than replaying N backspaces (audit 1.3). Most
apps coalesce a paste into one undo step, and blind backspaces eat real text
when the caret has moved.

---

## Verified against live APIs

Full pipeline, real generated speech:

```
[1] preprocess  2.4 ms   speech=True  dur=5.88s  gain=0.56  peak=0.37
[2] ASR      2401 ms   conf=0.994
    "Um, so I think we should not merge the Kubernetes changes until…"
[3] refine    216 ms   guard=accepted
    "So I think we should not merge the Kubernetes changes until…"

negation preserved    ✓
Kubernetes correct    ✓
filler removed        ✓
no invented content   ✓
```

Earlier clips measured **0% WER** at 99.6% confidence.

The live key also caught an API change no research surfaced: `speech_model`
(singular) is **deprecated**; it is now `speech_models: [...]`. Every
documented example online is stale.

## Tests

```
python -m pytest -q          # 81 tests
python eval/mock_api_test.py # HTTP flow against a fake AssemblyAI
python eval/run_wer.py       # WER on your own clips
```


---

## Streaming

Batch transcription makes the user finish speaking and *then* wait ~2 s
staring at nothing. Streaming uploads audio while they talk, so the wait
after they stop is only the tail.

Measured against the live API on a 7.44 s utterance:

| | Batch | Streaming |
|---|---|---|
| First feedback | none until done | **1.04 s** (partial text) |
| Wait after speech ends | ~2400 ms | **505 ms** |

That is a **79% cut in perceived latency** for identical compute.

The overlay switches from waveform to live text as soon as words arrive —
watching your sentence appear is far better feedback than a bouncing
equaliser. Text is measured against the gap between the cancel and stop
buttons and trimmed to fit, so it can never collide with them.

Audio is buffered locally throughout, so if the socket fails we finalise
from the batch endpoint and nothing is lost. This is the one fallback worth
keeping: it is between two paths to the *same* model, not a silent
downgrade to a weaker one.

## App context

The deleted `screen_context.py` screenshotted the whole desktop and ran
PaddleOCR on it: 0.3–2 s on the critical path, low-signal text (taskbar,
clock, unrelated windows), and your entire screen shipped to a third party
on every dictation.

`context/app_context.py` reads the same information from the OS:

| Source | Cost | Yields |
|---|---|---|
| `GetForegroundWindow` | ~0.1 ms | process name, window title |
| UI Automation | ~2–10 ms | focused control type, optionally its text |

Captured **at hotkey-down on a worker thread**, never after speech ends.
UIA calls are budgeted at 350 ms and disabled after three consecutive
failures, so a hung accessibility client can never stall dictation.

Reading the focused field's existing text is the most useful signal and the
most privacy-sensitive, so it is a separate toggle, off by default
(`WHISPRFLOW_READ_FIELD=1`).

## Per-app profiles

A profile matched on the process name contributes one instruction to the
refinement prompt. Verified live:

| App | Input | Output |
|---|---|---|
| VS Code | "a function called get user data that takes user id" | `get_user_data` … `user_id` |
| Terminal | "git checkout dash b feature slash new login" | `git checkout -b feature/new-login` |
| Slack | "hey can you take a look at the pr" | "hey can you take a look at the PR" |
| Outlook | "um so i wanted to follow up on the invoice" | "I wanted to follow up on the invoice." |

Profiles are **appended** to the base prompt, never substituted, so the
meaning-preserving rules always apply. Verified: negations and amounts
survive even under the most permissive profile (Email, `allow_restructure`).

Code and Terminal set `allow_restructure=False` — literal words matter
there, and paraphrasing them is destructive.

Users extend or override via `%APPDATA%/WhisprFlow/profiles.json`. A
malformed file is logged and ignored, never fatal.

## Auto-learn

Wispr Flow's real moat is that the dictionary fills itself. Theirs watches
you type over a correction; **we deliberately do not keylog** — reading
everything typed after a dictation would be a worse privacy trade than the
OCR we just removed.

Two safe signals instead:

1. **Repetition** — an unusual token transcribed N times across sessions
2. **Uncertainty** — a token AssemblyAI repeatedly flags low-confidence
   (weighted double: a word the recogniser keeps struggling with is a
   strong signal it belongs in the dictionary)

A heuristic filters ordinary English: internal capitals (`WhisprFlow`),
digits (`GPT4`), hyphens (`on-prem`), or capitalised proper nouns qualify;
`running`, `something`, `about` do not.

Candidates are **surfaced for one-click approval**, never added silently —
a dictionary that fills itself with garbage is worse than an empty one.
Dismissals are permanent.

> A bug the tests caught: `UserDictionary` defines `__len__`, so an *empty*
> dictionary is falsy. `if not self.dictionary` bailed out precisely when
> the dictionary was empty, meaning the very first learned term could never
> be added. Now `is None`.
