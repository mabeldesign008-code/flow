# Architecture

## Pipeline

```
hotkey  →  record  →  AssemblyAI  →  Groq refine  →  GUARD  →  paste
                      universal-3-5-pro   llama-3.1-8b    reject?
                      + keyterms          + evidence      → raw text
```

No fallback engines. A silent downgrade to a weaker model is worse than an
honest error: the user cannot tell their text got worse, so they ship it.
If AssemblyAI fails, the UI says so and offers retry.

## Modules

| Path | Role |
|---|---|
| `main.py` | App, overlay, hotkeys, pipeline orchestration |
| `recorder.py` | Audio capture |
| `injector.py` | Clipboard paste with save/restore |
| `stt/base.py` | `TranscriptionResult`, WAV encoding |
| `stt/assemblyai_client.py` | The transcription engine |
| `stt/dictionary.py` | User dictionary → keyterms + guard |
| `refine/refiner.py` | Groq LLM cleanup |
| `refine/guard.py` | **Hallucination rejection** |

## The guard (`refine/guard.py`)

The most important 150 lines in the codebase.

An LLM given only a 1-best transcript and told to "fix misheard words" will
convert *recognition* errors (visible, obviously wrong) into *fluent* errors
(invisible, plausibly wrong). The old code tried to prevent this with a
prompt rule — `"6. NEVER flip negations"`. A prompt rule is a hope.

Every rewrite is now checked mechanically and **rejected** if:

| Check | Why |
|---|---|
| Negation count changed | `"can't ship"` → `"can ship"` inverts meaning |
| A number changed or vanished | `"$50"` → `"$15"` is catastrophic and invisible |
| A dictionary term was dropped | The LLM "corrected" your product name away |
| >45% of words rewritten | Standard mode should correct, not reinvent |
| Much longer than the original | Content was invented |

On rejection the raw transcript is used and the UI says why. Two subtleties
that live testing forced:

- **Short text skips the edit-ratio check.** A 3-word phrase hits 50% from
  one legitimate change. Meaning-preserving checks still apply at all lengths.
- **`unable`/`impossible` count as negations**, so `"cannot ship"` →
  `"unable to ship"` stays balanced and is allowed, while
  `"cannot ship"` → `"can ship"` is not.

This guard caught a real hallucination during development. Given
`"i cant make the meeting at three thirty please reschedule"`,
llama-3.1-8b returned:

> "I'm afraid I won't be able to make the meeting at 3:30. Could we reschedule?"

Politeness the user never said. Rejected, and the prompt was hardened.

## Evidence-based refinement

The refiner receives more than the raw string:

```
APP: Code.exe
KNOWN TERMS: Kubernetes, WhisprFlow, Groq
UNCERTAIN: cuban, artists
TEXT:
deploy to cuban artists cluster
```

`UNCERTAIN` comes from AssemblyAI's per-word confidences
(`result.low_confidence_words()`). Correction becomes *selection among
candidates* — which LLMs do reliably — rather than free-form guessing.

Model is `llama-3.1-8b-instant`, not 70b. At temperature 0 the small model
handles punctuation, casing and filler removal just as well, at a fraction
of the latency. Latency is the product.

## Configuration

```
ASSEMBLYAI_API_KEY=...          # required
GROQ_API_KEY=...                # optional; without it, deterministic cleanup
ASSEMBLYAI_MODEL=universal-3-5-pro
```

Dictionary: `%APPDATA%\WhisprFlow\user_dictionary.txt`

## Verified against live APIs

| Test | Result |
|---|---|
| Real speech → transcript | 0% WER, 99.6% confidence |
| Filler removal | `"Um, so I think..."` → `"So I think..."` |
| Dictionary term | `"Kubernetes"` correct |
| Negation preservation | `"should not merge"` survived |
| Guard on 7 dictation samples | 0 false rejections |

The live key also caught an API change no amount of research would have:
`speech_model` (singular) is **deprecated**; the API now requires
`speech_models: ["..."]`. The documented examples are stale.
