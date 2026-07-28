# Transcription engines

## Why AssemblyAI replaced SenseVoice

The app was hard-locked to English (`language="en"` in three places plus a
whole `_enforce_english` layer) while running the model with the *worst*
English accuracy of any mainstream option.

| Engine | English WER | Notes |
|---|---|---|
| SenseVoice-Small (removed as primary) | **~14.7%** | Built for Chinese/Japanese/Korean/Cantonese. CTC → no hotwords possible. |
| Whisper large-v3 | ~9.4% | |
| Parakeet TDT 0.6B v2 (local fallback) | ~6.1% | Transducer → hotwords work. 465 MB. |
| **AssemblyAI Universal-3.5 Pro (now primary)** | **~2–5% clean / 7.7% hard** | Best-in-class; keyterms prompting. |

Sources: [SenseVoice vs Whisper benchmark](https://whispernotes.app/blog/sensevoice-fastest-cjk-transcription),
[AssemblyAI accuracy report](https://www.assemblyai.com/blog/how-accurate-speech-to-text),
[Open ASR Leaderboard summary](https://localaimaster.com/blog/parakeet-vs-whisper).

That is roughly a **3–5× reduction in raw error rate** before any other fix.

### Free tier

$50 credit, **no credit card required** — about **185 hours** of Universal
transcription. At realistic dictation volume (~30 min/day) that is over a
year of use. Sign up: https://www.assemblyai.com/dashboard/signup

---

## Architecture

```
main.py
  └── EngineRouter                    stt/router.py
        ├── AssemblyAIClient          stt/assemblyai_client.py   (primary)
        ├── LocalSherpaClient         stt/local_client.py        (fallback)
        └── UserDictionary            stt/dictionary.py          (keyterms)
```

Every engine returns a `TranscriptionResult` (`stt/base.py`) so they are
interchangeable — swapping providers later means writing one class, not
touching the pipeline.

### Policies

| `STT_POLICY` | Behaviour |
|---|---|
| `cloud_first` *(default)* | AssemblyAI; falls back to local on failure |
| `cloud_only` | AssemblyAI only; fails loudly |
| `local_only` | Offline, never touches the network (privacy mode) |

Fallback fires **only** on `ok=False`. An empty-but-`ok` result means the
user was genuinely silent — retrying locally would waste a second and risk
inventing text.

---

## Configuration

`.env` in the project root:

```
ASSEMBLYAI_API_KEY=your_key_here
ASSEMBLYAI_MODEL=universal-3-5-pro
STT_POLICY=cloud_first
LOCAL_STT_MODEL=parakeet
GROQ_API_KEY=your_groq_key
```

Or set the key in the UI: **Transcription Engine → AssemblyAI API Key → Save**.

### Models

| Model | Price/hr | Keyterms | Use |
|---|---|---|---|
| `universal-3-5-pro` | $0.21 | ✅ | Default — highest accuracy |
| `universal-3-pro` | $0.21 | ✅ | Previous flagship |
| `slam-1` | $0.27 | ✅ (1000 terms, semantic) | Heavy domain jargon |
| `universal-2` | $0.15 | ❌ | Cheapest; keyterms auto-suppressed |

---

## User dictionary

The highest-ROI accuracy feature, and the app previously had none.

Location: `%APPDATA%\WhisprFlow\user_dictionary.txt` — plain text, one term
per line, `#` for comments.

```
Kubernetes
WhisprFlow
Mabel Design
Accra
```

Terms are routed to all three stages:
- **AssemblyAI** → `keyterms_prompt` (semantic biasing, not just likelihood)
- **Local** → sherpa-onnx `hotwords_file` (transducer + `modified_beam_search`)
- **LLM** → refinement prompt, so Groq stops "correcting" your product names

Limits: 1000 terms, max 6 words per phrase, 50 chars per term (AssemblyAI's).

---

## Bugs fixed in this change

| Bug | Old behaviour | Now |
|---|---|---|
| **Silent data loss** | `_enforce_english()` returned `""` — discarding the whole transcript — on any non-English tag or >30% non-ASCII. `"— €50"` scores 0.5 and was deleted. | Never discards. Tag stripping is anchored to the known token set. |
| Error vs silence conflated | Both surfaced as "Empty transcription" | `ok=False` + `error` vs `ok=True` + empty text |
| Model-load hang | Exception died in a daemon thread; every later call blocked 30 s then raised `TimeoutError` | Error captured, surfaced in UI immediately |
| Temp WAV leak | `NamedTemporaryFile(delete=False)`, never cleaned up | In-memory encoding, no temp files |
| int16 overflow | `np.int16(x * 32767)` wraps → loud noise bursts | Clipped before conversion |
| Regex over-match | Catch-all `[A-Z_]+` ate legitimate `<\|A\|>` text | Anchored to real SenseVoice tokens |
| Implicit resampling | Captured at 44.1/48 kHz, sherpa resampled silently | Explicit `resample_poly` to 16 kHz |
| `h2` missing → crash | `http2=True` raised `ImportError` at construction | Detected; falls back to HTTP/1.1 |
| CWD-relative config | `.env` relative to working dir; shortcut launch lost settings | Dictionary in `%APPDATA%` |

---

## Measuring accuracy

Never trust vendor benchmarks — measure on your own audio.

```bash
# 1. Put WAVs in eval/clips/
# 2. Write eval/reference.jsonl:
#    {"file": "clip001.wav", "text": "what you actually said"}

python eval/run_wer.py --engine assemblyai
python eval/run_wer.py --engine assemblyai --dictionary
python eval/run_wer.py --engine local --local-model parakeet
python eval/run_wer.py --engine local --local-model sensevoice
```

Reports overall WER, median WER, perfect-clip rate, latency, and a
substitution/deletion/insertion breakdown. Clips over 15% WER are flagged
with `<<<` so you can find the hard cases fast.

---

## Testing

```bash
python -m pytest test_stt.py -v      # 36 unit tests
python eval/mock_api_test.py         # 27 checks against a fake AssemblyAI server
```

The mock server exercises the real upload → submit → poll HTTP flow,
verifies the uploaded WAV is intact 16 kHz mono 16-bit, confirms keyterms
are sent (and suppressed on `universal-2`), and covers job errors, HTTP
400s, missing keys, and router fallback.
