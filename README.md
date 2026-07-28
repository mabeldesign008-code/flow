# WhisprFlow

System-wide AI dictation for Windows. Hold a hotkey, speak, and polished
text appears at your cursor in any application.

> **Status:** actively being reworked. See [`AUDIT.md`](AUDIT.md) for the
> full engineering audit and roadmap. Phase 1 (transcription engine) is done.

---

## Quick start

```bash
git clone https://github.com/mabeldesign008-code/flow.git
cd flow
pip install -r requirements.txt

# Configure AssemblyAI (free $50 credit, no credit card)
python setup_stt.py

python main.py
```

Hold **Ctrl + Win**, speak, release. Text is injected at the cursor.

---

## How it works

```
always-on mic (500 ms pre-roll) → VAD → AssemblyAI → Groq → guard → paste
```

The mic is open before you press the key, so the first syllable is never
lost. Every LLM rewrite is checked before injection.

The guard rejects any rewrite that flips a negation, changes a number, or
drops a dictionary term — the raw transcript is used instead. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

| Stage | Component |
|---|---|
| Capture | Always-on 16 kHz stream + pre-roll ring (`audio/`) |
| Transcription | **AssemblyAI Universal-3.5 Pro** + keyterms |
| Dictionary | `%APPDATA%\WhisprFlow\user_dictionary.txt` |
| Refinement | Groq `llama-3.1-8b-instant`, output verified by a guard |
| Injection | Direct unicode ≤120 chars, else clipboard w/ restore |

Architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## Configuration

`.env` in the project root:

```
ASSEMBLYAI_API_KEY=...      # required for cloud accuracy
GROQ_API_KEY=...            # optional, for LLM refinement
ASSEMBLYAI_MODEL=universal-3-5-pro
```

Keys can also be set in the UI (system tray → Transcription History).

### Custom dictionary

Add your names, jargon, acronyms and product names to
`%APPDATA%\WhisprFlow\user_dictionary.txt` — one per line. This is the
single biggest accuracy improvement available.

## Hotkeys

| Key | Action |
|---|---|
| `Ctrl + Win` (hold) | Dictate |
| `Ctrl + Alt + Z` | Undo |

Click the left of the pill to cancel mid-recording, the right to stop
early, or the pill itself after an error to retry.

---

## Development

```bash
python -m pytest -q                    # 81 tests
python eval/mock_api_test.py           # end-to-end HTTP flow
python eval/run_wer.py                 # accuracy on your own clips
```

### Layout

```
main.py                   app, settings window, tray, pipeline
audio/
  capture.py              always-on stream + pre-roll ring
  process.py              VAD, high-pass, RMS levelling
ui/
  overlay.py              floating pill
  theme.py                design tokens
injector.py               text insertion
stt/
  base.py                 TranscriptionResult + WAV encoding
  assemblyai_client.py    transcription engine
  dictionary.py           user dictionary
refine/
  refiner.py              Groq LLM cleanup
  guard.py                hallucination rejection
eval/
  run_wer.py              accuracy harness
  mock_api_test.py        integration test
```

---

## Requirements

Windows 10/11, Python 3.9+, a microphone, and an internet connection for
transcription.

## Known limitations

Windows-only (`winsound`, `ctypes.windll`). Requires an internet
connection — there is no offline mode. Screen-context OCR was removed; app
context via UI Automation is the planned replacement (`AUDIT.md` §3.4).
