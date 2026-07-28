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
hotkey → record → AssemblyAI (local fallback) → Groq LLM cleanup → paste
```

| Stage | Component |
|---|---|
| Capture | `sounddevice` → `recorder.py` |
| Transcription | **AssemblyAI Universal-3.5 Pro**, sherpa-onnx local fallback |
| Dictionary | `%APPDATA%\WhisprFlow\user_dictionary.txt` → keyterms + hotwords |
| Refinement | Groq `llama-3.3-70b-versatile` |
| Injection | Clipboard paste with save/restore |

Full engine documentation: [`docs/STT_ENGINES.md`](docs/STT_ENGINES.md)

---

## Configuration

`.env` in the project root:

```
ASSEMBLYAI_API_KEY=...      # required for cloud accuracy
GROQ_API_KEY=...            # optional, for LLM refinement
ASSEMBLYAI_MODEL=universal-3-5-pro
STT_POLICY=cloud_first      # cloud_first | cloud_only | local_only
LOCAL_STT_MODEL=parakeet    # parakeet | sensevoice
```

Keys can also be set in the UI (system tray → Transcription History).

### Custom dictionary

Add your names, jargon, acronyms and product names to
`%APPDATA%\WhisprFlow\user_dictionary.txt` — one per line. This is the
single biggest accuracy improvement available.

### Offline mode

Set `STT_POLICY=local_only`. Requires a local model in `models/parakeet/`
(Parakeet TDT 0.6B, ~465 MB, [k2-fsa releases](https://github.com/k2-fsa/sherpa-onnx/releases)).
No audio leaves the machine.

---

## Hotkeys

| Key | Action |
|---|---|
| `Ctrl + Win` (hold) | Record |
| `Ctrl + Alt + Z` | Undo last injection |

---

## Development

```bash
python -m pytest test_stt.py -v        # STT layer (36 tests)
python -m pytest test_recorder.py -v   # audio capture
python eval/mock_api_test.py           # end-to-end HTTP flow (27 checks)
python eval/run_wer.py --engine assemblyai   # accuracy on your own clips
```

### Layout

```
main.py                   app, UI, pipeline
recorder.py               audio capture
injector.py               text injection
groq_client.py            LLM refinement
screen_context.py         OCR context (being replaced by UI Automation)
stt/
  base.py                 TranscriptionResult + engine protocol
  assemblyai_client.py    cloud engine
  local_client.py         sherpa-onnx offline engine
  router.py               policy + fallback
  dictionary.py           user dictionary
eval/
  run_wer.py              accuracy harness
  mock_api_test.py        integration test
```

---

## Requirements

Windows 10/11, Python 3.9+, a microphone, and an internet connection for
cloud transcription (offline mode available).

## Known limitations

Windows-only (`winsound`, `ctypes.windll`). Screen-context OCR is disabled
by default pending the UI Automation rewrite — see `AUDIT.md` §3.4 and §4.
