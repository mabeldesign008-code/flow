"""
WhisprFlow — system-wide AI dictation for Windows.

Pipeline:  hotkey → always-on capture (with pre-roll) → AssemblyAI
           → Groq refinement → hallucination guard → inject at cursor
"""

import asyncio
import ctypes
import logging
import os
import threading
import time
import traceback
import winsound
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import Label, messagebox, scrolledtext

import pystray
from dotenv import load_dotenv, set_key
from PIL import Image, ImageDraw
from pynput import keyboard
from pynput.keyboard import Controller as KeyboardController

from audio import AudioCapture, process as process_audio
from injector import TextInjector
from refine import Refiner, basic_cleanup
from stt import AssemblyAIClient, UserDictionary, default_config_dir
from ui import theme
from ui.overlay import FloatingPill, PillState

# High-DPI awareness so the overlay renders crisply.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

CONFIG_DIR = default_config_dir()
ENV_PATH = CONFIG_DIR / ".env"
load_dotenv(ENV_PATH)
load_dotenv()  # also honour a project-local .env during development

DEBUG = os.getenv("WHISPRFLOW_DEBUG", "").lower() in ("1", "true", "yes")

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("whisprflow")


def beep_async(freq: int, ms: int) -> None:
    """Never block the caller. The old code called winsound.Beep inline
    before opening the mic, which is how the first syllable got clipped."""
    threading.Thread(
        target=lambda: _safe_beep(freq, ms), daemon=True
    ).start()


def _safe_beep(freq: int, ms: int) -> None:
    try:
        winsound.Beep(freq, ms)
    except Exception:
        pass


class WhisprFlowApp:
    def __init__(self):
        self.dictionary = UserDictionary()
        self.injector = TextInjector()
        self.refiner = Refiner(api_key=os.getenv("GROQ_API_KEY", ""))
        self.stt = AssemblyAIClient(
            api_key=os.getenv("ASSEMBLYAI_API_KEY", ""),
            model=os.getenv("ASSEMBLYAI_MODEL", "universal-3-5-pro"),
        )

        # Always-on capture. The stream opens once and stays open, so the
        # pre-roll ring already holds the moment before the hotkey fired.
        self.capture = AudioCapture()

        self._state_lock = threading.RLock()
        self.is_recording = False
        self.generation = 0          # bumped on cancel to invalidate in-flight work
        self.last_raw_text = ""
        self.last_audio = None
        self.last_text_injected = ""
        self.last_failed = False
        self._start_time = 0.0

        self.loop = asyncio.new_event_loop()

        # Hotkeys
        self.hotkey_combo = {keyboard.Key.ctrl_l, keyboard.Key.cmd}
        self.pressed_keys = set()
        self.kb_controller = KeyboardController()

        # Window
        self.root = tk.Tk()
        self.root.title("WhisprFlow")
        self.root.geometry("640x720")
        self.root.configure(bg=theme.HEX_BG_APP)
        self.root.protocol("WM_DELETE_WINDOW", self.hide_window)

        self.refinement_enabled = tk.BooleanVar(value=True)

        self._build_ui()
        self.pill = FloatingPill(
            self.root,
            get_level=self.capture.get_level,
            on_stop=self.stop_recording,
            on_cancel=self.cancel_recording,
            on_retry=self.retry,
        )
        self.root.withdraw()
        self.tray = self._build_tray()

    # ══════════════════════════════════════════════════════════════════
    #  UI
    # ══════════════════════════════════════════════════════════════════

    def _build_ui(self):
        wrap = tk.Frame(self.root, bg=theme.HEX_BG_APP)
        wrap.pack(fill="both", expand=True, padx=24, pady=24)

        Label(wrap, text="WhisprFlow", font=(theme.UI_FONT, 22, "bold"),
              fg=theme.HEX_TEXT, bg=theme.HEX_BG_APP).pack(anchor="w")
        Label(wrap, text="Hold Ctrl + Win to dictate anywhere",
              font=(theme.UI_FONT, 10), fg=theme.HEX_MUTED,
              bg=theme.HEX_BG_APP).pack(anchor="w", pady=(0, 18))

        # ── Engine ──
        card = self._card(wrap, "Transcription")
        info = self.stt.get_info()
        self.engine_label = Label(
            card, text=f"AssemblyAI · {info['model']}",
            font=(theme.UI_FONT, 11, "bold"),
            fg=theme.HEX_SUCCESS if info["configured"] else theme.HEX_WARNING,
            bg=theme.HEX_BG_CARD)
        self.engine_label.pack(anchor="w")
        self.engine_sub = Label(
            card,
            text=("Ready" if info["configured"] else "Add an API key to start"),
            font=(theme.UI_FONT, 9), fg=theme.HEX_MUTED, bg=theme.HEX_BG_CARD)
        self.engine_sub.pack(anchor="w", pady=(1, 12))

        self.aai_entry = self._key_row(card, "AssemblyAI API key",
                                       os.getenv("ASSEMBLYAI_API_KEY", ""),
                                       self.save_assemblyai_key)
        self.groq_entry = self._key_row(card, "Groq API key (refinement)",
                                        os.getenv("GROQ_API_KEY", ""),
                                        self.save_groq_key)

        # ── Refinement ──
        card2 = self._card(wrap, "Refinement")
        tk.Checkbutton(
            card2, text="Clean up transcripts with AI",
            variable=self.refinement_enabled, command=self._on_refine_toggle,
            bg=theme.HEX_BG_CARD, fg=theme.HEX_TEXT,
            selectcolor=theme.HEX_BG_INPUT, activebackground=theme.HEX_BG_CARD,
            activeforeground=theme.HEX_TEXT, font=(theme.UI_FONT, 10),
            borderwidth=0, highlightthickness=0,
        ).pack(anchor="w")
        self.refine_status = Label(card2, text="", font=(theme.UI_FONT, 9),
                                   fg=theme.HEX_MUTED, bg=theme.HEX_BG_CARD)
        self.refine_status.pack(anchor="w", pady=(4, 0))

        # ── Dictionary ──
        card3 = self._card(wrap, "Dictionary")
        self.dict_label = Label(
            card3, text=f"{len(self.dictionary)} terms",
            font=(theme.UI_FONT, 10), fg=theme.HEX_TEXT, bg=theme.HEX_BG_CARD)
        self.dict_label.pack(anchor="w")
        Label(card3, text="Names and jargon the recogniser should never guess at.",
              font=(theme.UI_FONT, 9), fg=theme.HEX_MUTED,
              bg=theme.HEX_BG_CARD).pack(anchor="w", pady=(1, 8))

        row = tk.Frame(card3, bg=theme.HEX_BG_CARD)
        row.pack(fill="x")
        self.dict_entry = self._entry(row)
        self.dict_entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))
        self.dict_entry.bind("<Return>", lambda e: self.add_dictionary_term())
        self._button(row, "Add", self.add_dictionary_term).pack(side="left", padx=(0, 6))
        self._button(row, "Open file", self.open_dictionary,
                     subtle=True).pack(side="left")

        # ── Activity ──
        card4 = self._card(wrap, "Activity", expand=True)
        self.log_area = scrolledtext.ScrolledText(
            card4, height=9, font=(theme.MONO_FONT, 9),
            bg=theme.HEX_BG_APP, fg=theme.HEX_TEXT, relief="flat",
            wrap=tk.WORD, borderwidth=0, insertbackground=theme.HEX_TEXT)
        self.log_area.pack(fill="both", expand=True)
        self.log_area.tag_configure("error", foreground=theme.HEX_DANGER)
        self.log_area.tag_configure("warn", foreground=theme.HEX_WARNING)
        self.log_area.tag_configure("ok", foreground=theme.HEX_SUCCESS)
        self.log_area.tag_configure("dim", foreground=theme.HEX_FAINT)
        self.log_area.config(state="disabled")

        self._refresh_status()

    def _card(self, parent, title, expand=False):
        Label(parent, text=title.upper(), font=(theme.UI_FONT, 8, "bold"),
              fg=theme.HEX_FAINT, bg=theme.HEX_BG_APP).pack(anchor="w", pady=(8, 5))
        frame = tk.Frame(parent, bg=theme.HEX_BG_CARD)
        frame.pack(fill="both" if expand else "x", expand=expand, pady=(0, 6))
        inner = tk.Frame(frame, bg=theme.HEX_BG_CARD)
        inner.pack(fill="both", expand=True, padx=16, pady=14)
        return inner

    def _entry(self, parent, show=None):
        return tk.Entry(
            parent, font=(theme.UI_FONT, 10), bg=theme.HEX_BG_INPUT,
            fg=theme.HEX_TEXT, insertbackground=theme.HEX_TEXT,
            relief="flat", show=show, borderwidth=0, highlightthickness=0)

    def _key_row(self, parent, label, value, command):
        Label(parent, text=label, font=(theme.UI_FONT, 9),
              fg=theme.HEX_MUTED, bg=theme.HEX_BG_CARD).pack(anchor="w", pady=(6, 3))
        row = tk.Frame(parent, bg=theme.HEX_BG_CARD)
        row.pack(fill="x")
        entry = self._entry(row, show="\u2022")
        entry.insert(0, value)
        entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))
        self._button(row, "Save", command).pack(side="left")
        return entry

    def _button(self, parent, text, command, subtle=False):
        bg = theme.HEX_BG_INPUT if subtle else theme.HEX_ACCENT
        fg = theme.HEX_TEXT if subtle else "#0d1220"
        b = tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                      font=(theme.UI_FONT, 9, "bold"), relief="flat",
                      padx=14, pady=6, cursor="hand2",
                      activebackground=theme.HEX_BG_HOVER,
                      activeforeground=theme.HEX_TEXT,
                      borderwidth=0, highlightthickness=0)
        hover = theme.HEX_BG_HOVER if subtle else "#93b4ff"
        b.bind("<Enter>", lambda e: b.config(bg=hover))
        b.bind("<Leave>", lambda e: b.config(bg=bg))
        return b

    def _build_tray(self):
        img = Image.new("RGB", (64, 64), theme.BG_APP)
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([14, 26, 50, 38], radius=6, fill=theme.ACCENT)
        menu = pystray.Menu(
            pystray.MenuItem("Open WhisprFlow", self.show_window, default=True),
            pystray.MenuItem("Dictionary", lambda: self.open_dictionary()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self.quit_app),
        )
        return pystray.Icon("WhisprFlow", img, "WhisprFlow", menu)

    # ══════════════════════════════════════════════════════════════════
    #  Settings
    # ══════════════════════════════════════════════════════════════════

    def _persist(self, key: str, value: str) -> None:
        ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not ENV_PATH.exists():
            ENV_PATH.touch()
        set_key(str(ENV_PATH), key, value)
        os.environ[key] = value

    def save_assemblyai_key(self):
        key = self.aai_entry.get().strip()
        if not key:
            messagebox.showwarning("WhisprFlow", "Enter an AssemblyAI API key.")
            return
        self._persist("ASSEMBLYAI_API_KEY", key)
        self.stt.set_api_key(key)
        self.log("AssemblyAI key saved.", "ok")
        self._refresh_status()
        asyncio.run_coroutine_threadsafe(self.stt.warmup(), self.loop)

    def save_groq_key(self):
        key = self.groq_entry.get().strip()
        if not key:
            messagebox.showwarning("WhisprFlow", "Enter a Groq API key.")
            return
        self._persist("GROQ_API_KEY", key)
        self.refiner.set_api_key(key)
        self.log("Groq key saved.", "ok")
        self._refresh_status()

    def add_dictionary_term(self):
        term = self.dict_entry.get().strip()
        if not term:
            return
        if self.dictionary.add(term):
            self.log(f"Added \u201c{term}\u201d to dictionary.", "ok")
            self.dict_entry.delete(0, tk.END)
            self._refresh_status()
        else:
            self.log(f"\u201c{term}\u201d not added (duplicate or too long).", "warn")

    def open_dictionary(self):
        try:
            os.startfile(str(self.dictionary.path))
        except Exception as e:
            self.log(f"Could not open dictionary: {e}", "error")

    def _on_refine_toggle(self):
        self.log("Refinement " + ("enabled." if self.refinement_enabled.get()
                                   else "disabled — raw transcript only."))
        self._refresh_status()

    def _refresh_status(self):
        info = self.stt.get_info()
        self.engine_label.config(
            text=f"AssemblyAI · {info['model']}",
            fg=theme.HEX_SUCCESS if info["configured"] else theme.HEX_WARNING)
        self.engine_sub.config(
            text="Ready" if info["configured"] else "Add an API key to start")

        if not self.refinement_enabled.get():
            txt, col = "Off — raw transcript", theme.HEX_MUTED
        elif not self.refiner.is_configured:
            txt, col = "No Groq key — basic cleanup only", theme.HEX_WARNING
        else:
            st = self.refiner.get_stats()
            if st["rejections"]:
                txt = f"{st['model']} · guard blocked {st['rejections']}/{st['calls']}"
                col = theme.HEX_WARNING
            else:
                txt, col = f"{st['model']} · guarded", theme.HEX_SUCCESS
        self.refine_status.config(text=txt, fg=col)
        self.dict_label.config(text=f"{len(self.dictionary)} terms")

    # ══════════════════════════════════════════════════════════════════
    #  Logging
    # ══════════════════════════════════════════════════════════════════

    def log(self, message: str, tag: str = "dim"):
        try:
            self.root.after(0, self._log_ui, message, tag)
        except Exception:
            logger.info(message)

    def _log_ui(self, message: str, tag: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_area.config(state="normal")
        self.log_area.insert(tk.END, f"{ts}  ", "dim")
        self.log_area.insert(tk.END, f"{message}\n", tag)
        # Keep the buffer bounded; transcripts are sensitive and unbounded
        # growth was a slow leak in the old version.
        if int(self.log_area.index("end-1c").split(".")[0]) > 300:
            self.log_area.delete("1.0", "100.0")
        self.log_area.see(tk.END)
        self.log_area.config(state="disabled")

    # ══════════════════════════════════════════════════════════════════
    #  Recording
    # ══════════════════════════════════════════════════════════════════

    def start_recording(self):
        with self._state_lock:
            if self.is_recording:
                return
            if not self.capture.is_running:
                self.log("Microphone unavailable.", "error")
                self.pill.flash_error("No microphone")
                return
            self.is_recording = True
            self.generation += 1

        self._start_time = time.monotonic()
        self.capture.begin()
        beep_async(880, 60)
        self.pill.set_state(PillState.RECORDING)

    def stop_recording(self):
        with self._state_lock:
            if not self.is_recording:
                return
            self.is_recording = False
            gen = self.generation

        beep_async(660, 60)
        audio = self.capture.end()
        duration = time.monotonic() - self._start_time

        if audio is None or duration < 0.25:
            self.pill.set_state(PillState.IDLE)
            self.log("Too short.", "warn")
            return

        self.pill.set_state(PillState.PROCESSING)
        asyncio.run_coroutine_threadsafe(self._pipeline(audio, gen), self.loop)

    def cancel_recording(self):
        with self._state_lock:
            if not self.is_recording:
                return
            self.is_recording = False
            self.generation += 1     # invalidates any in-flight pipeline

        self.capture.discard()
        beep_async(440, 90)
        self.pill.set_state(PillState.IDLE)
        self.log("Cancelled.", "dim")

    def _stale(self, gen: int) -> bool:
        """True if the user cancelled or started a new take since `gen`."""
        with self._state_lock:
            return gen != self.generation

    # ══════════════════════════════════════════════════════════════════
    #  Pipeline
    # ══════════════════════════════════════════════════════════════════

    async def _pipeline(self, audio, gen: int):
        try:
            cleaned = await asyncio.to_thread(process_audio, audio, self.capture.sample_rate)
            if self._stale(gen):
                return

            if not cleaned.speech_detected:
                self.log("No speech detected.", "warn")
                self.pill.set_state(PillState.IDLE)
                return

            self.last_audio = (cleaned.samples, cleaned.sample_rate)

            result = await self.stt.transcribe(
                cleaned.samples, cleaned.sample_rate,
                keyterms=self.dictionary.as_keyterms(),
            )
            if self._stale(gen):
                return

            if not result.ok:
                self.last_failed = True
                self.log(f"Transcription failed: {result.error}", "error")
                self.pill.flash_error(_short_error(result.error))
                return

            raw = result.text
            self.last_raw_text = raw

            if not raw.strip():
                self.log("No speech detected.", "warn")
                self.pill.set_state(PillState.IDLE)
                return

            if DEBUG:
                self.log(f"[raw {result.latency_ms}ms conf={result.confidence:.2f}] {raw}")

            final = await self._refine(raw, result)
            if self._stale(gen):
                return

            self.last_text_injected = final
            await asyncio.to_thread(self.injector.inject, final)

            self.last_failed = False
            self.log(_preview(final), "ok")
            self.pill.flash_success(_preview(final, 22))

        except Exception as e:
            self.last_failed = True
            self.log(f"Failed: {e}", "error")
            self.pill.flash_error("Something went wrong")
            if DEBUG:
                traceback.print_exc()

    async def _refine(self, raw: str, result) -> str:
        if not (self.refinement_enabled.get() and self.refiner.is_configured):
            return basic_cleanup(raw)

        final = await self.refiner.refine(
            raw,
            uncertain_words=[w.text for w in result.low_confidence_words()],
            dictionary_terms=self.dictionary.as_keyterms(),
        )
        if self.refiner.last_rejected:
            self.log(f"Refinement rejected ({self.refiner.last_rejected}) — kept raw text.",
                     "warn")
        self.root.after(0, self._refresh_status)
        return final

    def retry(self):
        if self.last_audio is None:
            self.log("Nothing to retry.", "dim")
            return
        with self._state_lock:
            self.generation += 1
            gen = self.generation
        samples, sr = self.last_audio
        self.pill.set_state(PillState.PROCESSING)
        self.log("Retrying\u2026")
        asyncio.run_coroutine_threadsafe(self._retry_from(samples, sr, gen), self.loop)

    async def _retry_from(self, samples, sr, gen):
        result = await self.stt.transcribe(
            samples, sr, keyterms=self.dictionary.as_keyterms())
        if self._stale(gen):
            return
        if not result.ok:
            self.log(f"Retry failed: {result.error}", "error")
            self.pill.flash_error(_short_error(result.error))
            return
        final = await self._refine(result.text, result)
        if self._stale(gen):
            return
        self.last_text_injected = final
        await asyncio.to_thread(self.injector.inject, final)
        self.log(_preview(final), "ok")
        self.pill.flash_success(_preview(final, 22))

    # ══════════════════════════════════════════════════════════════════
    #  Hotkeys
    # ══════════════════════════════════════════════════════════════════

    def on_press(self, key):
        self.pressed_keys.add(key)
        # Exact match only. `issubset` meant Ctrl+Win+D (new desktop) and
        # Ctrl+Win+arrows all started phantom recordings.
        if self.pressed_keys == self.hotkey_combo and not self.is_recording:
            self.start_recording()

    def on_release(self, key):
        was_hotkey = key in self.hotkey_combo
        self.pressed_keys.discard(key)
        if was_hotkey and self.is_recording:
            self.stop_recording()

    def undo(self):
        """Ctrl+Z is safer than replaying N backspaces: most apps coalesce a
        paste into one undo step, and blind backspaces eat real text when
        the caret has moved."""
        if not self.last_text_injected:
            self.log("Nothing to undo.", "dim")
            return
        self.kb_controller.press(keyboard.Key.ctrl)
        self.kb_controller.press("z")
        self.kb_controller.release("z")
        self.kb_controller.release(keyboard.Key.ctrl)
        self.last_text_injected = ""
        self.log("Undo sent.", "dim")

    # ══════════════════════════════════════════════════════════════════
    #  Window / lifecycle
    # ══════════════════════════════════════════════════════════════════

    def hide_window(self):
        self.root.withdraw()

    def show_window(self, icon=None, item=None):
        self.root.after(0, lambda: (self.root.deiconify(), self.root.lift()))

    def quit_app(self, icon=None, item=None):
        self.log("Shutting down\u2026")
        try:
            self.tray.stop()
        except Exception:
            pass
        self.capture.stop_stream()
        for coro in (self.stt.close(), self.refiner.close()):
            try:
                asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout=3)
            except Exception:
                pass
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass
        try:
            self.pill.destroy()
            self.root.quit()
        except Exception:
            pass
        os._exit(0)

    def run(self):
        threading.Thread(
            target=lambda: (asyncio.set_event_loop(self.loop), self.loop.run_forever()),
            daemon=True, name="asyncio").start()

        if self.capture.start_stream():
            self.log("Microphone ready.", "ok")
        else:
            self.log(f"Microphone failed: {self.capture.last_error}", "error")

        asyncio.run_coroutine_threadsafe(self.stt.warmup(), self.loop)

        keyboard.Listener(on_press=self.on_press, on_release=self.on_release).start()
        keyboard.GlobalHotKeys({"<ctrl>+<alt>+z": self.undo}).start()
        threading.Thread(target=self.tray.run, daemon=True, name="tray").start()

        if not self.stt.is_configured:
            self.log("No AssemblyAI key — open settings to add one.", "warn")
            self.root.after(400, self.show_window)
        else:
            self.log("Ready. Hold Ctrl + Win to dictate.", "ok")

        self.root.mainloop()


def _preview(text: str, limit: int = 60) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def _short_error(error: str) -> str:
    e = (error or "").lower()
    if "key" in e or "401" in e or "unauthor" in e:
        return "Check API key"
    if "timed out" in e or "timeout" in e:
        return "Timed out"
    if "network" in e or "connect" in e:
        return "No connection"
    if "insufficient" in e or "credit" in e:
        return "Out of credit"
    return "Failed \u21ba"


if __name__ == "__main__":
    app = WhisprFlowApp()
    try:
        app.run()
    except KeyboardInterrupt:
        os._exit(0)
