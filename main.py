import asyncio
import os
import re
import math
import time
import threading
import traceback
import ctypes
import winsound
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, Label, Entry, Button, scrolledtext, Canvas

from PIL import Image, ImageDraw, ImageTk, ImageFont, ImageFilter
import pystray
from pynput import keyboard
from pynput.keyboard import Controller as KeyboardController

from recorder import AudioRecorder
from refine import Refiner, basic_cleanup
from stt import AssemblyAIClient, UserDictionary
from injector import TextInjector
from dotenv import load_dotenv, set_key

# ── High-DPI awareness ────────────────────────────────────────
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

load_dotenv()

DEBUG = False  # Flip to True for verbose console output


# ══════════════════════════════════════════════════════════════
#  Spring physics (unchanged — already clean)
# ══════════════════════════════════════════════════════════════

class Spring:
    def __init__(self, value, stiffness=200, damping=20):
        self.val = value
        self.target = value
        self.vel = 0.0
        self.stiffness = stiffness
        self.damping = damping

    def update(self, dt=0.016):
        force = (self.target - self.val) * self.stiffness
        self.vel += (force - self.damping * self.vel) * dt
        self.val += self.vel * dt
        return self.val


# ══════════════════════════════════════════════════════════════
#  Floating Overlay  — 15 fps idle / 30 fps recording
#                    — 2× supersampling (was 4×)
#                    — NO gaussian blur
# ══════════════════════════════════════════════════════════════

class FloatingOverlay:
    # ── CHANGED: reduced constants ──
    SS_SCALE = 2                # was 4
    FPS_IDLE = 15               # was 60
    FPS_ACTIVE = 30             # was 60
    INTERVAL_IDLE = 1000 // FPS_IDLE
    INTERVAL_ACTIVE = 1000 // FPS_ACTIVE

    def __init__(self, parent):
        self.parent = parent
        self.root = tk.Toplevel(parent.root)
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.98)
        self.root.configure(bg="black")
        self.root.attributes("-transparentcolor", "black")

        self.width = 180
        self.mini_height = 2
        self.expanded_height = 42
        self.height_spring = Spring(self.mini_height, stiffness=250, damping=22)

        self.expanded = False
        self.photo = None
        self.wave_history = [Spring(0.0, stiffness=300, damping=25) for _ in range(11)]
        self.wave_lean_spring = Spring(0.0, stiffness=150, damping=15)
        self.success_end_time = 0
        self.wpm = 0
        self.pulse_val = 0
        self.last_update_time = time.time()
        self.articulate_mode = False

        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.center_x = self.screen_width // 2
        self.center_y = self.screen_height - 60

        self.canvas = Canvas(
            self.root, width=self.width, height=self.expanded_height,
            bg="black", highlightthickness=0,
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)

        # ── CHANGED: pre-load font once ──
        try:
            self._font = ImageFont.truetype("arial.ttf", 8 * self.SS_SCALE)
        except Exception:
            self._font = ImageFont.load_default()

        self.update_position()
        self._schedule_loop()

    # ── CHANGED: adaptive frame interval ──
    def _schedule_loop(self):
        interval = self.INTERVAL_ACTIVE if self.parent.is_recording else self.INTERVAL_IDLE
        self.update_position()
        self.render()
        self.root.after(interval, self._schedule_loop)

    def update_position(self):
        now = time.time()
        dt = min(now - self.last_update_time, 0.05)
        self.last_update_time = now

        try:
            mx, my = self.root.winfo_pointerxy()
            mouse_near = abs(mx - self.center_x) < 120 and abs(my - self.center_y) < 80
            self.expanded = mouse_near or self.parent.is_recording or self.parent.last_transcription_failed
        except Exception:
            self.expanded = self.parent.is_recording or self.parent.last_transcription_failed

        self.height_spring.target = self.expanded_height if self.expanded else self.mini_height
        h = self.height_spring.update(dt)

        self.center_y = self.screen_height - 100
        x = self.center_x - self.width // 2
        y = self.center_y - int(h) // 2

        try:
            self.root.geometry(f"{self.width}x{int(max(1, h))}+{x}+{y}")
        except Exception:
            pass

    def on_click(self, event):
        if self.height_spring.val < self.expanded_height - 10:
            return
        x = event.x
        if x > self.width - 45:
            self.on_action_click()
        elif x < 45:
            self.on_cancel_click()
        elif 60 < x < 120:
            self.toggle_articulate_mode()

    # ── rendering ──────────────────────────────────────────────

    def render(self):
        S = self.SS_SCALE
        bw = self.width * S
        bh = self.expanded_height * S
        img = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        h = self.height_spring.val
        alpha = max(0.0, min(1.0,
            (h - self.mini_height) / (self.expanded_height - 10 - self.mini_height + 1e-5)
        ))

        if h <= self.mini_height + 1:
            jw = 60 * S
            jx = (bw - jw) // 2
            draw.rounded_rectangle(
                [jx, 0, jx + jw, self.mini_height * S],
                radius=1 * S, fill=(255, 255, 255, 180),
            )
        else:
            # Pulse glow while recording
            if self.parent.is_recording:
                self.pulse_val = (math.sin(time.time() * 4) + 1) / 2
                pa = int(30 * self.pulse_val * alpha)
                draw.rounded_rectangle(
                    [-5*S, -5*S, bw+5*S, bh+5*S],
                    radius=30*S, outline=(0, 255, 255, pa), width=3*S,
                )

            bg_a = int(230 * alpha)
            draw.rounded_rectangle(
                [0, 0, bw, bh], radius=21*S,
                fill=(18, 18, 20, bg_a),
                outline=(0, 0, 0, int(150 * alpha)), width=2*S,
            )
            draw.rounded_rectangle(
                [S, S, bw - S, bh - S], radius=20*S,
                outline=(255, 255, 255, int(45 * alpha)), width=S,
            )

            if alpha > 0.6:
                self._draw_left_button(draw, bw, bh, S, alpha)
                self._draw_waves(draw, bw, bh, S, alpha)
                self._draw_mode_indicator(draw, bw, bh, S, alpha)
                self._draw_success(draw, bw, bh, S, alpha)
                self._draw_right_button(draw, bw, bh, S, alpha)

        # ── CHANGED: 2× downsample, no blur ──
        final = img.resize((self.width, self.expanded_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(final)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

    # ── sub-draw helpers (extracted for clarity) ───────────────

    def _draw_left_button(self, draw, bw, bh, S, alpha):
        show = self.parent.is_recording or self.parent.last_transcription_failed
        if not show:
            return
        col = (255, 50, 50, 255) if self.parent.last_transcription_failed else (50, 50, 50, 255)
        draw.ellipse([12*S, 8*S, 34*S, 30*S], fill=col)
        cx, cy = 23*S, 19*S
        if self.parent.last_transcription_failed:
            r = 5 * S
            draw.arc([cx-r, cy-r, cx+r, cy+r], start=0, end=270, fill="white", width=2*S)
            draw.polygon([cx+r, cy, cx+r-3*S, cy-3*S, cx+r+3*S, cy-3*S], fill="white")
        else:
            xs = 5 * S
            draw.line([cx-xs, cy-xs, cx+xs, cy+xs], fill="white", width=2*S)
            draw.line([cx+xs, cy-xs, cx-xs, cy+xs], fill="white", width=2*S)

    def _draw_waves(self, draw, bw, bh, S, alpha):
        num_bars = 11
        cx, cy = bw // 2, bh // 2
        bar_w = 4 * S
        gap = 5 * S

        raw_level = self.parent.recorder.get_level() if self.parent.is_recording else 0.005
        level = min(1.0, raw_level * 20)

        try:
            mx, _ = self.root.winfo_pointerxy()
            rel_mx = (mx - self.center_x) / (self.width / 2)
        except Exception:
            rel_mx = 0

        self.wave_lean_spring.target = rel_mx if abs(rel_mx) < 1.5 else 0
        lean = self.wave_lean_spring.update(0.016) * 3 * S

        t = time.time()
        for i in range(num_bars):
            if self.parent.is_recording:
                var = 0.6 + 0.4 * math.sin(t * 12 + i * 0.7)
                target_h = 4*S + level * 25 * S * var
            else:
                target_h = 4*S + 2*S * math.sin(t * 3 + i)

            self.wave_history[i].target = target_h
            h = self.wave_history[i].update(0.016)

            offset = i - num_bars // 2
            bx = cx + offset * (bar_w + gap) + lean

            if self.parent.is_recording:
                # ── CHANGED: draw glow directly, no separate bloom layer / blur ──
                r = int(min(255, max(0, 100 + level * 155)))
                g = int(min(255, max(0, 200 - level * 50)))
                glow_a = int(60 * alpha)
                draw.rounded_rectangle(
                    [bx-bar_w//2-2*S, cy-h//2-2*S, bx+bar_w//2+2*S, cy+h//2+2*S],
                    radius=4*S, fill=(r, g, 255, glow_a),
                )
                draw.rounded_rectangle(
                    [bx-bar_w//2, cy-h//2, bx+bar_w//2, cy+h//2],
                    radius=2*S, fill=(255, 255, 255, int(255*alpha)),
                )
            else:
                draw.rounded_rectangle(
                    [bx-bar_w//2-S, cy-h//2-S, bx+bar_w//2+S, cy+h//2+S],
                    radius=3*S, fill=(100, 100, 100, int(30*alpha)),
                )
                draw.rounded_rectangle(
                    [bx-bar_w//2, cy-h//2, bx+bar_w//2, cy+h//2],
                    radius=2*S, fill=(255, 255, 255, int(120*alpha)),
                )

    def _draw_mode_indicator(self, draw, bw, bh, S, alpha):
        if alpha <= 0.7:
            return
        label = "A" if self.articulate_mode else "S"
        col = (100, 200, 255, int(255*alpha)) if self.articulate_mode else (150, 150, 150, int(180*alpha))
        ix, iy, ir = bw // 2, bh - 8*S, 6*S
        draw.ellipse([ix-ir, iy-ir, ix+ir, iy+ir], fill=col)
        bb = draw.textbbox((0, 0), label, font=self._font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        draw.text((ix - tw//2, iy - th//2), label, fill=(255, 255, 255, int(255*alpha)), font=self._font)

    def _draw_success(self, draw, bw, bh, S, alpha):
        if time.time() >= self.success_end_time:
            return
        font = ImageFont.load_default(size=9 * S)
        draw.text(
            (bw//2 - 28*S, bh//2 - 6*S), "READY",
            fill=(100, 255, 100, int(255*alpha)), font=font,
        )

    def _draw_right_button(self, draw, bw, bh, S, alpha):
        if self.parent.is_recording:
            pulse_r = int(12*S * (1 + 0.1 * math.sin(time.time() * 8)))
            bx, by = bw - 23*S, 19*S
            draw.ellipse([bx-pulse_r, by-pulse_r, bx+pulse_r, by+pulse_r], fill=(255, 255, 255, 255))
            s = 4 * S
            draw.rectangle([bx-s, by-s, bx+s, by+s], fill=(255, 50, 50, 255))
        else:
            draw.ellipse([bw-34*S, 8*S, bw-12*S, 30*S], fill=(255, 50, 50, 255))
            cx, cy, r = bw-23*S, 19*S, 4*S
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255, 255, 255, 255))

    # ── actions ────────────────────────────────────────────────

    def on_action_click(self):
        if self.parent.is_recording:
            self.parent.stop_recording_sequence()
        else:
            self.parent.start_recording_sequence()

    def on_cancel_click(self):
        if self.parent.is_recording:
            self.parent.cancel_recording_sequence()
        elif self.parent.last_transcription_failed:
            self.parent.retry_processing_sequence()

    def toggle_articulate_mode(self):
        self.articulate_mode = not self.articulate_mode
        self.parent.refiner.articulate_mode = self.articulate_mode
        mode = "Articulate" if self.articulate_mode else "Standard"
        self.parent.log_message(f"Mode: {mode}")
        winsound.Beep(1200 if self.articulate_mode else 800, 100)

    def trigger_success(self):
        self.success_end_time = time.time() + 2.0


# ══════════════════════════════════════════════════════════════
#  Main Application
# ══════════════════════════════════════════════════════════════

class WhisprFlowApp:
    def __init__(self):
        self.env_path = Path(".env")
        self.recorder = AudioRecorder()
        self.refiner = Refiner(api_key=os.getenv("GROQ_API_KEY", ""))
        self.injector = TextInjector()

        # ── Transcription: AssemblyAI only. No local fallback: a silent
        #    downgrade to a weaker model is worse than an honest error.
        self.dictionary = UserDictionary()
        self.stt = AssemblyAIClient(
            api_key=os.getenv("ASSEMBLYAI_API_KEY", ""),
            model=os.getenv("ASSEMBLYAI_MODEL", "universal-3-5-pro"),
        )


        # ── CHANGED: thread-safe recording flag ──
        self._state_lock = threading.Lock()
        self.is_recording = False
        self.is_cancelled = False
        self.last_transcription_failed = False
        self.last_raw_text = ""
        self.last_stt_result = None
        self.last_audio_data = None       # (np.ndarray, sr)  for retry
        self.last_audio_path = None       # file path fallback for retry
        self.last_text_injected = ""
        self.recording_duration = 0.0
        self.start_time = 0.0

        # ── async loop on a background thread ──
        self.loop = asyncio.new_event_loop()

        # ── hotkeys ──
        self.hotkey_combo = {keyboard.Key.ctrl_l, keyboard.Key.cmd}
        self.pressed_keys = set()
        self.undo_hotkey = "<ctrl>+<alt>+z"
        self.kb_controller = KeyboardController()

        # ── GUI ──
        self.root = tk.Tk()
        self.root.title("WhisprFlow — AI Speech Recognition")
        self.root.geometry("700x800")
        self.root.resizable(True, True)
        self.root.configure(bg="#1a1a1a")
        try:
            self.root.attributes("-alpha", 0.98)
        except Exception:
            pass
        self.root.protocol("WM_DELETE_WINDOW", self.hide_window)

        self.setup_gui()
        self.overlay = FloatingOverlay(self)
        self.root.withdraw()
        self.icon = self._create_tray_icon()

        # Open the HTTPS connection so the first dictation isn't slowed
        # by a cold TLS handshake.
        threading.Thread(target=self._warm_connection, daemon=True).start()

        self.log_message("System ready.  Hold Ctrl+Win to record.")
        print("WhisprFlow started.")

    # ══════════════════════════════════════════════════════════
    #  GUI setup  (mostly unchanged — trimmed for brevity
    #              where styling-only code repeats)
    # ══════════════════════════════════════════════════════════

    def setup_gui(self):
        self.root.configure(bg="#1a1a1a")
        main = tk.Frame(self.root, bg="#1a1a1a")
        main.pack(fill="both", expand=True, padx=20, pady=20)

        # ── header ──
        hdr = tk.Frame(main, bg="#2d2d2d")
        hdr.pack(fill="x", pady=(0, 20))
        tf = tk.Frame(hdr, bg="#2d2d2d")
        tf.pack(pady=20)
        Label(tf, text="WhisprFlow", font=("Segoe UI", 24, "bold"),
              fg="#fff", bg="#2d2d2d").pack()
        Label(tf, text="AI-Powered Speech Recognition",
              font=("Segoe UI", 11), fg="#888", bg="#2d2d2d").pack()
        self.status_indicator = Canvas(tf, width=12, height=12,
                                       bg="#2d2d2d", highlightthickness=0)
        self.status_indicator.pack(pady=5)
        self.status_indicator.create_oval(2, 2, 10, 10, fill="#4CAF50", outline="")

        # ── API key card ──
        api_card = self._card(main, "🔑 API Configuration")
        ac = tk.Frame(api_card, bg="#2d2d2d")
        ac.pack(fill="x", padx=20, pady=(0, 20))
        Label(ac, text="Groq API Key", font=("Segoe UI", 10, "bold"),
              fg="#fff", bg="#2d2d2d").pack(anchor="w", pady=(0, 5))
        af = tk.Frame(ac, bg="#2d2d2d")
        af.pack(fill="x", pady=(0, 10))
        self.api_entry = tk.Entry(af, font=("Segoe UI", 10), bg="#3d3d3d",
                                  fg="#fff", insertbackground="#fff",
                                  relief="flat", show="*")
        self.api_entry.insert(0, os.getenv("GROQ_API_KEY", ""))
        self.api_entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 10))
        self._btn(af, "Save", self.save_settings, "#4CAF50").pack(side="right")

        # ── engine card ──
        eng = self._card(main, "⚡ Transcription Engine")
        ec = tk.Frame(eng, bg="#2d2d2d")
        ec.pack(fill="x", padx=20, pady=(0, 20))
        info = self.stt.get_info()
        ok = info["configured"]
        Label(ec, text=f"AssemblyAI {info['model']}",
              font=("Segoe UI", 12, "bold"),
              fg="#4CAF50" if ok else "#FF9800",
              bg="#2d2d2d").pack(anchor="w")
        Label(ec,
              text=("English • Keyterms enabled" if ok
                    else "No API key — add one below to enable transcription"),
              font=("Segoe UI", 9), fg="#888", bg="#2d2d2d").pack(anchor="w")
        self.engine_status_label = Label(
            ec, text=f"Dictionary: {len(self.dictionary)} terms",
            font=("Segoe UI", 8), fg="#666", bg="#2d2d2d")
        self.engine_status_label.pack(anchor="w", pady=(2, 10))

        # ── AssemblyAI key ──
        Label(ec, text="AssemblyAI API Key", font=("Segoe UI", 10, "bold"),
              fg="#fff", bg="#2d2d2d").pack(anchor="w", pady=(0, 5))
        akf = tk.Frame(ec, bg="#2d2d2d")
        akf.pack(fill="x", pady=(0, 10))
        self.aai_entry = tk.Entry(akf, font=("Segoe UI", 10), bg="#3d3d3d",
                                  fg="#fff", insertbackground="#fff",
                                  relief="flat", show="*")
        self.aai_entry.insert(0, os.getenv("ASSEMBLYAI_API_KEY", ""))
        self.aai_entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 10))
        self._btn(akf, "Save", self.save_assemblyai_key, "#4CAF50").pack(side="right")

        # ── refinement card ──
        ref = self._card(main, "🎯 Refinement Settings")
        rc = tk.Frame(ref, bg="#2d2d2d")
        rc.pack(fill="x", padx=20, pady=(0, 20))
        self.refinement_enabled = tk.BooleanVar(value=True)
        tk.Checkbutton(
            rc, text="Enable Groq AI refinement",
            variable=self.refinement_enabled, command=self._on_refinement_toggle,
            bg="#2d2d2d", fg="#fff", selectcolor="#3d3d3d",
            activebackground="#2d2d2d", activeforeground="#fff",
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=2)
        rlf = tk.Frame(rc, bg="#2d2d2d")
        rlf.pack(fill="x", pady=(10, 0))
        self.rate_limit_label = Label(rlf, text="", font=("Segoe UI", 8),
                                      fg="#888", bg="#2d2d2d")
        self.rate_limit_label.pack(side="left")

        # ── log card ──
        log = self._card(main, "📝 Activity Log")
        lc = tk.Frame(log, bg="#2d2d2d")
        lc.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self.history_area = scrolledtext.ScrolledText(
            lc, height=10, font=("Consolas", 9),
            bg="#1a1a1a", fg="#fff", insertbackground="#fff",
            relief="flat", wrap=tk.WORD,
        )
        self.history_area.pack(fill="both", expand=True, pady=(10, 0))
        self.history_area.config(state="disabled")

        # ── footer ──
        ft = tk.Frame(main, bg="#2d2d2d")
        ft.pack(fill="x", pady=(20, 0))
        Label(ft, text="🎤 Hold Ctrl+Win to Record  •  ✨ Floating Pill at Bottom Center",
              font=("Segoe UI", 10), fg="#888", bg="#2d2d2d").pack(pady=15)

        # Periodic status updates
        self._periodic_status()

    # ── GUI helpers ────────────────────────────────────────────

    def _card(self, parent, title):
        f = tk.Frame(parent, bg="#2d2d2d")
        f.pack(fill="x", pady=(0, 15))
        h = tk.Frame(f, bg="#3d3d3d")
        h.pack(fill="x")
        Label(h, text=title, font=("Segoe UI", 12, "bold"),
              fg="#fff", bg="#3d3d3d").pack(anchor="w", padx=20, pady=15)
        return f

    def _btn(self, parent, text, cmd, color):
        b = tk.Button(parent, text=text, command=cmd, bg=color, fg="white",
                      font=("Segoe UI", 9, "bold"), relief="flat", padx=15, pady=8,
                      cursor="hand2")
        lighter = {
            "#4CAF50": "#5CBF60", "#607D8B": "#708D9B",
        }
        orig = color
        b.bind("<Enter>", lambda e: b.config(bg=lighter.get(orig, orig)))
        b.bind("<Leave>", lambda e: b.config(bg=orig))
        return b

    def _periodic_status(self):
        try:
            self._update_refiner_label()
        except Exception:
            pass
        self.root.after(5000, self._periodic_status)

    def _update_refiner_label(self):
        if not self.refinement_enabled.get():
            self.rate_limit_label.config(text="Refinement off — raw transcript", fg="#FF9800")
            return
        if not self.refiner.is_configured:
            self.rate_limit_label.config(text="No Groq key — raw transcript", fg="#FF9800")
            return
        st = self.refiner.get_stats()
        if st["rejections"]:
            self.rate_limit_label.config(
                text=f"Guard blocked {st['rejections']}/{st['calls']} rewrites", fg="#FF9800")
        else:
            self.rate_limit_label.config(text=f"Refiner ready ({st['model']})", fg="#4CAF50")

    def _on_refinement_toggle(self):
        if self.refinement_enabled.get():
            self.log_message("Refinement enabled")
        else:
            self.log_message("Refinement disabled — raw transcript only")
        self._update_refiner_label()

    # ── logging ────────────────────────────────────────────────

    def log_message(self, message, is_error=False):
        self.root.after(0, self._log, message, is_error)

    def _log(self, message, is_error):
        ts = datetime.now().strftime("%H:%M:%S")
        tag = "error" if is_error else "normal"
        self.history_area.config(state="normal")
        self.history_area.tag_configure("error", foreground="#FF5252")
        self.history_area.tag_configure("normal", foreground="#ffffff")
        self.history_area.insert(tk.END, f"[{ts}] {message}\n", tag)
        self.history_area.see(tk.END)
        self.history_area.config(state="disabled")

    # ── settings ───────────────────────────────────────────────

    def save_settings(self):
        key = self.api_entry.get().strip()
        if not key:
            messagebox.showwarning("Warning", "Enter an API key.")
            return
        if not self.env_path.exists():
            self.env_path.touch()
        set_key(str(self.env_path), "GROQ_API_KEY", key)
        os.environ["GROQ_API_KEY"] = key
        self.refiner.set_api_key(key)
        self.log_message("Groq API key saved.")
        self._update_refiner_label()

    def save_assemblyai_key(self):
        key = self.aai_entry.get().strip()
        if not key:
            messagebox.showwarning("Warning", "Enter an AssemblyAI API key.")
            return
        if not self.env_path.exists():
            self.env_path.touch()
        set_key(str(self.env_path), "ASSEMBLYAI_API_KEY", key)
        os.environ["ASSEMBLYAI_API_KEY"] = key
        self.stt.set_api_key(key)
        self.log_message("AssemblyAI key saved — cloud transcription enabled.")
        asyncio.run_coroutine_threadsafe(self.stt.warmup(), self.loop)

    def _warm_connection(self):
        try:
            asyncio.run_coroutine_threadsafe(self.stt.warmup(), self.loop)
        except Exception:
            pass

    # ── window / tray ──────────────────────────────────────────

    def hide_window(self):
        self.root.withdraw()

    def show_window(self, icon=None, item=None):
        self.root.after(0, self.root.deiconify)

    def _create_tray_icon(self):
        img = Image.new("RGB", (64, 64), (20, 20, 20))
        ImageDraw.Draw(img).ellipse([10, 10, 54, 54], fill=(255, 60, 60))
        menu = pystray.Menu(
            pystray.MenuItem("Transcription History", self.show_window, default=True),
            pystray.MenuItem("Exit", self.quit_app),
        )
        return pystray.Icon("WhisprFlow", img, "WhisprFlow", menu)

    def quit_app(self, icon=None, item=None):
        # ── CHANGED: graceful shutdown ──
        if self.icon:
            self.icon.stop()
        # Close persistent HTTP client
        try:
            asyncio.run_coroutine_threadsafe(self.refiner.close(), self.loop).result(timeout=5)
            asyncio.run_coroutine_threadsafe(self.stt.close(), self.loop).result(timeout=5)
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.root.quit()
        os._exit(0)

    # ══════════════════════════════════════════════════════════
    #  Recording sequences
    # ══════════════════════════════════════════════════════════

    def start_recording_sequence(self):
        with self._state_lock:
            if self.is_recording:
                return
            self.is_recording = True
            self.is_cancelled = False
            self.last_transcription_failed = False

        winsound.Beep(1000, 100)
        self.log_message("🎤 Recording...")
        self.start_time = time.time()
        self.recorder.start()

    def stop_recording_sequence(self):
        with self._state_lock:
            if not self.is_recording:
                return
            self.is_recording = False

        self.recording_duration = time.time() - self.start_time
        winsound.Beep(800, 100)
        self.log_message("⚙️ Processing...")

        # ── CHANGED: get numpy array directly ──
        result = self.recorder.stop()
        if result is None:
            self.log_message("No speech detected.", is_error=True)
            return

        audio_data, sample_rate = result
        self.last_audio_data = (audio_data, sample_rate)
        asyncio.run_coroutine_threadsafe(
            self._process_audio(audio_data, sample_rate), self.loop
        )

    def cancel_recording_sequence(self):
        with self._state_lock:
            if not self.is_recording:
                return
            self.is_recording = False
            self.is_cancelled = True

        winsound.Beep(400, 200)
        self.log_message("Cancelled.")
        self.recorder.stop()  # Discard the data

    # ══════════════════════════════════════════════════════════
    #  Core pipeline  — numpy in, text out
    # ══════════════════════════════════════════════════════════

    async def _process_audio(self, samples, sample_rate):
        if self.is_cancelled:
            return
        try:
            # ── Step 1: transcription ──
            self.log_message("Transcribing...")
            result = await self.stt.transcribe(
                samples, sample_rate, keyterms=self.dictionary.as_keyterms()
            )
            self.last_stt_result = result

            # Cancelled while the request was in flight? Do not inject into
            # whatever window the user has moved on to.
            if self.is_cancelled:
                return

            # A failure and genuine silence are different things and must
            # produce different UI. Only a failure is worth retrying.
            if not result.ok:
                self.log_message(f"Transcription failed: {result.error}", is_error=True)
                self.last_transcription_failed = True
                return

            raw = result.text
            self.last_raw_text = raw

            if not raw.strip():
                self.log_message("No speech detected.")
                self.last_transcription_failed = False
                return

            if DEBUG:
                self.log_message(
                    f"[RAW] ({result.model}, {result.latency_ms}ms, "
                    f"conf={result.confidence:.2f}) {raw}"
                )

            # ── Step 2: refinement, guarded ──
            if self.refinement_enabled.get() and self.refiner.is_configured:
                uncertain = [w.text for w in result.low_confidence_words()]
                final = await self.refiner.refine(
                    raw,
                    uncertain_words=uncertain,
                    dictionary_terms=self.dictionary.as_keyterms(),
                )
                if self.refiner.last_rejected:
                    # The rewrite looked like a hallucination and was
                    # discarded. The raw transcript is used instead.
                    self.log_message(
                        f"Refinement rejected ({self.refiner.last_rejected}) — using raw text."
                    )
                self._update_refiner_label_safe()
            else:
                final = basic_cleanup(raw)

            if self.is_cancelled:
                return

            # ── Step 3: inject ──
            self.log_message(f"\u2713 {final[:60]}{'\u2026' if len(final) > 60 else ''}")
            self.last_text_injected = final
            await asyncio.to_thread(self.injector.inject, final)
            self.overlay.trigger_success()
            self.last_transcription_failed = False

        except Exception as e:
            self.last_transcription_failed = True
            self.log_message(f"FAILED: {e}", is_error=True)
            if DEBUG:
                traceback.print_exc()

    def _update_refiner_label_safe(self):
        try:
            self.root.after(0, self._update_refiner_label)
        except Exception:
            pass

    # ── retry ──────────────────────────────────────────────────

    def retry_processing_sequence(self):
        if self.last_audio_data is not None:
            self.log_message("Retrying from cached audio...")
            self.last_transcription_failed = False
            samples, sr = self.last_audio_data
            asyncio.run_coroutine_threadsafe(self._process_audio(samples, sr), self.loop)
        elif self.last_raw_text:
            self.log_message("Retrying refinement only...")
            self.last_transcription_failed = False
            asyncio.run_coroutine_threadsafe(self._retry_refinement(), self.loop)
        else:
            self.last_transcription_failed = False
            self.log_message("Nothing to retry.")

    async def _retry_refinement(self):
        try:
            final = await self.refiner.refine(
                self.last_raw_text,
                dictionary_terms=self.dictionary.as_keyterms(),
            )
            if self.is_cancelled:
                return
            self.log_message(f"\u2713 (retry) {final[:60]}")
            self.last_text_injected = final
            await asyncio.to_thread(self.injector.inject, final)
            self.overlay.trigger_success()
            self.last_transcription_failed = False
        except Exception as e:
            self.last_transcription_failed = True
            self.log_message(f"Retry failed: {e}", is_error=True)

    # ── undo ───────────────────────────────────────────────────

    def undo_injection(self):
        if not self.last_text_injected:
            self.log_message("Nothing to undo.")
            return
        n = len(self.last_text_injected)
        self.log_message(f"Undoing ({n} chars)...")
        for _ in range(n):
            self.kb_controller.press(keyboard.Key.backspace)
            self.kb_controller.release(keyboard.Key.backspace)
        self.last_text_injected = ""

    # ── keyboard listener ──────────────────────────────────────

    def on_press(self, key):
        self.pressed_keys.add(key)
        if self.hotkey_combo.issubset(self.pressed_keys) and not self.is_recording:
            self.start_recording_sequence()

    def on_release(self, key):
        self.pressed_keys.discard(key)
        if key in self.hotkey_combo and self.is_recording:
            self.stop_recording_sequence()

    # ══════════════════════════════════════════════════════════
    #  Run
    # ══════════════════════════════════════════════════════════

    def run(self):
        # Background async loop
        threading.Thread(
            target=lambda: (asyncio.set_event_loop(self.loop), self.loop.run_forever()),
            daemon=True,
        ).start()

        # Open the TLS connection ahead of the first dictation (~200ms saved)
        asyncio.run_coroutine_threadsafe(self.stt.warmup(), self.loop)

        # Keyboard listeners
        keyboard.Listener(on_press=self.on_press, on_release=self.on_release).start()
        keyboard.GlobalHotKeys({self.undo_hotkey: self.undo_injection}).start()

        # System tray
        threading.Thread(target=self.icon.run, daemon=True).start()

        # Tkinter main loop (blocks)
        self.root.mainloop()


if __name__ == "__main__":
    app = WhisprFlowApp()
    try:
        app.run()
    except KeyboardInterrupt:
        os._exit(0)