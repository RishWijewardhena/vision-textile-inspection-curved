#!/usr/bin/env python3
"""
A&C Textile Inspection System — Service Control Panel
Controls Thread.service via systemctl
Also manages seam/stitch offset values in .env
"""

import tkinter as tk
from tkinter import messagebox
import subprocess
import threading
import time
from pathlib import Path

# ── Colour palette ────────────────────────────────────────────────────────
BG_DARK = "#0D0F14"
BG_CARD = "#161A23"
BG_STRIP = "#1C2130"
ACCENT_BLUE = "#3A7BF7"
GREEN_RUN = "#00C96E"
GREEN_HOVER = "#00E87E"
RED_STOP = "#E03A3A"
RED_HOVER = "#FF4F4F"
TEXT_PRIMARY = "#F0F4FF"
TEXT_MUTED = "#6B7A99"
TEXT_STATUS = "#A8B4CC"
BORDER_DIM = "#252D40"
INPUT_BG = "#10141D"
INPUT_BORDER = "#2D364A"

# ── Changeable environment configuration ──────────────────────────────────
ENV_FILE_NAME = ".env"

SEAM_LENGTH_ENV_KEY = "SEAM_ALLOWANCE_OFFSET_MM" 
STITCH_WIDTH_ENV_KEY = "STITCH_LENGTH_OFFSET_MM"

SEAM_LENGTH_LABEL = "Seam length adjustment"
STITCH_WIDTH_LABEL = "Stitch length adjustment"

DEFAULT_OFFSET_VALUE = "0.0"


def env_file_path() -> Path:
    """
    Search for:
        ~/Desktop/THREAD/.env

    regardless of username.
    """
    home = Path.home()
    return home / "Desktop" / "THREAD" / ENV_FILE_NAME


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith('"') and value.endswith('"') and len(value) >= 2:
            value = value[1:-1]
        values[key] = value
    return values


def write_env_file(path: Path, updates: dict[str, str]) -> None:
    existing_lines: list[str] = []
    seen_keys: set[str] = set()

    if path.exists():
        existing_lines = path.read_text(encoding="utf-8").splitlines()

    output_lines: list[str] = []

    for raw_line in existing_lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in raw_line:
            output_lines.append(raw_line)
            continue

        key, _ = raw_line.split("=", 1)
        key = key.strip()
        if key in updates:
            output_lines.append(f"{key}={updates[key]}")
            seen_keys.add(key)
        else:
            output_lines.append(raw_line)

    for key, value in updates.items():
        if key not in seen_keys and all(not line.startswith(f"{key}=") for line in output_lines):
            output_lines.append(f"{key}={value}")

    # Keep file tidy with a trailing newline.
    path.write_text("\n".join(output_lines).rstrip() + "\n", encoding="utf-8")


def ensure_offset_keys_exist(path: Path) -> dict[str, str]:
    values = parse_env_file(path)
    changed = False

    if SEAM_LENGTH_ENV_KEY not in values:
        values[SEAM_LENGTH_ENV_KEY] = DEFAULT_OFFSET_VALUE
        changed = True
    if STITCH_WIDTH_ENV_KEY not in values:
        values[STITCH_WIDTH_ENV_KEY] = DEFAULT_OFFSET_VALUE
        changed = True

    if changed:
        if path.exists():
            write_env_file(path, {
                SEAM_LENGTH_ENV_KEY: values[SEAM_LENGTH_ENV_KEY],
                STITCH_WIDTH_ENV_KEY: values[STITCH_WIDTH_ENV_KEY],
            })
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                f"# offsets for seam length and stitch width\n"
                f"{SEAM_LENGTH_ENV_KEY}={values[SEAM_LENGTH_ENV_KEY]}\n"
                f"{STITCH_WIDTH_ENV_KEY}={values[STITCH_WIDTH_ENV_KEY]}\n",
                encoding="utf-8",
            )

    return values


def run_systemctl(action: str):
    """Use sudo -n (requires passwordless sudoers rule for Thread.service)."""
    try:
        result = subprocess.run(
            ["sudo", "-n", "systemctl", action, "Thread.service"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return True, ""

        err = result.stderr.strip()
        if "password" in err.lower() or "sudoers" in err.lower():
            return False, "No sudo rule found. Run: sudo visudo -f /etc/sudoers.d/thread-control"
        return False, err

    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def get_service_status() -> str:
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "Thread.service"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


class PulsingDot(tk.Canvas):
    COLORS = {
        "active": ("#00C96E", "#00FF8A"),
        "inactive": ("#6B7A99", "#8A9BB8"),
        "failed": ("#E03A3A", "#FF5555"),
        "unknown": ("#F0A500", "#FFC840"),
    }

    def __init__(self, parent):
        super().__init__(parent, width=20, height=20, bg=BG_STRIP, highlightthickness=0)
        self._state = "unknown"
        self._radius = 7.0
        self._growing = False
        self._animate()

    def set_state(self, state: str):
        self._state = state if state in self.COLORS else "unknown"

    def _animate(self):
        glow, core = self.COLORS.get(self._state, self.COLORS["unknown"])
        r = self._radius
        cx, cy = 10, 10
        self.delete("all")
        self.create_oval(cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2, fill="", outline=glow, width=1)
        self.create_oval(cx - r + 1, cy - r + 1, cx + r - 1, cy + r - 1, fill=core, outline="")

        if self._growing:
            self._radius = min(8.0, self._radius + 0.25)
            if self._radius >= 8.0:
                self._growing = False
        else:
            self._radius = max(5.5, self._radius - 0.25)
            if self._radius <= 5.5:
                self._growing = True

        self.after(60, self._animate)


class RoundedButton(tk.Canvas):
    def __init__(self, parent, text, color, hover_color, command=None, btn_width=350, btn_height=55):
        super().__init__(parent, width=btn_width, height=btn_height, bg=BG_CARD, highlightthickness=0)
        self._text = text
        self._color = color
        self._hover_color = hover_color
        self._command = command
        self._bw = btn_width
        self._bh = btn_height
        self._enabled = True

        self.after(0, lambda: self._draw(self._color))
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _rounded_rect(self, x1, y1, x2, y2, r, **kw):
        pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
               x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
               x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
        return self.create_polygon(pts, smooth=True, **kw)

    def _darken(self, hex_color):
        try:
            r = max(0, int(hex_color[1:3], 16) - 30)
            g = max(0, int(hex_color[3:5], 16) - 30)
            b = max(0, int(hex_color[5:7], 16) - 30)
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return hex_color

    def _lighten(self, hex_color):
        try:
            r = min(255, int(hex_color[1:3], 16) + 25)
            g = min(255, int(hex_color[3:5], 16) + 25)
            b = min(255, int(hex_color[5:7], 16) + 25)
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return hex_color

    def _draw(self, fill_color, label=None):
        self.delete("all")
        w, h, r = self._bw, self._bh, 14
        self._rounded_rect(0, 0, w, h, r, fill=fill_color, outline="")
        self._rounded_rect(2, h // 2, w - 2, h - 2, r - 2, fill=self._darken(fill_color), outline="")
        self._rounded_rect(2, 2, w - 2, h // 2, r - 2, fill=self._lighten(fill_color), outline="")
        self.create_text(
            w // 2, h // 2,
            text=label or self._text,
            fill=TEXT_PRIMARY if self._enabled else TEXT_MUTED,
            font=("Courier New", 12, "bold"),
            anchor="center",
        )

    def _on_enter(self, _):
        if self._enabled:
            self._draw(self._hover_color)

    def _on_leave(self, _):
        if self._enabled:
            self._draw(self._color)

    def _on_click(self, _):
        if self._enabled and self._command:
            self._command()

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        self.config(cursor="hand2" if enabled else "watch")
        self._draw(self._color if enabled else "#2A3040", None if enabled else "Please wait…")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("A&C Textile Inspection System")
        self.resizable(True, True)
        self.minsize(760, 700)
        self.configure(bg=BG_DARK)

        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        W = max(700, int(sw * 0.36))
        H = max(700, int(sh * 0.65))
        x = (sw - W) // 2
        y = (sh - H) // 2
        self.geometry(f"{W}x{H}+{x}+{y}")

        self.env_path = env_file_path()
        print("ENV FILE:", self.env_path)
        self.env_values = ensure_offset_keys_exist(self.env_path)

        self._build_ui()
        self._load_env_to_ui()
        self._refresh_status()
        self._start_status_poll()

        self.bind("<Configure>", self._on_resize)

    def _build_ui(self):
        tk.Frame(self, bg=ACCENT_BLUE, height=4).pack(fill="x", side="top")

        wrapper = tk.Frame(self, bg=BG_DARK)
        wrapper.pack(fill="both", expand=True, padx=16, pady=(10, 16))

        card = tk.Frame(wrapper, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER_DIM)
        card.pack(fill="both", expand=True)

        card.columnconfigure(0, weight=1)
        card.rowconfigure(2, weight=0)
        card.rowconfigure(3, weight=0)
        card.rowconfigure(4, weight=0)
        card.rowconfigure(5, weight=1)

        header = tk.Frame(card, bg=BG_CARD)
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(20, 0))
        header.columnconfigure(1, weight=1)

        title_block = tk.Frame(header, bg=BG_CARD)
        title_block.grid(row=0, column=0, sticky="w")

        tk.Label(title_block, text="A&C Textile Inspection", fg=TEXT_PRIMARY, bg=BG_CARD, font=("Georgia", 17, "bold")).pack(anchor="w")
        tk.Label(title_block, text="Service Control Panel  ·  Thread.service", fg=TEXT_MUTED, bg=BG_CARD, font=("Courier New", 9)).pack(anchor="w")

        tk.Frame(card, bg=BORDER_DIM, height=1).grid(row=1, column=0, sticky="ew", padx=24, pady=(16, 0))

        strip = tk.Frame(card, bg=BG_STRIP, highlightthickness=1, highlightbackground=BORDER_DIM)
        strip.grid(row=2, column=0, sticky="ew", padx=24, pady=(14, 0))
        strip.columnconfigure(0, weight=1)

        status_inner = tk.Frame(strip, bg=BG_STRIP)
        status_inner.pack(fill="x", padx=18, pady=16)

        tk.Label(status_inner, text="SERVICE STATUS", fg=TEXT_MUTED, bg=BG_STRIP, font=("Courier New", 8, "bold")).pack(anchor="w")

        dot_row = tk.Frame(status_inner, bg=BG_STRIP)
        dot_row.pack(anchor="w", pady=(8, 0))

        self._dot = PulsingDot(dot_row)
        self._dot.pack(side="left", padx=(0, 12))

        self._status_label = tk.Label(dot_row, text="Checking…", fg=TEXT_STATUS, bg=BG_STRIP, font=("Georgia", 16, "italic"))
        self._status_label.pack(side="left")

        self._detail_label = tk.Label(status_inner, text="", fg=TEXT_MUTED, bg=BG_STRIP, font=("Courier New", 8))
        self._detail_label.pack(anchor="w", pady=(6, 0))

        btn_frame = tk.Frame(card, bg=BG_CARD)
        btn_frame.grid(row=3, column=0, sticky="ew", padx=24, pady=(20, 0))
        btn_frame.columnconfigure(0, weight=1, uniform="btn")
        btn_frame.columnconfigure(1, weight=1, uniform="btn")

        self._run_btn = RoundedButton(btn_frame, text="▶   START SERVICE", color=GREEN_RUN, hover_color=GREEN_HOVER, command=self._start_service)
        self._run_btn.grid(row=0, column=0, sticky="ew", padx=(0, 10), ipady=6)

        self._stop_btn = RoundedButton(btn_frame, text="■   STOP SERVICE", color=RED_STOP, hover_color=RED_HOVER, command=self._stop_service)
        self._stop_btn.grid(row=0, column=1, sticky="ew", padx=(10, 0), ipady=6)

        # ── Offset settings section ──
        settings_outer = tk.Frame(card, bg=BG_STRIP, highlightthickness=1, highlightbackground=BORDER_DIM)
        settings_outer.grid(row=4, column=0, sticky="ew", padx=24, pady=(18, 0))
        settings_outer.columnconfigure(0, weight=1)
        settings_outer.columnconfigure(1, weight=1)

        settings_inner = tk.Frame(settings_outer, bg=BG_STRIP)
        settings_inner.pack(fill="both", expand=True, padx=18, pady=16)
        settings_inner.columnconfigure(0, weight=1)
        settings_inner.columnconfigure(1, weight=1)

        tk.Label(
            settings_inner,
            text="OFFSET SETTINGS",
            fg=TEXT_MUTED,
            bg=BG_STRIP,
            font=("Courier New", 8, "bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        # Seam length
        tk.Label(settings_inner, text=SEAM_LENGTH_LABEL, fg=TEXT_PRIMARY, bg=BG_STRIP, font=("Courier New", 10, "bold")).grid(
            row=1, column=0, sticky="w", pady=(12, 4)
        )
        seam_entry_frame = tk.Frame(settings_inner, bg=INPUT_BG, highlightthickness=1, highlightbackground=INPUT_BORDER)
        seam_entry_frame.grid(row=2, column=0, sticky="ew", padx=(0, 12))
        seam_entry_frame.columnconfigure(0, weight=1)
        self._seam_var = tk.StringVar()
        self._seam_entry = tk.Entry(
            seam_entry_frame,
            textvariable=self._seam_var,
            bg=INPUT_BG,
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            relief="flat",
            justify="left",
            font=("Courier New", 11),
        )
        self._seam_entry.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        # Stitch width / length label as requested
        tk.Label(settings_inner, text=STITCH_WIDTH_LABEL, fg=TEXT_PRIMARY, bg=BG_STRIP, font=("Courier New", 10, "bold")).grid(
            row=1, column=1, sticky="w", pady=(12, 4)
        )
        stitch_entry_frame = tk.Frame(settings_inner, bg=INPUT_BG, highlightthickness=1, highlightbackground=INPUT_BORDER)
        stitch_entry_frame.grid(row=2, column=1, sticky="ew", padx=(12, 0))
        stitch_entry_frame.columnconfigure(0, weight=1)
        self._stitch_var = tk.StringVar()
        self._stitch_entry = tk.Entry(
            stitch_entry_frame,
            textvariable=self._stitch_var,
            bg=INPUT_BG,
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            relief="flat",
            justify="left",
            font=("Courier New", 11),
        )
        self._stitch_entry.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        self._save_btn = RoundedButton(
            settings_inner,
            text="SAVE OFFSETS",
            color=ACCENT_BLUE,
            hover_color="#5A93FF",
            command=self._save_offsets,
            btn_width=260,
            btn_height=55
        )

        self._save_btn.grid(
            row=3,
            column=0,
            columnspan=2,
            pady=(14, 0)
        )

        self._note_label = tk.Label(
            settings_inner,
            text="Note: after changing the adjustment values, restart the system.",
            fg=TEXT_MUTED,
            bg=BG_STRIP,
            font=("Courier New", 8),
            wraplength=520,
            justify="left",
            anchor="w",
        )
        self._note_label.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        log_outer = tk.Frame(card, bg=BG_DARK, highlightthickness=1, highlightbackground=BORDER_DIM)
        log_outer.grid(row=5, column=0, sticky="nsew", padx=24, pady=(18, 0))
        log_outer.columnconfigure(0, weight=1)
        log_outer.rowconfigure(0, weight=1)

        self._log_label = tk.Label(
            log_outer,
            text="Ready. Use the buttons above to control the service.",
            fg=TEXT_MUTED,
            bg=BG_DARK,
            font=("Courier New", 9),
            wraplength=420,
            justify="left",
            anchor="w",
        )
        self._log_label.grid(row=0, column=0, sticky="ew", padx=16, pady=12)

        tk.Label(card, text="A&C Textile · Vision Inspection Platform", fg=BORDER_DIM, bg=BG_CARD, font=("Courier New", 8)).grid(
            row=6, column=0, pady=(12, 14)
        )

        self._log_outer = log_outer

    def _load_env_to_ui(self):
        self.env_values = ensure_offset_keys_exist(self.env_path)
        self._seam_var.set(self.env_values.get(SEAM_LENGTH_ENV_KEY, DEFAULT_OFFSET_VALUE))
        self._stitch_var.set(self.env_values.get(STITCH_WIDTH_ENV_KEY, DEFAULT_OFFSET_VALUE))

    def _save_offsets(self):
        seam_value = self._seam_var.get().strip()
        stitch_value = self._stitch_var.get().strip()

        try:
            float(seam_value)
            float(stitch_value)
        except ValueError:
            messagebox.showerror("Invalid value", "Please enter valid numeric values for both adjustments.")
            return

        updates = {
            SEAM_LENGTH_ENV_KEY: seam_value,
            STITCH_WIDTH_ENV_KEY: stitch_value,
        }

        try:
            self.env_path.parent.mkdir(parents=True, exist_ok=True)
            write_env_file(self.env_path, updates)
            self._log(f"✓ Saved offsets")
            messagebox.showinfo("Saved", "Offset values saved successfully.\n\nRestart the system for changes to take effect.")
        except Exception as e:
            messagebox.showerror("Save failed", f"Could not save .env file: {e}")

    # ── Service actions ───────────────────────────────────────────────────

    def _set_buttons_busy(self, busy: bool):
        self._run_btn.set_enabled(not busy)
        self._stop_btn.set_enabled(not busy)
        self._save_btn.set_enabled(not busy)

    def _start_service(self):
        self._set_buttons_busy(True)
        self._log("Sending START command to Thread.service…")
        threading.Thread(target=self._do_action, args=("start",), daemon=True).start()

    def _stop_service(self):
        self._set_buttons_busy(True)
        self._log("Sending STOP command to Thread.service…")
        threading.Thread(target=self._do_action, args=("stop",), daemon=True).start()

    def _do_action(self, action: str):
        ok, msg = run_systemctl(action)
        self.after(0, self._on_action_done, action, ok, msg)

    def _on_action_done(self, action: str, ok: bool, msg: str):
        if ok:
            if action == "start":
                self._log("✓ Service started successfully.")
            elif action == "stop":
                self._log("✓ Service stopped successfully.")
            else:
                self._log("✓ Service action completed successfully.")
        else:
            self._log(f"✗ Failed to {action}: {msg or 'unknown error'}")

        self._set_buttons_busy(False)
        self._refresh_status()

    # ── Status polling ────────────────────────────────────────────────────

    def _refresh_status(self):
        threading.Thread(target=self._poll_once, daemon=True).start()

    def _poll_once(self):
        status = get_service_status()
        self.after(0, self._update_status_ui, status)

    def _update_status_ui(self, status: str):
        self._dot.set_state(status)
        labels = {
            "active": ("Running", TEXT_PRIMARY),
            "inactive": ("Stopped", TEXT_MUTED),
            "failed": ("Failed", RED_STOP),
            "unknown": ("Unknown", "#F0A500"),
        }
        text, color = labels.get(status, ("Unknown", "#F0A500"))
        self._status_label.config(text=text, fg=color)
        self._detail_label.config(text=f"systemctl is-active → {status}  ·  {time.strftime('%H:%M:%S')}")

    def _start_status_poll(self):
        def loop():
            self._refresh_status()
            self.after(5000, loop)
        self.after(5000, loop)

    def _log(self, message: str):
        self._log_label.config(text=message)

    def _on_resize(self, _event):
        width = max(320, self.winfo_width() - 120)
        self._log_label.configure(wraplength=width)
        self._note_label.configure(wraplength=width)


if __name__ == "__main__":
    app = App()
    app.mainloop()