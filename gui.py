import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from main import VirtualMouse

PREVIEW_W = 640
PREVIEW_H = 360
BG = "#1e1e2e"
FG = "#cdd6f4"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Virtual Mouse")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.mouse = VirtualMouse()
        self._build_ui()
        self._update_preview()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        # Title
        tk.Label(self, text="AI Virtual Mouse", font=("Segoe UI", 15, "bold"),
                 bg=BG, fg=FG).pack(pady=(10, 4))

        # Webcam canvas
        self.canvas = tk.Canvas(self, width=PREVIEW_W, height=PREVIEW_H,
                                bg="#000000", highlightthickness=0)
        self.canvas.pack(padx=16, pady=6)

        # Status labels
        self.status_var = tk.StringVar(value="Status: Stopped")
        tk.Label(self, textvariable=self.status_var, font=("Segoe UI", 10),
                 bg=BG, fg="#a6e3a1").pack()

        self.gesture_var = tk.StringVar(value="Gesture: —")
        tk.Label(self, textvariable=self.gesture_var, font=("Segoe UI", 10),
                 bg=BG, fg="#89b4fa").pack(pady=(2, 8))

        # Buttons
        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.pack(pady=4)

        self.start_btn = tk.Button(btn_frame, text="▶  Start", width=12,
                                   command=self._start, bg="#a6e3a1", fg="#1e1e2e",
                                   font=("Segoe UI", 10, "bold"), relief="flat",
                                   cursor="hand2")
        self.start_btn.grid(row=0, column=0, padx=6)

        self.stop_btn = tk.Button(btn_frame, text="■  Stop", width=12,
                                  command=self._stop, bg="#f38ba8", fg="#1e1e2e",
                                  font=("Segoe UI", 10, "bold"), relief="flat",
                                  cursor="hand2", state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=6)

        self.pause_btn = tk.Button(btn_frame, text="⏸  Pause", width=12,
                                   command=self._toggle_pause, bg="#fab387", fg="#1e1e2e",
                                   font=("Segoe UI", 10, "bold"), relief="flat",
                                   cursor="hand2", state="disabled")
        self.pause_btn.grid(row=0, column=2, padx=6)

        # Settings
        sf = tk.LabelFrame(self, text="  Settings  ", bg=BG, fg=FG,
                           font=("Segoe UI", 10), padx=16, pady=8)
        sf.pack(padx=16, pady=10, fill="x")

        self._slider(sf, "Smoothing (speed)", 0.05, 0.5, 0.18,
                     lambda v: setattr(self.mouse, "smoothing", float(v)), 0, res=0.01)
        self._slider(sf, "Click Cooldown (s)", 0.1, 2.0, 0.5,
                     lambda v: setattr(self.mouse, "click_cooldown", float(v)), 1, res=0.1)
        self._slider(sf, "Scroll Sensitivity", 0.1, 3.0, 0.8,
                     lambda v: setattr(self.mouse, "scroll_sensitivity", float(v)), 2, res=0.1)

        # Gesture hints
        hints = ("Index finger → Move cursor     |     Pinch (thumb+index) → Left click\n"
                 "Index + Middle → Right click   |     Open palm (move up/down) → Scroll\n"
                 "Fist → Pause movement")
        tk.Label(self, text=hints, font=("Segoe UI", 9), bg=BG,
                 fg="#6c7086", justify="center").pack(pady=(0, 12))

    def _slider(self, parent, label, lo, hi, default, cmd, row, res=1):
        tk.Label(parent, text=label, bg=BG, fg=FG,
                 font=("Segoe UI", 9), width=24, anchor="w").grid(
            row=row, column=0, sticky="w", pady=4)
        var = tk.DoubleVar(value=default)

        def _cmd(v):
            cmd(v)
            val_lbl.config(text=f"{float(v):.1f}")

        ttk.Scale(parent, from_=lo, to=hi, variable=var,
                  orient="horizontal", length=220, command=_cmd).grid(
            row=row, column=1, padx=10)
        val_lbl = tk.Label(parent, text=f"{default:.1f}", bg=BG, fg="#89b4fa",
                           font=("Segoe UI", 9), width=5)
        val_lbl.grid(row=row, column=2)

    # --- Controls ---
    def _start(self):
        self.mouse.start()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.pause_btn.config(state="normal")
        self.status_var.set("Status: Running")

    def _stop(self):
        self.mouse.stop()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.pause_btn.config(state="disabled", text="⏸  Pause")
        self.status_var.set("Status: Stopped")
        self.canvas.delete("all")

    def _toggle_pause(self):
        self.mouse.toggle_pause()
        if self.mouse.paused:
            self.pause_btn.config(text="▶  Resume")
            self.status_var.set("Status: Paused")
        else:
            self.pause_btn.config(text="⏸  Pause")
            self.status_var.set("Status: Running")

    # --- Preview loop ---
    def _update_preview(self):
        if self.mouse.running:
            frame = self.mouse.get_frame()
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb).resize((PREVIEW_W, PREVIEW_H))
                self._imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor="nw", image=self._imgtk)
            self.gesture_var.set(f"Gesture: {self.mouse.gesture_label}")
        self.after(30, self._update_preview)

    def _on_close(self):
        self.mouse.stop()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
