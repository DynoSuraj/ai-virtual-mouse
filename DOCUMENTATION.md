
# AI Virtual Mouse — Project Documentation

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Technology Stack & Packages](#2-technology-stack--packages)
3. [Project Structure](#3-project-structure)
4. [Installation — Step by Step](#4-installation--step-by-step)
5. [How the System Works](#5-how-the-system-works)
6. [MediaPipe Hand Landmarks](#6-mediapipe-hand-landmarks)
7. [Gesture Reference](#7-gesture-reference)
8. [Code Walkthrough — main.py](#8-code-walkthrough--mainpy)
9. [Code Walkthrough — gui.py](#9-code-walkthrough--guipy)
10. [EMA Smoothing Explained](#10-ema-smoothing-explained)
11. [Settings & Tuning Guide](#11-settings--tuning-guide)
12. [Common Errors & Fixes](#12-common-errors--fixes)
13. [Development Phases](#13-development-phases)

---

## 1. Project Overview

**Project Name:** AI Virtual Mouse Using Hand Gesture Recognition

**Goal:** Control your Windows desktop mouse cursor using only your hand in front of a webcam — no physical mouse needed.

**How it works in one line:**  
Webcam → OpenCV reads frames → MediaPipe detects 21 hand landmarks → Python identifies your gesture → PyAutoGUI moves/clicks the mouse.

**What it can do:**

| Gesture | Action |
|---|---|
| Index finger only | Move mouse cursor |
| Thumb + index finger pinch | Left click |
| Index + middle finger | Right click |
| Open palm (move up/down) | Scroll |
| Fist | Pause all mouse control |

---

## 2. Technology Stack & Packages

### Python Version
- **Python 3.11** or **3.12** (both work)
- Download: https://www.python.org/downloads/

---

### Package Details

#### `opencv-python == 4.10.0.84`
- **Full name:** Open Source Computer Vision Library
- **What it does:**
  - Opens and reads your webcam feed frame by frame
  - Flips the image horizontally (mirror effect)
  - Converts color from BGR (OpenCV default) to RGB (MediaPipe requires RGB)
  - Draws landmark dots and gesture labels on the preview frame
  - Handles FPS timing display
- **Key functions used:**
  - `cv2.VideoCapture(0)` — opens webcam at index 0
  - `cv2.flip(frame, 1)` — mirrors the image
  - `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` — converts color
  - `cv2.putText(...)` — draws text on frame
  - `cv2.circle(...)` — draws fingertip dots
- **Why this version:** 4.10.x is stable and compatible with mediapipe 0.10.21 and numpy 1.26.4

---

#### `mediapipe == 0.10.21`
- **What it does:**
  - Runs a pre-trained neural network model to detect a hand in every frame
  - Returns 21 landmark points (x, y, z coordinates) for every joint and fingertip
  - Works in real time on CPU — no GPU needed
- **Key functions used:**
  - `mp.solutions.hands` — the hand detection module
  - `mp_hands.Hands(...)` — creates the detector with config options
  - `hands.process(rgb_frame)` — runs detection on one frame
  - `mp.solutions.drawing_utils.draw_landmarks(...)` — draws the skeleton
- **Configuration options used:**

  | Option | Value | Meaning |
  |---|---|---|
  | `static_image_mode` | `False` | Video mode (tracking between frames) |
  | `max_num_hands` | `1` | Only detect one hand |
  | `min_detection_confidence` | `0.7` | 70% confidence needed to detect |
  | `min_tracking_confidence` | `0.7` | 70% confidence needed to keep tracking |

- **Why this version:** 0.10.21 is the latest stable version compatible with numpy 1.26.4

---

#### `numpy == 1.26.4`
- **What it does:**
  - Maps finger tip pixel coordinates to screen pixel coordinates using `np.interp`
  - Clips mapped values so they stay within screen bounds using `np.clip`
  - Used for coordinate arithmetic
- **Key functions used:**
  - `np.interp(value, [in_min, in_max], [out_min, out_max])` — remaps a value from one range to another
  - `np.clip(value, min, max)` — clamps a value within bounds
- **Why this version:** 1.26.4 is the last 1.x version. Required because mediapipe 0.10.21 is not compatible with numpy 2.x

---

#### `PyAutoGUI == 0.9.54`
- **What it does:**
  - Moves the actual OS mouse cursor to any screen coordinate
  - Performs left and right mouse clicks
  - Performs scroll wheel actions
- **Key functions used:**
  - `pyautogui.size()` — gets screen resolution (width, height)
  - `pyautogui.moveTo(x, y)` — moves mouse to absolute position
  - `pyautogui.click()` — left click at current position
  - `pyautogui.rightClick()` — right click at current position
  - `pyautogui.scroll(amount)` — scroll up (positive) or down (negative)
- **Safety settings used:**
  - `pyautogui.FAILSAFE = True` — moving cursor to top-left corner (0,0) raises an exception and stops the program
  - `pyautogui.PAUSE = 0.01` — 10ms pause between each PyAutoGUI command to prevent overwhelming the OS

---

#### `Pillow`
- **What it does:**
  - Converts OpenCV numpy frames (BGR arrays) into PIL Image objects
  - Resizes the frame to fit the GUI preview canvas
  - Converts to `ImageTk.PhotoImage` format which Tkinter can display
- **Key functions used:**
  - `Image.fromarray(frame_rgb)` — numpy array → PIL Image
  - `img.resize((width, height))` — resize for preview
  - `ImageTk.PhotoImage(image=img)` — PIL Image → Tkinter-compatible image

---

#### `tkinter` (built into Python — no install needed)
- **What it does:**
  - Creates the desktop GUI window
  - Provides buttons (Start, Stop, Pause), sliders, labels
  - Shows a live webcam preview using a Canvas widget
  - Updates the preview every 30ms using `.after(30, callback)`

---

#### `math` (built into Python)
- **What it does:**
  - Calculates the Euclidean distance between two landmark points
  - Used to detect pinch gesture (distance between thumb tip and index tip)
- **Function used:**
  - `math.hypot(dx, dy)` — calculates √(dx² + dy²)

---

#### `time` (built into Python)
- **What it does:**
  - Tracks last click time to enforce cooldown between clicks
  - Measures FPS (frames per second)

---

#### `threading` (built into Python)
- **What it does:**
  - Runs the webcam capture and hand detection loop in a background thread
  - Prevents the GUI from freezing while the camera is active
  - Uses a Lock (`threading.Lock`) to safely share the current frame between threads

---

## 3. Project Structure

```
AI-Virtual-Mouse/
│
├── main.py              # Core engine: webcam, hand tracking, gesture logic, mouse control
├── gui.py               # Tkinter desktop GUI: preview, buttons, sliders
├── requirements.txt     # Package list with pinned versions
└── DOCUMENTATION.md     # This file
```

---

## 4. Installation — Step by Step

### Step 1: Install Python
1. Download Python 3.11 or 3.12 from https://www.python.org/downloads/
2. During installation, check **"Add Python to PATH"**
3. Click Install Now

Verify in terminal:
```
python --version
```

---

### Step 2: Create project folder
```
mkdir AI-Virtual-Mouse
cd AI-Virtual-Mouse
```

---

### Step 3: Create a virtual environment
```
python -m venv venv
```

Activate it (Windows):
```
venv\Scripts\activate
```

You should see `(venv)` at the start of your terminal line.

---

### Step 4: Upgrade pip
```
python -m pip install --upgrade pip
```

---

### Step 5: Install packages
```
pip install -r requirements.txt
```

Or install manually:
```
pip install opencv-python==4.10.0.84
pip install mediapipe==0.10.21
pip install numpy==1.26.4
pip install PyAutoGUI==0.9.54
pip install Pillow
```

**Important:** If you have `opencv-python-headless` installed from before, remove it first to avoid conflicts:
```
pip uninstall opencv-python-headless -y
```

---

### Step 6: Verify installation
```
python -c "import cv2, mediapipe, numpy, pyautogui; print('All OK')"
```

Expected output:
```
All OK
```

---

### Step 7: Run the app
```
python gui.py
```

Then click the **▶ Start** button to activate the webcam.

---

## 5. How the System Works

```
Webcam (cv2.VideoCapture)
        ↓
   Read frame (cap.read())
        ↓
   Flip horizontally (mirror)
        ↓
   Convert BGR → RGB
        ↓
   MediaPipe processes frame
        ↓
   21 landmark points returned
        ↓
   get_finger_states() — which fingers are up?
        ↓
   Gesture identified
        ↓
   PyAutoGUI executes action
        ↓
   Frame annotated and shown in GUI
```

---

## 6. MediaPipe Hand Landmarks

MediaPipe returns 21 points on your hand. Each has an `x`, `y`, `z` coordinate normalized between 0.0 and 1.0.

```
                8   12  16  20
                |   |   |   |
                7   11  15  19
            4   |   |   |   |
            |   6   10  14  18
            3   |   |   |   |
            |   5---9--13--17
            2       |
             \      |
              1     |
               \    0 (Wrist)
```

**Key landmark IDs used in this project:**

| ID | Location | Used for |
|---|---|---|
| 0 | Wrist | Scroll tracking, thumb detection |
| 3 | Thumb IP joint | Thumb extension check |
| 4 | Thumb tip | Pinch distance calculation |
| 5 | Index MCP (knuckle) | Index finger "up" check |
| 8 | Index tip | Cursor position, pinch |
| 9 | Middle MCP | Middle finger "up" check |
| 12 | Middle tip | Right click detection |
| 13 | Ring MCP | Ring finger "up" check |
| 16 | Ring tip | Gesture filtering |
| 17 | Pinky MCP | Pinky "up" check |
| 20 | Pinky tip | Gesture filtering |

---

## 7. Gesture Reference

### How finger detection works

For each finger (except thumb):
- Compare the **tip y coordinate** with the **MCP (knuckle) y coordinate**
- If `tip_y < mcp_y` → finger is extended (pointing up)
- If `tip_y >= mcp_y` → finger is curled down

For thumb:
- Compare horizontal distance of thumb tip from wrist vs thumb IP joint from wrist
- If thumb tip is further from wrist → thumb is extended

### Gesture logic

| Gesture | Condition |
|---|---|
| **Move cursor** | Only index finger extended |
| **Left click** | Only index finger extended AND pinch distance < 0.045 |
| **Right click** | Index AND middle extended, ring AND pinky down |
| **Scroll** | All 4 fingers extended (open palm), track wrist Y movement |
| **Fist / pause** | No fingers extended |

---

## 8. Code Walkthrough — main.py

### Configuration block
```python
SMOOTHING = 0.18       # EMA alpha: how fast cursor follows hand (0.05–0.5)
PINCH_THRESHOLD = 0.045  # Normalized distance for pinch detection
CLICK_COOLDOWN = 0.5   # Seconds between clicks (prevents double-click spam)
SCROLL_SENSITIVITY = 0.8  # Multiplier for scroll speed
```

### `get_finger_states(landmarks, w, h)`
- Takes all 21 landmark objects and the frame dimensions
- Returns a dict: `{"index": True, "middle": False, ...}` — True if that finger is extended
- Also returns pixel coordinates of each fingertip

### `get_distance(p1, p2, w, h, landmarks)`
- Takes two landmark IDs (e.g. 4 for thumb tip, 8 for index tip)
- Returns normalized Euclidean distance (0.0 to ~1.4)
- Used to detect pinch: if distance < `PINCH_THRESHOLD`, it's a pinch

### `VirtualMouse` class

| Method | Purpose |
|---|---|
| `start()` | Launches background thread running `_loop()` |
| `stop()` | Sets `self.running = False`, thread exits cleanly |
| `toggle_pause()` | Flips `self.paused` — hand detection continues but mouse is frozen |
| `get_frame()` | Thread-safe copy of the latest annotated frame |
| `_loop()` | Main loop: reads webcam, runs MediaPipe, identifies gesture, calls PyAutoGUI |

### Cursor mapping in `_loop()`
```python
margin_x = w * 0.03   # 3% border on left and right
margin_y = h * 0.03   # 3% border on top and bottom

mapped_x = np.interp(index_tip[0], [margin_x, w - margin_x], [0, SCREEN_WIDTH])
mapped_y = np.interp(index_tip[1], [margin_y, h - margin_y], [0, SCREEN_HEIGHT])
```
- The camera frame is `1280×720` pixels
- The usable area is from 3% to 97% of the frame
- That usable area maps to the full screen resolution (e.g. 1920×1080)
- Moving your finger to the frame edge → cursor reaches screen edge

### EMA smoothing
```python
alpha = self.smoothing   # e.g. 0.18
smooth_x = alpha * mapped_x + (1 - alpha) * self._prev_x
smooth_y = alpha * mapped_y + (1 - alpha) * self._prev_y
```
- Each frame, the cursor moves 18% toward the new raw position
- Remaining 82% stays at the previous smoothed position
- Eliminates jitter without introducing noticeable lag

---

## 9. Code Walkthrough — gui.py

### `App` class (extends `tk.Tk`)

| Method | Purpose |
|---|---|
| `__init__()` | Creates window, builds UI, starts preview loop |
| `_build_ui()` | Lays out all widgets: title, canvas, buttons, sliders, hints |
| `_slider()` | Helper: creates label + ttk.Scale + value label in a grid row |
| `_start()` | Calls `mouse.start()`, updates button states and status label |
| `_stop()` | Calls `mouse.stop()`, clears canvas |
| `_toggle_pause()` | Calls `mouse.toggle_pause()`, updates pause button text |
| `_update_preview()` | Called every 30ms via `self.after(30, ...)` — pulls latest frame and draws it on canvas |
| `_on_close()` | Stops mouse engine before destroying window |

### Preview update loop
```python
def _update_preview(self):
    if self.mouse.running:
        frame = self.mouse.get_frame()          # get latest frame from background thread
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).resize((640, 360))
        self._imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor="nw", image=self._imgtk)
    self.after(30, self._update_preview)        # schedule next update in 30ms (~33 FPS)
```

### Slider → engine connection
Each slider directly updates the engine's config in real time:
```python
lambda v: setattr(self.mouse, "smoothing", float(v))
```
No restart needed — the change takes effect on the very next frame.

---

## 10. EMA Smoothing Explained

**EMA = Exponential Moving Average**

Formula:
```
smooth = alpha × new_value + (1 - alpha) × previous_smooth
```

**Example:**
- Previous smooth position: `400px`
- New raw detected position: `500px`
- Alpha: `0.18`
- Result: `0.18 × 500 + 0.82 × 400 = 90 + 328 = 418px`

The cursor moves *toward* the finger but doesn't jump all the way — this hides jitter from hand tremor and camera noise.

**Alpha tuning guide:**

| Alpha | Feel |
|---|---|
| `0.05` | Very smooth, but cursor lags behind your hand |
| `0.10` | Smooth with slight lag |
| `0.18` | Default — good balance |
| `0.30` | Responsive, small jitter visible |
| `0.50` | Fast response, more jitter |

**Why not simple averaging?**  
The old method (`position / N`) moves the cursor by a fixed step size. For fast movements it feels too slow; for slow fine movements it overshoots. EMA always moves a *percentage* of the remaining gap, which feels physically natural.

---

## 11. Settings & Tuning Guide

### Smoothing (alpha: 0.05 → 0.5)
- **Too shaky?** Lower the value (e.g. 0.10)
- **Too laggy/slow?** Raise the value (e.g. 0.30)

### Click Cooldown (seconds: 0.1 → 2.0)
- **Registering double clicks?** Raise to 0.8 or 1.0
- **Clicking feels unresponsive?** Lower to 0.3

### Scroll Sensitivity (0.1 → 3.0)
- **Scrolling too fast?** Lower to 0.3–0.5
- **Barely scrolling?** Raise to 1.5–2.0

### Camera index
If your webcam doesn't open, change line 115 in `main.py`:
```python
cap = cv2.VideoCapture(0)   # try 1 or 2 if 0 doesn't work
```

### Lighting
- MediaPipe performs best with **front-facing light** on your hand
- Avoid bright light directly behind your hand (backlight confuses detection)
- A desk lamp facing you improves detection accuracy significantly

---

## 12. Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `No module named cv2` | OpenCV not installed | `pip install opencv-python==4.10.0.84` |
| `module 'cv2' has no attribute '__version__'` | Conflicting OpenCV packages | `pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y` then `pip install opencv-python==4.10.0.84` |
| `No module named mediapipe` | Wrong Python env | Make sure virtual environment is activated |
| `numpy incompatible` | numpy 2.x installed | `pip install numpy==1.26.4` |
| Webcam window stays black | Camera not started | Click the **▶ Start** button in the GUI |
| Webcam won't open | Wrong camera index | Change `VideoCapture(0)` to `VideoCapture(1)` |
| Cursor shaking heavily | Smoothing too high | Lower alpha slider to 0.08–0.12 |
| Can't reach screen edges | Margins too large | Margins are already set to 3% — ensure your hand moves to the edge of the camera frame |
| Multiple clicks firing | Cooldown too short | Raise Click Cooldown slider to 0.8–1.0 |
| Hand not detected | Poor lighting or angle | Improve lighting, face your palm directly at camera |
| PyAutoGUI FailSafeException | Cursor hit top-left corner | Move cursor away from top-left corner, or set `pyautogui.FAILSAFE = False` |
| MediaPipe TensorFlow warnings on startup | Normal MediaPipe logs | Not an error — the app works fine |

---

## 13. Development Phases

### Phase 1 — Hand Tracking
- Webcam initialization with OpenCV
- MediaPipe hand detection
- 21 landmark point visualization on frame
- FPS counter

### Phase 2 — Mouse Control
- Index finger → cursor movement with coordinate mapping
- Thumb+index pinch → left click with cooldown
- Index+middle → right click with cooldown
- Open palm vertical movement → scroll

### Phase 3 — Accuracy Improvements
- EMA smoothing (replaced simple division with exponential moving average)
- Tight frame margins (3%) for full screen reach
- Click cooldown to prevent repeated triggers
- Coordinate clamping to prevent out-of-bounds crashes
- Fist gesture to pause all movement

### Phase 4 — Desktop GUI
- Tkinter window with dark theme
- Live webcam preview (640×360) updated at ~33 FPS
- Start / Stop / Pause controls
- Real-time sliders: Smoothing, Click Cooldown, Scroll Sensitivity
- Status label and gesture name display
- Background threading so GUI stays responsive
- Thread-safe frame sharing with `threading.Lock`
