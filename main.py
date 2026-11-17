import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import math
import threading
import urllib.request
import os

# ---------------- CONFIGURATION ----------------
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
SMOOTHING = 0.18
PINCH_THRESHOLD = 0.045
CLICK_COOLDOWN = 0.5
SCROLL_SENSITIVITY = 0.8

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model (~25 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model ready.")

# ---------------- MEDIAPIPE TASKS API ----------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Hand skeleton connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# ---------------- LANDMARK INDICES ----------------
TIP_IDS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}
MCP_IDS = {
    "index": 5,
    "middle": 9,
    "ring": 13,
    "pinky": 17,
}


def get_finger_states(landmarks, w, h):
    pts = {name: (int(landmarks[tip].x * w), int(landmarks[tip].y * h))
           for name, tip in TIP_IDS.items()}
    mcp = {name: (int(landmarks[mid].x * w), int(landmarks[mid].y * h))
           for name, mid in MCP_IDS.items()}

    thumb_tip = pts["thumb"]
    thumb_ip = (int(landmarks[3].x * w), int(landmarks[3].y * h))
    wrist = (int(landmarks[0].x * w), int(landmarks[0].y * h))
    thumb_extended = abs(thumb_tip[0] - wrist[0]) > abs(thumb_ip[0] - wrist[0])

    fingers = {
        "thumb": thumb_extended,
        "index": pts["index"][1] < mcp["index"][1],
        "middle": pts["middle"][1] < mcp["middle"][1],
        "ring": pts["ring"][1] < mcp["ring"][1],
        "pinky": pts["pinky"][1] < mcp["pinky"][1],
    }
    return fingers, pts


def get_distance(p1, p2, landmarks):
    x1, y1 = landmarks[p1].x, landmarks[p1].y
    x2, y2 = landmarks[p2].x, landmarks[p2].y
    return math.hypot(x2 - x1, y2 - y1)


def draw_hand(frame, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, pts[start], pts[end], (80, 110, 255), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)


class VirtualMouse:
    def __init__(self):
        self.running = False
        self.paused = False
        self.smoothing = SMOOTHING
        self.pinch_threshold = PINCH_THRESHOLD
        self.click_cooldown = CLICK_COOLDOWN
        self.scroll_sensitivity = SCROLL_SENSITIVITY

        self._prev_x = 0
        self._prev_y = 0
        self._last_click_time = 0
        self._last_right_click_time = 0
        self._prev_wrist_y = None
        self._thread = None
        self.frame = None
        self.fps = 0
        self.gesture_label = "No hand"
        self._frame_lock = threading.Lock()

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def get_frame(self):
        with self._frame_lock:
            return self.frame.copy() if self.frame is not None else None

    def _loop(self):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        landmarker = HandLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        prev_time = time.time()
        start_time = prev_time

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp_ms = int((time.time() - start_time) * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            gesture = "No hand"

            if result.hand_landmarks and not self.paused:
                landmarks = result.hand_landmarks[0]
                draw_hand(frame, landmarks, w, h)

                fingers, pts = get_finger_states(landmarks, w, h)
                index_tip = pts["index"]
                wrist_y = int(landmarks[0].y * h)

                extended = sum(fingers[f] for f in ["index", "middle", "ring", "pinky"])
                all_up = extended == 4
                only_index = (fingers["index"] and not fingers["middle"]
                              and not fingers["ring"] and not fingers["pinky"])
                index_middle = (fingers["index"] and fingers["middle"]
                                and not fingers["ring"] and not fingers["pinky"])
                fist = extended == 0
                pinch_dist = get_distance(4, 8, landmarks)
                now = time.time()

                if fist:
                    gesture = "Fist - Paused"

                elif all_up:
                    gesture = "Open Palm - Scroll"
                    if self._prev_wrist_y is not None:
                        delta = self._prev_wrist_y - wrist_y
                        if abs(delta) > 2:
                            pyautogui.scroll(int(delta * self.scroll_sensitivity))
                    self._prev_wrist_y = wrist_y

                elif index_middle:
                    gesture = "Index+Middle - Right Click"
                    if now - self._last_right_click_time > self.click_cooldown:
                        pyautogui.rightClick()
                        self._last_right_click_time = now

                elif only_index:
                    gesture = "Index - Move"
                    margin_x = w * 0.03
                    margin_y = h * 0.03
                    mapped_x = np.interp(index_tip[0], [margin_x, w - margin_x], [0, SCREEN_WIDTH])
                    mapped_y = np.interp(index_tip[1], [margin_y, h - margin_y], [0, SCREEN_HEIGHT])
                    mapped_x = float(np.clip(mapped_x, 0, SCREEN_WIDTH))
                    mapped_y = float(np.clip(mapped_y, 0, SCREEN_HEIGHT))

                    alpha = self.smoothing
                    smooth_x = alpha * mapped_x + (1 - alpha) * self._prev_x
                    smooth_y = alpha * mapped_y + (1 - alpha) * self._prev_y
                    self._prev_x, self._prev_y = smooth_x, smooth_y
                    pyautogui.moveTo(smooth_x, smooth_y)

                    if pinch_dist < self.pinch_threshold:
                        gesture = "Pinch - Left Click"
                        if now - self._last_click_time > self.click_cooldown:
                            pyautogui.click()
                            self._last_click_time = now
                else:
                    self._prev_wrist_y = None

                cv2.putText(frame, gesture, (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                for name, pt in pts.items():
                    color = (0, 255, 0) if fingers[name] else (0, 0, 255)
                    cv2.circle(frame, pt, 8, color, -1)

            elif self.paused:
                cv2.putText(frame, "PAUSED", (w // 2 - 60, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            curr_time = time.time()
            self.fps = 1 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            self.gesture_label = gesture
            with self._frame_lock:
                self.frame = frame

        cap.release()
        landmarker.close()

    def toggle_pause(self):
        self.paused = not self.paused
