import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import math
import threading

# ---------------- CONFIGURATION ----------------
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
SMOOTHING = 0.18   # EMA factor: lower = smoother but more lag (0.1–0.4)
PINCH_THRESHOLD = 0.045
CLICK_COOLDOWN = 0.5
SCROLL_SENSITIVITY = 0.8

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# ---------------- MEDIAPIPE SETUP ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

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
    """Return dict of which fingers are extended."""
    pts = {name: (int(landmarks[tip].x * w), int(landmarks[tip].y * h))
           for name, tip in TIP_IDS.items()}
    mcp = {name: (int(landmarks[mid].x * w), int(landmarks[mid].y * h))
           for name, mid in MCP_IDS.items()}

    # Thumb: compare x with MCP of thumb
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


def get_distance(p1, p2, w, h, landmarks):
    """Normalized distance between two landmark tips."""
    x1 = landmarks[p1].x
    y1 = landmarks[p1].y
    x2 = landmarks[p2].x
    y2 = landmarks[p2].y
    return math.hypot(x2 - x1, y2 - y1)


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
        self._cap = None
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
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        prev_time = time.time()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            gesture = "No hand"

            if result.multi_hand_landmarks and not self.paused:
                lm = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                fingers, pts = get_finger_states(lm.landmark, w, h)
                index_tip = pts["index"]
                wrist_y = int(lm.landmark[0].y * h)

                # --- Gesture detection ---
                extended = sum(fingers[f] for f in ["index", "middle", "ring", "pinky"])
                all_up = extended == 4

                only_index = fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]
                index_middle = fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]
                fist = extended == 0
                pinch_dist = get_distance(4, 8, w, h, lm.landmark)

                now = time.time()

                if fist:
                    gesture = "Fist - Paused"

                elif all_up:
                    gesture = "Open Palm - Scroll"
                    if self._prev_wrist_y is not None:
                        delta = self._prev_wrist_y - wrist_y
                        if abs(delta) > 2:
                            scroll_amount = int(delta * self.scroll_sensitivity)
                            pyautogui.scroll(scroll_amount)
                    self._prev_wrist_y = wrist_y

                elif index_middle:
                    # Right click
                    gesture = "Index+Middle - Right Click"
                    if now - self._last_right_click_time > self.click_cooldown:
                        pyautogui.rightClick()
                        self._last_right_click_time = now

                elif only_index:
                    gesture = "Index - Move"
                    # Tight margins so hand can reach all screen edges
                    margin_x = w * 0.03
                    margin_y = h * 0.03
                    mapped_x = np.interp(index_tip[0], [margin_x, w - margin_x], [0, SCREEN_WIDTH])
                    mapped_y = np.interp(index_tip[1], [margin_y, h - margin_y], [0, SCREEN_HEIGHT])
                    mapped_x = float(np.clip(mapped_x, 0, SCREEN_WIDTH))
                    mapped_y = float(np.clip(mapped_y, 0, SCREEN_HEIGHT))

                    # EMA smoothing: alpha closer to 1 = faster, closer to 0 = smoother
                    alpha = self.smoothing
                    smooth_x = alpha * mapped_x + (1 - alpha) * self._prev_x
                    smooth_y = alpha * mapped_y + (1 - alpha) * self._prev_y
                    self._prev_x, self._prev_y = smooth_x, smooth_y

                    pyautogui.moveTo(smooth_x, smooth_y)

                    # Check pinch for left click
                    if pinch_dist < self.pinch_threshold:
                        gesture = "Pinch - Left Click"
                        if now - self._last_click_time > self.click_cooldown:
                            pyautogui.click()
                            self._last_click_time = now
                else:
                    self._prev_wrist_y = None

                # Draw gesture label on frame
                cv2.putText(frame, gesture, (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Draw fingertips
                for name, pt in pts.items():
                    color = (0, 255, 0) if fingers[name] else (0, 0, 255)
                    cv2.circle(frame, pt, 8, color, -1)

            elif self.paused:
                cv2.putText(frame, "PAUSED", (w // 2 - 60, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            # FPS
            curr_time = time.time()
            self.fps = 1 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            self.gesture_label = gesture
            with self._frame_lock:
                self.frame = frame

        cap.release()

    def toggle_pause(self):
        self.paused = not self.paused
