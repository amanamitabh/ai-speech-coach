import cv2
import mediapipe as mp
import time
from collections import deque

class GazeEstimator:
    def __init__(self):

        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)   # enable iris tracking

        # Store last 5 vertical and horizontal ratios
        self.ratios_v = deque(maxlen=5)
        self.ratios_h = deque(maxlen=5)

        # Calibration centers
        self.center_v = None
        self.center_h = None

        self.down_start = None
        self.THRESHOLD = 2


    def compute_ratios(self, lm, w, h):

        # Landmarks [468:iris-center, 159:eyelid-top, 145:eyelid-bottom, 33/133:eye corners]
        iris = lm[468]
        top = lm[159]   
        bottom = lm[145]
        left = lm[33]
        right = lm[133]

        # Convert to pixel coordinates
        iris_x = iris.x * w
        iris_y = iris.y * h

        left_x = left.x * w
        right_x = right.x * w
        top_y = top.y * h
        bottom_y = bottom.y * h

        # Compute vertical and horizontal ratios
        v_ratio = (iris_y - top_y) / max((bottom_y - top_y), 1)     # 0:looking up, 1:looking down
        h_ratio = (iris_x - left_x) / max((right_x - left_x), 1)    # 0:looking left, 1:looking right   

        return v_ratio, h_ratio
    

    def calibrate(self, frame_generator, duration=3):
        calib_v, calib_h = [], []
        start_time = time.time()

        # Capture frames for given duration
        while time.time() - start_time < duration:
            frame = next(frame_generator)

            # Preprocess the raw frames
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            # Get landmarks once face is detected
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                v_ratio, h_ratio = self.compute_ratios(lm, w, h)
                calib_v.append(v_ratio)
                calib_h.append(h_ratio)

        # Calculate avg center
        self.center_v = sum(calib_v) / len(calib_v)
        self.center_h = sum(calib_h) / len(calib_h)


    def process(self, frame):

        # Preprocess the raw frames
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        # Default gaze and alert state
        gaze = "CENTER"
        alert = None

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark   # get landmarks for detected face

            # Compute vertical and horizontal ratios
            v_ratio, h_ratio = self.compute_ratios(lm, w, h)

            # Add to buffers for smoothing
            self.ratios_v.append(v_ratio)
            self.ratios_h.append(h_ratio)
            smooth_v = sum(self.ratios_v) / len(self.ratios_v)
            smooth_h = sum(self.ratios_h) / len(self.ratios_h)

            # Gaze classification based on thresholds
            if smooth_v > self.center_v + 0.08:
                gaze = "UP"
            elif smooth_v < self.center_v - 0.08:
                gaze = "DOWN"
            elif smooth_h > self.center_h + 0.08:
                gaze = "RIGHT"
            elif smooth_h < self.center_h - 0.08:
                gaze = "LEFT"

            # Alert logic
            if gaze == "DOWN":
                if self.down_start is None:
                    self.down_start = time.time()
                elif time.time() - self.down_start > self.THRESHOLD:
                    alert = "LOOK UP!"
            else:
                self.down_start = None

        return frame, gaze, alert