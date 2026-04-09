from collections import deque
import time


class EngagementTracker:
    def __init__(self):
        # Initialize sliding window buffers
        self.eye_buffer = deque(maxlen=30)       # 30 frames
        self.wpm_buffer = deque(maxlen=5)
        self.filler_buffer = deque(maxlen=5)

        # Internal state
        self.prev_score = 50
        self.last_update_time = time.time()


    def update_eye_contact(self, eye_score):
        self.eye_buffer.append(eye_score)   # eye score: 0–100


    def update_speech(self, wpm, filler_density):
        self.wpm_buffer.append(wpm) # live estimate of wpm
        self.filler_buffer.append(filler_density)   # fillers per window


    def compute_speech_score(self, wpm):
        # Maximum score for maintaining ideal range (120-160)
        if 120 <= wpm <= 160:
            return 100
        deviation = abs(wpm - 140)
        return max(0, 100 - deviation * 0.8)


    def compute_filler_score(self, filler_density):
        return max(0, 100 - filler_density * 8)


    def get_avg(self, buffer, default=0):
        return sum(buffer) / len(buffer) if buffer else default


    def get_realtime_score(self):

        # Provide slightly delayed updates
        current_time = time.time()
        if current_time - self.last_update_time < 0.3:
            return round(self.prev_score, 2), "Stable"

        self.last_update_time = current_time

        # Aggregate inputs
        eye_score = self._get_avg(self.eye_buffer, 50)
        wpm = self._get_avg(self.wpm_buffer, 140)
        filler_density = self._get_avg(self.filler_buffer, 0)

        # Handle silence
        if wpm < 20:
            speech_score = 50
        else:
            speech_score = self._compute_speech_score(wpm)

        filler_score = self._compute_filler_score(filler_density)

        # Instant score
        instant_score = (
            0.4 * eye_score +
            0.3 * speech_score +
            0.3 * filler_score
        )

        # EWMA smoothing
        alpha = 0.2
        smoothed = alpha * instant_score + (1 - alpha) * self.prev_score

        # ---- Trend detection ----
        if smoothed > self.prev_score + 1:
            trend = "Improving"
        elif smoothed < self.prev_score - 1:
            trend = "Dropping"
        else:
            trend = "Stable"

        self.prev_score = smoothed

        return round(smoothed, 2), trend