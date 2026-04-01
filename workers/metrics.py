from collections import deque

class SpeechMetrics:
    def __init__(self, window_size=5, smoothing=0.2):
        self.window_size = window_size
        self.buckets = deque()   # stored as (second, word_count)
        self.smoothed_wpm = 0
        self.alpha = smoothing   # smoothing factor

    def add_words(self, word_count, timestamp):
        sec = int(timestamp)

        if self.buckets and self.buckets[-1][0] == sec:
            # Add to last bucket if timestamp and 'second' of last bucket are the same
            prev_sec, prev_count = self.buckets[-1]
            self.buckets[-1] = (prev_sec, prev_count + word_count)
        else:
            # Add new bucket
            self.buckets.append((sec, word_count))

        # Remove buckets outside the sliding window
        cutoff = sec - self.window_size
        while self.buckets and self.buckets[0][0] < cutoff:
            self.buckets.popleft()

    def get_wpm(self):
        if not self.buckets:
            return 0

        total_words = sum(count for _, count in self.buckets)
        raw_wpm = (total_words / self.window_size) * 60

        # EWMA smoothing to avoid jittery WPM
        self.smoothed_wpm = (
            self.alpha * raw_wpm + (1 - self.alpha) * self.smoothed_wpm
        )

        return round(self.smoothed_wpm, 1)