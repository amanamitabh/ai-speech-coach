import cv2
import os
import time
import threading
import queue
from dotenv import load_dotenv

def now():
    # Get current time in global clock
    return time.perf_counter()


# Adds variables from .env file to program environment variables
load_dotenv()

# Get ffmpeg binaries path and add it to PATH
ffmpeg_bin_path = os.environ["FFMPEG_BIN_PATH"]
os.environ["PATH"] = f"{ffmpeg_bin_path}{os.pathsep}{os.environ.get('PATH','')}"

# Queues for processing audio and video
video_queue = queue.Queue(maxsize=30)
audio_queue = queue.Queue(maxsize=30)

# Thread safe signal flag
STOP = threading.Event()

# Video Capture Thread
def video_capture():
    cap = cv2.VideoCapture(0)

    while not STOP.is_set():
        ret, frame = cap.read()

        if not ret:
            continue

        # Get current timestamp
        timestamp = now()

        # Enqueue the frame if video queue is empty, or else frame is dropped
        try:
            video_queue.put_nowait((timestamp, frame))
            print(f"[VIDEO]: {timestamp:.6f}")
        
        except queue.Full:
            pass

        # Display the frame
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Free allocated resources
    cap.release()
    cv2.destroyAllWindows()
