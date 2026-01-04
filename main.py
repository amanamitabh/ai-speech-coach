import cv2
import os
import time
import threading
import sounddevice as sd
import numpy as np
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

# VIDEO THREAD

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

    # Free allocated resources
    cap.release()


# AUDIO THREAD

def audio_callback(indata, frames, time, status):
    timestamp = now()

    # Enqueue the audio chunk if audio queue is empty, or else chunk is dropped
    try:
        audio_queue.put_nowait((timestamp, indata.copy()))
        print(f"[AUDIO]: {timestamp:.6f}")
    
    except queue.Full:
        pass


def audio_capture():
    with sd.InputStream(
        samplerate=16000,
        channels=1,
        dtype=np.float32,
        blocksize=800,
        callback=audio_callback
    ):
        while not STOP.is_set():
            time.sleep(0.01)


# Separate threads for audio and video
threads = [
    threading.Thread(target=video_capture, daemon=True),
    threading.Thread(target=audio_capture, daemon=True)
]

# Start each thread
for thread in threads:
    thread.start()


# Loop to test program
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    STOP.set()
    print("Exiting...")