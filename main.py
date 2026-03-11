import cv2
import os
import time
import threading
import sounddevice as sd
import numpy as np
import queue
from faster_whisper import WhisperModel
from dotenv import load_dotenv

STT_MODEL_SIZE = "small"
STT_DEVICE = "cuda"
STT_COMPUTE_TYPE = "float16"

# Load the faster-whisper model
whisper_model = WhisperModel(
    STT_MODEL_SIZE,
    device=STT_DEVICE,
    compute_type=STT_COMPUTE_TYPE
)


def now():
    # Get current time in global clock
    return time.perf_counter()


# Adds variables from .env file to program environment variables
load_dotenv()

# Get ffmpeg binaries path and add it to PATH
ffmpeg_bin_path = os.environ["FFMPEG_BIN_PATH"]
os.environ["PATH"] = f"{ffmpeg_bin_path}{os.pathsep}{os.environ.get('PATH','')}"

# Queues for processing audio and video
video_queue = queue.Queue()
audio_queue = queue.Queue()

# Thread safe signal flag
STOP = threading.Event()

# VIDEO PRODUCER THREAD

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
            #print(f"[VIDEO]: {timestamp:.6f}")
        
        except queue.Full:
            pass

    # Free allocated resources
    cap.release()


# AUDIO PRODUCER THREAD

def audio_callback(indata, frames, time, status):
    timestamp = now()

    # Enqueue the audio chunk if audio queue is empty, or else chunk is dropped
    try:
        audio_queue.put_nowait((timestamp, indata.copy()))
        #print(f"[AUDIO]: {timestamp:.6f}")
    
    except queue.Full:
        pass


def audio_capture():
    with sd.InputStream(
        samplerate=16000,
        channels=1,
        dtype=np.float32,
        blocksize=1600,
        callback=audio_callback
    ):
        while not STOP.is_set():
            time.sleep(0.01)


# AUDIO CONSUMER THREAD

def stt_consumer():
    audio_buffer = []
    BUFFER_DURATION = 1.5 # In Seconds
    SAMPLE_RATE = 16000

    while not STOP.is_set():
        try:
            # Dequeue the audio chunk and load it to audio buffer
            timestamp, chunk = audio_queue.get(timeout=1)
            audio_buffer.append((timestamp, chunk))    

            # Convert buffered chunks to single numpy array
            chunks = [c for _, c in audio_buffer]
            audio_data = np.concatenate(chunks).astype(np.float32).flatten()
            start_ts = audio_buffer[0][0]
            end_ts = audio_buffer[-1][0]

            duration = len(audio_data) / SAMPLE_RATE

            if duration >= BUFFER_DURATION:
                segments, info = whisper_model.transcribe(
                    audio_data.flatten(),
                    language="en",
                    beam_size=1,
                    vad_filter=True
                )

                for segment in segments:
                        print(f"[STT {timestamp:.2f}] {segment.text}")

                # Clear buffer
                audio_buffer = []

        except queue.Empty:
            pass


# Separate threads for audio and video
print("Starting audio and video capture...")
threads = [
    threading.Thread(target=video_capture, daemon=True),
    threading.Thread(target=audio_capture, daemon=True),
    threading.Thread(target=stt_consumer, daemon=True)
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