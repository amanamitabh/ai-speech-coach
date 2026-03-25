import os
import time
from dotenv import load_dotenv
from multiprocessing import Process

# Adds variables from .env file to program environment variables
load_dotenv()

# Get ffmpeg binaries path and add it to PATH
ffmpeg_bin_path = os.environ["FFMPEG_BIN_PATH"]
os.environ["PATH"] = f"{ffmpeg_bin_path}{os.pathsep}{os.environ.get('PATH','')}"

# Separate processes for audio and video processing
print("Starting audio and video capture...")
processes = [
    Process(target=video_capture, args=()),
    Process(target=audio_capture, args=())
]

# Spawn child processes
for process in processes:
    process.start()

# Loop to test program
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    STOP.set()
    print("Exiting...")