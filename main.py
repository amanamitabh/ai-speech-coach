import cv2
import os
import whisper
import json
from dotenv import load_dotenv

# Adds variables from .env file to program environment variables
load_dotenv()

# Get ffmpeg binaries path and add it to PATH
ffmpeg_bin_path = os.environ["FFMPEG_BIN_PATH"]
os.environ["PATH"] = f"{ffmpeg_bin_path}{os.pathsep}{os.environ.get('PATH','')}"

# Get video directory path and join path to video
video_dir = os.environ["VIDEO_DIR"]
video = os.path.join(video_dir, "vid2.mp4")

# Load video
cap = cv2.VideoCapture(video)

# Get FPS from properties and calculate delay
fps = cap.get(cv2.CAP_PROP_FPS)
if fps > 0:
    delay = int(1000/fps)   # Calculate delay in ms
else:
    delay = 33    # Set default delay in case fps is 0s

# Load the whisper model (use base or small)
model = whisper.load_model("base")

# Transcribe and print the result
result = model.transcribe(video, fp16=False)
print(result)

# Display video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Free resources and destroy windows
cap.release()
cv2.destroyAllWindows()