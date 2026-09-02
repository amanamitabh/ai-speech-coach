# Virtual Coach

Virtual Coach is a local, real-time speaking assistant that analyzes a presenter’s microphone and webcam feed to provide feedback on pacing, filler words, and eye contact. It helps improve public speaking, delivery, and spoken clarity before a live presentation.

## What this project does

The system combines multiple AI and signal-processing components:

- Live audio capture from the microphone
- Speech-to-text transcription using Faster Whisper
- Real-time speaking-rate tracking (WPM) and filler-word detection
- Eye contact estimation from face landmarks and camera input
- Transcript cleanup and refinement using a local Ollama model

## Key features

- Real-time speech transcription with `faster-whisper`
- WPM monitoring and haptic/auditory feedback when speaking becomes too slow
- Gaze and eye-contact estimation using MediaPipe face mesh landmarks
- Text cleanup and filler-word analysis via a local LLM with Ollama
- Parsed JSON transcript output for downstream analytics
- Multi-process orchestration to keep audio and video pipelines running independently

## Architecture overview

The project is organized around a small orchestration loop in `main.py`.

1. `main.py` creates separate processes for the audio and video pipelines.
2. The audio worker captures live audio, runs speech-to-text, computes speech metrics, and sends transcript text to the main process.
3. The video worker captures frames, calibrates gaze, estimates eye direction, and emits eye-contact scores.
4. The LLM worker calls a local Ollama endpoint to clean the transcript and detect filler words.
5. The coaching workflow aggregates these signals into a real-time speaking assessment.

## Project structure

```text
ai-speech-coach/
├── .env       
├── .gitignore
├── main.py                     # application entry point and process orchestration   
├── requirements.txt            # Python dependencies
├── README.md                   # project documentation
├── ffmpeg/                     # bundled FFmpeg binaries used by the app
├── config/                     # configuration folder
├── utils/
│   ├── esp32_feedback.py       # buzzer / feedback utilities
│   └── json_parser.py          # cleans and parses LLM JSON response
├── workers/
│   ├── audio_process.py        # audio capture + STT pipeline
│   ├── engagement_tracker.py   # realtime scoring helpers
│   ├── gaze_estimation.py      # MediaPipe-based gaze tracking
│   ├── llm_speech_analyzer.py  # Ollama prompt + transcript cleanup
│   ├── metrics.py              # speech metrics utilities
│   └── video_process.py        # camera processing and gaze pipeline
└── .venv/                      
```

## Requirements

Before running the project, install:

- Python 3.10+
- A working webcam and microphone
- FFmpeg on the system path or configured via `FFMPEG_BIN_PATH`
- Ollama with a compatible model available locally

## Environment setup

The app reads a local `.env`:

```env
FFMPEG_BIN_PATH=<path_to_ffmpeg>
```


## Installation

### 1. Clone the repository:

```bash
git clone https://github.com/amanamitabh/ai-speech-coach
cd ai-speech-coach
```

### 2. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies:

```bash
pip install -r requirements.txt
```

### 4. Install and run Ollama:

```bash
# Install Ollama from https://ollama.com/download
ollama pull gemma3:12b
ollama serve
```

### 5. Ensure FFmpeg is available to the app.

If your environment does not already include FFmpeg, the bundled local copy under `ffmpeg/bin` should be used and referenced through `FFMPEG_BIN_PATH`.

## Running the app

From the project root:

```bash
python main.py
```

This starts the audio/video capture and begins the real-time coaching pipeline. The app will attempt to:

- open the default webcam
- open the default microphone
- track speaking flow and gaze
- send a final transcript to the local LLM for cleanup
- print cleaned transcript output and filler information before exiting
