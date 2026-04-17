# Virtual Coach

## Overview
Virtual Coach is a real time, multimodal system designed to analyze and improve public speaking performance. It processes live audio and video streams to provide feedback on speech clarity, eye contact, and speaking speed.

## Features
- Real time speech transcription using Faster-Whisper
- Eye gaze tracking using facial landmarks
- Local LLM powered transcript refinement and feedback generation
- Real time auditory feedback on speaking performance

## Setup
1. Clone the repository:
   ```bash
   git clone
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Install ffmpeg from https://ffmpeg.org/download.html and place it in the root project directory.

4. Install Ollama from https://ollama.com/download and pull the required model (Gemma3).

5. Start the Ollama server:
    ```bash
    ollama serve
    ```

6. Run the application:
    ```bash
    python main.py
    ```