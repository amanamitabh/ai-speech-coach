import time
import queue
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from multiprocessing import Queue
from threading import Thread

# Move these into config later
STT_MODEL_SIZE = "small"
STT_DEVICE = "cuda"
STT_COMPUTE_TYPE = "float16"


def now():
    return time.time()


def make_audio_callback(audio_queue):

    # Use closure to maintain the signature of audio_callback()
    def audio_callback(indata, frames, time_info, status):
        timestamp = now()

        # Enqueue the audio chunk if audio queue is empty, or else chunk is dropped
        try:
            audio_queue.put_nowait((timestamp, indata.copy()))
    
        except queue.Full:
            pass
    
    return audio_callback


def audio_capture(audio_queue, stop_event):
    with sd.InputStream(
        samplerate=16000,
        channels=1,
        dtype=np.float32,
        blocksize=1600,
        callback=make_audio_callback(audio_queue)
    ):
        while not stop_event.is_set():
            time.sleep(0.01)


def stt_consumer(audio_queue, stop_event):

    # Load the faster whisper model
    whisper_model = WhisperModel(
        STT_MODEL_SIZE,
        device=STT_DEVICE,
        compute_type=STT_COMPUTE_TYPE
    )

    # Debug statement (add to logging later)
    print(f"Model: {STT_MODEL_SIZE} loaded on {STT_DEVICE}")

    audio_buffer = []
    BUFFER_DURATION = 1.5 # In Seconds
    SAMPLE_RATE = 16000

    while not stop_event.is_set():
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

            # Start transcription when audio chunk exceeds duration threshold
            if duration >= BUFFER_DURATION:
                segments, info = whisper_model.transcribe(
                    audio_data.flatten(),
                    language="en",
                    beam_size=1,
                    vad_filter=True
                )

                for segment in segments:
                        print(f"[STT {timestamp}] {segment.text}")

                # Clear buffer after transcription
                audio_buffer = []

        except queue.Empty:
            pass


def audio_pipeline(stop_event):

    # Create multiprocessing safe audio queue
    audio_queue = Queue(maxsize=100)

    # Separate threads for audio caprue and speech-to-text
    threads = [
        Thread(target=audio_capture, args=(audio_queue, stop_event), daemon=True),
        Thread(target=stt_consumer, args=(audio_queue, stop_event), daemon=True)
    ]

    # Start threads
    for thread in threads:
        thread.start()

    try:
        stop_event.wait()
    finally:
        # Shutdown threads when stop event is received from main process
        print("Audio Pipeline shutting down...")
        for thread in threads:
            thread.join(timeout=2.0)