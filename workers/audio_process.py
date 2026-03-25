import sounddevice as sd
import numpy as np


STT_MODEL_SIZE = "small"
STT_DEVICE = "cuda"
STT_COMPUTE_TYPE = "float16"


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


def stt_consumer():

    # Load the faster whisper model
    whisper_model = WhisperModel(
        STT_MODEL_SIZE,
        device=STT_DEVICE,
        compute_type=STT_COMPUTE_TYPE
    )

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