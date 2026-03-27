import os
import time
from dotenv import load_dotenv
from multiprocessing import Process, Event
from workers.video_process import video_pipeline
from workers.audio_process import audio_pipeline


def check_process_health(processes, stop_event):

    if stop_event.is_set():
        return

    # Check if processes die mid execution
    for process in processes:
        if not process.is_alive():
            print(f"CRITICAL: {process.name} has stopped unexpectedly!")
            raise Exception("Worker failure")


def initiate_shutdown(processes):
    
    # Wait for child processes to exit
    for process in processes:
        process.join(timeout=5.0)

    # Forcefully terminate hanging processes
    for process in processes:
        if process.is_alive():
            print(f"Force killing {process.name}")
            process.terminate()
            process.join()    


def main():
        
    # Adds variables from .env file to program environment variables
    load_dotenv()

    # Get ffmpeg binaries path and add it to PATH
    ffmpeg_bin_path = os.environ["FFMPEG_BIN_PATH"]
    os.environ["PATH"] = f"{ffmpeg_bin_path}{os.pathsep}{os.environ.get('PATH','')}"

    # Global stop event for all processes
    stop_event = Event()

    # Separate processes for audio and video processing
    print("Starting audio and video capture...")
    processes = [
        Process(target=video_pipeline, args=(stop_event,), name="VideoWorker"),
        Process(target=audio_pipeline, args=(stop_event,), name="AudioWorker")
    ]

    # Spawn child processes
    for process in processes:
        process.start()

    try:
        while not stop_event.is_set():
            time.sleep(1)
            check_process_health(processes, stop_event)


    except (KeyboardInterrupt, Exception) as e:

        # Set the stop event and signal shut down
        print(f"Shutting down: {type(e).__name__}...")
        stop_event.set()
        initiate_shutdown(processes)


if __name__ == "__main__":
    main()