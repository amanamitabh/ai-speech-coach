import os
import time
from dotenv import load_dotenv
from multiprocessing import Process, Event, Queue
from workers.video_process import video_pipeline
from workers.audio_process import audio_pipeline
from workers.llm_speech_analyzer import analyze_transcript


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

    # Multiprocessing queue for transcript
    transcript_queue = Queue()
    raw_transcript = ""

    # Separate processes for audio and video processing
    print("Starting audio and video capture...")
    processes = [
        Process(target=video_pipeline, args=(stop_event,), name="VideoWorker"),
        Process(target=audio_pipeline, args=(stop_event, transcript_queue), name="AudioWorker")
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
        
        # Wait to ensure child processes push transcript
        time.sleep(2)



    finally:
        
        # Ensure stop event is sent on normal exit
        stop_event.set()

        # Read transcript from queue
        try:
            if not transcript_queue.empty():
                raw_transcript = transcript_queue.get()
            else:
                raw_transcript = transcript_queue.get(timeout=5)                
        except:
            print("No transcript received!")

        # Shutdown all processes
        initiate_shutdown(processes)

        # Print transcripts
        print(f"Raw transcript: {raw_transcript}")
        refined_transcript = analyze_transcript(raw_transcript)
        print("\nRefined Transcript:")
        print(refined_transcript)

if __name__ == "__main__":
    main()