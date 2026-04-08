import cv2
import queue
import time
from multiprocessing import Queue
from threading import Thread
from workers import gaze_estimation
from workers.gaze_estimation import GazeEstimator

def now():
    return time.time()

def video_capture(video_queue, stop_event):
    cap = cv2.VideoCapture(0)

    while not stop_event.is_set():
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
    

def frame_generator(video_queue, stop_event):
    """Generator for calibrating eye gaze on initialization"""
    while not stop_event.is_set():
        try:
            # Send frame to caller and save state of generator
            _, frame = video_queue.get(timeout=0.1)
            yield frame 
        except queue.Empty:
            continue


def video_pipeline(stop_event):
    
    # Create multiprocessing safe queue for video frames
    video_queue = Queue(maxsize=2)  # smaller maxsize for frame dropping on delay

    # Create threads for 
    threads = [
        Thread(target=video_capture, args=(video_queue, stop_event), daemon=True, name="FrameCaptureThread")
    ]

    # Start threads
    for thread in threads:
        thread.start()
    
    # Create gaze estimator object
    gaze_estimator = GazeEstimator()
    
    # Calibrate eye gaze
    print("Calibrating gaze... look straight")
    gaze_estimator.calibrate(
        frame_generator(video_queue, stop_event)
    )
    print("Calibration done")

    try:
        while not stop_event.is_set():
        
            try:
                # Get the frame and timestamp from video queue
                timestamp, frame = video_queue.get(timeout=0.1)
            
            except queue.Empty:
                continue  
            
            # Process frame to estimate gaze
            frame, gaze, alert = gaze_estimator.process(frame)

            # Overlay over UI
            if alert:
                cv2.putText(frame, alert, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(frame, f"Gaze: {gaze}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Display frame and overlayed text
            cv2.imshow("Video Coach", frame)

            if cv2.waitKey(1) & 0xFF == 27: # End on ESC key
                stop_event.set()
                break
            
    except KeyboardInterrupt:
        pass    # Avoid printing traceback

    finally:
        # Shutdown threads when stop event is received
        print("Video Pipeline shutting down...")
        cv2.destroyAllWindows()
        for thread in threads:
            thread.join(timeout=2.0)