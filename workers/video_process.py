import cv2
import time
from multiprocessing import Queue
from threading import Thread
import queue

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
    
    try:
        while not stop_event.is_set():
        
            try:
                
                # Get the frame and timestamp from video queue and display
                timestamp, frame = video_queue.get(timeout=0.1)
                cv2.imshow("Video Coach", frame)

                if cv2.waitKey(1) & 0xFF == 27: # End on ESC key
                    stop_event.set()
                    break
            
            except queue.Empty:
                continue  

    except KeyboardInterrupt:
        pass    # Avoid printing traceback

    finally:
        # Shutdown threads when stop event is received
        print("Video Pipeline shutting down...")
        cv2.destroyAllWindows()
        for thread in threads:
            thread.join(timeout=2.0)