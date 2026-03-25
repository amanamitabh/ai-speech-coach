import cv2
import threading

def video_capture():
    cap = cv2.VideoCapture(0)

    while not STOP.is_set():
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
    

def run_video():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()

run_video()