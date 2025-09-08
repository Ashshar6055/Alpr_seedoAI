import cv2
import time
from detection import load_model, detect_plates
from tracking import initialize_tracker, track_objects
from recognition import recognize_plate
from utils import calculate_fps
from ocr_utils import process_lost_tracks # <-- IMPORTANT: Import changed

def main(video_path):
    model = load_model()
    tracker = initialize_tracker()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_path}")
        return

    cv2.namedWindow("License Plate Detector", cv2.WINDOW_NORMAL)

    # --- NEW: Keep track of active and lost track IDs ---
    active_track_ids = set()

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended.")
            break

        detections = detect_plates(model, frame)
        tracked_objects = track_objects(tracker, detections)

        # Get the set of currently tracked IDs
        current_track_ids = {int(obj[4]) for obj in tracked_objects}

        # --- NEW: Find which tracks were lost in this frame ---
        lost_track_ids = active_track_ids - current_track_ids
        if lost_track_ids:
            process_lost_tracks(lost_track_ids)

        # Update the active tracks for the next frame
        active_track_ids = current_track_ids

        # This function now just saves the best shot, it doesn't do OCR
        recognize_plate(frame, tracked_objects)

        avg_fps = calculate_fps(start_time)
        cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("License Plate Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # --- NEW: Process any remaining tracks when the video ends ---
    print("Processing any remaining tracks...")
    process_lost_tracks(active_track_ids)
    time.sleep(2) # Give threads time to finish

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # --- CHOOSE YOUR VIDEO SOURCE ---
    # To use a local video file:
    main("testing1.mp4")

    # To use a webcam:
    # main(0)

    # To use an RTSP stream:
    # main("rtsp://username:password@your_camera_ip:554/stream1")

