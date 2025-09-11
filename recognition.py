import cv2
from ocr_utils import calculate_sharpness, plate_best_shots, realtime_ocr_results

def recognize_plate(frame, tracked_objects):
    """
    Saves the best (sharpest) image of a plate and draws the latest OCR result on the frame.
    """
    h, w, _ = frame.shape

    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, obj)

        # Add padding to the bounding box to ensure the whole plate is captured.
        padding = 5
        x1_pad, y1_pad = max(0, x1 - padding), max(0, y1 - padding)
        x2_pad, y2_pad = min(w, x2 + padding), min(h, y2 + padding)
        plate_image = frame[y1_pad:y2_pad, x1_pad:x2_pad]

        if plate_image.size == 0:
            continue

        # Calculate sharpness and update the 'best shot' if the current one is better.
        sharpness = calculate_sharpness(plate_image)
        if track_id not in plate_best_shots or sharpness > plate_best_shots[track_id]['sharpness']:
            plate_best_shots[track_id] = {'image': plate_image.copy(), 'sharpness': sharpness}

        # --- REAL-TIME DISPLAY LOGIC ---
        # Draw the bounding box.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Get the latest OCR result for this track_id from the shared dictionary.
        # If no result is available yet, default to showing the track ID.
        display_text = realtime_ocr_results.get(track_id, f"ID {track_id}")

        # Draw the text on the frame.
        cv2.putText(frame, display_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


