import cv2
from ocr_utils import calculate_sharpness, plate_best_shots

def recognize_plate(frame, tracked_objects):
    """
    Finds the best (sharpest) image of a license plate for each tracked object.
    This function does NOT perform OCR directly anymore.
    """
    h, w, _ = frame.shape

    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, obj)
        padding = 5
        x1_pad, y1_pad = max(0, x1 - padding), max(0, y1 - padding)
        x2_pad, y2_pad = min(w, x2 + padding), min(h, y2 + padding)

        plate_image = frame[y1_pad:y2_pad, x1_pad:x2_pad]

        if plate_image.size == 0:
            continue

        # Calculate the sharpness of the current plate crop
        sharpness = calculate_sharpness(plate_image)

        # Check if we have a record for this track_id
        if track_id not in plate_best_shots:
            # If not, this is the best shot so far
            plate_best_shots[track_id] = {'image': plate_image, 'sharpness': sharpness, 'bbox': (x1, y1, x2, y2)}
        else:
            # If we do, check if the current shot is sharper
            if sharpness > plate_best_shots[track_id]['sharpness']:
                plate_best_shots[track_id] = {'image': plate_image, 'sharpness': sharpness, 'bbox': (x1, y1, x2, y2)}
    
    # Draw bounding boxes on the frame for visualization
    for track_id, data in plate_best_shots.items():
        x1, y1, x2, y2 = data['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
