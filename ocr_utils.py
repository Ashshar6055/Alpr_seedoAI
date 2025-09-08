import cv2
import time
from collections import Counter
import easyocr
from concurrent.futures import ThreadPoolExecutor
import requests
import numpy as np
import csv
from datetime import datetime

# Initialize OCR reader globally
reader = easyocr.Reader(['en'], gpu=True)
executor = ThreadPoolExecutor(max_workers=2) # OCR is heavy, fewer workers are better

# Constants
CONFIDENCE_THRESHOLD = 0.7 # Lowered slightly for the single best shot
LOG_FILE = 'detection_log.csv'

# --- NEW: Dictionary to store the best shot for each track ID ---
plate_best_shots = {}

def calculate_sharpness(image):
    """Calculates the sharpness of an image using the variance of the Laplacian."""
    if image is None or image.size == 0:
        return 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # A higher variance corresponds to a sharper image
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def enhance_plate(plate):
    """Enhances the plate image for better OCR results."""
    if plate is None or plate.size == 0:
        return plate
    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_plate = cv2.filter2D(plate, -1, sharpening_kernel)
    gray = cv2.cvtColor(sharpened_plate, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced

def log_plate_to_csv(plate_text, timestamp):
    """Logs a detected plate number and timestamp to a CSV file."""
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['Timestamp', 'License Plate'])
        writer.writerow([timestamp, plate_text])

def ocr_and_log_plate(track_id, plate_image):
    """Performs OCR on the best plate image and logs the result."""
    if plate_image is None:
        return

    enhanced_plate = enhance_plate(plate_image)
    detected_texts = reader.readtext(enhanced_plate, detail=1, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", paragraph=False, decoder="wordbeamsearch")

    if not detected_texts:
        return

    # Get the result with the highest confidence
    best_text, best_conf = "", 0.0
    for _, text, conf in detected_texts:
        if conf > best_conf:
            best_text, best_conf = text, conf
            
    stable_plate = best_text.strip().upper()

    if stable_plate and len(stable_plate) > 3 and best_conf > CONFIDENCE_THRESHOLD:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"✅ Final OCR for ID-{track_id}: {stable_plate} (Confidence: {best_conf:.2f})")
        log_plate_to_csv(stable_plate, timestamp)
        
        # --- You can uncomment this section when your server is back online ---
        # vehicle_data = {"vehicle_id": stable_plate, "type": "CheckIn"}
        # try:
        #     response = requests.post("http://localhost:9000/api/v1/vehicle/mark", json=vehicle_data, timeout=5)
        #     print(f"API Response: {response.json()}")
        # except requests.RequestException as e:
        #     print(f"API Error for plate {stable_plate}: {e}")

def process_lost_tracks(lost_track_ids):
    """Processes tracks that have been lost, performing OCR on their best shot."""
    for track_id in lost_track_ids:
        if track_id in plate_best_shots:
            best_shot_info = plate_best_shots[track_id]
            # Submit the final OCR task to the thread pool
            executor.submit(ocr_and_log_plate, track_id, best_shot_info['image'])
            # Clean up the memory for this track ID
            del plate_best_shots[track_id]

