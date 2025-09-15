import cv2
import time
import pytesseract
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import csv
from datetime import datetime
import re

# --- Global Initialization ---
executor = ThreadPoolExecutor(max_workers=2)

# --- Constants ---
CONFIDENCE_THRESHOLD = 0.15 # Minimum OCR confidence to accept a result.
PLAUSIBILITY_THRESHOLD = 0.4 # Minimum score for a text to be considered a valid plate.
LOG_FILE = 'detection_log.csv'

# --- Data Storage ---
plate_best_shots = {}
realtime_ocr_results = {}

def calculate_sharpness(image):
    """Calculates the sharpness of an image using the variance of the Laplacian."""
    if image is None or image.size == 0:
        return 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def enhance_plate(plate):
    """Applies an advanced image processing pipeline to enhance the plate for OCR."""
    if plate is None or plate.size == 0:
        return plate

    TARGET_PLATE_HEIGHT = 80
    height, width = plate.shape[:2]
    scale = TARGET_PLATE_HEIGHT / height
    new_width = int(width * scale)
    resized_plate = cv2.resize(plate, (new_width, TARGET_PLATE_HEIGHT), interpolation=cv2.INTER_LANCZOS4)

    gray = cv2.cvtColor(resized_plate, cv2.COLOR_BGR2GRAY)
    gaussian_blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.5, gaussian_blur, -0.5, 0)
    alpha = 1.2
    beta = -20
    auto_contrast = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)
    denoised = cv2.bilateralFilter(auto_contrast, 9, 75, 75)
    _, thresholded = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresholded

def log_plate_to_csv(plate_text, timestamp):
    """Logs a detected plate and timestamp to a CSV file."""
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['Timestamp', 'License Plate'])
        writer.writerow([timestamp, plate_text])

def post_process_ocr_text(text):
    """Corrects common OCR character misinterpretations."""
    if not text:
        return ""
    char_map = { 'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8' }
    return "".join(char_map.get(char, char) for char in text)

def score_plate_candidate(plate_text):
    """
    Scores a string based on how much it resembles an Indian license plate.
    Returns a score between 0.0 and 1.0.
    """
    score = 0.0
    
    # 1. Length Score (worth 30%)
    if 6 <= len(plate_text) <= 10:
        score += 0.3

    # 2. Starts with two letters (worth 30%)
    if len(plate_text) >= 2 and plate_text[0].isalpha() and plate_text[1].isalpha():
        score += 0.3

    # 3. Has a good mix of letters and numbers (worth 40%)
    num_digits = sum(c.isdigit() for c in plate_text)
    num_letters = sum(c.isalpha() for c in plate_text)
    if num_letters >= 2 and num_digits >= 3:
        score += 0.4
        
    return score

def ocr_and_log_plate(track_id, plate_image):
    """Orchestrates the OCR process using Tesseract and a scoring model."""
    if plate_image is None:
        return

    print(f"[DEBUG] Starting Tesseract OCR for track ID: {track_id}...")
    enhanced_plate = enhance_plate(plate_image)
    
    custom_config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    ocr_data = pytesseract.image_to_data(enhanced_plate, config=custom_config, output_type=pytesseract.Output.DICT)
    
    best_candidate = ""
    best_confidence = 0.0
    best_score = 0.0

    # Find the best candidate from all text fragments found by Tesseract.
    for i in range(len(ocr_data['text'])):
        confidence = int(ocr_data['conf'][i]) / 100.0
        if confidence > 0: # Consider only actual words
            raw_text = ocr_data['text'][i]
            cleaned_text = post_process_ocr_text(raw_text.strip().upper())
            
            plausibility_score = score_plate_candidate(cleaned_text)
            
            # We are looking for the candidate with the best combination of
            # looking like a plate and having high OCR confidence.
            if plausibility_score * confidence > best_score * best_confidence:
                best_candidate = cleaned_text
                best_confidence = confidence
                best_score = plausibility_score
    
    # Update the real-time display with the best attempt we found.
    realtime_ocr_results[track_id] = best_candidate + "?" if best_candidate else f"ID {track_id}"

    if not best_candidate:
        print(f"[DEBUG] Tesseract found no plausible text for ID-{track_id}.")
        return

    # --- FINAL DECISION LOGIC ---
    if best_score >= PLAUSIBILITY_THRESHOLD and best_confidence >= CONFIDENCE_THRESHOLD:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"✅ Final OCR for ID-{track_id}: {best_candidate} (Plausibility: {best_score:.2f}, Confidence: {best_confidence:.2f})")
        log_plate_to_csv(best_candidate, timestamp)
        # Update the display to show it's a confirmed plate (remove the '?')
        realtime_ocr_results[track_id] = best_candidate
    else:
        # If not, print the best attempt for debugging and demo purposes.
        print(f"[INFO] Best attempt for ID-{track_id}: '{best_candidate}' (Plausibility: {best_score:.2f}, Confidence: {best_confidence:.2f}) - Rejected.")

def process_lost_tracks(lost_track_ids):
    """Processes vehicles that have left the view, sending their best image for OCR."""
    if lost_track_ids:
        print(f"[DEBUG] Processing lost track IDs: {lost_track_ids}")

    for track_id in lost_track_ids:
        if track_id in plate_best_shots:
            best_shot_info = plate_best_shots[track_id]
            executor.submit(ocr_and_log_plate, track_id, best_shot_info['image'])
            del plate_best_shots[track_id]
        if track_id in realtime_ocr_results:
            del realtime_ocr_results[track_id]

