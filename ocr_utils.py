import cv2
import time
from collections import Counter
import easyocr
from concurrent.futures import ThreadPoolExecutor
import requests
import numpy as np
import csv
from datetime import datetime
import re # Import the regular expression module for advanced pattern matching

# --- Global Initialization ---
# Initialize the OCR reader once to avoid reloading the model. GPU is used if available.
reader = easyocr.Reader(['en'], gpu=True)
# Create a thread pool to run the heavy OCR task without blocking the main video loop.
executor = ThreadPoolExecutor(max_workers=2)

# --- Constants ---
# The minimum confidence score from OCR for a result to be considered.
CONFIDENCE_THRESHOLD = 0.15
# The name of the CSV file where detected plates will be logged.
LOG_FILE = 'detection_log.csv'

# --- Data Storage ---
# A dictionary to hold the best quality image ('best shot') for each tracked vehicle.
# Key: track_id, Value: {'image': np.array, 'sharpness': float, 'bbox': tuple}
plate_best_shots = {}

def calculate_sharpness(image):
    """
    Calculates the sharpness of a given image using the variance of the Laplacian.
    A higher variance value corresponds to a sharper, more in-focus image.

    Args:
        image (np.array): The input image.

    Returns:
        float: The calculated sharpness score. Returns 0 if the image is invalid.
    """
    if image is None or image.size == 0:
        return 0
    # Convert to grayscale as sharpness is independent of color.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # The variance of the Laplacian highlights edges; more variance means sharper edges.
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def enhance_plate(plate):
    """
    Applies a series of advanced image processing techniques to enhance the license plate
    image, making it cleaner and easier for the OCR engine to read. This version
    includes sharpening and auto-contrast adjustments as requested.

    Args:
        plate (np.array): The cropped license plate image.

    Returns:
        np.array: The enhanced, binarized (black and white) image.
    """
    if plate is None or plate.size == 0:
        return plate

    # 1. Resize to a larger, consistent height for more detail.
    TARGET_PLATE_HEIGHT = 80
    height, width = plate.shape[:2]
    scale = TARGET_PLATE_HEIGHT / height
    new_width = int(width * scale)
    resized_plate = cv2.resize(plate, (new_width, TARGET_PLATE_HEIGHT), interpolation=cv2.INTER_LANCZOS4)

    # 2. Convert to grayscale.
    gray = cv2.cvtColor(resized_plate, cv2.COLOR_BGR2GRAY)

    # 3. Increase Sharpness using Unsharp Masking.
    # This technique subtracts a blurred version of the image from the original,
    # which effectively sharpens the edges of the characters.
    gaussian_blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.5, gaussian_blur, -0.5, 0)

    # 4. Auto-adjust Brightness and Contrast.
    # This normalizes the image to use the full range of pixel intensities,
    # making the text stand out more regardless of lighting conditions.
    alpha = 1.2 # Contrast control
    beta = -20   # Brightness control (a negative value lowers brightness)
    auto_contrast = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)

    # 5. Denoise the image while preserving edges.
    denoised = cv2.bilateralFilter(auto_contrast, 9, 75, 75)

    # 6. Binarization using Otsu's thresholding for the final clean image.
    _, thresholded = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresholded

def log_plate_to_csv(plate_text, timestamp):
    """
    Logs a successfully detected plate number and the current timestamp to a CSV file.
    Creates the file and adds headers if it doesn't exist.

    Args:
        plate_text (str): The detected license plate number.
        timestamp (str): The formatted timestamp of the detection.
    """
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        # If the file is empty, write the header row first.
        if file.tell() == 0:
            writer.writerow(['Timestamp', 'License Plate'])
        writer.writerow([timestamp, plate_text])

def post_process_ocr_text(text):
    """
    Cleans up raw OCR text by correcting common character misinterpretations
    (e.g., mistaking the letter 'O' for the number '0').

    Args:
        text (str): The raw text from the OCR engine.

    Returns:
        str: The cleaned and corrected text.
    """
    if not text:
        return ""
    # This dictionary maps common OCR errors to their correct characters.
    char_map = { 'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8' }
    return "".join(char_map.get(char, char) for char in text)

def find_best_plate_candidate(detected_texts):
    """
    Intelligently analyzes a list of text fragments from OCR to find the most
    plausible license plate number. It tries multiple strategies.

    Args:
        detected_texts (list): The raw output list from easyocr.

    Returns:
        tuple: A tuple containing (best_plate_str, best_confidence_score).
               Returns ("", 0.0) if no plausible plate is found.
    """
    best_candidate = ""
    best_confidence = 0.0
    
    # Define a robust regex pattern to match common Indian license plates.
    # This looks for patterns like MH20EE7597, UP32AB1234, etc.
    plate_pattern = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$')

    # --- Strategy 1: Find a single fragment that is already a valid plate ---
    for _, raw_text, confidence in detected_texts:
        cleaned_text = post_process_ocr_text(raw_text.strip().upper())
        if plate_pattern.match(cleaned_text) and confidence > best_confidence:
            best_candidate = cleaned_text
            best_confidence = confidence
    
    # If we found a great single candidate, return it.
    if best_candidate and best_confidence > 0.4: # Higher threshold for a clean single find
        print(f"[DEBUG] Found high-confidence single fragment candidate: '{best_candidate}'")
        return best_candidate, best_confidence

    # --- Strategy 2: If no single fragment worked, combine everything and search within ---
    # This handles cases where OCR incorrectly splits a plate or adds junk characters.
    combined_text = "".join([res[1] for res in detected_texts]).replace(" ", "").strip().upper()
    cleaned_combined = post_process_ocr_text(combined_text)
    
    # Search for the plate pattern within the messy combined string.
    match = plate_pattern.search(cleaned_combined)
    if match:
        # If a match is found, extract it.
        extracted_plate = match.group(0)
        # Find the average confidence of the fragments that make up this extracted plate.
        # This is a proxy for the confidence of the extracted result.
        avg_confidence = np.mean([res[2] for res in detected_texts])
        print(f"[DEBUG] Extracted candidate '{extracted_plate}' from combined string '{cleaned_combined}'")
        return extracted_plate, avg_confidence

    # If all strategies fail, return the best single fragment we found, even if it's not perfect.
    return best_candidate, best_confidence

def ocr_and_log_plate(track_id, plate_image):
    """
    The core function that orchestrates the OCR process for a single vehicle's best image.
    It enhances the image, runs OCR, and uses intelligent logic to find and validate the plate.
    """
    if plate_image is None:
        return

    print(f"[DEBUG] Starting OCR for track ID: {track_id}...")
    enhanced_plate = enhance_plate(plate_image)
    
    detected_texts = reader.readtext(enhanced_plate, detail=1, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", paragraph=False, decoder="wordbeamsearch")

    if not detected_texts:
        print(f"[DEBUG] EasyOCR found no text for track ID: {track_id}.")
        return
    else:
        print(f"[DEBUG] EasyOCR raw output for ID-{track_id}: {detected_texts}")

    # Use our new intelligent function to find the best possible plate number.
    final_plate_text, final_confidence = find_best_plate_candidate(detected_texts)

    print(f"[DEBUG] Final candidate for ID-{track_id}: '{final_plate_text}' with confidence {final_confidence:.2f}")

    # Final check: Is the best candidate we found good enough?
    if final_plate_text and final_confidence > CONFIDENCE_THRESHOLD:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"✅ Final OCR for ID-{track_id}: {final_plate_text} (Confidence: {final_confidence:.2f})")
        log_plate_to_csv(final_plate_text, timestamp)
    else:
        print(f"[DEBUG] Result for ID-{track_id} rejected. Final candidate='{final_plate_text}', confidence={final_confidence:.2f}")


def process_lost_tracks(lost_track_ids):
    """
    Processes vehicles that have left the camera's view. It submits their best-shot
    image to the OCR thread pool for final processing.
    """
    if lost_track_ids:
        print(f"[DEBUG] Processing lost track IDs: {lost_track_ids}")

    for track_id in lost_track_ids:
        if track_id in plate_best_shots:
            best_shot_info = plate_best_shots[track_id]
            # Run the heavy OCR process in a separate thread.
            executor.submit(ocr_and_log_plate, track_id, best_shot_info['image'])
            # Clean up memory by removing the processed track.
            del plate_best_shots[track_id]

