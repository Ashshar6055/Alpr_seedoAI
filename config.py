import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model Paths
YOLO_MODEL_PATH = "models/license_plate_detector.pt"

# YOLO Settings
YOLO_CONFIDENCE_THRESHOLD = 0.4 # Lowered to catch more plates

# Tracker Settings
SORT_MAX_AGE = 5
SORT_MIN_HITS = 7
SORT_IOU_THRESHOLD = 0.2