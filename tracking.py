import sys
import numpy as np
from config import SORT_MAX_AGE, SORT_MIN_HITS, SORT_IOU_THRESHOLD

# Add the 'sort' directory to the Python path to allow the import
sys.path.append("./sort")
from sort import Sort  # Import the Sort class from sort/sort.py

def initialize_tracker():
    """Initializes the SORT tracker with parameters from the config file."""
    return Sort(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS, iou_threshold=SORT_IOU_THRESHOLD)

def track_objects(tracker, detections):
    """
    Updates the tracker with new detections from the frame.
    :param tracker: The initialized Sort tracker instance.
    :param detections: A numpy array of detections from the YOLO model.
    :return: A numpy array of tracked objects with their assigned IDs.
    """
    if detections.size == 0:
        # Pass an empty array with the correct shape if no detections
        return tracker.update(np.empty((0, 5)))
    return tracker.update(detections)
