import time
import collections

# Create a deque (a list-like container with fast appends and pops from either end)
# with a maximum length of 30. This will store the FPS of the last 30 frames.
fps_queue = collections.deque(maxlen=30)

def calculate_fps(start_time):
    """
    Calculates FPS using a moving average over the last 30 frames.
    This provides a much smoother and more stable FPS reading than calculating
    it for every single frame.
    """
    # Calculate the instantaneous FPS for the current frame
    end_time = time.time()
    instant_fps = 1.0 / (end_time - start_time)
    
    # Add the new FPS value to our queue
    fps_queue.append(instant_fps)
    
    # Calculate the average of all values in the queue
    average_fps = sum(fps_queue) / len(fps_queue)
    
    return average_fps
