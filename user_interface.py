import cv2

class UserInterface:
    def __init__(self):
        # Initialize user interface (e.g., create windows, buttons, sliders)
        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)

    def display_frame(self, frame):
        # Display frame on user interface
        cv2.imshow("Object Detection", frame)
        cv2.waitKey(1)

    def check_user_input(self):
        # Check for user input and update configuration if needed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        return True

def draw_results(frame, tracked_objects):
    # Draw bounding boxes around tracked objects on frame
    for bbox in tracked_objects:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame
