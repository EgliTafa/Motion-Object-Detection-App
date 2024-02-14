import cv2

class Tracker:
    def __init__(self):
        # Initialize tracker parameters and variables
        self.tracker = cv2.MultiTracker_create()

    def update(self, frame, detections):
        # Update tracker with new detections
        tracked_objects = []

        # Convert detections to bounding boxes
        for detection in detections:
            bbox = detection_to_bbox(detection)
            tracked_objects.append(bbox)

        # Update tracker with new bounding boxes
        success, boxes = self.tracker.update(frame)

        if success:
            # Convert boxes to detections format
            tracked_objects = []
            for box in boxes:
                tracked_objects.append(bbox_to_detection(box))

        return tracked_objects

def detection_to_bbox(detection):
    # Convert detection to bounding box format (x, y, w, h)
    # detection format: [xmin, ymin, xmax, ymax]
    xmin, ymin, xmax, ymax = detection[:4]
    bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
    return bbox

def bbox_to_detection(bbox):
    # Convert bounding box to detection format (xmin, ymin, xmax, ymax)
    # bbox format: (x, y, w, h)
    x, y, w, h = bbox
    detection = [x, y, x + w, y + h]
    return detection
