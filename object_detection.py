import cv2


def load_model():
    # Load pre-trained object detection model
    model_config = '../Pre-Trained Model/object_detection_nanodet_2022nov.onnx'
    model_weights = '../Pre-Trained Model/object_detection_nanodet_2022nov.onnx'
    model = cv2.dnn.readNet(model_config, model_weights)
    return model


def detect_objects(frame, model):
    # Resize frame to fit the model's input size (if necessary)
    input_blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)

    # Set input to the model
    model.setInput(input_blob)

    # Forward pass through the model to perform object detection
    detections = model.forward()

    return detections
