import sys
import cv2
from ultralytics import YOLO
import pyttsx3
import threading
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Add AdaBins directory to path
sys.path.append('C:\\Python project\\object_detection\\AdaBins')  # Adjust this path to your cloned AdaBins directory

from infer import InferenceHelper

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speaking rate
engine.setProperty('volume', 0.9)  # Adjust volume

# Function to speak text without blocking the main thread
def speak_text(text):
    def speak():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=speak).start()

# Load the YOLOv11 model
model = YOLO("yolo11m.pt")

# Load AdaBins model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
depth_helper = InferenceHelper(dataset='nyu', device=device)

# Set webcam resolution to lower frame size
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Function for depth estimation
def estimate_depth(frame):
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert frame to PIL image
    transform = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_frame = transform(pil_frame).unsqueeze(0).to(device)
    _, depth_map = depth_helper.predict(input_frame)
    depth_map = depth_map.squeeze()  # Remove singleton dimensions
    depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))  # Resize to match the input frame
    return depth_map

# Function to calculate distance and direction of detected objects
def calculate_distance_and_direction(bbox, depth_map, frame_width):
    x1, y1, x2, y2 = map(int, bbox)  # Ensure coordinates are integers
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    distance = np.mean(depth_map[y1:y2, x1:x2])
    
    # Determine direction based on the horizontal position of the center of the bounding box
    if center_x < frame_width * 0.33:
        direction = "left"
    elif center_x > frame_width * 0.66:
        direction = "right"
    else:
        direction = "center"
    
    return distance, direction

frame_skip_rate = 2  # Process every second frame to boost speed
frame_count = 0
last_detected = []  # To track the last detected objects

while True:
    ret, frame = cap.read()
    frame_count += 1
    if not ret or frame_count % frame_skip_rate != 0:
        continue

    # Run YOLOv11 model on the frame
    results = model(frame)

    # Extract the detected objects
    detections = results[0].boxes
    class_ids = detections.cls.cpu().numpy()
    confidences = detections.conf.cpu().numpy()
    names = model.names
    boxes = detections.xyxy.cpu().numpy()

    # Depth estimation
    depth_map = estimate_depth(frame)
    current_detections = []
    
    for i in range(len(class_ids)):
        label = names[int(class_ids[i])]
        confidence = confidences[i]
        bbox = boxes[i]
        distance, direction = calculate_distance_and_direction(bbox, depth_map, frame.shape[1])
        if distance < 0.5:
            voice_warning = f"Warning: {label} very close to you on your {direction}"
            speak_text(voice_warning)
        else:
            current_detections.append((label, confidence, distance, direction))

        # Display text on frame
        text = f"{label} {confidence:.2f} at {distance:.2f}m {direction}"
        cv2.putText(frame, text, (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Check for any new detections
    if current_detections != last_detected:
        last_detected = current_detections
        for detection in current_detections:
            label, confidence, distance, direction = detection
            voice = f"{label} detected with {confidence:.2f} confidence, approximately {distance:.2f} meters to your {direction}"
            print(voice)
            speak_text(voice)

    # Create a visual representation of the depth map
    depth_visual = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_visual = cv2.applyColorMap(depth_visual, cv2.COLORMAP_MAGMA)
    combined_frame = cv2.addWeighted(frame, 0.6, depth_visual, 0.4, 0)

    # Annotate the frame with detection results
    annotated_frame = results[0].plot()
    # Overlay the depth map on the annotated frame
    combined_frame = cv2.addWeighted(annotated_frame, 0.6, depth_visual, 0.4, 0)

    # Display the combined frame
    cv2.imshow('Live Object Detection and Depth Estimation', combined_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
