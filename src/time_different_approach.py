"""
Trial implementation: Temporal Depth Brightness Difference
This script uses a real-time webcam feed to track the average brightness 
(depth intensity) change over time. It compares the current frame's overall 
depth to a reference frame captured at startup. A significant increase in 
brightness indicates an approaching object.
Note: This is an earlier trial version.
"""
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import time

cap = cv2.VideoCapture(0)

image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")

start_time = time.time()
# first loop capture the first frame (when nothing is approaching to the camera)
first_loop = True
prev_depth_np = None

# first frame capture time
capture_time = 10
# comparison time gap
compare_time = 3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # prepare image for the model
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # interpolate to original size and visualize the prediction
    post_processed_output = image_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.height, image.width)],
    )

    predicted_depth = post_processed_output[0]["predicted_depth"]
    depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
    depth = depth.detach().cpu().numpy() * 255
    depth = Image.fromarray(depth.astype("uint8"))

    # Convert PIL image to numpy array
    depth_np = np.array(depth)

    threshold = 100  # The threshold value detecting the brightness increased

    # Capture the first frame after t=capture_time as prev_depth_np
    if first_loop and time.time() - start_time > capture_time:
        prev_depth_np = depth_np
        first_loop = False
        start_time = time.time()
        continue
    # Compare the current frame with the previous frame every three second
    elif not first_loop:
        if time.time() - start_time > compare_time:
            if prev_depth_np is not None:
                current_depth_np = depth_np
                diff = current_depth_np - prev_depth_np
                print(np.mean(diff))
                if np.mean(diff) > threshold:
                    print("yes")
                else:
                    print("no")
                start_time = time.time()
        
    
    cv2.imshow('Depth Image', depth_np)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # # Check if there exist class2 numbers
    # if np.any(class2):
    #     print("yes")
    # else:
    #     print("np")
    # # Display the image using OpenCV
    # cv2.imshow('Depth Image', highlighted_depth)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()