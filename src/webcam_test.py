"""
Trial implementation: Webcam Connected
This script uses a real-time webcam feed to generate a depth map using 
the Depth-Anything-V2-Small model. It thresholds the predicted depth values 
and uses OpenCV connected components to locate and bound the largest close object.
Note: This is an earlier trial version.
"""
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
# import requests
import cv2
from matplotlib import pyplot as plt

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
cap = cv2.VideoCapture(0)

# Get the width and height of the frames from the video capture
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
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

    # Classify different grey levels into three classes
    class1 = (depth_np >= 0) & (depth_np <= 99)
    class2 = (depth_np >= 100) & (depth_np <= 150)
    class3 = (depth_np >= 151) & (depth_np <= 255)

    # Display the image using OpenCV
    # cv2.imshow('Depth Image', depth_np)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Display the image using matplotlib
    # plt.imshow(depth_np, cmap='gray')
    # plt.title('Depth Image')
    # plt.axis('off')
    # plt.show()
    # Find the indices where the depth values are in the class3 range
    high_value_indices = np.where(class3)
    detect_value_indices = np.where(class2)
    far_value_indices = np.where(class1)

    # print("High value indices:", high_value_indices)
    # Optionally, you can highlight these areas in the image
    highlighted_depth = depth_np.copy()
    # highlighted_depth[high_value_indices] = 0  # Set high values to white for visualization
    # highlighted_depth[detect_value_indices] = 255
    # highlighted_depth[far_value_indices] = 0

    # # Set the third class to 255 in the grey scale picture
    # depth_np[class3] = 255
    # Find the region with the highest values in the grey scale picture
    num_labels, labels_im = cv2.connectedComponents(class3.astype(np.uint8))

    # Initialize variables to store the largest component
    max_size = 0
    max_label = 0

    # Iterate through all components to find the largest one
    for label in range(1, num_labels):
        size = np.sum(labels_im == label)
        if size > max_size:
            max_size = size
            max_label = label

    # Create a mask for the largest component
    largest_component_mask = (labels_im == max_label).astype(np.uint8)

    # Find the bounding box of the largest component
    x, y, w, h = cv2.boundingRect(largest_component_mask)

    # Increase the size of the rectangle to ensure the region is totally included
    padding = 30  # Increase the padding value to cover the whole region
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2 * padding, highlighted_depth.shape[1] - x)
    h = min(h + 2 * padding, highlighted_depth.shape[0] - y)

    # Draw a rectangle around the largest component
    cv2.rectangle(highlighted_depth, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Turn the area covered by the rectangle to 0
    highlighted_depth[y:y+h, x:x+w] = 0

    # Display the image with the rectangle
    cv2.imshow('Highlighted Depth Image', highlighted_depth)

    # Print the numpy array indicating the location of the rectangle
    rectangle_location = np.array([x, y, w, h])
    print("Rectangle location:", rectangle_location)
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