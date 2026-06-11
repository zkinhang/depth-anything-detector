# =============================================================================
# Block-Based Surface Detection Algorithm
# This is the final implemented approach for obstacle and close-surface detection.
# 1. Estimates depth on stationary images and normalizes the depth map.
# 2. Masks out excessively close/white spots to reduce noise.
# 3. Divides the depth image into uniform regions (blocks).
# 4. Computes mean depth intensity per block.
# 5. Algorithmically links neighboring blocks with similar high intensities.
# 6. Identifies if the linked blocks form a significantly large contiguous "close surface".
# =============================================================================
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import os

# Get all image paths from the folder PIC
image_folder = "src/PIC"
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

# Load images
images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
titles = [os.path.basename(img_path).split('.')[0] for img_path in image_paths]

image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")

for img, title in zip(images, titles):
    # prepare image for the model
    inputs = image_processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # interpolate to original size and visualize the prediction
    post_processed_output = image_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(img.height, img.width)],
    )

    predicted_depth = post_processed_output[0]["predicted_depth"]
    depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
    depth = depth.detach().cpu().numpy() * 255
    depth = Image.fromarray(depth.astype("uint8"))

    # Convert PIL image to numpy array
    depth_np = np.array(depth)
    # Copy the original depth image for compare
    org_np = np.copy(depth_np)

    # Mask the object in front to 0
    class3 = (depth_np >= 100) & (depth_np <= 255)
    high_value_indices = np.where(class3)
    depth_np[high_value_indices] = 0
    
    # Divide the image into equal regions
    height, width = depth_np.shape
    block_size = min(height, width) // 13  # Adjust the divisor to change the number of blocks
    num_blocks_x = width // block_size
    num_blocks_y = height // block_size
    
    blocks = []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = depth_np[y:y + block_size, x:x + block_size]
            blocks.append(block)

    # Calculate the mean value of grey scale for each block
    mean_values = [np.mean(block) for block in blocks]

    # Print the mean values in row and column order, in formatted manner
    num_blocks_x = width // block_size + 1 # padding/trunc
    num_blocks_y = height // block_size + 1
    for row in range(num_blocks_y):
        row_values = []
        for col in range(num_blocks_x):
            row_values.append(f"{mean_values[row * num_blocks_x + col]:6.2f}")
        print(" | ".join(row_values))
        
    # Draw lines around each block
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            cv2.rectangle(depth_np, (x, y), (x + block_size, y + block_size), (255, 255, 255), 1)

    # Check if 4 or above neighboring blocks have similar mean values and are larger than a threshold
    tensity_threshold = 70
    surface_size_threshold = 4
    # Find linked blocks, put the number of linked blocks and the indices of linked blocks inside
    linked_blocks = []

    # check all neighbors of a block (left, right, top, bottom) and return the indices of qualified (intensity > threshold & is similar) neighbors
    def check_neighbors(index, mean_values, checked_indices, threshold=60):
        neighbors = []
        if index % num_blocks_x != 0:  # left neighbor
            neighbors.append(index - 1)
        if (index + 1) % num_blocks_x != 0:  # right neighbor
            neighbors.append(index + 1)
        if index >= num_blocks_x:  # top neighbor
            neighbors.append(index - num_blocks_x)
        if index < len(mean_values) - num_blocks_x:  # bottom neighbor
            neighbors.append(index + num_blocks_x)
        
        qualified_neighbors = [i for i in neighbors if abs(mean_values[i] - mean_values[index]) < 20 and mean_values[i] > threshold and i not in checked_indices]
        return qualified_neighbors

    # check all blocks, skip the blocks that have been checked (include check neighbor function)
    checked_indices = set()
    for i in range(len(mean_values)):
        if i in checked_indices:
            continue
        #  check the neighbour, if the neighbour is qualified, check the further neighbours, until no more qualified neighbours (in stack)
        if mean_values[i] > tensity_threshold:
            stack = [i]
            qualified_neighbors = []
            while stack:
                current = stack.pop()
                if current not in checked_indices:
                    checked_indices.add(current)
                    qualified_neighbors.append(current)
                    further_neighbors = check_neighbors(current, mean_values, checked_indices)
                    stack.extend(further_neighbors)
            if len(qualified_neighbors) >= surface_size_threshold:
                linked_blocks.append((len(qualified_neighbors), qualified_neighbors))

    # Highlight linked blocks in the image by setting them to 255
    for _, block_indices in linked_blocks:
        for index in block_indices:
            y = (index // num_blocks_x) * block_size
            x = (index % num_blocks_x) * block_size
            depth_np[y:y + block_size, x:x + block_size] = 255

    print("Number of Linked Blocks:", len(linked_blocks), "\nregion size:", end='')
    for size, block_indices in linked_blocks:
        print(size, end=' ')
    print()

    # if there are only one linked region, check if it has close surface
    have_close_surface = False
    # Print the coordinates of linked blocks
    for _, block_indices in linked_blocks:
        coordinates = [(index % num_blocks_x, index // num_blocks_x) for index in block_indices]
        print("Linked Block Coordinates:", coordinates)
        if len(linked_blocks) == 1:
            have_close_surface = True
        else:
            # check if the linked blocks (more than one) is actually linked together
            all_linked = True
            for i in range(len(linked_blocks) - 1):
                for j in range(i + 1, len(linked_blocks)):
                    if not any(abs(a % num_blocks_x - b % num_blocks_x) <= 1 and abs(a // num_blocks_x - b // num_blocks_x) <= 1 for a in linked_blocks[i][1] for b in linked_blocks[j][1]):
                        all_linked = False
                        break
                if not all_linked:
                    break
            if all_linked:
                have_close_surface = True
                
    print("Object too close:", have_close_surface)

    highlighted_percentage = np.sum(depth_np == 255) / depth_np.size * 100
    print(f"Highlighted Region Percentage: {highlighted_percentage:.2f}%")

    plt.imshow(org_np, cmap='gray')
    plt.title(f'{title} - Depth Image')
    plt.axis('off')
    plt.show()

    plt.imshow(depth_np, cmap='gray')
    plt.title(f'{title} - Depth Image')
    plt.axis('off')
    plt.show()