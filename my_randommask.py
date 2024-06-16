import cv2
import numpy as np
import os

# Function to generate random masks
def generate_random_mask(size=(256, 256), num_masks=1000):
    masks = []
    for _ in range(num_masks):
        mask = np.zeros(size, dtype=np.uint8)
        shape = np.random.choice(['circle', 'rectangle'])
        if shape == 'circle':
            radius = np.random.randint(30, 80)
            center = (np.random.randint(radius, size[1] - radius),
                      np.random.randint(radius, size[0] - radius))
            cv2.circle(mask, center, radius, (255), -1)
        else:  # rectangle
            x1, y1 = np.random.randint(0, size[1] - 150), np.random.randint(0, size[0] - 150)
            x2, y2 = x1 + np.random.randint(40, 150), y1 + np.random.randint(40, 150)
            cv2.rectangle(mask, (x1, y1), (x2, y2), (255), -1)
        masks.append(mask)
    return masks

# Create directory to save masks
os.makedirs("random_masks", exist_ok=True)

# Generate masks
masks = generate_random_mask()

# Save masks
for i, mask in enumerate(masks):
    cv2.imwrite(f"random_masks/mask_{i}.png", mask)
