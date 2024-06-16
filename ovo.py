from PIL import Image

# Define the dimensions of the image
width = 1680
height = 1120

# Create a new black image
image = Image.new("RGB", (width, height), color="white")

# Save the image

image.save("/home/oscar/Desktop/0301_dpdd_seg/dpdd_seg_mask_copy/1P0A2510.png")
