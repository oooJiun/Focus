import os
from PIL import Image, ImageFilter

# Define the folder containing the images
folder_path = "/home/oscar/Desktop/dataset/DPDD/dd_dp_dataset_png/train_c/target"

# Define the output folder where blurred images will be saved
output_folder = "./results_0423/blurred_target/"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define the radius of the median blur
radius = 13

# Function to apply median blur and save the image
def apply_blur_and_save(input_path, output_path):
    # Open the image
    img = Image.open(input_path)
    
    # Apply median blur
    img_blurred = img.filter(ImageFilter.MedianFilter(size=radius))
    
    # Save the blurred image
    img_blurred.save(output_path)

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file is an image (assuming all files in the folder are images)
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".bmp"):
        # Get the full path of the input image
        input_path = os.path.join(folder_path, filename)
        
        # Construct the output path for the blurred image
        output_path = os.path.join(output_folder, filename)
        
        # Apply blur and save the image
        apply_blur_and_save(input_path, output_path)

print("Blurring completed.")
