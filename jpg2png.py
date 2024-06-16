import os
from PIL import Image

def convert_jpg_to_png(input_path, output_path):
    try:
        # Open the JPG image
        with Image.open(input_path) as img:
            # Check if the image is in RGB mode, convert if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Save it as a PNG image
            img.save(output_path)
            print(f"Image successfully converted to {output_path}")
    except Exception as e:
        print(f"An error occurred while converting {input_path}: {e}")

def convert_all_jpgs_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.bmp'):
            # Construct full file path
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join("/home/oscar/Desktop/dataset/SG_dataset/dataset/train_data/FCFB/FBB/FB_png", f"{os.path.splitext(filename)[0]}.png")
            
            # Convert the file
            convert_jpg_to_png(input_path, output_path)

# # Example usage
folder_path = '/home/oscar/Desktop/dataset/SG_dataset/dataset/train_data/FCFB/FBB/FB/'  # Replace with your folder path
convert_all_jpgs_in_folder(folder_path)


def resize_and_convert_image(input_path, output_path, size=(1680, 1120)):
    try:
        # Open the image
        with Image.open(input_path) as img:
            # Resize the image
            img = img.resize(size, Image.Resampling.LANCZOS)
            
            # Save it as a PNG image
            img.save(output_path)
            print(f"Image successfully resized and converted to {output_path}")
    except Exception as e:
        print(f"An error occurred while processing {input_path}: {e}")

def process_all_images_in_folder(folder_path, size=(1680, 1120)):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            # Construct full file path
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}.png")
            
            # Resize and convert the file
            resize_and_convert_image(input_path, output_path, size)

# Example usage
# folder_path = '/home/oscar/Desktop/EBB!/Training/original _1680'  # Replace with your folder path
# process_all_images_in_folder(folder_path)
