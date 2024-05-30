import os
import shutil

original_img_dir = 'datasets/custom_images/images'
output_img_dirs = 'datasets/custom_images/raw'

for filename in os.listdir(original_img_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # change to .ipg if that's actually the case
        # Create a new directory path for this image
        new_dir = os.path.join(output_img_dirs, filename[:-4])  # Remove the extension from the filename

        # Check if the directory already exists, if not, create it
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        # Define the current path of the file and the new path where it will be moved
        current_file = os.path.join(original_img_dir, filename)
        new_file_path = os.path.join(new_dir, filename)

        # Move the file
        shutil.copy(current_file, new_file_path)
        print(f"Copy '{filename}' to '{new_dir}'")