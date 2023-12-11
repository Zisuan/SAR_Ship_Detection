import os
import numpy as np
import tifffile as tiff

def calculate_channel_statistics(img_dir):
    pixel_sum = np.array([0.0, 0.0, 0.0, 0.0])
    pixel_sum_squared = np.array([0.0, 0.0, 0.0, 0.0])
    num_pixels = 0

    for filename in os.listdir(img_dir):
        if not filename.endswith('.tif'):
            continue

        image_path = os.path.join(img_dir, filename)
        image = tiff.imread(image_path)
        
        # Accumulate the sum and sum of squares
        pixel_sum += image.sum(axis=(0, 1))
        pixel_sum_squared += (image ** 2).sum(axis=(0, 1))
        
        num_pixels += image.shape[0] * image.shape[1]

    # Compute the mean and std dev
    mean = pixel_sum / num_pixels
    std_dev = np.sqrt((pixel_sum_squared / num_pixels) - (mean ** 2))

    return mean, std_dev

# Replace 'train' with the correct subdirectory or path to your image directory
mean, std_dev = calculate_channel_statistics('originalTIF/train')
print(f"Mean: {mean}")
print(f"Standard Deviation: {std_dev}")
