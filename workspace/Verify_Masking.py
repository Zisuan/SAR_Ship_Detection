import numpy as np
from PIL import Image

def rle_decode(rle_string, img_shape):
    """
    Decodes an RLE string into a mask image.
    
    Parameters:
    rle_string (str): The string containing the RLE encoded mask.
    img_shape (tuple): The shape of the array to be returned (height, width).

    Returns:
    numpy.ndarray: The decoded mask image.
    """
     # Handle RLE strings separated by commas
    rle_numbers = [int(num) for num in rle_string.replace(' ', '').split(',') if num.isdigit()]
    rle_pairs = np.array(rle_numbers).reshape(-1, 2)
    img = np.zeros(img_shape[0] * img_shape[1], dtype=np.uint8)
    
    for start, length in rle_pairs:
        start_index = start - 1
        end_index = start_index + length
        img[start_index:end_index] = 1
    
    # Reshape and transpose the mask array to match the given image shape
    return img.reshape(img_shape, order='F')

# Example usage:
rle_string = "0, 137, 1, 67, 2, 67, 2, 66, 3, 65, 4, 64, 5, 64, 5, 63, 6, 63, 6, 62, 7, 61, 8, 61, 8, 60, 9, 59, 10, 58, 11, 58, 11, 57, 12, 56, 13, 56, 13, 55, 14, 54, 15, 54, 15, 53, 16, 53, 16, 52, 17, 51, 18, 51, 18, 50, 19, 49, 20, 48, 21, 48, 21, 47, 22, 47, 22, 46, 23, 45, 24, 45, 24, 44, 25, 43, 26, 42, 27, 42, 27, 41, 28, 41, 28, 40, 29, 39, 30, 39, 30, 38, 31, 37, 32, 37, 32, 36, 33, 35, 34, 35, 34, 34, 35, 33, 36, 33, 36, 32, 37, 31, 38, 30, 39, 30, 39, 29, 40, 28, 41, 27, 42, 27, 42, 26, 43, 26, 43, 25, 44, 25, 44, 24, 45, 23, 46, 23, 46, 22, 47, 21, 48, 21, 48, 20, 49, 19, 50, 19, 50, 18, 51, 18, 51, 17, 52, 16, 53, 15, 54, 15, 54, 14, 55, 13, 56, 13, 56, 12, 57, 11, 58, 11, 58, 10, 59, 10, 59, 9, 60, 8, 61, 8, 61, 7, 62, 6, 63, 6, 63, 5, 64, 4, 65, 4, 65, 3, 66, 2, 67, 1, 68, 1, 68"
img_shape = (103,69)  # Replace with your image's dimensions
mask = rle_decode(rle_string, img_shape)

# To save the mask as an image file:
mask_img = Image.fromarray(mask * 255)  # Convert to uint8
mask_img.save('mask.png')
