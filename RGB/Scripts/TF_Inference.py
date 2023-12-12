import io
import os
import scipy.misc
import numpy as np
import six
import time
import glob
from IPython.display import display
import random
from six import BytesIO

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import config_util
from object_detection.builders import model_builder


def load_image_into_numpy_array(path):
   
    image = Image.open(path)

    image = image.resize((512, 512), Image.LANCZOS)
    
    return np.array(image)

category_index = label_map_util.create_category_index_from_labelmap('workspace/training_demo/ship_label_map.pbtxt', use_display_name=True) # Replace with your label map path

tf.keras.backend.clear_session()
detect_fn = tf.saved_model.load('workspace/training_demo/exported_models/ship_model/saved_model') # Replace with your trained model path

IMAGE_DIR = 'A_Dual-polarimetric_SAR_Ship_Detection_Dataset-main/PNGImages/test/' # Replace with your dataset path

# Get all image paths from the directory
all_image_paths = glob.glob(os.path.join(IMAGE_DIR, '*.png'))

# Number of random images you want to process
num_random_images = 10  # Change this value based on your requirement

# Select random images
random_image_paths = random.sample(all_image_paths, num_random_images)

for image_path in random_image_paths:

    print('Running inference for {}... '.format(image_path), end='')

    image_np = load_image_into_numpy_array(image_path)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}

    print("Detection Scores:", detections['detection_scores'])

    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    '''
    image_np_with_detections = image_np.copy()

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0,
        agnostic_mode=False,
        line_thickness=1  
        )

    plt.figure()
    plt.imshow(image_np_with_detections)
    print('Done')
    plt.show()
    '''
    # Create a PIL image from the numpy array
    original_image_pil = Image.fromarray(image_np)
    detected_image_pil = original_image_pil.copy()
    draw = ImageDraw.Draw(detected_image_pil)
    font = ImageFont.truetype("arial.ttf", 15)  # Choose a font and size
    GAP_WIDTH = 20
    # Iterate over the detections and add annotations
    for box, score, class_id in zip(detections['detection_boxes'], detections['detection_scores'], detections['detection_classes']):
        if score > 0.5:  # Adjust the threshold as needed
            ymin, xmin, ymax, xmax = box
            (left, right, top, bottom) = (xmin * image_np.shape[1], xmax * image_np.shape[1], ymin * image_np.shape[0], ymax * image_np.shape[0])
            class_name = category_index[class_id]['name']
            label = '{}: {:.2f}%'.format(class_name, score*100)

            # Draw bounding box and label
            draw.rectangle([(left, top), (right, bottom)], outline='lime', width=2)
            draw.text((left, top - 20), label, fill='lime', font=font)

    # To display the original and detected images side by side
    combined_width = original_image_pil.width + detected_image_pil.width + GAP_WIDTH
    combined_height = max(original_image_pil.height, detected_image_pil.height)

    combined_image = Image.new('RGB', (combined_width, combined_height), color='white')
    combined_image.paste(original_image_pil, (0, 0))
    combined_image.paste(detected_image_pil, (original_image_pil.width + GAP_WIDTH, 0))
    combined_image.show()
    
'''
# Create an output directory if it doesn't exist
output_dir = 'output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# save the image using PIL's save function
im = Image.fromarray(image_np_with_detections)
image_name = os.path.basename(image_path)
im.save(os.path.join(output_dir, f'annotated_{image_name}'))

# you could also print out a message saying the file has been saved
print(f"Annotated image saved as 'annotated_{image_name}'")
'''