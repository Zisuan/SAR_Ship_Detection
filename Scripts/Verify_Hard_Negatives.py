import cv2
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import tifffile
from matplotlib import pyplot as plt

# Define the directories where your images and XML files are stored
inference_results_dir = 'inference_results'
hard_negatives_folder = 'hard_negatives'

def resize_with_aspect_ratio(image, target_height, inter=cv2.INTER_AREA):
    # Calculate the aspect ratio and determine new dimensions
    (h, w) = image.shape[:2]
    ratio = target_height / float(h)
    dim = (int(w * ratio), target_height)
    # Resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# Loop through the inference images and display them alongside the hard negatives
for inference_image_name in os.listdir(inference_results_dir):
    if inference_image_name.endswith('_comparison.png'):
        # Read the inference image
        inference_image_path = os.path.join(inference_results_dir, inference_image_name)
        inference_image = cv2.cvtColor(cv2.imread(inference_image_path), cv2.COLOR_BGR2RGB)

        mid_point = inference_image.shape[1] // 2
        left_inference_image = inference_image[:, :mid_point]
        right_inference_image = inference_image[:, mid_point:]
        
        # Find the corresponding XML file in hard_negatives_folder
        xml_filename = inference_image_name.replace('_comparison.png', '.xml')
        xml_file_path = os.path.join(hard_negatives_folder, xml_filename)

        if not os.path.exists(xml_file_path):
            print(f"XML file {xml_filename} does not exist. Skipping.")
            continue

        # Parse the XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # Construct the image filename from the XML filename
        hard_negative_image_name = xml_filename.replace('.xml', '.tif')
        hard_negative_image_path = os.path.join(hard_negatives_folder, hard_negative_image_name)

        # Read the hard negative image
        hard_negative_image = tifffile.imread(hard_negative_image_path)
        hard_negative_image = cv2.cvtColor(hard_negative_image, cv2.COLOR_BGR2RGB) if hard_negative_image.ndim == 3 else hard_negative_image
        
        # Determine the target height based on the hard negative image
        target_height = hard_negative_image.shape[0]

        # Resize the inference images to the target height
        resized_left_inference_image = resize_with_aspect_ratio(left_inference_image, target_height)
        resized_right_inference_image = resize_with_aspect_ratio(right_inference_image, target_height)


        # Draw the bounding boxes from the XML on the hard negative image
        for object_element in root.findall('object'):
            bndbox = object_element.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # Draw a rectangle on the hard negative image
            cv2.rectangle(hard_negative_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Plot both images side by side
        plt.figure(figsize=(12,8))  # Adjust the figure size as needed

         # Display left inference image
        plt.subplot(1, 3, 1)
        plt.imshow(resized_left_inference_image)
        plt.title('Inference Result')
        plt.axis('off')

        # Display right inference image
        plt.subplot(1, 3, 2)
        plt.imshow(resized_right_inference_image)
        plt.title('Ground truth Result')
        plt.axis('off')

        # Display hard negative image
        plt.subplot(1, 3, 3)
        plt.imshow(hard_negative_image)
        plt.title('Hard Negative with FP Boxes')
        plt.axis('off')

        plt.tight_layout()
        plt.show()
        plt.close()
