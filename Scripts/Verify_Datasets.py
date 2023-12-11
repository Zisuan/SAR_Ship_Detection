import os
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import tifffile

# Function to load the annotations from a given XML file
def load_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = []

    for member in root.findall('object'):
        bndbox = member.find('bndbox')
        annotations.append({
            'label': member.find('name').text,
            'bbox': [
                int(bndbox.find('xmin').text),
                int(bndbox.find('ymin').text),
                int(bndbox.find('xmax').text),
                int(bndbox.find('ymax').text)
            ]
        })
    
    return annotations

# Function to plot the annotations on the image
def plot_annotations(img_path, annotations):
    # Read the image using tifffile
    img = tifffile.imread(img_path)
    
    # Plot the image
    plt.imshow(img)
    
    # Plot the annotations
    for ann in annotations:
        bbox = ann['bbox']
        label = ann['label']
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        plt.text(bbox[0], bbox[1], label, color='red', fontsize=8)
    
    # Show the plot
    plt.show()

# Define the paths to the images and annotations
img_dir = 'combined_dataset/images'  # Replace with the path to your images
xml_dir = 'combined_dataset/annotations'  # Replace with the path to your annotations

# Verify the annotations for each image
for img_file in os.listdir(img_dir):
    if img_file.endswith('.tif'):
        # Corresponding XML file for the image
        if img_file.startswith('hard_neg'):
            xml_file = os.path.join(xml_dir, os.path.splitext(img_file)[0] + '.xml')
            
            # Load the annotations
            annotations = load_annotations(xml_file)
            
            # Plot the annotations on the image
            plot_annotations(os.path.join(img_dir, img_file), annotations)
