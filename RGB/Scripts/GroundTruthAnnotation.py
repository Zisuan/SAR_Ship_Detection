import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    boxes = []
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        if bndbox is not None:
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            boxes.append((xmin, ymin, xmax, ymax))
        
        robndbox = obj.find('robndbox')
        if robndbox is not None:
            # Additional code to handle the rotation can be added here
            pass
            
    return boxes

def plot_annotations(image_dir, annotation_dir):
    for filename in os.listdir(annotation_dir):
        if filename.endswith('.xml'):
            xml_path = os.path.join(annotation_dir, filename)
            img_path = os.path.join(image_dir, filename.replace('.xml', '.png'))
            
            boxes = parse_xml(xml_path)
            
            # Load image
            image = Image.open(img_path)
            fig, ax = plt.subplots(1)
            ax.imshow(image)
            
            # Plot bounding boxes
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                         linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            
            plt.show()

def plot_annotations_for_single_image(img_path, xml_path):
    boxes = parse_xml(xml_path)
            
    # Load image
    image = Image.open(img_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    # Plot bounding boxes
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    plt.show()


# Replace with your own Directory Path or File Path, use accordingly for single image or a folder of images.
# Directory paths
#IMAGE_DIR = "A_Dual-polarimetric_SAR_Ship_Detection_Dataset-main/PNGImages/train"
#ANNOTATION_DIR = "A_Dual-polarimetric_SAR_Ship_Detection_Dataset-main/annotations/HorizontalBox/train"

IMAGE_PATH = "A_Dual-polarimetric_SAR_Ship_Detection_Dataset-main/PNGImages/test/000495.png"
XML_PATH = "A_Dual-polarimetric_SAR_Ship_Detection_Dataset-main/annotations/HorizontalBox/test/000495.xml"

# Plotting annotations on the images
#plot_annotations(IMAGE_DIR, ANNOTATION_DIR)
plot_annotations_for_single_image(IMAGE_PATH, XML_PATH)