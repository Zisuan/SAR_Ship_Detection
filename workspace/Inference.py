from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog , DatasetCatalog
import tifffile
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os
import cv2
import xml.etree.ElementTree as ET
import torch

# Load the config
cfg = get_cfg()
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.merge_from_file("config.yaml")  # Path to the config file used for training
cfg.MODEL.WEIGHTS = "model2/model_0099999.pth"  # Path to the trained model weights
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # Update to match the number of classes in your dataset
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9 # Set the testing threshold for this model

class_names = ["ship"]#,"non_ship"]
thing_classes = class_names  # This should be a list of strings, representing each class

# Register the metadata
MetadataCatalog.get("sar_ships_test").set(thing_classes=thing_classes)

metadata = MetadataCatalog.get("sar_ships_test")

# Create the predictor using the DefaultPredictor class
predictor = DefaultPredictor(cfg)

def calculate_iou(box1, box2):
  
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # If there's no overlap, return 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The area of the intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # The area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the IoU by dividing the intersection area by the union area
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou
'''
def create_xml_file(filename, boxes, folder):
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = filename
    ET.SubElement(root, "folder").text = folder

    for box in boxes:
        object_tag = ET.SubElement(root, "object")
        ET.SubElement(object_tag, "name").text = "non_ship"  # Or "negative" or any other label you wish to use for false positives
        ET.SubElement(object_tag, "pose").text = "Unspecified"
        ET.SubElement(object_tag, "truncated").text = "0"
        ET.SubElement(object_tag, "difficult").text = "0"
        bndbox_tag = ET.SubElement(object_tag, "bndbox")
        ET.SubElement(bndbox_tag, "xmin").text = str(int(box[0]))
        ET.SubElement(bndbox_tag, "ymin").text = str(int(box[1]))
        ET.SubElement(bndbox_tag, "xmax").text = str(int(box[2]))
        ET.SubElement(bndbox_tag, "ymax").text = str(int(box[3]))

    # Write the XML file
    tree = ET.ElementTree(root)
    tree.write(os.path.join(hard_negatives_folder, f"{filename}.xml"))
'''

# Path to the inference folder and corresponding PNG folder
tif_inference_folder = "originalTIF/For_Inference"
png_inference_folder = "PNGImages/test"

# Path to the folder containing XML annotation files
annotation_folder = "annotations/HorizontalBox/test"

# Define the desired display size
display_size = (256,256)  # You can adjust this to your preferred display size
'''
# Path to hard negatives mining
hard_negatives_folder = 'hard_negatives'
if not os.path.exists(hard_negatives_folder):
    os.makedirs(hard_negatives_folder)
'''

# Path to results folder
plot_folder = "inference_results_2"
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)


# Process each TIF image in the folder
for tif_filename in os.listdir(tif_inference_folder):
    if tif_filename.endswith('.tif'):
        # Load the TIF image using tifffile
        tif_image_path = os.path.join(tif_inference_folder, tif_filename)
        tif_image = tifffile.imread(tif_image_path)
        if tif_image.ndim == 3 and tif_image.shape[-1] == 4:
            outputs = predictor(tif_image)
        else:
            raise ValueError("Image does not have the correct shape.")
      
    
        # Create a Visualizer instance using the original TIF image
        v = Visualizer(tif_image, metadata=metadata, scale=1.0)
        #v._default_font_size = max(tif_image.shape[:2]) // 200  # Set the default font size based on the image dimensions
        '''
        # Get the predicted classes and filter out non-ship detections
        pred_classes = outputs["instances"].pred_classes.cpu().numpy()
        ship_indices = np.where(pred_classes == 0)[0]  # assuming ship class has index 0

        # Filter predictions for visualization
        instances = outputs["instances"][ship_indices]

        # Draw the instance predictions
        out = v.draw_instance_predictions(instances.to("cpu"))
        '''
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # Retrieve the visualized image
        vis_image = out.get_image()

        # Resize the visualized image if necessary
        result_tif_image = cv2.resize(vis_image, display_size, interpolation=cv2.INTER_AREA)
        
        # Load the corresponding PNG image
        png_image_path = tif_image_path.replace('originalTIF/For_Inference', 'PNGImages/test').replace('.tif', '.png')
        png_image = np.array(Image.open(png_image_path))
        png_image = cv2.resize(png_image, display_size, interpolation=cv2.INTER_AREA)

        # Load the corresponding XML annotation file
        xml_filename = tif_filename.replace('originalTIF/For_Inference', 'annotations/HorizontalBox/test').replace('.tif', '.xml')
        xml_path = os.path.join(annotation_folder, xml_filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Initialize a list to store bounding box coordinates
        bounding_boxes = []

        # Extract all bounding box information within the XML
        for object_element in root.findall("object"):
            xmin = int(object_element.find("bndbox").find("xmin").text)
            ymin = int(object_element.find("bndbox").find("ymin").text)
            xmax = int(object_element.find("bndbox").find("xmax").text)
            ymax = int(object_element.find("bndbox").find("ymax").text)
    
            bounding_boxes.append((xmin, ymin, xmax, ymax))
       # Draw all bounding boxes on the PNG image
        for xmin, ymin, xmax, ymax in bounding_boxes:
            cv2.rectangle(png_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)  # Green rectangle for each bounding box
        '''
        false_positives = []

         # Extract the predicted bounding boxes and scores
        predicted_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
  
        
        # Compare each predicted box with the ground truth
        for pred_box in predicted_boxes:
            max_iou = 0.0
            for true_box in bounding_boxes:
                iou = calculate_iou(pred_box, true_box)
                max_iou = max(max_iou, iou)
                #print(f"Comparing predicted box {pred_box} with true box {true_box}, IoU: {iou}")  # Debugging output
            
            # If the max IoU is below a threshold, consider it a false positive
            if max_iou < 0.1:
                #print(f"Appending false positive box {pred_box} with max IoU {max_iou}")  # Debugging output
                false_positives.append(pred_box)
        
        if false_positives:
            create_xml_file(tif_filename.replace('.tif', ''), false_positives, hard_negatives_folder)

            # Save the associated image for the false positives
            false_positive_image_path = os.path.join(hard_negatives_folder, tif_filename)
            cv2.imwrite(false_positive_image_path, tif_image)
        '''
        # Plot both TIF inference result and PNG image side by side
        fig, axes = plt.subplots(1, 2, figsize=(16, 12))  # Adjust the figure size as needed

        axes[0].imshow(result_tif_image)
        axes[0].set_title("Inference on TIF image")
        axes[0].axis('off')

        axes[1].imshow(png_image)
        axes[1].set_title("Corresponding PNG image")
        axes[1].axis('off')

        #plot_filename = os.path.join(plot_folder, tif_filename.replace('.tif', '_comparison.png'))
        #plt.savefig(plot_filename, bbox_inches='tight')
        #plt.close()
        
        plt.tight_layout()
        plt.show()
        