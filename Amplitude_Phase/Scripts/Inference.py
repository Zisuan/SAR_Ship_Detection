from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog , DatasetCatalog
from detectron2.structures import Instances , Boxes
import tifffile
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
import xml.etree.ElementTree as ET
import torch


def load_mask_from_png(mask_file_path):
    mask_image = Image.open(mask_file_path)
    mask_array = np.array(mask_image)
    if len(mask_array.shape) == 3:  # If it's a color image
        mask_array = np.any(mask_array > 0, axis=-1)  # Check any channel has non-zero values
    else:
        mask_array = mask_array > 0  # If it's a grayscale image, check for non-zero values

    return mask_array.astype(bool)
    
# Function to get the mask file path from the XML annotation
def get_mask_files_from_xml(xml_file, image_name, masks_folder_path):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    mask_files = []

    for obj in root.iter('object'):
        mask_tag = obj.find('segm/mask')
        if mask_tag is not None:
            mask_file = mask_tag.text
            if image_name in mask_file:
                # Include the correct path by combining the masks_folder_path with the mask file name
                mask_file_path = os.path.normpath(os.path.join(masks_folder_path, mask_file))
                mask_files.append(mask_file_path)

    return mask_files

def modify_model_conv1(model):
    # Get the original first convolutional layer from the pre-trained model
    old_conv1 = model.backbone.bottom_up.stem.conv1

    # Create a new Conv2d layer with 4 input channels and the same output channels, kernel size, etc.
    new_conv1 = torch.nn.Conv2d(
        in_channels=4,  # Change the number of input channels to 4
        out_channels=old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        bias=(old_conv1.bias is not None)
    )
  
    # Replace the first conv layer with the new one
    model.backbone.bottom_up.stem.conv1 = new_conv1

    return model

# Load the config
cfg = get_cfg()
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.merge_from_file("config.yaml")  # Path to the config file used for training
cfg.MODEL.WEIGHTS = "model2/model_0099999.pth"  # Path to the trained model weights
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # Update to match the number of classes in your dataset
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8 # Set the testing threshold for this model
cfg.MODEL.PIXEL_MEAN = [0.00941263, -0.00114559, 0.00115459, 0.10023721]
cfg.MODEL.PIXEL_STD = [1.81057896, 1.49695959, 2.00576686, 7.44893589]

# Build the model
model = build_model(cfg)
model = modify_model_conv1(model)  # Modify the model

# Load weights
checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)

# Move the model to GPU if available
if torch.cuda.is_available():
    model.to("cuda")
model.eval()

# Register the metadata
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
metadata.set(thing_classes=["ship"])

'''
## For hard negative data mining
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

# Function to visualize the mask on the image

def visualize_mask_on_image(image, mask, color=[0, 255, 0], alpha=0.8):
 
    # Ensure the mask is boolean
    mask = mask.astype(bool)
    
    # Create an RGB overlay with the same spatial dimensions as the original image
    overlay = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Set the color on the overlay where the mask is true
    for i in range(3):  # Set color for each RGB channel
        overlay[:, :, i][mask] = color[i]

    # Extract the RGB channels from the 4-channel image
    image_rgb = image[:, :, :3].astype(np.uint8)

    # Combine the RGB overlay with the RGB channels of the original image using the specified alpha
    # Ensure both are of the same data type
    combined = cv2.addWeighted(overlay.astype(np.uint8), alpha, image_rgb, 1 - alpha, 0)

    # Concatenating to maintain the 4th channel as it is 
    if image.shape[2] == 4:
        combined = np.concatenate((combined, image[:, :, 3:4]), axis=2)

    return combined

# Path to the inference folder and corresponding PNG folder
tif_inference_folder = "originalTIF/For_Inference"
png_inference_folder = "PNGImages/For_Inference"

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

xml_directory = "Masking_Annotation/default"
masks_folder_path = "Masking_Annotation/default/Masks"

# Process each TIF image in the folder
for tif_filename in os.listdir(tif_inference_folder):
    if tif_filename.endswith('.tif'):
        # Load the TIF image using tifffile
        tif_image_path = os.path.join(tif_inference_folder, tif_filename)
        tif_image = tifffile.imread(tif_image_path)
        tif_image_tensor = torch.as_tensor(tif_image.astype("float32")).permute(2, 0, 1)
        # Move to CUDA if available
        if torch.cuda.is_available():
            tif_image_tensor = tif_image_tensor.to("cuda")

        # Inference
        with torch.no_grad():
            outputs = model([{"image": tif_image_tensor}])[0]
        
         # Extract prediction results
        instances = outputs["instances"].to("cpu")
        pred_boxes = instances.pred_boxes if instances.has("pred_boxes") else None
        # If there are no predicted boxes, continue to the next image
        if pred_boxes is None:
            print("No predictions found for image:", tif_filename)
            continue
        
        # Get pred_boxes, pred_classes, and scores from the model output
        instances = outputs["instances"].to("cpu")
        pred_boxes = instances.pred_boxes if instances.has("pred_boxes") else None
        scores = instances.scores if instances.has("scores") else None
        pred_classes = instances.pred_classes if instances.has("pred_classes") else None

        # Filter predictions to get only ships
        ship_indices = [i for i, pc in enumerate(pred_classes) if pc == 0]
        pred_boxes = pred_boxes[ship_indices]
        scores = scores[ship_indices]
        pred_classes = pred_classes[ship_indices]

        # Get the mask file paths from XML and load the masks
        image_name_without_extension = os.path.splitext(tif_filename)[0]
        xml_file = os.path.join(xml_directory, image_name_without_extension + '.xml')
        mask_file_paths = get_mask_files_from_xml(xml_file, image_name_without_extension, masks_folder_path)
        combined_mask = np.zeros(tif_image.shape[:2], dtype=bool)  # Initialize combined mask with False
        for mask_file_path in mask_file_paths:
            mask = load_mask_from_png(mask_file_path)
            combined_mask = np.logical_or(combined_mask, mask)

        # Combined mask to apply on image 
        if combined_mask is not None:
            # Filter out detections that overlap with the masked area
            keep_indices = []
            for i, box in enumerate(pred_boxes):
                box_np = box.cpu().numpy().astype(int)
                x1, y1, x2, y2 = box_np
                # Calculate the overlap with the mask
                overlap = combined_mask[y1:y2, x1:x2].mean()
                if overlap < 0.1:
                    keep_indices.append(i)

            # If there are detections that pass the mask filter, visualize and process them
            if keep_indices:
                filtered_instances = instances[keep_indices]
                masked_image = visualize_mask_on_image(tif_image[:, :, :3], combined_mask)
                # Create a Visualizer instance using the original TIF image
                v = Visualizer(masked_image, metadata=metadata, scale=1.0)
                out = v.draw_instance_predictions(filtered_instances)
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
                ## For hard negative mining
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

                plot_filename = os.path.join(plot_folder, tif_filename.replace('.tif', '_comparison.png'))
                plt.savefig(plot_filename, bbox_inches='tight')
                plt.close()
                
                plt.tight_layout()
                plt.show()
                