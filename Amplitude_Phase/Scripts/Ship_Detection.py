# Import necessary libraries
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import Instances, Boxes
import tifffile
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
import xml.etree.ElementTree as ET
import torch
from sentinelhub import SentinelHubRequest, DataCollection, MosaickingOrder, MimeType, BBox, CRS , SHConfig
from scipy import ndimage
import argparse
from skimage.morphology import opening, closing, square
from skimage.filters import median
from skimage import measure, morphology
from matplotlib.colors import ListedColormap

def calculate_mndwi(vv_channel, vh_channel):
    # Convert to float for division and handling negative values
    vv = vv_channel.astype(float)
    vh = vh_channel.astype(float)

    # Calculate MNDWI
    mndwi = (vh - vv) / (vh + vv + 1e-10)
    return mndwi

def modify_model_conv1(model):
    # Get the original first convolutional layer from the pre-trained model
    old_conv1 = model.backbone.bottom_up.stem.conv1

    # Create a new Conv2d layer with 4 input channels and the same output channels, kernel size, etc.
    new_conv1 = torch.nn.Conv2d(
        in_channels=4,  #input channels changed to 4
        out_channels=old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        bias=(old_conv1.bias is not None)
    )
  
    # Replace the first conv layer with the new one
    model.backbone.bottom_up.stem.conv1 = new_conv1

    return model

# Sentinel Hub Configuration
config = SHConfig()
config.instance_id = 'xxxx'  # Replace with your Sentinel Hub instance ID
config.sh_client_id = 'xxxx'  # Replace with your client ID
config.sh_client_secret = 'xxxx'  # Replace with your client secret

# Ensure you have correctly set your credentials
if config.instance_id == '' or config.sh_client_id == '' or config.sh_client_secret == '':
    print("Warning: Please set up your Sentinel Hub configuration with your credentials.")

evalscript_s1 = """
   //VERSION=3
        function setup() {
            return {
                input: ["VV", "VH"],  // For dual-polarization GRD mode
                output: {
                    bands: 2,
                    sampleType: "FLOAT32"
                }
            };
        }
        function evaluatePixel(sample) {
            return [Math.sqrt(sample.VV), Math.sqrt(sample.VH)];
        }
    """

parser = argparse.ArgumentParser(description="Sentinel-1 Data Request")

# Step 2: Define arguments
parser.add_argument('--lat1', type=float, required=True, help='Latitude of the first point of the bounding box')
parser.add_argument('--lon1', type=float, required=True, help='Longitude of the first point of the bounding box')
parser.add_argument('--lat2', type=float, required=True, help='Latitude of the second point of the bounding box')
parser.add_argument('--lon2', type=float, required=True, help='Longitude of the second point of the bounding box')
parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYY-MM-DD format')
parser.add_argument('--end_date', type=str, required=True, help='End date in YYYY-MM-DD format')

args = parser.parse_args()

# --lat1 1.2661 --lon1 103.8727 --lat2 1.2806 --lon2 103.8908 --start_date 2022-01-01 --end_date 2022-12-31
# --lat1 1.2450 --lon1 103.7000 --lat2 1.2800 --lon2 103.8000 --start_date 2022-01-01 --end_date 2022-12-31
# --lat1 1.2600 --lon1 103.7600 --lat2 1.3100 --lon2 103.8100 --start_date 2022-01-01 --end_date 2022-12-31
# --lat1 53.525 --lon1 9.930 --lat2 53.545 --lon2 9.980 --start_date 2022-01-01 --end_date 2022-12-31

# Define the time interval (YYYY-MM-DD format)
singapore_port_bbox = BBox(bbox=[args.lon1, args.lat1, args.lon2, args.lat2], crs=CRS.WGS84)

# Define the time interval (YYYY-MM-DD format)
my_time_interval = (args.start_date, args.end_date)

# Sentinel Hub Request for Sentinel-1 dual-polarimetric data
request_s1 = SentinelHubRequest(
    evalscript=evalscript_s1,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL1_IW,  # IW mode for dual-polarimetric data
            time_interval=my_time_interval,
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
    bbox=singapore_port_bbox,
    size=(256, 256),  # Output size in pixels (width, height)
    config=config,
)

# Execute the request
data = request_s1.get_data()

# Check if data is received
if data:
    image_data = data[0]  # First element in the data list

    # Save the data as a TIFF file
    output_file = 'sentinel_1_data.tif'
    tifffile.imwrite(output_file, image_data)
    print(f"Data saved as {output_file}")
else:
    print("No data received.")

#print("Requested BBox:", singapore_port_bbox)
#print("Known request size:", (256, 256))

display_size = (256,256)


bbox_list = [args.lon1, args.lat1, args.lon2, args.lat2]
#print("bbox_list: ", bbox_list)

tif_image = tifffile.imread('sentinel_1_data.tif')

#print("tif_image shape:" ,tif_image.shape)
#Assuming tif_image is in the shape of [height, width, 2]
vv_channel = tif_image[:, :, 0]
#print("Shape of SAR data (VV channel):", vv_channel.shape)
vh_channel = tif_image[:, :, 1]
#print("Shape of SAR data (VH channel):", vh_channel.shape)
img_size = vv_channel.shape

# Apply a contrast stretch by clipping the intensity values to the 5th and 95th percentiles
vv_stretched = np.clip(vv_channel, np.percentile(vv_channel, 2), np.percentile(vv_channel, 98))
vv_normalized = (vv_stretched - np.min(vv_stretched)) / (np.max(vv_stretched) - np.min(vv_stretched))

vh_stretched = np.clip(vh_channel, np.percentile(vh_channel, 2), np.percentile(vh_channel, 98))
vh_normalized = (vh_stretched - np.min(vh_stretched)) / (np.max(vh_stretched) - np.min(vh_stretched))

mndwi_threshold = 0.5
land_threshold = np.percentile(vv_channel, 95)

mndwi = calculate_mndwi(vv_normalized, vh_normalized)
water_mask = mndwi > mndwi_threshold
land_mask = vv_channel > land_threshold

min_ship_size = 10  # minimum pixel area for the smallest ship you want to detect
max_ship_size = 20  # maximum pixel area for the largest ship you want to detect

# Label and filter based on size to identify ships
label_objects, nb_labels = ndimage.label(land_mask)
sizes = ndimage.sum(land_mask, label_objects, range(nb_labels + 1))
ship_mask = ((sizes > min_ship_size) & (sizes < max_ship_size))[label_objects]

# Separate ship detection from land
refined_ship_mask = np.zeros_like(ship_mask)
refined_ship_mask[ship_mask] = land_mask[ship_mask]

# Create the combined mask
combined_mask = np.logical_or(water_mask, land_mask)

# Refine the land mask to exclude ships, if desired
refined_land_mask = np.logical_and(combined_mask, ~refined_ship_mask)

water_color = [0, 0, 255]  # Blue color for water
ship_color = [255, 0, 0]   # Red color for ships

# Apply colors
rgb_image = np.zeros((vv_channel.shape[0], vv_channel.shape[1], 3), dtype=np.uint8)
rgb_image[water_mask, 2] = water_color[2]
rgb_image[refined_ship_mask, 0] = ship_color[0]

# Overlay the refined masks onto the RGB image
rgb_image[..., 2] = np.where(water_mask, water_color[2], 0)
rgb_image[..., 0] = np.where(refined_land_mask, ship_color[0], 0)


plt.figure(figsize=(8, 6))
plt.hist(mndwi.ravel(), bins=50, color='blue', alpha=0.7)
plt.title('Histogram of MNDWI Values')
plt.xlabel('MNDWI Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
# Visualization

plt.figure(figsize=(14, 8))
# Display the original VV channel data
plt.subplot(1, 4, 1)
plt.imshow(vv_normalized, cmap='gray')
plt.title('VV Channel')
plt.axis('off')

# Display the MNDWI
plt.subplot(1, 4, 2)
plt.imshow(mndwi, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('MNDWI')
plt.axis('off')

# Display the land mask
plt.subplot(1, 4, 3)
plt.imshow(water_mask, cmap='Blues')
plt.title('Water Mask')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(rgb_image)
plt.title('Water and Land mask Overlay')
plt.axis('off')

plt.tight_layout()

# Simulate the real part of C12 using the Sobel filter
sobel_x = ndimage.sobel(vv_channel, axis=0)
sobel_y = ndimage.sobel(vv_channel, axis=1)
simulated_real = np.hypot(sobel_x, sobel_y)

# Simulate the imaginary part of C12 using the Gabor filter
gabor_kernel = cv2.getGaborKernel((5, 5), 4.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
simulated_imag = cv2.filter2D(vv_channel, cv2.CV_32F, gabor_kernel)

# Stack the channels to create a 4-channel image
simulated_4_channel_image = np.stack((vv_channel, simulated_real, simulated_imag, vh_channel), axis=-1)

sar_image_tensor = torch.as_tensor(simulated_4_channel_image.astype("float32")).permute(2, 0, 1)
print("SAR image tensor shape:", sar_image_tensor.shape)
print("SAR image tensor value range:", sar_image_tensor.min(), sar_image_tensor.max())
print("Bounding Box:", bbox_list)

# Load the config
cfg = get_cfg()
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.merge_from_file("config.yaml")  # Path to the config file used for training
cfg.MODEL.WEIGHTS = "model2/model_0099999.pth"  # Path to the trained model weights
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # Update to match the number of classes in your dataset
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05 # Set the testing threshold for this model
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

 # Move to CUDA if available
if torch.cuda.is_available():
    sar_image_tensor = sar_image_tensor.to("cuda")

# Inference
with torch.no_grad():
    outputs = model([{"image": sar_image_tensor}])[0]

# Extract prediction results
instances = outputs["instances"].to("cpu")
pred_boxes = instances.pred_boxes if instances.has("pred_boxes") else None
scores = instances.scores if instances.has("scores") else None
pred_classes = instances.pred_classes if instances.has("pred_classes") else None
# If there are no predicted boxes, continue to the next image
if pred_boxes is None:
    print("No predictions found for image")
else:
# Filter predictions to get only ships
    ship_indices = [i for i, pc in enumerate(pred_classes) if pc == 0]
    pred_boxes = pred_boxes[ship_indices]
    scores = scores[ship_indices]
    pred_classes = pred_classes[ship_indices]
    pred_boxes_list = pred_boxes.tensor.tolist()
           
vis_image_np = sar_image_tensor.permute(1, 2, 0).cpu().numpy().astype(int)

v = Visualizer(vis_image_np, metadata=metadata, scale=1.0)

print(outputs["instances"].to("cpu"))
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
vis_image = out.get_image()

# Resize the visualized image if necessary
result_tif_image = cv2.resize(vis_image, display_size, interpolation=cv2.INTER_AREA)

fig, axes = plt.subplots(1, 3, figsize=(16, 12))  # Create a figure with 3 subplots

# Visualize the inference result
axes[0].imshow(result_tif_image)
axes[0].set_title("Inference on TIF image")
axes[0].axis('off')

# Visualize the VV polarization channel
axes[1].imshow(vv_channel, cmap='gray')
axes[1].set_title("VV Channel")
axes[1].axis('off')

# Visualize the VH polarization channel
axes[2].imshow(vh_channel, cmap='gray')
axes[2].set_title("VH Channel")
axes[2].axis('off')


#plt.figure(figsize=(10, 10))
#plt.imshow(result_tif_image)
#plt.axis('off')  # Hide the axes
plt.tight_layout()
plt.show()

'''
### Without masking detection

# Extract prediction results
instances = outputs["instances"].to("cpu")
pred_boxes = instances.pred_boxes if instances.has("pred_boxes") else None
# If there are no predicted boxes, continue to the next image
if pred_boxes is None:
    print("No predictions found for image")

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

vis_image_np = sar_image_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

v = Visualizer(vis_image_np, metadata=metadata, scale=1.0)
print(outputs["instances"].to("cpu"))
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
vis_image = out.get_image()


plt.figure(figsize=(14, 8))
# Display the original VV channel data
plt.subplot(1, 3, 1)
plt.imshow(vis_image)
plt.title("Inference on TIF image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(vv_normalized, cmap='gray')
plt.title('VV Channel')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(vh_normalized, cmap='Blues')
plt.title('VH Channel')
plt.axis('off')
plt.tight_layout()
plt.show()
'''