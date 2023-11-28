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
import cv2
import xml.etree.ElementTree as ET
import torch
from global_land_mask import globe
from mpl_toolkits.basemap import Basemap
from sentinelhub import SentinelHubRequest, DataCollection, MosaickingOrder, MimeType, BBox, CRS , SHConfig
from pyproj import Transformer

# Sentinel Hub Configuration
config = SHConfig()
config.instance_id = '657f9fba-b10d-49ac-9c16-3b00766d71ad'  # Replace with your Sentinel Hub instance ID
config.sh_client_id = '331d6059-8e44-48c1-9f36-95b6cb76a1dc'  # Replace with your client ID
config.sh_client_secret = '68Jk1Um1NM1RDMo56TYyrgYXermLsasGip4guJNt'  # Replace with your client secret

# Ensure you have correctly set your credentials
if config.instance_id == '' or config.sh_client_id == '' or config.sh_client_secret == '':
    print("Warning: Please set up your Sentinel Hub configuration with your credentials.")
'''
# Define an evalscript for dual-polarimetric SAR data (VV and VH)
evalscript_s1 = """
//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["VV", "VH"],
            units: "LINEAR_POWER"
        }],
        output: {
            bands: 2,
            sampleType: "FLOAT32"
        }
    };
}

function evaluatePixel(sample) {
    return [sample.VV, sample.VH];
}
"""
'''

evalscript_s1 = """
   //VERSION=3
        function setup() {
            return {
                input: ["B01", "B02", "B03", "B04"],
                output: { bands: 4 }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B01, sample.B02, sample.B03, sample.B04];
        }
    """

width = height = 2560
center_lat = 51.94  # Approximate latitude for the Port of Rotterdam
center_lon = 4.25  # Approximate longitude for the Port of Rotterdam
# Define the transformer for WGS84 (epsg:4326) and UTM zone 31N
transformer = Transformer.from_crs("epsg:4326", "epsg:32631", always_xy=True)

# Convert the center point from WGS84 to UTM coordinates
center_x, center_y = transformer.transform(center_lon, center_lat)

# Calculate the corner points of the bounding box in UTM coordinates
min_x = center_x - width / 2
min_y = center_y - height / 2
max_x = center_x + width / 2
max_y = center_y + height / 2

# Define the transformer for UTM to WGS84
transformer_back = Transformer.from_crs("epsg:32631", "epsg:4326", always_xy=True)

# Convert the corner points back to WGS84 coordinates
min_lon, min_lat = transformer_back.transform(min_x, min_y)
max_lon, max_lat = transformer_back.transform(max_x, max_y)

(min_lon, min_lat, max_lon, max_lat)
# Define the bounding box for your area of interest
# Example coordinates (min_lon, min_lat, max_lon, max_lat)
rotterdam_bbox = BBox(
    bbox=[
        min_lon,
        min_lat,
        max_lon,
        max_lat
    ],
    crs=CRS.WGS84
)

# Define the time interval (YYYY-MM-DD format)
my_time_interval = ('2020-01-01', '2020-12-31')

# Sentinel Hub Request for Sentinel-1 dual-polarimetric data
request_s1 = SentinelHubRequest(
    evalscript=evalscript_s1,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,  # IW mode for dual-polarimetric data
            time_interval=my_time_interval,
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
    bbox=rotterdam_bbox,
    size=(256, 256),  # Output size in pixels (width, height)
    config=config,
)

# Execute the request
data = request_s1.get_data()

# Check if data is received
if data:
    image_data = data[0]  # First element in the data list

    # Check the number of bands
    if image_data.ndim == 3 and image_data.shape[2] == 2:
        print("Both VV and VH bands are present.")
    else:
        print(f"Unexpected data shape: {image_data.shape}")

    # Save the data as a TIFF file
    output_file = 'sentinel_1_data.tif'
    tifffile.imwrite(output_file, image_data)
    print(f"Data saved as {output_file}")
else:
    print("No data received.")

'''
# Optional: Display the data
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(data[:, :, 0], cmap='gray')
plt.title('VV Band')
plt.subplot(1, 2, 2)
plt.imshow(data[:, :, 1], cmap='gray')
plt.title('VH Band')
plt.show()
'''
tif_image = tifffile.imread('sentinel_1_data.tif')
print("tif_image shape:" ,tif_image.shape)
sar_image_resized = cv2.resize(tif_image, (256, 256))

blurred_vv = cv2.GaussianBlur(sar_image_resized[:, :, 0], (3, 3), 0)
blurred_vh = cv2.GaussianBlur(sar_image_resized[:, :, 1], (3, 3), 0)

# Stack the channels to form a four-channel image
four_channel_image = np.stack((sar_image_resized[:, :, 0], 
                               sar_image_resized[:, :, 1],
                               blurred_vv,
                               blurred_vh), axis=-1)

sar_image_tensor = torch.as_tensor(sar_image_resized.astype("float32")).permute(2, 0, 1)

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

# Convert the image for saving
vis_image_pil = Image.fromarray(vis_image)
vis_image_pil.save('detection_result.png')