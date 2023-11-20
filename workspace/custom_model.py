import os
import xml.etree.ElementTree as ET
import tifffile as tiff
from detectron2.structures import BoxMode , Instances , Boxes
from detectron2.data import DatasetCatalog, MetadataCatalog , build_detection_train_loader , DatasetMapper , build_detection_test_loader
import copy
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.utils.events import EventWriter
from detectron2.engine import HookBase
from detectron2.engine import hooks as detectron2_hooks

def annotations_to_instances(annotations, image_height, image_width):
    instances = Instances((image_height, image_width))
    boxes = [ann["bbox"] for ann in annotations]
    boxes = BoxMode.convert(boxes, annotations[0]["bbox_mode"], BoxMode.XYXY_ABS)
    instances.gt_boxes = Boxes(boxes)
    instances.gt_classes = torch.tensor([ann["category_id"] for ann in annotations])
    return instances


class CustomDatasetMapper(DatasetMapper):
    def __init__(self, cfg, is_train: bool = True):
        super().__init__(cfg, is_train=is_train)
        self.is_train = is_train
        self.image_format = cfg.INPUT.FORMAT

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        
        # Use tifffile to read the image
        image = tiff.imread(dataset_dict["file_name"])
        
        # Check if the image has 4 channels (amplitude and phase)
        if image.shape[-1] != 4:
            raise ValueError(f"The image {dataset_dict['file_name']} does not have 4 channels")
        
        # Convert the image to a format Detectron2 can work with (C, H, W)
        image = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        
        # Update the dataset dict
        dataset_dict["image"] = image
        
        if "annotations" in dataset_dict:
            annotations = dataset_dict.pop("annotations")
            dataset_dict["instances"] = annotations_to_instances(
                annotations, dataset_dict["height"], dataset_dict["width"]
            )

        return dataset_dict


def get_dicts(img_dir, annotation_dir):
    dataset_dicts = []
    for idx, filename in enumerate(os.listdir(img_dir)):
        if not filename.endswith('.tif'):
            continue
        
        record = {}
        
        image_path = os.path.join(img_dir, filename)
        image = tiff.imread(image_path)
        height, width = image.shape[:2]
        
        record["file_name"] = image_path
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        
        annotation_file = os.path.splitext(filename)[0] + '.xml'
        annotation_path = os.path.join(annotation_dir, annotation_file)
        if not os.path.exists(annotation_path):
            continue

        tree = ET.parse(annotation_path)
        root = tree.getroot()

        objs = []
        for member in root.findall('object'):
            class_name = member.find('name').text
            category_id = class_names.index(class_name)
            bndbox = member.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            xmax = int(bndbox.find('xmax').text)
            ymin = int(bndbox.find('ymin').text)
            ymax = int(bndbox.find('ymax').text)
            obj = {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": category_id,  
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts
'''
for d in ["train", "test"]:
    DatasetCatalog.register("sar_ships_" + d, lambda d=d: get_dicts(f'originalTIF/{d}', f'annotations/HorizontalBox/{d}'))
    MetadataCatalog.get("sar_ships_" + d).set(thing_classes=["ship","non_ship"])
'''

# Update the class names to include 'non_ship'
class_names = ["ship", "non_ship"]
thing_classes = class_names  # This should be a list of strings, representing each class

DatasetCatalog.register("sar_ships_train", lambda: get_dicts("combined_dataset/images", "combined_dataset/annotations"))
MetadataCatalog.get("sar_ships_train").set(thing_classes=thing_classes)

# Visualize the dataset to confirm everything is correct 
from detectron2.utils.visualizer import Visualizer
import random
import os
import matplotlib.pyplot as plt
from tifffile import imread
'''
png_img_dir = 'PNGImages/train'

metadata = MetadataCatalog.get("sar_ships_train")
dataset_dicts = get_dicts('originalTIF/train', 'annotations/HorizontalBox/train')

for d in random.sample(dataset_dicts, 3):
    tiff_image_path = d["file_name"]
    png_image_path = tiff_image_path.replace('originalTIF', 'PNGImages').replace('.tif', '.png')
    
    if not os.path.exists(png_image_path):
        print(f"Corresponding PNG image not found for {tiff_image_path}")
        continue

    try:
        # Attempt to read the TIFF image for detection data
        tiff_img = imread(tiff_image_path)
        print(f"Image {tiff_image_path} read successfully with shape {tiff_img.shape}")
    except Exception as e:
        print(f"Error reading {tiff_image_path}: {e}")
        continue
    

    # Use the corresponding PNG image for visualization
    png_img = plt.imread(png_image_path)
    if png_img is None:
        print(f"Failed to read corresponding PNG image for {tiff_image_path}")
        continue

    visualizer = Visualizer(png_img[:, :, :3], metadata=metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)

    # Create a subplot to display images side by side
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    
    # Display TIFF image with detections on the left
    ax[0].imshow(vis.get_image())
    ax[0].set_title("Detection on TIFF image")
    ax[0].axis('off')
    
    # Display corresponding PNG image on the right
    ax[1].imshow(png_img)
    ax[1].set_title("Corresponding PNG image")
    ax[1].axis('off')
    
    plt.show()
'''

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

def setup(cfg):

    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("sar_ships_train",)
    #cfg.DATASETS.TEST = ("sar_ships_test",)
    cfg.DATALOADER.NUM_WORKERS = 4

    # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001  # pick a good LR
    cfg.SOLVER.MAX_ITER = 100000    # number of iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)  
    
    # Adjust pixel mean and std for 4-channel images
    cfg.MODEL.PIXEL_MEAN = [ 0.00941263, -0.00114559,  0.00115459,  0.10023721]  
    cfg.MODEL.PIXEL_STD = [1.81057896, 1.49695959 ,2.00576686, 7.44893589]               

    # Specify the output directory
    cfg.OUTPUT_DIR = "model5"
    if not os.path.exists( cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    print(cfg)
    with open("config4.yaml", "w") as f:
        f.write(cfg.dump())

    return cfg

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

class SimpleLogger(HookBase):
    def after_step(self):
        # This will print iteration number every training step
        print(f"Iteration: {self.trainer.iter}")


class NoOpWriter(EventWriter):
    def write(self):
        pass

    def close(self):
        pass

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_model(cls, cfg):
        # Build the original model
        model = build_model(cfg)
        
        # Modify the first convolutional layer to accept 4-channel input
        model = modify_model_conv1(model)

       
        model.to(cfg.MODEL.DEVICE)

        return model
    
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=CustomDatasetMapper(cfg))
    
    def build_writers(self):
        # Return a list with a no-op writer to disable all writing
        return [NoOpWriter()]
    
    def build_hooks(self):
        # Call the parent method to get the list of default hooks
        default_hooks = super().build_hooks()

        # Filter out hooks related to Tensorboard
        hooks_without_tensorboard = [hook for hook in default_hooks if not hasattr(hook, '_writer')]

        # Add the custom simple logger hook
        hooks_without_tensorboard.append(SimpleLogger())

        return hooks_without_tensorboard
    
def main():
    cfg = get_cfg()
    cfg = setup(cfg)
    trainer = CustomTrainer(cfg)
    #resume_dir = "model4\last_checkpoint" # get the last resume checkpoint file path
    #trainer.resume_or_load(resume_dir)# pass the resume_dir path here 
    trainer.resume_or_load(resume=False)
    trainer.train()
    #evaluator = COCOEvaluator("sar_ships_test", cfg, False, output_dir=cfg.OUTPUT_DIR)
    #val_loader = build_detection_test_loader(cfg, "sar_ships_test")
    #inference_on_dataset(trainer.model, val_loader, evaluator)

if __name__ == '__main__':
    main()