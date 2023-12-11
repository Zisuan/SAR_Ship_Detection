import os
import xml.etree.ElementTree as ET
import tifffile as tiff
from tifffile import imread
import copy
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from detectron2.structures import BoxMode , Instances , Boxes
from detectron2.data import DatasetCatalog, MetadataCatalog , build_detection_train_loader , DatasetMapper , build_detection_test_loader
from detectron2.engine import DefaultPredictor, HookBase,  DefaultTrainer
from detectron2.utils.events import EventWriter
from detectron2.engine import hooks as detectron2_hooks
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

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
            category_id = 0
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

for d in ["train", "test"]:
    DatasetCatalog.register("sar_ships_" + d, lambda d=d: get_dicts(f'originalTIF/{d}', f'annotations/HorizontalBox/{d}'))
    MetadataCatalog.get("sar_ships_" + d).set(thing_classes=["ship"])

# Update the class names to include 'non_ship'
#class_names = ["ship", "non_ship"]
#thing_classes = class_names  # This should be a list of strings, representing each class

#DatasetCatalog.register("sar_ships_train", lambda: get_dicts("combined_dataset/images", "combined_dataset/annotations"))
#MetadataCatalog.get("sar_ships_train").set(thing_classes=thing_classes)


def setup(cfg):

    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("sar_ships_train",)
    cfg.DATASETS.TEST = ("sar_ships_test",)
    cfg.DATALOADER.NUM_WORKERS = 4

    # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001  # pick a LR
    cfg.SOLVER.MAX_ITER = 100000 # number of iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
    
    # Adjust pixel mean and std for 4-channel images
    cfg.MODEL.PIXEL_MEAN = [ 0.00941263, -0.00114559,  0.00115459,  0.10023721]  
    cfg.MODEL.PIXEL_STD = [1.81057896, 1.49695959 ,2.00576686, 7.44893589]               

    # Specify the output directory
    cfg.OUTPUT_DIR = "model_last"
    if not os.path.exists( cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    print(cfg)
    with open("config_last.yaml", "w") as f:
        f.write(cfg.dump())

    return cfg

def modify_model_conv1(model):
    # Get the original first convolutional layer from the pre-trained model
    old_conv1 = model.backbone.bottom_up.stem.conv1

    # Create a new Conv2d layer with 4 input channels and the same output channels, kernel size, etc.
    new_conv1 = torch.nn.Conv2d(
        in_channels=4,  # Input channels changed to 4
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
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self._data_loader_iter = iter(self.data_loader)

    def run_step(self):
        assert self.model.training, "[CustomTrainer] model was changed to eval mode!"
        data = next(self._data_loader_iter)
        loss_dict = self.model(data)
        self.optimizer.zero_grad()
        losses = sum(loss_dict.values())
        losses.backward()
        
        # Gradient clipping and checks
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"NaN or Inf found in gradients of {name}")

        self.optimizer.step()
    
def main():
    cfg = get_cfg()
    cfg = setup(cfg)
    trainer = CustomTrainer(cfg)
    
    # Use below two commented lines to resume training from previous checkpoint
    #resume_dir = "model6\last_checkpoint" # get the last resume checkpoint file path
    #trainer.resume_or_load(resume_dir)# pass the resume_dir path here 

    # Use commented line below to start training from scratch
    #trainer.resume_or_load(resume=False)
    trainer.train()
    evaluator = COCOEvaluator("sar_ships_test", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "sar_ships_test")
    inference_on_dataset(trainer.model, val_loader, evaluator)

if __name__ == '__main__':
    main()