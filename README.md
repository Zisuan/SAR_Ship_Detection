# SAR_Ship_Detection

Ship Detection on Synthetic Aperture Radar (SAR) Images using amplitude and phase data

## Dataset Description
### DSSDD (Dual-polarimetric SAR Ship Detection Dataset)
- **Content**: 50 dual-polarimetric SAR images from Sentinel-1.
- **Format**: Cropped into 1236 image slices (256x256 pixels).
- **Polarizations**: VV and VH, fused into RGB channels for pseudo-color images.
- **Ship Instances**: 3540, labeled with rotatable (RBox) and horizontal bounding boxes (BBox).
- **Colour Depth**: 8 bits/channel (PNG), original 16-bit images (TIFF) also included.

### SSDD (SAR Ship Detection Dataset) 
- **Content**: 1160 images from three prominent satellites: RadarSat-2, TerraSAR-X, and Sentinel-1.
- **Format**:  Each image having dimensions of up to 500×350 pixels.
- **Polarizations**:  Mix of polarization modes, including HH, HV, VV, and VH.
- **Ship Instances**: 2358 ship instances, labeled with rotatable (RBox) and horizontal bounding boxes (BBox).
- **Colour Depth**: .jpeg format, color depth of 24 bits.  

### OpenSARShip 
- **Content**: Sentienl-1 interferometric wide (IW) ground range detected (GRD) products and Sentienl-1 interferometric wide (IW) single look complex (SLC) products.
- **Format**:  For each Sentinel-1 SAR image, four subfolders provide four formats of ship chips - Original data, calibrated data, visualized data in pseudo-color, and visualized data in
grey scale.
- **Polarizations**:  VV and VH.  
  
## Repository Structure
- [Amplitude_Phase](https://github.com/Zisuan/SAR_Ship_Detection/tree/main/Amplitude_Phase): Contains data and file informations used for amplitude and phase data detection  
- [RGB](https://github.com/Zisuan/SAR_Ship_Detection/tree/main/RGB): Contains data and file informations used for RGB data detection.
- [Classification](https://github.com/Zisuan/SAR_Ship_Detection/tree/main/Classification): Contains data and file informations used for Ships Classification.
The scripts used in the project are located in the Scripts folders found in each folder.  
Datasets and annotations utilized in the project are located in other folders.  

## Installation and Usage
For Amplitude and Phase data detection, Detectron2 models are used.  
[Detectron2 Installation Guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)  
After Installation, Pick a model and its config file from [model zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#coco-person-keypoint-detection-baselines-with-keypoint-r-cnn) and train a custom model using [custom_model.py](https://github.com/Zisuan/SAR_Ship_Detection/blob/main/Amplitude_Phase/Scripts/custom_model.py).  
After training, Run Inference Test on custom dataset using trained model weights using [Inference.py](https://github.com/Zisuan/SAR_Ship_Detection/blob/main/Amplitude_Phase/Scripts/Inference.py).  
For Real-time SAR Data detection, Set up a [Sentinel Account](https://www.sentinel-hub.com/) and [install SentinelHub](https://sentinelhub-py.readthedocs.io/en/latest/install.html) to use SentinelHub API to extract data, Run [Ship_Detection.py](https://github.com/Zisuan/SAR_Ship_Detection/blob/main/Amplitude_Phase/Scripts/Ship_Detection.py) using your own SH Configuration.     
  
For RGB data detection, Tensorflow models are used.  
[Tensorflow Installation Guide](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)  
After Installation, Download Pre-Trained Model from [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) and Configure the Training Pipeline to train a model.  
Various scripts needed for training can be found the [Script Folder](https://github.com/Zisuan/SAR_Ship_Detection/tree/main/RGB/Scripts).   
After training, Run Inference Test on custom dataset using trained model weights using [TF_Inference.py](https://github.com/Zisuan/SAR_Ship_Detection/blob/main/RGB/Scripts/TF_Inference.py).   

**Files and Directory Paths may differ based on Personal Folder Structure**  
**Install any other required dependencies as needed**

