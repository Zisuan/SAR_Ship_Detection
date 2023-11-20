# SAR_Ship_Detection

Ship Detection on Synthetic Aperture Radar (SAR) Images using amplitude and phase data

## Dataset Description
### DSSDD (Dual-polarimetric SAR Ship Detection Dataset)
- **Content**: 50 dual-polarimetric SAR images from Sentinel-1.
- **Format**: Cropped into 1236 image slices (256x256 pixels).
- **Polarizations**: VV and VH, fused into RGB channels for pseudo-color images.
- **Ship Instances**: 3540, labeled with rotatable (RBox) and horizontal bounding boxes (BBox).
- **Colour Depth**: 8 bits/channel (PNG), original 16-bit images (TIFF) also included.

## Repository Structure
root/
│
├── scripts/ # Python scripts used for image processing and machine learning models.
├── data/
│ ├── SAR_images/ # Original SAR images in TIFF format.
│ ├── RGB_images/ # Processed RGB images in PNG format.
│ └── annotations/ # Annotation files for ship instances.
│
├── results/ # Output results and model performance metrics.
├── figures/ # Figures and visualizations generated from the study.
└── README.md

## Installation and Usage
