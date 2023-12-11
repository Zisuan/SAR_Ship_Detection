import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

def prepare_combined_dataset(train_image_dir, train_xml_dir, hard_negatives_dir, hard_negatives_xml_dir, combined_dir, prefix='hard_neg_'):
    # Create new directories for combined dataset
    combined_image_dir = Path(combined_dir) / 'images'
    combined_xml_dir = Path(combined_dir) / 'annotations'
    combined_image_dir.mkdir(parents=True, exist_ok=True)
    combined_xml_dir.mkdir(parents=True, exist_ok=True)

    # Copy and rename hard negatives
    for image_file in Path(hard_negatives_dir).glob('*.tif'):
        new_image_name = prefix + image_file.name
        shutil.copy(image_file, combined_image_dir / new_image_name)

        # Update and copy XML files
        xml_file = Path(hard_negatives_xml_dir) / image_file.with_suffix('.xml').name
        if xml_file.exists():
            tree = ET.parse(xml_file)
            root = tree.getroot()
            filename_tag = root.find('filename')
            if filename_tag is not None:
                filename_tag.text = new_image_name
            tree.write(combined_xml_dir / (prefix + xml_file.name))
    
    # Copy original training data .tif files
    for image_file in Path(train_image_dir).glob('*.tif'):
        shutil.copy(image_file, combined_image_dir / image_file.name)

    # Copy original training data .xml files
    for xml_file in Path(train_xml_dir).glob('*.xml'):
        shutil.copy(xml_file, combined_xml_dir / xml_file.name)

# Example usage
prepare_combined_dataset(
    train_image_dir='originalTIF/train', 
    train_xml_dir='annotations/HorizontalBox/train',
    hard_negatives_dir='hard_negatives', 
    hard_negatives_xml_dir = 'hard_negatives_annotations',
    combined_dir='combined_dataset'
)
