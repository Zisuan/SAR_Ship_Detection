import os
import xml.etree.ElementTree as ET

def merge_xml(folder1, folder2, output_folder):
    for filename in os.listdir(folder1):
        if not filename.endswith('.xml'):
            continue
        
        # Construct file paths
        xml_file1 = os.path.join(folder1, filename)
        xml_file2 = os.path.join(folder2, filename)
        
        if not os.path.exists(xml_file2):
            continue
        
        # Parse the XML files
        tree1 = ET.parse(xml_file1)
        tree2 = ET.parse(xml_file2)
        root1 = tree1.getroot()
        root2 = tree2.getroot()
        
        # Merge objects from xml_file2 into xml_file1
        for object2 in root2.findall('object'):
            root1.append(object2)
        
        # Save the merged XML to the output folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        tree1.write(os.path.join(output_folder, filename))

# Replace 'path_to_folder1', 'path_to_folder2', and 'path_to_output_folder' with your actual paths
merge_xml('annotations/HorizontalBox/train', 'hard_negatives', 'hard_negatives_annotations')
