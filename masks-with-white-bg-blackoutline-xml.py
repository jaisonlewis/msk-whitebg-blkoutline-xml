import pandas as pd
import cv2
import os
import ast
from tqdm import tqdm
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET

# Define directories and directory name for masks
mask_dir = 'norm_img_mask'
dirname = 'images2'

# Read the CSV file into a DataFrame
df = pd.read_csv('nor_img.csv')

# Extract filenames and annotations from the DataFrame
filenames = df['id'].values
annotations = df['annotations'].apply(ast.literal_eval).values

# Initialize a counter for naming the saved files
start_no = 0

# Loop through each image and its corresponding annotations
for filename, annotation_list in tqdm(zip(filenames, annotations)):

    # Create a blank white image
    blank_image = np.ones((512, 512, 3), dtype=np.uint8) * 255

    # Loop through each annotation and create a black outline on the blank image
    for annotation in annotation_list:
        for coord_set in annotation['coordinates']:
            points = np.array(coord_set, dtype=np.int32)
            cv2.polylines(blank_image, [points], isClosed=True, color=(0, 0, 0), thickness=2)

    # Save the masked image
    im_masked_name = os.path.join(mask_dir, "{}{}.png".format(filename, start_no))
    im_masked = Image.fromarray(blank_image)
    im_masked.save(im_masked_name)

    # Save the corresponding annotations as an XML file
    xml_name = os.path.join(mask_dir, "{}{}.xml".format(filename, start_no))
    root = ET.Element("annotation")
    object_node = ET.SubElement(root, "object")
    type_node = ET.SubElement(object_node, "type")
    type_node.text = annotation_list[0]['type']
    for annotation in annotation_list:
        for coord_set in annotation['coordinates']:
            points = np.array(coord_set, dtype=np.int32)
            bndbox_node = ET.SubElement(object_node, "bndbox")
            xmin_node = ET.SubElement(bndbox_node, "xmin")
            ymin_node = ET.SubElement(bndbox_node, "ymin")
            xmax_node = ET.SubElement(bndbox_node, "xmax")
            ymax_node = ET.SubElement(bndbox_node, "ymax")
            xmin_node.text = str(np.min(points[:, 0]))
            ymin_node.text = str(np.min(points[:, 1]))
            xmax_node.text = str(np.max(points[:, 0]))
            ymax_node.text = str(np.max(points[:, 1]))
    tree = ET.ElementTree(root)
    tree.write(xml_name)

    # Update the counter for naming the saved files
    start_no += 1
