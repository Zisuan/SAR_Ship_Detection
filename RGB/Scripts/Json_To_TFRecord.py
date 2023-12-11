import os
import io
import json
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util

import argparse

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Sample TensorFlow XML-to-TFRecord converter")
parser.add_argument("-j",
                    "--json_dir",
                    help="Path to the folder where the input .json files are stored.",
                    type=str)
parser.add_argument("-l",
                    "--labels_path",
                    help="Path to the labels (.pbtxt) file.", type=str)
parser.add_argument("-o",
                    "--output_path",
                    help="Path of output TFRecord (.record) file.", type=str)
parser.add_argument("-i",
                    "--image_dir",
                    help="Path to the folder where the input image files are stored. "
                         "Defaults to the same directory as XML_DIR.",
                    type=str, default=None)
parser.add_argument("-c",
                    "--csv_path",
                    help="Path of output .csv file. If none provided, then no file will be "
                         "written.",
                    type=str, default=None)

args = parser.parse_args()

def json_to_examples(json_path, image_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image  = fid.read()

    image = Image.open(image_path)
    width, height = image.size

    filename = os.path.basename(image_path).encode('utf8')
    image_format = b'jpg'
    examples = []
    
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        xmin = min(point[0] for point in points) / width
        xmax = max(point[0] for point in points) / width
        ymin = min(point[1] for point in points) / height
        ymax = max(point[1] for point in points) / height

        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
        classes_text.append(label.encode('utf8'))
        classes.append(1)  # Assuming there's only one class (ship)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    examples.append(tf_example)
    
    return examples


def main(_):

    folder_path = args.json_dir  # Specify the path to the folder containing both images and JSON files

    writer = tf.io.TFRecordWriter(args.output_path)

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            json_path = os.path.join(folder_path, filename)
            image_name = os.path.splitext(filename)[0] + '.jpg'
            image_path = os.path.join(folder_path, image_name)
            
            tf_examples = json_to_examples(json_path, image_path)
            for example in tf_examples:
                writer.write(example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.compat.v1.app.run()
