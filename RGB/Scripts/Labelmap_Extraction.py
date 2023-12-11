import tensorflow as tf
from collections import defaultdict
from PIL import Image
from io import BytesIO

def extract_label_map_from_tfrecords(tfrecords_filename):
    # Extract class labels and IDs from TFRecords
    class_labels = defaultdict(set)

    raw_dataset = tf.data.TFRecordDataset(tfrecords_filename)
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }

    for raw_record in raw_dataset:
        parsed_record = tf.io.parse_single_example(raw_record, feature_description)
        class_texts = tf.sparse.to_dense(parsed_record['image/object/class/text']).numpy()
        class_labels_list = tf.sparse.to_dense(parsed_record['image/object/class/label']).numpy()
        '''
        if b'ship-29' in class_texts:
            # Decode and display the image using PIL
            image_data = parsed_record['image/encoded'].numpy()
            image = Image.open(BytesIO(image_data))
            image.show(title="ship-29")
        '''
        for class_text, class_label in zip(class_texts, class_labels_list):
            class_labels[class_text.decode('utf-8')].add(class_label)

    # Generate label map
    label_map = ""
    for class_text, class_labels_set in class_labels.items():
        # Since class labels should be unique, we can directly pick an element from the set
        class_label = next(iter(class_labels_set))
        label_map_entry = "item {\n  id: %d\n  name: '%s'\n}\n" % (class_label, class_text)
        label_map += label_map_entry

    return label_map

# Example usage:
tfrecords_filename = 'TFRecords'
label_map = extract_label_map_from_tfrecords(tfrecords_filename)
print(label_map)