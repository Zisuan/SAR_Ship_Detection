import tensorflow as tf
import matplotlib.pyplot as plt

def read_tfrecord(tfrecord_path):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

    # Define the feature description
    feature_description = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    }

    height_to_width_ratios = []
    width_to_height_ratios = []

    for raw_record in raw_dataset:
        example = tf.io.parse_single_example(raw_record, feature_description)
        
        # Extract bounding box
        width = tf.cast(example['image/width'], tf.float32)
        height = tf.cast(example['image/height'], tf.float32)
        xmin = example['image/object/bbox/xmin'].values * width
        xmax = example['image/object/bbox/xmax'].values * width
        ymin = example['image/object/bbox/ymin'].values * height
        ymax = example['image/object/bbox/ymax'].values * height
        
        box_heights = ymax - ymin
        box_widths = xmax - xmin

        ratios_h_w = box_heights / box_widths
        ratios_w_h = box_widths / box_heights

        height_to_width_ratios.extend(ratios_h_w.numpy())
        width_to_height_ratios.extend(ratios_w_h.numpy())

    return height_to_width_ratios, width_to_height_ratios

def plot_ratios(height_to_width_ratios, width_to_height_ratios):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(height_to_width_ratios, bins=50, facecolor='blue', alpha=0.7)
    plt.title('Height-to-Width Ratios')
    plt.xlabel('Ratio')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(width_to_height_ratios, bins=50, facecolor='green', alpha=0.7)
    plt.title('Width-to-Height Ratios')
    plt.xlabel('Ratio')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    tfrecord_path = 'workspace/training_demo/tfrecord_annotations.tfrecord'  # replace with your TFRecord path
    h_w_ratios, w_h_ratios = read_tfrecord(tfrecord_path)
    plot_ratios(h_w_ratios, w_h_ratios)
