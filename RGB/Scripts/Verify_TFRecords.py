import tensorflow as tf
import io
import matplotlib.pyplot as plt
from PIL import Image

def decode_tfrecord(tfrecord_path):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        height = int(example.features.feature['image/height'].int64_list.value[0])
        width = int(example.features.feature['image/width'].int64_list.value[0])
        filename = example.features.feature['image/filename'].bytes_list.value[0].decode('utf-8')
        encoded_image = example.features.feature['image/encoded'].bytes_list.value[0]
        xmins = example.features.feature['image/object/bbox/xmin'].float_list.value
        xmaxs = example.features.feature['image/object/bbox/xmax'].float_list.value
        ymins = example.features.feature['image/object/bbox/ymin'].float_list.value
        ymaxs = example.features.feature['image/object/bbox/ymax'].float_list.value
        classes_text = example.features.feature['image/object/class/text'].bytes_list.value
        classes = example.features.feature['image/object/class/label'].int64_list.value

        img = Image.open(io.BytesIO(encoded_image))
        print("Image shape:", img.size, "Channels:", img.mode)  # Print the shape and number of channels of the image
        plt.imshow(img)
        ax = plt.gca()

        for i in range(len(xmins)):
            ymin = ymins[i] * height
            ymax = ymaxs[i] * height
            xmin = xmins[i] * width
            xmax = xmaxs[i] * width
            class_name = classes_text[i].decode('utf-8')

            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(xmin, ymin, class_name, bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 1})

        plt.show()


if __name__ == '__main__':
    tfrecord_path = 'workspace/training_demo/TFRecords' 
    decode_tfrecord(tfrecord_path)
