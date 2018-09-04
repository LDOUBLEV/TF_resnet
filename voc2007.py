#coding:utf-8
import numpy as np
import os
import scipy.misc as scm
import tensorflow as tf
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']


def read_image_label(file):
    print('[dataset] read ' + file)
    data = dict()
    with open(file, 'r') as f:
        for line in f:
            tmp = line.split(' ')
            name = tmp[0]
            label = int(round((int(tmp[-1]) + 1)/2))
            data[name] = label
            # data.append([name, label])
            # print('%s  %d' % (name, label))
    return data


def read_object_labels(root, set):
    """
    :param root: ./voc2007trainval : which include Annotations,ImageseSets and JPEGImages
    :param set:  trainval  or train or val or test
    :return: label
    """
    path_labels = os.path.join(root, 'ImageSets', 'Main')
    labeled_data = dict()
    num_classes = len(object_categories)

    for i in range(num_classes):
        file = os.path.join(path_labels, object_categories[i] + '_' + set + '.txt')
        data = read_image_label(file)

        if i == 0:
            for (name, label) in data.items():
                labels = np.zeros(num_classes)
                labels[i] = label
                labeled_data[name] = labels
        else:
            for (name, label) in data.items():
                labeled_data[name][i] = label

    return labeled_data


def _max_top_k(vectors):
    index = []
    for i in range(len(vectors)):
        if vectors[i]:
            index.append(i)
    return  index


def voc_to_tfrecord(root, set, jpegpath, record_name, scale_size=500):
    """
    :param root:  ./voc2007trainval : which include Annotations,ImageseSets and JPEGImages
    :param set:  trainval or test
    :param jpegpath:
    :param record_name:
    :return:  nothing
    """
    label_data = read_object_labels(root, set)

    data_num = len(label_data)
    print('the total data num  is :', data_num)
    print('please wait, the files are tfrecording!!!')

    writer = tf.python_io.TFRecordWriter(record_name+'.tfrecord')
    for (name, label) in label_data.items():
        img_path = jpegpath + '/' + str(name) + '.jpg'
        # print(label)

        index = _max_top_k(label)
        # print(index)
        catagory = [object_categories[int(id_)] for id_ in index]
        print(name, len(label), catagory)

        img = scm.imread(img_path)
        img = scm.imresize(img, (scale_size, scale_size))

        img_raw = img.tobytes()
        label_raw = np.array(label, dtype=np.int32)

        example = tf.train.Example(features=tf.train.Features(feature={
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=label_raw))}
                ))  #  将数据整理成 TFRecord 需要的数据结构

        # "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}

        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()

    print('data transfer done')


def read_and_decode(filename, scale_size=500):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                           'label' : tf.FixedLenFeature(shape=(20,), dtype=tf.int64)
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [scale_size, scale_size, 3])
    # img = tf.cast(img, tf.float32)
    # label = tf.decode_raw(features['label'], tf.float64)

    label = tf.reshape(features['label'], [20])
    label = tf.cast(label, tf.float64)

    return img, label

def read_and_decode_coco(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer(filename)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature(shape=(80,), dtype=tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [400, 400, 3])
    # img = tf.cast(img, tf.float32)

    label = tf.reshape(features['label'], [80])
    label = tf.cast(label, tf.float64)

    return img, label

"""
###############################################################################
###############################################################################
###############################################################################
"""
slim = tf.contrib.slim

_FILE_PATTERN = 'flowers_%s_*.tfrecord'

SPLITS_TO_SIZES = {'trainval': 5011, 'test': 350}

_NUM_CLASSES = 1

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 19',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading flowers.
  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.
  Returns:
    A `Dataset` namedtuple.
  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  # if dataset_utils.has_labels(dataset_dir):
  #   labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)

def _get_image_label(dataset):
    with tf.device('/cpu:0'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=4,
            common_queue_capacity=20 * 16,
            common_queue_min=10 * 16)
        [image, label] = provider.get(['image', 'label'])
    return image, label

"""
###############################################################################
###############################################################################
###############################################################################
"""


if __name__ == '__main__':

    build_records = False
    if build_records:

        rootpath = '/home/liuweiwei02/Projects/tf_resnet/data/voc2007trainval'
        img_path = rootpath +'/JPEGImages'
        record_name = 'trainval'
        set = 'trainval'
        voc_to_tfrecord(rootpath, set, img_path, record_name)

        # train images num: 5011
        # test imgs num: 4952

    else:

        with tf.Session() as sess:

            print('open quene done')
            split_name = 'trainval'
            dir = '/home/liuweiwei02/Projects/tf_resnet/data'
            data_set = get_split(split_name, dir, file_pattern=None, reader=None)
            img, label = _get_image_label(data_set)
            # img, label = read_and_decode('/home/liuweiwei02/Projects/tf_resnet/trainval.tfrecord')
            # img_batch, label_batch = tf.train.shuffle_batch([img, label],
            #                                                 batch_size=1, capacity=220,
            #                                                 min_after_dequeue=210)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            # init = tf.initialize_all_variables()
            # sess.run(init)
            print('data load done')
            img = sess.run(img)
            label = sess.run(label)
            print('data run done')
            print(img)
            print(label)

            coord.request_stop()
            coord.join(threads)

