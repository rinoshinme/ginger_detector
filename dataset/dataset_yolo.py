import tensorflow as tf
import cv2
import os
import random
import numpy as np
from utils import io
from config import abs_path
from dataset import augmentation
from utils.preprocess import preprocess_detection

TRAIN_SIZES = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
TEST_SIZES = [544]
YOLO_STRIDES = [8, 16, 32]
NUM_CLASSES = 20
ANCHORS_FILE = abs_path('config/anchors/baseline_anchors.txt')


class Dataset(object):
    def __init__(self, phase, annot_path, batch_size, data_aug=True):
        assert phase in ['train', 'test']
        self.annot_path = annot_path
        if phase == 'train':
            self.input_sizes = TRAIN_SIZES
        else:
            self.input_sizes = TEST_SIZES
        self.batch_size = batch_size
        self.data_aug = data_aug

        self.strides = np.array(YOLO_STRIDES)
        self.num_classes = NUM_CLASSES
        self.anchors = np.array(io.read_anchors(ANCHORS_FILE))
        self.anchors_per_scale = 3
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations()
        self.num_samples = len(self.annotations)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def load_annotations(self):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        return annotations

    def parse_annotation(self, annotation, target_size):
        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError('%s does not exists...' % image_path)
        image = cv2.imread(image_path)
        bboxes = [[int(v) for v in item.split(',')] for item in line[1:]]
        bboxes = np.array(bboxes)

        if self.data_aug:
            image, bboxes = augmentation.random_horizontal_flip(image, bboxes)
            # do other augmentation functions

        image, bboxes = preprocess_detection(image, target_size, bboxes)
        return image, bboxes

    def preprocess_true_boxes(self, bboxes, output_size):
        label = [np.zeros((output_size[0], output_size[1], self.anchors_per_scale,
                           5 + self.num_classes)) for _ in range(3)]
        boxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros(3)  # number of boxes in each scale

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device('/cpu:0'):
            input_size = random.choice(self.input_sizes)
            output_size = input_size // self.strides

            batch_image = np.zeros((self.batch_size, input_size, input_size, 3))
            batch_label_sbbox = np.zeros((self.batch_size, output_size[0], output_size[0],
                                          self.anchors_per_scale, 5 + self.num_classes))
            batch_label_mbbox = np.zeros((self.batch_size, output_size[1], output_size[1],
                                          self.anchors_per_scale, 5 + self.num_classes))
            batch_label_lbbox = np.zeros((self.batch_size, output_size[2], output_size[2],
                                          self.anchors_per_scale, 5 + self.num_classes))
            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))

            num = 0
            if self.batch_count < self.num_batches:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:
                        index -= self.num_samples
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotation(annotation, [input_size, input_size])

                    # preprocess true boxes.


if __name__ == '__main__':
    dataset_text = abs_path('config/data/train.txt')
    dataset = Dataset('train', dataset_text, 1)

    rimg, rbboxes = next(dataset)
    cv2.imshow('image', rimg)
    cv2.waitKey(0)
    print(rbboxes)
