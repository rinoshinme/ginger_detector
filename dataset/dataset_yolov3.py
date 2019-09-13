import tensorflow as tf
import cv2
import os
import random
import numpy as np
from utils import io
from config import abs_path
from dataset import augmentation
from utils.preprocess import preprocess_detection
from utils import bbox_np

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
        label = [np.zeros((output_size[i], output_size[i], self.anchors_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros(3)  # number of boxes in each scale

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]
            one_hot = np.zeros(self.num_classes, dtype=np.float)
            one_hot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = one_hot * (1 - deta) + uniform_distribution * deta

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                                        (bbox_coor[2:] - bbox_coor[:2])], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchors_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = bbox_np.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchors_per_scale)
                best_anchor = int(best_anchor_ind % self.anchors_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh

        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_batches

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
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = \
                        self.preprocess_true_boxes(bboxes, output_size)

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                return batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                    batch_sbboxes, batch_mbboxes, batch_lbboxes
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration


if __name__ == '__main__':
    dataset_text = abs_path('config/data/train.txt')
    dataset = Dataset('train', dataset_text, 1)

    img, ls, lm, ll, bs, bm, bl = next(dataset)

    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    print(img.shape)
    print(ls.shape)
    print(lm.shape)
    print(ll.shape)
