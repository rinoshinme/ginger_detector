import xml.etree.ElementTree as ElementTree
import os
import numpy as np
import cv2
import pickle
import copy
from config import abs_path

DATA_FLIP = True
CACHE_PATH = abs_path('dataset/cache')


class Dataset(object):
    def __init__(self, data_path, phase, batch_size, class_names, rebuild=False):
        # VOC2007 path
        self.data_path = data_path
        self.cache_path = CACHE_PATH
        self.batch_size = batch_size
        self.image_size = 448
        self.cell_size = 7
        self.classes = class_names
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        self.flipped = DATA_FLIP
        self.phase = phase
        self.rebuild = rebuild
        self.cursor = 0
        self.epoch = 1
        self.gt_labels = None  # image path and box annotations
        self.prepare()

    def get(self):
        images = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros(
            (self.batch_size, self.cell_size, self.cell_size, 5 + self.num_classes))
        count = 0
        while count < self.batch_size:
            img_name = self.gt_labels[self.cursor]['imname']
            flipped = self.gt_labels[self.cursor]['flipped']
            images[count, :, :, :] = self.image_read(img_name, flipped)
            labels[count, :, :, :] = self.gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels

    def image_read(self, img_name, flipped=False):
        image = cv2.imread(img_name)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            image = image[:, ::-1, :]
        return image

    def prepare(self):
        gt_labels = self.load_labels()
        if self.flipped:
            print('Appending horizontally flipped training examples...')
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                gt_labels_cp[idx]['label'] = gt_labels_cp[idx]['label'][:, ::-1, :]
                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                            gt_labels_cp[idx]['label'][i, j, 1] = \
                                self.image_size - 1 - gt_labels_cp[idx]['label'][i, j, 1]
            gt_labels += gt_labels_cp
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        return gt_labels

    def load_labels(self):
        cache_file = os.path.join(self.cache_path, 'pascal_' + self.phase + '_gt_labels.pkl')
        if os.path.isfile(cache_file) and not self.rebuild:
            print('Loading gt_labels from: {}'.format(cache_file))
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels
        print('Processing gt_labels from: {}'.format(self.data_path))
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        if self.phase == 'train':
            txtname = os.path.join(self.data_path, 'ImageSets', 'Main', 'trainval.txt')
        else:
            txtname = os.path.join(self.data_path, 'ImageSets', 'Main', 'test.txt')

        with open(txtname, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]

        gt_labels = []
        for index in self.image_index:
            label, num = self.load_pascal_annotation(index)
            if num == 0:
                continue
            img_name = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
            gt_labels.append({
                'imname': img_name,
                'label': label,
                'flipped': False
            })
        print('Saving gt_labels to: {}'.format(cache_file))
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)
        return gt_labels

    def load_pascal_annotation(self, index):
        img_name = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        img = cv2.imread(img_name)
        h_ratio = 1.0 * self.image_size / img.shape[0]
        w_ratio = 1.0 * self.image_size / img.shape[1]

        label = np.zeros((self.cell_size, self.cell_size))
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ElementTree.parse(filename)
        objs = tree.findall('object')

        for obj in objs:
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            x1 = max(min(x1 * w_ratio, self.image_size - 1), 0)
            y1 = max(min(y1 * h_ratio, self.image_size - 1), 0)
            x2 = max(min(x2 * w_ratio, self.image_size - 1), 0)
            y2 = max(min(y2 * h_ratio, self.image_size - 1), 0)

            cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]
            box_coord = [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1]
            x_ind = int(box_coord[0] * self.cell_size / self.image_size)
            y_ind = int(box_coord[1] * self.cell_size / self.image_size)
            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = box_coord
            label[y_ind, x_ind, 5 + cls_ind] = 1
        return label, len(objs)


if __name__ == '__main__':
    voc_path = '/Users/liyu/data/VOCdevkit/VOC2007'
    ph = 'train'
    bs = 1
    voc_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    dataset = Dataset(voc_path, ph, bs, voc_names)

    im, lbl = dataset.get()
    print(im)
    print(lbl)
    print(im.shape)
    print(lbl.shape)
