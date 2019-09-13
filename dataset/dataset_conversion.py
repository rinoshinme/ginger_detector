import os
from config import os_type, abs_path
from dataset.pascalvoc import PascalVOC

VOC_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
             'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
             'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def pascal_to_yolo(voc_root, voc_names, target_txt, phase):
    assert phase in ['train', 'val', 'test', 'trainval']
    annot_folder = os.path.join(voc_root, 'Annotations')
    image_folder = os.path.join(voc_root, 'JPEGImages')
    src_text = os.path.join(voc_root, 'ImageSets', 'Main', phase + '.txt')

    # read indices
    indices = []
    with open(src_text, 'r') as f:
        for line in f.readlines():
            indices.append(line.strip())

    pv = PascalVOC()
    target_fp = open(target_txt, 'wt')
    for index in indices:
        image_path = os.path.join(image_folder, index + '.jpg')
        annot_path = os.path.join(annot_folder, index + '.xml')
        fname, imgsize, boxes = pv.read_xml(annot_path)

        target_fp.write('%s' % image_path)
        for box in boxes:
            name = box['name']
            label = voc_names.index(name)
            rect = box['bndbox']
            target_fp.write(' %d,%d,%d,%d,%d' % (rect[0], rect[1], rect[2], rect[3], label))

        target_fp.write('\n')
    target_fp.close()


if __name__ == '__main__':
    if os_type == 'Windows':
        src_path = r'D:\data\VOCdevkit\VOC2007'
    else:
        src_path = '/Users/liyu/data/VOCdevkit/VOC2007'

    pascal_to_yolo(src_path, VOC_NAMES, abs_path('config/data/train.txt'), 'train')
