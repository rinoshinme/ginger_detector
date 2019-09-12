import numpy as np


def read_class_names(class_file):
    names = {}
    with open(class_file, 'r') as f:
        for idx, name in enumerate(f):
            names[idx] = name.strip()
    return names


def read_anchors(anchor_file):
    with open(anchor_file, 'r') as f:
        data = f.readline()
        values = [float(v) for v in data.split(',')]
    anchors = np.array(values)
    return anchors.reshape(3, 3, 2)


if __name__ == '__main__':
    anchor_file1 = '../config/anchors/baseline_anchors.txt'
