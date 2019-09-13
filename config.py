"""
Global configurations
"""
import os
import platform

# Windows, Darwin, Linux?
os_type = platform.system()

if os_type == 'Windows':
    PROJECT_DIR = r'D:\projects\ginger_detector'
elif os_type == 'Darwin':
    PROJECT_DIR = '/Users/liyu/Desktop/Projects/ginger_detector'
else:
    raise ValueError('os type not supported yet')


def abs_path(path):
    return os.path.join(PROJECT_DIR, path)


ANCHORS_FILE = abs_path('config/anchors/baseline_anchors.txt')


if __name__ == '__main__':
    print(platform.system())
