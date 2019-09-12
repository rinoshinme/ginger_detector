"""
Global configurations
"""
import os
import platform

# Windows, Darwin, Linux?
os_type = platform.system()

PROJECT_DIR = r'D:\projects\ginger_detector'


def abs_path(path):
    return os.path.join(PROJECT_DIR, path)


if __name__ == '__main__':
    print(platform.system())
