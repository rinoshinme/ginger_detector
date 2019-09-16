import time


def get_time_stamp():
    return time.strftime('%Y%m%d|%H:%M:%S', time.localtime(time.time()))


if __name__ == '__main__':
    print(get_time_stamp())
