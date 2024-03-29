import time
import datetime


class Timer(object):
    def __init__(self):
        self.init_time = time.time()
        self.total_time = 0.
        self.calls = 0.
        self.start_time = 0.
        self.diff = 0.
        # self.average_time = 0.
        self.remain_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        if average:
            return self.total_time / self.calls
        else:
            return self.diff

    def remain(self, iters, max_iters):
        if iters == 0:
            self.remain_time = 0
        else:
            self.remain_time = (time.time() - self.init_time) * (max_iters - iters) / iters
        return str(datetime.timedelta(seconds=int(self.remain_time)))
