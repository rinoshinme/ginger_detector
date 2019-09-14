import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import datetime
from utils.timer import Timer
from dataset.dataset_yolo import Dataset
from models.yolo.yolo import Yolo

WEIGHT_FILE = None
LEARNING_RATE = 0.000
DECAY_STEPS = 30000
DECAY_RATE = 0.1
STAIRCASE = True
BATCH_SIZE = 45
MAX_ITER = 15000
SUMMARY_ITER = 10
SAVE_ITER = 1000


class YoloTrain(object):
    def __init__(self, net, data):
        self.net = net
        self.data = data
        self.weight_file = WEIGHT_FILE

        # training parameter
        self.max_iter = 10000
        self.initial_learning_rate = LEARNING_RATE
        self.decay_steps = DECAY_STEPS
        self.decay_rate = DECAY_RATE
        self.staircase = STAIRCASE
        self.summary_iter = SUMMARY_ITER
        self.save_iter = SAVE_ITER

        self.output_dir = './logs'

        # training perparation
        self.variable_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.ckpt_file = os.path.join(self.output_dir, 'yolo')
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)

        self.global_step = tf.train.create_global_step()
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
        self.optimiser = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.train_op = slim.learning.create_train_op(
            self.net.total_loss, self.optimiser, global_step=self.global_step)

        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        if self.weight_file is not None:
            print('Restoring weights from: {}'.format(self.weight_file))
            self.saver.restore(self.sess, self.weight_file)
        self.writer.add_graph(self.sess.graph)

    def train(self):
        train_timer = Timer()
        load_timer = Timer()

        for step in range(1, self.max_iter + 1):

            load_timer.tic()
            images, labels = self.data.get()
            load_timer.toc()
            feed_dict = {self.net.images: images,
                         self.net.labels: labels}

            if step % self.summary_iter == 0:
                if step % (self.summary_iter * 10) == 0:

                    train_timer.tic()
                    summary_str, loss, _ = self.sess.run(
                        [self.summary_op, self.net.total_loss, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                    # log_str = '''{} Epoch: {}, Step: {}, Learning rate: {},'''
                    # ''' Loss: {:5.3f}\nSpeed: {:.3f}s/iter,'''
                    # '''' Load: {:.3f}s/iter, Remain: {}'''.format(
                    #     datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    #     self.data.epoch,
                    #     int(step),
                    #     round(self.learning_rate.eval(session=self.sess), 6),
                    #     loss,
                    #     train_timer.average_time,
                    #     load_timer.average_time,
                    #     train_timer.remain(step, self.max_iter))
                    # print(log_str)

                else:
                    train_timer.tic()
                    summary_str, _ = self.sess.run(
                        [self.summary_op, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                self.writer.add_summary(summary_str, step)

            else:
                train_timer.tic()
                self.sess.run(self.train_op, feed_dict=feed_dict)
                train_timer.toc()

            if step % self.save_iter == 0:
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    self.output_dir))
                self.saver.save(
                    self.sess, self.ckpt_file, global_step=self.global_step)


if __name__ == '__main__':
    net = Yolo(num_classes=20, is_training=True)
    data_path = None
    phase = None
    batch_size = 0
    class_names = None
    data = Dataset(data_path, phase, batch_size, class_names)
    trainer = YoloTrain(net, data)
    trainer.train()
