import tensorflow as tf
import numpy as np
from config import abs_path
# from utils.io import read_class_names
import time
import os
import shutil
from dataset.dataset_yolov3 import Dataset
from models.yolov3.yolov3 import YoloV3

# Training Parameters
TRAIN_DATASET_PATH = abs_path('config/data/train.txt')
TEST_DATASET_PATH = abs_path('config/data/test.txt')
BATCH_SIZE = 2
NUM_CLASSES = 20
LEARN_RATE_INIT = 1.0e-4
LEARN_RATE_END = 1.0e-6
WARMUP_EPOCHS = 2
FIRST_STAGE_EPOCHS = 20
SECONDS_STAGE_EPOCHS = 30
DATA_AUG = True
INITIAL_WEIGHT = './'


class YoloTrain(object):
    def __init__(self):
        # define parameters
        self.anchor_per_scale = 3
        # self.classes = read_class_names(CLASS_NAME_FILE)
        self.num_classes = NUM_CLASSES

        self.learn_rate_init = LEARN_RATE_INIT
        self.learn_rate_end = LEARN_RATE_END

        # training strategy parameters
        self.first_stage_epochs = FIRST_STAGE_EPOCHS
        self.second_stage_epochs = SECONDS_STAGE_EPOCHS
        self.warmup_periods = WARMUP_EPOCHS
        self.initial_weight = INITIAL_WEIGHT
        self.time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_avg_decay = 0.9995
        # self.max_bbox_per_scale = 150
        self.train_logdir = abs_path('logs/')
        self.trainset = Dataset('train', TRAIN_DATASET_PATH, BATCH_SIZE)
        self.testset = Dataset('test', TEST_DATASET_PATH, BATCH_SIZE)
        self.steps_per_period = len(self.trainset)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        # create network
        with tf.name_scope('define_inputs'):
            # no shape is defined for multi-scale training.
            self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.label_sbbox = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
            self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.trainable = tf.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope('define_loss'):
            self.model = YoloV3(self.input_data, self.trainable)
            self.net_var = tf.global_variables()
            self.giou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(
                self.label_sbbox,  self.label_mbbox,  self.label_lbbox,
                self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
            self.loss = self.giou_loss + self.conf_loss + self.prob_loss

        with tf.name_scope('learn'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                       dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant((self.first_stage_epochs + self.second_stage_epochs) * self.steps_per_period,
                                      dtype=tf.float64, name='train_steps')
            self.learn_rate = tf.cond(
                pred=self.global_step / warmup_steps * self.learn_rate_init,
                true_fn = lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) * (
                    1 + tf.cos((self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi)))
            global_step_update = tf.assign_add(self.global_step, 1.0)

        with tf.name_scope('define_weight_decay'):
            moving_avg = tf.train.ExponentialMovingAverage(self.moving_avg_decay).apply(tf.trainable_variables())

        with tf.name_scope("define_first_stage_train"):
            self.first_stage_trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                    self.first_stage_trainable_var_list.append(var)

            first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=self.first_stage_trainable_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_avg]):
                        self.train_op_with_frozen_variables = tf.no_op()

        with tf.name_scope("define_second_stage_train"):
            second_stage_trainable_var_list = tf.trainable_variables()
            second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=second_stage_trainable_var_list)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_avg]):
                        self.train_op_with_all_variables = tf.no_op()

        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(self.net_var)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        with tf.name_scope('summary'):
            tf.summary.scalar('learn_rate', self.learn_rate)
            tf.summary.scalar('giou_loss', self.giou_loss)
            tf.summary.scalar('conf_loss', self.conf_loss)
            tf.summary.scalar('prob_loss', self.prob_loss)
            tf.summary.scalar('total_loss', self.loss)

            logdir = abs_path('log/')
            if os.path.exists(logdir):
                shutil.rmtree(logdir)
            os.mkdir(logdir)
            self.write_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(logdir, graph=self.sess.graph)

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        try:
            print('=> Restoring weights from: %s ...' % self.initial_weight)
            self.loader.restore(self.sess, self.initial_weight)
        except:
            print('=> %s does not exists !!!' % self.initial_weight)
            print('=> Now it starts to train YoloV3 from scratch...')
            self.first_stage_epochs = 0

        for epoch in range(1, 1 + self.first_stage_epochs + self.second_stage_epochs):
            if epoch <= self.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables
            else:
                train_op = self.train_op_with_all_variables

            train_epoch_loss = []
            test_epoch_loss = []
            for train_data in self.trainset:
                _, summary, train_step_loss ,global_step_val = self.sess.run(
                    [train_op, self.write_op, self.loss, self.global_step],
                    feed_dict={
                        self.input_data: train_data[0],
                        self.label_sbbox: train_data[1],
                        self.label_mbbox: train_data[2],
                        self.label_lbbox: train_data[3],
                        self.true_sbboxes: train_data[4],
                        self.true_mbboxes: train_data[5],
                        self.true_lbboxes: train_data[6],
                        self.trainable: True})

                train_epoch_loss.append(train_step_loss)
                self.summary_writer.add_summary(summary, global_step_val)

            for test_data in self.testset:
                test_step_loss = self.sess.run(self.loss, feed_dict={
                    self.input_data: test_data[0],
                    self.label_sbbox: test_data[1],
                    self.label_mbbox: test_data[2],
                    self.label_lbbox: test_data[3],
                    self.true_sbboxes: test_data[4],
                    self.true_mbboxes: test_data[5],
                    self.true_lbboxes: test_data[6],
                    self.trainable: False})

                test_epoch_loss.append(test_step_loss)

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            ckpt_file = abs_path("checkpoint/yolov3_test_loss=%.4f.ckpt" % test_epoch_loss)
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                  % (epoch, log_time, float(train_epoch_loss), float(test_epoch_loss), ckpt_file))
            self.saver.save(self.sess, ckpt_file, global_step=epoch)


if __name__ == '__main__':
    trainer = YoloTrain()
    trainer.train()
