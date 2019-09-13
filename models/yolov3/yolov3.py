import tensorflow as tf
import numpy as np
from backend.darknet53 import darknet53
from utils import layers
from utils.io import read_anchors
from utils import bbox_tf
from config import ANCHORS_FILE


class YoloV3(object):
    def __init__(self, inputs, num_classes, trainable=False):
        self.trainable = trainable
        self.num_classes = num_classes

        # set yolo parameters
        self.anchors = read_anchors(ANCHORS_FILE)
        self.strides = np.array([8, 16, 32])
        self.anchor_per_scale = 3
        self.iou_loss_thresh = 0.5
        self.upsample_method = 'resize'

        # build output tensors
        self.conv_lbbox, self.conv_mbbox, self.conv_sbbox = self._build_network(inputs)
        with tf.variable_scope('pred_sbbox'):
            self.pred_sbbox = self._decode(self.conv_sbbox, self.anchors[0], self.strides[0])
        with tf.variable_scope('pred_mbbox'):
            self.pred_mbbox = self._decode(self.conv_mbbox, self.anchors[1], self.strides[1])
        with tf.variable_scope('pred_lbbox'):
            self.pred_lbbox = self._decode(self.conv_lbbox, self.anchors[2], self.strides[2])

    def _build_network(self, inputs):
        route_1, route_2, input_data = darknet53(inputs, trainable=self.trainable)

        input_data = layers.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv52')
        input_data = layers.convolutional(input_data, (3, 3, 512, 1024), self.trainable, 'conv53')
        input_data = layers.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv54')
        input_data = layers.convolutional(input_data, (3, 3, 512, 1024), self.trainable, 'conv55')
        input_data = layers.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv56')

        # 5 + num_classes = 4(box) + 1(obj conf) + num_classes(class prob)
        conv_lobj_branch = layers.convolutional(input_data, (3, 3, 512, 1024), self.trainable, name='conv_lobj_branch')
        conv_lbbox = layers.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (self.num_classes + 5)),
                                          trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)

        input_data = layers.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv57')
        input_data = layers.upsample(input_data, name='upsample0', method=self.upsample_method)

        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)

        input_data = layers.convolutional(input_data, (1, 1, 768, 256), self.trainable, 'conv58')
        input_data = layers.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv59')
        input_data = layers.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv60')
        input_data = layers.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv61')
        input_data = layers.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv62')

        conv_mobj_branch = layers.convolutional(input_data, (3, 3, 256, 512), self.trainable, name='conv_mobj_branch')
        conv_mbbox = layers.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (self.num_classes + 5)),
                                          trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)

        input_data = layers.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv63')
        input_data = layers.upsample(input_data, name='upsample1', method=self.upsample_method)

        with tf.variable_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)

        input_data = layers.convolutional(input_data, (1, 1, 384, 128), self.trainable, 'conv64')
        input_data = layers.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv65')
        input_data = layers.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv66')
        input_data = layers.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv67')
        input_data = layers.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv68')

        conv_sobj_branch = layers.convolutional(input_data, (3, 3, 128, 256), self.trainable, name='conv_sobj_branch')
        conv_sbbox = layers.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (self.num_classes + 5)),
                                          trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)

        return conv_lbbox, conv_mbbox, conv_sbbox

    def _decode(self, conv_output, anchors, stride):
        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        anchor_per_scale = len(anchors)

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size,
                                               anchor_per_scale, 5 + self.num_classes))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5:]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors[tf.newaxis, tf.newaxis, tf.newaxis, :, :]) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    @staticmethod
    def focal(target, actual, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):
        """
        conv: raw output of feature layers
        pred: encoded conv output
        label: gt encoded label
        bboxes: gt encoded box
            ground truth values should be encoded in preprocessing
        """
        conv_shape = tf.shape(conv)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        input_size = stride * output_size  # size of the input image (416)
        conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                 self.anchor_per_scale, 5 + self.num_classes))
        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]
        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 4:5]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]  # 1 with box correspondence and 0 not
        label_prob = label[:, :, :, :, 5:]

        giou = tf.expand_dims(bbox_tf.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        iou = bbox_tf.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thresh, tf.float32)

        conf_focal = self.focal(respond_bbox, pred_conf)

        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

        return giou_loss, conf_loss, prob_loss

    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):
        with tf.name_scope('smaller_box_loss'):
            loss_sbbox = self.loss_layer(self.conv_sbbox, self.pred_sbbox,
                                         label_sbbox, true_sbbox,
                                         anchors=self.anchors[0], stride=self.strides[0])
        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(self.conv_mbbox, self.pred_mbbox, label_mbbox, true_mbbox,
                                         anchors=self.anchors[1], stride=self.strides[1])

        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox,
                                         anchors=self.anchors[2], stride=self.strides[2])

        with tf.name_scope('giou_loss'):
            giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('prob_loss'):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

        return giou_loss, conf_loss, prob_loss


if __name__ == '__main__':
    xinputs = tf.placeholder(shape=[2, 416, 416, 3], dtype=tf.float32)
    model = YoloV3(xinputs, num_classes=20, trainable=True)

    print(model.conv_sbbox)
    print(model.conv_mbbox)
    print(model.conv_lbbox)

    print(model.pred_sbbox)
    print(model.pred_mbbox)
    print(model.pred_lbbox)
