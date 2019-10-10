import tensorflow as tf
import numpy as np
from constant.constant import *




def import_weight(file_path):
    print(file_path)
    w = np.load(file_path,encoding='latin1').item()
    return w


def model_vgg(input, model_path):
    imported_weight = import_weight(VGG_PATH)
    mean = tf.constant([R_MEAN, G_MEAN, B_Mean], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    input = tf.clip_by_value((input*255.0),clip_value_max=255.0,clip_value_min=0.0) - mean

    conv1_1_weights = tf.Variable(imported_weight['conv1_1'][0], name='conv1_1_weights',trainable=False)
    conv1_1_biases  = tf.Variable(imported_weight['conv1_1'][1], name='conv1_1_biases',trainable=False)
    conv1_2_weights = tf.Variable(imported_weight['conv1_2'][0], name='conv1_2_weights',trainable=False)
    conv1_2_biases  = tf.Variable(imported_weight['conv1_2'][1], name='conv1_2_biases',trainable=False)

    conv1_1 = tf.nn.conv2d(input, conv1_1_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu1_1 = tf.nn.relu(tf.nn.bias_add(conv1_1, conv1_1_biases))
    conv1_2 = tf.nn.conv2d(relu1_1, conv1_2_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu1_2 = tf.nn.relu(tf.nn.bias_add(conv1_2, conv1_2_biases))
    pool1 = tf.nn.max_pool(relu1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv2_1_weights = tf.Variable(imported_weight['conv2_1'][0], name='conv2_1_weights',trainable=False)
    conv2_1_biases = tf.Variable(imported_weight['conv2_1'][1], name='conv2_1_biases',trainable=False)
    conv2_2_weights = tf.Variable(imported_weight['conv2_2'][0], name='conv2_2_weights',trainable=False)
    conv2_2_biases = tf.Variable(imported_weight['conv2_2'][1], name='conv2_2_biases',trainable=False)

    conv2_1 = tf.nn.conv2d(pool1, conv2_1_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu2_1 = tf.nn.relu(tf.nn.bias_add(conv2_1, conv2_1_biases))
    conv2_2 = tf.nn.conv2d(relu2_1, conv2_2_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu2_2 = tf.nn.relu(tf.nn.bias_add(conv2_2, conv2_2_biases))
    pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # conv3_1 128x3x3 filters -> 256, stride 1
    # conv3_2 256x3x3 filters -> 256, stride 1
    # conv3_3 128x3x3 filters -> 256, stride 1
    # maxpool (2,2) (2,2)
    conv3_1_weights = tf.Variable(imported_weight['conv3_1'][0], name='conv3_1_weights',trainable=False)
    conv3_1_biases = tf.Variable(imported_weight['conv3_1'][1], name='conv3_1_biases',trainable=False)
    conv3_2_weights = tf.Variable(imported_weight['conv3_2'][0], name='conv3_2_weights',trainable=False)
    conv3_2_biases = tf.Variable(imported_weight['conv3_2'][1], name='conv3_2_biases',trainable=False)
    conv3_3_weights = tf.Variable(imported_weight['conv3_3'][0], name='conv3_3_weights',trainable=False)
    conv3_3_biases = tf.Variable(imported_weight['conv3_3'][1], name='conv3_3_biases',trainable=False)

    conv3_1 = tf.nn.conv2d(pool2, conv3_1_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu3_1 = tf.nn.relu(tf.nn.bias_add(conv3_1, conv3_1_biases))
    conv3_2 = tf.nn.conv2d(relu3_1, conv3_2_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu3_2 = tf.nn.relu(tf.nn.bias_add(conv3_2, conv3_2_biases))
    conv3_3 = tf.nn.conv2d(relu3_2, conv3_3_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu3_3 = tf.nn.relu(tf.nn.bias_add(conv3_3, conv3_3_biases))
    pool3 = tf.nn.max_pool(relu3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv4_1 256x3x3 filters -> 512, stride 1
    # conv4_2 512x3x3 filters -> 512, stride 1
    # conv4_3 512x3x3 filters -> 512, stride 1
    # maxpool (2,2) (2,2)
    conv4_1_weights = tf.Variable(imported_weight['conv4_1'][0], name='conv4_1_weights',trainable=False)
    conv4_1_biases = tf.Variable(imported_weight['conv4_1'][1], name='conv4_1_biases',trainable=False)
    conv4_2_weights = tf.Variable(imported_weight['conv4_2'][0], name='conv4_2_weights',trainable=False)
    conv4_2_biases = tf.Variable(imported_weight['conv4_2'][1], name='conv4_2_biases',trainable=False)
    conv4_3_weights = tf.Variable(imported_weight['conv4_3'][0], name='conv4_3_weights',trainable=False)
    conv4_3_biases = tf.Variable(imported_weight['conv4_3'][1], name='conv4_3_biases',trainable=False)
    conv4_1 = tf.nn.conv2d(pool3, conv4_1_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu4_1 = tf.nn.relu(tf.nn.bias_add(conv4_1, conv4_1_biases))
    conv4_2 = tf.nn.conv2d(relu4_1, conv4_2_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu4_2 = tf.nn.relu(tf.nn.bias_add(conv4_2, conv4_2_biases))
    conv4_3 = tf.nn.conv2d(relu4_2, conv4_3_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu4_3 = tf.nn.relu(tf.nn.bias_add(conv4_3, conv4_3_biases))
    pool4 = tf.nn.max_pool(relu4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv5_1 512x3x3 filters -> 512, stride 1
    # conv5_2 512x3x3 filters -> 512, stride 1
    # conv5_3 512x3x3 filters -> 512, stride 1
    # maxpool (2,2) (2,2)
    conv5_1_weights = tf.Variable(imported_weight['conv5_1'][0], name='conv5_1_weights',trainable=False)
    conv5_1_biases = tf.Variable(imported_weight['conv5_1'][1], name='conv5_1_biases',trainable=False)
    conv5_2_weights = tf.Variable(imported_weight['conv5_2'][0], name='conv5_2_weights',trainable=False)
    conv5_2_biases = tf.Variable(imported_weight['conv5_2'][1], name='conv5_2_biases',trainable=False)
    conv5_3_weights = tf.Variable(imported_weight['conv5_3'][0], name='conv5_3_weights',trainable=False)
    conv5_3_biases = tf.Variable(imported_weight['conv5_3'][1], name='conv5_3_biases',trainable=False)

    conv5_1 = tf.nn.conv2d(pool4, conv5_1_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu5_1 = tf.nn.relu(tf.nn.bias_add(conv5_1, conv5_1_biases))
    conv5_2 = tf.nn.conv2d(relu5_1, conv5_2_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu5_2 = tf.nn.relu(tf.nn.bias_add(conv5_2, conv5_2_biases))
    conv5_3 = tf.nn.conv2d(relu5_2, conv5_3_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu5_3 = tf.nn.relu(tf.nn.bias_add(conv5_3, conv5_3_biases))
    pool5 = tf.nn.max_pool(relu5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    shape = pool5.get_shape().as_list()
    pool5 = tf.reshape(pool5, [-1, shape[1] * shape[2] * shape[3]])

    fc6_weights = tf.Variable(imported_weight['fc6'][0], name='fc6_weights',trainable=False)
    fc6_biases = tf.Variable(imported_weight['fc6'][1], name='fc6_biases',trainable=False)
    fc6 = tf.nn.bias_add(tf.matmul(pool5, fc6_weights), fc6_biases)
    fc6 = tf.nn.l2_normalize(fc6, -1, epsilon=1)

    weights = []
    weights.append(conv1_1_weights)
    weights.append(conv1_1_biases)
    weights.append(conv1_2_weights)
    weights.append(conv1_2_biases)
    weights.append(conv2_1_weights)
    weights.append(conv2_1_biases)
    weights.append(conv2_2_weights)
    weights.append(conv2_2_biases)
    weights.append(conv3_1_weights)
    weights.append(conv3_1_biases)
    weights.append(conv3_2_weights)
    weights.append(conv3_2_biases)
    weights.append(conv3_3_weights)
    weights.append(conv3_3_biases)
    weights.append(conv4_1_weights)
    weights.append(conv4_1_biases)
    weights.append(conv4_2_weights)
    weights.append(conv4_2_biases)
    weights.append(conv4_3_weights)
    weights.append(conv4_3_biases)
    weights.append(conv5_1_weights)
    weights.append(conv5_1_biases)
    weights.append(conv5_2_weights)
    weights.append(conv5_2_biases)
    weights.append(conv5_3_weights)
    weights.append(conv5_3_biases)
    weights.append(fc6_weights)
    weights.append(fc6_biases)

    return fc6, weights
