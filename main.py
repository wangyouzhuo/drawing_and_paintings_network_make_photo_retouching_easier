from train.train import *
import tensorflow as tf





if __name__ == "__main__":
    sess = tf.Session()
    train(sess=sess)