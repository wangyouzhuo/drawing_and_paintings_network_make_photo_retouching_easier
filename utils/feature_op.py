import tensorflow as tf
import numpy as np
from utils.train_op import distribution_color,distribution_gray

def get_color_feature(image):
    hist_one,hist_two,hist_three = distribution_color(image)
    color_feature = np.concatenate([hist_one,hist_two,hist_three],axis=0)
    return color_feature


def get_gray_feature(image):
    gray_feature = distribution_gray(image)
    return gray_feature


