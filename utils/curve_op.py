import matplotlib.pyplot as plt
import math
from PIL import Image
from PIL import ImageEnhance
import numpy as np
from utils.load_image import show_image,load_image2numpy
from scipy import interpolate
import random


"""
 all pixels in images are between 0 and 1

"""


#   图像变黑变灰
def curve_dark_gray(x):
    """
    make image dark and gray
    :param x:
    :return:
    """
    # sample_x = [item/256.0 for item in [0 ,57,107,163,203,230,255]  ]
    # sample_y = [item/256.0 for item in [19,55,93 ,144,187,219,255] ]
    # func_a = np.polyfit(sample_x,sample_y,3)
    # func_b = np.poly1d(func_a)
    # # func = interpolate.interp1d(sample_x, sample_y, kind='cubic')
    # print("-------------------------------------")
    # print(func_b)
    y = 0.1895*pow(x,3)+0.1331*pow(x,2)+0.603*x+0.07363
    y = np.clip(y,a_min=0.0,a_max=1.0)
    return y

#   图像变亮
def curve_light(x):
    """
    make the image more lightful
    :param x:
    :return:
    """
    y = 0.005266 * pow(x, 3) - 0.1417 * pow(x, 2) + 1.136 * pow(x, 1) - 0.0003793
    y = np.clip(y,a_min=0.0,a_max=1.0)
    return y

#   图像变暗
def curve_dark(x):
    y = 0.03097*pow(x,3) + 0.1086*pow(x,2) + 0.8615*x + 0.001639
    y = np.clip(y,a_max=1.0,a_min=0.0)
    return y

#   图像对比度增高
def curve_contrast(x):
    y = -0.5574*pow(x,3) + 0.8379*pow(x,2) + 0.7185*x - 0.0001706
    y = np.clip(y,a_min=0.0,a_max=1.0)
    return y

#   图像减红加蓝
def curve_red_blue(x):
    beta = 0.7
    raw = x.copy()
    red  = x[:,:,0]
    blue = x[:,:,2]
    red = curve_dark(red)
    blue = curve_light(blue)
    raw[:,:,0] = red
    raw[:,:,2] = blue
    result = beta*raw + (1-beta)*x
    result = np.clip(result,a_min=0.0,a_max=1.0)
    return result

def curve_less_green(x):
    beta = 0.7
    raw = x.copy()
    green = x[:,:,1]
    green = curve_dark(green)
    raw[:,:,1] = green
    result = beta*raw + (1-beta)*x
    result = np.clip(result,a_min=0.0,a_max=1.0)
    return result

def curve_color_change(image,target_color,action):
    raw_image = image.copy()
    if target_color =='red':
        target_channel = image[:,:,0]
        channel_index = 0
    elif target_color == 'green':
        target_channel = image[:,:,1]
        channel_index = 1
    elif target_color == 'blue':
        target_channel = image[:,:,2]
        channel_index = 2
    else:
        print("target_color must be red or green or blue!")
        return
    if action == "left":
        target_channel = curve_dark(target_channel)
    elif action == 'right':
        target_channel = curve_light(target_channel)
    else:
        print("action must be left or right!")
        return
    raw_image[:, :, channel_index] = target_channel
    raw_image = np.clip(raw_image,a_min=0.0,a_max=1.0)
    return raw_image





if __name__ == '__main__':
    root_path = '/home/wyz/PycharmProjects/deep_drawing_and_paintings_network_make_photo_retouching_easier/image/raw_image/'
    file = '14470426704.jpg'
    filepath = root_path+file

    image_np = load_image2numpy(filepath)

    show_image(image_np)

    for i in range(1):
        image_np = curve_red_blue(image_np)
    show_image(image_np)
    # #
    # x = [x/256.0 for x in range(0,256) ]
    #
    # # x = [float(x) for x in range(0,1,0.01)]
    # y = [curve_contrast(x_) for x_ in x]
    #
    # plt.plot(x,y)
    # plt.plot(x,x)
    # plt.show()


