import sys
sys.path.append('/home/wyz/PycharmProjects/deep_drawing_and_paintings_network_make_photo_retouching_easier/')
import numpy as np
import pandas as pd
from PIL import Image
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
from constant.constant import *


def resize_image(image,outpath,size=(224,224)):
    resized_img = cv2.resize(image,size)
    save_image(resized_img,filepath=outpath)



def load_image2numpy(path):
    img = np.array(Image.open(path))
    img = img/256.0
    return img

def show_image(image,info='SHOW'):
    image = image*255.0
    image = np.clip(image,a_max=255,a_min=0)
    image = Image.fromarray(image.astype(np.uint8))
    image.show(title=info+'.png')
    return

def save_image(image,filepath):
    image = image*255.0
    image = np.clip(image, a_max=255, a_min=0)
    image = Image.fromarray(image.astype(np.uint8))
    image.save(filepath)
    return

def show_histogram(image,whe_close=False):
    plt.figure(figsize=(20, 5))
    image = (image*255.0).astype(np.uint8)
    color = ('r', 'g', 'b')
    for i, col in enumerate(color):
        histr = cv2.calcHist([image], [i], None, [256], [0, 256])
        histr = histr/(image.shape[0]*image.shape[1])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()
    if whe_close == True:
        plt.pause(0.5)
        plt.close()


def count_pixel(channel_pixel):
    # channel between 0~255
    count_left,count_right = 0,0
    if channel_pixel <= 10:
        count_left = 1
    if channel_pixel >= 240:
        count_right = 1
    return count_left,count_right


def count_blue_pixel(channel_pixel):
    count_left, count_right = 0, 0
    if channel_pixel <= 3:
        count_left = 1
    if channel_pixel >= 240:
        count_right = 1
    return count_left, count_right



def check_histogram(image,che_sat = False):
    result_dict = {}
    area = image.shape[0]*image.shape[1]
    image = (image*255.0).astype(np.uint8)
    r_channel,g_channel,b_channel = image[:,:,0],image[:,:,1],image[:,:,2]
    distance = np.sum(np.square((r_channel-g_channel)))+np.sum(np.square((r_channel-b_channel)))+np.sum(np.square((b_channel-g_channel)))
    np_count_pixel = np.vectorize(count_pixel)
    np_count_blue_pixel = np.vectorize(count_blue_pixel)


    r_left,r_right = np_count_pixel(r_channel)
    r_left,r_right = np.sum(r_left),np.sum(r_right)
    result_dict['r_left'],result_dict['r_right'] = r_left * 1.0 / area, r_right * 1.0 / area

    g_left,g_right = np_count_pixel(g_channel)
    g_left,g_right = np.sum(g_left),np.sum(g_right)
    result_dict['g_left'],result_dict['g_right'] = g_left * 1.0 / area, g_right * 1.0 / area

    _,b_right = np_count_pixel(b_channel)
    b_right = np.sum(b_right)
    b_left = np_count_blue_pixel(b_channel)
    b_left = np.sum(b_left)
    result_dict['b_left'],result_dict['b_right'] = b_left * 1.0 / area, b_right * 1.0 / area

    result_dict['all_left'],result_dict['all_right'] = \
        ((r_left+g_left+b_left)*1.0)/(3*area), ((r_right+b_right+g_right )*1.0)/(3*area)
    for key in result_dict.keys():
        result_dict[key] = round(result_dict[key],4)
        # return result_dict
    result_list = [ result_dict['r_left'], result_dict['r_right'], result_dict['g_left'], result_dict['g_right'], result_dict['b_right'] ]
    result = np.array([int(item < 0.1) for item in result_list])
    if che_sat:
        if np.sum(result) == 5 and result_dict['b_left']<0.8 and distance>45385788: # 45385788这个值根据实验多次得出
            return True
        else:
            return False
    else:
        if np.sum(result) == 5 and result_dict['b_left']<0.8 :
            return True
        else:
            return False



if __name__ == '__main__':
    file_path = root + 'prepare_data_pair/raw_data/'
    r_left_min,r_right_min,g_left_min,g_right_min,b_left_min,b_right_min,all_left_min,all_right_min = 1,1,1,1,1,1,1,1
    attributes_list = ['r_left','r_right','g_left','g_right','b_left','b_right','all_left','all_right']
    target_list = [r_left_min,r_right_min,g_left_min,g_right_min,b_left_min,b_right_min,all_left_min,all_right_min]
    count = 0
    for pic_name in os.listdir(file_path):
        count = count + 1
        print(count)
        image_path = file_path + pic_name
        image = load_image2numpy(image_path)
        dict = check_histogram(image)
        current_list = [dict['r_left'],dict['r_right'],dict['g_left']  ,dict['g_right'],
                        dict['b_left'],dict['b_right'],
                        # dict['all_left'],dict['all_right']
                        ]
        result = np.array([ int(item<0.1) for item in current_list])
        error = np.sum(image[300:400, 300:400, 0] - image[300:400, 300:400, 1])  # 排除黑白照片

        if np.sum(result) == 6 and error > 0:

        # print("-------------------------------------------")
            print(pic_name)
        # print(dict)
            show_image(image,info='fuck')
            show_histogram(image,whe_close=False)
        # print("For %s : y or n?"%(pic_name))
        # whether = input()
        # # print(sys.argv)
        # if  whether == 'n':
        #     for i in range(len(target_list)):
        #         target_list[i] = min([target_list[i],current_list[i]])
        #         print(attributes_list[i],'  ',target_list[i])
        # elif whether == 'y':
        #     pass
        # else:
        #     print("Input Error :must y or n")









