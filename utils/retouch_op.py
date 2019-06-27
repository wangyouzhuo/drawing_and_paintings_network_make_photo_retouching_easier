import pandas as pd
import numpy as np
import cv2
import os
import random
from utils.curve_op import *
from utils.color_op import *
from utils.load_image import *
from constant.constant import *


#-------------------------------------------------------hold保持不动-------------------------------------------
def hold(image):
    return image

# ------------------------第一组动作----5个---------调整画面 黑白色阶分布的op  beta代表op作用的力度-------------------------------

def light(image,beta=0.2):  # 整体变亮
    raw_image = image.copy()
    image = curve_light(image)
    result = beta*image + (1-beta)*raw_image
    result = np.clip(result,a_min=0.0,a_max=1.0)
    if check_histogram(result):
        return result
    else:
        return raw_image


def dark(image,beta=0.2):   # 整体变黑
    raw_image = image.copy()
    image = curve_dark(image)
    result = beta * image + (1 - beta) * raw_image
    result = np.clip(result,a_min=0.0,a_max=1.0)
    if check_histogram(result):
        return result
    else:
        return raw_image


def contrast(image,beta=0.2):  # 加对比
    raw_image = image.copy()
    image = curve_contrast(image)
    result = beta * image + (1 - beta) * raw_image
    result = np.clip(result,a_min=0.0,a_max=1.0)
    if check_histogram(result):
        return result
    else:
        return raw_image


def gray(image,beta=0.2):   # 降对比
    raw_image = image.copy()
    image = curve_dark_gray(image)
    result = beta * image + (1 - beta) * image
    result = np.clip(result,a_min=0.0,a_max=1.0)
    if check_histogram(result):
        return result
    else:
        return raw_image


def dark_more_dark(image,beta=0.2):   # 暗部更暗
    return image


#-------------------------第二组动作-----6个-------调节画面的 色相 beta代表作用的力度-------------------------------------


# 蓝色色相偏左
def hue_blue_hue_left(image,beta=0.2):
    result = update_specific_hue(image, target_color_hsv=blue, action='left', step_size=beta)
    if check_histogram(result):
        return result
    else:
        return image
# 蓝色色相偏右
def hue_blue_hue_right(image,beta=0.2):
    result = update_specific_hue(image, target_color_hsv=blue, action='right', step_size=beta)
    if check_histogram(result):
        return result
    else:
        return image
# 绿色色相偏左
def hue_green_hue_left(image,beta=0.2):
    result = update_specific_hue(image, target_color_hsv=green, action='left', step_size=beta)
    if check_histogram(result):
        return result
    else:
        return image
# 绿色色相偏右
def hue_green_hue_right(image,beta=0.2):
    result = update_specific_hue(image, target_color_hsv=green, action='right', step_size=beta)
    if check_histogram(result):
        return result
    else:
        return image
# 黄色色相偏左
def hue_yellow_hue_left(image,beta=0.2):
    result = update_specific_hue(image, target_color_hsv=yellow, action='left', step_size=beta)
    if check_histogram(result):
        return result
    else:
        return image
# 黄色色相偏右
def hue_yellow_hue_right(image,beta=0.2):
    result = update_specific_hue(image, target_color_hsv=yellow, action='right', step_size=beta)
    if check_histogram(result):
        return result
    else:
        return image



#----------------------------第三组动作-----8个----调节画面的 饱和 beta代表作用的力度-------------------------------------

# 整体加饱和
def glo_saturation_up(image,beta=0.2):
    raw_img = image.copy()
    image = update_global_saturation(img=image,action=0.3)
    result = beta * image + (1 - beta) * raw_img
    result = np.clip(result,a_min=0.0,a_max=1.0)
    if check_histogram(result):
        return result
    else:
        return image
# 整体降饱和
def glo_saturation_down(image,beta=0.2):
    raw_img = image.copy()
    image = update_global_saturation(img=image,action=-0.3)
    result = beta * image + (1 - beta) * raw_img
    result = np.clip(result,a_min=0.0,a_max=1.0)
    if check_histogram(result):
        return result
    else:
        return image

# 蓝色饱和增大
def sat_blue_up(image,beta=0.2):
    result = update_specific_saturation(image,target_color_hsv=blue,action='right',step_size=beta)
    if check_histogram(result):
        return result
    else:
        return image
# 蓝色饱和减小
def sat_blue_down(image,beta=0.2):
    result = update_specific_saturation(image,target_color_hsv=blue,action='left',step_size=beta)
    if check_histogram(result):
        return result
    else:
        return image

# 绿色饱和增大
def sat_green_up(image,beta=0.2):
    result = update_specific_saturation(image,target_color_hsv=green,action='right',step_size=beta)
    if check_histogram(result):
        return result
    else:
        return image
# 绿色饱和减小
def sat_green_down(image,beta=0.2):
    result = update_specific_saturation(image,target_color_hsv=green,action='left',step_size=beta)
    if check_histogram(result):
        return result
    else:
        return image


# 黄色饱和增大
def sat_yellow_up(image,beta=0.2):
    result = update_specific_saturation(image,target_color_hsv=yellow,action='right',step_size=beta)
    if check_histogram(result):
        return result
    else:
        return image
# 黄色饱和减小
def sat_yellow_down(image,beta=0.2):
    result = update_specific_saturation(image,target_color_hsv=yellow,action='left',step_size=beta)
    if check_histogram(result):
        return result
    else:
        return image


#---------------------------第四组动作-----6个-----调节画面的 色彩曲线（白平衡）beta代表作用的力度-------------------------------------


# 提亮蓝色曲线
def glo_blue_curve_light(image,beta = 0.4):
    result = curve_color_change(image,action='right',target_color='blue')
    result = beta*result + (1-beta)*image
    result = np.clip(result,a_max=1.0,a_min=0.0)
    if check_histogram(result):
        return result
    else:
        return image

# 压暗蓝色曲线
def glo_blue_curve_dark(image,beta = 0.4):
    result = curve_color_change(image,action='left',target_color='blue')
    result = beta*result + (1-beta)*image
    result = np.clip(result,a_max=1.0,a_min=0.0)
    if check_histogram(result):
        return result
    else:
        return image

# 提亮红色曲线
def glo_red_curve_light(image,beta = 0.4):
    result = curve_color_change(image,action='right',target_color='red')
    result = beta*result + (1-beta)*image
    result = np.clip(result,a_max=1.0,a_min=0.0)
    if check_histogram(result):
        return result
    else:
        return image
# 压暗红色曲线
def glo_red_curve_dark(image,beta = 0.4):
    result = curve_color_change(image,action='left',target_color='red')
    result = beta*result + (1-beta)*image
    result = np.clip(result,a_max=1.0,a_min=0.0)
    if check_histogram(result):
        return result
    else:
        return image

# 提亮绿色曲线
def glo_green_curve_light(image,beta = 0.4):
    result = curve_color_change(image,action='right',target_color='green')
    result = beta*result + (1-beta)*image
    result = np.clip(result,a_max=1.0,a_min=0.0)
    if check_histogram(result):
        return result
    else:
        return image
# 压暗绿色曲线
def glo_green_curve_dark(image,beta = 0.4):
    result = curve_color_change(image,action='left',target_color='green')
    result = beta*result + (1-beta)*image
    result = np.clip(result,a_max=1.0,a_min=0.0)
    if check_histogram(result):
        return result
    else:
        return image


# 4 actions in Master_action
master_action_list = [light,dark,contrast,gray,
                      # hold
                      ]


# 18 actions in Sub_action
sub_action_list = [
                     hue_blue_hue_left   ,hue_blue_hue_right  ,
                     hue_green_hue_left  ,hue_green_hue_right ,
                     hue_yellow_hue_left ,hue_yellow_hue_right,
                     glo_saturation_up   ,glo_saturation_down,
                     sat_blue_down       ,sat_blue_up,
                     sat_green_down      ,sat_green_up,
                     sat_yellow_down     ,sat_yellow_up,
                     glo_blue_curve_dark ,glo_blue_curve_light,
                     glo_green_curve_dark,glo_green_curve_light,
                     glo_red_curve_dark  ,glo_red_curve_light,
                     # hold
                    ]

gray_action_list = [light,dark,contrast,gray,dark_more_dark]

hue_action_list = [hue_blue_hue_left   , hue_blue_hue_right  , hue_green_hue_left,
                   hue_green_hue_right , hue_yellow_hue_left , hue_yellow_hue_right]

saturation_action_list = [glo_saturation_up,glo_saturation_down,sat_blue_up,
                          sat_blue_down,sat_green_up,sat_green_down,sat_yellow_up,sat_yellow_down]

whitebalance_action_list = [glo_blue_curve_light,glo_blue_curve_dark,glo_red_curve_light,
                            glo_red_curve_dark,glo_green_curve_light,glo_green_curve_dark]