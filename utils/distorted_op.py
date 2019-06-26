import os
import random
from utils.curve_op import *
from utils.color_op import *
from utils.load_image import *
from constant.constant import *


"""
    调整特定颜色的色相
    def update_specific_hue(image,target_color,color_threshold,action,step_size):
        
    调整特定颜色的饱和度
    def update_specific_saturation(image, target_color, color_threshold, action, step_size):
        
    调整全局饱和度
    def update_global_saturation(img, action):
    
    图像变黑变灰
    def curve_dark_gray(x, beta=0.6)
    
    图像变亮
    def curve_light(x, beta=0.5):
        
    图像变暗
    def curve_dark(x, beta=0.3):
        
    图像对比度增高
    def curve_contrast(x, beta=0.4):
        
    图像减红加蓝
    def curve_red_blue(x):
    
    需要调整的几个主要色彩 
      [79 95 59]   [53 64 34]   green   
      [173 86 32]  [109 67 19]  yellow    
      [79 102 154] [56 110 174] blue   
      [212 149 33] [210 157 61] sun 
      [208 185 169]             skin
      [111 88 132]              purple
"""

def normalized(target_color_list):
    result_list = target_color_list.copy()
    for i in range(len(target_color_list)):
        for  j in range(3):
            result_list[i][j] = target_color_list[i][j]/255.0
    return result_list

# ---------------------------------curve_op--------------------------------
#  全局变暗
def global_dark(image):
    return curve_dark(image)

#  全局变黑
def global_gray(image):
    return curve_dark_gray(image)

#  全局变亮
def global_light(image):
    return curve_light(image)

#  全局加对比
def global_contrast(image):
    return curve_contrast(image)

#  全局减红加蓝
def global_red_blue(image):
    return curve_red_blue(image)

# 全局减少绿色
def global_green_less(image):
    return curve_less_green(image)

# --------------------------------color_op--------------------------------

# 全局加饱和
def global_saturation_up(image,action=0.3):
    if action>1 or action < -1:
        raise NameError("action must between -1 and 1")
    return update_global_saturation(image,action)

# 全局减饱和
def global_saturation_down(image,action=-0.2):
    if action>1 or action < -1:
        raise NameError("action must between -1 and 1")
    return update_global_saturation(image,action)

# 调节特定颜色色相
def hue_red_left(image,step_size):
    return update_specific_hue(image,target_color_hsv=red,action='left',step_size=step_size)

def hue_red_right(image,step_size):
    return update_specific_hue(image, target_color_hsv=red, action='right', step_size=step_size)

def hue_green_left(image,step_size):
    return update_specific_hue(image,target_color_hsv=green,action='left',step_size=step_size)

def hue_green_right(image,step_size):
    return update_specific_hue(image, target_color_hsv=green, action='right', step_size=step_size)

def hue_yellow_left(image,step_size):
    return update_specific_hue(image,target_color_hsv=yellow,action='left',step_size=step_size)

def hue_yellow_right(image,step_size):
    return update_specific_hue(image,target_color_hsv=yellow,action='right',step_size=step_size)

def hue_blue_left(image,step_size):
    return update_specific_hue(image,target_color_hsv=blue,action='left',step_size=step_size)

def hue_blue_right(image,step_size):
    return update_specific_hue(image,target_color_hsv=blue,action='right',step_size=step_size)

def hue_sun_left(image,step_size):
    return update_specific_hue(image,target_color_hsv=sun,action='left',step_size=step_size)

def hue_sun_right(image,step_size):
    return update_specific_hue(image,target_color_hsv=sun,action='right',step_size=step_size)

def hue_purple_left(image,step_size):
    return update_specific_hue(image,target_color_hsv=purple,action='left',step_size=step_size)

def hue_purple_right(image,step_size):
    return update_specific_hue(image,target_color_hsv=purple,action='right',step_size=step_size)

def hue_skin_left(image,step_size):
    return update_specific_hue(image,target_color_hsv=skin,action='left',step_size=step_size)

def hue_skin_right(image,step_size):
    return update_specific_hue(image,target_color_hsv=skin,action='right',step_size=step_size)


#调节特定颜色饱和度
saturation_step_size = 0.2

def saturation_red_strong(image,stepsize):
    return update_specific_saturation(image,target_color_hsv=red,action='right',step_size=stepsize)

def saturation_red_thin(image,stepsize):
    return update_specific_saturation(image,target_color_hsv=red,action='left',step_size=stepsize)

def saturation_green_strong(image,stepsize):
    return update_specific_saturation(image,target_color_hsv=green,action='right',step_size=stepsize)

def saturation_green_thin(image,stepsize):
    return update_specific_saturation(image,target_color_hsv=green,action='left',step_size=stepsize)

def saturation_yellow_strong(image,stepsize):
    return update_specific_saturation(image,target_color_hsv=yellow,action='right',step_size=stepsize)

def saturation_yellow_thin(image,stepsize):
    return update_specific_saturation(image,target_color_hsv=yellow,action='left',step_size=stepsize)

def saturation_blue_strong(image,stepsize):
    return update_specific_saturation(image,target_color_hsv=blue,action='right',step_size=stepsize)

def saturation_blue_thin(image,stepsize):
    return update_specific_saturation(image,target_color_hsv=blue,action='left',step_size=stepsize)

def saturation_sun_strong(image,stepsize):
    return update_specific_saturation(image,target_color_hsv=sun,action='right',step_size=stepsize)

def saturation_sun_thin(image,stepsize):
    return update_specific_saturation(image,target_color_hsv=sun,action='left',step_size=stepsize)

def saturation_purple_strong(image,stepsize):
    return update_specific_saturation(image,target_color_hsv=purple,action='right',step_size=stepsize)

def saturation_purple_thin(image,stepsize):
    return update_specific_saturation(image,target_color_hsv=purple,action='left',step_size=stepsize)

def saturation_skin_strong(image,stepsize):
    return update_specific_saturation(image,target_color_hsv=skin,action='right',step_size=stepsize)

def saturation_skin_thin(image,stepsize):
    return update_specific_saturation(image,target_color_hsv=skin,action='left',step_size=stepsize)


global_action_list = [
                        global_dark,
                        global_gray,
                        global_light,
                        global_contrast,
                        global_red_blue,
                        global_saturation_up,
                        global_saturation_down
                    ]

hue_action_list =   [
                        hue_green_left,
                        hue_green_right,
                        hue_yellow_left,
                        hue_yellow_right,
                        hue_blue_left,
                        hue_blue_right,
                        hue_sun_left,
                        hue_sun_right,
                        hue_purple_left,
                        hue_purple_right,
                        hue_skin_left,
                        hue_skin_right
                    ]

saturation_action_list = [
                        saturation_green_strong,
                        saturation_green_thin,
                        saturation_yellow_strong,
                        saturation_yellow_thin,
                        saturation_blue_strong,
                        saturation_blue_thin,
                        saturation_sun_strong,
                        saturation_sun_thin,
                        saturation_purple_strong,
                        saturation_purple_thin,
                        saturation_skin_strong,
                        saturation_skin_thin
                        ]

action_list = [
                    global_dark,                #1
                    global_gray,                #2
                    global_light,               #3
                    global_contrast,            #4
                    global_red_blue,            #5
                    global_saturation_up,       #6
                    global_saturation_down,     #7
                    hue_green_left,             #8
                    hue_green_right,            #9
                    hue_yellow_left,            #10
                    hue_yellow_right,           #11
                    hue_blue_left,              #12
                    hue_blue_right,             #13
                    hue_sun_left,               #14
                    hue_sun_right,              #15
                    hue_purple_left,            #16
                    hue_purple_right,           #17
                    hue_skin_left,              #18
                    hue_skin_right,             #19
                    saturation_green_strong,    #20
                    saturation_green_thin,      #21
                    saturation_yellow_strong,   #22
                    saturation_yellow_thin,     #23
                    saturation_blue_strong,     #24
                    saturation_blue_thin,       #25
                    saturation_sun_strong,      #26
                    saturation_sun_thin,        #27
                    saturation_purple_strong,   #28
                    saturation_purple_thin,     #29
                    saturation_skin_strong,     #30
                    saturation_skin_thin        #31
            ]


def distorted_landscapes(image):
    image = global_red_blue(image=image)
    image = saturation_green_thin(image=image, stepsize=0.4)
    image = saturation_blue_thin(image=image, stepsize=0.5)
    image = hue_green_left(image=image, step_size=0.4)
    image = hue_blue_right(image=image, step_size=0.6)
    image = hue_yellow_right(image=image, step_size=0.6)
    image = global_gray(image=image)
    image = global_saturation_down(image=image, action=-0.2)
    # image = hue_skin_right(image,step_size=0.01)
    image = global_light(image)
    image = global_light(image)
    image = global_gray(image=image)
    image = global_dark(image=image)
    image = saturation_blue_thin(image=image, stepsize=0.4)
    image = global_green_less(image=image)
    image = global_light(image=image)
    image = global_light(image=image)
    image = global_light(image=image)



    return image


def distorted_indoor(image):
    image = global_red_blue(image=image)
    # image = global_red_blue(image=image)
    image = global_saturation_down(image=image, action=-0.3)
    image = global_gray(image=image)
    image = global_green_less(image=image)
    image = saturation_red_thin(image,stepsize=0.2)
    image = saturation_red_thin(image,stepsize=0.15)
    image = saturation_green_thin(image,stepsize=0.3)
    image = saturation_blue_thin(image,stepsize=0.4)
    image = global_gray(image=image)
    image = saturation_blue_thin(image=image, stepsize=0.4)
    image = hue_green_left(image=image, step_size=0.4)
    image = hue_blue_right(image=image, step_size=0.6)
    image = hue_yellow_right(image=image, step_size=0.6)
    image = global_light(image=image)
    image = global_light(image=image)
    image = global_light(image=image)
    image = global_light(image=image)
    image = global_light(image=image)

    return image










if __name__ == '__main__':
    count = 0
    file_path = root + 'prepare_data_pair/tailoring_data/'  # 用来扭曲的数据 被存放的位置
    out_path  = root + 'data/train_data/'  # 存放 扭曲过后的源数据
    source_out_path = root + 'data/source_data/'  # 存放 干净的源数据
    length = len(os.listdir(file_path))
    for pic_name in os.listdir(file_path):
        # count = count + 1
        # image_path = file_path + pic_name
        # img = load_image2numpy(image_path)
        # resize_image(image=img,outpath=file_path+str(count)+'.jpg')
        count = count + 1
        image_path = file_path + pic_name
        image = load_image2numpy(image_path)
        save_image(image,source_out_path+str(count)+'.jpg')  # 干净的源数据 格式为  xx.jpg

        indoor_image = distorted_indoor(image)
        save_image(indoor_image,out_path+str(count)+"_a"+'.jpg') # 扭曲过的数据 格式为 xx_a.jpg

        # outdoor_image = distorted_landscapes(image)
        # save_image(outdoor_image,out_path+str(count)+"_b"+'.jpg')

        # show_image(indoor_image)
        # show_image(outdoor_image)

        print(count/length)








