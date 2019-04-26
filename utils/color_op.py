import cv2 as cv
import numpy as np
from utils.load_image import *
import math
import cv2
from constant.constant import *
"""
 in open-cv HSV:
        H 0-180
        S 0-255
        V 0-255
        
 in open-cv HLS:
        H 0-180
        L 0-255
        S 0-255
        
 in open-cv LAB:
        L   0  - 100
        A -127 - 128
        B -127 - 128
"""


def RGB_to_LAB(img):
    img = (img*256.0).astype(np.uint8)
    lab = cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
    return lab

def HLS_to_RGB(hls_image):
    cv_hls = hls_image
    cv_rgb = cv2.cvtColor(cv_hls,cv2.COLOR_HLS2RGB)
    rgd = cv_rgb/256.0
    return rgd

def RGB_to_HLS(image):
    cv_rgb = (image*255.0).astype(np.uint8)
    cv_hls = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2HLS)
    return cv_hls

def HSV_to_RGB(hsv_image):
    cv_hls = hsv_image
    cv_rgb = cv2.cvtColor(cv_hls,cv2.COLOR_HSV2RGB)
    rgd = cv_rgb/256.0
    return rgd

def RGB_to_HSV(image):
    cv_rgb = (image*256.0).astype(np.uint8)
    cv_hsv = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2HSV)
    return cv_hsv

def ColourDistance(image, target_color_list):
    result = [new_ColourDistance(image,target_pixel) for target_pixel in target_color_list]
    result = tuple(result)
    result = np.stack(result)
    result = result.min(0)
    return result

def new_ColourDistance(image,target_pixel):
    target_img = np.ones(image.shape)
    target_img[:,:,0],target_img[:,:,1],target_img[:,:,2] \
        = target_img[:,:,0]*target_pixel[0]  , target_img[:,:,1]*target_pixel[1] , target_img[:,:,2]*target_pixel[2]
    a_R,a_G,a_B = image[:,:,0],image[:,:,1],image[:,:,2]
    b_R,b_G,b_B = target_img[:,:,0],target_img[:,:,1],target_img[:,:,2]
    r_mean = (a_R+b_R)/2.0
    R = a_R - b_R
    G = a_G - b_G
    B = a_B - b_B
    distance = (((2+r_mean)/256.0)*(R*R)) + (4*G*G) + ((2+((255.0-r_mean)/256.0))*(B*B))
    return np.sqrt(distance)

def get_beta(distance,threshold):
    if distance<=0.45:
        return 1.0
    elif distance<threshold:
        k = 1/(0.45-threshold)
        b = (-threshold)*k
        beta = k*distance + b
        return beta
    else:
        return 0.0

def keep_hue_stable(max_index,mid_index,min_index,rgb,h):
    """
    max_index,mid_index,min_index 根据调整之前来计算
    输入的rgb是 由 调整后的两个通道 + 未调整的一个通道 构成的
    返回的rgb是 由 调整后的两个通道 + 调整后的一个通道 构成的
    :param h: hsl[0] scalar
    :param max_index:
    :param min_index:
    :param rgb: list like [0.1,0.2,0.3]
    :return:
    """
    rgb_max,rgb_min = max(rgb),min(rgb)
    r,g,b = rgb[0],rgb[1],rgb[2]
    distance = rgb_max-rgb_min
    r,g,b = rgb[0],rgb[1],rgb[2]
    if max_index == 0:
        if mid_index == 1:
            g = b + 6*h*distance
        if mid_index == 2:
            b = g - 6*h*distance
    if max_index == 1:
        if mid_index == 2:
            b = r + (6.0*h-2.0)*distance
        if mid_index == 0:
            r = b - (6.0*h-2.0)*distance
    if max_index == 2:
        if mid_index == 0:
            r = g + (6.0*h-4.0)*distance
        if mid_index == 1:
            g = r - (6.0*h-4.0)*distance
    return [r,g,b]


def choose_similar_hsv_pixle(channel_pixel,up,low):
    if channel_pixel>=low and channel_pixel<=up:
        return 1
    else:
        return 0


def choose_similar_color(image,target_hsv_color,whe_show=False):
    """

    :param image: np.array
    :param target_hsv_color: dict
    :param whe_show: bool
    :return: result_image np.array
    """
    h_channel,s_channel,v_channel = RGB_to_HSV(image)[:,:,0],RGB_to_HSV(image)[:,:,1],RGB_to_HSV(image)[:,:,2]
    np_choose_hsv_pixel = np.vectorize(choose_similar_hsv_pixle)
    # compute h
    if 'h_low' not in target_hsv_color.keys() or 'h_up' not in target_hsv_color.keys():
        raise NameError('target_hsv_color must have h_low and h_up! ')
    else:
        h_low,h_up = target_hsv_color['h_low'],target_hsv_color['h_up']
        h = np_choose_hsv_pixel(channel_pixel = h_channel,up=h_up,low=h_low)
    # compute s
    if 's_low' in target_hsv_color.keys() and 's_up' in target_hsv_color.keys():
        s_low,s_up = target_hsv_color['s_low'],target_hsv_color['s_up']
        s = np_choose_hsv_pixel(channel_pixel=s_channel, up=s_up, low=s_low)
    else:
        s = np.ones(s_channel.shape)
    # compute v
    if 'v_low' in target_hsv_color.keys() and 'v_up' in target_hsv_color.keys():
        v_low, v_up = target_hsv_color['v_low'], target_hsv_color['v_up']
        v = np_choose_hsv_pixel(channel_pixel=v_channel, up=v_up, low=v_low)
    else:
        v = np.ones(v_channel.shape)
    result = h*s*v
    if whe_show:
        beta_metrix = np.tile(result[:,:,np.newaxis],[1,1,3])
        np_ones = np.ones(image.shape)
        image_show = beta_metrix*np_ones + (1.0-beta_metrix)*image
        show_image(image_show)
    return result




    return beta_metrix


def update_globa_hue(image,action,step_size):
    if action == 'left':
        step_size = -1.0 * 10 * step_size
    elif action == 'right':
        step_size = 1.0 * 10 * step_size
    cv_hls = RGB_to_HLS(image)
    cv_hls[:, :, 0] = np.clip(cv_hls[:, :, 0] + step_size, a_min=0, a_max=255)
    new_color = HLS_to_RGB(hls_image=cv_hls)
    return new_color


def update_specific_hue(image,target_color_hsv,action,step_size):
    """

    :param image:
    :param target_color_hsvaction:
    :param step_size:
    :return:
    """
    if action =='left':
        step_size = -1.0*10*step_size
    elif action == 'right':
        step_size = 1.0*10*step_size
    height,width,n_channels = image.shape
    image_show = image.copy()
    beta_metrix = choose_similar_color(image=image,target_hsv_color=target_color_hsv,whe_show=False)
    beta_metrix = np.tile(beta_metrix[:,:,np.newaxis],[1,1,3])
    cv_hls = RGB_to_HLS(image)
    cv_hls[:, :, 0] = np.clip(cv_hls[:,:,0] + step_size,a_min=0,a_max=255)
    new_color = HLS_to_RGB(hls_image=cv_hls)
    result = beta_metrix*new_color + (1-beta_metrix)*image
    result = np.clip(result,a_max=1.0,a_min=0.0)
    return result

# 调整特定颜色的饱和度
def update_specific_saturation(image,target_color_hsv,action,step_size,raw_img=None):
    """
    :param image:  input_image as a numpy array
    :param target_color:  like [0.4,0.0,1.0]
    :param color_threshold: the min distance between pixel color and the target_color,must>0.55
    :param action: make the target attributes bigger or small
    :return:
    """
    height,width,n_channels = image.shape
    image_show = image.copy()
    whe_scalar = choose_similar_color(image=image,target_hsv_color=target_color_hsv,whe_show=False)
    whe_scalar = whe_scalar[:,:,np.newaxis]
    whe_scalar = np.tile(whe_scalar,[1,1,3])
    if action == "left":
        action = -0.5*step_size
    elif action == 'right':
        action = 0.5*step_size
    color_image = update_global_saturation(img=image,action=action)
    result = whe_scalar*color_image + (1-whe_scalar)*image
    result = np.clip(result,a_max=1.0,a_min=0.0)
    return result

# 调整全局饱和度
def update_global_saturation(img,action):
    img = 256.0*img
    img_out = img
    if action>1:  action=1
    if action<-1: action=-1
    Increment = action# -1 ~ 1
    img_min = img.min(axis=2)
    img_max = img.max(axis=2)
    Delta = (img_max - img_min)/ 256.0
    value = (img_max + img_min)/ 256.0
    L = value / 2.0
    mask_1 = L < 0.5
    s1 = Delta / (value + 0.001)
    s2 = Delta / (2 - value + 0.001)
    s = s1 * mask_1 + s2 * (1 - mask_1)
    if Increment >= 0:
        temp = Increment + s
        mask_2 = temp > 1
        alpha_1 = s
        alpha_2 = s * 0 + 1 - Increment
        alpha = alpha_1 * mask_2 + alpha_2 * (1 - mask_2)
        alpha = 1 / (alpha + 0.001) - 1
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 256.0) * alpha
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 256.0) * alpha
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 256.0) * alpha
    else:
        alpha = Increment
        img_out[:, :, 0] = L * 256.0 + (img[:, :, 0] - L * 256.0) * (1 + alpha)
        img_out[:, :, 1] = L * 256.0 + (img[:, :, 1] - L * 256.0) * (1 + alpha)
        img_out[:, :, 2] = L * 256.0 + (img[:, :, 2] - L * 256.0) * (1 + alpha)
    img_out = img_out / 256.0
    # 饱和处理
    mask_1 = img_out < 0
    mask_2 = img_out > 1
    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2

    '''
    hsv = RGB_to_HSV(img)
    s_channel = hsv[:,:,1]
    hsv[:,:,1] = np.clip(s_channel+action,a_min=0,a_max=255)
    img = HSV_to_RGB(hsv)
    '''
    img_out = np.clip(img_out,a_max=1.0,a_min=0.0)

    return img_out

def normalized(target_color_list):
    result_list = target_color_list.copy()
    for i in range(len(target_color_list)):
        for j in range(3):
            result_list[i][j] = target_color_list[i][j] / 256.0
    return result_list



if __name__ == '__main__':
    root_path = '/home/wyz/PycharmProjects/deep_drawing_and_paintings_network_make_photo_retouching_easier/'
    file = 'prepare_data_pair/fuck/144722436540.jpg'
    filepath = root_path + file

    raw_image_np = load_image2numpy(filepath)
    show_image(raw_image_np)
    show_histogram(raw_image_np)

    #  树干绿色 [58,71,27]  天空蓝色[126,152,187]  地面黄色[169,132,80]

    # green = normalized(green_list)
    # yellow = normalized(yellow_list)
    # blue = normalized(blue_list)
    # sun = normalized(sun_list)
    # skin = normalized(skin_list)
    # purple = normalized(purple_list)

    # test hue -- blue

    # choose_similar_color(raw_image_np,target_hsv_color=blue,whe_show=True)
    # image_np = update_specific_hue(image=raw_image_np,target_color_list=blue ,color_threshold=0.6,action='left',step_size=0.7)
    # image_np = update_specific_hue(image=image_np,target_color_list=green,color_threshold=0.6,action='left', step_size=0.7)
    # raw_image_np = update_specific_saturation(raw_image_np, target_color_hsv=green,  action='right', step_size=0.8)
    # show_image(raw_image_np,info='blue_hue')
    # show_histogram(raw_image_np)
    #
    # raw_image_np = update_specific_saturation(raw_image_np, target_color_hsv=green, action='right', step_size=0.8)
    # show_image(raw_image_np, info='blue_hue')
    # show_histogram(raw_image_np)

            # test saturation
    # image_np = update_specific_saturation(image=raw_image_np, target_color=target_tree,
    #                                       color_threshold=0.6, action='left', step_size=0.8)
    # show_image(image_np,info='blue_saturation')

    # # test hue -- green
    # # choose_similar_color(image_np,target_color=target_tree,threshold=0.9,whe_show=True)
    # image_np = update_hue(image=raw_image_np,target_color=target_tree,color_threshold=0.7,action='left',step_size=0.1)
    # show_image(image_np,info='green_hue')
    #
    #         # test saturation
    # image_np = update_saturation(image=image_np, target_color=target_tree, color_threshold=0.7, action='right',
    #                              step_size=0.1)
    # show_image(image_np,info='green_saturation')

    # print(result)
    # print(np.sum(result))

    # for i in range(1):
    #     image_np = curve_contrast(image_np)





