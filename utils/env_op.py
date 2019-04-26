import numpy as np


def color2gray(image):
    result = image.copy()
    r,g,b = image[:,:,0],image[:,:,1],image[:,:,2]
    gray_channel = (r+g+b)/3.0
    result[:,:,0],result[:,:,1],result[:,:,2] = gray_channel,gray_channel,gray_channel
    return result


def compute_color_l2(target_image,current_image):
    distance = np.sum( np.square(target_image-current_image) )
    return distance


def compute_black_l2(target_image,current_image):
    target_image  = color2gray(target_image)
    current_image = color2gray(current_image)
    return compute_color_l2(target_image=target_image,current_image=current_image)


