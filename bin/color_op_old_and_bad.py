import cv2 as cv
import numpy as np
from utils.load_image import load_image2numpy,show_image
import math

def HSL_to_RGB(h,s,l):
    ''' Converts HSL colorspace (Hue/Saturation/Value) to RGB colorspace.
        Formula from http://www.easyrgb.com/math.php?MATH=M19#text19

        Input:
            h (float) : Hue (0...1, but can be above or below
                              (This is a rotation around the chromatic circle))
            s (float) : Saturation (0...1)    (0=toward grey, 1=pure color)
            l (float) : Lightness (0...1)     (0=black 0.5=pure color 1=white)

        Ouput:
            (r,g,b) (integers 0...255) : Corresponding RGB values

        Examples:
            >>> print HSL_to_RGB(0.7,0.7,0.6)
            (110, 82, 224)
            >>> r,g,b = HSL_to_RGB(0.7,0.7,0.6)
            >>> print g
            82
    '''
    def Hue_2_RGB( v1, v2, vH ):
        while vH<0.0: vH += 1.0
        while vH>1.0: vH -= 1.0
        if 6*vH < 1.0 : return v1 + (v2-v1)*6.0*vH
        if 2*vH < 1.0 : return v2
        if 3*vH < 2.0 : return v1 + (v2-v1)*((2.0/3.0)-vH)*6.0
        return v1

    if not (0 <= s <=1): raise ValueError("s (saturation) parameter must be between 0 and 1.")
    if not (0 <= l <=1): raise ValueError("l (lightness) parameter must be between 0 and 1.")

    r,b,g = (l*255,)*3
    if s!=0.0:
       if l<0.5 : var_2 = l * ( 1.0 + s )
       else     : var_2 = ( l + s ) - ( s * l )
       var_1 = 2.0 * l - var_2
       r = 255 * Hue_2_RGB( var_1, var_2, h + ( 1.0 / 3.0 ) )
       g = 255 * Hue_2_RGB( var_1, var_2, h )
       b = 255 * Hue_2_RGB( var_1, var_2, h - ( 1.0 / 3.0 ) )

    return (int(round(r)),int(round(g)),int(round(b)))

def RGB_to_HSL(image):
    ''' Converts RGB colorspace to HSL (Hue/Saturation/Value) colorspace.
        Formula from http://www.easyrgb.com/math.php?MATH=M18#text18
    '''

    # r,g,b = image[:,:,0],image[:,:,1],image[:,:,2]
    r,g,b = image[0],image[1],image[2]


    if not (0 <= r <=1): raise ValueError("r (red) parameter must be between 0 and 1.")
    if not (0 <= g <=1): raise ValueError("g (green) parameter must be between 0 and 1.")
    if not (0 <= b <=1): raise ValueError("b (blue) parameter must be between 0 and 1.")


    rgb_min = min( r, g, b )    # Min. value of RGB
    rgb_max = max( r, g, b )    # Max. value of RGB
    dis_max_min = rgb_max - rgb_min # Delta RGB value
    # compute l
    l = ( rgb_max + rgb_min ) / 2.0
    h = 0.0
    s = 0.0
    if dis_max_min!=0.0:
       # compute s
       if l<0.5:
           s = dis_max_min / ( rgb_max + rgb_min )
       else:
           s = dis_max_min / ( 2.0 - rgb_max - rgb_min )

       # compure h
       del_R = ( (( rgb_max - r )/6.0) + (dis_max_min/2.0) ) / dis_max_min
       del_G = ( (( rgb_max - g )/6.0) + (dis_max_min/2.0) ) / dis_max_min
       del_B = ( (( rgb_max - b )/6.0) + (dis_max_min/2.0) ) / dis_max_min
       if    r == rgb_max :
           h = del_B - del_G
       elif  g == rgb_max :
           h = ( 1.0 / 3.0 ) + del_R - del_B
       elif  b == rgb_max :
           h = ( 2.0 / 3.0 ) + del_G - del_R

       # h between 0 ~ 1
       while h < 0.0:
           h += 1.0
       while h > 1.0:
           h -= 1.0

    return (h,s,l)

def ColourDistance(rgb_1, rgb_2):
    R_1, G_1, B_1 = rgb_1
    R_2, G_2, B_2 = rgb_2
    rmean = (R_1 + R_2) / 2
    R = R_1 - R_2
    G = G_1 - G_2
    B = B_1 - B_2
    return math.sqrt( (2+rmean/256)*(R**2) + 4*(G**2)+(2+(255-rmean)/256)*(B**2)  )

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
    return math.sqrt(distance)

def get_beta(distance,threshold):
    if distance<=0.55:
        return 1.0
    elif distance<threshold:
        k = 1/(0.55-threshold)
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

def choose_similar_color(image,target_color_list,threshold=0.55,whe_show=False):
    """
    :param image: input_image as a numpy array
    :param target_color: like [0.4,0.0,1.0]
    :param threshold: the min distance between pixel color and the target_color
    :param whe_show: whe show the changed image
    :return: the shape is the same as the input image,and return a scalar(0~1) : 1 means similar ,0 means not similar
    """
    if threshold<0.55:
        raise RuntimeError("The threshold is too small! Please reset! ")
    height,width,n_channels = image.shape
    image_show = image.copy()
    result_scalar,result_binary = np.zeros(shape=[height,width]),np.zeros(shape=[height,width])
    # targer_r,target_g,target_b = target_color[0],target_color[1],target_color[2]
    for i in range(0,height):
        for j in range(0,width):
            pixel_color = image[i,j,:]
            distance = min([ColourDistance(item,pixel_color) for item in target_color_list])
            # distance = ColourDistance(target_color,pixel_color)
            if distance < threshold:
                beta = get_beta(distance,threshold)
                result_scalar[i][j],result_binary[i][j] = beta,1.0
                if whe_show:
                    a = [ item*beta for item in [1.0,1.0,1.0]]
                    b = [ item*(1-beta) for item in [image[i][j][0],image[i][j][1],image[i][j][2]] ]
                    c = [x + y for x, y in zip(a, b)]
                    image_show[i][j][0],image_show[i][j][1],image_show[i][j][2] = c
    if whe_show:
        show_image(image_show)
    return result_scalar,result_binary

def change_hsl(rgb_pixle,attributes,action,step_size=0.005):
    """

    :param rgb_pixle like (0.1,0.2,0.5)
    :param hsl  rgb ---> hsl
    :param attributes: 'hue' or 'saturation'
    :param action:  'left' or 'right'
    :return: rgb like [0.1,0.2,0.5]
    """
    hsl = RGB_to_HSL(rgb_pixle)
    rgb_pixle,hsl = list(rgb_pixle),list(hsl)
    rgb_set = set()
    for i in range(3):
        len_before = len(rgb_set)
        rgb_set.add(rgb_pixle[i])
        len_after = len(rgb_set)
        if len_after == len_before:
            # print("RGB内出现相同色彩")
            rgb_pixle[i] = rgb_pixle[i] + 0.00001*(rgb_pixle[i]-0.49999)
    index_set = set(range(3))
    min_rdg,min_index = min(rgb_pixle),rgb_pixle.index(min(rgb_pixle))
    max_rdg,max_index = max(rgb_pixle),rgb_pixle.index(max(rgb_pixle))
    index_set.remove(min_index)
    index_set.remove(max_index)
    mid_index = list(index_set)[0]
    if attributes == 'hue':
        """
        change the hue: 
            right--->make rgb_mid larger 
            left --->make rgb_mid smaller
        """
        if action == 'right':
            if rgb_pixle[mid_index]+step_size <= rgb_pixle[max_index]:
                rgb_pixle[mid_index] = rgb_pixle[mid_index] + step_size
        if action == 'left':
            if rgb_pixle[mid_index]-step_size >= rgb_pixle[min_index]:
                rgb_pixle[mid_index] = rgb_pixle[mid_index] - step_size
    if attributes == 'l':
        """
            change the light:
        """
        raise NotImplementedError("Change image brightness based on RGB : Method Not Implemented! ERROR!")
    if attributes == 'saturation':
        """
        change the saturation:
            right--->make the saturation larger 
                1. rgb_max larger and rgb_min smaller
                2. change the rgb_mid and keep the h not change
            left --->make the saturation smaller--->
                1. rgb_max smaller and rgb_min bigger
                2. change the rgb_mid and keep the h not change
        """
        if action =='right' :
            if rgb_pixle[max_index] + step_size < 1.0 and rgb_pixle[min_index] - step_size > 0.0:
                rgb_pixle[max_index] = rgb_pixle[max_index] + step_size
                rgb_pixle[min_index] = rgb_pixle[min_index] - step_size
                rgb_pixle = keep_hue_stable(h=hsl[0], rgb=rgb_pixle,
                                      max_index=max_index,
                                      mid_index=mid_index,
                                      min_index=min_index)
            else:
                pass # rgb no change
        if action == 'left':
            if rgb_pixle[max_index] - step_size > rgb_pixle[mid_index] and rgb_pixle[min_index] + step_size < rgb_pixle[mid_index]:
                rgb_pixle[max_index] = rgb_pixle[max_index] - step_size
                rgb_pixle[min_index] = rgb_pixle[min_index] + step_size
                rgb_pixle = keep_hue_stable(h=hsl[0],rgb=rgb_pixle,
                                      max_index=max_index,
                                      mid_index=mid_index,
                                      min_index=min_index )
            else:
                pass # rgb no change
    return rgb_pixle

# 调整特定颜色的色相
def update_specific_hue(image,target_color_list,color_threshold,action,step_size):
    """
    :param image:  input_image as a numpy array
    :param target_color:  like [0.4,0.0,1.0]
    :param color_threshold: the min distance between pixel color and the target_color,must>0.55
    :param action: make the target attributes bigger or small
    :return:
    """
    if color_threshold<0.55:
        raise RuntimeError("The threshold is too small! Please reset! ")
    height,width,n_channels = image.shape
    image_show = image.copy()
    # beta_metrix = map(lambda x:get_beta(rgb_1=x,rgb_2=target_color,threshold=color_threshold),image)
    # new_image = map(lambda x:change_hsl(rgb_pixle=x,attributes='hue',action=action,step_size=step_size),image)
    # result_image = beta_metrix*new_image + (1-beta_metrix)*image
    for i in range(0,height):
        for j in range(0,width):
            old_pixel = image[i,j,:]
            # distance = ColourDistance(target_color,old_pixel)
            distance = min([ColourDistance(item,old_pixel) for item in target_color_list])
            if distance < color_threshold:
                beta = get_beta(distance,color_threshold)
            else:
                beta = 0
            if  beta>0:
                new_pixel = change_hsl(rgb_pixle=old_pixel,attributes='hue',action=action,step_size=step_size)
                new_color = [ beta*x    for x in new_pixel]
                old_color = [(1-beta)*x for x in old_pixel]
                color = [x+y for x,y in zip(new_color,old_color)]
                image_show[i, j, :] = color
            else:
                image_show[i, j, :] = image[i, j, :]
    return np.clip(image_show,a_min=0.0,a_max=1.0)

# 调整特定颜色的饱和度
def update_specific_saturation(image,target_color_list,color_threshold,action,step_size):
    """
    :param image:  input_image as a numpy array
    :param target_color:  like [0.4,0.0,1.0]
    :param color_threshold: the min distance between pixel color and the target_color,must>0.55
    :param action: make the target attributes bigger or small
    :return:
    """
    if color_threshold<0.55:
        raise RuntimeError("The threshold is too small! Please reset! ")
    height,width,n_channels = image.shape
    image_show = image.copy()
    whe_scalar = choose_similar_color(image=image,target_color_list=target_color_list,threshold=color_threshold,whe_show=False)
    whe_scalar = whe_scalar[:,:,np.newaxis]
    whe_scalar = np.tile(whe_scalar,[1,1,3])
    if action == "left":
        action = -1.0*step_size
    elif action == 'right':
        action = step_size
    color_image = update_global_saturation(img=image,action=action)
    result = whe_scalar*color_image + (1-whe_scalar)*image
    return result

# 调整全局饱和度
def update_global_saturation(img,action):
    img = 255.0*img
    img_out = img
    Increment = action# -1 ~ 1

    img_min = img.min(axis=2)
    img_max = img.max(axis=2)

    Delta = (img_max - img_min)/ 255.0
    value = (img_max + img_min)/ 255.0
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
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha
    else:
        alpha = Increment
        img_out[:, :, 0] = L * 255.0 + (img[:, :, 0] - L * 255.0) * (1 + alpha)
        img_out[:, :, 1] = L * 255.0 + (img[:, :, 1] - L * 255.0) * (1 + alpha)
        img_out[:, :, 2] = L * 255.0 + (img[:, :, 2] - L * 255.0) * (1 + alpha)
    img_out = img_out / 255.0
    # 饱和处理
    mask_1 = img_out < 0
    mask_2 = img_out > 1
    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    return img_out

def normalized(target_color_list):
    result_list = target_color_list.copy()
    for i in range(len(target_color_list)):
        for j in range(3):
            result_list[i][j] = target_color_list[i][j] / 255.0
    return result_list

if __name__ == '__main__':
    root_path = '/home/wyz/PycharmProjects/deep_drawing_and_paintings_network_make_photo_retouching_easier/image/raw_image/'
    file = '14470426704.jpg'
    filepath = root_path + file

    raw_image_np = load_image2numpy(filepath)
    show_image(raw_image_np,info='raw')

    #  树干绿色 [58,71,27]  天空蓝色[126,152,187]  地面黄色[169,132,80]





    green = normalized([[79, 95, 59], [53, 64, 34], [102, 126, 48], [80, 119, 64], [164, 179, 34]])
    yellow = normalized([[173, 86, 32], [109, 67, 19], [129, 116, 38], [128, 78, 43]])
    blue = normalized([[79, 102, 154], [56, 110, 174], [71, 110, 165], [58, 84, 121], [25, 116, 161], [73, 172, 203]])
    sun = normalized([[212, 149, 33], [210, 157, 61], [232, 186, 64], [250, 216, 48]])
    skin = normalized([[208, 185, 169], [197, 167, 169], [210, 157, 125], [207, 155, 97], [171, 115, 56]])
    purple = normalized([[111, 88, 132], [49, 19, 83], [128, 67, 126]])



    # test hue -- blue
    choose_similar_color(raw_image_np,target_color_list=green,threshold=0.7,whe_show=True)
    # image_np = update_specific_hue(image=raw_image_np,target_color_list=green ,color_threshold=0.7,action='left',step_size=0.3)
    # show_image(image_np,info='blue_hue')

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





