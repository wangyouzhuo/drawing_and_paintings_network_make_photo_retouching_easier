import numpy as np
from utils.load_image import *
from constant.constant import root
from utils.env_op import color2gray
from utils.color_op import *
import tensorflow as tf

def distribution_color(image,channel_kind='lab'):
    w,h,c = image.shape
    if channel_kind=='lab':
        image = RGB_to_LAB(image)
    else:
        image = np.clip(image*255.0,a_min=0,a_max=255).astype(np.uint8)
    # print("l--max:%3s  min:%3s  ||  a--max:%3s  min:%3s ||  b--max:%3s  min:%3s"
    #       %(np.max(image[:,:,0]),np.min(image[:,:,0]),np.max(image[:,:,1]),np.min(image[:,:,1]),np.max(image[:,:,2]),np.min(image[:,:,2])
    #         ))
    hist_one,_   = np.histogram(image[:,:,0].ravel(),bins=256,range=(0,256))
    hist_two,_   = np.histogram(image[:,:,1].ravel(),bins=256,range=(0,256))
    hist_three,_ = np.histogram(image[:,:,2].ravel(),bins=256,range=(0,256))
    hist_one   = hist_one*1.0/(w*h)
    hist_two   = hist_two*1.0/(w*h)
    hist_three = hist_three*1.0/(w*h)
    return hist_one,hist_two,hist_three


def distribution_gray(image):
    w, h, c = image.shape
    img = color2gray(image)
    img = np.clip(img*255.0, a_min=0, a_max=255).astype(np.uint8)
    hist,_ = np.histogram(img[:, :, 0].ravel(), bins=256, range=(0, 256))
    hist = hist*1.0/(w*h)
    return hist


def count_3d_hist(image,interval):
    image = RGB_to_LAB(image)
    l_channel,a_channel,b_channel = image[:,:,0],image[:,:,1],image[:,:,2]
    l_channel = count_channel(l_channel)
    a_channel = count_channel(a_channel)
    b_channel = count_channel(b_channel)
    count_object = Count_hist(l_channel=l_channel,a_channel=a_channel,b_channel=b_channel,interval=interval)
    result_list = count_object.compute()



def compute_result(l_pixel,a_pixel,b_pixel,result):
    print("result:",result)
    result[l_pixel][a_pixel][b_pixel] = result[l_pixel,a_pixel,b_pixel] + 1


def count_pixel(n_interval=20,pixel=None,):
    list = [i*(255.0/n_interval) for i in range(0,n_interval)]
    list.append(255.0)
    for i in range(len(list)):
        if list[i]<=pixel and list[i+1]>=pixel:
            pixel=i
            break
    return pixel


def count_channel(channel,n_interval=20):
    np_count_pixel = np.vectorize(count_pixel,excluded='n_interval')
    result = np_count_pixel(pixel=channel,n_interval=n_interval)
    return result


class Count_hist(object):

    def __init__(self,l_channel,a_channel,b_channel,interval):
        w,h = l_channel.shape
        self.l_channel = l_channel.reshape([1,w*h]).tolist()[0]
        self.a_channel = a_channel.reshape([1,w*h]).tolist()[0]
        self.b_channel = b_channel.reshape([1,w*h]).tolist()[0]
        # self.l_channel = l_channel
        # self.a_channel = a_channel
        # self.b_channel = b_channel
        self.interval = interval
        self.result = np.zeros(shape=(interval,interval,interval))

    def compute_pixel(self,l,a,b):
        result = np.zeros(shape=(self.interval,self.interval,self.interval))
        result[l,a,b] =  1
        result = result.reshape([1,self.interval*self.interval*self.interval]).tolist()[0]
        index = result.index(1.0)
        return index


    def compute(self):
        np_compute_pixel = np.vectorize(self.compute_pixel)
        index_list = np_compute_pixel(self.l_channel,self.a_channel,self.b_channel).tolist()
        result_list = list(map(lambda i:index_list.count(i),range(0,self.interval*self.interval*self.interval)))
        show_list = []
        for item in result_list:
            if item>0:
                show_list.append(item)
        return result_list


def generate_fc_weight(shape, whe_train, name):
    threshold = 1.0 / np.sqrt(shape[0])
    weight_matrix = tf.random_uniform(shape, minval=-threshold, maxval=threshold)
    weight = tf.Variable(weight_matrix, name=name, trainable=whe_train)
    return weight

def generate_fc_bias(shape, whe_train, name):
    bias_distribution = tf.constant(0.0, shape=shape)
    bias = tf.Variable(bias_distribution, name=name, trainable=whe_train)
    return bias
















if __name__ == '__main__':
    img = load_image2numpy(path='/home/wyz/PycharmProjects/'
                          'deep_drawing_and_paintings'
                          '_network_make_photo_retouchi'
                          'ng_easier/prepare_data_pai'
                          'r/tailoring_data/4.jpg')
    count_3d_hist(interval=20,image=img)





