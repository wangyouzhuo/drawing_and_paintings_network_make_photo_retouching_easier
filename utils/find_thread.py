import os
import sys
sys.path.append('/home/wyz/PycharmProjects/deep_drawing_and_paintings_network_make_photo_retouching_easier/')

from PIL import Image
import numpy as np
from utils.load_image import *
from utils.distorted_op import *


loadpath = '/home/wyz/PycharmProjects/deep_drawing_and_paintings_network_make_photo_retouching_easier/' \
            + 'prepare_data_pair/tailoring_data/'

all_count = len(os.listdir(loadpath))
count = 0
distance_list = []

for pic_name in os.listdir(loadpath):
    count = count + 1
    picture = Image.open(loadpath + pic_name)
    shape = picture.size
    img_np = np.array(picture)

    image = load_image2numpy(loadpath + pic_name)
    for i in range(3):
        image = global_saturation_down(image)
    show_image(image)
    _,distance = check_histogram(image,che_sat=True)
    print("img:%s   distance:%s"%(pic_name,distance))
    # if whe_use != 'n':
    #     distance_list.append(distance)
    #     print("distance_min:%5s  diatance_max:%5s"%((min(distance_list) , max(distance_list))))
    # elif whe_use == 'n':
    #     print("%s Pass!"%pic_name)
    #     pass


    if count>2:
        break

