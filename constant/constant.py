
root = '/home/wyz/PycharmProjects/drawing_and_paintings_network_make_photo_retouching_easier/'

dirty_data_path  = root + '/data/source_data/'

target_data_path = root + '/data/train_data/'

result_save_path = root +'/data/result_image/'

"""
    green_list  = [[0, 255, 0], [110, 157, 45],[83,131,79],[84,121,69],[79,116,75],[67,189,28]]
    yellow_list = [[178,186,38],[194,222,50],[218,221,72],[171,170,2],[172,133,42],[225,189,95]]
    sun_list    = [[212, 149, 33], [210, 157, 61], [232, 186, 64], [250, 216, 48]]
    skin_list   = [[208, 185, 169], [197, 167, 169], [210, 157, 125], [207, 155, 97], [171, 115, 56]]
    purple_list = [[111, 88, 132], [49, 19, 83], [128, 67, 126]]
"""

MATSER_ACTION_ITERA = 10
SUB_ACTION_ITER = 20

WIDTH  = 224
HEIGHT = 224
N_CHANNEL = 3

GAMMA = 0.99

green = {'h_low': 38, 'h_up':75  }
blue  = {'h_low': 75, 'h_up':130 }
red   = {'h_low': 160,'h_up':179 }
Violet= {'h_low': 130,'h_up':160 }
yellow= {'h_low': 22 ,'h_up':38  }
orange= {'h_low': 0  ,'h_up':22  }

VGG_PATH = '/home/wyz/PycharmProjects/VGG_PARAMS/vgg16.npy'

R_MEAN = 128.0
G_MEAN = 128.0
B_Mean = 128.0

A_ITER = 10
C_ITER = 10



dim_image_feature = 4096

dim_color_hist = 256*3
dim_gray_hist = 256


TERMINAL_THRESHOLD = 1000

EP_MAX = 1000000

MAX_STEPS_IN_EPISODE = 400




