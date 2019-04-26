from constant.constant import *
import numpy as np

def test_load():
    data_path = VGG_PATH # 文件保存路径
    data_dict = np.load(data_path, encoding='latin1').tolist()
    print("keys:  ",data_dict.keys())
    print("data_dict['conv1_1']: ",data_dict['conv1_1'][0].shape , data_dict['conv1_1'][1].shape)
    keys = sorted(data_dict.keys())
    for key in keys:
        weights = data_dict[key][0]
        biases = data_dict[key][1]
        print('\n')
        print(key)
        print('weights shape: ', weights.shape)
        print('biases shape: ', biases.shape)


if __name__ == '__main__':
    test_load()