from utils.generate_data_op import generate_data
from constant.constant import root


load_path = '/home/wyz/PycharmProjects/ddddeep_drawing_and_paintings_network_make_photo_retouching_easier/' \
            + 'prepare_data_pair/raw_data/'

save_path = root + 'prepare_data_pair/tailoring_data/'


generate_data(loadpath=load_path,savepath=save_path)