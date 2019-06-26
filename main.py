from train.train import *
import tensorflow as tf
from train.model import *
import threading
from constant.constant import *
from train.work import *
from constant.params import *
from utils.global_count import _init_train_count,_init_reward_list,_init_trajectory_dict





if __name__ == "__main__":
    # for PPO
    """
        sess = tf.Session()
        train(sess=sess)
    """

    with tf.device(device):


        # initial count_op
        _init_train_count()
        _init_reward_list()
        _init_trajectory_dict()


        config = tf.ConfigProto(allow_soft_placement=True)

        SESS = tf.Session(config=config)

        COORD = tf.train.Coordinator()

        tf.set_random_seed(-1)

        GLOBAL_MASTER = A3C_Net(type='global',name='Global_Master', sess=SESS,
                            dim_color_feature=dim_gray_hist,
                            dim_vgg_feature=dim_image_feature,
                            a_dim=dim_master_action,
                            LR_A=LR_A,LR_C=LR_A,devcie=device,
                            global_AC=None)


        GLOBAL_SUB    = A3C_Net(type='global',name='Global_Sub', sess=SESS,
                                       dim_color_feature=dim_color_hist,
                                       dim_vgg_feature=dim_image_feature,
                                       a_dim=dim_sub_action,
                                       LR_A=LR_A,LR_C=LR_A,devcie=device,
                                       global_AC=None)

        VGG = A3C_Net(type='VGG',name='VGG', sess=None,dim_color_feature=None,
                     dim_vgg_feature=None,a_dim=None,LR_A=None,LR_C=None,devcie=None,global_AC=None)

        workers = []

        for i in range(N_WORKERS):
            i_name = 'W_%i' % i
            workers.append(AC_Worker(name=i_name,
                                     master_global=GLOBAL_MASTER,sub_global=GLOBAL_SUB,
                                     sess=SESS,device=device,LR_A=LR_A,LR_C=LR_C,
                                     dim_color_feature=dim_color_hist,
                                     dim_black_feature=dim_gray_hist,
                                     dim_vgg_feature=dim_image_feature,
                                     master_a_dim=dim_master_action,
                                     sub_a_dim=dim_sub_action,
                                     coord = COORD,
                                     vgg=VGG))

        SESS.run(tf.global_variables_initializer())

        worker_threads = []
        for worker in workers:
            job = lambda: worker.work()
            t = threading.Thread(target=job)
            t.start()
            worker_threads.append(t)

        COORD.join(worker_threads)






