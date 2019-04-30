import tensorflow as tf
from train.model import PPO_Net
from constant.constant import *
from constant.params import *
from environment.env import *

def train(sess,BATCH_SIZE=20):

    adaptive_dict = {'KL_Min': KL_Min,
                     'KL_Max': KL_Max,
                     'KL_BETA': KL_BETA}

    with tf.device(DEVICE):

        master_ppo = PPO_Net(name='Master_PPO', sess=sess,dim_global_feature=dim_gray_hist,
                             dim_image_feature=dim_image_feature,a_dim=dim_master_action,
                             LR_A=LR_A, LR_C=LR_C,adaptive_dict=adaptive_dict,
                             THRESHOLD=2,device=DEVICE)

        sub_ppo    = PPO_Net(name='Sub_PPO', sess=sess,dim_global_feature=dim_color_hist,
                             dim_image_feature=dim_image_feature,a_dim=dim_sub_action,
                             LR_A=LR_A, LR_C=LR_C,adaptive_dict=adaptive_dict,
                             THRESHOLD=2,device=DEVICE)

    sess.run(tf.global_variables_initializer())

    env = Environment(dirty_path=dirty_data_path,target_path=target_data_path)

    ep_reward_list = []

    buffer_s_image,buffer_s_feature, buffer_a, buffer_q,buffer_r = [],[],[],[],[]
    EP_COUNT = 0
    while EP_COUNT<=EP_MAX:
        s_image,s_color_feature,s_gray_feature = env.reset()
        EP_COUNT = EP_COUNT + 1
        ep_r  = 0
        steps = 0

        # result,self.get_color_feature(result),self.get_gray_feature(result),reward,self.done

        while True:
            # update master_ppo
            if (steps + 1) % MATSER_ACTION_ITERA == 0:
                buffer_s_image_for_master, buffer_s_feature_for_master, buffer_a_for_master = [], [], []
                a, _ = master_ppo.choose_action(s_feature=s_gray_feature,s_image=s_image)
                # print("choose_master_action")
                s_image_next,s_color_feature_next,s_gray_feature_next, r, done\
                    = env.take_master_action(a)
                steps = steps + 1
                ep_r = ep_r + r
                if done:
                    q = 0
                else:
                    q = master_ppo.get_value(s_feature=s_gray_feature_next,s_image=s_image_next)
                buffer_q_for_master = []
                q = r + GAMMA * q
                buffer_s_image_for_master.append(s_image)
                buffer_s_feature_for_master.append(s_gray_feature)
                buffer_a_for_master.append(a)
                buffer_q_for_master.append(q)
                buffer_s_image_for_master,buffer_s_feature_for_master, \
                buffer_a_for_master, buffer_q_for_master = np.array(buffer_s_image_for_master),\
                                                           np.vstack(buffer_s_feature_for_master),\
                                                           np.vstack(buffer_a_for_master),\
                                                           np.vstack(buffer_q_for_master)
                # print("buffer_s_feature gray:",buffer_s_feature_for_master)
                master_ppo.update_network(s_image   = buffer_s_image_for_master,
                                          s_feature = buffer_s_feature_for_master,
                                          a         = buffer_a_for_master,
                                          q_value   = buffer_q_for_master)
                buffer_s_image_for_master,\
                buffer_s_feature_for_master,\
                buffer_a_for_master, \
                buffer_q_for_master = [], [], [], []
                s_image,s_color_feature,s_gray_feature = s_image_next, s_color_feature_next, s_gray_feature_next
                if done or steps>=MAX_STEPS_IN_EPISODE:
                    print("Epi:%6s || Success:%5s || Steps:%3s || ep_r:%6s || Img::%8s"%(EP_COUNT,done,steps,round(ep_r,3),env.img_name))
                    ep_reward_list.append(ep_r)
                    if done is False:
                        env.save_env_image(success=False,epi=EP_COUNT)
                    elif done is True:
                        env.save_env_image(success=True,epi=EP_COUNT)
                    break
            # update sub_ppo
            a, _ = sub_ppo.choose_action(s_feature=s_color_feature,s_image=s_image)
            # print("choose_sub_action")
            s_image_next, s_color_feature_next, s_gray_feature_next, r, done = env.take_sub_action(a)
            steps = steps + 1
            ep_r = ep_r + r
            buffer_s_image.append(s_image)
            buffer_s_feature.append(s_color_feature)
            buffer_a.append(a)
            buffer_r.append(r)
            if (steps+1)%SUB_ACTION_ITER == 0 or done:
                if done:
                    q = 0
                else:
                    q = sub_ppo.get_value(s_feature=s_color_feature_next,s_image=s_image_next)
                buffer_q = []
                for r in buffer_r:
                    q = r + GAMMA * q
                    buffer_q.append(q)
                buffer_q.reverse()
                buffer_s_image, buffer_s_feature, buffer_a, buffer_q = \
                    np.array(buffer_s_image), \
                    np.array(buffer_s_feature), \
                    np.array(buffer_a), \
                    np.array(buffer_q)
                # print("buffer_s_feature color:",buffer_s_feature)
                # print("buffer_a :",buffer_a)
                sub_ppo.update_network(s_image   = buffer_s_image  ,
                                       s_feature = buffer_s_feature ,
                                       a         = buffer_a[:,np.newaxis],
                                       q_value   = buffer_q[:,np.newaxis])
                buffer_s_image, buffer_s_feature, buffer_a, buffer_r = [], [], [], []
            s_image, s_color_feature, s_gray_feature = s_image_next, s_color_feature_next, s_gray_feature_next
            if done or steps>=MAX_STEPS_IN_EPISODE:
                print("Epi:%6s || Success:%5s || Steps:%3s || ep_r:%6s || Img::%8s" % (
                EP_COUNT, done, steps, round(ep_r, 3), env.img_name))
                ep_reward_list.append(ep_r)
                if done is False:
                    env.save_env_image(success=False,epi=EP_COUNT)
                elif done is True:
                    env.save_env_image(success=True,epi=EP_COUNT)
                break




