import tensorflow as tf
from train.model import A3C_Net
from constant.constant import *
from constant.params import *
from environment.env import *
from utils.global_count import _get_train_count,_append_trajectory_dict,_append_reward_list,_add_train_count



class AC_Worker(object):

    def __init__(self,name,master_global,sub_global,sess,device,
                 LR_A,LR_C,dim_color_feature,dim_gray_feature,
                 dim_vgg_feature,master_a_dim,sub_a_dim,coord,vgg):

        self.coord = coord

        self.name = name

        self.env = Environment(dirty_path=dirty_data_path,target_path=target_data_path)

        self.Master_net = A3C_Net(type='local',name='Master_local_'+name,sess=sess,
                                   dim_color_feature=dim_color_feature,
                                   dim_gray_feature=dim_gray_feature,
                                   dim_vgg_feature=dim_vgg_feature,
                                   a_dim=master_a_dim,
                                   LR_A=LR_A,LR_C=LR_C,devcie=device,
                                   global_AC=master_global,
                                   vgg = vgg)

        self.Sub_Net    = A3C_Net(type='local',name='Sub_local_'+name,sess=sess,
                                  dim_color_feature=dim_color_feature,
                                  dim_gray_feature=dim_gray_feature,
                                  dim_vgg_feature=dim_vgg_feature,
                                  a_dim=sub_a_dim,
                                  LR_A=LR_A,
                                  LR_C=LR_C,
                                  devcie=device,
                                  global_AC=sub_global,
                                  vgg = vgg)

    def work(self):

        while not self.coord.should_stop() and _get_train_count() <= EP_MAX:
            EP_COUNT = _add_train_count()
            s_image,s_color_feature,s_gray_feature =self.env.reset()
            buffer_s_image,buffer_s_feature, buffer_a, buffer_q,buffer_r = [],[],[],[],[]
            trajectory = []
            ep_r  = 0
            steps = 0
            # result,self.get_color_feature(result),self.get_gray_feature(result),reward,self.done
            while True:
                # update master_net
                if (steps + 1) % MATSER_ACTION_ITERA == 0:
                    buffer_s_image_for_master, buffer_s_feature_for_master, buffer_a_for_master = [], [], []
                    a, _ = self.Master_net.choose_action(s_feature=s_gray_feature,s_image=s_image)
                    # print("choose_master_action")
                    s_image_next,s_color_feature_next,s_gray_feature_next, r, done = self.env.take_master_action(a)
                    trajectory.append(a)
                    steps = steps + 1
                    ep_r = ep_r + r
                    if done:
                        q = 0
                    else:
                        q = self.Master_net.get_value(s_feature=s_gray_feature_next,s_image=s_image_next)
                    buffer_q_for_master = []
                    q = r + GAMMA * q
                    buffer_s_image_for_master.append(s_image)
                    buffer_s_feature_for_master.append(s_gray_feature)
                    buffer_a_for_master.append(a)
                    buffer_q_for_master.append(q)
                    buffer_s_image_for_master,buffer_s_feature_for_master, \
                    buffer_a_for_master, buffer_q_for_master = np.array(buffer_s_image_for_master), \
                                                               np.vstack(buffer_s_feature_for_master), \
                                                               np.vstack(buffer_a_for_master), \
                                                               np.vstack(buffer_q_for_master)
                    # print("buffer_s_feature gray:",buffer_s_feature_for_master)
                    self.Master_net.update_network(s_image   = buffer_s_image_for_master,
                                                   s_feature = buffer_s_feature_for_master,
                                                   a         = buffer_a_for_master,
                                                   q_value   = buffer_q_for_master)
                    buffer_s_image_for_master, \
                    buffer_s_feature_for_master, \
                    buffer_a_for_master, \
                    buffer_q_for_master = [], [], [], []
                    s_image,s_color_feature,s_gray_feature = s_image_next, s_color_feature_next, s_gray_feature_next
                    if done or steps>=MAX_STEPS_IN_EPISODE:
                        print("Epi:%6s || Success:%5s || Steps:%3s || ep_r:%6s || Img::%8s"%(EP_COUNT,done,steps,round(ep_r,3),self.env.img_name))
                        _append_reward_list(ep_r)
                        _add_train_count()
                        if done is False:
                            #self.env.save_env_image(success=False,epi=EP_COUNT)
                            pass
                        elif done is True:
                            _append_trajectory_dict(EP_COUNT,trajectory)
                            self.env.save_env_image(success=True,epi=EP_COUNT)
                        break
                # update sub_ppo
                a, _ = self.Sub_Net.choose_action(s_feature=s_color_feature,s_image=s_image)
                # print("choose_sub_action")
                s_image_next, s_color_feature_next, s_gray_feature_next, r, done = self.env.take_sub_action(a)
                trajectory.append(a)
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
                        q = self.Sub_Net.get_value(s_feature=s_color_feature_next,s_image=s_image_next)
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
                    self.Sub_Net.update_network(s_image   = buffer_s_image  ,
                                           s_feature = buffer_s_feature ,
                                           a         = buffer_a[:,np.newaxis],
                                           q_value   = buffer_q[:,np.newaxis])
                    buffer_s_image, buffer_s_feature, buffer_a, buffer_r = [], [], [], []
                s_image, s_color_feature, s_gray_feature = s_image_next, s_color_feature_next, s_gray_feature_next
                if done or steps>=MAX_STEPS_IN_EPISODE:
                    _append_reward_list(ep_r)
                    print("Epi:%6s || Success:%5s || Steps:%3s || ep_r:%6s || Img::%8s" % (EP_COUNT,
                    done, steps, round(ep_r, 3), self.env.img_name))
                    _add_train_count()
                    if done is False:
                        #self.env.save_env_image(success=False,epi=EP_COUNT)
                        pass
                    elif done is True:
                        _append_trajectory_dict(EP_COUNT,trajectory)
                        self.env.save_env_image(success=True,epi=EP_COUNT)
                    break










