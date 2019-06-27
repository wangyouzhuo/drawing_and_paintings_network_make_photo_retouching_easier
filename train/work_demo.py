import tensorflow as tf
from train.model import A3C_Net
from constant.constant import *
from constant.params import *
from environment.env import *
from utils.global_count import _get_train_count,_append_trajectory_dict,_append_reward_list,_add_train_count



class AC_Worker(object):

    def __init__(self,name,master_global,sub_global,sess,device,
                 LR_A,LR_C,dim_color_feature,dim_black_feature,
                 dim_vgg_feature,master_a_dim,sub_a_dim,coord,vgg):

        self.coord = coord

        self.name = name

        self.env = Environment(dirty_path=dirty_data_path,target_path=target_data_path)

        self.Master_net = A3C_Net(type='local',name='Master_local_'+name,sess=sess,
                                   dim_color_feature=dim_black_feature,
                                   dim_vgg_feature=dim_vgg_feature,
                                   a_dim=master_a_dim,
                                   LR_A=LR_A,LR_C=LR_C,devcie=device,
                                   global_AC=master_global,
                                   vgg = vgg)

        self.Sub_Net    = A3C_Net(type='local',name='Sub_local_'+name,sess=sess,
                                  dim_color_feature=dim_color_feature,
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
            s_image,color_feature,gray_feature =self.env.reset()
            trajectory = []
            ep_r  = 0
            steps = 0
            master_reward = []
            while True:
                #-----------------------------------sample sub_policy from master_net-----------------------------------
                if steps%K == 0:
                    start_color_feature,start_gray_feature,start_s_image \
                        = color_feature,gray_feature,s_image
                    sub_policy_index, _ = self.Master_net.choose_action(color_feature = start_color_feature,
                                                                        gray_feature  = start_gray_feature,
                                                                        s_image       = start_s_image)
                #------------------------------------sample action from sub_policy-------------------------------
                a, _ = self.Sub_Nets[sub_policy_index].choose_action(gray_feature  = gray_feature,
                                                                     color_feature = color_feature,
                                                                     s_image       = s_image)
                trajectory.append([sub_policy_index,a])
                # take the sampled sub_action
                s_image_next, color_feature_next, gray_feature_next, r, done \
                    = self.env.take_sub_action(action_index = a,policy_index = sub_policy_index )
                master_reward.append(r)
                steps = steps + 1
                ep_r = ep_r + r
                # append the buffer
                buffer_s_image.append(s_image)
                buffer_color_feature.append(color_feature)
                buffer_gray_feature.append(gray_feature)
                buffer_a.append(a)
                buffer_r.append(r)

                # update the master_net and sub_net
                if steps%(K+N) == 0 or done:
                    #------------------------------------update sub_net-------------------------------------
                    if done:
                        q = 0
                    else:
                        q = self.Sub_Nets[sub_policy_index].get_value(gray_feature = gray_feature_next,
                                                                      color_feature=color_feature_next,
                                                                      s_image=s_image)
                    buffer_q = []
                    for r in buffer_r:
                        q = r + GAMMA * q
                        buffer_q.append(q)
                    buffer_q.reverse()
                    buffer_s_image, buffer_color_feature,buffer_gray_feature,buffer_a, buffer_q = \
                    np.array(buffer_s_image)      , np.array(buffer_color_feature),\
                    np.array(buffer_gray_feature) , np.array(buffer_a),np.array(buffer_q)
                    self.Sub_Nets[sub_policy_index].update_network(s_image       = buffer_s_image  ,
                                                                   gray_feature  = buffer_gray_feature ,
                                                                   color_feature = buffer_color_feature ,
                                                                   a             = buffer_a[:,np.newaxis],
                                                                   q_value       = buffer_q[:,np.newaxis])
                    buffer_s_image, buffer_color_feature,buffer_gray_feature,buffer_a, buffer_r = [],[],[],[],[]

                    #-----------------------------------update master_net-----------------------------------
                    end_color_feature, end_gray_feature, end_s_image \
                        = color_feature_next, gray_feature_next, s_image_next
                    """
                        traisition for master_net:
                            state:        (start_color_feature , start_gray_feature , start_s_image)
                            action:       sub_policy_index
                            reward:       sum(master_reward)
                            next_state:   (end_color_feature   , end_gray_feature   , end_s_image)
                    """
                    if done:
                        v_next_master = 0
                    else:
                        v_next_master = self.Master_net.get_value(color_feature = start_color_feature,
                                                                  gray_feature  = start_gray_feature,
                                                                  s_image       = start_s_image)
                    q_master = sum(master_reward) + GAMMA*v_next_master
                    buffer_s_image_for_master       = [start_s_image]
                    buffer_gray_feature_for_master  = [start_gray_feature]
                    buffer_color_feature_for_master = [start_color_feature]
                    buffer_a_for_master             = [sub_policy_index]
                    buffer_q_for_master             = [q_master]

                    buffer_s_image_for_master, buffer_gray_feature_for_master,buffer_color_feature_for_master \
                    buffer_a_for_master, buffer_q_for_master \
                        = \
                    np.array(buffer_s_image_for_master), np.vstack(buffer_gray_feature_for_master), \
                    np.vstack(buffer_color_feature_for_master), np.vstack(buffer_a_for_master), \
                    np.vstack(buffer_q_for_master)

                    self.Master_net.update_network(s_image=buffer_s_image_for_master,
                                                   color_feature=buffer_color_feature_for_master,
                                                   gray_feature=buffer_gray_feature_for_master,
                                                   a=buffer_a_for_master,
                                                   q_value=buffer_q_for_master)
                    master_reward = []

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










