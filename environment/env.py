import numpy as np
from utils.env_op import compute_color_l2,compute_black_l2
import os
import random
from utils.load_image import *
from utils.retouch_op import *
from utils.train_op import *
import time


class Environment(object):

    def __init__(self,dirty_path,target_path):
        self.dirty_path = dirty_path  # 脏数据
        self.target_path = target_path  # 干净数据
        self.source_image = None
        self.target_image = None
        self.current_image = None
        self.l2_distance_old = None
        self.dirty_name_list = os.listdir(self.dirty_path)
        self.target_name_list = os.listdir(self.target_path)
        self.sub_action_list = sub_action_list
        self.master_action_list = master_action_list
        self.dim_s = None
        self.dim_a_master = len(master_action_list)
        self.dim_a_sub = len(sub_action_list)
        self.action_trajectory = []
        self.done = False

        self.gray_action_list = gray_action_list
        self.hue_action_list = hue_action_list
        self.saturation_action_list = saturation_action_list
        self.whitebalance_action_list = whitebalance_action_list

        self.sub_policy_list = []
        self.sub_policy_list.append(gray_action_list)
        self.sub_policy_list.append(hue_action_list)
        self.sub_policy_list.append(saturation_action_list)
        self.sub_policy_list.append(whitebalance_action_list)




    def reset(self):
        target_name = random.choice(self.target_name_list)
        self.img_name = target_name
        index_str = target_name[0:target_name.find('.')]
        for name in self.dirty_name_list:
            index = name[0:name.find('_')]
            if index == index_str:
                dirty_name = name
                break
        self.dirty_image = load_image2numpy(self.dirty_path+dirty_name) # 脏数据
        self.target_image = load_image2numpy(self.target_path+target_name) # 干净数据
        self.current_image = self.dirty_image
        self.l2_distance_old = compute_color_l2(current_image=self.dirty_image,target_image=self.target_image)
        self.action_trajectory = []
        self.done = False
        # result = np.clip(self.current_image*255.0,a_min=0.0,a_max=255.0).astype(np.uint8)
        return self.current_image ,self.get_color_feature(self.current_image),self.get_gray_feature(self.current_image)


    def take_sub_action(self,action_index):
        self.action_trajectory.append(self.sub_action_list[action_index])
        result = self.sub_action_list[action_index](self.current_image)
        new_l2_distance = compute_color_l2(current_image=result, target_image=self.target_image)
        reward = self.l2_distance_old - new_l2_distance
        self.l2_distance_old = new_l2_distance
        self.current_image = result
        #print("sub_distance: ",new_l2_distance)
        if new_l2_distance < TERMINAL_THRESHOLD:
            self.done = True
        # result = np.clip(result*255.0,a_min=0.0,a_max=255.0).astype(np.uint8)
        return result,self.get_color_feature(result),self.get_gray_feature(result),reward,self.done


    def take_master_action(self, action_index):
        self.action_trajectory.append(self.master_action_list[action_index])
        result = self.master_action_list[action_index](self.current_image)
        new_l2_distance = compute_color_l2(current_image=result, target_image=self.target_image)
        reward = self.l2_distance_old - new_l2_distance
        self.l2_distance_old = new_l2_distance
        self.current_image = result
        #print("master_distance: ",new_l2_distance)
        if new_l2_distance < TERMINAL_THRESHOLD:
            self.done = True
        # result = np.clip(result*255.0,a_min=0.0,a_max=255.0).astype(np.uint8)
        return result,self.get_color_feature(result),self.get_gray_feature(result),reward,self.done


    def take_action(self,action_index,policy_index):
        result = self.sub_policy_list[policy_index][action_index](self.current_image)

        new_l2_distance = compute_color_l2(current_image=result, target_image=self.target_image)
        reward = self.l2_distance_old - new_l2_distance

        self.l2_distance_old = new_l2_distance
        self.current_image = result
        if new_l2_distance < TERMINAL_THRESHOLD:
            self.done = True
        # result = np.clip(result*255.0,a_min=0.0,a_max=255.0).astype(np.uint8)
        return result,self.get_color_feature(result),self.get_gray_feature(result),reward,self.done



    def show(self):
        show_image(self.current_image)

    def show_target(self):
        show_image(self.target_image)

    def get_action_trajectory(self):
        return  self.action_trajectory

    def get_color_feature(self,image):
        channel_one,channel_two,channel_three = distribution_color(image)
        result = np.concatenate([channel_one,channel_two,channel_three],axis=0).reshape([1,-1])[0]
        # print("color_feature:",result)
        return result

    def get_gray_feature(self,image):
        result = distribution_gray(image).reshape([1,-1])[0]
        # print("gray_feature:",result)
        return result

    def save_env_image(self,success,epi):
        time_current = time.strftime("[%Y-%m-%d-%H-%M-%S]", time.localtime(time.time()))
        if success:
            save_image(self.current_image, filepath=result_save_path +str(epi)+'_'+'Success_'+ str(time_current) + '_No:'+ self.img_name )
        else:
            save_image(self.current_image, filepath=result_save_path +str(epi)+'_'+'False_'  + str(time_current) + '_No:'+ self.img_name )




if __name__ == "__main__":
       env = Environment(target_path=root + 'data/source_data/',source_path=root + 'data/train_data/')
       env.reset()