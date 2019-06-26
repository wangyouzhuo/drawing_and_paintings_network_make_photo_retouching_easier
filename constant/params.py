from utils.retouch_op import *




LR_A = 0.0001
LR_C = 0.00015

dim_master_action = len(master_action_list)
dim_sub_action = len(sub_action_list)

DEVICE = '/gpu:0'