

# -----------------------------------global_count--------------------------------
def _init_train_count():
    global EPISODE_COUNT
    EPISODE_COUNT = 0

def _add_train_count():
    global EPISODE_COUNT
    EPISODE_COUNT = EPISODE_COUNT + 1
    return EPISODE_COUNT

def _get_train_count():
    global EPISODE_COUNT
    return EPISODE_COUNT


#------------------------------------global_reward-----------------------------------
def _init_reward_list():
    global GLOBAL_REWARD
    GLOBAL_REWARD = []

def _append_reward_list(reward):
    global GLOBAL_REWARD
    GLOBAL_REWARD.append(reward)

def _get_reward_list():
    global GLOBAL_REWARD
    return GLOBAL_REWARD


#------------------------------------global_trajectory-----------------------------------
def _init_trajectory_dict():
    global GLOBAL_TRAJECTORY
    GLOBAL_TRAJECTORY = dict()

def _append_trajectory_dict(epi_count,trajectory):
    global GLOBAL_TRAJECTORY
    GLOBAL_TRAJECTORY[epi_count] = trajectory

def _get_trajectory_dict():
    global GLOBAL_TRAJECTORY
    return GLOBAL_TRAJECTORY


