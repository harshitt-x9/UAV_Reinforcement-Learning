import numpy as np

class Environment:
    def __init__(self, my_UAV2Users, x_min=0, x_max=130, y_min=0,\
         y_max=130, rate_thr=10, penalty=-1):
        
        self.my_UAV2Users = my_UAV2Users
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.rate_thr = rate_thr
        self.penalty = penalty
    
    def step(self, action):
    
        curr_loc = self.my_UAV2Users.my_UAV.location[0, :2].reshape(1, 2)
        done = False
        # (curr_loc[0, 0] + action[0, 0] > 100) or 
        if (curr_loc[0, 0] + action[0, 0] < self.x_min) or \
            (curr_loc[0, 0] + action[0, 0] > self.x_max):
            done = True
        # (curr_loc[0, 0] + action[0, 0] > 100) or 
        if(curr_loc[0, 1] + action[0, 1] < self.y_min) or \
            (curr_loc[0, 1] + action[0, 1] > self.y_max):
            done = True
        if done:
            self.my_UAV2Users.my_UAV.update_location(action[0, 0], action[0, 1])
            # C = f_sim_UAV_sum_power_PSOiter(my_UAV, my_users)
            reward = -5
            curr_loc_ = self.my_UAV2Users.my_UAV.location[0, 0:2].reshape(1, 2)/100
            return curr_loc_, reward, done
        else:
            self.my_UAV2Users.my_UAV.update_location(action[0, 0], action[0, 1])
            reward_ = self.my_UAV2Users.f_calc_rate_2()
            if reward_ < self.rate_thr:
                reward_ = self.penalty
            reward = reward_ 
            curr_loc_ = self.my_UAV2Users.my_UAV.location[0, 0:2].reshape(1, 2)/100
            return curr_loc_, reward, done