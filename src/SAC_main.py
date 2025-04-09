import numpy as np
import matplotlib.pyplot as plt
import yaml

from src.components.UAV import UAV
from src.components.BS import BS
from src.components.users import Users

from src.utils.BS2UAV import BS2UAV
from src.utils.UAV2Users import UAV2Users
from src.utils.Environment import Environment

from src.SAC.SAC_tf.Agent import Agent

if __name__ == "__main__":

    my_BS = BS(x=0, y=0, z=0, N_s=6, N_T=144,
               P_t=30, mean_AAoD=120, sep_AAoD=10,
               mean_EAoD=60, sep_EAoD=10)
    
    my_UAV = UAV(x=50, y=50, z=20, N_r=144, N_s=6,
                 N_t=144, P_t=30, mean_AAoA=120,
                 sep_AAoA=10, mean_EAoA=60,
                 sep_EAoA=10, mean_AAoD=120,
                 sep_AAoD=10, mean_EAoD=60,
                 sep_EAoD=10)
    
    my_users = Users(num_users=4, x_min=92.5,
                     y_min=72.5, z_min=1.5,
                     x_max=97.5, y_max=77.5,
                     z_max=2)
    
    my_BS2UAV = BS2UAV(my_BS=my_BS, my_UAV=my_UAV, B_W=5e8,
                      num_path=10, f_c=28, path_loss=3.6,
                      rep=70)
    
    my_UAV2Users = UAV2Users(my_UAV=my_UAV, my_users=my_users, B_W=5e8,
                      num_path=10, f_c=28, path_loss=3.6,
                      rep=70)
    
    env = Environment(my_UAV2Users=my_UAV2Users, rate_thr=18)

    
    
    n_runs = 5
    n_games = 500
    n_timesteps = 200
    
    
    for run in n_runs:

        print(f'********** run = {run} **********')
        best_score = -np.inf
        score_history = []
        load_checkpoint = False


        agent = Agent(input_dims=(2, ), alpha=0.001, beta=0.001,
        tau=0.01, n_actions=2)
        

        if load_checkpoint:
            agent.load_models()


        for i in range(n_games):
            my_UAV.set_location(np.random.uniform(50, 100), np.random.uniform(50, 100))
            observation = my_UAV.location[0, 0:2].reshape(1, 2)/100
            score = 0
            done = False
            for s in range(n_timesteps):
                action = agent.choose_action(observation)
                observation_, reward, done = env.step(action)
                score += reward
                agent.remember(observation, action, reward, observation_, done)
                if not load_checkpoint:
                    agent.learn()
                observation = observation_
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                if not load_checkpoint:
                    agent.save_models()
            print(my_UAV.location)
            print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    agent.load_models()
    location = []

    my_UAV.set_location(55, 75)
    observation = my_UAV.location[0, 0:2].reshape(1, 2)/100
    score = 0
    done = False
    for s in range(200):
        location.append(observation)
        action = agent.choose_action(observation)
        observation_, reward, done = env.step(action)
        score += reward
        agent.remember(observation, action, reward, observation_, done)
        if not load_checkpoint:
            agent.learn()
        observation = observation_
