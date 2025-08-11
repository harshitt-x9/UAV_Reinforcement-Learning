import numpy as np

from src.components.UAV import UAV
from src.components.BS import BS
from src.components.users import Users

from src.utils.BS2UAV import BS2UAV
from src.utils.UAV2Users import UAV2Users
from src.utils.Environment import Environment

from src.DDPG.Agent import Agent

from src.utils.f_plot import f_plot

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
                      rep=100)
    
    my_UAV2Users = UAV2Users(my_UAV=my_UAV, my_users=my_users, B_W=5e8,
                      num_path=10, f_c=28, path_loss=3.6,
                      rep=100)
    
    env = Environment(my_UAV2Users=my_UAV2Users, rate_thr=18)

    
    n_runs = 2
    n_games = 5
    T = 10
    min_step = -1.0
    max_step = 1.0
    score_dict = dict()

    alphas = [0.0001, 0.001, 0.01]
    for alpha in alphas:
        score_dict[f"$\\alpha$={alpha}, $\\beta$={alpha * 2}"] = np.zeros((n_runs, n_games))

        for run in range(n_runs):
            agent = Agent(input_dims=(2, ), alpha=alpha, beta=2*alpha, number=1)

            temp_score = []
            avg_score = []
            best_score = float('-inf')
            load_checkpoint = False

            if load_checkpoint:
                n_steps = 0
                while n_steps <= agent.batch_size:
                    observation = my_UAV.location[0, 0:2].reshape(1, 2)/100 
                    action = agent.choose_action(observation, False)
                    observation_, reward, done = env.step(action)
                    agent.store_transition(observation, action, reward, observation_, done)
                    n_steps += 1
                agent.learn()
                agent.load_models()
                evaluate = True
            else:
                evaluate = False

            for i in range(n_games):
                my_UAV.set_location(np.random.uniform(50, 100), np.random.uniform(50, 100))
                observation = my_UAV.location[0, 0:2].reshape(1, 2)/100
                score = 0
                done = False
                for s in range(T):
                    action = agent.choose_action(observation, evaluate)
                    observation_, reward, done = env.step(action)
                    score += reward
                    agent.store_transition(observation, action, reward, observation_, done)
                    if not load_checkpoint:
                        agent.learn()
                    observation = observation_
                score_dict[f"$\\alpha$={alpha}, $\\beta$={alpha * 2}"][run, i] = score
                temp_score.append(score)
                avg_score.append(np.mean(temp_score[-100:]))
                if avg_score[-1] > best_score:
                    best_score = avg_score[-1]
                    print("best_score")
                    opt_location = my_UAV.location[0, :2]
                    if not load_checkpoint:
                        agent.save_models()
                print(my_UAV.location)
                print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score[i])

f_plot(score_dict)
