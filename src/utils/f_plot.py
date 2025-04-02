import numpy as np
import matplotlib.pyplot as plt


def f_plot(scores, file_path="figs"):

    for key, value in scores.items():
        num_episodes = value.shape[1]
        mean = np.mean(value, axis=0)
        std = np.std(value, axis=0)
        plt.plot(mean, label=key)
        plt.fill_between(range(num_episodes), mean-std, mean+std, alpha=0.3)
    
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Accumulated Reward")
    plt.savefig(f"{file_path}/Reward.pdf")
    plt.show()

