import numpy as np


class BS2UAV:
    def __init__(self, my_BS, my_UAV, B_W=5e8, ref_path_loss=61.34,\
                 num_path=10, f_c=28, path_loss=3.6, rep=100):
        self.my_UAV = my_UAV
        self.my_BS = my_BS
        self.noise_PSD = 174 + 10 * np.log10(B_W) 
        self.ref_path_loss = ref_path_loss
        self.num_path = num_path
        self.path_loss = path_loss
        self.PL_dB = 32.4 + 20 * np.log10(f_c)
        self.rep = rep

    def db2pow(self, pow_dB):
        return np.power(10, pow_dB/10)