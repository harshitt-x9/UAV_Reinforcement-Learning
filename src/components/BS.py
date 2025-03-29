import numpy as np

class BS:
    def __init__(self, x, y, z, N_s, N_rf_b, N_T):
        self.location = np.array([x, y, z]).reshape((1, 3))
        self.N_s = N_s
        self.N_rf_b = N_rf_b
        self.N_T = N_T