import numpy as np

class BS:
    def __init__(self, x=0, y=0, z=10, N_s=7, N_rf_b=7, N_T=144, P_t=30,\
                 mean_AAoD=120, sep_AAoD=10, mean_EAoD=60, sep_EAoD=10):
        self.location = np.array([[x, y, z]])
        self.N_s = N_s
        self.N_rf_b = N_rf_b
        self.N_T = N_T
        self.P_t = P_t
        self.mean_AAoD = mean_AAoD
        self.sep_AAoD = sep_AAoD
        self.mean_EAoD = mean_EAoD
        self.sep_EAoD = sep_EAoD

    def calc_f_b(self):
        N_rf = 0
        ID_N_rf = np.zeros(self.N_T)
        M_Tx = np.sqrt(self.N_T)
        M_Ty = np.sqrt(self.N_T)
        sample_cluster = 100

        # Calculate step sizes for theta_t and phi_t ranges
        theta_t_step = (2 * self.sep_EAoD) / (sample_cluster - 1)
        phi_t_step = (2 * self.sep_AAoD) / (sample_cluster - 1)

        # Generate the angular ranges for theta_t and phi_t
        theta_t_range = np.arange(-self.sep_EAoD, self.sep_EAoD + theta_t_step, theta_t_step)
        phi_t_range = np.arange(-self.sep_AAoD, self.sep_AAoD + phi_t_step, phi_t_step)

        # Calculate the final theta_t and phi_t
        theta_t = (self.mean_EAoD + theta_t_range)[:, np.newaxis]
        phi_t = self.mean_AAoD + phi_t_range

        ## Gamma
        gamma_xt = np.zeros((sample_cluster, sample_cluster), dtype='float')
        gamma_yt = np.zeros((sample_cluster, sample_cluster), dtype='float')
        gamma_xt[0: sample_cluster+1, :] = np.sin(theta_t * np.pi / 180.0) * np.cos(phi_t * np.pi / 180.0)
        gamma_yt[0: sample_cluster+1, :] = np.sin(theta_t * np.pi / 180.0) * np.sin(phi_t * np.pi / 180.0)  

        ## lambda
        lambda_xt = -1 + (2 * np.arange(1, M_Tx + 1) - 1) / M_Tx
        lambda_yt = -1 + (2 * np.arange(1, M_Ty + 1) - 1) / M_Ty
        [lambda_xt, lambda_yt] = np.meshgrid(lambda_xt, lambda_yt)
        lambda_xt = lambda_xt.flatten(order='F')[:, np.newaxis]
        lambda_yt = lambda_yt.flatten(order='F')[:, np.newaxis]
        for i in range(self.N_T):
            lambda_xt_low = lambda_xt[i] - (1 / M_Tx)
            lambda_xt_high = lambda_xt[i] + (1 / M_Tx)
            lambda_yt_low = lambda_yt[i] - (1 / M_Ty)
            lambda_yt_high = lambda_yt[i] + (1 / M_Ty)
            XX = (lambda_xt_low < gamma_xt) * (lambda_xt_high > gamma_xt)
            YY = (lambda_yt_low < gamma_yt) * (lambda_yt_high > gamma_yt)
            if np.sum(np.sum((1 * XX) * (1 * YY))) > 0:
                N_rf = N_rf + 1
                ID_N_rf[N_rf - 1] = i
        ID_N_rf = ID_N_rf[0:N_rf]

        # Generate TX-F-BF
        x = np.arange(0, M_Tx, dtype=int)
        y = np.arange(0, M_Ty, dtype=int)
        [x, y] = np.meshgrid(x, y)
        x = x.flatten(order='F')[:, np.newaxis]
        y = y.flatten(order='F')[:, np.newaxis]
        F = np.sqrt(1 / self.N_T) * (np.exp(
        1j * np.pi * (x * ((lambda_xt[ID_N_rf.astype(int)]).transpose()) + \
                       (y * ((lambda_yt[ID_N_rf.astype(int)]).transpose())))))
        self.f_b = F

    def calc_b_b(self, K, V_1):
        self.b_b = np.sqrt(self.P_t/K) * V_1
        return self.b_b 
    
