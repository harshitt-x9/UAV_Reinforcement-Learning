import numpy as np


class UAV:
    def __init__(self, x=50, y=50, z=50, N_r=144, N_s=7, N_t=144,
                 P_t=30, mean_AAoA=120, sep_AAoA=10, mean_EAoA=60,
                 sep_EAoA=10, mean_AAoD=120, sep_AAoD=10, mean_EAoD=60,
                 sep_EAoD=10, user_range=np.array([0.1, 0.9]), ID_SimScenario=1,
                 Psi_g=np.array([15, 55, 95, 135]), Delta=15):
        self.location = np.array([[x, y, z]], dtype='float32')
        self.N_r = N_r
        self.N_s = N_s
        self.N_t = N_t
        self.P_t = P_t
        self.mean_AAoA = mean_AAoA
        self.sep_AAoA = sep_AAoA
        self.mean_EAoA = mean_EAoA
        self.sep_EAoA = sep_EAoA
        self.mean_AAoD = mean_AAoD
        self.sep_AAoD = sep_AAoD
        self.mean_EAoD = mean_EAoD
        self.sep_EAoD = sep_EAoD
        self.Psi_g = np.array([Psi_g, Psi_g + Delta]).T
        self.Theta, _ = self.f_findElevation(user_range, ID_SimScenario)


    def update_location(self, mov_x, mov_y):
        self.location[0, 0] += mov_x
        self.location[0, 1] += mov_y

    def f_findElevation(self, UserRange, ID_SimScenario):
        # UMI
        if ID_SimScenario == 1:
            hBS = 10
            hUT_1 = [1.5, 2.5]
            ISD = 200
        # UMa
        elif ID_SimScenario == 2:
            hBS = 25
            hUT_1 = [1.5, 2.5]
            ISD = 500

        Radius = ISD / 2
        d2D_lim = Radius * UserRange
        d2D_lim_1 = d2D_lim.tolist()
        fra = 1e-3
        d2D = (np.tile(np.linspace(d2D_lim_1[0], d2D_lim_1[1], num=1001, dtype=float), (int(1.0 / fra + 1.0), 1))).tolist()
        d2D = np.array(d2D)
        hUT = (np.tile(np.linspace(hUT_1[0], hUT_1[1], num=1001, dtype=float), (int(1.0 / fra + 1.0), 1))).tolist()
        hUT = np.transpose(hUT)
        d3D = np.sqrt(np.square(d2D) + np.square(hBS - hUT))
        Theta_1 = [np.min(np.min(np.degrees(np.arcsin(d2D / d3D)))), np.max(np.max(np.degrees(np.arcsin(d2D / d3D))))]
        dist_1 = [np.min(d3D.flatten()), np.max(d3D.flatten())]
        return Theta_1, dist_1


    def calc_f_ur(self):
        N_rf = 0
        ID_N_rf = np.zeros(self.N_t)
        M_Rx = np.sqrt(self.N_t)
        M_Ry = np.sqrt(self.N_t)
        sample_cluster = 100
        
    
        # Calculate step size for the range of angles
        theta_step = (2 * self.sep_EAoA) / (sample_cluster - 1)
        phi_step = (2 * self.sep_AAoA) / (sample_cluster - 1)

        # Generate the angular ranges for theta_r and phi_r
        theta_range = np.arange(-self.sep_EAoA, self.sep_EAoA + theta_step, theta_step)
        phi_range = np.arange(-self.sep_AAoA, self.sep_AAoA + phi_step, phi_step)

        # Calculate the final theta_r and phi_r
        theta_r = (self.mean_EAoA + theta_range)[:, np.newaxis]
        phi_r = self.mean_AAoA + phi_range

        ## Gamma
        gamma_xr = np.zeros((sample_cluster, sample_cluster), dtype='float')
        gamma_yr = np.zeros((sample_cluster, sample_cluster), dtype='float')
        gamma_xr[0: sample_cluster+1, :] = np.sin(theta_r * np.pi / 180.0) * np.cos(phi_r * np.pi / 180.0)
        gamma_yr[0: sample_cluster+1, :] = np.sin(theta_r * np.pi / 180.0) * np.sin(phi_r * np.pi / 180.0)   

        ## lambda
        lambda_xr = -1 + (2 * np.arange(1, M_Rx + 1) - 1) / M_Rx
        lambda_yr = -1 + (2 * np.arange(1, M_Ry + 1) - 1) / M_Ry
        [lambda_xr, lambda_yr] = np.meshgrid(lambda_xr, lambda_yr)
        lambda_xr = lambda_xr.flatten(order='F')[:, np.newaxis]
        lambda_yr = lambda_yr.flatten(order='F')[:, np.newaxis]
        for i in range(self.N_t):
            lambda_xr_low = lambda_xr[i] - (1 / M_Rx)
            lambda_xr_high = lambda_xr[i] + (1 / M_Rx)
            lambda_yr_low = lambda_yr[i] - (1 / M_Ry)
            lambda_yr_high = lambda_yr[i] + (1 / M_Ry)
            XX = (lambda_xr_low < gamma_xr) * (lambda_xr_high > gamma_xr)
            YY = (lambda_yr_low < gamma_yr) * (lambda_yr_high > gamma_yr)
            if np.sum(np.sum((1 * XX) * (1 * YY))) > 0:
                N_rf = N_rf + 1
                ID_N_rf[N_rf - 1] = i
        ID_N_rf = ID_N_rf[0:N_rf]

        # Generate TX-F-BF
        x = np.arange(0, M_Rx, dtype=int)
        y = np.arange(0, M_Ry, dtype=int)
        [x, y] = np.meshgrid(x, y)
        x = x.flatten(order='F')
        y = y.flatten(order='F')
        F = np.sqrt(1 / self.N_t) * (np.exp(
            -1j * np.pi * (x * (lambda_xr[ID_N_rf.astype(int)]) + (y * (lambda_yr[ID_N_rf.astype(int)])))))
        self.f_ur = F
    
    def calc_b_ur(self, U_1):
        self.b_ur = U_1.conj().T
        return self.b_ur

    def calc_b_ut(self, H_eff_2, noise_PSD, K, P_IWF):
        alpha = np.power(10, (noise_PSD - 30)/10) / np.power(10, (self.P_t - 30)/10)
        W = np.linalg.pinv((H_eff_2.conj().T @ H_eff_2) + (K * alpha * np.eye(np.size(H_eff_2, 1))))
        B = W @ H_eff_2.conj().T @ P_IWF
        Eps = np.sqrt(np.power(10, (self.P_t - 30)/10) * (1 / (np.real(np.trace(B.conj().T @ self.f_ut.conj().T @ self.f_ut @ B)))))
        B = Eps * W @ H_eff_2.conj().T @ P_IWF
        self.b_ut = B
        return self.b_ut

    def find_NRF_g(self, lambda_x, lambda_y, gamma_x, gamma_y, M_Tx, M_Ty):
        ID_g = np.zeros(M_Tx * M_Ty)
        NRF_g = 0
        ## special cases M_Tx == 1 & M_Ty == 1 not supported
        for i in range(M_Tx * M_Ty):
            lambda_x_low = lambda_x[i] - (1 / M_Tx)
            lambda_x_high = lambda_x[i] + (1 / M_Tx)
            lambda_y_low = lambda_y[i] - (1 / M_Ty)
            lambda_y_high = lambda_y[i] + (1 / M_Ty)

            XX = (lambda_x_low < gamma_x) & (lambda_x_high > gamma_x)
            YY = (lambda_y_low < gamma_y) & (lambda_y_high > gamma_y)

            if np.sum(XX & YY) > 0:
                ID_g[NRF_g] = i  # Adjusting for index starting at 1
                NRF_g += 1
        return NRF_g, ID_g[:NRF_g]

    def find_NRF(self, M_Tx, M_Ty, Theta, Psi_g):

        frac = 1e-2

        lambda_x = (-1 + (2 * np.arange(1, M_Tx + 1) - 1) / M_Tx)
        lambda_y = (-1 + (2 * np.arange(1, M_Ty + 1) - 1) / M_Ty)
        lambda_x = np.tile(lambda_x, (M_Ty, 1)).flatten(order='F')[:, np.newaxis]
        lambda_y = np.tile(lambda_y, (1, M_Tx)).T.flatten()[:, np.newaxis]

        theta_g = np.arange(Theta[0], Theta[1] + frac * (Theta[1] - Theta[0]), frac * (Theta[1] - Theta[0])).reshape((101, 1))
        psi_g = np.arange(Psi_g[0], Psi_g[1] + frac * (Psi_g[1] - Psi_g[0]), frac * (Psi_g[1] - Psi_g[0])).reshape((1, 101))


        gamma_x = np.sin(theta_g * np.pi / 180.) @ np.cos(psi_g * np.pi / 180.)
        gamma_y = np.sin(theta_g * np.pi / 180.) @ np.sin(psi_g * np.pi / 180.)
        
    # Flatten gamma_x and gamma_y to make them 1D arrays
        gamma_x = gamma_x.flatten(order='F')
        gamma_y = gamma_y.flatten(order='F')

        NRF, ID = self.find_NRF_g(lambda_x, lambda_y, gamma_x, gamma_y, M_Tx, M_Ty)

        gamma_x = [np.min(gamma_x), np.max(gamma_x)]
        gamma_y = [np.min(gamma_y), np.max(gamma_y)]

        return NRF, ID

    def calc_f_ut(self):
        
        M_Tx = int(np.sqrt(self.N_t))
        M_Ty = int(np.sqrt(self.N_t))
        NRF, ID_g = self.find_NRF(M_Tx, M_Ty, self.Theta, self.Psi_g[0])

        ## lambda
        lambda_xt = -1 + (2 * np.arange(1, M_Tx + 1) - 1) / M_Tx
        lambda_yt = -1 + (2 * np.arange(1, M_Ty + 1) - 1) / M_Ty
        [lambda_xt, lambda_yt] = np.meshgrid(lambda_xt, lambda_yt)
        lambda_xt = lambda_xt.flatten(order='F')[:, np.newaxis]
        lambda_yt = lambda_yt.flatten(order='F')[:, np.newaxis]

        # Generate TX-F-BF
        x = np.arange(0, M_Tx, dtype=int)
        y = np.arange(0, M_Ty, dtype=int)
        [x, y] = np.meshgrid(x, y)
        x = x.flatten(order='F')[:, np.newaxis]
        y = y.flatten(order='F')[:, np.newaxis]
        F = np.sqrt(1 / self.N_t) * (np.exp(
        1j * np.pi * (x * ((lambda_xt[ID_g.astype(int)]).transpose()) + \
                       (y * ((lambda_yt[ID_g.astype(int)]).transpose())))))
        self.f_ut = F
