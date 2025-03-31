import numpy as np
import matplotlib.pyplot as plt


class UAV2Users():
    def __init__(self, my_UAV, my_users, B_W=5e8,
                  ref_path_loss=61.34, num_path=10,
                    f_c=28, path_loss=3.6, rep=100):
        self.my_UAV = my_UAV
        self.my_users = my_users
        self.num_users = my_users.locations.shape[0]
        self.noise_PSD = -174 + 10 * np.log10(B_W) 
        self.ref_path_loss = ref_path_loss
        self.num_path = num_path
        self.path_loss = path_loss
        self.PL_dB = 32.4 + 20 * np.log10(f_c)
        self.rep = rep

    def db2pow(self, pow_dB):
        return np.power(10, pow_dB/10)  
    
    def f_gen_phase_xy(self, Psi_g_1, Theta):
        Theta_gp_1 = (Theta[1] - Theta[0]) * np.random.rand(self.num_path, 1) + Theta[0]
        Psi_g_1 = (Psi_g_1[1] - Psi_g_1[0]) * np.random.rand(self.num_path, 1) + Psi_g_1[0]
        gamma_xgp_1 = np.sin(Theta_gp_1 * np.pi / 180.) * np.cos(Psi_g_1 * np.pi / 180.)
        gamma_ygp_1 = np.sin(Theta_gp_1 * np.pi / 180.) * np.sin(Psi_g_1 * np.pi / 180.)
        return gamma_xgp_1, gamma_ygp_1

    def f_gen_channel2(self):
        Z = np.zeros((self.num_users, self.num_path), dtype='complex')
        A = np.zeros((self.num_path, self.my_UAV.N_t), dtype='complex')
        Z[:, :] = np.linalg.norm(self.my_users.locations - self.my_UAV.location, axis=1)[:, np.newaxis]
        ## path gain
        Kg, Path = Z.shape
        Z = Z ** (-self.path_loss)
        Z = np.sqrt(1 / (2 * self.num_path)) * (np.random.randn(Kg, Path) + 1j * np.random.randn(Kg, Path)) * Z
        ## phase matrix
        [Gamma_xgp, Gamma_ygp] = self.f_gen_phase_xy(self.my_UAV.Psi_g[0], self.my_UAV.Theta)
        M_Tx = int(np.sqrt(self.my_UAV.N_t))
        M_Ty = int(np.sqrt(self.my_UAV.N_t))
        x = np.arange(0, M_Tx, dtype=int)
        x = np.tile(x, (M_Tx, 1))
        x = x.reshape(1, M_Tx * M_Ty, order='F')
        y = np.arange(0, M_Ty, dtype=int)
        y = np.tile(y, (1, M_Ty))
        A = np.exp(-1j * np.pi * (Gamma_xgp * x)) * np.exp(-1j * np.pi * (Gamma_ygp * y))
        H_2 = Z @ A

        return H_2
    
    def f_sim_OFDM_EQ(self):
        # Generate BB Precoder
        H_2 = self.f_gen_channel2()
        F_ut = self.my_UAV.f_ut
        H_eff = H_2 @ F_ut
        B_ut = self.my_UAV.calc_b_ut(H_eff, self.noise_PSD, self.num_users, np.eye(self.num_users))
        # Find Powers
        H_eff_ALL = H_2 @ F_ut @ B_ut
        P_gk = np.abs(np.diag(H_eff_ALL)) ** 2
        INT_gk = np.sum(np.abs(H_eff_ALL - np.diag(np.diag(H_eff_ALL))) ** 2, axis=1)
        # SINR
        SINR = P_gk / (INT_gk + self.db2pow(self.noise_PSD - 30))
        # Sum-Rate
        Rate = np.sum(np.log2(1 + SINR))

        return Rate
    
    def f_calc_rate_2(self):
        rate = np.zeros(self.rep)
        # Simulations
        for j in range(self.rep):
            rate[j] = self.f_sim_OFDM_EQ()
        # Rate.Rates = 0.5*np.mean(np.asarray(rate), axis=0)
        Rate = 0.5 * np.mean(rate)

        return Rate
    
    def plot_UAV2Users(self, n_steps=10, n_levels=10, x_min=50,
                        x_max=100, y_min=50, y_max=100,\
                            file_path="figs"):
        
        x = np.linspace(x_min, x_max, n_steps)
        y = np.linspace(y_min, y_max, n_steps)
        X, Y = np.meshgrid(x, y)
        Rate = np.zeros((n_steps, n_steps))

        for i in range(n_steps):
            for j in range(n_steps):
                self.my_UAV.set_location(x[i], y[j])
                Rate[i, j] = self.f_calc_rate_2()

        plt.contourf(Y, X, Rate, levels=n_levels)
        plt.colorbar(label='Achievable Rate [bps/Hz]')
        plt.scatter(my_users.locations[:, 0], my_users.locations[:, 1], color='k',
                    label = "Users")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig("figs/UAV2Users.pdf")
        plt.show()

        

if __name__ == "__main__": 

    from src.components.UAV import UAV
    from src.components.users import Users

    my_UAV = UAV()
    my_users = Users()
    my_UAV2Users = UAV2Users(my_UAV, my_users)
    my_UAV2Users.plot_UAV2Users()
    