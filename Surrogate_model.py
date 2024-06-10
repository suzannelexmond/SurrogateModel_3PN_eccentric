from RB_and_EIM import *

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
            

class Training_set(Empirical_Interpolation_Method, Reduced_basis, Dataset):
    
    def __init__(self, eccmin_list, waveform_size=None, total_mass=10, mass_ratio=1, freqmin=5):
        
        self.Dphase_training = None
        self.Damp_training = None
        self.TS_training = None

        Dataset.__init__(self, eccmin_list=eccmin_list, waveform_size=waveform_size, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin)
        Simulate_Inspiral.__init__(self, eccmin=None, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin, waveform_size=waveform_size)
        Waveform_properties.__init__(self, eccmin=None, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin, waveform_size=waveform_size)
        Reduced_basis.__init__(self, eccmin_list, waveform_size = waveform_size, total_mass=10, mass_ratio=1, freqmin=5)

        # print(f"Initialized Dataset with waveform_size={waveform_size}")

    # def __init__(self, eccmin_list, waveform_size=None, total_mass=10, mass_ratio=1, freqmin=5):
            
    #     self.Dphase_training = None
    #     self.Damp_training = None
    #     self.TS_training = None
        
    #     # Initialize all parent classes
    #     super(Training_set, self).__init__(eccmin_list, waveform_size, total_mass, mass_ratio, freqmin)

    #     print(f"Initialized Dataset with waveform_size={waveform_size}")

    def generate_training_set(self, property='Phase', save_dataset=False, eccmin_list=None, val_vecs_num=False):
        
        hp_TS, hc_TS, self.TS_M = self.import_polarisations(save_dataset, eccmin_list)
        reduced_basis_hp = self.reduced_basis(hp_TS)

        if val_vecs_num is False:
            residual, self.TS_M = self.import_waveform_property(property, save_dataset, eccmin_list)
        else:
            residual = self.calc_validation_vectors(val_vecs_num, save_dataset, property)
            
        empirical_nodes = self.calc_empirical_nodes(reduced_basis_hp, self.TS_M)
        
        residual_training = np.zeros((residual.shape[0], len(empirical_nodes)))
        time_training = np.zeros((residual.shape[0], len(empirical_nodes)))

        for i in range(len(self.eccmin_list)):
            residual_training[i] = residual[i][empirical_nodes]
        
        self.TS_training = self.TS_M[empirical_nodes]

        if property == 'Phase':
            self.Dphase_training = residual_training
        elif property == 'Amplitude':
            self.Damp_training = residual_training
        else:
            print('Choose property= "Phase" or "Amplitude"')

        return residual_training, empirical_nodes
    
    
    def plot_training_set_at_node(self, time_node_idx, save_dataset=False):

        if self.Damp_training is None:
            self.generate_training_set('Amplitude', save_dataset)
        
        if self.Dphase_training is None:
            self.generate_training_set('Phase', save_dataset)

        fig_training_set, ax1 = plt.subplots(figsize=(8, 8))
        ax2 = ax1.twinx()

        ax1.scatter(self.eccmin_list, self.Dphase_training.T[time_node_idx], color='black')
        ax2.scatter(self.eccmin_list, self.Damp_training.T[time_node_idx], color='black')
        
        ax1.plot(self.eccmin_list, self.Dphase_training.T[time_node_idx], color='orange')
        ax2.plot(self.eccmin_list, self.Damp_training.T[time_node_idx], color='blue')

        ax1.set_xlabel("eccentricity")
        ax1.set_ylabel(f"$\Delta\phi$ at T_{time_node_idx}", fontsize=14)
        ax1.tick_params(axis="y", labelcolor='orange')

        ax2.set_ylabel(f"$\Delta$A at T_{time_node_idx} [M]", fontsize=14)
        ax2.tick_params(axis="y", labelcolor='blue')

        plt.grid()
        # plt.show()

    def largest_pointwise_error(self, time_node_idx, validation_num, save_dataset=False):

        if self.Damp_training is None:
            self.generate_training_set('Amplitude', save_dataset)
        
        if self.Dphase_training is None:
            self.generate_training_set('Phase', save_dataset)


        parameter_space = np.linspace(np.min(self.eccmin_list), np.max(self.eccmin_list), num=500)
        validation_params = np.random.choice(parameter_space, size=validation_num, replace=False)
        
        validation_set_amp = self.generate_training_set('Amplitude', save_dataset=True, eccmin_list=validation_params)[0]
        validation_set_phase = self.generate_training_set('Phase', save_dataset=True, eccmin_list=validation_params)[0]

        amp_error, phase_error = [], []

        for i in range(len(self.eccmin_list)):
            amp_error = np.abs((self.Damp_training.T[i][time_node_idx] - validation_set_amp.T[i])/self.Damp_training.T[i][time_node_idx])
            phase_error = np.abs(self.Dphase_training.T[i][time_node_idx] - validation_set_phase.T[i])

        fig_relative_errors, ax1 = plt.subplots(figsize=(8, 8))
        ax2 = ax1.twinx()

        ax1.plot(self.eccmin_list, phase_error, color='orange')
        ax2.plot(self.eccmin_list, amp_error, color='blue')

        ax1.set_xlabel("eccentricity")
        ax1.set_ylabel("$\Delta\phi$ error at T_{time_node_idx} [M]", fontsize=14)
        ax1.tick_params(axis="y", labelcolor='orange')

        ax2.set_ylabel("$\Delta$A error at T_{time_node_idx} [M]", fontsize=14)
        ax2.tick_params(axis="y", labelcolor='blue')

        plt.grid()

    
    def gaussian_process_regression(self, time_node_idx, training_set):

        X = np.linspace(min(self.eccmin_list), max(self.eccmin_list), self.waveform_size)[:, np.newaxis]
        
        X_train = self.eccmin_list.reshape(-1, 1)
        y_train = np.squeeze(training_set.T[time_node_idx])
        
        kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        gaussian_process.fit(X_train, y_train)
        gaussian_process.kernel_

        mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

        return mean_prediction, [(mean_prediction - 1.96 * std_prediction), (mean_prediction + 1.96 * std_prediction)]


    def plot_gaussian_process_regression(self, time_node_idx, property='Phase', save_dataset=False):
        
        X_train = self.eccmin_list.reshape(-1, 1)
        X = np.linspace(min(self.eccmin_list), max(self.eccmin_list), 1000)[:, np.newaxis]

        if property == 'Phase':
            if self.Dphase_training is None:
                residual_training = self.generate_training_set('Phase', save_dataset)[0]
            else:
                residual_training = self.Dphase_training

            mean_prediction, uncertainty_region = self.gaussian_process_regression(time_node_idx, residual_training)
            residual_training = residual_training.T[time_node_idx]

        elif property == 'Amplitude':
            if self.Damp_training is None:
                residual_training = self.generate_training_set('Amplitude', save_dataset)[0]*1e26

            else:
                residual_training = self.Damp_training*1e26

            mean_prediction, uncertainty_region = self.gaussian_process_regression(time_node_idx, residual_training)
            
            residual_training = residual_training.T[time_node_idx]*1e-26
            mean_prediction = mean_prediction*1e-26
            uncertainty_region = uncertainty_region[0]*1e-26, uncertainty_region[1]*1e-26
        else:
            print('Choose property = "Phase" or "Amplitude"')
            sys.exit(1)

        
        fig_GPR = plt.figure()

        plt.scatter(X_train, residual_training, color='red', label="Observations", s=10)
        plt.plot(X, mean_prediction, label="Mean prediction", linewidth=0.8)
        plt.fill_between(
            X.ravel(),
            uncertainty_region[0],
            uncertainty_region[1],
            alpha=0.5,
            label=r"95% confidence interval",
        )
        plt.legend()
        plt.xlabel("$e$")
        plt.ylabel("$f(e)$")
        # _ = plt.title(f"GPR {property} on T_{time_node_idx}")
        # plt.show()

    def generate_surrogate_waveform(self, polarisation='plus', save_dataset=False):
        hp_DS, hc_DS, TS_M = self.import_polarisations(save_dataset)

        if polarisation == 'plus':
            reduced_basis = self.reduced_basis(hp_DS)
        elif polarisation == 'cross':
            reduced_basis = self.reduced_basis(hc_DS)
        else:
            print('Choose polarisation ="plus" or "cross"')
            sys.exit(1)

        # Generate training set for amplitude and phase
        res_amp_training, emp_nodes_idx = self.generate_training_set('Amplitude', save_dataset)
        res_phase_training = self.generate_training_set('Phase', save_dataset)[0]

        # Create empty arrays for best approximation ampitude and phase with length m
        m = len(emp_nodes_idx)
        A = np.zeros((m, self.waveform_size))
        phi = np.zeros((m, self.waveform_size))
        
        # Create surrogate model amplitude and phase arrays for every time node
        for node in range(m):
            # print(type(res_amp_training), res_amp_training.shape)
            A[node] = self.gaussian_process_regression(node, res_amp_training)[0]
            phi[node] = self.gaussian_process_regression(node, res_phase_training)[0]

        #B_i(t) = sum_{j=i}^m e_j(t)(V^-1)_ji for e_j and T_i
        B_j_vec = np.zeros((reduced_basis.shape[1], m))

        V = np.zeros((m, m))
        for j in range(m):
            for i in range(m):
                V[j][i] = reduced_basis[i][emp_nodes_idx[j]]

        for j in range(V.shape[1]): 
            B_j = 0
            for i in range(m):
                B_j += reduced_basis[i].T * np.linalg.inv(V)[i][j]

            B_j_vec[:, j] = B_j



            

        def calculate_surrogate_waveform(B, A, phi):
            # Switch back from residuals to actual amplitude and phase
            # Run circulair waveform properties
            self.circulair_wf()
            amp_circ = np.array(waveform.utils.amplitude_from_polarizations(self.hp_TS_circ, self.hc_TS_circ))
            phase_circ = np.array(waveform.utils.phase_from_polarizations(self.hp_TS_circ, self.hc_TS_circ))
            
            surrogate = 0
            for i in range(m):
                # element-wise exponential with imaginary unit
                y = (A[i] + amp_circ) * np.exp(-1j * (phi[i] + phase_circ))
                y = y.reshape(-1, 1).T
                surrogate += np.dot(B[:, i].reshape(-1, 1), y)
                
            return surrogate
              

        
        surrogate = calculate_surrogate_waveform(B_j_vec, A, phi)

        
        plt.figure(figsize=(8, 5))
        plt.plot(self.TS_M, surrogate[:, 0])
        plt.xlabel('t [M]')
        plt.grid(True)
        plt.show()
        # Build the surrogate
        # h_surrogate = 

# wp_05 = Waveform_properties(eccmin=0.5)
# sim = Simulate_Inspiral(0.181)
# sim.plot_sim_inspiral_mass_indp()
# sim.plot_sim_inspiral_mass_indp_multiple([50], [1], [0.181])
# wp_01 = Waveform_properties(eccmin=0.1)

# wp_05.plot_residuals(property='Amplitude')
# wp_01.plot_residuals(property='Amplitude')
# wp_05.plot_residuals(property='Phase')
# wp_01.plot_residuals(property='Phase')


# ds = Dataset(eccmin_list=np.linspace(0.01, 0.2, num=20).round(3), waveform_size=100000)
# ds.plot_dataset_polarisations()


# EMP = Empirical_Interpolation_Method(eccmin_list=np.linspace(0.01, 0.2, num=20).round(3), waveform_size=1000)
# EMP.plot_empirical_nodes(save_dataset=True)

# RB = Reduced_basis(eccmin_list=np.linspace(0.01, 0.5, num=20).round(3), waveform_size=100000)
# RB.plot_dataset_properties('Phase', save_dataset=True)
# RB.plot_dataset_properties('Amplitude', save_dataset=True)

TS = Training_set(eccmin_list=np.linspace(0.01, 0.2, num=20).round(3), waveform_size=1000)
# TS.plot_empirical_nodes(polarisation='plus', save_dataset=True)
# TS.plot_training_set_at_node(10)
# TS.plot_gaussian_process_regression(10, 'Phase', save_dataset=True)
# TS.plot_gaussian_process_regression(10, 'Amplitude', save_dataset=True)


TS.generate_surrogate_waveform('plus')

# plt.show()

# Example usage
# training_set = Training_set([0.1, 0.2, 0.3], waveform_size=1000, total_mass=10, mass_ratio=1, freqmin=5)