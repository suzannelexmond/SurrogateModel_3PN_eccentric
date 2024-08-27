from RB_and_EIM import *

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Product, ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel, ExpSineSquared, ConstantKernel as C
from sklearn.model_selection import train_test_split

import warnings
from sklearn.exceptions import ConvergenceWarning

import re

# Ignore ConvergenceWarning warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
            

class Training_set(Empirical_Interpolation_Method, Reduced_basis, Dataset):
    
    def __init__(self, eccmin_list, waveform_size=None, total_mass=50, mass_ratio=1, freqmin=20):
        
        self.Dphase_training = None
        self.Damp_training = None
        self.TS_training = None

        self.surrogate = None

        self.greedy_params_amp = None
        self.greedy_params_phase = None

        Dataset.__init__(self, eccmin_list=eccmin_list, waveform_size=waveform_size, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin)
        Simulate_Inspiral.__init__(self, eccmin=None, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin, waveform_size=waveform_size)
        Waveform_properties.__init__(self, eccmin=None, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin, waveform_size=waveform_size)
        Reduced_basis.__init__(self, eccmin_list, waveform_size = waveform_size, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin)

        self.parameter_space = np.linspace(min(self.eccmin_list), max(self.eccmin_list), 500).round(4)

        # print(f"Initialized Dataset with waveform_size={waveform_size}")

    # def __init__(self, eccmin_list, waveform_size=None, total_mass=10, mass_ratio=1, freqmin=5):
            
    #     self.Dphase_training = None
    #     self.Damp_training = None
    #     self.TS_training = None
        
    #     # Initialize all parent classes
    #     super(Training_set, self).__init__(eccmin_list, waveform_size, total_mass, mass_ratio, freqmin)

    #     print(f"Initialized Dataset with waveform_size={waveform_size}")

    def generate_training_set(self, min_greedy_error, property=None, save_dataset=False, eccmin_list=None):
        # loaded_tr = np.load(f'Straindata/Greedy_Res_Training_{property}_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')
        # greedy_parameters_idx = loaded_tr['eccentricities_idx']
        # self.TS_M = np.load(f'Straindata/Training_TS_{min(eccmin_list)}_{max(eccmin_list)}.npz')['time'][-self.waveform_size:]
        try:

            loaded_tr = np.load(f'Straindata/Greedy_Res_Training_{property}_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')
            loaded_GB = np.load(f'Straindata/Greedy_Res_{property}_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')

            loaded_h = np.load(f'Straindata/Greedy_h_wfs_{property}_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')

            greedy_parameters = loaded_tr['eccentricity']
            greedy_parameters_idx = loaded_tr['eccentricities_idx']
            residual_training = loaded_tr['greedy_training']

            greedy_basis = loaded_GB['greedy_basis']

            reduced_basis_h = loaded_h['reduced_basis_h']
            empirical_nodes = loaded_h['empirical_nodes']
            
            self.TS_M = np.load(f'Straindata/Training_TS_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')['time'][-self.waveform_size:]


        except:
            print('No greedy set available')
            
            training_set, self.TS_M = self.import_waveform_property(property=property, save_dataset=save_dataset, training_set=True)
            greedy_basis, greedy_errors, greedy_parameters, greedy_parameters_idx = self.plot_greedy_error(min_greedy_error=min_greedy_error, training_set=training_set, property=property, plot_greedy_error=False)

            def compute_waveform(h_plus, h_cross):
                return h_plus + 1j * h_cross
            
            hp, hc, self.TS_M = self.generate_dataset_polarisations(save_dataset=False, eccmin_list=greedy_parameters)
            # Complex waveform
            h = compute_waveform(hp, hc)

            reduced_basis_h = self.reduced_basis(h)  

            np.savez(f'Straindata/Greedy_Res_{property}_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz', eccentricity=greedy_parameters, greedy_basis=greedy_basis)
            # np.savez(f'Straindata/Greedy_h_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz', eccentricity=greedy_parameters, reduced_basis_h=reduced_basis_h)

            empirical_nodes = self.calc_empirical_nodes(greedy_basis, self.TS_M)
            # empirical_nodes = self.calc_empirical_nodes(reduced_basis_h, self.TS_M)
            
            residual_training = np.zeros((greedy_basis.shape[0], len(empirical_nodes)))
            # time_training = np.zeros((residual.shape[0], len(empirical_nodes)))

            for i in range(len(greedy_parameters)):
                residual_training[i] = greedy_basis[i][empirical_nodes]
            
            self.TS_training = self.TS_M[empirical_nodes]
            print('greedy parameters:', greedy_parameters, property)

            np.savez(f'Straindata/Greedy_Res_Training_{property}_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz', eccentricity=greedy_parameters, eccentricities_idx=greedy_parameters_idx, greedy_training=residual_training)
            np.savez(f'Straindata/Greedy_h_wfs_{property}_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz', eccentricity=greedy_parameters, reduced_basis_h=reduced_basis_h, empirical_nodes=empirical_nodes)

        # if property == 'Phase':
        #     self.Dphase_training = residual_training
        #     self.greedy_params_phase = greedy_parameters
        # elif property == 'Amplitude':
        #     self.Damp_training = residual_training
        #     self.greedy_params_amp = greedy_parameters
        # else:
        #     print('Choose property= "Phase" or "Amplitude"')
        #     sys.exit(1)

        # def sort_arrays(parameters, residual_training):
        #     parameters = np.array(parameters).flatten()
        #     sorted_indices = np.argsort(parameters)

        #     sorted_parameters = parameters[sorted_indices]
        #     sorted_residual = residual_training[sorted_indices]
            
        #     return sorted_parameters, sorted_residual

        # fig_dataset = plt.figure()
        # # sorted_params, sorted_basis = sort_arrays(greedy_parameters, greedy_basis)
        
        # for i in range(10, 15):
        #     # plt.plot(self.TS_M, greedy_basis[i], label=greedy_parameters[i], linewidth=0.6)
        #     # plt.scatter(self.TS_M[empirical_nodes], greedy_basis[i][empirical_nodes], label='greedy basis emp nodes')
        #     plt.scatter(greedy_parameters, residual_training.T[i], label=empirical_nodes[i], s=4)
        # plt.legend()
        # plt.title('residual training')
        # plt.show()
        print('RT:', residual_training, empirical_nodes, greedy_parameters)
        return greedy_basis, residual_training, empirical_nodes, greedy_parameters, greedy_parameters_idx, reduced_basis_h
    
    
    def plot_training_set_at_node(self, time_node_idx, min_greedy_err_phase, min_greedy_err_amp, save_dataset=False):

        GB_amp, self.Damp_training, emp_nodes_amp, params_amp, params_amp_idx, reduced_basis_amp = self.generate_training_set(min_greedy_error=min_greedy_err_amp, property='Amplitude', save_dataset=save_dataset)
        GB_phase, self.Dphase_training, emp_nodes_phase, params_phase, params_phase_idx, reduced_basis_phase = self.generate_training_set(min_greedy_error=min_greedy_err_phase, property='Phase', save_dataset=save_dataset)
        
        def sort_arrays(parameters, residual_training):
            parameters = np.array(parameters).flatten()
            sorted_indices = np.argsort(parameters)

            sorted_parameters = parameters[sorted_indices]

            # Step 2: Use the sorting indices to sort the greedy_basis array
            sorted_residual = residual_training[sorted_indices]
            return sorted_parameters, sorted_residual
        
        sorted_amp_params, sorted_Damp = sort_arrays(params_amp, self.Damp_training)
        sorted_phase_params, sorted_Dphase = sort_arrays(params_phase, self.Dphase_training)
        
        
        fig_training_set, ax1 = plt.subplots(figsize=(7, 7))
        ax2 = ax1.twinx()
        
        ax1.scatter(sorted_phase_params, sorted_Dphase.T[time_node_idx], color='orange')
        ax2.scatter(sorted_amp_params, sorted_Damp.T[time_node_idx], color='blue')

        ax1.plot(sorted_phase_params, sorted_Dphase.T[time_node_idx], color='orange', label='Phase')
        ax2.plot(sorted_amp_params, sorted_Damp.T[time_node_idx], color='blue', label='Amplitude')

        ax1.set_xlabel("eccentricity")
        ax1.set_ylabel(f"$\Delta\phi$ at T_{time_node_idx}", fontsize=14)
        ax1.tick_params(axis="y", labelcolor='orange')

        ax2.set_ylabel(f"$\Delta$A at T_{time_node_idx} [M]", fontsize=14)
        ax2.tick_params(axis="y", labelcolor='blue')

        # ax1.legend(*ax1.get_legend_handles_labels(), *ax2.get_legend_handles_labels(), loc='lower left')

        # Adjust layout to fit the ylabels properly
        fig_training_set.tight_layout()  # Adjust the padding between and around subplots
        fig_training_set.subplots_adjust(right=0.85)  # Adjust the right margin to fit the ylabel

        plt.grid()
        plt.title(f'T_{time_node_idx} A = {self.TS_M[emp_nodes_amp[time_node_idx]].round(2)}, T_{time_node_idx} $\phi$ = {self.TS_M[emp_nodes_phase[time_node_idx]].round(2)}')
        # plt.show()

        # fig_dataset = plt.figure()
        # for i in range(5):
        #     plt.plot(self.TS_M, self.Dphase_training[-5:][i])
        # plt.show()
        # print('points_amp_x = ',[sorted(params_amp)] )
        # print('points_amp_y = ', [self.Damp_training.T[time_node_idx]])
        # print('points_phase_x = ',[sorted(params_phase)] )
        # print('points_phase_y = ', [self.Dphase_training.T[time_node_idx]])
    
    def largest_pointwise_error(self, time_node_idx, validation_num, min_greedy_error, save_dataset=False):

        if self.Damp_training is None:
            self.generate_training_set(min_greedy_error, 'Amplitude', save_dataset)
        
        if self.Dphase_training is None:
            self.generate_training_set(min_greedy_error, 'Phase', save_dataset)


        parameter_space = np.linspace(np.min(self.eccmin_list), np.max(self.eccmin_list), num=500)
        validation_params = np.random.choice(parameter_space, size=validation_num, replace=False)
        
        # Set min_greedy_error to very low value so it doesn't stop generating vecs for low error
        GB_amp, validation_set_amp, emp_nodes_amp, params_amp, params_amp_idx, reduced_basis_amp = self.generate_training_set(min_greedy_error=1e-20, property='Amplitude', save_dataset=True, eccmin_list=validation_params)[0]
        GB_phase, validation_set_phase, emp_nodes_phase, params_phase, params_phase_idx, reduced_basis_phase = self.generate_training_set(min_greedy_error=1e-20, property='Phase', save_dataset=True, eccmin_list=validation_params)[0]

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


    def gaussian_process_regression_fit(self, min_greedy_error, property='Phase', save_dataset=False, plot_time_node_idx=None ):

        def GPR(time_node_idx, training_set, greedy_parameters, property='Phase'):
            
            X = self.parameter_space[:, np.newaxis]

            X_train = np.array(greedy_parameters).reshape(-1, 1)
            y_train = np.squeeze(training_set.T[time_node_idx])
            
            # Scale y_train
            scaler = StandardScaler()
            y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            periodic_kernel = ExpSineSquared(length_scale=10.0, periodicity=1.0, length_scale_bounds=(1e-2, 1e3), periodicity_bounds=(1e-2, 1e1))
            rbf_kernel = RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e3))
            locally_periodic_kernel = C(1.0, (1e-4, 1e1)) * Product(periodic_kernel, rbf_kernel) + WhiteKernel(noise_level=1)
            
            kernels = [
                # C(1.0, (1e-4, 1e1)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
                C(1.0, (1e-4, 1e1)) * Matern(length_scale=5.0, length_scale_bounds=(1e-2, 1e3), nu=1.5),
                # C(1.0, (1e-4, 1e1)) * Product(periodic_kernel, rbf_kernel) + WhiteKernel(noise_level=1)
                # C(1.0, (1e-4, 1e1)) * RationalQuadratic(length_scale=1.0, alpha=0.1),
                # C(1.0, (1e-4, 1e1)) * ExpSineSquared(length_scale=1.0, periodicity=3.0, length_scale_bounds=(1e-2, 1e2), periodicity_bounds=(1e-2, 1e1)),
                # C(1.0, (1e-4, 1e1)) * ExpSineSquared(length_scale=1.0, periodicity=1.0) + WhiteKernel(noise_level=1),
                # C(1.0, (1e-4, 1e1)) * ExpSineSquared(length_scale=1.0, periodicity=1.0) + WhiteKernel(noise_level=1)
            ]

            # fig_GPR = plt.figure()

            for kernel in kernels:
            # kernel = C(1.0, (1e-4, 1e1)) * RationalQuadratic(length_scale=1.0, alpha=0.1)

                gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
                gaussian_process.fit(X_train, y_train_scaled)
                gaussian_process.kernel_

                mean_prediction_scaled, std_prediction_scaled = gaussian_process.predict(X, return_std=True)
                mean_prediction = scaler.inverse_transform(mean_prediction_scaled.reshape(-1, 1)).flatten()
                std_prediction = std_prediction_scaled * scaler.scale_[0]

            #     # plt.scatter(X_train, y_train, color='red', label="Observations", s=10)
            #     # plt.plot(X, mean_prediction, label="Mean prediction", linewidth=0.8)
            #     plt.scatter(X_train, y_train, color='red', s=10)
            #     plt.plot(X, mean_prediction, label="Mean prediction", linewidth=0.8)
            #     plt.fill_between(
            #         X.ravel(),
            #     (mean_prediction - 1.96 * std_prediction), 
            #     (mean_prediction + 1.96 * std_prediction),
            #         alpha=0.5,
            #         # label=r"95% confidence interval",
            #     )
            # plt.legend(loc = 'upper right')
            # plt.xlabel("$e$")
            # plt.ylabel("$f(e)$")
            # plt.title(f"GPR {property} on T_{time_node_idx} first")
            

            return mean_prediction, [(mean_prediction - 1.96 * std_prediction), (mean_prediction + 1.96 * std_prediction)]

        
        GB, residual_training, emp_nodes, greedy_params, greedy_params_idx, reduced_basis = self.generate_training_set(min_greedy_error, property, save_dataset)
        property_fit = np.zeros((len(emp_nodes), len(self.parameter_space)))
        property_uncertainty_region = []

        print('Interpolate phase and amplitude...')

        for node_i in range(len(emp_nodes)):
            mean_prediction, uncertainty_region = GPR(node_i, residual_training, greedy_params, property)
            
            property_fit[node_i] = mean_prediction
            property_uncertainty_region.append(uncertainty_region)


        residual_training_node = residual_training.T[plot_time_node_idx]

        X_train = np.array(greedy_params).reshape(-1, 1)
        X = self.parameter_space[:, np.newaxis]

        if plot_time_node_idx is not None:
            fig_GPR = plt.figure()

            # plt.scatter(X_train, residual_training_node, color='red', label="Observations", s=10)
            # plt.plot(X, property_fit[plot_time_node_idx], label="Mean prediction", linewidth=0.8)
            # # plt.fill_between(
            # #     X.ravel(),
            # #     property_uncertainty_region[time_node_idx][0],
            # #     uncertainty_region[time_node_idx][1],
            # #     alpha=0.5,
            # #     label=r"95% confidence interval",
            # # )
            # plt.legend()
            # plt.xlabel("$e$")
            # plt.ylabel("$f(e)$")
            # plt.title(f"GPR {property} on T_{plot_time_node_idx} second")
            # plt.show()
        
        return property_fit, property_uncertainty_region

    def generate_surrogate_waveform(self, min_greedy_error_amp, min_greedy_error_phase, save_dataset=False):

        # Generate training set for amplitude and phase
        GB_amp, res_amp_training, emp_nodes_idx_amp, greedy_params_amp, greedy_params_amp_idx, reduced_basis_amp = self.generate_training_set(min_greedy_error_amp, 'Amplitude', save_dataset)
        GB_phase, res_phase_training, emp_nodes_idx_phase, greedy_params_phase, greedy_params_phase_idx, reduced_basis_phase = self.generate_training_set(min_greedy_error_phase, 'Phase', save_dataset)
        # print('res shapes', res_phase_training.shape, res_amp_training.shape)
        
        amp_fit = self.gaussian_process_regression_fit(min_greedy_error=min_greedy_error_amp, property='Amplitude')[0]
        phase_fit = self.gaussian_process_regression_fit(min_greedy_error=min_greedy_error_phase, property='Phase')[0]

        
        
        fig_res_training, axs = plt.subplots(2)

        for i in range(10):
            # axs[0].scatter(self.parameter_space[greedy_params_amp_idx], res_amp_training.T[i])
            axs[0].plot(self.parameter_space, amp_fit[i])

            # axs[1].scatter(self.parameter_space[greedy_params_phase_idx], res_phase_training.T[i])
            axs[1].plot(self.parameter_space, phase_fit[i])
        
        axs[0].set_title('fit amp')
        axs[1].set_title('fit phi')
        plt.show()



        # print('res A', res_A[10], 'res phi', res_phi[10])

        # fig_gauss, axs = plt.subplots(2)
        
        # for i in range(len(res_A)):
        #     axs[0].plot(self.parameter_space, res_A[i])

        # for j in range(len(res_phi)):
        #     axs[1].plot(self.parameter_space, res_phi[j])

        # axs[0].set_title('gauss amp')
        # axs[1].set_title('gauss phi')
        # plt.show()

        def B_vec(reduced_basis, emp_nodes_idx):
            
            m = len(emp_nodes_idx)
            B_j_vec = np.zeros((reduced_basis.shape[1], m), dtype=complex)  # Ensure complex dtype

            V = np.zeros((m, m), dtype=complex)
            for j in range(m):
                for i in range(m):
                    V[j][i] = reduced_basis[i][emp_nodes_idx[j]]

            V_inv = np.linalg.inv(V)

            for t in range(reduced_basis.shape[1]):
                B_j = 0
                for i in range(m):
                    # B_j += reduced_basis[i].conj().T * V_inv[i][j]  # Use conjugate transpose for complex numbers
                    B_j_vec[t, i] = np.dot(reduced_basis[:, t], V_inv[:, i])
                    # print('fix?', reduced_basis[:, t], V_inv[:, i], B_j_vec[t, i])
                # B_j_vec[:, j] = B_j
            

            return B_j_vec


        def calculate_surrogate_datapiece(reduced_basis, residual_fit_dataset, emp_nodes_idx, greedy_params_idx, property='Phase'):
            # Switch back from residuals to actual amplitude and phase
            # Run circulair waveform properties
            B = B_vec(reduced_basis, emp_nodes_idx)
            print('RB', reduced_basis)
            self.circulair_wf()

            if property == 'Phase':
                circ = np.array(waveform.utils.phase_from_polarizations(self.hp_TS_circ, self.hc_TS_circ))
                circ = circ - circ[len(circ) - self.waveform_size]

            elif property == 'Amplitude':
                circ = np.array(waveform.utils.amplitude_from_polarizations(self.hp_TS_circ, self.hc_TS_circ))

            length_diff = len(circ) - self.waveform_size
            
            surrogate_datapiece = 0
            original_dataset = np.zeros((len(emp_nodes_idx), len(self.parameter_space)))

            for i, emp_node in enumerate(emp_nodes_idx):
                # element-wise exponential with imaginary unit
                original_dataset[i] = residual_fit_dataset[i] + circ[length_diff:][emp_node]
                """
                Maybe circ causes troubles?
                
                """
                # y = amp * np.exp(-1j * phase)

                surrogate_datapiece = np.dot(B, original_dataset)
            
            print(surrogate_datapiece.shape, original_dataset.shape, original_dataset.T[0])

            # fig_residual_datapieces, axs = plt.subplots(2, figsize=(10, 10))
            # plt.subplots_adjust(hspace=0.4)

            # print(res_phase_training.shape)
            # for i in range(5):
            #     axs[0].scatter(self.TS_M[emp_nodes_idx_phase], res_phase_training[i], label='residual phase emp nodes', s=3)

            #     # axs[1].plot(self.parameter_space[greedy_params_idx], res_phase_training[i])
            #     axs[1].plot(self.parameter_space, residual_dataset[i], label='residual')
            #     axs[1].plot(self.parameter_space, original_dataset[i], label='original')


            # axs[0].set_xlabel('t [M]')
            # # axs[0].set_ylabel('$\Delta\phi_{22}$ [radians]')
            # axs[0].set_ylabel('$A_{22}$')
            # axs[0].grid()

            # # axs[0].scatter(self.TS_M[emp_nodes_idx], circ[length_diff:][emp_nodes_idx], label='circ', s=3)
            # # axs[0].scatter(self.TS_M[emp_nodes_idx], residual_dataset.T[0], label='residual', s=3)
            # axs[0].legend()
            # axs[1].legend()

            # plt.title('original datapiece')
            # plt.show()

            # axs[1].plot(self.TS_M, surrogate_datapiece.T[0])
            

            return surrogate_datapiece
            
        # def calculate_surrogate_waveform(B, residual_amp, residual_phase, emp_nodes_idx):
        #     # Switch back from residuals to actual amplitude and phase
        #     # Run circulair waveform properties
        #     self.circulair_wf()
        #     amp_circ = np.array(waveform.utils.amplitude_from_polarizations(self.hp_TS_circ, self.hc_TS_circ))
        #     phase_circ = np.array(waveform.utils.phase_from_polarizations(self.hp_TS_circ, self.hc_TS_circ))
            
        #     length_diff = len(amp_circ) - self.waveform_size
        #     phase_circ = phase_circ - phase_circ[len(phase_circ) - self.waveform_size]

        #     surrogate = 0
        #     for i, emp_node in enumerate(emp_nodes_idx):
        #         # element-wise exponential with imaginary unit
        #         amp = residual_amp[i] + amp_circ[length_diff:][emp_node]
        #         phase = residual_phase[i] + phase_circ[length_diff:][emp_node]
                
        #         y = amp * np.exp(-1j * phase)
        #         y = y.reshape(-1, 1).T
        #         surrogate += np.dot(B[:, i].reshape(-1, 1), y)

        #     return surrogate
        # print('RB amp', reduced_basis_amp[10], 'RB phase', reduced_basis_phase[10])
        print('empies', emp_nodes_idx_amp, emp_nodes_idx_phase)
        surrogate_phase = calculate_surrogate_datapiece(GB_phase, phase_fit, emp_nodes_idx_phase, greedy_params_phase_idx, property='Phase')
        surrogate_amp = calculate_surrogate_datapiece(GB_amp, amp_fit, emp_nodes_idx_amp, greedy_params_amp_idx, property='Amplitude')
        print('shapes', surrogate_phase.shape, surrogate_amp.shape)
        # print('amp:', surrogate_amp, '\n phase:', surrogate_phase)
        self.surrogate = surrogate_amp * np.exp(-1j * surrogate_phase)
        print('after amp')
        print('surro', surrogate_phase, surrogate_amp, np.real(self.surrogate))

        # fig_surrogate_datapieces, axs = plt.subplots(2, figsize=(10, 10))
        # plt.subplots_adjust(hspace=0.4)

        # axs[0].plot(self.TS_M, surrogate_amp.T[0], linewidth=0.6)
        # axs[0].set_xlabel('t [M]')
        # # axs[0].set_ylabel('$\Delta\phi_{22}$ [radians]')
        # axs[0].set_ylabel('$A_{22}$')
        # axs[0].grid()

        
        # axs[1].plot(self.TS_M, surrogate_phase.T[0], linewidth=0.6)
        # axs[1].set_ylabel('$\phi_{22}$')
        
        # # axs[1].set_xlim(-50000, 0)
        # axs[1].set_xlabel('t [M]')
        # axs[1].grid()

        # print('after after amp')

        # plt.show()

    # def generate_surrogate_waveform(self, polarisation, min_greedy_error_phase=10e-3, min_greedy_error_amp=0.7, save_dataset=False):

    #     hp_DS, hc_DS, TS_M = self.import_polarisations(save_dataset)

    #     if polarisation == 'plus':
    #         reduced_basis = self.reduced_basis(hp_DS)
    #     elif polarisation == 'cross':
    #         reduced_basis = self.reduced_basis(hc_DS)
    #     else:
    #         print('Choose polarisation = "plus" or "cross"')
    #         sys.exit(1)

    #     # Generate training set for amplitude and phase
    #     res_amp_training, emp_nodes_idx_amp, params_amp = self.generate_training_set(min_greedy_error_amp, 'Amplitude', save_dataset)
    #     res_phase_training, emp_nodes_idx_phase, params_phase = self.generate_training_set(min_greedy_error_phase, 'Phase', save_dataset)

    #     # Create empty arrays for best approximation ampitude and phase with length m
    #     def get_surrogate_A_phase(emp_nodes_idx):
    #         m = len(emp_nodes_idx)

            
    #         print('Interpolate phase and amplitude...')
    #         # Create surrogate model amplitude and phase arrays for every time node
    #         for node in range(m):
    #             # print(type(res_amp_training), res_amp_training.shape)
    #             surrogate[node] = self.gaussian_process_regression(node, res_amp_training)[0]
    #             phi[node] = self.gaussian_process_regression(node, res_phase_training)[0]

    #     A = np.zeros((self.parameter_space, self.waveform_size))
    #     phi = np.zeros((self.parameter_space, self.waveform_size))

    #     #B_i(t) = sum_{j=i}^m e_j(t)(V^-1)_ji for e_j and T_i
    #     B_j_vec = np.zeros((reduced_basis.shape[1], m))

    #     V = np.zeros((m, m))
    #     for j in range(m):
    #         for i in range(m):
    #             V[j][i] = reduced_basis[i][emp_nodes_idx[j]]

    #     for j in range(V.shape[1]): 
    #         B_j = 0
    #         for i in range(m):
    #             B_j += reduced_basis[i].T * np.linalg.inv(V)[i][j]

    #         B_j_vec[:, j] = B_j


    #     def calculate_surrogate_waveform(B, residual_amp, residual_phase, emp_nodes_idx):
    #         # Switch back from residuals to actual amplitude and phase
    #         # Run circulair waveform properties
    #         self.circulair_wf()
    #         amp_circ = np.array(waveform.utils.amplitude_from_polarizations(self.hp_TS_circ, self.hc_TS_circ))
    #         phase_circ = np.array(waveform.utils.phase_from_polarizations(self.hp_TS_circ, self.hc_TS_circ))
            
    #         length_diff = len(amp_circ) - self.waveform_size
    #         phase_circ = phase_circ - phase_circ[len(phase_circ) - self.waveform_size]

    #         surrogate = 0
    #         for i, emp_node in enumerate(emp_nodes_idx):
    #             # element-wise exponential with imaginary unit
    #             amp = residual_amp[i] + amp_circ[length_diff:][emp_node]
    #             phase = residual_phase[i] + phase_circ[length_diff:][emp_node]
                
    #             y = amp * np.exp(-1j * phase)
    #             y = y.reshape(-1, 1).T
    #             surrogate += np.dot(B[:, i].reshape(-1, 1), y)

    #         return surrogate
              
    #     self.surrogate = calculate_surrogate_waveform(B_j_vec, A, phi, emp_nodes_idx)


    def calc_surrogate_error(self, true_h, idx):
        
        sum_squared_diff = np.zeros(len(true_h))

        for i in range(len(true_h)):
            sum_squared_diff[i] = (true_h[i] - np.real(self.surrogate[i, idx]))**2

        surrogate_error = (self.TS_M[1] - self.TS_M[0]) * sum_squared_diff
        print('surrogate error = ', surrogate_error, ' , eccmin = ', self.parameter_space[idx])

        fig_surrogate_err = plt.figure(figsize=(8, 5))
        
        plt.plot(self.TS_M, sum_squared_diff)  
        plt.title(self.parameter_space[idx])
        plt.ylim(-1e-42, 1e-42)
        # plt.show()

        return surrogate_error


    def plot_surrogate(self, idx, min_greedy_error_amp, min_greedy_error_phase, save_dataset=False):

        if self.surrogate is None:
            self.generate_surrogate_waveform(min_greedy_error_amp, min_greedy_error_phase, save_dataset=False)


        fig_surrogate = plt.figure(figsize=(8, 5))

        true_h = self.plot_sim_inspiral_mass_indp(polarisation='plus', eccmin=self.parameter_space[idx])
        print(true_h)
        plt.plot(self.TS_M, np.real(self.surrogate[:, idx]), linewidth=0.6, label='Surrogate')
        plt.scatter(self.TS_M[100], self.surrogate[100, idx])
        plt.scatter(self.TS_M[100], true_h[100])
        # plt.ylim(-1e-21, 1e-21)
        plt.xlabel('t [M]')
        plt.ylim(-1e-20, 1e-20)
        plt.grid(True)
        plt.legend()
        # plt.show()
        print(len(true_h))
        surrogate_error = self.calc_surrogate_error(true_h, idx)

        figname = f'Surrogate total mass = {self.total_mass}, mass ratio = {self.mass_ratio}, ecc = {min(self.eccmin_list)}_{max(self.eccmin_list)}, eccsur = {self.parameter_space[idx]}.png'
        fig_surrogate.savefig('Images/Surrogate_model/' + figname)
        print('fig is saved')    

        # plt.show()


    
    # def plot_surrogate_error(self, polarisation= 'plus'):

    #     if self.surrogate is None:
    #         self.generate_surrogate_waveform(min_greedy_error_amp, min_greedy_error_phase, save_dataset=False)
    
    #     print('Calculating validation set...')
    #     hp_validation_set, hc_validation_set, TS_M = self.generate_dataset_polarisations(self, eccmin_list=self.parameter_space)
        
    #     print('Validation set calculated')

    #     if polarisation == 'plus':
    #         true_h = hp_validation_set
    #     elif polarisation == 'cross':
    #         true_h = hc_validation_set
    #     else:
    #         print('Choose polarisation = "plus" or "cross"')
    #         sys.exit(1)

    #     surrogate_errors = []

    #     for idx in range(len(self.parameter_space)):
    #         surrogate_errors.append(self.calc_surrogate_error(true_h[idx], idx))

    #     fig_surrogate_errors = plt.figure(figsize=(8, 5))

    #     plt.plot(self.parameter_space, surrogate_errors)
    #     plt.xlabel('eccentricity')
    #     plt.ylabel('surrogate error')

    #     figname = f'Surrogate total mass = {self.total_mass}, mass ratio = {self.mass_ratio}, ecc = {min(self.eccmin_list)}_{max(self.eccmin_list)}, eccsur = {self.parameter_space[idx]}.png'
    #     fig_surrogate_errors.savefig('Images/Surrogate_errors/' + figname)
    #     print('fig is saved') 


# wp_05 = Waveform_properties(eccmin=0.181, total_mass=50, freqmin=18, waveform_size=3000)
# wp_05.plot_residuals('Amplitude')
# wp_05.plot_residuals('Phase')
# wp_05.plot_residuals('Frequency')


# wp_06 = Waveform_properties(eccmin=0.181, total_mass=50, freqmin=15, waveform_size=3000)
# wp_06.plot_residuals('Amplitude')
# wp_06.plot_residuals('Phase')
# wp_05.plot_sim_inspiral_mass_indp('cross')
# wp_05.plot_constructed_waveform(1000)
# plt.show()

# sim = Simulate_Inspiral(0.2, 50, 1, freqmin=20, waveform_size=None)
# sim.plot_sim_inspiral_mass_indp()
# plt.show()
# sim2 = Simulate_Inspiral(0.2, 50, 1, freqmin=15, waveform_size=3000)
# sim2.plot_sim_inspiral_mass_indp()
# sim.plot_sim_inspiral_mass_indp_multiple([10, 50], [1], [0.2], 3000)

# sim2 = Simulate_Inspiral(0.2, 10, 1, waveform_size=1000)
# sim2.plot_sim_inspiral_mass_indp()
# wp_01 = Waveform_properties(eccmin=0.1)
# plt.show()
# wp_05.plot_residuals(property='Amplitude')
# wp_01.plot_residuals(property='Amplitude')
# wp_05.plot_residuals(property='Phase')
# wp_01.plot_residuals(property='Phase')


# ds = Dataset(eccmin_list=np.linspace(0.01, 0.2, num=20).round(3), waveform_size=100000)
# ds.plot_dataset_polarisations()


# EMP = Empirical_Interpolation_Method(eccmin_list=np.linspace(0.01, 0.2, num=20).round(3), waveform_size=1000)
# EMP.plot_empirical_nodes(save_dataset=True)

# RB = Reduced_basis(eccmin_list=np.linspace(0.01, 0.2, num=20).round(3), waveform_size=3000, total_mass=50, mass_ratio=1, freqmin=20)
# RB.plot_greedy_error('Phase', save_dataset=True, polarisation='cross')
# RB.plot_greedy_error('Phase', save_dataset=True, polarisation='plus')
# RB.greedy_error_efficienty(save_dataset=True, polarisation = 'plus')
# RB.greedy_error_efficienty(save_dataset=True, polarisation = 'cross')
# RB.greedy_error_efficiency(save_dataset=True, property = 'Phase')
# RB.greedy_error_efficiency(save_dataset=True, property = 'Amplitude')
# RB.plot_greedy_error('Amplitude', save_dataset=True)
# RB.plot_dataset_properties('Phase', save_dataset=True)
# RB.plot_dataset_properties('Amplitude', save_dataset=True)

# RB.import_polarisations(save_dataset = True)

# RB.plot_dataset_polarisations()
# RB.import_waveform_property('Phase', save_dataset=True)
# RB.import_waveform_property('Amplitude', save_dataset= True)

TS = Training_set(eccmin_list=np.linspace(0.01, 0.2, num=20).round(3), freqmin=18, waveform_size=3500)
# TS.generate_training_set('Phase')
# TS.gaussian_process_regression()

# TS.plot_empirical_nodes(min_greedy_error=1e-2, property='Amplitude', save_dataset=True, plot_greedy_error=True)
# TS.plot_empirical_nodes(min_greedy_error=1e-3, property='Phase', save_dataset=True, plot_greedy_error=True)
# TS.plot_training_set_at_node(time_node_idx=2, min_greedy_err_phase=1e-3, min_greedy_err_amp=1e-2)
# TS.plot_training_set_at_node(time_node_idx=8, min_greedy_err_phase=1e-3, min_greedy_err_amp=1e-2)
# TS.plot_training_set_at_node(time_node_idx=15, min_greedy_err_phase=1e-3, min_greedy_err_amp=1e-2)
# TS.plot_training_set_at_node(time_node_idx=20, min_greedy_err_phase=1e-3, min_greedy_err_amp=1e-2)
print('gaussian')
# TS.gaussian_process_regression_fit(min_greedy_error=1e-3, property='Phase', save_dataset=True, plot_time_node_idx=2)
# TS.gaussian_process_regression_fit(min_greedy_error=1e-2, property='Amplitude', save_dataset=True, plot_time_node_idx=2)
# TS.gaussian_process_regression_fit(time_node_idx=8, min_greedy_error=1e-3, property='Phase', save_dataset=True, plot_GPR=True)
# TS.gaussian_process_regression_fit(time_node_idx=8, min_greedy_error=1e-2, property='Amplitude', save_dataset=True, plot_GPR=True)
# TS.gaussian_process_regression_fit(time_node_idx=15, min_greedy_error=1e-3, property='Phase', save_dataset=True, plot_GPR=True)
# TS.gaussian_process_regression_fit(time_node_idx=15, min_greedy_error=1e-2, property='Amplitude', save_dataset=True, plot_GPR=True)
# TS.plot_gaussian_process_regression_fit(time_node_idx=20, min_greedy_error=1e-3, property='Phase', save_dataset=True)
# TS.plot_gaussian_process_regression_fit(time_node_idx=20, min_greedy_error=1e-2, property='Amplitude', save_dataset=True)


TS.generate_surrogate_waveform(min_greedy_error_amp=1e-2, min_greedy_error_phase=1e-3)
# print('surrogate')
TS.plot_surrogate(25, 1e-2, 1e-3, save_dataset=True)
# TS.plot_surrogate(1000, 'plus')
# TS.plot_surrogate(2000, 'plus')
# TS.plot_surrogate(2800, 'plus')
# TS.calc_surrogate_error()

plt.show()

# Example usage
# training_set = Training_set([0.1, 0.2, 0.3], waveform_size=1000, total_mass=10, mass_ratio=1, freqmin=5)


"""
It should be greedy basis instead of training set!
"""