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
        # self.TS_M = np.load(f'Straindata/TS_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')['time'][-self.waveform_size:]
        
        try:
            
            loaded_tr = np.load(f'Straindata/Greedy_Res_Training_{property}_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')
            print(1)
            loaded_GB = np.load(f'Straindata/Greedy_Res_{property}_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')
            print(2)
            loaded_h = np.load(f'Straindata/Greedy_h_wfs_{property}_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')
            print(3)
            greedy_parameters = loaded_tr['eccentricity']
            greedy_parameters_idx = loaded_tr['eccentricities_idx']
            residual_training = loaded_tr['greedy_training']
            print(4)
            greedy_basis = loaded_GB['greedy_basis']
            print(5)
            loaded_DS_phase = np.load(f'Straindata/Training_Res_{property}_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')
            Res_DS_phase = loaded_DS_phase['Residual_dataset']
            print(6)
            # fig_GB = plt.figure()
            
            # for i in range(len(greedy_basis)):
            #     # print(Res_DS_phase[self.parameter_space[greedy_parameters_idx[i]])
            #     real_wf = Waveform_properties(self.parameter_space[greedy_parameters_idx[i]])
            #     real_wf.plot_residuals('Phase')
            #     plt.plot(self.TS_M, greedy_basis[i])
            #     plt.plot(self.TS_M, Res_DS_phase[greedy_parameters_idx[i]])
            # plt.title('GB') 
            # plt.show()
            
            print(6)

            print(7)
            # reduced_basis_h = loaded_h['reduced_basis_h']
            empirical_nodes = loaded_h['empirical_nodes']
            print(8)
            self.TS_M = np.load(f'Straindata/Training_TS_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')['time'][-self.waveform_size:]
            print(9)
            training_set, self.TS_M = self.import_waveform_property(property=property, save_dataset=True, training_set=True)

            print('Greedy set available')
            

        except:
            print('No greedy set available')
            loaded_DS_phase = np.load(f'Straindata/Training_Res_{property}_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')
            Res_DS_phase = loaded_DS_phase['Residual_dataset']
            
            self.import_waveform_property(property=property)
            training_set, self.TS_M = self.import_waveform_property(property=property, save_dataset=save_dataset, training_set=True)
            greedy_basis, greedy_errors, greedy_parameters, greedy_parameters_idx = self.plot_greedy_error(min_greedy_error=min_greedy_error, training_set=training_set, property=property, plot_greedy_error=False)
            
            # fig_GB = plt.figure()
            
            # for i in range(len(greedy_basis)):
            #     # print(Res_DS_phase[self.parameter_space[greedy_parameters_idx[i]])
            #     real_wf = Waveform_properties(self.parameter_space[greedy_parameters_idx[i]])
            #     real_wf.plot_residuals('Phase')
            #     plt.plot(self.TS_M, greedy_basis[i])
            #     plt.plot(self.TS_M, Res_DS_phase[greedy_parameters_idx[i]])
            # plt.title('GB') 
            # plt.show()
            # fig_GB = plt.figure()

            # for i in range(len(greedy_basis)):
            #     plt.plot(self.TS_M, greedy_basis[i], label=greedy_parameters[i])
            # plt.show()

            # def compute_waveform(h_plus, h_cross):
            #     return h_plus + 1j * h_cross
            
            # hp, hc, self.TS_M = self.generate_dataset_polarisations(save_dataset=False, eccmin_list=greedy_parameters)
            
            # # Complex waveform
            # h = compute_waveform(hp, hc)
            # print('hp', hp.shape)

            # reduced_basis_h = self.reduced_basis(h)  

            np.savez(f'Straindata/Greedy_Res_{property}_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz', eccentricity=greedy_parameters, greedy_basis=greedy_basis)

            empirical_nodes = self.calc_empirical_nodes(greedy_basis, self.TS_M)
            
            residual_training = np.zeros((greedy_basis.shape[0], len(empirical_nodes)))
            # time_training = np.zeros((residual.shape[0], len(empirical_nodes)))

            for i in range(len(greedy_parameters)):
                residual_training[i] = greedy_basis[i][empirical_nodes]
            
            self.TS_training = self.TS_M[empirical_nodes]

            np.savez(f'Straindata/Greedy_Res_Training_{property}_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz', eccentricity=greedy_parameters, eccentricities_idx=greedy_parameters_idx, greedy_training=residual_training)
            np.savez(f'Straindata/Greedy_h_wfs_{property}_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz', eccentricity=greedy_parameters, empirical_nodes=empirical_nodes)


        if property == 'Phase':
            self.Dphase_training = residual_training
            self.greedy_params_phase = greedy_parameters
        elif property == 'Amplitude':
            self.Damp_training = residual_training
            self.greedy_params_amp = greedy_parameters
        else:
            print('Choose property= "Phase" or "Amplitude"')
            sys.exit(1)

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
        # print('RT:', residual_training, empirical_nodes, greedy_parameters)
        return greedy_basis, residual_training, empirical_nodes, greedy_parameters, greedy_parameters_idx
    
    
    def plot_training_set_at_node(self, time_node_idx, min_greedy_err_phase, min_greedy_err_amp, save_dataset=False):

        GB_amp, self.Damp_training, emp_nodes_amp, params_amp, params_amp_idx= self.generate_training_set(min_greedy_error=min_greedy_err_amp, property='Amplitude', save_dataset=save_dataset)
        GB_phase, self.Dphase_training, emp_nodes_phase, params_phase, params_phase_idx = self.generate_training_set(min_greedy_error=min_greedy_err_phase, property='Phase', save_dataset=save_dataset)
        
        loaded_DS_amp = np.load(f'Straindata/Training_Res_Amplitude_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')
        Res_DS_amp = loaded_DS_amp['Residual_dataset']
        loaded_DS_phase = np.load(f'Straindata/Training_Res_Phase_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')
        Res_DS_phase = loaded_DS_phase['Residual_dataset']

        def sort_arrays(parameters, parameters_idx, residual_training):
            parameters = np.array(parameters).flatten()
            sorted_indices = np.argsort(parameters)

            sorted_parameters = parameters[sorted_indices]

            # Step 2: Use the sorting indices to sort the greedy_basis array
            sorted_residual = residual_training[sorted_indices]
            return sorted_parameters, sorted_residual
        
        sorted_amp_params, sorted_Damp = sort_arrays(params_amp, params_amp_idx, self.Damp_training)
        sorted_phase_params, sorted_Dphase = sort_arrays(params_phase, params_phase_idx, self.Dphase_training)
        # print(params_amp_idx[0], self.Dphase_training.shape, Res_DS_amp.shape)

        # fig_test2 = plt.figure()
        # for i in range(1):
        #     plt.scatter(self.parameter_space[greedy_parameters_idx], greedy_basis.T[i], label='GB')
        #     plt.scatter(self.parameter_space[greedy_parameters_idx], training_set[greedy_parameters_idx].T[i], linestyle='dashed', label='TS')
        #     plt.plot(self.parameter_space, training_set.T[i])
        # plt.legend()
        # plt.show()
        # print(params_amp, params_amp_idx)
        # fig_test = plt.figure()
        # plt.plot(self.parameter_space, Res_DS_amp.T[time_node_idx])
        
        # plt.scatter(self.parameter_space[params_amp_idx], self.Damp_training.T[time_node_idx])
        # plt.ylim(-1e-22, 1e-22)
        # plt.title('')
        # plt.show()
        
        # fig_training_set, ax1 = plt.subplots(figsize=(7, 7))
        # ax2 = ax1.twinx()
        
        # ax1.scatter(sorted_phase_params, sorted_Dphase.T[time_node_idx], color='orange')
        # ax1.plot(self.parameter_space, Res_DS_amp[:, time_node_idx])
        # ax2.scatter(sorted_amp_params, sorted_Damp.T[time_node_idx], color='blue')
        # ax2.plot(self.parameter_space, Res_DS_phase[:, time_node_idx])

        # ax1.plot(sorted_phase_params, sorted_Dphase.T[time_node_idx], color='orange', label='Phase')
        # ax2.plot(sorted_amp_params, sorted_Damp.T[time_node_idx], color='blue', label='Amplitude')

        # ax1.set_xlabel("eccentricity")
        # ax1.set_ylabel(f"$\Delta\phi$ at T_{time_node_idx}", fontsize=14)
        # ax1.tick_params(axis="y", labelcolor='orange')

        # ax2.set_ylabel(f"$\Delta$A at T_{time_node_idx} [M]", fontsize=14)
        # ax2.tick_params(axis="y", labelcolor='blue')

        # # ax1.legend(*ax1.get_legend_handles_labels(), *ax2.get_legend_handles_labels(), loc='lower left')

        # # Adjust layout to fit the ylabels properly
        # fig_training_set.tight_layout()  # Adjust the padding between and around subplots
        # fig_training_set.subplots_adjust(right=0.85)  # Adjust the right margin to fit the ylabel

        # plt.grid()
        # plt.title(f'T_{time_node_idx} A = {self.TS_M[emp_nodes_amp[time_node_idx]].round(2)}, T_{time_node_idx} $\phi$ = {self.TS_M[emp_nodes_phase[time_node_idx]].round(2)}')
        # # plt.show()

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
        GB_amp, validation_set_amp, emp_nodes_amp, params_amp, params_amp_idx = self.generate_training_set(min_greedy_error=1e-20, property='Amplitude', save_dataset=True, eccmin_list=validation_params)[0]
        GB_phase, validation_set_phase, emp_nodes_phase, params_phase, params_phase_idx = self.generate_training_set(min_greedy_error=1e-20, property='Phase', save_dataset=True, eccmin_list=validation_params)[0]

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

        
        GB, residual_training, emp_nodes, greedy_params, greedy_params_idx = self.generate_training_set(min_greedy_error, property, save_dataset)
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

        # if plot_time_node_idx is not None:
            # fig_GPR = plt.figure()

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

        try:
            ############ also save self.TS_M

            loaded_fits = np.load(f'Straindata/gauss_fits_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')
            loaded_props = np.load(f'Straindata/gauss_fits_properties_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')

            res_amp_fit = loaded_fits['fit_amp']
            res_phase_fit = loaded_fits['fit_phase']
            GB_amp=loaded_props['GB_amp']
            GB_phase=loaded_props['GB_phase']
            emp_nodes_idx_amp=loaded_props['emp_nodes_idx_amp']
            emp_nodes_idx_phase=loaded_props['emp_nodes_idx_phase']
            greedy_params_amp_idx=loaded_props['greedy_params_amp_idx']
            greedy_params_phase_idx=loaded_props['greedy_params_phase_idx']
            res_amp_training = loaded_props['res_amp_training']
            res_phase_training=loaded_props['res_phase_training']
            self.TS_M = loaded_fits['TS']

            # fig_GB = plt.figure()
            # for i in range(len(GB_phase)):
            #     plt.scatter(self.parameter_space[greedy_params_phase_idx], GB_phase[:, i])
            # plt.show()

            print('Loading fits succefull')

        except:
            print('No loaded fits found')
            # Generate training set for amplitude and phase
            GB_amp, res_amp_training, emp_nodes_idx_amp, greedy_params_amp, greedy_params_amp_idx = self.generate_training_set(min_greedy_error_amp, 'Amplitude', save_dataset)
            GB_phase, res_phase_training, emp_nodes_idx_phase, greedy_params_phase, greedy_params_phase_idx = self.generate_training_set(min_greedy_error_phase, 'Phase', save_dataset)
            # print('res shapes', res_phase_training.shape, res_amp_training.shape)
            
            res_amp_fit = self.gaussian_process_regression_fit(min_greedy_error=min_greedy_error_amp, property='Amplitude')[0]
            res_phase_fit = self.gaussian_process_regression_fit(min_greedy_error=min_greedy_error_phase, property='Phase')[0]

            np.savez(f'Straindata/gauss_fits_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz', fit_phase=res_phase_fit, fit_amp=res_amp_fit, TS=self.TS_M)
            np.savez(f'Straindata/gauss_fits_properties_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz', GB_amp=GB_amp, GB_phase=GB_phase, emp_nodes_idx_amp=emp_nodes_idx_amp, emp_nodes_idx_phase=emp_nodes_idx_phase, greedy_params_amp_idx=greedy_params_amp_idx, greedy_params_phase_idx=greedy_params_phase_idx, res_amp_training=res_amp_training, res_phase_training=res_phase_training)

        loaded_DS_amp = np.load(f'Straindata/Training_Res_Amplitude_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')
        Res_DS_amp = loaded_DS_amp['Residual_dataset']
        loaded_DS_phase = np.load(f'Straindata/Training_Res_Phase_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')
        Res_DS_phase = loaded_DS_phase['Residual_dataset']

        fig_ = plt.figure()
        for i in range(len(Res_DS_phase)):
            plt.plot(self.TS_M, Res_DS_phase[i])

        plt.show()
        

        # idx = [100, 200, 300]
        # fig2 = plt.figure()
        # for i in idx:
        #     plt.plot(self.parameter_space, Res_DS_amp[:, i])
        #     plt.title('Training')


        # Define a distinct color palette
        color_palette = plt.cm.tab10.colors  # You can choose another colormap like 'tab20', 'Set1', etc.

        # Number of distinct colors needed (based on the number of lines you want to plot)
        if len(res_amp_fit) >= len(res_phase_fit):
            num_colors = len(res_amp_fit)
        else:
            num_colors = len(res_phase_fit)

        # Create a color map to use consistent colors for matching fits and data sets
        colors = [color_palette[i % len(color_palette)] for i in range(num_colors)]

        fig_res_training, axs = plt.subplots(2, 2, figsize=(11,6), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.4})

        # Top left subplot for amplitude main plot
        for i in range(len(res_amp_fit)):
            # Use the same color for the fit and the corresponding empirical data
            axs[0, 0].plot(self.parameter_space, res_amp_fit.T[:, i], color=colors[i])
            axs[0, 0].scatter(self.parameter_space[greedy_params_amp_idx], res_amp_training[:, i], color=colors[i])
            axs[0, 0].plot(self.parameter_space, Res_DS_amp[:, emp_nodes_idx_amp[i]], linestyle='dashed', color=colors[i], label=f't={int(self.TS_M[emp_nodes_idx_amp[i]])}')

        # Adjust legend: smaller font, inside figure, upper left
        axs[0, 0].legend(loc='upper left', ncol=3, bbox_to_anchor=(0.01, 0.99), fontsize='small')
        axs[0, 0].set_xlabel('eccentricity')
        axs[0, 0].set_ylabel('$\Delta$ A')
        axs[0, 0].set_title('GPRfit amp')
        axs[0, 0].grid()

        # Top right subplot for phase main plot
        for i in range(len(res_phase_fit)):
            # Use the same color for the fit and the corresponding empirical data
            # axs[0, 1].plot(self.parameter_space, res_phase_fit.T[:, i], color=colors[i])
            # axs[0, 1].scatter(self.parameter_space[greedy_params_phase_idx], res_phase_training[:, i], color=colors[i])
            axs[0, 1].plot(self.parameter_space, Res_DS_phase[:, emp_nodes_idx_phase[i]], linestyle='dashed', color=colors[i], label=f't={int(self.TS_M[emp_nodes_idx_phase[i]])}')

        # Adjust legend: smaller font, inside figure, upper left
        axs[0, 1].legend(loc='upper left', ncol=3, bbox_to_anchor=(0.01, 0.99), fontsize='small')
        axs[0, 1].set_xlabel('eccentricity')
        axs[0, 1].set_ylabel('$\Delta \phi$')
        axs[0, 1].set_title('GPRfit phase')
        axs[0, 1].grid()

        # Bottom left subplot for amplitude error plot
        for i in range(len(res_amp_fit)):
            error_amp = abs(Res_DS_amp[:, emp_nodes_idx_amp[i]] - res_amp_fit.T[:, i]) / Res_DS_amp[:, emp_nodes_idx_amp[i]]
            axs[1, 0].plot(self.parameter_space, error_amp, color=colors[i], label=f'Error {i+1} (t={int(self.TS_M[emp_nodes_idx_amp[i]])})')

        # Adjust legend: smaller font, inside figure, upper right
        axs[1, 0].set_xlabel('eccentricity')
        axs[1, 0].set_ylabel('Error')
        axs[1, 0].set_title('Amplitude Fit Error')
        # axs[1, 0].set_xlim(0., 0.105)
        # axs[1, 0].set_ylim(-100, 100)
        axs[1, 0].grid()

        # Bottom right subplot for phase error plot
        for i in range(len(res_phase_fit)):
            error_phase = abs(Res_DS_phase[:, emp_nodes_idx_phase[i]] - res_phase_fit.T[:, i]) / Res_DS_phase[:, emp_nodes_idx_phase[i]]
            axs[1, 1].plot(self.parameter_space, error_phase, color=colors[i], label=f'Error {i+1} (t={int(self.TS_M[emp_nodes_idx_phase[i]])})')

        # Adjust legend: smaller font, inside figure, lower right
        axs[1, 1].set_xlabel('eccentricity')
        axs[1, 1].set_ylabel('Error')
        axs[1, 1].set_title('Phase Fit Error')
        # axs[1, 1].set_xlim(0.02, 0.105)
        # axs[1, 1].set_ylim(-100, 100)
        axs[1, 1].grid()

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()
###################################################3
        # plt.show()


        # fig_res_training, axs = plt.subplots(2)
        # plt.subplots_adjust(hspace=0.4)
        # print('shape', res_phase_training.shape)
        # for i in range(len(amp_fit)):
        #     # axs[0].scatter(self.parameter_space[greedy_params_amp_idx], res_amp_training.T[i])
        #     axs[0].plot(self.parameter_space, amp_fit.T[:, i])
        #     axs[0].scatter(self.parameter_space[greedy_params_amp_idx], res_amp_training[:, i])
        #     axs[0].plot(self.parameter_space, Res_DS_amp[:, emp_nodes_idx_amp[i]], linestyle='dashed', label=f't_idx={int(self.TS_M[emp_nodes_idx_amp[i]])}')
        #     axs[0].legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.15))
        #     axs[0].set_xlabel('eccentricity')
        #     axs[0].set_ylabel('$\Delta$ A')

        # for i in range(len(phase_fit)):
        #     # axs[1].scatter(self.parameter_space[greedy_params_phase_idx], res_phase_training.T[i])
        #     # axs[1].plot(self.parameter_space, phase_fit.T[:, i],)
        #     # axs[1].scatter(self.parameter_space[greedy_params_phase_idx], res_phase_training[:, i])
        #     axs[1].plot(self.parameter_space, Res_DS_phase[:, emp_nodes_idx_phase[i]], linestyle='dashed', label=f't_idx={int(self.TS_M[emp_nodes_idx_phase[i]])}')
        #     axs[1].legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.15))
        #     axs[1].set_xlabel('eccentricity')
        #     axs[1].set_ylabel('$\Delta \phi$')
        
        # axs[0].set_title('GPRfit amp')
        # axs[1].set_title('GPRfit phase')

        # figname = f'GPRfits greedy err amp = {min_greedy_error_amp}, greedy error phase = {min_greedy_error_phase}, ecc = {min(self.eccmin_list)}_{max(self.eccmin_list)}.png'
        # fig_res_training.savefig('Images/GPRfits/' + figname)
        
        
        
        # fig_res_t_training, axs = plt.subplots(2)

        # for i in range(10):
        #     # axs[0].scatter(self.parameter_space[greedy_params_amp_idx], res_amp_training.T[i])
        #     axs[0].plot(self.TS_M, amp_fit.T[i])

        #     # axs[1].scatter(self.parameter_space[greedy_params_phase_idx], res_phase_training.T[i])
        #     axs[1].plot(self.TS_M, phase_fit.T[i])
        
        # axs[0].set_title('fit amp')
        # axs[1].set_title('fit phi')



        # print('res A', res_A[10], 'res phi', res_phi[10])

        # fig_gauss, axs = plt.subplots(2)
        
        # for i in range(len(res_A)):
        #     axs[0].plot(self.parameter_space, res_A[i])

        # for j in range(len(res_phi)):
        #     axs[1].plot(self.parameter_space, res_phi[j])

        # axs[0].set_title('gauss amp')
        # axs[1].set_title('gauss phi')
        # plt.show()

        def compute_B_matrix(reduced_basis, emp_nodes_idx):

            """
            Computes the B matrix for all empirical nodes and basis functions.
            
            e_matrix: Array of shape (m, time_samples) representing the reduced basis functions evaluated at different time samples.
            V_inv: Inverse of the interpolation matrix of shape (m, m).
            
            Returns:
            B_matrix: Array of shape (m, time_samples) where each row represents B_j(t) for j=1,2,...,m
            """

            m, time_samples = reduced_basis.shape
            B_matrix = np.zeros((m, time_samples))

            V = np.zeros((m, m))
            for j in range(m):
                for i in range(m):
                    V[j][i] = reduced_basis[i][emp_nodes_idx[j]]

            V_inv = np.linalg.pinv(V)

            
            # Compute each B_j(t) for j = 1, 2, ..., m
            for j in range(m):
                # Compute B_j(t) as a linear combination of all e_i(t) with weights from V_inv[:, j]
                for i in range(m):
                    B_matrix[j] += reduced_basis[i] * V_inv[i, j]
                    
            
            return B_matrix
    
        # def compute_B_matrix(reduced_basis, emp_nodes_idx):
        #     """
        #     Computes the B matrix for a given set of empirical nodes.
            
        #     e_matrix: Array of shape (m, time_samples) representing the reduced basis.
        #     V_inv: Inverse of the interpolation matrix of shape (m, m).
            
        #     Returns:
        #     B_matrix: Array of shape (m, time_samples).
        #     """
        #     e_matrix = reduced_basis
            
        #     m, time_samples = e_matrix.shape
        #     B_matrix = np.zeros((m, time_samples))

        #     V = np.zeros((m, m), dtype=complex)
        #     for j in range(m):
        #         for i in range(m):
        #             V[j][i] = reduced_basis[i][emp_nodes_idx[j]]
        #     # print(reduced_basis[:10])
        #     # print('V: ', V, np.linalg.det(V))
        #     V_inv = np.linalg.pinv(V)
            
        #     for i in range(m):
        #         B_matrix[i, :] = e_matrix.T @ V_inv[:, i]
            
            
        #     return B_matrix
        
        def residual_to_original(residual_dataset, property='Phase'):
            # Switch back from residuals to actual amplitude and phase
            # Run circulair waveform properties
            # B = B_vec(reduced_basis, emp_nodes_idx)
            # print('RB', reduced_basis)
            self.circulair_wf()

            if property == 'Phase':
                circ = np.array(waveform.utils.phase_from_polarizations(self.hp_TS_circ, self.hc_TS_circ))
                # circ_cutoff = circ - circ[len(circ) - self.waveform_size]

            elif property == 'Amplitude':
                circ = np.array(waveform.utils.amplitude_from_polarizations(self.hp_TS_circ, self.hc_TS_circ))

            length_diff = len(circ) - self.waveform_size
            
            surrogate_datapiece = 0
            original_dataset = np.zeros((residual_dataset.shape[0], residual_dataset.shape[1]))

            # for i, emp_node in enumerate(emp_nodes_idx):
            #     # element-wise exponential with imaginary unit
            #     original_dataset[i] = residual_fit_dataset[i] + circ[length_diff:][emp_node]
            
            # fig = plt.figure()
            for i in range(residual_dataset.shape[1]):
                original_dataset[:, i] = residual_dataset[:, i] + circ[length_diff:]

            # for i in [100, 200]:
            #     real_wf = Waveform_properties(self.parameter_space[i])
            #     real_wf.plot_residuals('Phase')
            #     plt.plot(self.TS_M, residual_dataset[:, i], label='residual fit')
            #     # plt.plot(self.TS_M, original_dataset[:, i], label='original')
            # # plt.plot(self.TS_M, circ[length_diff:], label='circ')
            # plt.legend()
            # plt.show()
            return original_dataset

        def reconstruct_surrogate(B_amp_matrix, amp_matrix, B_phase_matrix, phase_matrix, time_samples):
            """
            Reconstructs the surrogate model for a given parameter using different empirical nodes for amplitude and phase.
            
            B_amp_matrix: Array of shape (m_amp, time_samples) for amplitude.
            A_matrix: Array of shape (m_amp, n) with n as the number of parameters.
            B_phi_matrix: Array of shape (m_phi, time_samples) for phase.
            phi_matrix: Array of shape (m_phi, n) with n as the number of parameters.
            time_samples: Array representing the time samples.
            param_index: Integer representing the index of the parameter to reconstruct.
            
            Returns:
            h_s: Array representing the reconstructed surrogate waveform.
            """
            m_amp, _ = B_amp_matrix.shape
            m_phase, _ = B_phase_matrix.shape
            h_s = np.zeros(len(time_samples), dtype=complex)

            

            amp_term_residual = np.zeros((len(time_samples), len(self.parameter_space)))
            phase_term_residual = np.zeros((len(time_samples), len(self.parameter_space)))
            # print(amp_term_residual.T[0], phase_term_residual.T[0])
            # fig_matrix = plt.figure()
            for i in range(m_amp):
                print('?', B_amp_matrix[i, :].reshape(-1, 1).shape, amp_matrix[i, :].reshape(1, -1).shape, type(B_amp_matrix), type(amp_matrix))
                amp_term_residual += np.dot(B_amp_matrix[i, :].reshape(-1, 1), amp_matrix[i, :].reshape(1, -1))
                
                # plt.plot(self.TS_M, amp_term_residual)
            for i in range(m_phase):
                phase_term_residual += np.dot(B_phase_matrix[i, :].reshape(-1, 1), phase_matrix[i, :].reshape(1, -1))
                # plt.plot(self.parameter_space, phase_term_residual[i])
                # plt.plot(self.parameter_space, phase_matrix[i, :], label='phase fits')
            # for i in [100, 200, 300]:
            #     plt.plot(self.parameter_space, phase_term_residual[i], label= 'check')
            # plt.legend()
            # plt.title('1')
            # # plt.show()
            # print(phase_matrix.shape, phase_term_residual.shape)

            print(amp_term_residual.T[0], phase_term_residual.T[0])


            

            surrogate_amp = residual_to_original(residual_dataset=amp_term_residual, property='Amplitude')
            surrogate_phase = residual_to_original(residual_dataset=phase_term_residual, property='Phase')
            print('surr shape', surrogate_phase.shape, surrogate_amp.shape)
            # plt.plot(self.parameter_space, surrogate_phase[0], label='surr')
            
            # fig3 = plt.figure()
            # for i in [100, 200, 300]:
            #     plt.plot(self.TS_M, phase_term_residual[:, i], label='phase term residual')
            #     # plt.plot(self.TS_M, )
            #     plt.plot(self.TS_M, surrogate_phase[:, i], label='surr')
            # plt.title('phase term residual')

            h_s = np.zeros((len(time_samples), len(self.parameter_space)), dtype=complex)
            h_s = surrogate_amp * np.exp(-1j * surrogate_phase)


            idx = [100, 500, 1000, 2000, 3000]
            idx2 = np.linspace(0, 499, num=10)


            # Define the color palette for consistency
            color_palette = plt.cm.tab10.colors  # Choose a colormap with distinct colors

            # Create a 2x2 subplot grid with height ratios 3:1
            fig_surrogate_datapieces, axs = plt.subplots(2, 2, figsize=(11, 6), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.4})
            # print(surrogate_amp.T[0], surrogate_phase.T[0])
            for i in idx2:
                i = int(i)
                
                # Select a distinct color for the current parameter
                color = color_palette[i % len(color_palette)]

                # Simulate the real waveform and extract amplitude and phase
                real_hp, real_hc, real_TS = self.sim_inspiral_mass_indp(self.parameter_space[i])
                real_amp = np.array(waveform.utils.amplitude_from_polarizations(real_hp, real_hc))
                real_phase = np.array(waveform.utils.phase_from_polarizations(real_hp, real_hc))

                # Ensure the time series lengths match by accounting for differences
                length_diff = len(real_TS) - self.waveform_size
                print('9', len(self.TS_M), surrogate_amp.T[i].shape)
                # Plot Surrogate and Real Amplitude (Top Left)
                axs[0, 0].plot(self.TS_M, surrogate_amp.T[i], color=color, linewidth=0.6, label=f'Surrogate: e = {self.parameter_space[i]}')
                axs[0, 0].plot(self.TS_M, real_amp[length_diff:], color=color, linewidth=0.6, linestyle='dashed')
                axs[0, 0].set_xlabel('t [M]')
                axs[0, 0].set_ylabel('$A_{22}$')
                axs[0, 0].grid(True)
                axs[0, 0].set_title('Surrogate vs Real Amplitude')
                axs[0, 0].legend(loc='upper left', ncol=2, fontsize='small')

                # Plot Surrogate and Real Phase (Top Right)
                axs[0, 1].plot(self.TS_M, surrogate_phase.T[i], color=color, linewidth=0.6, label=f'Surrogate: e = {self.parameter_space[i]}')
                axs[0, 1].plot(self.TS_M, real_phase[length_diff:], color=color, linewidth=0.6, linestyle='dashed')
                axs[0, 1].set_ylabel('$\phi_{22}$')
                axs[0, 1].set_xlabel('t [M]')
                axs[0, 1].grid(True)
                axs[0, 1].set_title('Surrogate vs Real Phase')
                axs[0, 1].legend(loc='upper left', ncol=2, fontsize='small')

                # Calculate and Plot Amplitude Error (Bottom Left)
                amp_error = abs(surrogate_amp.T[i] - real_amp[length_diff:]) / real_amp[length_diff:]
                axs[1, 0].plot(self.TS_M, amp_error, color=color, linewidth=0.6)
                axs[1, 0].set_ylabel('Rel. Error in $A_{22}$')
                axs[1, 0].set_xlabel('t [M]')
                axs[1, 0].grid(True)
                axs[1, 0].set_title('Relative Amplitude Error')

                # Calculate and Plot Phase Error (Bottom Right)
                phase_error = abs(surrogate_phase.T[i] - real_phase[length_diff:]) / real_phase[length_diff:]
                axs[1, 1].plot(self.TS_M, phase_error, color=color, linewidth=0.6)
                axs[1, 1].set_ylabel('Rel. Error in $\phi_{22}$')
                axs[1, 1].set_xlabel('t [M]')
                axs[1, 1].grid(True)
                axs[1, 1].set_title('Relative Phase Error')

                # Adjust layout to prevent overlap
                plt.tight_layout()

            # plt.show()




            # plt.show()

            figname = f'Surrogate greedy datapieces amp = {min_greedy_error_amp}, greedy error phase = {min_greedy_error_phase}, ecc = {min(self.eccmin_list)}_{max(self.eccmin_list)}.png'
            fig_surrogate_datapieces.savefig('Images/Surrogate_datapieces/' + figname)
                
            return h_s

        B_amp_matrix = compute_B_matrix(GB_amp, emp_nodes_idx_amp)
        B_phase_matrix = compute_B_matrix(GB_phase, emp_nodes_idx_phase)
        print('B matrix amp: ', emp_nodes_idx_amp, GB_amp, B_amp_matrix, GB_amp.shape, B_amp_matrix.shape )
        print('B matrix phase: ', emp_nodes_idx_phase, GB_phase, B_phase_matrix, GB_phase.shape, B_phase_matrix.shape)

        reconstruct_surrogate(B_amp_matrix, res_amp_fit, B_phase_matrix, res_phase_fit, self.TS_M)


            # axs[2].plot(self.TS_M, np.real(h_s[:, i]), linewidth=0.6, label=f'Surrogate: e={self.parameter_space[i]}')
            # axs[2].plot(self.TS_M, real_hp[length_diff:], linewidth=0.6, linestyle='dashed', label='Real')
            # axs[2].set_xlabel('t [M]')
            # axs[2].set_ylabel('$h_+$')
            # axs[2].legend()

        # def B_vec(reduced_basis, emp_nodes_idx):
            
        #     m = len(emp_nodes_idx)
        #     B_j_vec = np.zeros((reduced_basis.shape[1], m), dtype=complex)  # Ensure complex dtype

        #     V = np.zeros((m, m), dtype=complex)
        #     for j in range(m):
        #         for i in range(m):
        #             V[j][i] = reduced_basis[i][emp_nodes_idx[j]]

        #     V_inv = np.linalg.inv(V)

        #     for t in range(reduced_basis.shape[1]):
        #         B_j = 0
        #         for i in range(m):
        #             # B_j += reduced_basis[i].conj().T * V_inv[i][j]  # Use conjugate transpose for complex numbers
        #             B_j_vec[t, i] = np.dot(reduced_basis[:, t], V_inv[:, i])
        #             # print('fix?', reduced_basis[:, t], V_inv[:, i], B_j_vec[t, i])
        #         # B_j_vec[:, j] = B_j
            

        #     return B_j_vec




        # surrogate_phase = calculate_surrogate_datapiece(GB_phase, phase_fit, emp_nodes_idx_phase, greedy_params_phase_idx, property='Phase')
        # surrogate_amp = calculate_surrogate_datapiece(GB_amp, amp_fit, emp_nodes_idx_amp, greedy_params_amp_idx, property='Amplitude')
        
        # np.savez(f'Straindata/surrogate_datapieces.npz', surrogate_phase=surrogate_phase, surrogate_amp=surrogate_amp, TS=self.TS_M)

        # self.surrogate = surrogate_amp * np.exp(-1j * surrogate_phase)
       


    def calc_surrogate_error(self, true_h, idx):
        
        sum_squared_diff = np.zeros(len(true_h))

        for i in range(len(true_h)):
            sum_squared_diff[i] = (true_h[i] - np.real(self.surrogate[i, idx]))**2

        surrogate_error = (self.TS_M[1] - self.TS_M[0]) * sum_squared_diff
        # print('surrogate error = ', surrogate_error, ' , eccmin = ', self.parameter_space[idx])

        fig_surrogate_err = plt.figure(figsize=(8, 5))
        
        plt.plot(self.TS_M, sum_squared_diff)  
        plt.title(self.parameter_space[idx])
        # plt.ylim(-1e-42, 1e-42)
        # plt.show()

        return surrogate_error


    def plot_surrogate(self, idx, min_greedy_error_amp, min_greedy_error_phase, save_dataset=False):

        if self.surrogate is None:
            self.generate_surrogate_waveform(min_greedy_error_amp, min_greedy_error_phase, save_dataset=False)


        fig_surrogate = plt.figure(figsize=(8, 5))
        print(self.TS_M)
        true_h = self.plot_sim_inspiral_mass_indp(polarisation='plus', eccmin=self.parameter_space[idx])
        plt.plot(self.TS_M, np.real(self.surrogate[:, idx]), linewidth=0.6, label='Surrogate')
        plt.scatter(self.TS_M[100], self.surrogate[100, idx])
        plt.scatter(self.TS_M[100], true_h[100])
        plt.xlabel('t [M]')
        plt.grid(True)
        plt.legend()


        surrogate_error = self.calc_surrogate_error(true_h, idx)

        figname = f'Surrogate total mass = {self.total_mass}, mass ratio = {self.mass_ratio}, ecc = {min(self.eccmin_list)}_{max(self.eccmin_list)}, eccsur = {self.parameter_space[idx]}.png'
        fig_surrogate.savefig('Images/Surrogate_model/' + figname)
        # print('fig is saved')   

        # plt.show()


# print(indexes)
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


# ds = Dataset(eccmin_list=np.linspace(0.01, 0.1, num=20).round(3), waveform_size=3500)
# ds.generate_dataset_property('Phase', save_dataset=True, training_set=True)
# ds.generate_dataset_property('Amplitude', save_dataset=True, training_set=True)

ds = Dataset(eccmin_list=np.linspace(0.1, 0.2, num=500).round(4), waveform_size=3500)
ds.generate_dataset_property('Phase', save_dataset=True, training_set=True)
ds.generate_dataset_property('Amplitude', save_dataset=True, training_set=True)


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

TS1 = Training_set(eccmin_list=np.linspace(0.01, 0.1, num=20).round(3), freqmin=18, waveform_size=3500)

# TS1.generate_training_set(min_greedy_error=1e-3, property='Phase')
# TS.gaussian_process_regression()

# TS1.plot_empirical_nodes(min_greedy_error=1e-2, property='Amplitude', save_dataset=True, plot_greedy_error=True)
# TS1.plot_empirical_nodes(min_greedy_error=1e-3, property='Phase', save_dataset=True, plot_greedy_error=True)
# TS1.plot_training_set_at_node(time_node_idx=2, min_greedy_err_phase=1e-3, min_greedy_err_amp=1e-2)
# TS1.plot_training_set_at_node(time_node_idx=8, min_greedy_err_phase=1e-3, min_greedy_err_amp=1e-2)
# TS.plot_training_set_at_node(time_node_idx=15, min_greedy_err_phase=1e-3, min_greedy_err_amp=1e-2)
# TS.plot_training_set_at_node(time_node_idx=20, min_greedy_err_phase=1e-3, min_greedy_err_amp=1e-2)
# print('gaussian')
# TS1.gaussian_process_regression_fit(min_greedy_error=1e-3, property='Phase', save_dataset=True, plot_time_node_idx=2)
# TS.gaussian_process_regression_fit(min_greedy_error=1e-2, property='Amplitude', save_dataset=True, plot_time_node_idx=2)
# TS.gaussian_process_regression_fit(time_node_idx=8, min_greedy_error=1e-3, property='Phase', save_dataset=True, plot_GPR=True)
# TS.gaussian_process_regression_fit(time_node_idx=8, min_greedy_error=1e-2, property='Amplitude', save_dataset=True, plot_GPR=True)
# TS.gaussian_process_regression_fit(time_node_idx=15, min_greedy_error=1e-3, property='Phase', save_dataset=True, plot_GPR=True)
# TS.gaussian_process_regression_fit(time_node_idx=15, min_greedy_error=1e-2, property='Amplitude', save_dataset=True, plot_GPR=True)
# TS.plot_gaussian_process_regression_fit(time_node_idx=20, min_greedy_error=1e-3, property='Phase', save_dataset=True)
# TS.plot_gaussian_process_regression_fit(time_node_idx=20, min_greedy_error=1e-2, property='Amplitude', save_dataset=True)


TS1.generate_surrogate_waveform(min_greedy_error_amp=1e-3, min_greedy_error_phase=1e-3)
# print('surrogate')
# TS1.plot_surrogate(25, 1e-2, 1e-3, save_dataset=True)
# TS.plot_surrogate(1000, 'plus')
# TS.plot_surrogate(2000, 'plus')
# TS.plot_surrogate(2800, 'plus')
# TS.calc_surrogate_error()

plt.show()


TS2 = Training_set(eccmin_list=np.linspace(0.1, 0.2, num=20).round(3), freqmin=18, waveform_size=3500)
# TS2.plot_empirical_nodes(min_greedy_error=1e-2, property='Amplitude', save_dataset=True, plot_greedy_error=True)
# TS2.plot_empirical_nodes(min_greedy_error=1e-3, property='Phase', save_dataset=True, plot_greedy_error=True)
# TS2.plot_training_set_at_node(time_node_idx=2, min_greedy_err_phase=1e-3, min_greedy_err_amp=1e-2)
# TS2.plot_training_set_at_node(time_node_idx=8, min_greedy_err_phase=1e-3, min_greedy_err_amp=1e-2)


TS2.generate_surrogate_waveform(min_greedy_error_amp=1e-2, min_greedy_error_phase=1e-3)
# # print('surrogate')
# TS2.plot_surrogate(25, 1e-2, 1e-3, save_dataset=True)
# # Example usage
# # training_set = Training_set([0.1, 0.2, 0.3], waveform_size=1000, total_mass=10, mass_ratio=1, freqmin=5)
plt.show()