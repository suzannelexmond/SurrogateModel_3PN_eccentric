from generate_eob_training_set import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Product, ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel, ExpSineSquared, DotProduct, ConstantKernel as C
from sklearn.model_selection import train_test_split
import faulthandler
from pycbc.filter import match
from pycbc.psd import aLIGOZeroDetHighPower
import pycbc.psd
import time
import seaborn as sns
from matplotlib.lines import Line2D
from inspect import currentframe, getframeinfo

f = currentframe()
plt.switch_backend('WebAgg')


faulthandler.enable()

class Generate_Surrogate(Generate_TrainingSet):

    def __init__(self, parameter_space, amount_input_wfs, amount_output_wfs, N_greedy_vecs_amp=None, N_greedy_vecs_phase=None, min_greedy_error_amp=None, min_greedy_error_phase=None, waveform_size=None, mass_ratio=1, freqmin=650):
        
        self.parameter_space_input = np.linspace(parameter_space[0], parameter_space[1], amount_input_wfs).round(4)
        self.parameter_space_output = np.linspace(parameter_space[0], parameter_space[1], amount_output_wfs).round(4)
        
        self.min_greedy_error_amp = min_greedy_error_amp
        self.min_greedy_error_phase = min_greedy_error_phase
        self.N_greedy_vecs_amp = N_greedy_vecs_amp
        self.N_greedy_vecs_phase = N_greedy_vecs_phase
        self.surrogate = None
        self.surrogate_amp = None
        self.surrogate_phase = None
        
        Generate_TrainingSet.__init__(self, parameter_space_input=self.parameter_space_input, waveform_size=waveform_size, mass_ratio=mass_ratio, freqmin=freqmin)

    def fit_to_training_set(self, property, min_greedy_error=None, N_greedy_vecs=None, save_fits_to_file=True, plot_kernels=False, plot_fits=False, save_fig_kernels=False, save_fig_fits=False, plot_residuals_ecc_evolve=False, save_fig_ecc_evolve=False, plot_residuals_time_evolve=False, save_fig_time_evolve=False):
        
        def gaussian_process_regression(time_node, training_set, optimized_kernel=None, plot_kernels=plot_kernels, save_fig_kernels=save_fig_kernels):
            # Extract X and training data
            X = self.parameter_space_output[:, np.newaxis]
            X_train = np.array(self.parameter_space_input[self.greedy_parameters_idx]).reshape(-1, 1)
            y_train = np.squeeze(training_set.T[time_node])

            # Scale X_train
            scaler_x = StandardScaler()
            X_train_scaled = scaler_x.fit_transform(X_train)

            # Scale X (for predictions)
            X_scaled = scaler_x.transform(X)

            # Scale y_train
            scaler_y = StandardScaler()
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

            kernels = [
                Matern(length_scale=0.1, length_scale_bounds=(1e-1, 1), nu=1.5)  # <= 0.3 eccentricity
            ]

            mean_prediction_per_kernel = []
            std_predictions_per_kernel = []
            lml_per_kernel = []

            for kernel in kernels:
                start = time.time()
                if optimized_kernel is None:
                    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
                else:
                    gaussian_process = GaussianProcessRegressor(kernel=optimized_kernel, optimizer=None)

                # Fit the GP model on scaled data
                gaussian_process.fit(X_train_scaled, y_train_scaled)
                optimized_kernel = gaussian_process.kernel_

                end = time.time()

                # Log-Marginal Likelihood
                lml = gaussian_process.log_marginal_likelihood_value_
                lml_per_kernel.append(lml)

                # Print the optimized kernel and hyperparameters
                print(f"kernel = {kernel}; Optimized kernel: {optimized_kernel} | time = {end - start:.2f}s | LML = {lml:.4f}")

                # Make predictions on scaled X
                mean_prediction_scaled, std_prediction_scaled = gaussian_process.predict(X_scaled, return_std=True)
                mean_prediction = scaler_y.inverse_transform(mean_prediction_scaled.reshape(-1, 1)).flatten()
                std_prediction = std_prediction_scaled * scaler_y.scale_[0]

                mean_prediction_per_kernel.append(mean_prediction)
                std_predictions_per_kernel.append(std_prediction)
            
            if plot_kernels is True:
                GPR_fit = plt.figure()

                for i in range(len(mean_prediction_per_kernel)):
                    plt.scatter(X_train, y_train, color='red', label="Observations", s=10)
                    plt.plot(X, mean_prediction_per_kernel[i], label='Mean prediction', linewidth=0.8)
                    plt.fill_between(
                        X.ravel(),
                    (mean_prediction_per_kernel[i] - 1.96 * std_predictions_per_kernel[i]), 
                    (mean_prediction_per_kernel[i] + 1.96 * std_predictions_per_kernel[i]),
                        alpha=0.5,
                        label=r"95% confidence interval",
                    )
                plt.legend(loc = 'upper left')
                plt.xlabel("$e$")
                if property == 'amplitude':
                    plt.ylabel("$f_A(e)$")
                elif property == 'phase':
                    plt.ylabel("$f_{\phi}(e)$")
                # plt.title(f"GPR {property} at T_{time_node}")
                # plt.show()

                if save_fig_kernels is True:
                    figname = f'Gaussian_kernels_{property}_ecc=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}]_q={self.mass_ratio}_fmin={self.freqmin}_iN={len(self.parameter_space_input)}_oN={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}.png'
                    
                    # Ensure the directory exists, creating it if necessary and save
                    os.makedirs('Images/Gaussian_kernels', exist_ok=True)
                    GPR_fit.savefig('Images/Gaussian_kernels/' + figname)

                    print('Figure is saved in Images/Gaussian_kernels')

            return mean_prediction, [(mean_prediction - 1.96 * std_prediction), (mean_prediction + 1.96 * std_prediction)], optimized_kernel, lml_per_kernel

        try:
            # if self.waveform_size is None:
                # self.waveform_size = self.simulate_inspiral_mass_independent(max(self.parameter_space_input))[0].shape[1]
            load_GPRfits = np.load(f'Straindata/GPRfits/{property}_q={self.mass_ratio}_fmin={self.freqmin}_ecc=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}]_Ni={len(self.parameter_space_input)}_No={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}_size={len(self.simulate_inspiral_mass_independent(max(self.parameter_space_input))[0]) - 20}.npz', allow_pickle=True)
            
            gaussian_fit = load_GPRfits['GPR_fit']
            training_set = load_GPRfits['training_set']
            uncertainty_region = load_GPRfits['uncertainty_region']
            self.greedy_parameters_idx = load_GPRfits['greedy_parameters']
            self.empirical_nodes_idx = load_GPRfits['empirical_nodes']
            self.residual_greedy_basis = load_GPRfits['residual_greedy_basis']
            lml_fits = load_GPRfits['lml_fits']
            self.TS = load_GPRfits['TS']
            self.waveform_size = len(self.residual_greedy_basis[0])
            
            print(f'GPRfit {property} load succeeded')

        except Exception as e:
            print(f'line {getframeinfo(f).lineno}: {e}')

            # Generate the training set of greedy parameters at empirical nodes
            training_set = self.get_training_set(property=property, min_greedy_error=min_greedy_error, N_greedy_vecs=N_greedy_vecs)
            # Create empty arrays to save fitvalues
            gaussian_fit = np.zeros((len(training_set.T), len(self.parameter_space_output)))
            uncertainty_region = []
            lml_fits = []

            print(f'Interpolate {property}...')

            # start1 = time.time()
            # mean_prediction, uncertainty_region = gaussian_process_regression_all(training_set, self.greedy_parameters_idx, plot_fits)
            # end1 = time.time()
            # print(f'time1 = {end1 - start1}')
            start2 = time.time()
            optimized_kernel = None
            for node_i in range(len(self.empirical_nodes_idx)):
                
                mean_prediction, uncertainty_region, optimized_kernel, lml = gaussian_process_regression(node_i, training_set, optimized_kernel, plot_kernels)
                
                gaussian_fit[node_i] = mean_prediction # Best prediction 
                uncertainty_region.append(uncertainty_region) # 95% confidence level
                lml_fits.append(lml) # Log-Marginal likelihood

            end2 = time.time()
            print(f'time full GPR = {end2 - start2}')

        if plot_fits is True:
            load_parameterspace_input = np.load(f'Straindata/Residuals/residuals_{property}_q={self.mass_ratio}_fmin={self.freqmin}_e=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_N={len(self.parameter_space_input)}].npz')
            
            residual_parameterspace_input = load_parameterspace_input['residual']
            self.TS = load_parameterspace_input['TS']

            try:
                load_residual_output = np.load(f'Straindata/Residuals/residuals_{property}_q={self.mass_ratio}_fmin={self.freqmin}_e=[{min(self.parameter_space_output)}_{max(self.parameter_space_output)}_N={len(self.parameter_space_output)}].npz')
                residual_parameterspace_output = load_residual_output['residual']

            except Exception as e:
                print(f'line {getframeinfo(f).lineno}: {e}')
                residual_parameterspace_output = self.generate_property_dataset(eccmin_list=self.parameter_space_output, property=property, save_dataset_to_file=True)
            
            # print(0, residual_parameterspace_output.shape, len(self.TS))
            # fig_test = plt.figure()
            # for i in range(len(self.empirical_nodes_idx)):
            #     # print( residual_parameterspace_output.T[i])
            #     print('emp nodes: ', self.empirical_nodes_idx[i])
            # for i in range(len(self.TS)):
            #     plt.plot(self.parameter_space_output, residual_parameterspace_output.T[i])
            # plt.plot(self.TS, residual_parameter)
            # plt.ylabel('residuals')
            # plt.xlabel('eccentricity')
            # plt.show()

            # Define a distinct color palette
            color_palette = plt.cm.tab10.colors 

            # Number of distinct colors 
            num_colors = len(gaussian_fit)

            # Create a color map to use consistent colors for matching fits and data sets
            colors = [color_palette[i % len(color_palette)] for i in range(num_colors)]

            fig_residual_training_fit, axs = plt.subplots(2, 1, figsize=(11,6), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1}, sharex=True)

            # Create a colormap from Viridis
            color_palette = sns.color_palette("tab10", as_cmap=True)
            # Number of distinct colors needed
            num_colors = len(gaussian_fit)  # Replace with the actual number of datasets (e.g., len(gaussian_fit))
            # Evenly sample the colormap
            colors = [color_palette(i / (num_colors - 1)) for i in range(num_colors)]

            # Define custom legend elements
            custom_legend_elements = [
                Line2D([0], [0], linestyle='dashed', color='black', linewidth=1, label=f'true {property}'),
                Line2D([0], [0], marker='o', linestyle='None', color='black', label='training Points')
            ]

            # Create a list to hold dynamic handles and labels
            dynamic_handles = []
            dynamic_labels = []

            # Sort gaussian fits by worst likelihood 
            sorted_indices = np.argsort(np.array(lml_fits).flatten())

            # Plotting data
            for i in range(len(gaussian_fit)):
                # Plot Gaussian fit
                line_fit, = axs[0].plot(self.parameter_space_output, gaussian_fit.T[:, i], linewidth=0.6, 
                                        label=f't={int(self.TS[self.empirical_nodes_idx[i]])}')
                
                # Scatter plot for training points
                axs[0].scatter(self.parameter_space_input[self.greedy_parameters_idx], training_set[:, i], s=6)
               
                # Plot residuals (true property)
                axs[0].plot(self.parameter_space_output, residual_parameterspace_output[:, self.empirical_nodes_idx[i]], 
                            linestyle='dashed', linewidth=0.6, label=f'{i}, {self.empirical_nodes_idx[i]}')
                
                # Collect handles and labels for the dynamic fits
                dynamic_handles.append(line_fit)
                dynamic_labels.append(f't={int(self.TS[self.empirical_nodes_idx[i]])} [M]')
                
                # Relative error plot
                axs[1].plot(self.parameter_space_output, 
                            abs(residual_parameterspace_output[:, self.empirical_nodes_idx[i]] - gaussian_fit.T[:, i]), 
                            linewidth=0.6, 
                            label=f'Error {i+1} (t={int(self.TS[self.empirical_nodes_idx[i]])})')

            # Combine custom and dynamic legend elements
            combined_handles = custom_legend_elements + dynamic_handles
            combined_labels = [handle.get_label() for handle in custom_legend_elements] + dynamic_labels

            # Add the combined legend to the top-left subplot
            axs[0].legend(combined_handles, combined_labels, loc='lower left', ncol=2)

            # Set labels and titles
            if property == 'phase':
                axs[0].set_ylabel('$\Delta \phi$')
                # axs[0].set_title(f'GPRfit $\phi$; greedy error = {min_greedy_error},N={len(self.greedy_parameters_idx)}')
            elif property == 'amplitude':
                axs[0].set_ylabel('$\Delta$ A')
                # axs[0].set_title(f'GPRfit A; greedy error = {min_greedy_error}, N={len(self.greedy_parameters_idx)}')
            axs[0].grid()

            axs[1].set_xlabel('eccentricity')
            if property == 'phase':
                axs[1].set_ylabel('|$\Delta \phi_{S} - \Delta \phi|$')
            else:
                axs[1].set_ylabel('|$\Delta A_{S} - \Delta A|$')
            axs[1].grid()

            plt.tight_layout()

            if save_fig_fits is True:
                figname = f'GPR_fits_{property}_ecc=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}]_q={self.mass_ratio}_fmin={self.freqmin}_iN={len(self.parameter_space_input)}_oN={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}_size={self.waveform_size}.png'
                
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Gaussian_fits', exist_ok=True)
                fig_residual_training_fit.savefig('Images/Gaussian_fits/' + figname)

                print('Figure is saved in Images/Gaussian_fits')

        if save_fits_to_file is True and not os.path.isfile(f'Straindata/GPRfits/{property}_q={self.mass_ratio}_fmin={self.freqmin}_ecc=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}]_Ni={len(self.parameter_space_input)}_No={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}_size={len(self.simulate_inspiral_mass_independent(max(self.parameter_space_input))[0]) - 20}.npz'):

            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Straindata/GPRfits', exist_ok=True)
            np.savez(f'Straindata/GPRfits/{property}_q={self.mass_ratio}_fmin={self.freqmin}_ecc=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}]_Ni={len(self.parameter_space_input)}_No={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}_size={self.waveform_size}.npz', GPR_fit=gaussian_fit, training_set=training_set, uncertainty_region=np.array(uncertainty_region, dtype=object), greedy_parameters=self.greedy_parameters_idx, empirical_nodes=self.empirical_nodes_idx, residual_greedy_basis=self.residual_greedy_basis, lml_fits=lml_fits, TS=self.TS)
            print('GPR fits saved in Straindata/GPRfits')
        
        if (plot_residuals_time_evolve is True) or (plot_residuals_time_evolve is True):
            load_parameterspace_input = np.load(f'Straindata/Residuals/residuals_{property}_q={self.mass_ratio}_fmin={self.freqmin}_e=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_N={len(self.parameter_space_input)}].npz')
            residual_parameterspace_input = load_parameterspace_input['residual']
            
            self._plot_residuals(residual_dataset=residual_parameterspace_input, eccmin_list=self.parameter_space_input, property=property, plot_eccentric_evolv=plot_residuals_ecc_evolve, save_fig_eccentric_evolve=save_fig_ecc_evolve, plot_time_evolve=plot_residuals_time_evolve, save_fig_time_evolve=save_fig_time_evolve)
        
        #     fig_residual_training_fit, axs = plt.subplots(2, 1, figsize=(11,6), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.4}, sharex=True)

        #     # Top left subplot for amplitude main plot
        #     for i in range(len(gaussian_fit[:3])):
        #         # Use the same color for the fit and the corresponding empirical data
        #         axs[0].plot(self.parameter_space_output, gaussian_fit.T[:, i], color=colors[i], linewidth=0.6)
        #         axs[0].scatter(self.parameter_space_input[self.greedy_parameters_idx], training_set[:, i], color=colors[i])
        #         # axs[0].plot(self.parameter_space_output, residual_parameterspace_output[:, self.empirical_nodes_idx[i]], linestyle='dashed', linewidth=0.6, color=colors[i], label=f't={self.TS[self.empirical_nodes_idx[i]]}')
        #         axs[0].plot(self.parameter_space_output, residual_parameterspace_output[:, self.empirical_nodes_idx[i]], linestyle='dashed', linewidth=0.6, color=colors[i])

        #         # axs[0].plot(self.parameter_space_output, residual_parameterspace_output[:, -1], linestyle='dashed', linewidth=0.6, color=colors[i], label=f't={self.TS[self.empirical_nodes_idx[i]]}')
        #         # axs[0].plot(self.parameter_space_input, residual_parameterspace_input[:, -1])
        #         relative_error = abs(residual_parameterspace_output[:, self.empirical_nodes_idx[i]] - gaussian_fit.T[:, i]) / abs(residual_parameterspace_output[:, self.empirical_nodes_idx[i]])
        #         axs[1].plot(self.parameter_space_output, relative_error, color=colors[i], linewidth=0.6, label=f'Error {i+1} (t={int(self.TS[self.empirical_nodes_idx[i]])})')
        #         axs[1].set_ylim(0, 2)
        #     # Adjust legend: smaller font, inside figure, upper left
        #     axs[0].legend(loc='upper left', ncol=3, bbox_to_anchor=(0.01, 0.99), fontsize='small')
        #     if property == 'phase':
        #         axs[0].set_ylabel('$\Delta \phi$ ')
        #         axs[0].set_title(f'GPRfit $\phi$; greedy error = {min_greedy_error}')
        #     elif property == 'amplitude':
        #         axs[0].set_ylabel('$\Delta$ A')
        #         axs[0].set_title(f'GPRfit A; greedy error = {min_greedy_error}')
        #     axs[0].grid()

        #     # Adjust legend: smaller font, inside figure, upper right
        #     axs[1].set_xlabel('eccentricity')
        #     axs[1].set_ylabel('Relative fit error')
        #     # axs[1].set_xlim(0., 0.105)
        #     # axs[1].set_ylim(-100, 100)
        #     axs[1].grid()

        #     # Adjust layout to prevent overlap
        #     plt.tight_layout()

        #     if save_fig_fits is True:
        #         figname = f'GPR_fits_{property}_ecc=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}]_q={self.mass_ratio}_fmin={self.freqmin}_iN={len(self.parameter_space_input)}_oN={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}.png'
                
        #         # Ensure the directory exists, creating it if necessary and save
        #         os.makedirs('Images/Gaussian_fits', exist_ok=True)
        #         fig_residual_training_fit.savefig('Images/Gaussian_fits/' + figname)

        #         print('Figure is saved in Images/Gaussian_fits')
        
        # if save_fits_to_file is True and not os.path.isfile(f'Straindata/GPRfits/{property}_q={self.mass_ratio}_fmin={self.freqmin}_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_Ni={len(self.parameter_space_input)}_No={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}_size={self.waveform_size}.npz'):

        #     # Ensure the directory exists, creating it if necessary and save
        #     os.makedirs('Straindata/GPRfits', exist_ok=True)
        #     np.savez(f'Straindata/GPRfits/{property}_q={self.mass_ratio}_fmin={self.freqmin}_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_Ni={len(self.parameter_space_input)}_No={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}_size={self.waveform_size}.npz', GPR_fit=gaussian_fit, training_set=training_set, uncertainty_region=np.array(uncertainty_region, dtype=object), greedy_parameters=self.greedy_parameters_idx, empirical_nodes=self.empirical_nodes_idx, residual_greedy_basis=self.residual_greedy_basis)
        #     print('GPR fits saved in Straindata/GPRfits')

        return gaussian_fit, uncertainty_region

    def generate_surrogate_model(self, plot_surr_datapiece_at_ecc=None, save_fig_datapiece=False, plot_surr_at_ecc=None, save_fig_surr=False, plot_GPRfit=False, save_fits_to_file=True, save_fig_fits=False, save_surr_to_file=False):

        def compute_B_matrix():

            """
            Computes the B matrix for all empirical nodes and basis functions.
            
            e_matrix: Array of shape (m, time_samples) representing the reduced basis functions evaluated at different time samples.
            V_inv: Inverse of the interpolation matrix of shape (m, m).
            
            Returns:
            B_matrix: Array of shape (m, time_samples) where each row represents B_j(t) for j=1,2,...,m
            """

            m, time_samples = self.residual_greedy_basis.shape
            B_matrix = np.zeros((m, time_samples))

            V = np.zeros((m, m))
            for j in range(m):
                for i in range(m):
                    V[j][i] = self.residual_greedy_basis[i][self.empirical_nodes_idx[j]]

            V_inv = np.linalg.pinv(V)

            
            # Compute each B_j(t) for j = 1, 2, ..., m
            for j in range(m):
                # Compute B_j(t) as a linear combination of all e_i(t) with weights from V_inv[:, j]
                for i in range(m):
                    B_matrix[j] += self.residual_greedy_basis[i] * V_inv[i, j]
                    
            
            return B_matrix
        
        
        def residual_to_original(residual_dataset, property):

            self.circulair_wf()

            if property == 'phase':
                circ = np.array(waveform.utils.phase_from_polarizations(self.hp_TS_circ, self.hc_TS_circ)) 
            elif property == 'amplitude':
                circ = np.array(waveform.utils.amplitude_from_polarizations(self.hp_TS_circ, self.hc_TS_circ))
            else:
                print('Choose property = "phase", "amplitude"')
                sys.exit(1)
            

            # length_diff = len(circ) - len(residual_dataset[0])
            maximum_length = min(len(circ), residual_dataset.shape[0])
            original_dataset = np.zeros((maximum_length, residual_dataset.shape[1]))

            for i in range(residual_dataset.shape[1]):
                if property == 'phase':
                    original_dataset[:, i] = circ[-maximum_length:] - residual_dataset[:, i][-maximum_length:]
                elif property == 'amplitude':
                    original_dataset[:, i] = residual_dataset[:, i][-maximum_length:] + circ[-maximum_length:]
                else:
                    print('Choose property == "amplitude"or "phase"')
                    sys.exit(1)

            return original_dataset

        def reconstruct_surrogate_datapiece(property, B_matrix, fit_matrix, plot_surr_datapiece_at_ecc=plot_surr_datapiece_at_ecc, save_fig_datapiece=save_fig_datapiece):
            """
            Reconstructs the surrogate model for a given parameter using different empirical nodes for amplitude and phase.
            
            Parameters:
            ------------------
            B_matrix (numpy.ndarray), shape (m, time_samples): Empricial interpolant matrix
            fit_matrix (numpy.ndarray), shape (m, lambda): Array of fitted greedy parameters at time nodes with lambda as the number of parameters in parameter_space.
            time_samples (numpy.ndarray), shape (time_samples, 1): Array representing the time-domain samples.
            plot_surr_datapiece_at_ecc (float) : Set this to a ecc_min value for plot of surrogate datapiece as comparison with real estimated value at given ecc_min.
            
            Returns:
            ------------------
            surrogate_datapiece (numpy.ndarray), shape (time_samples, lambda): Array representing the reconstructed surrogate waveform datapiece (amplitude or phase).
            """
            try:
                load_surrogate = np.load(f'Straindata/Surrogate_datapieces/Surrogate_datapieces_q={self.mass_ratio}_fmin={self.freqmin}_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_Ni={len(self.parameter_space_input)}_No={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}_size={self.waveform_size}.npz')
                

                if property == 'phase':
                    surrogate_datapiece = load_surrogate['surrogate_phase']
                    computation_time = load_surrogate['computation_t_phase']
                elif property == 'amplitude':
                    surrogate_datapiece = load_surrogate['surrogate_amp']
                    computation_time = load_surrogate['computation_t_amp']
                
                self.TS = load_surrogate['TS']
                self.waveform_size = len(self.TS)

            except Exception as e:
                print(f'line {getframeinfo(f).lineno}: {e}')
                
                computation_time = None
                m, _ = B_matrix.shape

                reconstructed_residual = np.zeros((len(self.TS), len(self.parameter_space_output)))

                for i in range(m):
                    reconstructed_residual += np.dot(B_matrix[i, :].reshape(-1, 1), fit_matrix[i, :].reshape(1, -1))


                # Change back from residual to original (+ circulair)
                surrogate_datapiece = residual_to_original(residual_dataset=reconstructed_residual, property=property)

            if (plot_surr_datapiece_at_ecc is not None) and (not isinstance(plot_surr_datapiece_at_ecc, float)):
                print('plot_surr_datapiece_at_ecc must be float value! Plot did not get generated.')

            if isinstance(plot_surr_datapiece_at_ecc, float):
                try:
                    index_ecc_min = np.where(self.parameter_space_output == plot_surr_datapiece_at_ecc)[0][0]
                except:
                    plot_surr_datapiece_at_ecc = self.parameter_space_output[np.abs(self.parameter_space_output - plot_surr_datapiece_at_ecc).argmin()]
                    index_ecc_min = np.where(self.parameter_space_output == plot_surr_datapiece_at_ecc)[0][0]
                    print(f'Eccentricity value {plot_surr_datapiece_at_ecc} not in ouput parameterspace. Eccentricity calculated for closest existing value at e={plot_surr_datapiece_at_ecc}.')

                # Create a 2x1 subplot grid with height ratios 3:1
                fig_surrogate_datapieces, axs = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1}, sharex=True)

                # Simulate the real waveform datapiece
                real_hp, real_hc, real_TS = self.simulate_inspiral_mass_independent(plot_surr_datapiece_at_ecc)
                minimum_length = min(len(real_TS), self.waveform_size)

                if property == 'amplitude':
                    real_datapiece = np.array(waveform.utils.amplitude_from_polarizations(real_hp, real_hc))[-minimum_length:]
                    units = ''
                elif property == 'phase':
                    real_datapiece = np.array(waveform.utils.phase_from_polarizations(real_hp, real_hc))[-minimum_length:]
                    units = ' [radians]'

                # Plot Surrogate and Real Amplitude (Top Left)
                # axs[0].plot(self.parameter_space_output, surrogate_datapiece[index_ecc_min], label='surr')
                axs[0].plot(self.TS, surrogate_datapiece.T[index_ecc_min][-minimum_length:], linewidth=0.6, label=f'surrogate e = {plot_surr_datapiece_at_ecc}')
                # axs[0].plot(self.TS, true_phase[index_ecc_min], linewidth=0.6, label=f'Surrogate: e = {plot_surr_datapiece_at_ecc}')
                axs[0].plot(self.TS, real_datapiece, linewidth=0.6, linestyle='dashed', label=f'true {property} e = {plot_surr_datapiece_at_ecc}')
                # axs[0].plot(self.parameter_space_output, true_phase[:, index_ecc_min], label='real')
                # axs[0].set_xlabel('t [M]')
                if property == 'phase':
                    axs[0].set_ylabel('$\phi$' + units)
                else:
                    axs[0].set_ylabel('A' + units)
                axs[0].grid(True)
                # axs[0].set_title(f'Surrogate vs Real {property}, ga={self.min_greedy_error_amp}, gp={self.min_greedy_error_phase}')
                axs[0].legend(loc='upper left')

                # Calculate and Plot Phase Error (Bottom Right)
                # Define a small threshold value to handle small or zero values in real_datapiece
                threshold = 1e-30  # You can adjust this value based on the scale of your data

                # Avoid division by very small numbers by using np.maximum to set a lower limit
                relative_error = abs(surrogate_datapiece.T[index_ecc_min] - real_datapiece) / abs(real_datapiece)
                axs[1].plot(self.TS, relative_error, linewidth=0.6)
                if property == 'phase':
                    axs[1].set_ylabel('|($\phi_S$ - $\phi$) / $\phi$|')
                else:
                    axs[1].set_ylabel('|($A_S$ - A) / A|')
                axs[1].set_xlabel('t [M]')
                axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
                axs[1].grid(True)

                # axs[1].set_title('Relative error')

                # Adjust layout to prevent overlap
                plt.tight_layout()

                if save_fig_datapiece is True:
                    figname = f'Surrogate_{property}_eccmin={plot_surr_datapiece_at_ecc}_for_ecc=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}]_q={self.mass_ratio}_fmin={self.freqmin}_iN={len(self.parameter_space_input)}_oN={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.min_greedy_error_phase}_Nga={self.N_greedy_vecs_amp}_size={self.N_greedy_vecs_phase}.png'
                    
                    # Ensure the directory exists, creating it if necessary and save
                    os.makedirs('Images/Surrogate_datapieces', exist_ok=True)
                    fig_surrogate_datapieces.savefig('Images/Surrogate_datapieces/' + figname)

                    print('Figure is saved in Images/Surrogate_datapieces')
                    
            return surrogate_datapiece, computation_time

        # Set timer for computational time of the surrogate model
        start_time_amp = time.time()
        # Get matrix with interpolated fits and B_matrix
        fit_matrix_amp = self.fit_to_training_set(min_greedy_error=self.min_greedy_error_amp, N_greedy_vecs=self.N_greedy_vecs_amp, property='amplitude', plot_fits=plot_GPRfit, save_fig_fits=save_fig_fits, save_fits_to_file=save_fits_to_file)[0]
        B_matrix_amp = compute_B_matrix()
        # Reconstruct amplitude datapiece
        surrogate_amp, computation_time_amp = reconstruct_surrogate_datapiece(property='amplitude', B_matrix=B_matrix_amp, fit_matrix=fit_matrix_amp, plot_surr_datapiece_at_ecc=plot_surr_datapiece_at_ecc)
        self.surrogate_amp = surrogate_amp[:-30, :] # Cut off first few points because of the tapering effects of the shortest length waveform

        end_time_amp = time.time()
        start_time_phase = time.time()

        # Get matrix with interpolated fits and B_matrix
        fit_matrix_phase = self.fit_to_training_set(min_greedy_error=self.min_greedy_error_phase, N_greedy_vecs=self.N_greedy_vecs_phase, property='phase', plot_fits=plot_GPRfit, save_fig_fits=save_fig_fits, save_fits_to_file=save_fits_to_file)[0]
        B_matrix_phase = compute_B_matrix()

        # Reconstruct phase datapiece
        surrogate_phase, computation_time_phase = reconstruct_surrogate_datapiece(property='phase', B_matrix=B_matrix_phase, fit_matrix=fit_matrix_phase, plot_surr_datapiece_at_ecc=plot_surr_datapiece_at_ecc)
        self.surrogate_phase = surrogate_phase[:-30, :] # Cut off first few points because of the tapering effects of the shortest length waveform

        # End timer for computation of surrogate model
        end_time_phase = time.time()
        # Compute total computational time of the surrogate datapieces
        if computation_time_phase is None:
            computation_time_amp = end_time_amp - start_time_amp
            computation_time_phase = end_time_phase - start_time_phase

        if save_surr_to_file is True and not os.path.isfile(f'Straindata/Surrogate_datapieces/Surrogate_datapieces_q={self.mass_ratio}_fmin={self.freqmin}_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_Ni={len(self.parameter_space_input)}_No={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}_size={self.waveform_size}.npz'):
            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Straindata/Surrogate_datapieces', exist_ok=True)
            np.savez(f'Straindata/Surrogate_datapieces/Surrogate_datapieces_q={self.mass_ratio}_fmin={self.freqmin}_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_Ni={len(self.parameter_space_input)}_No={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}_size={self.waveform_size}.npz', surrogate_amp=surrogate_amp, surrogate_phase=surrogate_phase, computation_t_amp=computation_time_amp, computation_t_phase=computation_time_phase, TS=self.TS)
            print('Surrogate datapieces saved in Straindata/Surrogate_datapieces')


        h_surrogate = np.zeros((len(self.TS), len(self.parameter_space_output)), dtype=complex)
        h_surrogate = surrogate_amp * np.exp(1j * surrogate_phase)


        if plot_surr_at_ecc is not False and (not isinstance(plot_surr_at_ecc, float)):
            print('plot_surr_at_ecc must be float value!')

        if isinstance(plot_surr_at_ecc, float):

            try:
                index_ecc_min = np.where(self.parameter_space_output == plot_surr_at_ecc)[0][0]
            except:
                plot_surr_at_ecc = self.parameter_space_output[np.abs(self.parameter_space_output - plot_surr_at_ecc).argmin()]
                print(f'Eccentricity value {plot_surr_at_ecc} not in ouput parameterspace. Eccentricity calculated for closest existing value at e={plot_surr_at_ecc}.')

            
            # fig_surrogate, axs = plt.subplots(4, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1, 3, 1], 'hspace': 0.2}, sharex=True)
            # fig_surrogate, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

            # try:
            #     index_ecc_min = np.where(self.parameter_space_output == plot_surr_at_ecc)[0][0]
            # except:
            #     print(f'Eccentricity value {plot_surr_at_ecc} not in parameterspace!')
            #     sys.exit(1)

            # true_hp, true_hc, TS = self.simulate_inspiral_mass_independent(plot_surr_at_ecc)
            # phase = np.array(waveform.utils.phase_from_polarizations(true_hp, true_hc))
            # amp = np.array(waveform.utils.amplitude_from_polarizations(true_hp, true_hc))

            # true_h = amp * np.exp(1j * phase)
            # length_diff = len(true_hp) - self.waveform_size

            # axs[0].set_title(f'Eccentricity = {plot_surr_at_ecc}')
            # # axs[0].plot(self.TS, true_hp[length_diff:], linewidth=0.6, label='True waveform before')
            # axs[0].plot(self.TS, np.real(true_h)[length_diff:], linewidth=0.6, label='True waveform after')
            # axs[0].plot(self.TS, np.real(h_surrogate[:, index_ecc_min]), linewidth=0.6, label='Surrogate')
            # axs[0].set_ylabel('$h_+$')
            # axs[0].grid(True)
            # axs[0].legend()

            # # Calculate and Plot plus polarisation error 
            # relative_error_hp = abs(np.real(h_surrogate[:, index_ecc_min]) - np.real(true_h)[length_diff:]) / abs(np.real(true_h)[length_diff:])
            # relative_error_hp[relative_error_hp > 1] = 0
            
            # # axs[1].plot(self.TS, relative_error_hp, linewidth=0.6)
            # # axs[1].set_ylabel(f'Rel. Error in $h_+$')
            # # axs[1].grid(True)
            # # # axs[1].set_ylim(0, 10)
            # # # axs[1].set_title('Relative error $h_x$')

            # # # axs[2].plot(self.TS, true_hc[length_diff:], linewidth=0.6, label='True waveform before')
            # # axs[2].plot(self.TS, np.imag(true_h)[length_diff:], linewidth=0.6, label='True waveform after')
            # # axs[2].plot(self.TS, np.imag(h_surrogate[:, index_ecc_min]), linewidth=0.6, label='Surrogate')
            # # axs[2].grid(True)
            # # axs[2].set_ylabel('$h_x$')
            # # axs[2].legend()

            # # axs[2].plot(self.TS, true_hc[length_diff:], linewidth=0.6, label='True waveform before')
            # axs[1].plot(self.TS, np.imag(true_h)[length_diff:], linewidth=0.6, label='True waveform after')
            # axs[1].plot(self.TS, np.imag(h_surrogate[:, index_ecc_min]), linewidth=0.6, label='Surrogate')
            # axs[1].grid(True)
            # axs[1].set_ylabel('$h_x$')
            # axs[1].legend()

            # # # Calculate and Plot cross polarisation error
            # # relative_error_hc = abs(np.imag(h_surrogate[:, index_ecc_min]) - np.imag(true_h)[length_diff:]) / abs(np.imag(true_h)[length_diff:])
            # # relative_error_hc[relative_error_hc > 1] = 0
            # # axs[3].plot(self.TS, relative_error_hc, linewidth=0.6)
            # # axs[3].set_ylabel(f'Rel. Error in $h_x$')
            # # axs[3].set_xlabel('t [M]')
            # # axs[3].grid(True)
            # # # axs[3].set_ylim(0, 10)


            # if save_fig_surr is True:
            #     figname = f'Surrogate_wf_eccmin={plot_surr_at_ecc}_for_ecc=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}]_q={self.mass_ratio}_fmin={self.freqmin}_iN={len(self.parameter_space_input)}_oN={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}.png'
                
            #     # Ensure the directory exists, creating it if necessary and save
            #     os.makedirs('Images/Surrogate_wf', exist_ok=True)
            #     fig_surrogate.savefig('Images/Surrogate_wf/' + figname)

            #     print('Figure is saved in Images/Surrogate_wf')

            fig_surrogate, axs = plt.subplots(4, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1, 3, 1], 'hspace': 0.2}, sharex=True)
            # fig_surrogate, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

            true_hp, true_hc, TS = self.simulate_inspiral_mass_independent(plot_surr_at_ecc)

            phase = np.array(waveform.utils.phase_from_polarizations(true_hp, true_hc))
            amp = np.array(waveform.utils.amplitude_from_polarizations(true_hp, true_hc))
            """ COULD BE PHASE SHIFT"""
            true_h = amp * np.exp(1j * phase)
            length_diff = len(true_hp) - self.waveform_size

            # axs[0].set_title(f'Eccentricity = {plot_surr_at_ecc}, ga={self.min_greedy_error_amp}, gp={self.min_greedy_error_phase}')
            # axs[0].plot(self.TS, true_hp[length_diff:], linewidth=0.6, label='True waveform before')
            axs[0].plot(self.TS, np.real(true_h)[length_diff:], linewidth=0.6, label=f'true waveform e = {plot_surr_at_ecc}')
            axs[0].plot(self.TS, np.real(h_surrogate[:, index_ecc_min]), linewidth=0.6, label=f'surrogate e = {plot_surr_at_ecc}')
            axs[0].set_ylabel('$h_+$')
            axs[0].grid(True)
            axs[0].legend()

            # Calculate and Plot plus polarisation error 
            relative_error_hp = abs(np.real(h_surrogate[:, index_ecc_min]) - np.real(true_h)[length_diff:]) / abs(np.real(true_h)[length_diff:])
            relative_error_hp[relative_error_hp > 1] = 0
            
            axs[1].plot(self.TS, abs(np.real(h_surrogate[:, index_ecc_min]) - np.real(true_h)[length_diff:]), linewidth=0.6)
            axs[1].set_ylabel('|$h_{+, S} - h_+$|')
            axs[1].grid(True)
            # axs[1].set_ylim(0, 10)
            # axs[1].set_title('Relative error $h_x$')

            # axs[2].plot(self.TS, true_hc[length_diff:], linewidth=0.6, label='True waveform before')
            axs[2].plot(self.TS, np.imag(true_h)[length_diff:], linewidth=0.6, label=f'true waveform e = {plot_surr_at_ecc}')
            axs[2].plot(self.TS, np.imag(h_surrogate[:, index_ecc_min]), linewidth=0.6, label=f'surrogate e = {plot_surr_at_ecc}')
            axs[2].grid(True)
            axs[2].set_ylabel('$h_x$')
            axs[2].legend()

            # # axs[2].plot(self.TS, true_hc[length_diff:], linewidth=0.6, label='True waveform before')
            # axs[1].plot(self.TS, np.imag(true_h)[length_diff:], linewidth=0.6, label='True waveform after')
            # axs[1].plot(self.TS, np.imag(h_surrogate[:, index_ecc_min]), linewidth=0.6, label='Surrogate')
            # axs[1].grid(True)
            # axs[1].set_ylabel('$h_x$')
            # axs[1].legend()

            # Calculate and Plot cross polarisation error
            relative_error_hc = abs(np.imag(h_surrogate[:, index_ecc_min]) - np.imag(true_h)[length_diff:]) / abs(np.imag(true_h)[length_diff:])
            relative_error_hc[relative_error_hc > 1] = 0
            axs[3].plot(self.TS, abs(np.imag(h_surrogate[:, index_ecc_min]) - np.imag(true_h)[length_diff:]), linewidth=0.6)
            axs[3].set_ylabel('|$h_{x, S} - h_x$|')
            axs[3].set_xlabel('t [M]')
            axs[3].grid(True)
            # axs[3].set_ylim(0, 10)

            # axs[4].plot(np.arange(len(relative_error_hc)), relative_error_hc, linewidth=0.6, label='rel err')
            # axs[5].plot(np.arange(len(relative_error_hc)), abs(np.imag(true_h)[length_diff:]), linewidth=0.6, label='abs')
            # axs[4].set_ylabel(f'Rel. Error in $h_x$')
            # axs[4].set_xlabel('t [M]')
            # axs[4].grid(True)
            # axs[4].legend()


            if save_fig_surr is True:
                figname = f'Surrogate_wf_eccmin={plot_surr_at_ecc}_for_ecc=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}]_q={self.mass_ratio}_fmin={self.freqmin}_iN={len(self.parameter_space_input)}_oN={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}_size={self.waveform_size}.png'
                
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Surrogate_wf', exist_ok=True)
                fig_surrogate.savefig('Images/Surrogate_wf/' + figname)

                print('Figure is saved in Images/Surrogate_wf')

        return h_surrogate, surrogate_amp, surrogate_phase, computation_time_amp, computation_time_phase

# gs = Generate_Surrogate(parameter_space=[0.01, 0.3], amount_input_wfs=60, amount_output_wfs=300, N_greedy_vecs_amp=50, N_greedy_vecs_phase=40, freqmin=650)
# gs.fit_to_training_set(property='phase', N_greedy_vecs=30, plot_fits=True, save_fig_fits=True, save_fits_to_file=True, plot_residuals_time_evolve=True, plot_residuals_ecc_evolve=True)
# gs.fit_to_training_set(property='amplitude', N_greedy_vecs=50, plot_fits=True, save_fig_fits=True, save_fits_to_file=True, plot_residuals_ecc_evolve=True, plot_kernels=True)
# gs.generate_surrogate_model(plot_surr_datapiece_at_ecc=0.18, plot_surr_at_ecc=0.17, plot_GPRfit=True)
# plt.show()
# print(gs.parameter_space_output)
# gs.generate_property_dataset(eccmin_list=ecc_list1, property='phase', plot_residuals=True, save_dataset_to_file='test1.npz')
# ecc_list2 = np.linspace(0.01, 0.3, num=500).round(5)
# gs.generate_property_dataset(eccmin_list=ecc_list2, property='phase', plot_residuals=True, save_dataset_to_file='test2.npz')

# gs.generate_property_dataset(eccmin_list=ecc_list, property='amplitude', plot_residuals=True)
# plt.show()

# fig__ = plt.figure
# plt.plot(ecc_list1, test1)
# plt.plot(ecc_list2, test2)


# gs.fit_to_training_set(property='phase', min_greedy_error=3e-4, plot_fits=True, save_fits_to_file='GPRfits_0.01_0.3')
# gs.fit_to_training_set(property='amplitude', min_greedy_error=1e-2, plot_fits=True, save_fits_to_file='GPRfits_0.01_0.2')


# fig1 = plt.figure()
# 'residual_phase_full_parameterspace_0.1_0.2.npz'

# gs.generate_surrogate_model(plot_surr_datapiece_at_ecc=0.1905, plot_surr_at_ecc=0.1905, plot_GPRfit=True)
# gs.generate_surrogate_model(plot_surr_datapiece_at_ecc=0.0336, plot_surr_at_ecc=0.0336, plot_GPRfit=True)

# plt.show()
