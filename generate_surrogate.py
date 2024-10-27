from generate_training_set import *

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Product, ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel, ExpSineSquared, ConstantKernel as C
from sklearn.model_selection import train_test_split
import faulthandler

faulthandler.enable()

class Generate_Surrogate(Generate_TrainingSet):

    def __init__(self, parameter_space, amount_input_wfs, amount_output_wfs,  min_greedy_error_amp, min_greedy_error_phase, waveform_size=None, total_mass=50, mass_ratio=1, freqmin=20):
        
        self.parameter_space_input = np.linspace(parameter_space[0], parameter_space[1], amount_input_wfs).round(4)
        self.parameter_space_output = np.linspace(parameter_space[0], parameter_space[1], amount_output_wfs).round(4)
        
        self.min_greedy_error_amp = min_greedy_error_amp
        self.min_greedy_error_phase = min_greedy_error_phase
        self.surrogate = None

        self.phase_shift_total_output = np.zeros(len(self.parameter_space_output))
        
        Generate_TrainingSet.__init__(self, parameter_space_input=self.parameter_space_input, waveform_size=waveform_size, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin)

    def fit_to_training_set(self, property, min_greedy_error, save_fits_to_file=False, plot_kernels=False, plot_fits=False, save_fig_kernels=False, save_fig_fits=False):

        def gaussian_process_regression(time_node, training_set, plot_kernels=plot_kernels, save_fig_kernels=save_fig_kernels):
            X = self.parameter_space_output[:, np.newaxis]

            X_train = np.array(self.parameter_space_input[self.greedy_parameters_idx]).reshape(-1, 1)
            y_train = np.squeeze(training_set.T[time_node])
            
            # Scale y_train
            scaler = StandardScaler()
            y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            # periodic_kernel = ExpSineSquared(length_scale=10.0, periodicity=1.0, length_scale_bounds=(1e-2, 1e3), periodicity_bounds=(1e-2, 1e1))
            # rbf_kernel = RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e3))
            # locally_periodic_kernel = C(1.0, (1e-4, 1e1)) * Product(periodic_kernel, rbf_kernel) + WhiteKernel(noise_level=1)
            
            kernels = [
                # C(1.0, (1e-4, 1e1)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
                # C(1.0, (1e-4, 1e1)) * Matern(length_scale=10.0, length_scale_bounds=(1e-4, 1e3), nu=2.5),
                C(1.0, (0.1, 10)) * Matern(length_scale=5.0, length_scale_bounds=(0.001, 0.3), nu=1.5)
                # C(1.0, (1e-4, 1e1)) * Product(periodic_kernel, rbf_kernel) + WhiteKernel(noise_level=1),
                # C(1.0, (1e-4, 1e1)) * RationalQuadratic(length_scale=1.0, alpha=0.1),
                # C(1.0, (1e-4, 1e1)) * ExpSineSquared(length_scale=1.0, periodicity=3.0, length_scale_bounds=(1e-2, 1e2), periodicity_bounds=(1e-2, 1e1)),
                # C(1.0, (1e-4, 1e1)) * ExpSineSquared(length_scale=1.0, periodicity=1.0) + WhiteKernel(noise_level=1),
                # C(1.0, (1e-4, 1e1)) * ExpSineSquared(length_scale=1.0, periodicity=1.0) + WhiteKernel(noise_level=1)
            ]
            """
            GPR parameters:
            - length_scale: A larger length_scale means the GP will consider data points further apart as similar, resulting in a smoother function.
            - nu: The nu parameter in the Matern kernel controls the smoothness of the kernel. Higher values of nu (e.g., 2.5) will produce a smoother function.


            """

            mean_prediction_per_kernel = []
            std_predictions_per_kernel = []

            for kernel in kernels:
         
                gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
                gaussian_process.fit(X_train, y_train_scaled)
                gaussian_process.kernel_
                # Print the optimized kernel with its hyperparameters
                print(f"Optimized kernel: {gaussian_process.kernel_}")

                mean_prediction_scaled, std_prediction_scaled = gaussian_process.predict(X, return_std=True)
                mean_prediction = scaler.inverse_transform(mean_prediction_scaled.reshape(-1, 1)).flatten()
                std_prediction = std_prediction_scaled * scaler.scale_[0]

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
                plt.legend(loc = 'upper right')
                plt.xlabel("$e$")
                if property == 'amplitude':
                    plt.ylabel("$f_A(e)$")
                elif property == 'phase':
                    plt.ylabel("$f_{\phi}(e)$")
                plt.title(f"GPR {property} at T_{time_node}")
                # plt.show()

                if save_fig_kernels is True:
                    figname = 'Gaussian kernels {property} M={}, q={}, ecc=[{},{}].png'.format(self.total_mass, self.mass_ratio, min(self.parameter_space_input), max(self.parameter_space_input))
                    
                    # Ensure the directory exists, creating it if necessary and save
                    os.makedirs('Images/Gaussian_kernels', exist_ok=True)
                    fig_residual_training_fit.savefig('Images/Gaussian_kernels/' + figname)

                    print('Figure is saved in Images/Gaussian_kernels')

            return mean_prediction, [(mean_prediction - 1.96 * std_prediction), (mean_prediction + 1.96 * std_prediction)]

        try:
            load_GPRfits = np.load(f'Straindata/GPRfits/{property}_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_{len(self.parameter_space_input)}wfsIN_{len(self.parameter_space_output)}wfsOUT.npz', allow_pickle=True)
            gaussian_fit = load_GPRfits['GPR_fit']
            training_set = load_GPRfits['training_set']
            uncertainty_region = load_GPRfits['uncertainty_region']
            self.greedy_parameters_idx = load_GPRfits['greedy_parameters']
            self.empirical_nodes_idx = load_GPRfits['empirical_nodes']
            self.residual_greedy_basis = load_GPRfits['residual_greedy_basis']
            
            print('GPRfit load succeeded')

        except:

            # Generate the training set of greedy parameters at empirical nodes
            training_set = self.get_training_set(property=property, min_greedy_error=min_greedy_error)

            # Create empty arrays to save fitvalues
            gaussian_fit = np.zeros((len(training_set.T), len(self.parameter_space_output)))
            uncertainty_region = []

            print(f'Interpolate {property}...')

            for node_i in range(len(self.empirical_nodes_idx)):
                
                mean_prediction, uncertainty_region = gaussian_process_regression(node_i, training_set, self.greedy_parameters_idx, plot_fits)
                
                gaussian_fit[node_i] = mean_prediction
                uncertainty_region.append(uncertainty_region)

        if plot_fits is True:
            try:
                load_parameterspace_input = np.load(f'Straindata/Residuals/residual_{property}_full_parameterspace_input_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}.npz')
                residual_parameterspace_input = load_parameterspace_input['residual']
                self.phase_shift_total_input = load_parameterspace_input['total_phase_shift']
                self.TS_M = load_parameterspace_input['TS_M']

                load_parameterspace_output = np.load(f'Straindata/Residuals/residual_{property}_full_parameterspace_output_{min(self.parameter_space_output)}_{max(self.parameter_space_output)}.npz')
                residual_parameterspace_output = load_parameterspace_output['residual']
                self.phase_shift_total_output = load_parameterspace_output['total_phase_shift']
            except:
                residual_parameterspace_output, self.phase_shift_total_output = self.generate_property_dataset(eccmin_list=self.parameter_space_output, property=property, save_dataset_to_file=f'residual_{property}_full_parameterspace_output_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_{len(self.parameter_space_output)}wfs.npz')

            # Define a distinct color palette
            color_palette = plt.cm.tab10.colors 

            # Number of distinct colors 
            num_colors = len(gaussian_fit)

            # Create a color map to use consistent colors for matching fits and data sets
            colors = [color_palette[i % len(color_palette)] for i in range(num_colors)]

            fig_residual_training_fit, axs = plt.subplots(2, 1, figsize=(11,6), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.4}, sharex=True)

            # Top left subplot for amplitude main plot
            for i in range(len(gaussian_fit)):
                # Use the same color for the fit and the corresponding empirical data
                axs[0].plot(self.parameter_space_output, gaussian_fit.T[:, i], color=colors[i], linewidth=0.6)
                axs[0].scatter(self.parameter_space_input[self.greedy_parameters_idx], training_set[:, i], color=colors[i])
                axs[0].plot(self.parameter_space_output, residual_parameterspace_output[:, self.empirical_nodes_idx[i]], linestyle='dashed', linewidth=0.6, color=colors[i], label=f't={self.TS_M[self.empirical_nodes_idx[i]]}')
                # axs[0].plot(self.parameter_space_output, residual_parameterspace_output[:, -1], linestyle='dashed', linewidth=0.6, color=colors[i], label=f't={self.TS_M[self.empirical_nodes_idx[i]]}')
                # axs[0].plot(self.parameter_space_input, residual_parameterspace_input[:, -1])
                relative_error = abs(residual_parameterspace_output[:, self.empirical_nodes_idx[i]] - gaussian_fit.T[:, i]) / abs(residual_parameterspace_output[:, self.empirical_nodes_idx[i]])

                # print(f'rel error {property}',(residual_parameterspace_input[:, self.empirical_nodes_idx[i]] - gaussian_fit.T[:, i])[35:45], abs(residual_parameterspace_input[:, self.empirical_nodes_idx[i]])[35:45], relative_error[35:45])
                axs[1].plot(self.parameter_space_output, relative_error, color=colors[i], label=f'Error {i+1} (t={int(self.TS_M[self.empirical_nodes_idx[i]])})')
                axs[1].scatter(self.parameter_space_output[30], relative_error[30], s=30)
                axs[1].scatter(self.parameter_space_output[45], relative_error[45], s=30)
            # Adjust legend: smaller font, inside figure, upper left
            axs[0].legend(loc='upper left', ncol=3, bbox_to_anchor=(0.01, 0.99), fontsize='small')
            if property == 'phase':
                axs[0].set_ylabel('$\Delta \phi$ ')
                axs[0].set_title('GPRfit $\phi$')
            elif property == 'amplitude':
                axs[0].set_ylabel('$\Delta$ A')
                axs[0].set_title('GPRfit A')
            axs[0].grid()

            # Adjust legend: smaller font, inside figure, upper right
            axs[1].set_xlabel('eccentricity')
            axs[1].set_ylabel('Relative fit error')
            # axs[1].set_xlim(0., 0.105)
            # axs[1].set_ylim(-100, 100)
            axs[1].grid()

            # Adjust layout to prevent overlap
            plt.tight_layout()

            if save_fig_fits is True:
                figname = 'Gaussian fits {property} M={}, q={}, ecc=[{},{}].png'.format(self.total_mass, self.mass_ratio, min(self.parameter_space_input), max(self.parameter_space_input))
                
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Gaussian_fits', exist_ok=True)
                fig_residual_training_fit.savefig('Images/Gaussian_fits/' + figname)

                print('Figure is saved in Images/Gaussian_fits')
        
        if save_fits_to_file is not None and not os.path.isfile(f'Straindata/GPRfits/{save_fits_to_file}'):

            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Straindata/GPRfits', exist_ok=True)
            np.savez(f'Straindata/GPRfits/{property}_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_{len(self.parameter_space_input)}wfsIN_{len(self.parameter_space_output)}wfsOUT.npz', GPR_fit=gaussian_fit, training_set=training_set, uncertainty_region=np.array(uncertainty_region, dtype=object), greedy_parameters=self.greedy_parameters_idx, empirical_nodes=self.empirical_nodes_idx, residual_greedy_basis=self.residual_greedy_basis)
            print('GPR fits saved')

        return gaussian_fit, uncertainty_region

    def generate_surrogate_model(self, plot_surr_datapiece_at_ecc=False, save_fig_datapiece=False, plot_surr_at_ecc=False, save_fig_surr=False, plot_GPRfit=False):

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
                circ -= circ[-self.waveform_size]

            elif property == 'amplitude':
                circ = np.array(waveform.utils.amplitude_from_polarizations(self.hp_TS_circ, self.hc_TS_circ))
            else:
                print('Choose property = "phase", "amplitude"')
                sys.exit(1)
            

            length_diff = len(circ) - self.waveform_size
            original_dataset = np.zeros((residual_dataset.shape[0], residual_dataset.shape[1]))

            for i in range(residual_dataset.shape[1]):
                if property == 'phase':
                    original_dataset[:, i] = circ[length_diff:] - residual_dataset[:, i]
                elif property == 'amplitude':
                    original_dataset[:, i] = residual_dataset[:, i] + circ[length_diff:]
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

            m, _ = B_matrix.shape

            reconstructed_residual = np.zeros((len(self.TS_M), len(self.parameter_space_output)))

            for i in range(m):
                reconstructed_residual += np.dot(B_matrix[i, :].reshape(-1, 1), fit_matrix[i, :].reshape(1, -1))


            # Change back from residual to original (+ circulair)
            surrogate_datapiece = residual_to_original(residual_dataset=reconstructed_residual, property=property)
            
            if property == 'phase':
                # extend phase shifts for longer parameterspace by using interpolation method
                old_indices = np.linspace(0, len(self.phase_shift_total_input) - 1, len(self.phase_shift_total_input))
                new_indices = np.linspace(0, len(self.phase_shift_total_output) - 1, len(self.parameter_space_output)) 
                self.phase_shift_total_output = np.interp(new_indices, old_indices, self.phase_shift_total_input)

                surrogate_datapiece = surrogate_datapiece + self.phase_shift_total_output
                print(self.phase_shift_total_input, self.phase_shift_total_output)

            if (plot_surr_datapiece_at_ecc is not False) and (not isinstance(plot_surr_datapiece_at_ecc, float)):
                print('plot_surr_datapiece_at_ecc must be float value!')

            if isinstance(plot_surr_datapiece_at_ecc, float):
                # Create a 2x1 subplot grid with height ratios 3:1
                fig_surrogate_datapieces, axs = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.4})

                # Simulate the real waveform datapiece
                real_hp, real_hc, real_TS = self.simulate_inspiral_mass_independent(plot_surr_datapiece_at_ecc)
                length_diff = len(real_TS) - self.waveform_size

                if property == 'amplitude':
                    real_datapiece = np.array(waveform.utils.amplitude_from_polarizations(real_hp, real_hc))[length_diff:]
                    units = ''
                elif property == 'phase':
                    real_datapiece = np.array(waveform.utils.phase_from_polarizations(real_hp, real_hc))
                    real_datapiece = real_datapiece[length_diff:]

                    surrogate_datapiece += real_datapiece[-self.waveform_size]

                    units = ' [radians]'

                # Ensure the time series lengths match by accounting for differences

                try:
                    index_ecc_min = np.where(self.parameter_space_output == plot_surr_datapiece_at_ecc)[0][0]
                except:
                    print(f'Eccentricity value {plot_surr_datapiece_at_ecc} not in parameterspace')
                    sys.exit(1)

                # true_phase = np.load(f'/home/suzannelexmond/anaconda3/envs/igwn-py39/Python_scripts/Thesis_Eccentric_BBHs/Classes/Straindata/Residuals/residual_{property}_full_parameterspace_output_0.01_0.3_500wfs.npz')['residual']
                # self.circulair_wf
                # phase0 = np.array(waveform.utils.phase_from_polarizations(self.hp_TS_circ, self.hc_TS_circ))[-self.waveform_size:]
                # true_phase += phase0
                # (10, surrogate_datapiece.shape, true_phase.shape)
                # Plot Surrogate and Real Amplitude (Top Left)
                # axs[0].plot(self.parameter_space_output, surrogate_datapiece[index_ecc_min], label='surr')
                axs[0].plot(self.TS_M, surrogate_datapiece.T[index_ecc_min], linewidth=0.6, label=f'Surrogate: e = {plot_surr_datapiece_at_ecc}')
                # axs[0].plot(self.TS_M, true_phase[index_ecc_min], linewidth=0.6, label=f'Surrogate: e = {plot_surr_datapiece_at_ecc}')
                axs[0].plot(self.TS_M, real_datapiece, linewidth=0.6, linestyle='dashed', label=f'Real: e = {plot_surr_datapiece_at_ecc}')
                # axs[0].plot(self.parameter_space_output, true_phase[:, index_ecc_min], label='real')
                axs[0].set_xlabel('t [M]')
                axs[0].set_ylabel(property + units)
                axs[0].grid(True)
                axs[0].set_title(f'Surrogate vs Real {property}')
                axs[0].legend(loc='upper left', ncol=2, fontsize='small')

                # Calculate and Plot Phase Error (Bottom Right)
                # Define a small threshold value to handle small or zero values in real_datapiece
                threshold = 1e-30  # You can adjust this value based on the scale of your data

                # Avoid division by very small numbers by using np.maximum to set a lower limit
                relative_error = abs(surrogate_datapiece.T[index_ecc_min] - real_datapiece) / abs(real_datapiece)
                # print(np.diff(relative_error), abs(real_datapiece))
                axs[1].plot(self.TS_M, relative_error, linewidth=0.6)
                axs[1].scatter(self.TS_M[100], relative_error[100], s=20)
                axs[1].set_ylabel(f'Rel. Error in {property}')
                axs[1].set_xlabel('t [M]')
                axs[1].grid(True)
                axs[1].set_title('Relative error')

                # Adjust layout to prevent overlap
                plt.tight_layout()

                if save_fig_datapiece is True:
                    figname = f'Surrogate_{property}_eccmin={plot_surr_datapiece_at_ecc}_for_ecc=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}].png'
                    
                    # Ensure the directory exists, creating it if necessary and save
                    os.makedirs('Images/Surrogate_datapieces', exist_ok=True)
                    fig_surrogate_datapieces.savefig('Images/Surrogate_datapieces/' + figname)

                    print('Figure is saved in Images/Surrogate_datapieces')
                    
            return surrogate_datapiece
        if self.surrogate is not None:
            h_surrogate = self.surrogate
        else:
            # Get matrix with interpolated fits and B_matrix
            fit_matrix_amp = self.fit_to_training_set(min_greedy_error=self.min_greedy_error_amp, property='amplitude', plot_fits=plot_GPRfit)[0]
            B_matrix_amp = compute_B_matrix()
            # Reconstruct amplitude datapiece
            surrogate_amp = reconstruct_surrogate_datapiece(property='amplitude', B_matrix=B_matrix_amp, fit_matrix=fit_matrix_amp, plot_surr_datapiece_at_ecc=plot_surr_datapiece_at_ecc)

            # Get matrix with interpolated fits and B_matrix
            fit_matrix_phase = self.fit_to_training_set(min_greedy_error=self.min_greedy_error_phase, property='phase', plot_fits=plot_GPRfit)[0]
            B_matrix_phase = compute_B_matrix()

            # Reconstruct phase datapiece
            surrogate_phase = reconstruct_surrogate_datapiece(property='phase', B_matrix=B_matrix_phase, fit_matrix=fit_matrix_phase, plot_surr_datapiece_at_ecc=plot_surr_datapiece_at_ecc)
            h_surrogate = np.zeros((len(self.TS_M), len(self.parameter_space_output)), dtype=complex)
            h_surrogate = surrogate_amp * np.exp(-1j * surrogate_phase)

            # self.surrogate = h_surrogate

        if (plot_surr_at_ecc is not False) and (not isinstance(plot_surr_at_ecc, float)):
            print('plot_surr_datapiece_at_ecc must be float value!')

        if isinstance(plot_surr_at_ecc, float):
            
            fig_surrogate, axs = plt.subplots(4, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1, 3, 1], 'hspace': 0.2}, sharex=True)

            try:
                index_ecc_min = np.where(self.parameter_space_output == plot_surr_datapiece_at_ecc)[0][0]
            except:
                print(f'Eccentricity value {plot_surr_datapiece_at_ecc} not in parameterspace!')
                sys.exit(1)

            true_hp, true_hc, TS_M = self.simulate_inspiral_mass_independent(plot_surr_datapiece_at_ecc)
            phase = np.array(waveform.utils.phase_from_polarizations(true_hp, true_hc))
            phase += self.phase_shift_total_output[index_ecc_min]
            amp = np.array(waveform.utils.amplitude_from_polarizations(true_hp, true_hc))

            true_h = amp * np.exp(-1j * phase)
            # print(true_hp, np.real(true_h))
            # print(true_hc, np.imag(true_h))

            length_diff = len(true_hp) - self.waveform_size
            axs[0].set_title(f'Eccentricity = {plot_surr_datapiece_at_ecc}')
            # axs[0].plot(self.TS_M, true_hp[length_diff:], linewidth=0.6, label='True waveform before')
            axs[0].plot(self.TS_M, np.real(true_h)[length_diff:], linewidth=0.6, label='True waveform after')
            axs[0].plot(self.TS_M, np.real(h_surrogate[:, index_ecc_min]), linewidth=0.6, label='Surrogate')
            axs[0].set_ylabel('$h_+$')
            axs[0].grid(True)
            axs[0].legend()

            # Calculate and Plot plus polarisation error 
            relative_error_hp = abs(np.real(h_surrogate[:, index_ecc_min]) - true_hp[length_diff:]) / true_hp[length_diff:]
            axs[1].plot(self.TS_M, relative_error_hp, linewidth=0.6)
            axs[1].set_ylabel(f'Rel. Error in $h_+$')
            axs[1].grid(True)
            # axs[1].set_title('Relative error $h_x$')

            # axs[2].plot(self.TS_M, true_hc[length_diff:], linewidth=0.6, label='True waveform before')
            axs[2].plot(self.TS_M, np.imag(true_h)[length_diff:], linewidth=0.6, label='True waveform after')
            axs[2].plot(self.TS_M, np.imag(h_surrogate[:, index_ecc_min]), linewidth=0.6, label='Surrogate')
            axs[2].grid(True)
            axs[2].legend()

            # Calculate and Plot cross polarisation error
            relative_error_hc = abs(np.imag(h_surrogate[:, index_ecc_min]) - true_hc[length_diff:]) / true_hc[length_diff:]
            axs[3].plot(self.TS_M, relative_error_hc, linewidth=0.6)
            axs[3].set_ylabel(f'Rel. Error in $h_+$')
            axs[3].set_xlabel('t [M]')
            axs[3].grid(True)
            # axs[3].set_title('Relative error $h_+$')


            if save_fig_surr is True:
                figname = f'Surrogate_wf_eccmin={plot_surr_at_ecc}_for_ecc=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}].png'
                
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Surrogate_wf', exist_ok=True)
                fig_surrogate.savefig('Images/Surrogate_wf/' + figname)

                print('Figure is saved in Images/Surrogate_wf')

        # surrogate_error = self.calc_surrogate_error(true_hp, index_ecc_min)

        return h_surrogate

# residual_input = np.load('/home/suzannelexmond/anaconda3/envs/igwn-py39/Python_scripts/Thesis_Eccentric_BBHs/Classes/Straindata/Residuals/residual_phase_full_parameterspace_input_0.01_0.2.npz')['residual']
# residual_output = np.load('/home/suzannelexmond/anaconda3/envs/igwn-py39/Python_scripts/Thesis_Eccentric_BBHs/Classes/Straindata/Residuals/residual_phase_full_parameterspace_output_0.01_0.2.npz')['residual']
# print(residual_input.shape, residual_output.shape)

# param_out = np.linspace(0.01, 0.2, num=500).round(4)
# param_in = np.linspace(0.01, 0.2, num=200).round(4)
# ecc = 0.1
# try:
#     index_ecc_min_i = np.where(param_out == ecc)[0][0]
# except:
#     print(f'Eccentricity value {ecc} not in parameterspace')
#     sys.exit(1)

# fig_c = plt.figure()
# plt.plot(np.linspace(0.01, 0.2, num=200).round(4), residual_input[:, 200])
# plt.plot(np.linspace(0.01, 0.2, num=500).round(4), residual_output[:, 500])
# plt.show()
test = Simulate_Inspiral(0.4)

gs = Generate_Surrogate(parameter_space=[0.01, 0.2], amount_input_wfs=50, amount_output_wfs=500, min_greedy_error_amp=1e-2, min_greedy_error_phase=1e-3, waveform_size=3500, freqmin=18)
# ecc_list = np.linspace(0.01, 0.2, num=20).round(5)

print(gs.parameter_space_output)
# gs.generate_property_dataset(eccmin_list=ecc_list1, property='phase', plot_residuals=True, save_dataset_to_file='test1.npz')
# ecc_list2 = np.linspace(0.01, 0.3, num=500).round(5)
# gs.generate_property_dataset(eccmin_list=ecc_list2, property='phase', plot_residuals=True, save_dataset_to_file='test2.npz')

# gs.generate_property_dataset(eccmin_list=ecc_list, property='amplitude', plot_residuals=True)
# plt.show()
# test1 = np.load('/home/suzannelexmond/anaconda3/envs/igwn-py39/Python_scripts/Thesis_Eccentric_BBHs/Classes/Straindata/Residuals/test1.npz')['total_phase_shift']
# test2= np.load('/home/suzannelexmond/anaconda3/envs/igwn-py39/Python_scripts/Thesis_Eccentric_BBHs/Classes/Straindata/Residuals/test2.npz')['total_phase_shift']

# fig__ = plt.figure
# plt.plot(ecc_list1, test1)
# plt.plot(ecc_list2, test2)


# gs.fit_to_training_set(property='phase', min_greedy_error=3e-4, plot_fits=True, save_fits_to_file='GPRfits_0.01_0.3')
# gs.fit_to_training_set(property='amplitude', min_greedy_error=1e-2, plot_fits=True, save_fits_to_file='GPRfits_0.01_0.2')


# fig1 = plt.figure()
# 'residual_phase_full_parameterspace_0.1_0.2.npz'

gs.generate_surrogate_model(plot_surr_datapiece_at_ecc=0.1905, plot_surr_at_ecc=0.1905, plot_GPRfit=True)
gs.generate_surrogate_model(plot_surr_datapiece_at_ecc=0.0336, plot_surr_at_ecc=0.0336, plot_GPRfit=True)

plt.show()