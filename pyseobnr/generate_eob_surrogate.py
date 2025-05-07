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

    def fit_to_training_set(self, property, min_greedy_error=None, N_greedy_vecs=None, save_fits_to_file=True, plot_kernels=False, plot_fits=False, save_fig_kernels=False, save_fig_fits=False):
        
        def gaussian_process_regression(time_node, training_set, optimized_kernel = None, plot_kernels=plot_kernels, save_fig_kernels=save_fig_kernels):
            X = self.parameter_space_output[:, np.newaxis]

            X_train = np.array(self.parameter_space_input[self.greedy_parameters_idx]).reshape(-1, 1)
            y_train = np.squeeze(training_set.T[time_node])
            
            # Scale y_train
            scaler = StandardScaler()
            y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

            kernels = [
                # Matern(length_scale=0.1, length_scale_bounds=(1e-3, 1), nu=1.5)
                Matern(length_scale=0.1, length_scale_bounds=(1e-1, 1), nu=1.5) # <= 0.3 eccentricity
            ]

            """
            GPR parameters:
            - length_scale: A larger length_scale means the GP will consider data points further apart as similar, resulting in a smoother function.
            - nu: The nu parameter in the Matern kernel controls the smoothness of the kernel. Higher values of nu (e.g., 2.5) will produce a smoother function.


            """

            mean_prediction_per_kernel = []
            std_predictions_per_kernel = []

            # region_mask = [[X_train < 0.2], [0.2 < X_train < 0.3]]
            # for mask in region_mask is True:
            #     kernel = 
            for kernel in kernels:

                if optimized_kernel is None:
                    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)

                else:
                    gaussian_process = GaussianProcessRegressor(kernel=optimized_kernel, optimizer=None)

                gaussian_process.fit(X_train, y_train_scaled)
                optimized_kernel = gaussian_process.kernel_

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
                plt.legend(loc = 'upper left')
                plt.xlabel("$e$")
                if property == 'amplitude':
                    plt.ylabel("$f_A(e)$")
                elif property == 'phase':
                    plt.ylabel("$f_{\phi}(e)$")
                plt.title(f"GPR {property} at T_{time_node}")
                # plt.show()

                if save_fig_kernels is True:
                    figname = f'Gaussian_kernels_{property}_ecc=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}]_q={self.mass_ratio}_fmin={self.freqmin}_iN={len(self.parameter_space_input)}_oN={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}.png'
                    
                    # Ensure the directory exists, creating it if necessary and save
                    os.makedirs('Images/Gaussian_kernels', exist_ok=True)
                    fig_residual_training_fit.savefig('Images/Gaussian_kernels/' + figname)

                    print('Figure is saved in Images/Gaussian_kernels')

            return mean_prediction, [(mean_prediction - 1.96 * std_prediction), (mean_prediction + 1.96 * std_prediction)], optimized_kernel

        try:

            load_GPRfits = np.load(f'Straindata/GPRfits/{property}_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_Ni={len(self.parameter_space_input)}_No={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}.npz', allow_pickle=True)
            
            gaussian_fit = load_GPRfits['GPR_fit']
            training_set = load_GPRfits['training_set']
            uncertainty_region = load_GPRfits['uncertainty_region']
            self.greedy_parameters_idx = load_GPRfits['greedy_parameters']
            self.empirical_nodes_idx = load_GPRfits['empirical_nodes']
            self.residual_greedy_basis = load_GPRfits['residual_greedy_basis']
            
            print('GPRfit load succeeded')

        except:

            # Generate the training set of greedy parameters at empirical nodes
            training_set = self.get_training_set(property=property, min_greedy_error=min_greedy_error, N_greedy_vecs=N_greedy_vecs)

            # Create empty arrays to save fitvalues
            gaussian_fit = np.zeros((len(training_set.T), len(self.parameter_space_output)))
            uncertainty_region = []

            print(f'Interpolate {property}...')

            # start1 = time.time()
            # mean_prediction, uncertainty_region = gaussian_process_regression_all(training_set, self.greedy_parameters_idx, plot_fits)
            # end1 = time.time()
            # print(f'time1 = {end1 - start1}')
            start2 = time.time()
            optimized_kernel = None
            for node_i in range(len(self.empirical_nodes_idx)):
                
                mean_prediction, uncertainty_region, optimized_kernel = gaussian_process_regression(node_i, training_set, optimized_kernel, plot_kernels)
                
                gaussian_fit[node_i] = mean_prediction
                uncertainty_region.append(uncertainty_region)
            end2 = time.time()
            print(f'time full GPR = {end2 - start2}')

        if plot_fits is True:
            load_parameterspace_input = np.load(f'Straindata/Residuals/residuals_{property}_e=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_N={len(self.parameter_space_input)}].npz')
            residual_parameterspace_input = load_parameterspace_input['residual']

            self.TS = load_parameterspace_input['TS']
            try:
                # load_parameterspace_input = np.load(f'Straindata/Residuals/residual_{property}_full_parameterspace_input_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}.npz')
                # residual_parameterspace_input = load_parameterspace_input['residual']
                # self.TS = load_parameterspace_input['TS']

                # load_parameterspace_output = np.load(f'Straindata/Residuals/residuals_{property}_e=[{min(self.parameter_space_output)}_{max(self.parameter_space_output)}]_N={len(self.parameter_space_output)}.npz')
                # residual_parameterspace_output = load_parameterspace_output['residual']

                true_h = np.load(f'true_h_PS_500wfs.npz')
                residual_parameterspace_output = np.zeros(len(true_h, self.waveform_size))

                for i in range(len(true_h)):
                    true_hp, true_hc = np.real(true_h[i]), np.imag(true_h[i])
                    phase = np.array(waveform.utils.phase_from_polarizations(true_hp, true_hc))
                    # amp = np.array(waveform.utils.amplitude_from_polarizations(true_hp, true_hc))
                    # true_h2 = amp * np.exp(1j * phase)
                    # print('c', true_hp, np.real(true_h2))
                    residual_parameterspace_output[i] = phase

            except:
                residual_parameterspace_output = self.generate_property_dataset(eccmin_list=self.parameter_space_output, property=property, save_dataset_to_file=True)

            # Define a distinct color palette
            color_palette = plt.cm.tab10.colors 

            # Number of distinct colors 
            num_colors = len(gaussian_fit)

            # Create a color map to use consistent colors for matching fits and data sets
            colors = [color_palette[i % len(color_palette)] for i in range(num_colors)]

            fig_residual_training_fit, axs = plt.subplots(2, 1, figsize=(11,6), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.4}, sharex=True)

            # Top left subplot for amplitude main plot
            for i in range(len(gaussian_fit[:3])):
                # Use the same color for the fit and the corresponding empirical data
                axs[0].plot(self.parameter_space_output, gaussian_fit.T[:, i], color=colors[i], linewidth=0.6)
                axs[0].scatter(self.parameter_space_input[self.greedy_parameters_idx], training_set[:, i], color=colors[i])
                axs[0].plot(self.parameter_space_output, residual_parameterspace_output[:, self.empirical_nodes_idx[i]], linestyle='dashed', linewidth=0.6, color=colors[i], label=f't={self.TS[self.empirical_nodes_idx[i]]}')
            
                # axs[0].plot(self.parameter_space_output, residual_parameterspace_output[:, -1], linestyle='dashed', linewidth=0.6, color=colors[i], label=f't={self.TS[self.empirical_nodes_idx[i]]}')
                # axs[0].plot(self.parameter_space_input, residual_parameterspace_input[:, -1])
                relative_error = abs(residual_parameterspace_output[:, self.empirical_nodes_idx[i]] - gaussian_fit.T[:, i]) / abs(residual_parameterspace_output[:, self.empirical_nodes_idx[i]])
                axs[1].plot(self.parameter_space_output, relative_error, color=colors[i], linewidth=0.6, label=f'Error {i+1} (t={int(self.TS[self.empirical_nodes_idx[i]])})')
                axs[1].set_ylim(0, 2)
            # Adjust legend: smaller font, inside figure, upper left
            axs[0].legend(loc='upper left', ncol=3, bbox_to_anchor=(0.01, 0.99), fontsize='small')
            if property == 'phase':
                axs[0].set_ylabel('$\Delta \phi$ ')
                axs[0].set_title(f'GPRfit $\phi$; greedy error = {min_greedy_error}')
            elif property == 'amplitude':
                axs[0].set_ylabel('$\Delta$ A')
                axs[0].set_title(f'GPRfit A; greedy error = {min_greedy_error}')
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
                figname = f'GPR_fits_{property}_ecc=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}]_q={self.mass_ratio}_fmin={self.freqmin}_iN={len(self.parameter_space_input)}_oN={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}.png'
                
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Gaussian_fits', exist_ok=True)
                fig_residual_training_fit.savefig('Images/Gaussian_fits/' + figname)

                print('Figure is saved in Images/Gaussian_fits')
        
        if save_fits_to_file is True and not os.path.isfile(f'Straindata/GPRfits/{property}_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_Ni={len(self.parameter_space_input)}_No={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}.npz'):

            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Straindata/GPRfits', exist_ok=True)
            np.savez(f'Straindata/GPRfits/{property}_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_Ni={len(self.parameter_space_input)}_No={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}.npz', GPR_fit=gaussian_fit, training_set=training_set, uncertainty_region=np.array(uncertainty_region, dtype=object), greedy_parameters=self.greedy_parameters_idx, empirical_nodes=self.empirical_nodes_idx, residual_greedy_basis=self.residual_greedy_basis)
            print('GPR fits saved in Straindata/GPRfits')

        return gaussian_fit, uncertainty_region

    def generate_surrogate_model(self, plot_surr_datapiece_at_ecc=False, save_fig_datapiece=False, plot_surr_at_ecc=False, save_fig_surr=False, plot_GPRfit=False, save_fits_to_file=False, save_fig_fits=False, save_surr_to_file=False):

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
            try:
                load_surrogate = np.load(f'Straindata/Surrogate_datapieces/Surrogate_datapieces_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_Ni={len(self.parameter_space_input)}_No={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}.npz')
                surrogate_amp = load_surrogate['surrogate_amp']
                surrogate_phase = load_surrogate['surrogate_phase']
                generation_time = load_surrogate['generation_time']
                self.TS = load_surrogate['TS']

                if property == 'phase':
                    surrogate_datapiece = surrogate_phase
                elif property == 'amplitude':
                    surrogate_datapiece = surrogate_amp

                print('Surrogate loaded')
            except Exception as e:
                print(e)

                m, _ = B_matrix.shape

                reconstructed_residual = np.zeros((len(self.TS), len(self.parameter_space_output)))

                for i in range(m):
                    reconstructed_residual += np.dot(B_matrix[i, :].reshape(-1, 1), fit_matrix[i, :].reshape(1, -1))


                # Change back from residual to original (+ circulair)
                surrogate_datapiece = residual_to_original(residual_dataset=reconstructed_residual, property=property)

            if (plot_surr_datapiece_at_ecc is not False) and (not isinstance(plot_surr_datapiece_at_ecc, float)):
                print('plot_surr_datapiece_at_ecc must be float value!')

            if isinstance(plot_surr_datapiece_at_ecc, float):
                # Create a 2x1 subplot grid with height ratios 3:1
                fig_surrogate_datapieces, axs = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.4})

                # Simulate the real waveform datapiece
                real_hp, real_hc, real_TS = self.simulate_inspiral_mass_independent(plot_surr_datapiece_at_ecc)
                minimum_length = min(len(real_TS), self.waveform_size)

                if property == 'amplitude':
                    real_datapiece = np.array(waveform.utils.amplitude_from_polarizations(real_hp, real_hc))[-minimum_length:]
                    units = ''
                elif property == 'phase':
                    real_datapiece = np.array(waveform.utils.phase_from_polarizations(real_hp, real_hc))[-minimum_length:]
                    units = ' [radians]'

                try:
                    index_ecc_min = np.where(self.parameter_space_output == plot_surr_datapiece_at_ecc)[0][0]
                except:
                    print(f'Eccentricity value {plot_surr_datapiece_at_ecc} not in parameterspace')
                    sys.exit(1)

                # Plot Surrogate and Real Amplitude (Top Left)
                # axs[0].plot(self.parameter_space_output, surrogate_datapiece[index_ecc_min], label='surr')
                axs[0].plot(self.TS, surrogate_datapiece.T[index_ecc_min][-minimum_length:], linewidth=0.6, label=f'Surrogate: e = {plot_surr_datapiece_at_ecc}')
                # axs[0].plot(self.TS, true_phase[index_ecc_min], linewidth=0.6, label=f'Surrogate: e = {plot_surr_datapiece_at_ecc}')
                axs[0].plot(self.TS, real_datapiece, linewidth=0.6, linestyle='dashed', label=f'Real: e = {plot_surr_datapiece_at_ecc}')
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
                axs[1].plot(self.TS, relative_error, linewidth=0.6)
                axs[1].set_ylabel(f'Rel. Error in {property}')
                axs[1].set_xlabel('t [M]')
                axs[1].grid(True)
                axs[1].set_title('Relative error')

                # Adjust layout to prevent overlap
                plt.tight_layout()

                if save_fig_datapiece is True:
                    figname = f'Surrogate_{property}_eccmin={plot_surr_datapiece_at_ecc}_for_ecc=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}]_q={self.mass_ratio}_fmin={self.freqmin}_iN={len(self.parameter_space_input)}_oN={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}.png'
                    
                    # Ensure the directory exists, creating it if necessary and save
                    os.makedirs('Images/Surrogate_datapieces', exist_ok=True)
                    fig_surrogate_datapieces.savefig('Images/Surrogate_datapieces/' + figname)

                    print('Figure is saved in Images/Surrogate_datapieces')
                    
            return surrogate_datapiece

        # Set timer for computational time of the surrogate model
        start_time = time.time()
        # Get matrix with interpolated fits and B_matrix
        fit_matrix_amp = self.fit_to_training_set(min_greedy_error=self.min_greedy_error_amp, N_greedy_vecs=self.N_greedy_vecs_amp, property='amplitude', plot_fits=plot_GPRfit, save_fits_to_file=save_fits_to_file)[0]
        B_matrix_amp = compute_B_matrix()
        # Reconstruct amplitude datapiece
        surrogate_amp = reconstruct_surrogate_datapiece(property='amplitude', B_matrix=B_matrix_amp, fit_matrix=fit_matrix_amp, plot_surr_datapiece_at_ecc=plot_surr_datapiece_at_ecc)
        self.surrogate_amp = surrogate_amp
        # Get matrix with interpolated fits and B_matrix
        fit_matrix_phase = self.fit_to_training_set(min_greedy_error=self.min_greedy_error_phase, N_greedy_vecs=self.N_greedy_vecs_phase, property='phase', plot_fits=plot_GPRfit, save_fig_fits=save_fig_fits, save_fits_to_file=save_fits_to_file)[0]
        B_matrix_phase = compute_B_matrix()

        # Reconstruct phase datapiece
        surrogate_phase = reconstruct_surrogate_datapiece(property='phase', B_matrix=B_matrix_phase, fit_matrix=fit_matrix_phase, plot_surr_datapiece_at_ecc=plot_surr_datapiece_at_ecc)
        self.surrogate_phase = surrogate_phase
        # End timer for computation of surrogate model
        end_time = time.time()
        generation_time = end_time - start_time

        if save_surr_to_file is True and not os.path.isfile(f'Straindata/Surrogate_datapieces/Surrogate_datapieces_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_Ni={len(self.parameter_space_input)}_No={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}.npz'):
            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Straindata/Surrogate_datapieces', exist_ok=True)
            np.savez(f'Straindata/Surrogate_datapieces/Surrogate_datapieces_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_Ni={len(self.parameter_space_input)}_No={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}.npz', surrogate_amp=surrogate_amp, surrogate_phase=surrogate_phase, generation_time=generation_time, TS=self.TS)
            print('GPR fits saved in Straindata/Surrogate_datapieces')


        h_surrogate = np.zeros((len(self.TS), len(self.parameter_space_output)), dtype=complex)
        h_surrogate = surrogate_amp * np.exp(1j * surrogate_phase)


        if plot_surr_at_ecc is not False and (not isinstance(plot_surr_at_ecc, float)):
            print('plot_surr_at_ecc must be float value!')

        if isinstance(plot_surr_at_ecc, float):
            
            fig_surrogate, axs = plt.subplots(4, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1, 3, 1], 'hspace': 0.2}, sharex=True)
            fig_surrogate, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

            try:
                index_ecc_min = np.where(self.parameter_space_output == plot_surr_at_ecc)[0][0]
            except:
                print(f'Eccentricity value {plot_surr_at_ecc} not in parameterspace!')
                sys.exit(1)

            true_hp, true_hc, TS_M = self.simulate_inspiral_mass_independent(plot_surr_at_ecc)

            phase = np.array(waveform.utils.phase_from_polarizations(true_hp, true_hc))
            amp = np.array(waveform.utils.amplitude_from_polarizations(true_hp, true_hc))

            true_h = amp * np.exp(1j * phase)
            length_diff = len(true_hp) - self.waveform_size

            axs[0].set_title(f'Eccentricity = {plot_surr_at_ecc}')
            # axs[0].plot(self.TS, true_hp[length_diff:], linewidth=0.6, label='True waveform before')
            axs[0].plot(self.TS, np.real(true_h)[length_diff:], linewidth=0.6, label='True waveform after')
            axs[0].plot(self.TS, np.real(h_surrogate[:, index_ecc_min]), linewidth=0.6, label='Surrogate')
            axs[0].set_ylabel('$h_+$')
            axs[0].grid(True)
            axs[0].legend()

            # Calculate and Plot plus polarisation error 
            relative_error_hp = abs(np.real(h_surrogate[:, index_ecc_min]) - np.real(true_h)[length_diff:]) / abs(np.real(true_h)[length_diff:])
            relative_error_hp[relative_error_hp > 1] = 0
            
            # axs[1].plot(self.TS, relative_error_hp, linewidth=0.6)
            # axs[1].set_ylabel(f'Rel. Error in $h_+$')
            # axs[1].grid(True)
            # # axs[1].set_ylim(0, 10)
            # # axs[1].set_title('Relative error $h_x$')

            # # axs[2].plot(self.TS, true_hc[length_diff:], linewidth=0.6, label='True waveform before')
            # axs[2].plot(self.TS, np.imag(true_h)[length_diff:], linewidth=0.6, label='True waveform after')
            # axs[2].plot(self.TS, np.imag(h_surrogate[:, index_ecc_min]), linewidth=0.6, label='Surrogate')
            # axs[2].grid(True)
            # axs[2].set_ylabel('$h_x$')
            # axs[2].legend()

            # axs[2].plot(self.TS, true_hc[length_diff:], linewidth=0.6, label='True waveform before')
            axs[1].plot(self.TS, np.imag(true_h)[length_diff:], linewidth=0.6, label='True waveform after')
            axs[1].plot(self.TS, np.imag(h_surrogate[:, index_ecc_min]), linewidth=0.6, label='Surrogate')
            axs[1].grid(True)
            axs[1].set_ylabel('$h_x$')
            axs[1].legend()

            # # Calculate and Plot cross polarisation error
            # relative_error_hc = abs(np.imag(h_surrogate[:, index_ecc_min]) - np.imag(true_h)[length_diff:]) / abs(np.imag(true_h)[length_diff:])
            # relative_error_hc[relative_error_hc > 1] = 0
            # axs[3].plot, relative_error_hc, linewidth=0.6)
            # axs[3].set_ylabel(f'Rel. Error in $h_x$')
            # axs[3].set_xlabel('t [M]')
            # axs[3].grid(True)
            # # axs[3].set_ylim(0, 10)


            if save_fig_surr is True:
                figname = f'Surrogate_wf_eccmin={plot_surr_at_ecc}_for_ecc=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}]_q={self.mass_ratio}_fmin={self.freqmin}_iN={len(self.parameter_space_input)}_oN={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}.png'
                
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Surrogate_wf', exist_ok=True)
                fig_surrogate.savefig('Images/Surrogate_wf/' + figname)

                print('Figure is saved in Images/Surrogate_wf')

        return h_surrogate, surrogate_amp, surrogate_phase, generation_time

gs = Generate_Surrogate(parameter_space=[0.01, 0.3], amount_input_wfs=40, N_greedy_vecs_amp=20, N_greedy_vecs_phase=20, amount_output_wfs=100)
print(1)
gs.get_training_set(property='phase', N_greedy_vecs=20)
print(2)
gs.fit_to_training_set(property='phase', N_greedy_vecs=20)
plt.show()