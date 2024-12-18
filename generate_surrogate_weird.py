from generate_training_set import *

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Product, ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel, ExpSineSquared, ConstantKernel as C
from sklearn.model_selection import train_test_split
import faulthandler
from pycbc.filter import match
from pycbc.psd import aLIGOZeroDetHighPower
import pycbc.psd
import time
import torch
import gpytorch
from gpytorch.constraints import Interval

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define custom Matern kernel with GPyTorch and constraints
class CustomMaternKernel(gpytorch.kernels.MaternKernel):
    def __init__(self):
        super().__init__(nu=1.5)  # Set nu=1.5 for smoothness
        # Set initial lengthscale and bounds similar to sklearn's kernel
        self.lengthscale = torch.nn.Parameter(torch.tensor(5.0).float().to(device))
        self.lengthscale_constraint = Interval(0.0001, 0.3)

# Define GP model with a custom kernel and likelihood
class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # Use ScaleKernel with our custom Matern kernel, similar to scikit-learn’s kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(CustomMaternKernel()).to(device)
        self.covar_module.outputscale = torch.tensor(1.0).to(device)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class Generate_Surrogate(Generate_TrainingSet):

    def __init__(self, parameter_space, amount_input_wfs, amount_output_wfs, N_greedy_vecs_amp=None, N_greedy_vecs_phase=None, min_greedy_error_amp=None, min_greedy_error_phase=None, waveform_size=None, total_mass=50, mass_ratio=1, freqmin=20):
        
        self.parameter_space_input = np.linspace(parameter_space[0], parameter_space[1], amount_input_wfs).round(4)
        self.parameter_space_output = np.linspace(parameter_space[0], parameter_space[1], amount_output_wfs).round(4)
        
        self.min_greedy_error_amp = min_greedy_error_amp
        self.min_greedy_error_phase = min_greedy_error_phase
        self.N_greedy_vecs_amp = N_greedy_vecs_amp
        self.N_greedy_vecs_phase = N_greedy_vecs_phase
        self.surrogate = None
        self.surrogate_amp = None
        self.surrogate_phase = None

        self.phase_shift_total_output = np.zeros(len(self.parameter_space_output))
        
        Generate_TrainingSet.__init__(self, parameter_space_input=self.parameter_space_input, waveform_size=waveform_size, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin)

    def fit_to_training_set(self, property, min_greedy_error=None, N_greedy_vecs=None, save_fits_to_file=False, plot_kernels=False, plot_fits=False, save_fig_kernels=False, save_fig_fits=False):

        def gaussian_process_regression(time_node, training_set, property, plot_kernels=plot_kernels, save_fig_kernels=save_fig_kernels):
            X = self.parameter_space_output[:, np.newaxis]

            X_train = np.array(self.parameter_space_input[self.greedy_parameters_idx]).reshape(-1, 1)
            y_train = np.squeeze(training_set.T[time_node])
            
            # Scale y_train
            scaler = StandardScaler()
            y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
           
            kernels = [
                C(1.0, (0.1, 20)) * Matern(length_scale=5.0, length_scale_bounds=(0.0001, 0.3), nu=1.5)
            ]
            """
            GPR parameters:
            - length_scale: A larger length_scale means the GP will consider data points further apart as similar, resulting in a smoother function.
            - nu: The nu parameter in the Matern kernel controls the smoothness of the kernel. Higher values of nu (e.g., 2.5) will produce a smoother function.


            """

            mean_prediction_per_kernel = []
            std_predictions_per_kernel = []

            for kernel in kernels:
                start_timer = time.time()
                gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
                gaussian_process.fit(X_train, y_train_scaled)
                gaussian_process.kernel_
                # Print the optimized kernel with its hyperparameters
                end_timer = time.time()
                print(f"Optimized kernel: {gaussian_process.kernel_} | time: {end_timer - start_timer}")

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

       # Define Gaussian Process Regression with GPU support
        # def gaussian_process_regression(time_node, training_set, property, plot_kernels=True, save_fig_kernels=save_fig_kernels):
        #     # Convert training data to torch tensors on GPU
        #     X_train = torch.tensor(self.parameter_space_input[self.greedy_parameters_idx]).reshape(-1, 1).float().to(device)
        #     y_train = torch.tensor(np.squeeze(training_set.T[time_node])).float().to(device)
        #     X = torch.tensor(self.parameter_space_output[:, np.newaxis]).float().to(device)

        #     # Standardize y_train for stable training
        #     scaler = StandardScaler()
        #     y_train_scaled = scaler.fit_transform(y_train.cpu().numpy().reshape(-1, 1)).flatten()
        #     y_train_scaled = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)

        #     # Set up model and likelihood on GPU
        #     likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        #     model = GPRegressionModel(X_train, y_train_scaled, likelihood).to(device)

        #     # Training function for the GPyTorch model
        #     def train_model(model, likelihood, X_train, y_train, training_iterations=50):
        #         model.train()
        #         likelihood.train()
        #         optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        #         mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        #         for i in range(training_iterations):
        #             optimizer.zero_grad()
        #             output = model(X_train)
        #             loss = -mll(output, y_train)
        #             loss.backward()
        #             optimizer.step()
        #             if i % 10 == 0:
        #                 print(f"Iteration {i + 1}/{training_iterations}, Loss: {loss.item()}")

        #     # Train the model
        #     train_model(model, likelihood, X_train, y_train_scaled)

        #     # Switch to evaluation mode
        #     model.eval()
        #     likelihood.eval()

        #     # Perform prediction and inverse transform
        #     with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #         posterior_distribution = likelihood(model(X))
        #         mean_posterior = scaler.inverse_transform(posterior_distribution.mean.cpu().numpy().reshape(-1, 1)).flatten()
        #         std_posterior = np.sqrt(posterior_distribution.variance.cpu().numpy()) * scaler.scale_[0]

        #     # Plot posterior predictions
        #     if plot_kernels:
        #         plt.figure(figsize=(10, 6))
        #         plt.plot(X.cpu().numpy(), mean_posterior, 'b', label='Posterior Mean')
        #         plt.fill_between(X.cpu().numpy().flatten(),
        #                         mean_posterior - 1.96 * std_posterior,
        #                         mean_posterior + 1.96 * std_posterior,
        #                         color='blue', alpha=0.2, label='Posterior 95% CI')
        #         plt.scatter(X_train.cpu().numpy(), scaler.inverse_transform(y_train_scaled.cpu().numpy().reshape(-1, 1)), color='red', label='Training Data')
        #         plt.title(f"Posterior Distribution - Gaussian Process Regression at Time Node {time_node}")
        #         plt.xlabel("$e$")
        #         plt.ylabel(f"$f_{{{property}}}(e)$")
        #         plt.legend()
        #         plt.show()

        #     return mean_posterior, [(mean_posterior - 1.96 * std_posterior), (mean_posterior + 1.96 * std_posterior)]



        try:

            load_GPRfits = np.load(f'Straindata/GPRfits_/{property}_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_Ni={len(self.parameter_space_input)}_No={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}.npz', allow_pickle=True)
            
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

            for node_i in range(len(self.empirical_nodes_idx)):
                
                mean_prediction, uncertainty_region = gaussian_process_regression(node_i, training_set, property, self.greedy_parameters_idx, plot_fits)
                
                gaussian_fit[node_i] = mean_prediction
                uncertainty_region.append(uncertainty_region)

        if plot_fits is True:
            load_parameterspace_input = np.load(f'Straindata/Residuals/residuals_{property}_e=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}]_N={len(self.parameter_space_input)}.npz')
            residual_parameterspace_input = load_parameterspace_input['residual']
            self.phase_shift_total_input = load_parameterspace_input['total_phase_shift']

            self.TS_M = load_parameterspace_input['TS_M']
            try:
                # load_parameterspace_input = np.load(f'Straindata/Residuals/residual_{property}_full_parameterspace_input_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}.npz')
                # residual_parameterspace_input = load_parameterspace_input['residual']
                # self.phase_shift_total_input = load_parameterspace_input['total_phase_shift']
                # self.TS_M = load_parameterspace_input['TS_M']

                # load_parameterspace_output = np.load(f'Straindata/Residuals/residuals_{property}_e=[{min(self.parameter_space_output)}_{max(self.parameter_space_output)}]_N={len(self.parameter_space_output)}.npz')
                # residual_parameterspace_output = load_parameterspace_output['residual']
                # self.phase_shift_total_output = load_parameterspace_output['total_phase_shift']
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
                residual_parameterspace_output, self.phase_shift_total_output = self.generate_property_dataset(eccmin_list=self.parameter_space_output, property=property, save_dataset_to_file=True)

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
                # axs[0].plot(self.parameter_space_output, residual_parameterspace_output[:, self.empirical_nodes_idx[i]], linestyle='dashed', linewidth=0.6, color=colors[i], label=f't={self.TS_M[self.empirical_nodes_idx[i]]}')
                axs[0].plot(self.parameter_space_output, residual_parameterspace_output[:, self.empirical_nodes_idx[i]], linestyle='dashed', linewidth=0.6, color=colors[i])

                # axs[0].plot(self.parameter_space_output, residual_parameterspace_output[:, -1], linestyle='dashed', linewidth=0.6, color=colors[i], label=f't={self.TS_M[self.empirical_nodes_idx[i]]}')
                # axs[0].plot(self.parameter_space_input, residual_parameterspace_input[:, -1])
                relative_error = abs(residual_parameterspace_output[:, self.empirical_nodes_idx[i]] - gaussian_fit.T[:, i]) / abs(residual_parameterspace_output[:, self.empirical_nodes_idx[i]])
                axs[1].plot(self.parameter_space_output, relative_error, color=colors[i], linewidth=0.6, label=f'Error {i+1} (t={int(self.TS_M[self.empirical_nodes_idx[i]])})')
                axs[1].set_ylim(0, 2)
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
        
        if save_fits_to_file is True and not os.path.isfile(f'Straindata/GPRfits/{property}_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_Ni={len(self.parameter_space_input)}_No={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}.npz'):

            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Straindata/GPRfits', exist_ok=True)
            np.savez(f'Straindata/GPRfits/{property}_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_Ni={len(self.parameter_space_input)}_No={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}.npz', GPR_fit=gaussian_fit, training_set=training_set, uncertainty_region=np.array(uncertainty_region, dtype=object), greedy_parameters=self.greedy_parameters_idx, empirical_nodes=self.empirical_nodes_idx, residual_greedy_basis=self.residual_greedy_basis)
            print('GPR fits saved in Straindata/GPRfits')

        return gaussian_fit, uncertainty_region

    def generate_surrogate_model(self, plot_surr_datapiece_at_ecc=False, save_fig_datapiece=False, plot_surr_at_ecc=False, save_fig_surr=False, plot_GPRfit=False, save_fits_to_file=False):

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
            if property == 'phase' and self.surrogate_phase is not None:
                surrogate_datapiece = self.surrogate_phase
            elif property == 'amplitude' and self.surrogate_amp is not None:
                surrogate_datapiece = self.surrogate_amp
            else:
                m, _ = B_matrix.shape

                reconstructed_residual = np.zeros((len(self.TS_M), len(self.parameter_space_output)))

                for i in range(m):
                    reconstructed_residual += np.dot(B_matrix[i, :].reshape(-1, 1), fit_matrix[i, :].reshape(1, -1))


                # Change back from residual to original (+ circulair)
                surrogate_datapiece = residual_to_original(residual_dataset=reconstructed_residual, property=property)

                if property == 'phase':

                    load_phase_shifts = np.load('Straindata/Phaseshift/estimated_phase_shift.npz')
                    loaded_phase_shift = load_phase_shifts['phase_shift']
                    loaded_parameter_space = load_phase_shifts['parameter_space']

                    original_array = loaded_phase_shift[loaded_parameter_space <= max(self.parameter_space_output)]
                    original_array_input = loaded_phase_shift[loaded_parameter_space <= max(self.parameter_space_input)]
                    # Define the new desired size, less than the original
                    new_size = len(self.parameter_space_output)

                    # Generate the original indices and new (downsampled) indices
                    old_indices = np.linspace(0, len(original_array) - 1, len(original_array))
                    new_indices = np.linspace(0, len(original_array) - 1, new_size)

                    # Interpolate the array at the new, downsampled indices
                    self.phase_shift_total_output = np.interp(new_indices, old_indices, original_array)

                    # fig_shift = plt.figure()
                    # plt.plot(loaded_parameter_space[loaded_parameter_space <= max(self.parameter_space_output)], original_array, label='old')
                    # plt.plot(loaded_parameter_space[loaded_parameter_space <= max(self.parameter_space_input)], original_array, label='input')
                    
                    # plt.plot(self.parameter_space_output, self.phase_shift_total_output, label='new')
                    # plt.plot(self.parameter_space_input, self.phase_shift_total_input)
                    # plt.legend()

                    surrogate_datapiece = surrogate_datapiece + self.phase_shift_total_output

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
                axs[1].plot(self.TS_M, relative_error, linewidth=0.6)
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

        # Get matrix with interpolated fits and B_matrix
        fit_matrix_amp = self.fit_to_training_set(min_greedy_error=self.min_greedy_error_amp, N_greedy_vecs=self.N_greedy_vecs_amp, property='amplitude', plot_fits=plot_GPRfit, save_fits_to_file=save_fits_to_file)[0]
        B_matrix_amp = compute_B_matrix()
        # Reconstruct amplitude datapiece
        surrogate_amp = reconstruct_surrogate_datapiece(property='amplitude', B_matrix=B_matrix_amp, fit_matrix=fit_matrix_amp, plot_surr_datapiece_at_ecc=plot_surr_datapiece_at_ecc)
        self.surrogate_amp = surrogate_amp
        # Get matrix with interpolated fits and B_matrix
        fit_matrix_phase = self.fit_to_training_set(min_greedy_error=self.min_greedy_error_phase, N_greedy_vecs=self.N_greedy_vecs_phase, property='phase', plot_fits=plot_GPRfit, save_fits_to_file=save_fits_to_file)[0]
        B_matrix_phase = compute_B_matrix()

        # Reconstruct phase datapiece
        surrogate_phase = reconstruct_surrogate_datapiece(property='phase', B_matrix=B_matrix_phase, fit_matrix=fit_matrix_phase, plot_surr_datapiece_at_ecc=plot_surr_datapiece_at_ecc)
        self.surrogate_phase = surrogate_phase

        h_surrogate = np.zeros((len(self.TS_M), len(self.parameter_space_output)), dtype=complex)
        h_surrogate = surrogate_amp * np.exp(1j * surrogate_phase)

        if (plot_surr_at_ecc is not False) and (not isinstance(plot_surr_at_ecc, float)):
            print('plot_surr_datapiece_at_ecc must be float value!')

        if isinstance(plot_surr_at_ecc, float):
            
            fig_surrogate, axs = plt.subplots(4, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1, 3, 1], 'hspace': 0.2}, sharex=True)
            # fig_surrogate, axs = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

            try:
                index_ecc_min = np.where(self.parameter_space_output == plot_surr_at_ecc)[0][0]
            except:
                print(f'Eccentricity value {plot_surr_at_ecc} not in parameterspace!')
                sys.exit(1)

            true_hp, true_hc, TS_M = self.simulate_inspiral_mass_independent(plot_surr_at_ecc)
            phase = np.array(waveform.utils.phase_from_polarizations(true_hp, true_hc))
            # phase += self.phase_shift_total_output[index_ecc_min]
            amp = np.array(waveform.utils.amplitude_from_polarizations(true_hp, true_hc))

            true_h = amp * np.exp(1j * phase)
            length_diff = len(true_hp) - self.waveform_size

            axs[0].set_title(f'Eccentricity = {plot_surr_at_ecc}')
            # axs[0].plot(self.TS_M, true_hp[length_diff:], linewidth=0.6, label='True waveform before')
            axs[0].plot(self.TS_M, np.real(true_h)[length_diff:], linewidth=0.6, label='True waveform after')
            axs[0].plot(self.TS_M, np.real(h_surrogate[:, index_ecc_min]), linewidth=0.6, label='Surrogate')
            axs[0].set_ylabel('$h_+$')
            axs[0].grid(True)
            axs[0].legend()

            # Calculate and Plot plus polarisation error 
            relative_error_hp = abs(np.real(h_surrogate[:, index_ecc_min]) - np.real(true_h)[length_diff:]) / abs(np.real(true_h)[length_diff:])
            relative_error_hp[relative_error_hp > 1] = 0
            
            axs[1].plot(self.TS_M, relative_error_hp, linewidth=0.6)
            axs[1].set_ylabel(f'Rel. Error in $h_+$')
            axs[1].grid(True)
            # axs[1].set_ylim(0, 10)
            # axs[1].set_title('Relative error $h_x$')

            # axs[2].plot(self.TS_M, true_hc[length_diff:], linewidth=0.6, label='True waveform before')
            axs[2].plot(self.TS_M, np.imag(true_h)[length_diff:], linewidth=0.6, label='True waveform after')
            axs[2].plot(self.TS_M, np.imag(h_surrogate[:, index_ecc_min]), linewidth=0.6, label='Surrogate')
            axs[2].grid(True)
            axs[2].set_ylabel('$h_x$')
            axs[2].legend()

            # # axs[2].plot(self.TS_M, true_hc[length_diff:], linewidth=0.6, label='True waveform before')
            # axs[1].plot(self.TS_M, np.imag(true_h)[length_diff:], linewidth=0.6, label='True waveform after')
            # axs[1].plot(self.TS_M, np.imag(h_surrogate[:, index_ecc_min]), linewidth=0.6, label='Surrogate')
            # axs[1].grid(True)
            # axs[1].set_ylabel('$h_x$')
            # axs[1].legend()

            # Calculate and Plot cross polarisation error
            relative_error_hc = abs(np.imag(h_surrogate[:, index_ecc_min]) - np.imag(true_h)[length_diff:]) / abs(np.imag(true_h)[length_diff:])
            relative_error_hc[relative_error_hc > 1] = 0
            axs[3].plot(self.TS_M, relative_error_hc, linewidth=0.6)
            axs[3].set_ylabel(f'Rel. Error in $h_x$')
            axs[3].set_xlabel('t [M]')
            axs[3].grid(True)
            # axs[3].set_ylim(0, 10)


            if save_fig_surr is True:
                figname = f'Surrogate_wf_eccmin={plot_surr_at_ecc}_for_ecc=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}].png'
                
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Surrogate_wf', exist_ok=True)
                fig_surrogate.savefig('Images/Surrogate_wf/' + figname)

                print('Figure is saved in Images/Surrogate_wf')

        return h_surrogate, surrogate_amp, surrogate_phase

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
# test = Simulate_Inspiral(0.4)


# gs = Generate_Surrogate(parameter_space=[0.01, 0.2], amount_input_wfs=35, amount_output_wfs=500, min_greedy_error_amp=1e-2, min_greedy_error_phase=1e-3, waveform_size=3500, freqmin=18)
# ecc_list = np.linspace(0.01, 0.2, num=20).round(5)
# gs.fit_to_training_set('phase', 1e-3)
# gs.fit_to_training_set('amplitude', 5e-2)
# print(gs.parameter_space_output)
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

# gs.generate_surrogate_model(plot_surr_datapiece_at_ecc=0.1905, plot_surr_at_ecc=0.1905, plot_GPRfit=True)
# gs.generate_surrogate_model(plot_surr_datapiece_at_ecc=0.0336, plot_surr_at_ecc=0.0336, plot_GPRfit=True)

# plt.show()
