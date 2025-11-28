from generate_greedy_training_set import *

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Product, ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel, ExpSineSquared, DotProduct, ConstantKernel as C
import time
import seaborn as sns
from matplotlib.lines import Line2D
from inspect import getframeinfo
from sklearn.preprocessing import StandardScaler
from fileinput import filename
from inspect import currentframe
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

f = currentframe()

class Generate_Offline_Surrogate(Generate_TrainingSet):

    def __init__(self, time_array, ecc_ref_parameterspace_range, amount_input_wfs, amount_output_wfs, N_basis_vecs_amp=None, N_basis_vecs_phase=None, min_greedy_error_amp=None, min_greedy_error_phase=None, training_set_selection='GPR_opt', minimum_spacing_greedy=0.008, f_lower=10, f_ref=20, chi1=0, chi2=0, phiRef=0., rel_anomaly=0., inclination=0., truncate_at_ISCO=True, truncate_at_tmin=True, geometric_units=True):
        
        
        if (N_basis_vecs_amp is None and N_basis_vecs_phase is None) and \
            (min_greedy_error_amp is None and min_greedy_error_phase is None):
                print('Choose either settings for the amount of basis_vecs OR the minimum greedy error.')
                sys.exit(1)

        self.ecc_ref_parameterspace_range = ecc_ref_parameterspace_range
        ecc_ref_parameter_space_input = np.linspace(ecc_ref_parameterspace_range[0], ecc_ref_parameterspace_range[1], amount_input_wfs).round(4)
        self.ecc_ref_parameter_space_output = np.linspace(ecc_ref_parameterspace_range[0], ecc_ref_parameterspace_range[1], amount_output_wfs).round(4)
        self.amount_input_wfs = amount_input_wfs
        self.amount_output_wfs = amount_output_wfs

        self.waveforms_in_geom_units = geometric_units
        self.training_set_selection = training_set_selection

        self.surrogate_amp = None
        self.surrogate_phase = None

        self.gaussian_fit_amp = None
        self.gaussian_fit_phase = None
        self.best_rep_parameters_idx_amp = None
        self.best_rep_parameters_idx_phase = None

        Generate_TrainingSet.__init__(self, time_array=time_array, ecc_ref_parameterspace=ecc_ref_parameter_space_input, N_basis_vecs_amp=N_basis_vecs_amp, N_basis_vecs_phase=N_basis_vecs_phase, min_greedy_error_amp=min_greedy_error_amp, min_greedy_error_phase=min_greedy_error_phase, minimum_spacing_greedy=minimum_spacing_greedy, chi1=chi1, chi2=chi2, phiRef=phiRef, rel_anomaly=rel_anomaly, inclination=inclination, truncate_at_ISCO=truncate_at_ISCO, truncate_at_tmin=truncate_at_tmin)

    
    def fit_to_training_set(self, property, min_greedy_error=None, N_basis_vecs=None, training_set=None, X_train=None, save_fits_to_file=True, plot_kernels=False, 
                            plot_GPR_fits=False, save_fig_GPR_fits=False, save_fig_kernels=False, plot_residuals_ecc_evolve=False, save_fig_ecc_evolve=False, plot_residuals_time_evolve=False, save_fig_time_evolve=False, no_file_load=False):

        if N_basis_vecs is not self.N_basis_vecs_amp and property == 'amplitude':
            print(self.colored_text('WARNING: N_basis_vecs does not match self.N_basis_vecs_amp. Changing self.N_basis_vecs_amp to N_basis_vecs', 'red'))
            self.N_basis_vecs_amp = N_basis_vecs
        elif N_basis_vecs is not self.N_basis_vecs_phase and property == 'phase':
            print(self.colored_text('WARNING: N_basis_vecs does not match self.N_basis_vecs_phase. Changing self.N_basis_vecs_phase to N_basis_vecs', 'red'))
            self.N_basis_vecs_phase = N_basis_vecs

        # Check for GPR issues
        def diagnose_gpr_issues(gaussian_process, X_train_scaled, y_train_scaled):
            issues = []
            
            # 1. Check if length scale is hitting bounds
            kernel = gaussian_process.kernel_
            if hasattr(kernel, 'length_scale_bounds'):
                length_scale = kernel.length_scale
                lb, ub = kernel.length_scale_bounds
                if np.abs(length_scale - lb) < 1e-6 or np.abs(length_scale - ub) < 1e-6:
                    issues.append("Length scale hitting optimization bounds")
            
            # 2. Check training fit quality
            y_pred, std_pred = gaussian_process.predict(X_train_scaled, return_std=True)
            residuals = y_pred - y_train_scaled
            if np.max(np.abs(residuals)) > 2.0:  # Large residuals in scaled space
                issues.append("Poor fit to training data")
            
            # 3. Check uncertainty calibration
            z_scores = residuals / std_pred
            if np.mean(z_scores**2) > 2.0:  # Poor uncertainty calibration
                issues.append("Poor uncertainty calibration")
            
            # 4. Check kernel matrix conditioning
            K = gaussian_process.kernel_(X_train_scaled)
            cond_num = np.linalg.cond(K + 1e-8 * np.eye(K.shape[0]))
            if cond_num > 1e10:
                issues.append(f"Severe ill-conditioning (cond={cond_num:.2e})")
            
            return issues

        def gaussian_process_regression_test(time_node, training_set, optimized_kernel=None, plot_kernels=plot_kernels, save_fig_kernels=save_fig_kernels):
            """
            fit_to_params: choose 'greedy' for fitting to greedy paramaters, or choors 'GPR_opt' for fitting to GPR optimized chosen parameters
            """
            # Extract X and training data
            X = self.ecc_ref_parameter_space_output[:, np.newaxis]
            X_train = np.array(self.best_rep_parameters).reshape(-1, 1)
            y_train = np.squeeze(training_set.T[time_node])

            # Scale X_train
            scaler_x = StandardScaler()
            X_train_scaled = scaler_x.fit_transform(X_train)

            # Scale X (for predictions)
            X_scaled = scaler_x.transform(X)

            # Scale y_train
            scaler_y = StandardScaler()
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

            # Define kernels to try

            # Median distance between nearest neighbors 
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(X_train_scaled)
            distances, indices = nn.kneighbors(X_train_scaled)
            median_nn_distance = np.median(distances[:, 1])  # distance to nearest neighbor
            base_ls = max(0.1, median_nn_distance * 2)  # Scale up from nearest neighbor distance
            print('base_ls = ', base_ls)

            # Length scale multipliers and upper bounds
            ls_multipliers = [0.3, 1.0, 3.0, 10.0] # length scale multipliers
            ls_upper_bounds = [0.5, 1.0, 2.0, 5.0] # length scale upper bounds
            smoothness_params = [1.0, 1.5, 2.0, 2.5]  # Matern nu values

            # Different kernel configurations
            kernels = []

            

            # Matern based kernels without noise
            for nu in smoothness_params:
                for ls_mult in ls_multipliers:
                    for ls_ub in ls_upper_bounds:
                        if base_ls * ls_mult <= ls_ub:
                            kernel = Matern(length_scale=base_ls * ls_mult, length_scale_bounds=(0.1, ls_ub), nu=nu)
                            kernels.append([kernel, ls_ub])
            
            # # Matern based kernels with noise terms
            # noise_levels = [1e-4, 1e-3, 1e-2, 1e-1]
            # for nu in smoothness_params:
            #     for ls_mult in ls_multipliers:
            #         for ls_ub in ls_upper_bounds:
            #             for noise in noise_levels:
            #                 kernel = Matern(length_scale=base_ls * ls_mult, length_scale_bounds=(1, ls_ub), nu=nu) + WhiteKernel(noise_level=noise, noise_level_bounds=(1e-6, 1))
            #                 kernels.append(kernel)

            # # RBF based kernels
            # for ls_mult in ls_multipliers:
            #     for ls_ub in ls_upper_bounds:
            #         kernel = RBF(length_scale=base_ls * ls_mult, length_scale_bounds=(1, ls_ub))
            #         kernels.append(kernel)

            # # Rational quadratic based kernels
            # alphas = [0.1, 1.0, 10.0]
            # for alpha in alphas:
            #     for ls_mult in ls_multipliers:
            #         for ls_ub in ls_upper_bounds:
            #             kernel = RationalQuadratic(length_scale=base_ls * ls_mult, alpha=alpha, length_scale_bounds=(1e-2, ls_ub), alpha_bounds=(1e-2, 100))
            #             kernels.append(kernel)

            # kernels = [
            #     Matern(length_scale=0.1, length_scale_bounds=(1e-1, 1), nu=1.5)  # <= 0.3 eccentricity,

            # ]

            mean_prediction_per_kernel = []
            std_predictions_per_kernel = []
            lml_per_kernel = []

            best_lml = -np.inf # log marginal likelihood

            for kernel in kernels:
                try:
                    start = time.time()
                    # if optimized_kernel is None:
                    #     gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
                    # else:
                    #     gaussian_process = GaussianProcessRegressor(kernel=optimized_kernel, optimizer=None)
                    # print(self.colored_text(f'Trying kernel: {kernel[0]} with ls_bounds=[0.1, {kernel[1]}]', 'blue'))
                    gaussian_process = GaussianProcessRegressor(kernel=kernel[0], n_restarts_optimizer=20, random_state=42)
                    
                    # Fit the GP model on scaled data
                    gaussian_process.fit(X_train_scaled, y_train_scaled)
                    optimized_kernel = gaussian_process.kernel_

                    end = time.time()

                    # Log-Marginal Likelihood
                    lml = gaussian_process.log_marginal_likelihood_value_

                    # Calculate training score (RMSE on training data)
                    y_pred_train, std_train = gaussian_process.predict(X_train_scaled, return_std=True)
                    train_rmse = np.sqrt(np.mean((y_pred_train - y_train_scaled)**2))
                    
                    # Print the optimized kernel and hyperparameters
                    # print(f"kernel = {kernel[0]}, ls_bounds=[0.1, {kernel[1]}]; Optimized kernel: {optimized_kernel} | time = {end - start:.2f}s | LML = {lml:.4f}, training_score = {train_rmse}")

                    # Make predictions on scaled X
                    mean_prediction_scaled, std_prediction_scaled = gaussian_process.predict(X_scaled, return_std=True)
                    mean_prediction = scaler_y.inverse_transform(mean_prediction_scaled.reshape(-1, 1)).flatten()
                    std_prediction = std_prediction_scaled * scaler_y.scale_[0]

                    mean_prediction_per_kernel.append(mean_prediction)
                    std_predictions_per_kernel.append(std_prediction)

                    issues = diagnose_gpr_issues(gaussian_process, X_train_scaled, y_train_scaled)
                    # if issues:
                        # print(self.colored_text("GPR ISSUES FOUND:", 'red'), issues)

                    if lml > best_lml:
                        best_lml = lml
                        best_guess_kernel = kernel
                        best_optimized_kernel = optimized_kernel
                        best_y_predict = mean_prediction
                        best_y_predict_std = std_prediction
                except Exception as e:
                    print(self.colored_text(f'GPR failed for kernel {kernel}: {e}', 'red'))
                    continue

            print(self.colored_text(f"Best guess kernel = {best_guess_kernel}; Optimized kernel: {best_optimized_kernel} | time = {end - start:.2f}s | LML = {best_lml:.4f} | X_train_scaled = {X_train_scaled[:10]} | Y_train_scaled = {y_train_scaled[:10]}", 'green'))
            lml_per_kernel.append(best_lml)

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
                    figname = f'Images/Gaussian_kernels/Gaussian_kernels_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_ref_parameter_space_input)}_{max(self.ecc_ref_parameter_space_input)}_Ni={len(self.ecc_ref_parameter_space_input)}]_No={len(self.ecc_ref_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_basis_vecs_phase}_Nga={self.N_basis_vecs_amp}.png'
                    
                    # Ensure the directory exists, creating it if necessary and save
                    os.makedirs('Images/Gaussian_kernels', exist_ok=True)
                    GPR_fit.savefig(figname)

                    print(self.colored_text(f'Figure is saved in {figname}', 'blue'))

            return best_y_predict, [(best_y_predict - 1.96 * best_y_predict_std), (best_y_predict + 1.96 * best_y_predict_std)], best_optimized_kernel, lml_per_kernel


        def gaussian_process_regression(time_node, training_set, optimized_kernel=None, plot_kernels=plot_kernels, save_fig_kernels=save_fig_kernels):
            # Extract X and training data
            X = self.ecc_ref_parameter_space_output[:, np.newaxis]
            X_train = np.array(self.best_rep_parameters).reshape(-1, 1)
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
                    figname = f'Gaussian_kernels_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_ref_parameter_space_input)}_{max(self.ecc_ref_parameter_space_input)}_Ni={len(self.ecc_ref_parameter_space_input)}]_No={len(self.ecc_ref_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}.png'
                    
                    # Ensure the directory exists, creating it if necessary and save
                    os.makedirs('Images/Gaussian_kernels', exist_ok=True)
                    GPR_fit.savefig('Images/Gaussian_kernels/' + figname)

                    print('Figure is saved in Images/Gaussian_kernels/' + figname)

            return mean_prediction, [(mean_prediction - 1.96 * std_prediction), (mean_prediction + 1.96 * std_prediction)], optimized_kernel, lml_per_kernel


        try:
            start = time.time()
            if no_file_load is True:
                raise FileNotFoundError
            
            filename = f'Straindata_/GPRfits/GPRfits_{self.training_set_selection}_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_ref_parameter_space_input)}_{max(self.ecc_ref_parameter_space_input)}_N={self.amount_input_wfs}]_No={self.amount_output_wfs}_g_err={min_greedy_error}_Ng_vecs={N_basis_vecs}_min_s={self.minimum_spacing_greedy}.npz'
            load_GPRfits = np.load(filename, allow_pickle=True)
            
            gaussian_fit = load_GPRfits['GPR_fit']
            self.empirical_nodes_idx = load_GPRfits['empirical_nodes']
            self.residual_reduced_basis = load_GPRfits['residual_reduced_basis']
            self.time = load_GPRfits['time']
            self.amp_circ = load_GPRfits['amp_circ']
            self.phase_circ = load_GPRfits['phase_circ']
            lml_fits = load_GPRfits['lml_fits']
            training_set = load_GPRfits['training_set']
            self.best_rep_parameters_idx = load_GPRfits['best_rep_parameters_idx']
            self.best_rep_parameters = load_GPRfits['best_rep_parameters']
            uncertainty_region = load_GPRfits['uncertainty_region'].tolist()
            
            print(f'GPRfit {property} load succeeded: {time.time() - start:.4f}s, emp_nodes={self.empirical_nodes_idx}, best_rep_params={self.best_rep_parameters_idx}')
            load_GPRfits.close()

        except Exception as e:
            print(f'line {getframeinfo(f).lineno}: {e}')

            if training_set is None:
                # Generate the training set of greedy parameters at empirical nodes
                training_set = self.get_training_set_greedy(property=property, min_greedy_error=min_greedy_error, N_greedy_vecs=N_basis_vecs)
            
            # Create empty arrays to save fitvalues
            gaussian_fit = np.zeros((len(training_set.T), len(self.ecc_ref_parameter_space_output)))
            uncertainty_region = []
            lml_fits = []

            print(f'Interpolate {property}...')

            start2 = time.time()
            optimized_kernel = None
            for node_i in range(len(self.empirical_nodes_idx)):
                
                mean_prediction, uncertainty_region, optimized_kernel, lml = gaussian_process_regression_test(node_i, training_set, optimized_kernel, X_train, plot_kernels)

                gaussian_fit[node_i] = mean_prediction # Best prediction 
                uncertainty_region.append(uncertainty_region) # 95% confidence level
                lml_fits.append(lml) # Log-Marginal likelihood

            end2 = time.time()
            print(f'time full GPR = {end2 - start2}')

        # If plot_fits is True, plot the GPR fits
        if plot_GPR_fits is True:
            self._plot_GPR_fits(property, gaussian_fit, training_set, lml_fits, save_fig_fits=save_fig_GPR_fits)

            # fig_greedy_params = plt.figure()
            # plt.scatter(self.ecc_ref_parameter_space_input[self.best_rep_parameters_idx], np.zeros(len(self.best_rep_parameters_idx)))
            # plt.plot(self.ecc_ref_parameter_space_input, np.zeros(len(self.ecc_ref_parameter_space_input)))
            # plt.xlabel('eccentricity')
            # plt.title(f'Chosen greedy parameters {property}')
            
        # If save_fits_to_file is True, save the GPR fits to a file
        if save_fits_to_file is True:
            # Save the GPR fits to a file
            self._save_GPR_fits_to_file(property, gaussian_fit, training_set, lml_fits, uncertainty_region)
        
        # If plot_residuals_ecc_evolve or plot_residuals_time_evolve is True, plot the residuals
        if (plot_residuals_ecc_evolve is True) or (plot_residuals_time_evolve is True):
            # Load residual input data
            load_parameterspace_input = np.load(f'Straindata/Residuals/residuals_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_ref_parameter_space_input)}_{max(self.ecc_ref_parameter_space_input)}_N={len(self.ecc_ref_parameter_space_input)}].npz')
            residual_parameterspace_input = load_parameterspace_input['residual']
            
            # Close file
            load_parameterspace_input.close()
            # Plot residuals
            self._plot_residuals(residual_dataset=residual_parameterspace_input, ecc_list=self.ecc_ref_parameter_space_input, property=property, plot_eccentric_evolv=plot_residuals_ecc_evolve, save_fig_eccentric_evolve=save_fig_ecc_evolve, plot_time_evolve=plot_residuals_time_evolve, save_fig_time_evolve=save_fig_time_evolve)

        return gaussian_fit, uncertainty_region
    
    def fit_to_training_set_GPR_opt(self, property, N_basis_vecs=None, save_fits_to_file=True, plot_kernels=False, 
                            plot_GPR_fits=False, save_fig_GPR_fits=False, save_fig_kernels=False, plot_residuals_ecc_evolve=False, save_fig_ecc_evolve=False, plot_residuals_time_evolve=False, save_fig_time_evolve=False,
                            plot_greedy_vecs=False, save_fig_greedy_vecs=False, plot_greedy_error=False, save_fig_greedy_error=False, plot_emp_nodes_at_ecc=False, save_fig_emp_nodes=False, plot_training_set=False, save_fig_training_set=False):
        save_fits_to_file_iter = False

        # Get first 3 points to produce a start for GPR
        residual_training_set = self.get_training_set_greedy(property, N_greedy_vecs=3, emp_nodes_of_full_dataset=True, plot_greedy_error=plot_greedy_error, save_fig_greedy_error=save_fig_greedy_error, plot_emp_nodes_at_ecc=plot_emp_nodes_at_ecc, save_fig_emp_nodes=save_fig_emp_nodes, plot_training_set=plot_training_set, save_fig_training_set=save_fig_training_set, 
                        save_dataset_to_file=True, plot_greedy_vecs=plot_greedy_vecs, save_fig_greedy_vecs=save_fig_greedy_vecs)

        while len(self.residual_reduced_basis) <= N_basis_vecs:

            # Update the length of the basis for every iteration
            if property == 'phase':
                self.N_basis_vecs_phase = len(residual_training_set)
            else:
                self.N_basis_vecs_amp = len(residual_training_set)

            # Save fits only for last iteration 
            if len(self.residual_reduced_basis) == N_basis_vecs:
                save_fits_to_file_iter = save_fits_to_file

            # Fit the basis vecs with GPR and save fits file at last iteration
            gaussian_fit, uncertainty_region = self.fit_to_training_set(property, N_basis_vecs=len(self.residual_reduced_basis), training_set=residual_training_set, no_file_load=True, save_fits_to_file=save_fits_to_file_iter, plot_kernels=plot_kernels, plot_GPR_fits=plot_GPR_fits, save_fig_kernels=save_fig_kernels, save_fig_GPR_fits=save_fig_GPR_fits, plot_residuals_ecc_evolve=plot_residuals_ecc_evolve, save_fig_ecc_evolve=save_fig_ecc_evolve, plot_residuals_time_evolve=plot_residuals_time_evolve, save_fig_time_evolve=save_fig_time_evolve)

            # Load in property residuals of full parameter space dataset
            try:
                filename = f'Straindata/Residuals/residuals_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_ref_parameter_space_output)}_{max(self.ecc_ref_parameter_space_output)}_N={len(self.ecc_ref_parameter_space_output)}].npz'
                with np.load(filename) as data:
                    residual_parameterspace_output = data['residual']
                    self.time = data['time']

            except Exception as e:
                print(f'line {getframeinfo(f).lineno}: {e}')
                residual_parameterspace_output = self.generate_property_dataset(ecc_list=self.ecc_ref_parameter_space_output, property=property, save_dataset_to_file=True)

            # Calculate the relative errors of GPR fits vs property dataset
            combined_gaussian_error = np.zeros(len(self.ecc_ref_parameter_space_output))
            for i in range(len(gaussian_fit)):
                combined_gaussian_error += abs(residual_parameterspace_output[:, self.empirical_nodes_idx[i]] - gaussian_fit.T[:, i])
            
            # Add new training set point at the place of worst fit error
            worst_relative_GPR_error_idx = np.argmax(combined_gaussian_error)

            # Add time-domain vec of worst fit parameter to the residual basis
            opt_residual_basis_vector = residual_parameterspace_output[worst_relative_GPR_error_idx]
            self.residual_reduced_basis = np.vstack([self.residual_reduced_basis, opt_residual_basis_vector])

            # Update the best pick parameters 
            self.best_rep_parameters_idx = np.append(list(self.best_rep_parameters_idx), worst_relative_GPR_error_idx)
            self.best_rep_parameters = np.append(list(self.best_rep_parameters), self.ecc_ref_parameter_space_output[worst_relative_GPR_error_idx])

            #Generate the training set at empirical nodes for next GPR iteration
            residual_training_set = self.residual_reduced_basis[:, self.empirical_nodes_idx]

        print(f'Final GPR optimized parameters {property}, N_basis_vecs={N_basis_vecs}, emp_nodes={len(self.empirical_nodes_idx)}')
        return gaussian_fit, uncertainty_region

    def _save_GPR_fits_to_file(self, property, gaussian_fit, training_set, lml_fits, uncertainty_region):
            if property == 'phase':
                min_greedy_error = self.min_greedy_error_phase
                N_basis_vecs=self.N_basis_vecs_phase
            else:
                min_greedy_error = self.min_greedy_error_amp
                N_basis_vecs=self.N_basis_vecs_amp

            filename = f'Straindata/GPRfits/GPRfits_{self.training_set_selection}_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_ref_parameter_space_input)}_{max(self.ecc_ref_parameter_space_input)}_N={self.amount_input_wfs}]_No={self.amount_output_wfs}_g_err={min_greedy_error}_Ng_vecs={N_basis_vecs}_min_s={self.minimum_spacing_greedy}.npz'
            
            if not os.path.isfile(filename):
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Straindata/GPRfits', exist_ok=True)

                if self.phase_circ is None or self.amp_circ is None:
                    self.circulair_wf()
                np.savez(filename, GPR_fit=gaussian_fit, empirical_nodes=self.empirical_nodes_idx, residual_reduced_basis=self.residual_reduced_basis, time=self.time, lml_fits=lml_fits, training_set=training_set, best_rep_parameters_idx=self.best_rep_parameters_idx, best_rep_parameters=self.best_rep_parameters, uncertainty_region=np.array(uncertainty_region, dtype=object), phase_circ=self.phase_circ, amp_circ=self.amp_circ)
                print('GPR fits saved in ', filename)
        

    def _plot_GPR_fits(self, property, gaussian_fit=None, training_set=None, lml_fits=None, save_fig_fits=False):

        if gaussian_fit is None or training_set is None or lml_fits is None:
            # Load the GPR fits from file if not provided
            if property == 'phase':
                filename = f'Straindata/GPRfits/GPRfits_{self.training_set_selection}_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_ref_parameter_space_input)}_{max(self.ecc_ref_parameter_space_input)}_N={self.amount_input_wfs}]_No={self.amount_output_wfs}_g_err={self.min_greedy_error_phase}_Ng_vecs={self.N_basis_vecs_phase}_min_s={self.minimum_spacing_greedy}.npz'
            else:
                filename = f'Straindata/GPRfits/GPRfits_{self.training_set_selection}_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_ref_parameter_space_input)}_{max(self.ecc_ref_parameter_space_input)}_N={self.amount_input_wfs}]_No={self.amount_output_wfs}_g_err={self.min_greedy_error_amp}_Ng_vecs={self.N_basis_vecs_amp}_min_s={self.minimum_spacing_greedy}.npz'
                        
            try:
                load_GPR_fits = np.load(filename)
                gaussian_fit = load_GPR_fits['GPR_fit']
                training_set = load_GPR_fits['training_set']
                lml_fits = load_GPR_fits['lml_fits']
                self.empirical_nodes_idx = load_GPR_fits['empirical_nodes']
                self.best_rep_parameters_idx = load_GPR_fits['best_rep_parameters_idx']
                self.best_rep_parameters = load_GPR_fits['best_rep_parameters']

                load_GPR_fits.close()
            except Exception as e:
                print(f'line {getframeinfo(f).lineno}: {e}')

        try:
            filename = f'Straindata/Residuals/residuals_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_ref_parameter_space_output)}_{max(self.ecc_ref_parameter_space_output)}_N={len(self.ecc_ref_parameter_space_output)}].npz'
            load_residual_output = np.load(filename)
            residual_parameterspace_output = load_residual_output['residual']
            self.time = load_residual_output['time']

        except Exception as e:
            print(f'line {getframeinfo(f).lineno}: {e}')
            residual_parameterspace_output = self.generate_property_dataset(ecc_list=self.ecc_ref_parameter_space_output, property=property, save_dataset_to_file=True)


        fig_residual_training_fit, axs = plt.subplots(3, 1, figsize=(11,6), gridspec_kw={'height_ratios': [3, 1, 1], 'hspace': 0.1}, sharex=True)

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
        lml_array = np.array(lml_fits).flatten()
        sorted_lml_fits = np.argsort(lml_array)[::-1] # Highest to lowest

        # Plotting data
        combined_rel_error = np.zeros(len(self.ecc_ref_parameter_space_output))

        # Plot 10 worst fits based on highest log-marginal likelihood

        for i in sorted_lml_fits[:5]:
            # Plot Gaussian fit
            # line_fit, = axs[0].plot(self.ecc_ref_parameter_space_output, gaussian_fit.T[:, i], linewidth=0.6, 
            #                         label=f't={int(self.time[self.empirical_nodes_idx[i]])}')
            line_fit, = axs[0].plot(self.ecc_ref_parameter_space_output, gaussian_fit.T[:, i], linewidth=0.6)
            # Scatter plot for training points
            axs[0].scatter(self.best_rep_parameters, training_set[:, i], s=6)
            
            # Plot residuals (true property)
            # axs[0].plot(self.ecc_ref_parameter_space_output, residual_parameterspace_output[:, self.empirical_nodes_idx[i]], 
            #             linestyle='dashed', linewidth=0.6, label=f'{i}, {self.empirical_nodes_idx[i]}')
            axs[0].plot(self.ecc_ref_parameter_space_output, residual_parameterspace_output[:, self.empirical_nodes_idx[i]], 
                        linestyle='dashed', linewidth=0.6)
            
            # Collect handles and labels for the dynamic fits
            dynamic_handles.append(line_fit)
            dynamic_labels.append(f't={int(self.time[self.empirical_nodes_idx[i]])} [M]')
            
            # Relative error plot
            # axs[1].plot(self.ecc_ref_parameter_space_output, 
            #             abs(residual_parameterspace_output[:, self.empirical_nodes_idx[i]] - gaussian_fit.T[:, i]), 
            #             linewidth=0.6, 
            #             label=f'Error {i+1} (t={int(self.time[self.empirical_nodes_idx[i]])})')
            relative_error = abs(residual_parameterspace_output[:, self.empirical_nodes_idx[i]] - gaussian_fit.T[:, i])
            combined_rel_error += relative_error
            axs[1].plot(self.ecc_ref_parameter_space_output, 
                        relative_error, 
                        linewidth=0.6)
            
            axs[2].scatter(self.best_rep_parameters, np.zeros(len(self.best_rep_parameters)), s=3)

        # Combine custom and dynamic legend elements
        combined_handles = custom_legend_elements + dynamic_handles
        combined_labels = [handle.get_label() for handle in custom_legend_elements] + dynamic_labels

        # Add the combined legend to the top-left subplot
        # axs[0].legend(combined_handles, combined_labels, ncol=2)
        print('combined relative error = ', np.sum(combined_rel_error))
        # Set labels and titles
        if property == 'phase':
            axs[0].set_ylabel('$\Delta \phi$')
            # axs[0].set_title(f'GPRfit $\phi$; greedy error = {min_greedy_error},N={len(self.best_rep_parameters_idx)}')
        elif property == 'amplitude':
            axs[0].set_ylabel('$\Delta$ A')
            # axs[0].set_title(f'GPRfit A; greedy error = {min_greedy_error}, N={len(self.best_rep_parameters_idx)}')
        axs[0].grid()

        axs[1].set_xlabel('eccentricity')
        if property == 'phase':
            axs[1].set_ylabel('|$\Delta \phi_{S} - \Delta \phi|$')
        else:
            axs[1].set_ylabel('|$\Delta A_{S} - \Delta A|$')
        axs[1].grid()

        plt.tight_layout()

        if save_fig_fits is True:
            if property == 'phase':
                figname = f'Images/Gaussian_fits/GPR_fits_new_{self.training_set_selection}_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_ref_parameter_space_input)}_{max(self.ecc_ref_parameter_space_input)}_Ni={len(self.ecc_ref_parameter_space_input)}]_No={len(self.ecc_ref_parameter_space_output)}_g_err={self.min_greedy_error_phase}_Ng_vecs={self.N_basis_vecs_phase}_min_s={self.minimum_spacing_greedy}.png'
            else:
                figname = f'Images/Gaussian_fits/GPR_fits_new_{self.training_set_selection}_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_ref_parameter_space_input)}_{max(self.ecc_ref_parameter_space_input)}_Ni={len(self.ecc_ref_parameter_space_input)}]_No={len(self.ecc_ref_parameter_space_output)}_g_err={self.min_greedy_error_amp}_Ng_vecs={self.N_basis_vecs_amp}_min_s={self.minimum_spacing_greedy}.png'

            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Images/Gaussian_fits', exist_ok=True)
            fig_residual_training_fit.savefig(figname)

            print(self.colored_text(f'Figure is saved in {figname}', 'blue'))

        plt.close(fig_residual_training_fit)

    

    def compute_B_matrix(self, property, save_matrix_to_file=True):

        """
        Computes the B matrix for all empirical nodes and basis functions.
        
        e_matrix: Array of shape (m, time_samples) representing the reduced basis functions evaluated at different time samples.
        V_inv: Inverse of the interpolation matrix of shape (m, m).
        
        Returns:
        B_matrix: Array of shape (m, time_samples) where each row represents B_j(t) for j=1,2,...,m
        """
        if property == 'phase':
            min_greedy_error = self.min_greedy_error_phase
            N_basis_vecs=self.N_basis_vecs_phase
        elif property == 'amplitude':
            min_greedy_error = self.min_greedy_error_amp
            N_basis_vecs=self.N_basis_vecs_amp

        filename = (
            f'Straindata/B_matrix/B_matrix_{self.training_set_selection}_{property}_'
            f'f_lower={self.f_lower}_f_ref={self.f_ref}_'
            f'e=[{self.ecc_ref_parameterspace_range[0]}_{self.ecc_ref_parameterspace_range[1]}_N={self.amount_input_wfs}]_'
            f'No={self.amount_output_wfs}_'
            f'g_err={min_greedy_error}_'
            f'Ng_vecs={N_basis_vecs}_'
            f'min_s={self.minimum_spacing_greedy}.npz'
        )

        try:
            load_B_matrix = np.load(filename)
            B_matrix = load_B_matrix['B_matrix']
            print(f'B_matrix {property} load succeeded: {filename}', B_matrix.shape)

            load_B_matrix.close()

        except Exception as e:
            print(f'line {getframeinfo(f).lineno}: {e}')
            print(self.residual_reduced_basis.shape, len(self.empirical_nodes_idx))

            m, time_samples = self.residual_reduced_basis.shape
            B_matrix = np.zeros((m, time_samples))

            V = np.zeros((m, m))
            for j in range(m):
                for i in range(m):
                    # print(i, j)
                    V[j][i] = self.residual_reduced_basis[i][self.empirical_nodes_idx[j]]

            V_inv = np.linalg.pinv(V)

            
            # Compute each B_j(t) for j = 1, 2, ..., m
            for j in range(m):
                # Compute B_j(t) as a linear combination of all e_i(t) with weights from V_inv[:, j]
                for i in range(m):
                    B_matrix[j] += self.residual_reduced_basis[i] * V_inv[i, j]
            
        
            if save_matrix_to_file is True and not os.path.isfile(filename):

                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Straindata/B_matrix', exist_ok=True)
                np.savez(filename, B_matrix=B_matrix)
                print('B_matrix fits saved in Straindata/B_matrix/' + filename)

        return B_matrix
    

sampling_frequency = 2048 # or 4096
duration = 3 # seconds
time_array = np.linspace(-duration, 0, int(sampling_frequency * duration))  # time in seconds

gt = Generate_Offline_Surrogate(time_array=time_array, ecc_ref_parameterspace_range=[0.0, 0.3], amount_input_wfs=40, amount_output_wfs=500, N_basis_vecs_amp=20, N_basis_vecs_phase=20,
                          minimum_spacing_greedy=0.003, training_set_selection='GPR_opt')

for vecs in [20]:
    # gt.generate_property_dataset(np.linspace(0.0, 0.3, 500).round(4), property='phase', plot_residuals_eccentric_evolv=True, save_fig_time_evolve=True, plot_residuals_time_evolv=True, save_fig_eccentric_evolv=True)
    gt.fit_to_training_set('phase', N_basis_vecs=vecs, save_fits_to_file=True, plot_GPR_fits=True, save_fig_GPR_fits=True, plot_residuals_ecc_evolve=True, save_fig_ecc_evolve=True, plot_residuals_time_evolve=True, save_fig_time_evolve=True)
    gt.fit_to_training_set('amplitude', N_basis_vecs=vecs, save_fits_to_file=True, plot_GPR_fits=True, save_fig_GPR_fits=True, plot_residuals_ecc_evolve=True, save_fig_ecc_evolve=True, plot_residuals_time_evolve=True, save_fig_time_evolve=True)

# plt.show()
# gt.fit_to_training_set('amplitude', N_basis_vecs=21, save_fits_to_file=True)