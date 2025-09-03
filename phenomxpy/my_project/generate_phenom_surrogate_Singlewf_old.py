from fileinput import filename
from generate_phenom_training_set_old import *

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
from sklearn.preprocessing import StandardScaler
from pathlib import Path

f = currentframe()
plt.switch_backend('WebAgg')


faulthandler.enable()

class Generate_Offline_Surrogate(Generate_TrainingSet):

    def __init__(self, time_array, ecc_ref_parameterspace_range, amount_input_wfs, amount_output_wfs, reference_total_mass=60, reference_luminosity_distance=200, N_greedy_vecs_amp=None, N_greedy_vecs_phase=None, min_greedy_error_amp=None, min_greedy_error_phase=None, f_lower=10, f_ref=20, chi1=0, chi2=0, phiRef=0., rel_anomaly=0., inclination=0., truncate_at_ISCO=True, truncate_at_tmin=True, geometric_units=True):
        
        
        if (N_greedy_vecs_amp is None and N_greedy_vecs_phase is None) and \
            (min_greedy_error_amp is None and min_greedy_error_phase is None):
                print('Choose either settings for the amount of greedy_vecs OR the minimum greedy error.')
                sys.exit(1)

        self.ecc_ref_parameterspace_range = ecc_ref_parameterspace_range
        self.ecc_parameter_space_input = np.linspace(ecc_ref_parameterspace_range[0], ecc_ref_parameterspace_range[1], amount_input_wfs).round(4)
        self.ecc_parameter_space_output = np.linspace(ecc_ref_parameterspace_range[0], ecc_ref_parameterspace_range[1], amount_output_wfs).round(4)
        self.amount_input_wfs = amount_input_wfs
        self.amount_output_wfs = amount_output_wfs

        self.min_greedy_error_amp = min_greedy_error_amp
        self.min_greedy_error_phase = min_greedy_error_phase
        self.N_greedy_vecs_amp = N_greedy_vecs_amp
        self.N_greedy_vecs_phase = N_greedy_vecs_phase
        
        self.reference_total_mass = reference_total_mass
        self.reference_luminosity_distance = reference_luminosity_distance
        self.surrogate_amp = None
        self.surrogate_phase = None
        self.waveforms_in_geom_units = geometric_units

        self.gaussian_fit_amp = None
        self.gaussian_fit_phase = None
        self.greedy_parameters_idx_amp = None
        self.greedy_parameters_idx_phase = None

        Generate_TrainingSet.__init__(self, time_array=time_array, ecc_ref_parameterspace=self.ecc_parameter_space_input, total_mass=reference_total_mass, luminosity_distance=reference_luminosity_distance, f_lower=f_lower, f_ref=f_ref, chi1=chi1, chi2=chi2, phiRef=phiRef, rel_anomaly=rel_anomaly, inclination=inclination, truncate_at_ISCO=truncate_at_ISCO, truncate_at_tmin=truncate_at_tmin)

    def simulate_inspiral_mass_dependent(self, total_mass, distance, custom_time_array=None, ecc_ref=None, truncate_at_ISCO=True, truncate_at_tmin=True, plot_polarisations=False, save_fig=False):
        """
        Simulate mass-independent plus and cross polarisations of the eccentric eob waveform (pyseobnr) (2,2) mode from f_start till t0 (waveform peak at t=0).
        
        Parameters:
        ----------------
        ecc_ref [dimensionless], float : For other eccentricity than Class specified ecc_ref, set new value.
        plot_polarisations, True OR False, bool : For a plot of the plus and cross polarisations, set to True.
        save_fig, True Or False, bool : If plot of the polarisations should be saved to a automatically created folder \Images, set to True.
        
        Returns:
        ----------------
        hp [dimensionless], np.array: Time-domain plus polarisation 
        hc [dimensionless], np.array: Time-domain cross polarisation 
        t [M], np.array: Time-domain in mass independent geometric units c=G=M=1
        """

        if custom_time_array is None:
            time_array = self.time
        else:
            time_array = custom_time_array

        # Either set ecc_ref specifically or use the class defined value
        if ecc_ref is None:
            ecc_ref = self.ecc_ref
        else:
            ecc_ref = ecc_ref

        # To compute the runtime for 1 simulated waveform
        start = timer()

        f_ref_geom = HztoMf(self.f_ref, self.reference_total_mass)
        f_lower_geom = HztoMf(self.f_lower, self.reference_total_mass)

        f_ref_SI = MftoHz(f_ref_geom, total_mass)
        f_lower_SI = MftoHz(f_lower_geom, total_mass)

        phen = phenomt.PhenomTE(
            mode=[2,2],
            times=time_array,
            eccentricity=ecc_ref,  
            total_mass=self.total_mass,
            distance=self.luminosity_distance,                
            f_ref=f_ref_SI,                   
            f_lower=f_lower_SI,
            phiRef=self.phiRef,
            inclination=self.inclination)
        
        phen.compute_polarizations(times=time_array, distance=distance, total_mass=total_mass)

        
        if phen.pWF.tmin > time_array[0]:
            warnings.warn(
                "t_min is larger than parts of the specified time-domain, resulting in unphysical waveforms. "
                "Either use the truncate_tmin=True setting to automatically truncate to start from t_min=time_array[0] "
                "or adjust the time-array manually to start at higher values."
            )
            # mask to only include the physical range of the time-domain
            if (self.truncate_at_tmin is True) and (truncate_at_tmin is True):
                mask = self.time >= phen.pWF.tmin

                time_array = time_array[mask]
                phen.hp = phen.hp[mask]
                phen.hc = phen.hc[mask]

                print(f'NEW TIME-DOMAIN (in geometric units): [{int(self.time[0])}, {int(self.time[-1])}] M')
                del mask # clear memory

        # True because it's smallest truncated waveform AND true because the surrogate is called with the ISCO cut-off.
        if (self.truncate_at_ISCO is True) and (truncate_at_ISCO is True):
            # Truncate the waveform at ISCO frequency
            idx_cut = self.truncate_waveform_at_isco(phen, time_array)
            time_array = time_array[:idx_cut]
        
        print(f'time : SimInspiral_M_independent ecc = {round(ecc_ref, 3)}, M = {self.total_mass}, lum_dist={self.luminosity_distance}, t=[{int(time_array[0])}, {int(time_array[-1])}, num={len(time_array)}], f_lower={self.f_lower}, f_ref={self.f_ref} | computation time = {(timer()-start)} seconds')

        if plot_polarisations is True:
            self._plot_polarisations(phen.hp, phen.hc, time_array, save_fig)

        if custom_time_array is None:
            self.time = time_array
            return phen.hp, phen.hc
        else:
            return phen.hp, phen.hc, time_array
    
    def fit_to_training_set(self, property, min_greedy_error=None, N_greedy_vecs=None, save_fits_to_file=True, plot_kernels=False, plot_fits=False, save_fig_kernels=False, save_fig_fits=False, plot_residuals_ecc_evolve=False, save_fig_ecc_evolve=False, plot_residuals_time_evolve=False, save_fig_time_evolve=False):
    
        def gaussian_process_regression(time_node, training_set, optimized_kernel=None, plot_kernels=plot_kernels, save_fig_kernels=save_fig_kernels):
            # Extract X and training data
            X = self.ecc_parameter_space_output[:, np.newaxis]
            X_train = np.array(self.ecc_parameter_space_input[self.greedy_parameters_idx]).reshape(-1, 1)
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
                    figname = f'Gaussian_kernels_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_parameter_space_input)}_{max(self.ecc_parameter_space_input)}_Ni={len(self.ecc_parameter_space_input)}]_No={len(self.ecc_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}.png'
                    
                    # Ensure the directory exists, creating it if necessary and save
                    os.makedirs('Images/Gaussian_kernels', exist_ok=True)
                    GPR_fit.savefig('Images/Gaussian_kernels/' + figname)

                    print('Figure is saved in Images/Gaussian_kernels/' + figname)

            return mean_prediction, [(mean_prediction - 1.96 * std_prediction), (mean_prediction + 1.96 * std_prediction)], optimized_kernel, lml_per_kernel

        try:
            start = time.time()
            filename = f'Straindata/GPRfits/GPRfits_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_parameter_space_input)}_{max(self.ecc_parameter_space_input)}_N={len(self.ecc_parameter_space_input)}]_No={len(self.ecc_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}.npz'
            load_GPRfits = np.load(filename, allow_pickle=True)
            
            gaussian_fit = load_GPRfits['GPR_fit']
            self.empirical_nodes_idx = load_GPRfits['empirical_nodes']
            self.residual_greedy_basis = load_GPRfits['residual_greedy_basis']
            self.time = load_GPRfits['time']
            self.amp_circ = load_GPRfits['amp_circ']
            self.phase_circ = load_GPRfits['phase_circ']
            lml_fits = load_GPRfits['lml_fits']
            training_set = load_GPRfits['training_set']
            self.greedy_parameters_idx = load_GPRfits['greedy_parameters_idx']
            uncertainty_region = load_GPRfits['uncertainty_region'].tolist()
            
            print(f'GPRfit {property} load succeeded: {time.time() - start:.4f}s')

        except Exception as e:
            print(f'line {getframeinfo(f).lineno}: {e}')

            # Generate the training set of greedy parameters at empirical nodes
            training_set = self.get_training_set(property=property, min_greedy_error=min_greedy_error, N_greedy_vecs=N_greedy_vecs)
            # Create empty arrays to save fitvalues
            gaussian_fit = np.zeros((len(training_set.T), len(self.ecc_parameter_space_output)))
            uncertainty_region = []
            lml_fits = []

            print(f'Interpolate {property}...')

            start2 = time.time()
            optimized_kernel = None
            for node_i in range(len(self.empirical_nodes_idx)):
                
                mean_prediction, uncertainty_region, optimized_kernel, lml = gaussian_process_regression(node_i, training_set, optimized_kernel, plot_kernels)
                
                gaussian_fit[node_i] = mean_prediction # Best prediction 
                uncertainty_region.append(uncertainty_region) # 95% confidence level
                lml_fits.append(lml) # Log-Marginal likelihood

            end2 = time.time()
            print(f'time full GPR = {end2 - start2}')

        # If plot_fits is True, plot the GPR fits
        if plot_fits is True:
            self._plot_GPR_fits(property, gaussian_fit, training_set, lml_fits, save_fig_fits=save_fig_fits)

            fig_greedy_params = plt.figure()
            plt.scatter(self.parameter_space_input[self.greedy_parameters_idx], np.zeros(len(self.greedy_parameters_idx)))
            plt.plot(self.parameter_space_input, np.zeros(len(self.parameter_space_input)))
            plt.xlabel('eccentricity')
            plt.title(f'Chosen greedy parameters {property}')
            
        # If save_fits_to_file is True, save the GPR fits to a file
        if save_fits_to_file is True:
            # Save the GPR fits to a file
            self._save_GPR_fits_to_file(property, gaussian_fit, training_set, lml_fits, uncertainty_region)
        
        # If plot_residuals_ecc_evolve or plot_residuals_time_evolve is True, plot the residuals
        if (plot_residuals_time_evolve is True) or (plot_residuals_time_evolve is True):
            load_parameterspace_input = np.load(f'Straindata/Residuals/residuals_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_parameter_space_input)}_{max(self.ecc_parameter_space_input)}_N={len(self.ecc_parameter_space_input)}].npz')
            residual_parameterspace_input = load_parameterspace_input['residual']
            
            self._plot_residuals(residual_dataset=residual_parameterspace_input, ecc_list=self.ecc_parameter_space_input, property=property, plot_eccentric_evolv=plot_residuals_ecc_evolve, save_fig_eccentric_evolve=save_fig_ecc_evolve, plot_time_evolve=plot_residuals_time_evolve, save_fig_time_evolve=save_fig_time_evolve)
   
        return gaussian_fit, uncertainty_region
    

    def _save_GPR_fits_to_file(self, property, gaussian_fit, training_set, lml_fits, uncertainty_region):
            filename = f'Straindata/GPRfits/GPRfits_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_parameter_space_input)}_{max(self.ecc_parameter_space_input)}_N={len(self.ecc_parameter_space_input)}]_No={len(self.ecc_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}.npz'
            if not os.path.isfile(filename):

                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Straindata/GPRfits', exist_ok=True)

                if self.phase_circ is None or self.amp_circ is None:
                    self.circulair_wf()
                    
                np.savez(filename, GPR_fit=gaussian_fit, empirical_nodes=self.empirical_nodes_idx, residual_greedy_basis=self.residual_greedy_basis, time=self.time, lml_fits=lml_fits, training_set=training_set, greedy_parameters_idx=self.greedy_parameters_idx, uncertainty_region=np.array(uncertainty_region, dtype=object), phase_circ=self.phase_circ, amp_circ=self.amp_circ)
                print('GPR fits saved in Straindata/GPRfits/' + filename)
        

    def _plot_GPR_fits(self, property, gaussian_fit=None, training_set=None, lml_fits=None, save_fig_fits=False):

            if gaussian_fit is None or training_set is None or lml_fits is None:
                # Load the GPR fits from file if not provided
                filename = filename = f'Straindata/GPRfits/GPRfits_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_parameter_space_input)}_{max(self.ecc_parameter_space_input)}_N={len(self.ecc_parameter_space_input)}]_No={len(self.ecc_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}.npz'
                load_GPR_fits = np.load(filename)

                gaussian_fit = load_GPR_fits['GPR_fit']
                training_set = load_GPR_fits['training_set']
                lml_fits = load_GPR_fits['lml_fits']
                self.empirical_nodes_idx = load_GPR_fits['empirical_nodes']
                self.greedy_parameters_idx = load_GPR_fits['greedy_parameters_idx']

            try:
                filename = f'Straindata/Residuals/residuals_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_parameter_space_output)}_{max(self.ecc_parameter_space_output)}_N={len(self.ecc_parameter_space_output)}].npz'
                load_residual_output = np.load(filename)
                residual_parameterspace_output = load_residual_output['residual']
                self.time = load_residual_output['time']

            except Exception as e:
                print(f'line {getframeinfo(f).lineno}: {e}')
                residual_parameterspace_output = self.generate_property_dataset(ecc_list=self.ecc_parameter_space_output, property=property, save_dataset_to_file=True)
            

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
                line_fit, = axs[0].plot(self.ecc_parameter_space_output, gaussian_fit.T[:, i], linewidth=0.6, 
                                        label=f't={int(self.time[self.empirical_nodes_idx[i]])}')
                
                # Scatter plot for training points
                axs[0].scatter(self.ecc_parameter_space_input[self.greedy_parameters_idx], training_set[:, i], s=6)
               
                # Plot residuals (true property)
                axs[0].plot(self.ecc_parameter_space_output, residual_parameterspace_output[:, self.empirical_nodes_idx[i]], 
                            linestyle='dashed', linewidth=0.6, label=f'{i}, {self.empirical_nodes_idx[i]}')
                
                # Collect handles and labels for the dynamic fits
                dynamic_handles.append(line_fit)
                dynamic_labels.append(f't={int(self.time[self.empirical_nodes_idx[i]])} [M]')
                
                # Relative error plot
                axs[1].plot(self.ecc_parameter_space_output, 
                            abs(residual_parameterspace_output[:, self.empirical_nodes_idx[i]] - gaussian_fit.T[:, i]), 
                            linewidth=0.6, 
                            label=f'Error {i+1} (t={int(self.time[self.empirical_nodes_idx[i]])})')

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
                figname = f'GPR_fits_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_parameter_space_input)}_{max(self.ecc_parameter_space_input)}_Ni={len(self.ecc_parameter_space_input)}]_No={len(self.ecc_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}.png'
                
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Gaussian_fits', exist_ok=True)
                fig_residual_training_fit.savefig('Images/Gaussian_fits/' + figname)

                print('Figure is saved in Images/Gaussian_fits/' + figname)

    

    def compute_B_matrix(self, property, save_matrix_to_file=True):

        """
        Computes the B matrix for all empirical nodes and basis functions.
        
        e_matrix: Array of shape (m, time_samples) representing the reduced basis functions evaluated at different time samples.
        V_inv: Inverse of the interpolation matrix of shape (m, m).
        
        Returns:
        B_matrix: Array of shape (m, time_samples) where each row represents B_j(t) for j=1,2,...,m
        """
        
        filename = f'Straindata/B_matrix/B_matrix_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_parameter_space_input)}_{max(self.ecc_parameter_space_input)}_N={len(self.ecc_parameter_space_input)}]_No={len(self.ecc_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}.npz'
        try:
            load_B_matrix = np.load(filename)
            B_matrix = load_B_matrix['B_matrix']
            print(f'B_matrix {property} load succeeded: {filename}')

        except Exception as e:
            print(f'line {getframeinfo(f).lineno}: {e}')
            
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
            
        
            if save_matrix_to_file is True and not os.path.isfile(filename):

                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Straindata/B_matrix', exist_ok=True)
                np.savez(filename, B_matrix=B_matrix)
                print('B_matrix fits saved in Straindata/B_matrix/' + filename)

        return B_matrix
    



class Load_Offline_Surrogate(Generate_Offline_Surrogate):
    def __init__(
        self,
        time_array,
        ecc_ref_parameterspace_range,
        amount_input_wfs,
        amount_output_wfs,
        reference_total_mass=60,
        reference_luminosity_distance=200,
        N_greedy_vecs_amp=None,
        N_greedy_vecs_phase=None,
        min_greedy_error_amp=None,
        min_greedy_error_phase=None,
        f_lower=10,
        f_ref=20,
        chi1=0,
        chi2=0,
        phiRef=0.,
        rel_anomaly=0.,
        inclination=0.,
        truncate_at_ISCO=True,
        truncate_at_tmin=True,
        waveforms_in_geom_units=True
    ):
        Generate_Offline_Surrogate.__init__(self, time_array=time_array, ecc_ref_parameterspace_range=ecc_ref_parameterspace_range, reference_total_mass=reference_total_mass, reference_luminosity_distance=reference_luminosity_distance, amount_input_wfs=amount_input_wfs, amount_output_wfs=amount_output_wfs, N_greedy_vecs_amp=N_greedy_vecs_amp, N_greedy_vecs_phase=N_greedy_vecs_phase, min_greedy_error_amp=min_greedy_error_amp, min_greedy_error_phase=min_greedy_error_phase, f_lower=f_lower, f_ref=f_ref, chi1=chi1, chi2=chi2, phiRef=phiRef, rel_anomaly=rel_anomaly, inclination=inclination, truncate_at_ISCO=truncate_at_ISCO, truncate_at_tmin=truncate_at_tmin, geometric_units=waveforms_in_geom_units)



    def create_offline_surrogate(self, plot_fits=False, save_fig_fits=False):
        """Load or fit surrogate model and save all necessary offline data."""

        # Try to load existing amplitude GPR fit data
        GPR_amp_data = self._load_gpr_data('amplitude', plot_fits=plot_fits, save_fig_fits=save_fig_fits)
        # Load corresponding B_matrices
        B_amp_data = self._load_b_matrix('amplitude')
        
        # Try to load existing phase GPR fit data
        GPR_phase_data = self._load_gpr_data('phase', plot_fits=plot_fits, save_fig_fits=save_fig_fits)
        # Try loading corresponding B_matrices
        B_phase_data = self._load_b_matrix('phase')



        # Save everything in one compressed file
        os.makedirs('Straindata/Offline_data', exist_ok=True)
        output_path = Path(
            f"Straindata/Offline_data/"
            f"Surrogate_OfflineData_"
            f"f_lower={self.f_lower}_"
            f"f_ref={self.f_ref}_"
            f"e=[{min(self.ecc_parameter_space_output)}_{max(self.ecc_parameter_space_output)}]_"
            f"Ni={self.amount_input_wfs}_"
            f"No={self.amount_output_wfs}_"
            f"gp={self.min_greedy_error_phase}_"
            f"ga={self.min_greedy_error_amp}_"
            f"Ngp={self.N_greedy_vecs_phase}_"
            f"Nga={self.N_greedy_vecs_amp}.npz"
        )

        np.savez_compressed(
            output_path,
            gaussian_fit_amp=GPR_amp_data['GPR_fit'],
            gaussian_fit_phase=GPR_phase_data['GPR_fit'],
            empirical_nodes_idx_amp=GPR_amp_data['empirical_nodes'],
            empirical_nodes_idx_phase=GPR_phase_data['empirical_nodes'],
            residual_greedy_basis_amp=GPR_amp_data['residual_greedy_basis'],
            residual_greedy_basis_phase=GPR_phase_data['residual_greedy_basis'],
            greedy_parameters_idx_amp=GPR_amp_data['greedy_parameters_idx'],
            greedy_parameters_idx_phase=GPR_phase_data['greedy_parameters_idx'],
            B_matrix_amp=B_amp_data['B_matrix'],
            B_matrix_phase=B_phase_data['B_matrix'],
            time=GPR_amp_data['time'],
            amp_circ=GPR_amp_data['amp_circ'],
            phase_circ=GPR_phase_data['phase_circ'],
        )
        print(f"Surrogate offline data saved to: {output_path}")


    def _gpr_filename(self, property):
        """Constructs a standardized filename for saved GPR fits."""
        return (
            f'Straindata/GPRfits/GPRfits_{property}_'
            f'f_lower={self.f_lower}_f_ref={self.f_ref}_'
            f'e=[{self.ecc_ref_parameterspace_range[0]}_{self.ecc_ref_parameterspace_range[1]}_N={self.amount_input_wfs}]_'
            f'No={self.amount_output_wfs}_'
            f'gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_'
            f'Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}.npz'
        )

    def _load_gpr_data(self, property, plot_fits=False, save_fig_fits=False):
        """Load GPR fit data for amplitude or phase."""
        filename = (
            f'Straindata/GPRfits/GPRfits_{property}_'
            f'f_lower={self.f_lower}_f_ref={self.f_ref}_'
            f'e=[{self.ecc_ref_parameterspace_range[0]}_{self.ecc_ref_parameterspace_range[1]}_N={self.amount_input_wfs}]_'
            f'No={self.amount_output_wfs}_'
            f'gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_'
            f'Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}.npz'
        )

        try:
            data = np.load(filename, allow_pickle=True)
            print(f'GPR fit data loaded: {filename}')

        except:
            self.fit_to_training_set(
                min_greedy_error=self.min_greedy_error_phase,
                N_greedy_vecs=self.N_greedy_vecs_phase,
                property=property,
                plot_fits=plot_fits,
                save_fig_fits=save_fig_fits,
                save_fits_to_file=True
            )


            data = np.load(filename, allow_pickle=True)

        return {
            'GPR_fit': data['GPR_fit'],
            'empirical_nodes': data['empirical_nodes'],
            'residual_greedy_basis': data['residual_greedy_basis'],
            'time': data['time'],
            'greedy_parameters_idx': data['greedy_parameters_idx'],
            'amp_circ': data['amp_circ'],
            'phase_circ': data['phase_circ']
        }
    
    def _load_b_matrix(self, property):
        """Try to load the B_matrix for a given property (amplitude or phase)."""
        filename = (
            f'Straindata/B_matrix/B_matrix_{property}_'
            f'f_lower={self.f_lower}_f_ref={self.f_ref}_'
            f'e=[{self.ecc_ref_parameterspace_range[0]}_{self.ecc_ref_parameterspace_range[1]}_N={self.amount_input_wfs}]_'
            f'No={self.amount_output_wfs}_'
            f'gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_'
            f'Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}.npz'
        )

        try:
            data = np.load(filename)
            print(f' B_matrix {property} loaded: {filename}')

        except FileNotFoundError:
            print(f'B_matrix file for {property} not found: {filename} .\n Calculate B_matrix...')
            
            data = self._load_gpr_data(property)
            self.residual_greedy_basis = data['residual_greedy_basis']
            self.empirical_nodes_idx = data['empirical_nodes']

            self.compute_B_matrix(
                    property=property,
                    save_matrix_to_file=True
                )
            data = np.load(filename)
        
        return {
            'B_matrix': data['B_matrix']
        }







# sampling_frequency = 2048 # or 4096
# duration = 6 # seconds
# time_array = np.linspace(-duration, 0, int(sampling_frequency * duration))  # time in seconds

# gs = Generate_Surrogate(time_array=time_array, output_ecc_ref=0.2, ecc_ref_parameterspace_range=[0, 0.2], total_mass_range=[60, 100], luminosity_distance_range=[200, 500], amount_input_wfs=80, amount_output_wfs=200, N_greedy_vecs_amp=40, N_greedy_vecs_phase=40)
# gs.fit_to_training_set(property='phase', N_greedy_vecs=40, plot_fits=True, save_fig_fits=True, save_fits_to_file=True, plot_residuals_time_evolve=True, plot_residuals_ecc_evolve=True)
# gs.fit_to_training_set(property='amplitude', N_greedy_vecs=40, plot_fits=True, save_fig_fits=True, save_fits_to_file=True, plot_residuals_ecc_evolve=True, plot_kernels=True)
# gs.generate_surrogate_model(plot_surr_datapiece=True, plot_surr_wf=True, plot_GPRfit=True, save_fig_datapiece=True, save_fig_surr=True, save_fits_to_file=True, save_surr_to_file=True, save_fig_fits=True)

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

# gs.generate_surrogate_model(plot_surr_datapiece=0.1905, plot_surr_wf=0.1905, plot_GPRfit=True)
# gs.generate_surrogate_model(plot_surr_datapiece=0.0336, plot_surr_wf=0.0336, plot_GPRfit=True)

# plt.show()
