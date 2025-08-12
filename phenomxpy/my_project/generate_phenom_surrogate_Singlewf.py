from fileinput import filename
from generate_phenom_training_set import *
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

f = currentframe()
plt.switch_backend('WebAgg')


faulthandler.enable()

class Generate_Surrogate(Generate_TrainingSet):

    def __init__(self, time_array, ecc_ref_parameterspace_range, amount_input_wfs, amount_output_wfs, total_mass_range=None, luminosity_distance_range=None, N_greedy_vecs_amp=None, N_greedy_vecs_phase=None, min_greedy_error_amp=None, min_greedy_error_phase=None, f_lower=10, f_ref=20, chi1=0, chi2=0, phiRef=0., rel_anomaly=0., inclination=0., truncate_at_ISCO=True, truncate_at_tmin=True, waveforms_in_geom_units=True):
        
        if (waveforms_in_geom_units is True) and ((total_mass_range is None) or (luminosity_distance_range is None)):
            print('Choose waveforms either in NR or SI units. Do this by either setting total_mass_range and luminosity_distance_range OR leave total_mass_range=luminosity_distance_range=None and set waveforms_in_geom_units=True.')
        if (N_greedy_vecs_amp is None and N_greedy_vecs_phase is None) and \
            (min_greedy_error_amp is None and min_greedy_error_phase is None):
                print('Choose either settings for the amount of greedy_vecs OR the minimum greedy error.')
                sys.exit(1)


        self.ecc_parameter_space_input = np.linspace(ecc_ref_parameterspace_range[0], ecc_ref_parameterspace_range[1], amount_input_wfs).round(4)
        self.ecc_parameter_space_output = np.linspace(ecc_ref_parameterspace_range[0], ecc_ref_parameterspace_range[1], amount_output_wfs).round(4)
        
        self.min_greedy_error_amp = min_greedy_error_amp
        self.min_greedy_error_phase = min_greedy_error_phase
        self.N_greedy_vecs_amp = N_greedy_vecs_amp
        self.N_greedy_vecs_phase = N_greedy_vecs_phase
        self.reference_total_mass = total_mass_range[0]
        self.reference_luminosity_distance = luminosity_distance_range[0]
        self.total_mass_range = total_mass_range
        self.luminosity_distance_range = luminosity_distance_range
        self.surrogate_amp = None
        self.surrogate_phase = None
        self.waveforms_in_geom_units = waveforms_in_geom_units

        self.gaussian_fit_amp = None
        self.gaussian_fit_phase = None

        
        Generate_TrainingSet.__init__(self, time_array, self.ecc_parameter_space_input, self.reference_total_mass, self.reference_luminosity_distance, f_lower, f_ref, chi1, chi2, phiRef, rel_anomaly, inclination, truncate_at_ISCO, truncate_at_tmin)

    
    def simulate_inspiral_mass_dependent(self, total_mass, distance, ecc_ref=None, plot_polarisations=False, save_fig=False, truncate_at_ISCO=False):
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
            times=self.time,
            eccentricity=ecc_ref,  
            total_mass=self.total_mass,
            distance=self.luminosity_distance,                
            f_ref=f_ref_SI,                   
            f_lower=f_lower_SI,
            phiRef=self.phiRef,
            inclination=self.inclination)
        
        phen.compute_polarizations(times=self.time, distance=distance, total_mass=total_mass)

        
        if phen.pWF.tmin > self.time[0]:
            warnings.warn(
                "t_min is larger than parts of the specified time-domain, resulting in unphysical waveforms. "
                "Either use the truncate_tmin=True setting to automatically truncate to start from t_min=time_array[0] "
                "or adjust the time-array manually to start at higher values."
            )
            # mask to only include the physical range of the time-domain
            if self.truncate_at_tmin is True:
                mask = self.time >= phen.pWF.tmin

                self.time = self.time[mask]
                phen.hp = phen.hp[mask]
                phen.hc = phen.hc[mask]

                print(f'NEW TIME-DOMAIN (in geometric units): [{int(self.time[0])}, {int(self.time[-1])}] M')
                del mask # clear memory

        # True because it's smallest truncated waveform AND true because the surrogate is called with the ISCO cut-off.
        if (truncate_at_ISCO is True) and (self.truncate_at_ISCO is True):
            # Truncate the waveform at ISCO frequency
            idx_cut = self.truncate_waveform_at_isco(phen)
            self.time = self.time[:idx_cut]
            

        print(f'time : SimInspiral_M_independent ecc = {round(ecc_ref, 3)}, M = {self.total_mass}, t=[{int(self.time[0])}, {int(self.time[-1])}, num={len(self.time)}] | computation time = {(timer()-start)} seconds')

        if plot_polarisations is True:

            fig_simulate_inspiral = plt.figure(figsize=(12,5))
            print(len(self.time), len(phen.hp))
            plt.plot(self.time[:len(phen.hp)], phen.hp, label = f'$h_+$', linewidth=0.6)
            plt.plot(self.time[:len(phen.hp)], phen.hc, label = f'$h_\times$', linewidth=0.6)

            plt.legend(loc = 'upper left')
            plt.xlabel('t [s]')
            plt.ylabel('$h_{22}]$')
            plt.title(f'M={self.total_mass}, e={round(ecc_ref, 3)}, f_min={self.f_lower} Hz')
            plt.grid(True)

            plt.tight_layout()

            if save_fig is True:
                figname = 'Polarisations_M={}_ecc={}.png'.format(self.total_mass, round(ecc_ref, 3))
                
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Polarisations', exist_ok=True)
                fig_simulate_inspiral.savefig('Images/Polarisations/' + figname, dpi=300, bbox_inches='tight')

                print('Figure is saved in Images/Polarisations/' + figname)

            plt.close('all')  # Clean up plots

        return phen.hp, phen.hc
    
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
            filename = f'Straindata/Residuals/residuals_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_parameter_space_input)}_{max(self.ecc_parameter_space_input)}_N={len(self.ecc_parameter_space_input)}].npz'
            load_parameterspace_input = np.load(filename)
            
            residual_parameterspace_input = load_parameterspace_input['residual']
            self.time = load_parameterspace_input['time']

            try:
                filename = f'Straindata/Residuals/residuals_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_parameter_space_output)}_{max(self.ecc_parameter_space_output)}_N={len(self.ecc_parameter_space_output)}].npz'
                load_residual_output = np.load(filename)
                residual_parameterspace_output = load_residual_output['residual']

            except Exception as e:
                print(f'line {getframeinfo(f).lineno}: {e}')
                residual_parameterspace_output = self.generate_property_dataset(ecc_list=self.ecc_parameter_space_output, property=property, save_dataset_to_file=True)
            
            # print(0, residual_parameterspace_output.shape, len(self.time))
            # fig_test = plt.figure()
            # for i in range(len(self.empirical_nodes_idx)):
            #     # print( residual_parameterspace_output.T[i])
            #     print('emp nodes: ', self.empirical_nodes_idx[i])
            # for i in range(len(self.time)):
            #     plt.plot(self.ecc_parameter_space_output, residual_parameterspace_output.T[i])
            # plt.plot(self.time, residual_parameter)
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

        filename = f'Straindata/GPRfits/GPRfits_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_parameter_space_input)}_{max(self.ecc_parameter_space_input)}_N={len(self.ecc_parameter_space_input)}]_No={len(self.ecc_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}.npz'
        if save_fits_to_file is True and not os.path.isfile(filename):

            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Straindata/GPRfits', exist_ok=True)
            print('circs:', self.amp_circ, self.phase_circ)
            np.savez(filename, GPR_fit=gaussian_fit, empirical_nodes=self.empirical_nodes_idx, residual_greedy_basis=self.residual_greedy_basis, time=self.time, lml_fits=lml_fits, training_set=training_set, greedy_parameters_idx=self.greedy_parameters_idx, uncertainty_region=np.array(uncertainty_region, dtype=object), phase_circ=self.phase_circ, amp_circ=self.amp_circ)
            print('GPR fits saved in Straindata/GPRfits/' + filename)
        
        if (plot_residuals_time_evolve is True) or (plot_residuals_time_evolve is True):
            load_parameterspace_input = np.load(f'Straindata/Residuals/residuals_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_parameter_space_input)}_{max(self.ecc_parameter_space_input)}_N={len(self.ecc_parameter_space_input)}].npz')
            residual_parameterspace_input = load_parameterspace_input['residual']
            
            self._plot_residuals(residual_dataset=residual_parameterspace_input, ecc_list=self.ecc_parameter_space_input, property=property, plot_eccentric_evolv=plot_residuals_ecc_evolve, save_fig_eccentric_evolve=save_fig_ecc_evolve, plot_time_evolve=plot_residuals_time_evolve, save_fig_time_evolve=save_fig_time_evolve)
        
        #     fig_residual_training_fit, axs = plt.subplots(2, 1, figsize=(11,6), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.4}, sharex=True)

        #     # Top left subplot for amplitude main plot
        #     for i in range(len(gaussian_fit[:3])):
        #         # Use the same color for the fit and the corresponding empirical data
        #         axs[0].plot(self.ecc_parameter_space_output, gaussian_fit.T[:, i], color=colors[i], linewidth=0.6)
        #         axs[0].scatter(self.ecc_parameter_space_input[self.greedy_parameters_idx], training_set[:, i], color=colors[i])
        #         # axs[0].plot(self.ecc_parameter_space_output, residual_parameterspace_output[:, self.empirical_nodes_idx[i]], linestyle='dashed', linewidth=0.6, color=colors[i], label=f't={self.time[self.empirical_nodes_idx[i]]}')
        #         axs[0].plot(self.ecc_parameter_space_output, residual_parameterspace_output[:, self.empirical_nodes_idx[i]], linestyle='dashed', linewidth=0.6, color=colors[i])

        #         # axs[0].plot(self.ecc_parameter_space_output, residual_parameterspace_output[:, -1], linestyle='dashed', linewidth=0.6, color=colors[i], label=f't={self.time[self.empirical_nodes_idx[i]]}')
        #         # axs[0].plot(self.ecc_parameter_space_input, residual_parameterspace_input[:, -1])
        #         relative_error = abs(residual_parameterspace_output[:, self.empirical_nodes_idx[i]] - gaussian_fit.T[:, i]) / abs(residual_parameterspace_output[:, self.empirical_nodes_idx[i]])
        #         axs[1].plot(self.ecc_parameter_space_output, relative_error, color=colors[i], linewidth=0.6, label=f'Error {i+1} (t={int(self.time[self.empirical_nodes_idx[i]])})')
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
        #         figname = f'GPR_fits_{property}_ecc=[{min(self.ecc_parameter_space_input)}_{max(self.ecc_parameter_space_input)}]_q={self.mass_ratio}_fmin={self.freqmin}_iN={len(self.ecc_parameter_space_input)}_oN={len(self.ecc_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}.png'
                
        #         # Ensure the directory exists, creating it if necessary and save
        #         os.makedirs('Images/Gaussian_fits', exist_ok=True)
        #         fig_residual_training_fit.savefig('Images/Gaussian_fits/' + figname)

        #         print('Figure is saved in Images/Gaussian_fits')
        
        # if save_fits_to_file is True and not os.path.isfile(f'Straindata/GPRfits/{property}_q={self.mass_ratio}_fmin={self.freqmin}_{min(self.ecc_parameter_space_input)}_{max(self.ecc_parameter_space_input)}_Ni={len(self.ecc_parameter_space_input)}_No={len(self.ecc_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}_size={self.waveform_size}.npz'):

        #     # Ensure the directory exists, creating it if necessary and save
        #     os.makedirs('Straindata/GPRfits', exist_ok=True)
        #     np.savez(f'Straindata/GPRfits/{property}_q={self.mass_ratio}_fmin={self.freqmin}_{min(self.ecc_parameter_space_input)}_{max(self.ecc_parameter_space_input)}_Ni={len(self.ecc_parameter_space_input)}_No={len(self.ecc_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}_size={self.waveform_size}.npz', GPR_fit=gaussian_fit, training_set=training_set, uncertainty_region=np.array(uncertainty_region, dtype=object), greedy_parameters=self.greedy_parameters_idx, empirical_nodes=self.empirical_nodes_idx, residual_greedy_basis=self.residual_greedy_basis)
        #     print('GPR fits saved in Straindata/GPRfits')

        return gaussian_fit, uncertainty_region

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
    
    def residual_to_original(self, residual_waveform, property):
        """
        Converts the residual waveform back to the original waveform by adding or subtracting the circular waveform depending on the property.
        """
        self.circulair_wf()  # ensure circular wf is updated
        
        if property == 'phase':
            circ = self.phase_circ
            original_waveform = circ - residual_waveform
        elif property == 'amplitude':
            circ = self.amp_circ
            original_waveform = residual_waveform + circ
        else:
            raise ValueError('property must be "phase" or "amplitude"')
        
        return original_waveform

    def reconstruct_surrogate_datapiece(self, property, B_matrix, fit_matrix, plot_surr_datapiece=True, save_fig_datapiece=False):
        """
        Reconstructs the surrogate model for a given parameter using different empirical nodes for amplitude and phase.
        
        Parameters:
        ------------------
        B_matrix (numpy.ndarray), shape (m, time_samples): Empricial interpolant matrix
        fit_matrix (numpy.ndarray), shape (m, lambda): Array of fitted greedy parameters at time nodes with lambda as the number of parameters in parameter_space.
        time_samples (numpy.ndarray), shape (time_samples, 1): Array representing the time-domain samples.
        plot_surr_datapiece (bool) : Set this to True for plot of surrogate datapiece as comparison with real estimated value at given output_ecc_ref.
        
        Returns:
        ------------------
        surrogate_datapiece (numpy.ndarray), shape (time_samples, lambda): Array representing the reconstructed surrogate waveform datapiece (amplitude or phase).
        """
        
        computation_time = None
        m, _ = B_matrix.shape

        # fit_vector2 = np.zeros(m)

        # for i in range(m):
        #     fit_vector2[i] = fit_matrix[i].predict(self.ecc_parameter_space_output)[0]


        fit_vector = fit_matrix.T[self.output_ecc_ref_idx]  # Get the fit vector for the specific output eccentricity reference
        reconstructed_residual = np.sum(B_matrix * fit_vector[:, None], axis=0)

        # Change back from residual to original (+ circulair)
        surrogate_datapiece = self.residual_to_original(residual_waveform=reconstructed_residual, property=property)
        
        if plot_surr_datapiece is True:
            print(0)
            # Create a 2x1 subplot grid with height ratios 3:1
            fig_surrogate_datapieces, axs = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1}, sharex=True)

            # Simulate the real waveform datapiece
            real_hp, real_hc = self.simulate_inspiral_mass_independent(self.output_ecc_ref)

            if property == 'amplitude':
                real_datapiece = self.amplitude(real_hp, real_hc)
                units = ''
            elif property == 'phase':
                real_datapiece = self.phase(real_hp, real_hc)
                units = ' [radians]'
            print('real_datapiece, surrogate_datapiece: ', real_datapiece, surrogate_datapiece)
            # Plot Surrogate and Real Amplitude (Top Left)
            # axs[0].plot(self.ecc_parameter_space_output, surrogate_datapiece[index_ecc_ref], label='surr')
            axs[0].plot(self.time, surrogate_datapiece, linewidth=0.6, label=f'surrogate e = {plot_surr_datapiece}')
            # axs[0].plot(self.time, true_phase[index_ecc_ref], linewidth=0.6, label=f'Surrogate: e = {plot_surr_datapiece}')
            axs[0].plot(self.time, real_datapiece, linewidth=0.6, linestyle='dashed', label=f'true {property} e = {plot_surr_datapiece}')
            # axs[0].plot(self.ecc_parameter_space_output, true_phase[:, index_ecc_ref], label='real')
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
            relative_error = abs(surrogate_datapiece - real_datapiece) / abs(real_datapiece)
            axs[1].plot(self.time, relative_error, linewidth=0.6)
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
                figname = f'Surrogate_{property}_ecc_ref={plot_surr_datapiece}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_parameter_space_input)}_{max(self.ecc_parameter_space_input)}_Ni={len(self.ecc_parameter_space_input)}]_No={len(self.ecc_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}.png'
                
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Surrogate_datapieces_Single', exist_ok=True)
                fig_surrogate_datapieces.savefig('Images/Surrogate_datapieces_Single/' + figname)

                print('Figure is saved in Images/Surrogate_datapieces_Single/' + figname)

        return surrogate_datapiece, computation_time

    def generate_surrogate_waveform(self, output_ecc_ref, plot_surr_datapiece=None, save_fig_datapiece=False, plot_surr_wf=None, save_fig_surr=False, plot_GPRfit=False, save_fits_to_file=True, save_fig_fits=False, save_matrix_to_file=True):

        if isinstance(output_ecc_ref, float):
            try:
                self.output_ecc_ref_idx = np.where(self.ecc_parameter_space_output == output_ecc_ref)[0][0]
                self.output_ecc_ref = output_ecc_ref
            except:
                output_ecc_ref_asked = output_ecc_ref
                self.output_ecc_ref = self.ecc_parameter_space_output[np.abs(self.ecc_parameter_space_output - output_ecc_ref).argmin()]
                self.output_ecc_ref_idx = np.where(self.ecc_parameter_space_output == self.output_ecc_ref)[0][0]
                print(f'Eccentricity value for output_ecc_ref={output_ecc_ref_asked} not in ouput parameterspace. Eccentricity calculated for closest existing value at e={self.output_ecc_ref}.')

        
        if self.gaussian_fit_amp is None:
            print('Loading surrogate amplitude...')
            # Set timer for computational time of the surrogate model
            # start_time_amp = time.time()

            # Get matrix with interpolated fits and B_matrix
            self.gaussian_fit_amp = self.fit_to_training_set(min_greedy_error=self.min_greedy_error_amp, N_greedy_vecs=self.N_greedy_vecs_amp, property='amplitude', plot_fits=plot_GPRfit, save_fig_fits=save_fig_fits, save_fits_to_file=save_fits_to_file)[0]
            # Get empirical nodes for amplitude
            self.empirical_nodes_idx_amp = self.empirical_nodes_idx
            # Get residual greedy basis of amplitude
            self.residual_greedy_basis_amp = self.residual_greedy_basis
            
        if self.B_matrix_amp is None:
            # Get B_matrix for amplitude
            self.B_matrix_amp = self.compute_B_matrix(property='amplitude', save_matrix_to_file=save_matrix_to_file)
            print('B_matrix: ', self.B_matrix_amp)
            # Reconstruct amplitude datapiece
            self.surrogate_amp, computation_time_amp = self.reconstruct_surrogate_datapiece(property='amplitude', B_matrix=self.B_matrix_amp, fit_matrix=self.gaussian_fit_amp, plot_surr_datapiece=plot_surr_datapiece, save_fig_datapiece=save_fig_datapiece)

        else:
            print('Reconstruct surrogate datapiece...')
            self.surrogate_amp, computation_time_amp = self.reconstruct_surrogate_datapiece(property='amplitude', B_matrix=self.B_matrix_amp, fit_matrix=self.gaussian_fit_amp, plot_surr_datapiece=plot_surr_datapiece, save_fig_datapiece=save_fig_datapiece)

        # # End timer for computation of surrogate model
        # end_time_amp = time.time()


        if self.gaussian_fit_phase is None:
            print('Loading surrogate phase...')
            # Set timer for computational time of the surrogate model
            # start_time_phase = time.time()

            # Get matrix with interpolated fits and B_matrix
            start1 = time.time()
            self.gaussian_fit_phase = self.fit_to_training_set(min_greedy_error=self.min_greedy_error_phase, N_greedy_vecs=self.N_greedy_vecs_phase, property='phase', plot_fits=plot_GPRfit, save_fig_fits=save_fig_fits, save_fits_to_file=save_fits_to_file)[0]
            print(f'GPR fit phase took {time.time() - start1:.4f}s')
            # Get empirical nodes of phase
            start2 = time.time()
            self.empirical_nodes_idx_phase = self.empirical_nodes_idx
            # Get residual greedy basis of phase
            self.residual_greedy_basis_phase = self.residual_greedy_basis
            
            print(f'Setting self took {time.time() - start2:.4f}s')
            start3 = time.time()

        if self.B_matrix_phase is None:
            # Get B_matrix for phase
            self.B_matrix_phase = self.compute_B_matrix(property='phase', save_matrix_to_file=save_matrix_to_file)
            print(f'B_matrix took {time.time() - start3:.4f}s')
            print('B_matrix: ', self.B_matrix_phase)
            # Reconstruct phase datapiece
            self.surrogate_phase, computation_time_phase = self.reconstruct_surrogate_datapiece(property='phase', B_matrix=self.B_matrix_phase, fit_matrix=self.gaussian_fit_phase, plot_surr_datapiece=plot_surr_datapiece)
        else:
            self.surrogate_phase, computation_time_phase = self.reconstruct_surrogate_datapiece(property='phase', B_matrix=self.B_matrix_phase, fit_matrix=self.gaussian_fit_phase, plot_surr_datapiece=plot_surr_datapiece)
        
        # # End timer for computation of surrogate model
        # end_time_phase = time.time()

        # # Compute total computational time of the surrogate datapieces
        # if computation_time_phase is None:
        #     computation_time_amp = end_time_amp - start_time_amp
        #     computation_time_phase = end_time_phase - start_time_phase

        # filename = f'Straindata/Surrogate_datapieces/Surrogate_datapieces_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_parameter_space_input)}_{max(self.ecc_parameter_space_input)}_N={len(self.ecc_parameter_space_input)}]_No={len(self.ecc_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}.npz'
        # if save_surr_to_file is True and not os.path.isfile(filename):
        #     # Ensure the directory exists, creating it if necessary and save
        #     os.makedirs('Straindata/Surrogate_datapieces', exist_ok=True)
        #     np.savez(filename, surrogate_amp=self.surrogate_amp, surrogate_phase=self.surrogate_phase, computation_t_amp=computation_time_amp, computation_t_phase=computation_time_phase, time=self.time)
        #     print('Surrogate datapieces saved in ' + filename)

        
        if self.waveforms_in_geom_units is False:
            # Convert mass-independent waveforms to a 3 dimensional mass-dependent grid of (total_mass x ecc_ref x time)
            surrogate_amp_SI, surrogate_phase_SI = self.surrogate_datapieces_from_NR_to_SI()

            h_surrogate = surrogate_amp_SI * np.exp(1j * surrogate_phase_SI)

        else:
            h_surrogate = self.surrogate_amp * np.exp(1j * self.surrogate_phase)


        if plot_surr_wf is True:
            # Plot surrogate waveform
            fig_surrogate, axs = plt.subplots(4, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1, 3, 1], 'hspace': 0.2}, sharex=True)

            if self.waveforms_in_geom_units is True:
                true_hp, true_hc = self.simulate_inspiral_mass_independent(self.output_ecc_ref)


            phase = self.phase(true_hp, true_hc)
            amp = self.amplitude(true_hp, true_hc)
            true_h = amp * np.exp(1j * phase)


            axs[0].plot(self.time, np.real(true_h), linewidth=0.6, label=f'true waveform e = {plot_surr_wf}')
            axs[0].plot(self.time, np.real(h_surrogate), linewidth=0.6, label=f'surrogate e = {plot_surr_wf}')
            axs[0].set_ylabel('$h_+$')
            axs[0].grid(True)
            axs[0].legend()

            # Calculate and Plot plus polarisation error 
            relative_error_hp = abs(np.real(h_surrogate) - np.real(true_h)) / abs(np.real(true_h))
            relative_error_hp[relative_error_hp > 1] = 0

            axs[1].plot(self.time, abs(np.real(h_surrogate) - np.real(true_h)), linewidth=0.6)
            axs[1].set_ylabel('|$h_{+, S} - h_+$|')
            axs[1].grid(True)
            # axs[1].set_ylim(0, 10)
            # axs[1].set_title('Relative error $h_x$')

            # axs[2].plot(self.time, true_hc[length_diff:], linewidth=0.6, label='True waveform before')
            axs[2].plot(self.time, np.imag(true_h), linewidth=0.6, label=f'true waveform e = {plot_surr_wf}')
            axs[2].plot(self.time, np.imag(h_surrogate), linewidth=0.6, label=f'surrogate e = {plot_surr_wf}')
            axs[2].grid(True)
            axs[2].set_ylabel('$h_x$')
            axs[2].legend()

            # # axs[2].plot(self.time, true_hc[length_diff:], linewidth=0.6, label='True waveform before')
            # axs[1].plot(self.time, np.imag(true_h)[length_diff:], linewidth=0.6, label='True waveform after')
            # axs[1].plot(self.time, np.imag(h_surrogate[:, index_ecc_ref]), linewidth=0.6, label='Surrogate')
            # axs[1].grid(True)
            # axs[1].set_ylabel('$h_x$')
            # axs[1].legend()

            # Calculate and Plot cross polarisation error
            relative_error_hc = abs(np.imag(h_surrogate) - np.imag(true_h)) / abs(np.imag(true_h))
            relative_error_hc[relative_error_hc > 1] = 0
            axs[3].plot(self.time, abs(np.imag(h_surrogate) - np.imag(true_h)), linewidth=0.6)
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
                figname = f'Surrogate_wf_ecc_ref={plot_surr_wf}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_parameter_space_input)}_{max(self.ecc_parameter_space_input)}_Ni={len(self.ecc_parameter_space_input)}]_No={len(self.ecc_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}.png'
                
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Surrogate_wf', exist_ok=True)
                fig_surrogate.savefig('Images/Surrogate_wf/' + figname)

                print('Figure is saved in Images/Surrogate_wf/' + figname)

        return self.surrogate_amp, self.surrogate_phase
    
    def surrogate_datapieces_from_NR_to_SI(self):
        # Phase is already unitless so doesn't need converting
        surrogate_amp_SI = np.zeros((len(self.total_mass_range), len(self.ecc_parameter_space_output), len(self.time))) #
        surrogate_phase_SI = np.zeros((len(self.total_mass_range), len(self.ecc_parameter_space_output), len(self.time))) #

        for total_mass, distance in zip(self.total_mass_range, self.luminosity_distance_range):
            self.time = MasstoSecond(self.time, total_mass)
            for ecc_ref in self.ecc_parameter_space_output:
                surrogate_amp_SI[total_mass, ecc_ref, :] = AmpNRtoSI(self.surrogate_amp.T[ecc_ref], distance, total_mass)
                surrogate_phase_SI[total_mass, ecc_ref, :] = self.surrogate_phase.t[ecc_ref]


        return surrogate_amp_SI, surrogate_phase_SI





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
