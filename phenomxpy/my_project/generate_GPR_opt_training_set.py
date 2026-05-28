from phenomxpy.my_project.fit_training_set import *

from inspect import getframeinfo


plt.switch_backend('WebAgg')

class Generate_TrainingSet_GPR_Opt(Generate_Offline_Surrogate):
    """
    Class to generate a training dataset for gravitational waveform simulations using a greedy algorithm and empirical interpolation.
    Inherits from WaveformProperties and SimulateInspiral to leverage methods for waveform 
    property calculations and waveform generation.

    """

    def __init__(
        self,
        time_array,
        mass_ratio_range, 
        ecc_ref_range, 
        mean_ano_ref_range, 
        chi1_range, 
        chi2_range, 
        amount_input_wfs,
        amount_output_wfs,
        minimum_spacing_greedy=0.008,
        f_lower=10,
        f_ref=20,
        phiRef=0.,
        inclination=0.,
        truncate_at_ISCO=True,
        truncate_at_tmin=True
    ):
        
        super().__init__(
            time_array=time_array, 
            mass_ratio_range=mass_ratio_range, 
            ecc_ref_range=ecc_ref_range, 
            mean_ano_ref_range=mean_ano_ref_range, 
            chi1_range=chi1_range, 
            chi2_range=chi2_range, 
            f_lower=f_lower, 
            f_ref=f_ref, 
            phiRef=phiRef, 
            inclination=inclination, 
            amount_input_wfs=amount_input_wfs, 
            amount_output_wfs=amount_output_wfs, 
            N_basis_vecs_amp=None, 
            N_basis_vecs_phase=None, 
            min_greedy_error_amp=None, 
            min_greedy_error_phase=None, 
            training_set_selection='GPR_opt', 
            minimum_spacing_greedy=minimum_spacing_greedy, 
            truncate_at_ISCO=truncate_at_ISCO, 
            truncate_at_tmin=truncate_at_tmin
            )


    def fit_to_training_set_GPR_opt(self, 
                                    property, 
                                    N_GPR_basis_vecs, 
                                    min_greedy_error=None, 
                                    N_basis_vecs=None, 
                                    N_ini_greedy_vecs=3,
                                    training_set=None, 
                                    X_train=None, 
                                    save_fits_to_file=True, 
                                    plot_kernels=False, save_fig_kernels=False,
                                    plot_GPR_fits=False, save_fig_GPR_fits=False,
                                    plot_residuals_ecc_evolve=False, save_fig_ecc_evolve=False, 
                                    plot_residuals_time_evolve=False, save_fig_time_evolve=False,
                                    plot_greedy_vecs=False, save_fig_greedy_vecs=False, 
                                    plot_greedy_error=False, save_fig_greedy_error=False, 
                                    plot_emp_nodes_at_ecc=False, save_fig_emp_nodes=False, 
                                    plot_training_set=False, save_fig_training_set=False
                                    ):
        
        # training object for the chosen property (phase or amplitude)
        gpr_obj = self._get_gpr_obj(property)
        train_obj = self._get_training_obj(property)

        # Get first 3 points to produce a start for GPR
        residual_training_set = self.get_training_set_greedy(property, 
                                                             N_greedy_vecs=N_ini_greedy_vecs, 
                                                             emp_nodes_of_full_dataset=True, 
                                                             plot_greedy_error=plot_greedy_error, 
                                                             save_fig_greedy_error=save_fig_greedy_error, 
                                                             plot_emp_nodes_at_ecc=plot_emp_nodes_at_ecc, save_fig_emp_nodes=save_fig_emp_nodes, 
                                                             plot_training_set=plot_training_set, save_fig_training_set=save_fig_training_set, 
                        save_dataset_to_file=True, plot_greedy_vecs=plot_greedy_vecs, save_fig_greedy_vecs=save_fig_greedy_vecs)

        while len(self.residual_reduced_basis) <= N_GPR_basis_vecs:
            # Update the length of the basis for every iteration
            if property == 'phase':
                self.N_basis_vecs_phase = len(residual_training_set)
            else:
                self.N_basis_vecs_amp = len(residual_training_set)
            
            try:
                start = time.time()

                filename = f'Straindata/GPRfits/GPRfits_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_ref_space)}_{max(self.ecc_ref_space)}_N={self.amount_input_wfs}]_No={self.amount_output_wfs}_g_err={min_greedy_error}_Ng_vecs={N_basis_vecs}_min_s={self.minimum_spacing_greedy}.npz'
                load_GPRfits = np.load(filename, allow_pickle=True)
                
                gaussian_fit = load_GPRfits['GPR_fit']
                self.empirical_nodes_idx = load_GPRfits['empirical_nodes']
                self.residual_reduced_basis = load_GPRfits['residual_reduced_basis']
                self.time = load_GPRfits['time']
                self.amp_circ = load_GPRfits['amp_circ']
                self.phase_circ = load_GPRfits['phase_circ']
                lml_fits = load_GPRfits['lml_fits']
                training_set = load_GPRfits['training_set']
                self.indices_basis = load_GPRfits['best_rep_parameters_idx']
                self.best_rep_parameters = load_GPRfits['best_rep_parameters']
                uncertainty_region = load_GPRfits['uncertainty_region'].tolist()

                
                print(f'GPRfit {property} load succeeded: {time.time() - start:.4f}s')
            except Exception as e:
                print(e)
                traceback.print_exc()

                # Fit the basis vecs with GPR
                gaussian_fit, uncertainty_region = self.fit_to_training_set(property, training_set=residual_training_set, save_fits_to_file=True, plot_kernels=plot_kernels, plot_GPR_fits=plot_GPR_fits, save_fig_kernels=save_fig_kernels, save_fig_GPR_fits=save_fig_GPR_fits, plot_residuals_ecc_evolve=plot_residuals_ecc_evolve, save_fig_ecc_evolve=save_fig_ecc_evolve, plot_residuals_time_evolve=plot_residuals_time_evolve, save_fig_time_evolve=save_fig_time_evolve)

            # Load in property residuals of full parameter space dataset
            try:
                filename = f'Straindata/Residuals/residuals_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_ref_parameter_space_output)}_{max(self.ecc_ref_parameter_space_output)}_N={len(self.ecc_ref_parameter_space_output)}].npz'
                with np.load(filename) as data:
                    residual_parameterspace_output = data['residual']
                    self.time = data['time']

            except Exception as e:
                print(e)
                traceback.print_exc()

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
            self.indices_basis.append(worst_relative_GPR_error_idx)
            self.best_rep_parameters.append(self.ecc_ref_parameter_space_output[worst_relative_GPR_error_idx])

            #Generate the training set at empirical nodes for next GPR iteration
            residual_training_set = self.residual_reduced_basis[:, self.empirical_nodes_idx]


sampling_frequency = 2048 # or 4096
duration = 4 # seconds
time_array = np.linspace(-duration, 0, int(sampling_frequency * duration))  # time in seconds

gt = Generate_TrainingSet_GPR_Opt(time_array=time_array, ecc_ref_parameterspace_range=[0.0,0.3], amount_input_wfs=40, amount_output_wfs=500, N_basis_vecs_amp=25,
                                  N_basis_vecs_phase=25)

gt.fit_to_training_set_GPR_opt('phase', N_GPR_basis_vecs=25, plot_GPR_fits=True, save_fig_GPR_fits=True, plot_emp_nodes_at_ecc=0.1, save_fig_emp_nodes=True)
gt.fit_to_training_set_GPR_opt('amplitude', N_GPR_basis_vecs=25, plot_GPR_fits=True, save_fig_GPR_fits=True, plot_emp_nodes_at_ecc=0.1, save_fig_emp_nodes=True)
