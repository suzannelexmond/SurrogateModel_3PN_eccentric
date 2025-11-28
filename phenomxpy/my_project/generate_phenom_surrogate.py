
from fit_training_set import *

import faulthandler
from pathlib import Path
from inspect import currentframe

f = currentframe()
plt.switch_backend('WebAgg')


faulthandler.enable()





class Load_Offline_Surrogate(Generate_Offline_Surrogate):
    def __init__(
        self,
        time_array,
        ecc_ref_parameterspace_range,
        amount_input_wfs,
        amount_output_wfs,
        N_basis_vecs_amp=None,
        N_basis_vecs_phase=None,
        min_greedy_error_amp=None,
        min_greedy_error_phase=None,
        training_set_selection='GPR_opt',
        minimum_spacing_greedy=0.008,
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

        Generate_Offline_Surrogate.__init__(
        self, 
        time_array=time_array, 
        ecc_ref_parameterspace_range=ecc_ref_parameterspace_range, 
        amount_input_wfs=amount_input_wfs, 
        amount_output_wfs=amount_output_wfs, 
        N_basis_vecs_amp=N_basis_vecs_amp, 
        N_basis_vecs_phase=N_basis_vecs_phase, 
        min_greedy_error_amp=min_greedy_error_amp, 
        min_greedy_error_phase=min_greedy_error_phase, 
        training_set_selection=training_set_selection,
        minimum_spacing_greedy=minimum_spacing_greedy, 
        f_lower=f_lower, 
        f_ref=f_ref, 
        chi1=chi1, 
        chi2=chi2, 
        phiRef=phiRef, 
        rel_anomaly=rel_anomaly, 
        inclination=inclination, 
        truncate_at_ISCO=truncate_at_ISCO, 
        truncate_at_tmin=truncate_at_tmin, 
        geometric_units=waveforms_in_geom_units
        )



        # Generate_TrainingSet_GPR_Opt.__init__(
        # self,
        # time_array,
        # ecc_ref_parameterspace_range=ecc_ref_parameterspace_range,
        # amount_input_wfs=amount_input_wfs,
        # amount_output_wfs=amount_output_wfs,
        # N_basis_vecs_amp=N_basis_vecs_amp,
        # N_basis_vecs_phase=N_basis_vecs_phase,
        # min_greedy_error_amp=min_greedy_error_amp,
        # min_greedy_error_phase=min_greedy_error_phase,
        # minimum_spacing_greedy=minimum_spacing_greedy,
        # f_lower=f_lower,
        # f_ref=f_ref,
        # chi1=chi1,
        # chi2=chi2,
        # phiRef=phiRef,
        # rel_anomaly=rel_anomaly,
        # inclination=inclination,
        # truncate_at_ISCO=truncate_at_ISCO,
        # truncate_at_tmin=truncate_at_tmin,
        # geometric_units=waveforms_in_geom_units
        # )


    def create_offline_surrogate(self, training_set_selection='GPR_opt', plot_fits=False, save_fig_fits=False):
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
            f"{self.training_set_selection}_"
            f"f_lower={self.f_lower}_"
            f"f_ref={self.f_ref}_"
            f"e=[{min(self.ecc_ref_parameter_space_output)}_{max(self.ecc_ref_parameter_space_output)}]_"
            f"Ni={self.amount_input_wfs}_"
            f"No={self.amount_output_wfs}_"
            f"gp={self.min_greedy_error_phase}_"
            f"ga={self.min_greedy_error_amp}_"
            f"Ngp={self.N_basis_vecs_phase}_"
            f"Nga={self.N_basis_vecs_amp}_"
            f"min_s={self.minimum_spacing_greedy}.npz"
        )       

        np.savez_compressed(
            output_path,
            gaussian_fit_amp=GPR_amp_data['GPR_fit'],
            gaussian_fit_phase=GPR_phase_data['GPR_fit'],
            empirical_nodes_idx_amp=GPR_amp_data['empirical_nodes'],
            empirical_nodes_idx_phase=GPR_phase_data['empirical_nodes'],
            residual_reduced_basis_amp=GPR_amp_data['residual_reduced_basis'],
            residual_reduced_basis_phase=GPR_phase_data['residual_reduced_basis'],
            best_rep_parameters_idx_amp=GPR_amp_data['best_rep_parameters_idx'],
            best_rep_parameters_idx_phase=GPR_phase_data['best_rep_parameters_idx'],
            best_rep_parameters_amp=GPR_amp_data['best_rep_parameters'],
            best_rep_parameters_phase=GPR_phase_data['best_rep_parameters'],
            B_matrix_amp=B_amp_data['B_matrix'],
            B_matrix_phase=B_phase_data['B_matrix'],
            time=GPR_amp_data['time'],
            amp_circ=GPR_amp_data['amp_circ'],
            phase_circ=GPR_phase_data['phase_circ'],
        )
        print(f"Surrogate offline data saved to: {output_path}")

    def _load_gpr_data(self, property, plot_fits=False, save_fig_fits=False):
        """ Load GPR fit data for amplitude or phase.
            
            training_set_selection: choose the way the best estimated training set gets selected. Either choose selection by the greedy algortihm (='greedy') or 
            choose the GPR optimization which adds new parameter based on worst GPR fit (='GPR_opt').  
        """

        if property == 'phase':
            min_greedy_error = self.min_greedy_error_phase
            N_basis_vecs = self.N_basis_vecs_phase
        else:
            min_greedy_error = self.min_greedy_error_amp
            N_basis_vecs = self.N_basis_vecs_amp

        filename = (
            f'Straindata/GPRfits/GPRfits_{self.training_set_selection}_{property}_'
            f'f_lower={self.f_lower}_f_ref={self.f_ref}_'
            f'e=[{self.ecc_ref_parameterspace_range[0]}_{self.ecc_ref_parameterspace_range[1]}_N={self.amount_input_wfs}]_'
            f'No={self.amount_output_wfs}_'
            f'g_err={min_greedy_error}_'
            f'Ng_vecs={N_basis_vecs}_'
            f'min_s={self.minimum_spacing_greedy}.npz'
        )
        print(0, self.training_set_selection, self.minimum_spacing_greedy)
        try:
            data = np.load(filename, allow_pickle=True)
            print(f'GPR fit data {property} loaded: {filename}', N_basis_vecs, len(data['empirical_nodes']))
            
        except:
            if self.training_set_selection == 'GPR_opt':
                self.fit_to_training_set_GPR_opt(
                    N_basis_vecs=N_basis_vecs,
                    property=property,
                    plot_GPR_fits=plot_fits,
                    save_fig_GPR_fits=save_fig_fits,
                    save_fits_to_file=True
                )
            elif self.training_set_selection == 'greedy':
                self.fit_to_training_set(
                min_greedy_error=min_greedy_error,
                N_basis_vecs=N_basis_vecs,
                property=property,
                plot_GPR_fits=plot_fits,
                save_fig_GPR_fits=save_fig_fits,
                save_fits_to_file=True
            )
            else:
                print(
                    'training_set_selection: choose the way the best estimated training set gets selected. '
                    'Either choose selection by the greedy algorithm (="greedy") or choose the GPR optimization which adds new parameter based on worst GPR fit (="GPR_opt").'
                )
            data = np.load(filename, allow_pickle=True)
            print(f'GPR fit data loaded: {filename}', N_basis_vecs, len(data['empirical_nodes']))

        print(6, data['GPR_fit'].shape, data['residual_reduced_basis'].shape, len(data['empirical_nodes']))
        result = {
        'GPR_fit': data['GPR_fit'],
        'empirical_nodes': data['empirical_nodes'],
        'residual_reduced_basis': data['residual_reduced_basis'],
        'time': data['time'],
        'best_rep_parameters_idx': data['best_rep_parameters_idx'],
        'best_rep_parameters': data['best_rep_parameters'],
        'amp_circ': data['amp_circ'],
        'phase_circ': data['phase_circ']
        }

        data.close()  # safely close the file
        return result
     
    
    def _load_b_matrix(self, property):
        """Try to load the B_matrix for a given property (amplitude or phase)."""
        if property == 'phase': 
            N_basis_vecs = self.N_basis_vecs_phase
            min_greedy_error = self.min_greedy_error_phase
        if property == 'amplitude':
            N_basis_vecs = self.N_basis_vecs_amp
            min_greedy_error = self.min_greedy_error_amp

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
            data = np.load(filename)
            print(f' B_matrix {property} loaded: {filename}')

        except FileNotFoundError:
            print(f'B_matrix file for {property} not found: {filename} .\n Calculate B_matrix...')

            data_GPR = self._load_gpr_data(property)
            self.residual_reduced_basis = data_GPR['residual_reduced_basis']
            self.empirical_nodes_idx = data_GPR['empirical_nodes']
            print(3, self.residual_reduced_basis.shape, len(self.empirical_nodes_idx))
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

# for N_vecs in [25]:
#     gs = Load_Offline_Surrogate(time_array=time_array, ecc_ref_parameterspace_range=[0.0, 0.3], amount_input_wfs=40,
#                                     amount_output_wfs=500, N_basis_vecs_amp=25, N_basis_vecs_phase=25, minimum_spacing_greedy=0.005)
# #     gs.fit_to_training_set(property='phase', N_basis_vecs=N_vecs, plot_fits=True, save_fig_fits=True, save_fits_to_file=True, plot_residuals_time_evolve=True, plot_residuals_ecc_evolve=True)
# #     gs.fit_to_training_set(property='amplitude', N_basis_vecs=N_vecs, plot_fits=True, save_fig_fits=True, save_fits_to_file=True, plot_residuals_ecc_evolve=True, plot_kernels=True)
#     gs.create_offline_surrogate(training_set_selection='greedy', plot_fits=True, save_fig_fits=True)
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
