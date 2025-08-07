from generate_phenom_surrogate import *

class call_surrogate_offline():

    def __init__(self, time_array, ecc_ref_parameterspace_range, amount_input_wfs, amount_output_wfs, total_mass_range=None, luminosity_distance_range=None, N_greedy_vecs_amp=None, N_greedy_vecs_phase=None, min_greedy_error_amp=None, min_greedy_error_phase=None, f_lower=10, f_ref=20, chi1=0, chi2=0, phiRef=0., rel_anomaly=0., inclination=0., truncate_at_ISCO=True, truncate_at_tmin=True, waveforms_in_geom_units=True):
        

        # Generate surrogate model
      
        generate_surrogate_fits = Generate_TrainingSet(time_array, ecc_ref_parameterspace_range, amount_input_wfs, amount_output_wfs, total_mass_range, luminosity_distance_range, N_greedy_vecs_amp, N_greedy_vecs_phase, min_greedy_error_amp, min_greedy_error_phase, f_lower, f_ref, chi1, chi2, phiRef, rel_anomaly, inclination, truncate_at_ISCO, truncate_at_tmin, waveforms_in_geom_units)
        # Fit the surrogate model to the training set for amplitude and phase
        generate_surrogate_fits.fit_to_training_set(min_greedy_error=self.min_greedy_error_amp, N_greedy_vecs=self.N_greedy_vecs_amp, property='amplitude', plot_fits=True, save_fig_fits=True, save_fits_to_file=True)
        generate_surrogate_fits.fit_to_training_set(min_greedy_error=self.min_greedy_error_phase, N_greedy_vecs=self.N_greedy_vecs_phase, property='phase', plot_fits=True, save_fig_fits=True, save_fits_to_file=True)

        # Extract amplitude specifics
        filename_amp = f'Straindata/GPRfits/GPRfits_amplitude_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_parameter_space_input)}_{max(self.ecc_parameter_space_input)}_N={len(self.ecc_parameter_space_input)}]_No={len(self.ecc_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}.npz'
        load_GPRfits = np.load(filename_amp, allow_pickle=True)
        
        gaussian_fit_amp = load_GPRfits['GPR_fit']
        empirical_nodes_idx_amp = load_GPRfits['empirical_nodes']
        residual_greedy_basis_amp = load_GPRfits['residual_greedy_basis']
        self.time = load_GPRfits['time']

        # Extract phase specifics
        filename_phase = f'Straindata/GPRfits/GPRfits_phase_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_parameter_space_input)}_{max(self.ecc_parameter_space_input)}_N={len(self.ecc_parameter_space_input)}]_No={len(self.ecc_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}.npz'
        load_GPRfits = np.load(filename_amp, allow_pickle=True)
        
        gaussian_fit_phase = load_GPRfits['GPR_fit']
        empirical_nodes_idx_phase = load_GPRfits['empirical_nodes']
        residual_greedy_basis_phase = load_GPRfits['residual_greedy_basis']

        # Compress everything in compact file for online call
        np.savez_compressed(
            'Straindata/GPRfits/Surrogate_OfflineData.npz',
            gaussian_fit_amp=gaussian_fit_amp,
            gaussian_fit_phase=gaussian_fit_phase,
            empirical_nodes_idx_amp=empirical_nodes_idx_amp,
            empirical_nodes_idx_phase=empirical_nodes_idx_phase,
            residual_greedy_basis_amp=residual_greedy_basis_amp,
            residual_greedy_basis_phase=residual_greedy_basis_phase,
            time=self.time
        )



class call_surrogate_online(call_surrogate_offline):

    def __init__(self, time_array, output_ecc_ref, ecc_ref_parameterspace_range, amount_input_wfs, amount_output_wfs, total_mass_range=None, luminosity_distance_range=None, N_greedy_vecs_amp=None, N_greedy_vecs_phase=None, min_greedy_error_amp=None, min_greedy_error_phase=None, f_lower=10, f_ref=20, chi1=0, chi2=0, phiRef=0., rel_anomaly=0., inclination=0., truncate_at_ISCO=True, truncate_at_tmin=True, waveforms_in_geom_units=True):
        Generate_Surrogate.__init__(self, time_array, output_ecc_ref, ecc_ref_parameterspace_range, amount_input_wfs, amount_output_wfs, total_mass_range, luminosity_distance_range, N_greedy_vecs_amp, N_greedy_vecs_phase, min_greedy_error_amp, min_greedy_error_phase, f_lower, f_ref, chi1, chi2, phiRef, rel_anomaly, inclination, truncate_at_ISCO, truncate_at_tmin, waveforms_in_geom_units)

    def load_surrogate_model(self):
        data = np.load('Straindata/GPRfits/Surrogate_OfflineData.npz', allow_pickle=True)

        self.gaussian_fit_amp = data['gaussian_fit_amp']
        self.gaussian_fit_phase = data['gaussian_fit_phase']
        self.empirical_nodes_idx_amp = data['empirical_nodes_idx_amp']
        self.empirical_nodes_idx_phase = data['empirical_nodes_idx_phase']
        self.residual_greedy_basis_amp = data['residual_greedy_basis_amp']
        self.residual_greedy_basis_phase = data['residual_greedy_basis_phase']
        self.time = data['time']

    def call_surrogate(self, plot_surr_datapiece=False, save_fig_datapiece=False, plot_surr_wf=False, save_fig_surr=False, plot_GPRfit=False, save_fits_to_file=False, save_fig_fits=False):
        """
        Call the surrogate model with the given output eccentricity reference.
        If plot_surr_wf is True, it will plot the surrogate waveform against the real waveform.
        If plot_surr_datapiece is True, it will plot the surrogate datapiece against the real datapiece.
        """
        
        # Load the surrogate model data
        if self.gaussian_fit_amp is None or self.gaussian_fit_phase is None:
            print("Loading surrogate model data...")
            self.load_surrogate_model()

        # Generate surrogate waveform
        surrogate_amp, surrogate_phase = self.generate_surrogate_waveform(plot_surr_datapiece=plot_surr_datapiece, save_fig_datapiece=save_fig_datapiece, plot_surr_wf=plot_surr_wf, save_fig_surr=save_fig_surr, plot_GPRfit=plot_GPRfit, save_fits_to_file=save_fits_to_file, save_fig_fits=save_fig_fits)

     