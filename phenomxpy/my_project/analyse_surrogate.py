from call_surrogate import *
from pycbc.types import TimeSeries
from pycbc.filter import optimized_match

class AnalyseSurrogate(Call_Surrogate):

    def __init__(self, time_array, eccentric_paramspace, mass_ratio_paramspace, chi_range, N_greedy_vecs_amp, N_greedy_vecs_phase, f_ref, f_lower):
        self.eccentric_paramspace = eccentric_paramspace
        self.mass_ratio_paramspace = mass_ratio_paramspace
        self.chi_range = chi_range
        self.N_greedy_vecs_amp = N_greedy_vecs_amp
        self.N_greedy_vecs_phase = N_greedy_vecs_phase
        self.f_ref = f_ref
        self.f_lower = f_lower

        Call_Surrogate.__init__(self,
            time_array=time_array,
            ecc_ref=None,
            total_mass=None,
            luminosity_distance=None, 
            f_lower=self.f_lower,
            f_ref=self.f_ref,
            chi1=0,
            chi2=0,
            phiRef=0.,
            rel_anomaly=0.,
            inclination=0.,
            truncate_at_ISCO=True,
            truncate_at_tmin=True
        )

    def calculate_mismatch(self, surrogate_waveform, true_waveform):
        # Wrap them in PyCBC TimeSeries
        delta_t = self.time[1] - self.time[0] # Get delta_t
        true_waveform_TS = TimeSeries(np.real(true_waveform), delta_t=delta_t) # Take real h+ part for mismatch
        surrogate_waveform_TS = TimeSeries(np.real(surrogate_waveform), delta_t=delta_t) # Take real h+ part for mismatch

        # Compute the match (overlap) and the time shift
        overlap, idx, phase_shift = optimized_match(surrogate_waveform_TS, true_waveform_TS, psd=None, return_phase=True)

        # Mismatch
        return 1 - overlap, idx, phase_shift

    
    def analyse_eccentric_mass_indepedent_accuracy(self, time_array, plot_GRP_fit=False, save_fig_fits=False, plot_surr_datapieces=False, save_fig_datapieces=False, plot_surr_wf=False, save_fig_surr=False):

        mean_relative_errors_waveform = np.zeros((len(self.eccentric_paramspace)))
        mean_relative_errors_amp = np.zeros((len(self.eccentric_paramspace)))
        mean_relative_errors_phase = np.zeros((len(self.eccentric_paramspace)))

        worst_relative_errors_waveform = np.zeros((len(self.eccentric_paramspace)))
        worst_relative_errors_amp = np.zeros((len(self.eccentric_paramspace)))
        worst_relative_errors_phase = np.zeros((len(self.eccentric_paramspace)))

        mismatches_waveform =  np.zeros((len(self.eccentric_paramspace)))
        phase_shift_mismatches =  np.zeros((len(self.eccentric_paramspace)))

        computation_times_surrogate = np.zeros((len(self.eccentric_paramspace)))
        computation_times_PhenomTE = np.zeros((len(self.eccentric_paramspace)))


        self.load_offline_surrogate(plot_GPRfit=plot_GRP_fit, save_fig_fits=save_fig_fits, plot_empirical_nodes_at_ecc=max(self.eccentric_paramspace), save_fig_empirical_nodes=True, plot_greedy_vectors=True, save_greedy_vectors_fig=True)

        
        for ecc_idx, ecc in enumerate(self.eccentric_paramspace):

                    # ---------------- Generate Phase & Amplitude --------------------------------
                    _, _, computation_time = self.generate_PhenomTE_surrogate(
                        ecc_ref=ecc,
                        geometric_units=True,
                        get_computation_time=True
                    )
                    # Get computation time of waveforms
                    computation_times_surrogate[ecc_idx] = computation_time
                    
                    # Calculate relative errors
                    # Get dictionary with keys 'amplitude', 'phase', 'waveform': surrogate_prop, true_prop, relative_error_prop
                    relative_errors_dict = self.calculate_relative_error(get_true_and_surrogate_output=True, get_computation_time_PhenomTE=True) 

                    computation_times_PhenomTE[ecc_idx] = relative_errors_dict['computation_time']

                    mean_relative_errors_amp[ecc_idx] = np.mean(relative_errors_dict['amplitude'][2])
                    worst_relative_errors_amp[ecc_idx] = np.max(relative_errors_dict['amplitude'][2])

                    mean_relative_errors_phase[ecc_idx] = np.mean(relative_errors_dict['phase'][2])
                    worst_relative_errors_phase[ecc_idx] = np.max(relative_errors_dict['phase'][2])

                    mean_relative_errors_waveform[ecc_idx] = np.mean(relative_errors_dict['waveform'][2])
                    worst_relative_errors_waveform[ecc_idx] = np.max(relative_errors_dict['waveform'][2])

                    surrogate_h, true_h = relative_errors_dict['waveform'][:2]

                    
                    # Calculate mismatches
                    mismatch, idx, phase_shift = self.calculate_mismatch(surrogate_h, true_h)
                    mismatches_waveform[ecc_idx] = mismatch
                    phase_shift_mismatches[ecc_idx] = phase_shift
                    


                    # # -------------------- Waveform --------------------------------------
                    # # Comparison with real waveform h_+, h_x
                    # surrogate_hp, surrogate_hc = self.get_surrogate_polarisations(
                    #     geometric_units=True,
                    #     plot_polarisations=False,
                    #     save_fig=False
                    # )

                    



        fig, axs = plt.subplots(1, 4, figsize=(16, 7))

        # Relative Amplitude
        axs[0].scatter(self.eccentric_paramspace, mean_relative_errors_amp, label='Mean relative time-domain error')
        axs[0].scatter(self.eccentric_paramspace, worst_relative_errors_amp, color='red', label='Worst relative time-domain error')
        axs[0].set_xlabel("Eccentricity")
        axs[0].set_ylabel("Mean Relative Error")
        axs[0].set_title("Mean Relative Error - Amplitude")
        axs[0].set_yscale('log')
        axs[0].grid(True)
        axs[0].legend()

        # Relative Phase
        axs[1].scatter(self.eccentric_paramspace, mean_relative_errors_phase, label='Mean relative time-domain error')
        axs[1].scatter(self.eccentric_paramspace, worst_relative_errors_phase, color='red', label='Worst relative time-domain error')
        axs[1].set_xlabel("Eccentricity")
        axs[1].set_ylabel("Mean Relative Error")
        axs[1].set_title("Mean Relative Error - Phase")
        axs[1].set_yscale('log')
        axs[1].grid(True)
        axs[1].legend()

        # Relative Waveform
        # axs[2].scatter(self.eccentric_paramspace[mean_relative_errors_waveform < 1], mean_relative_errors_waveform[mean_relative_errors_waveform < 1], label='Mean relative time-domain error')
        # axs[2].scatter(self.eccentric_paramspace[mean_relative_errors_waveform < 1], worst_relative_errors_waveform[mean_relative_errors_waveform < 1], color='red', label='Worst relative time-domain error')
        # axs[2].set_xlabel("Eccentricity")
        # axs[2].set_ylabel("Mean Relative Error")
        # axs[2].set_title("Mean Relative Error - Waveform")
        # axs[2].set_yscale('log')
        # axs[2].grid(True)
        # axs[2].legend()

        # Mismatch
        axs[2].scatter(self.eccentric_paramspace, mismatches_waveform, label='Mismatches')
        axs[2].scatter(self.eccentric_paramspace, phase_shift_mismatches, label='Phase shift mismatches')
        axs[2].set_xlabel("Eccentricity")
        axs[2].set_ylabel("Mismatch")
        axs[2].set_title("Mismatch - Waveform")
        axs[2].set_yscale('log')
        axs[2].grid(True)
        axs[2].legend()

        # Computation time
        axs[3].scatter(self.eccentric_paramspace, computation_times_surrogate, label='Computation time surrogate')
        axs[3].scatter(self.eccentric_paramspace, computation_times_PhenomTE, label='Computation time PhenomTE')
        axs[3].set_xlabel("Eccentricity")
        axs[3].set_ylabel("Computation time")
        axs[3].set_title("Computation time - Waveform")
        axs[3].set_yscale('log')
        axs[3].grid(True)
        axs[3].legend()


        plt.tight_layout()
        
        figname = f'Errors_and_efficiency_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_parameter_space_input)}_{max(self.ecc_parameter_space_input)}_Ni={len(self.ecc_parameter_space_input)}]_No={len(self.ecc_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}.png'
        # Ensure the directory exists, creating it if necessary and save
        os.makedirs('Images/Errors_and_efficiency', exist_ok=True)
        plt.savefig(f'Images/Errors_and_efficiency/{figname}', dpi=300)

        print(f'Figure saved to Images/Errors_and_efficiency/{figname}')

        return mismatch, computation_times_surrogate



sampling_frequency = 2048 # or 4096
duration = 4 # seconds
time_array = np.linspace(-duration, 0, int(sampling_frequency * duration))  # time in seconds


analyse_surrogate = AnalyseSurrogate(
    time_array=time_array,
    eccentric_paramspace = np.linspace(0.0, 0.3, 100),  
    mass_ratio_paramspace = np.linspace(1, 3, 3), 
    chi_range = (-0.5, 0.5),
    N_greedy_vecs_amp = 40,
    N_greedy_vecs_phase = 40,       
    f_ref = 20,
    f_lower = 10
)


analyse_surrogate.analyse_eccentric_mass_indepedent_accuracy(
    time_array = time_array,  # Example time array from -4 to 0 seconds with 8192 samples
    plot_GRP_fit = True,
    save_fig_fits = True,
    plot_surr_datapieces = True,
    save_fig_datapieces = True,
    plot_surr_wf = True,
    save_fig_surr = True
)

# plt.show()