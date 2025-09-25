from call_surrogate_old import *

class AnalyseSurrogate(Generate_Online_Surrogate):

    def __init__(self, time_array, mass_paramspace, eccentric_paramspace, mass_ratio_paramspace, luminosity_distance_paramspace, chi_range, N_greedy_vecs_amp, N_greedy_vecs_phase, f_ref, f_lower):
        
        self.mass_paramspace = mass_paramspace
        self.eccentric_paramspace = eccentric_paramspace
        self.mass_ratio_paramspace = mass_ratio_paramspace
        self.luminosity_distance_paramspace = luminosity_distance_paramspace
        self.chi_range = chi_range

        super().__init__(
            self,
            time=time_array,
            N_greedy_vecs_amp=N_greedy_vecs_amp,
            N_greedy_vecs_phase=N_greedy_vecs_phase,
            f_ref=f_ref,
            f_lower=f_lower,
        )
        
    def analyse_eccentric_mass_indepedent_accuracy(self, time_array, plot_GRP_fit=False, save_fig_fits=False, plot_surr_datapieces=False, save_fig_datapieces=False, plot_surr_wf=False, save_fig_surr=False):

        relative_errors_hp = np.zeros((len(self.eccentric_paramspace), len(self.mass_paramspace), len(self.luminosity_distance_paramspace)))
        relative_errors_hc = np.zeros((len(self.eccentric_paramspace), len(self.mass_paramspace), len(self.luminosity_distance_paramspace)))
        relative_errors_amp = np.zeros((len(self.eccentric_paramspace), len(self.mass_paramspace), len(self.luminosity_distance_paramspace)))
        relative_errors_phase = np.zeros((len(self.eccentric_paramspace), len(self.mass_paramspace), len(self.luminosity_distance_paramspace)))

        surrogate_generator = Generate_Online_Surrogate(
            time_array,
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
            truncate_at_tmin=True,
        )

        surrogate_generator.load_offline_surrogate(plot_GPRfit=plot_GRP_fit, save_fig_fits=save_fig_fits)

        for ecc_idx, ecc in enumerate(self.eccentric_paramspace):
            for mass_idx, mass in enumerate(self.mass_paramspace):
                    for ld_idx, luminosity_distance in enumerate(self.luminosity_distance_paramspace):
                        
                        # Comparison with real waveform datapieces
                        surrogate_amp, surrogate_phase = surrogate_generator.generate_PhenomTE_surrogate(
                            ecc_ref=ecc,
                            plot_surr_datapiece=plot_surr_datapieces,
                            save_fig_datapiece=save_fig_datapieces,
                            plot_surr_wf=plot_surr_wf,
                            save_fig_surr=save_fig_surr
                        )

                        surrogate_amp, true_amp, relative_error_amp = self._plot_surr_datapieces(
                            property='amplitude',
                            surrogate_datapiece=surrogate_amp,
                            save_fig_datapiece=save_fig_datapieces,
                            geometric_units=False,
                            total_mass=mass,
                            luminosity_distance=luminosity_distance
                        )

                        surrogate_phase, true_phase, relative_error_phase = self._plot_surr_datapieces(
                            property='phase',
                            surrogate_datapiece=surrogate_phase,
                            save_fig_datapiece=save_fig_datapieces,
                            geometric_units=False,
                            total_mass=mass,
                            luminosity_distance=luminosity_distance
                        )

                        relative_errors_amp[ecc_idx, mass_idx, ld_idx] = relative_error_amp
                        relative_errors_phase[ecc_idx, mass_idx, ld_idx] = relative_error_phase



                        # fig_compare_datapieces = plt.figure(figsize=(16, 7))

                        # # --- Left: Amplitude ---
                        # ax1 = fig_compare_datapieces.add_subplot(121, projection='3d')
                        # sc1 = ax1.scatter(E.flatten(),
                        #                 M.flatten(),
                        #                 L.flatten(),
                        #                 c=relative_errors_amp.flatten(),
                        #                 cmap='viridis', s=50)

                        # ax1.set_xlabel("Eccentricity")
                        # ax1.set_ylabel("Mass")
                        # ax1.set_zlabel("Luminosity Distance")
                        # ax1.set_title("Relative Error (Amplitude)")
                        # fig_compare_datapieces.colorbar(sc1, ax=ax1, shrink=0.6, label="Relative Error")

                        # # --- Right: Phase ---
                        # ax2 = fig_compare_datapieces.add_subplot(122, projection='3d')
                        # sc2 = ax2.scatter(E.flatten(),
                        #                 M.flatten(),
                        #                 L.flatten(),
                        #                 c=relative_errors_phase.flatten(),
                        #                 cmap='plasma', s=50)

                        # ax2.set_xlabel("Eccentricity")
                        # ax2.set_ylabel("Mass")
                        # ax2.set_zlabel("Luminosity Distance")
                        # ax2.set_title("Relative Error (Phase)")
                        # fig_compare_datapieces.colorbar(sc2, ax=ax2, shrink=0.6, label="Relative Error")

                        # plt.tight_layout()
                        # plt.show()



                        # Comparison with real waveform h_+, h_x
                        surrogate_hp, surrogate_hc = surrogate_generator.get_surrogate_polarisations(
                            total_mass=mass,
                            luminosity_distance=luminosity_distance,
                            geometric_units=False,
                            plot_polarisations=False,
                            save_fig=False
                        )

                        surrogate_h = surrogate_amp * np.exp(1j * surrogate_phase)

                        surrogate_h, true_h, relative_error_hp, relative_error_hc = self.plot_surrogate_waveform(
                            h_surrogate=surrogate_h,
                            save_fig_surr=save_fig_surr,
                            geometric_units=False
                        )

                        relative_errors_hp[ecc_idx, mass_idx, ld_idx] = relative_error_hp
                        relative_errors_hc[ecc_idx, mass_idx, ld_idx] = relative_error_hc
                        
                        # # Create grids of parameter values
                        # E, M, L = np.meshgrid(self.eccentric_paramspace,
                        #                     self.mass_paramspace,
                        #                     self.luminosity_distance_paramspace,
                        #                     indexing="ij")

                        # fig_compare_polarisations = plt.figure(figsize=(16, 7))

                        # # --- Left: h_plus ---
                        # ax1 = fig_compare_polarisations.add_subplot(121, projection='3d')
                        # sc1 = ax1.scatter(E.flatten(),
                        #                 M.flatten(),
                        #                 L.flatten(),
                        #                 c=relative_errors_hp.flatten(),
                        #                 cmap='viridis', s=50)

                        # ax1.set_xlabel("Eccentricity")
                        # ax1.set_ylabel("Mass")
                        # ax1.set_zlabel("Luminosity Distance")
                        # ax1.set_title("Relative Error (h_plus)")
                        # fig_compare_polarisations.colorbar(sc1, ax=ax1, shrink=0.6, label="Relative Error")

                        # # --- Right: h_cross ---
                        # ax2 = fig_compare_polarisations.add_subplot(122, projection='3d')
                        # sc2 = ax2.scatter(E.flatten(),
                        #                 M.flatten(),
                        #                 L.flatten(),
                        #                 c=relative_errors_hc.flatten(),
                        #                 cmap='plasma', s=50)

                        # ax2.set_xlabel("Eccentricity")
                        # ax2.set_ylabel("Mass")
                        # ax2.set_zlabel("Luminosity Distance")
                        # ax2.set_title("Relative Error (h_cross)")
                        # fig_compare_polarisations.colorbar(sc2, ax=ax2, shrink=0.6, label="Relative Error")

                        # plt.tight_layout()
                        # plt.show()



                        

                        print('worst relative errors: ', relative_error_amp.flatten().sorted()[:5])
                        print('worst relative errors: ', relative_error_phase.flatten().sorted()[:5])
                        print('worst relative errors: ', relative_error_hp.flatten().sorted()[:5])
                        print('worst relative errors: ', relative_error_hc.flatten().sorted()[:5])
                            
                    

    def run_analysis(self):
        # Initialize the surrogate model generator
        sampling_frequency = 2048 # or 4096
        duration = 4 # seconds
        time_array = np.linspace(-duration, 0, int(sampling_frequency * duration))  # time in seconds

        
        surrogate_generator = Generate_Online_Surrogate(
            time_array,
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
            truncate_at_tmin=True,
        )

        # Generate the surrogate model
        surrogate_model = surrogate_generator.generate_surrogate()

        # Perform analysis on the surrogate model
        analysis_results = surrogate_generator.analyze_surrogate(surrogate_model)

        return analysis_results

sampling_frequency = 2048 # or 4096
duration = 4 # seconds
time_array = np.linspace(-duration, 0, int(sampling_frequency * duration))  # time in seconds


analyse_surrogate = AnalyseSurrogate(
    time_array=time_array,
    mass_paramspace = np.linspace(20, 80, 2),  # Example mass values from 20 to 80
    eccentric_paramspace = np.linspace(0.1, 0.2, 2),  # Example eccentricity values from 0.0 to 0.2
    luminosity_distance_paramspace = np.linspace(100, 500, 2),  # Example distances from 100 to 500 Mpc
    mass_ratio_paramspace = np.linspace(1, 3, 3),  # Example mass ratios from 1 to 3
    chi_range = (-0.5, 0.5),
    N_greedy_vecs_amp = 40,
    N_greedy_vecs_phase = 40,       
    f_ref = 20,
    f_lower = 15
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

plt.show()