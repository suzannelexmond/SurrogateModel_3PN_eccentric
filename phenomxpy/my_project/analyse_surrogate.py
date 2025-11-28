from call_surrogate import *
from call_surrogate_for_testing import*
from pycbc.types import TimeSeries
from pycbc.filter import optimized_match
import itertools
import pandas as pd
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # activates 3D projection support

class AnalyseSurrogate(Call_Surrogate_Testing):

    def __init__(self, time_array, eccentric_paramspace, mass_ratio_paramspace, amount_input_wfs=60, amount_output_wfs=500, N_basis_vecs_amp=40, N_basis_vecs_phase=40, training_set_selection='GPR_opt', 
                 minimum_spacing_greedy=0.008, ecc_ref=None, total_mass=None, luminosity_disytance=None, f_ref=20, f_lower=10, chi1=0, chi2=0, phiRef=0., rel_anomaly=0., inlcination=0., truncate_at_ISCO=True, truncate_at_tmin=True):
        
        self.eccentric_paramspace = eccentric_paramspace
        self.mass_ratio_paramspace = mass_ratio_paramspace

        Call_Surrogate_Testing.__init__(
            self,
            time_array=time_array,
            ecc_ref_parameterspace_range=[0.0, 0.3],
            amount_input_wfs=amount_input_wfs,
            amount_output_wfs=amount_output_wfs,
            N_basis_vecs_amp=N_basis_vecs_amp, 
            N_basis_vecs_phase=N_basis_vecs_phase,
            training_set_selection=training_set_selection,
            minimum_spacing_greedy=minimum_spacing_greedy,
            ecc_ref=ecc_ref,
            total_mass=total_mass,
            luminosity_distance=luminosity_disytance, 
            f_lower=f_lower,
            f_ref=f_ref,
            chi1=chi1,
            chi2=chi2,
            phiRef=phiRef,
            rel_anomaly=rel_anomaly,
            inclination=inlcination,
            truncate_at_ISCO=truncate_at_ISCO,
            truncate_at_tmin=truncate_at_tmin
        )

    def calculate_mismatch(self, surrogate_waveform, true_waveform=None, ecc_ref=None):
        if ecc_ref is None:
            ecc_ref = self.output_ecc_ref

        if true_waveform is None:
            true_hp, true_hc = self.simulate_inspiral(
                ecc_ref=ecc_ref,
                geometric_units=True,
                truncate_at_ISCO=False,
                truncate_at_tmin=False
            )

            # Compute true amplitude and phase
            true_amp = self.amplitude(
                hplus_NR=true_hp,
                hcross_NR=true_hc,
                geometric_units=self.geometric_units,
                total_mass=self.total_mass,
                luminosity_distance=self.luminosity_distance
            )
            true_phase = self.phase(true_hp, true_hc)

            # Construct complex waveforms
            true_waveform = true_amp * np.exp(1j * true_phase)

        # Wrap them in PyCBC TimeSeries
        delta_t = self.time[1] - self.time[0] # Get delta_t
        true_waveform_TS = TimeSeries(np.real(true_waveform), delta_t=delta_t) # Take real h+ part for mismatch
        surrogate_waveform_TS = TimeSeries(np.real(surrogate_waveform), delta_t=delta_t) # Take real h+ part for mismatch

        # Compute the match (overlap) and the time shift
        overlap, idx, phase_shift = optimized_match(surrogate_waveform_TS, true_waveform_TS, psd=None, return_phase=True)

        # Mismatch
        return 1 - overlap, idx, phase_shift

    
    def analyse_eccentric_mass_indepedent_accuracy(self, training_set_selection='GPR_opt', plot_GRP_fit=False, save_fig_fits=False, plot_errors_and_efficiency=True):
        # print(0, self.N_basis_vecs_amp, self.N_basis_vecs_phase)

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

        self.load_offline_surrogate(plot_GPRfit=plot_GRP_fit, save_fig_fits=save_fig_fits, plot_empirical_nodes_at_ecc=max(self.eccentric_paramspace), save_fig_empirical_nodes=True, plot_greedy_vectors=True, save_fig_greedy_vectors=True)

        
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

                    


        if plot_errors_and_efficiency is True:
              
            fig_errors_and_efficiency, axs = plt.subplots(1, 4, figsize=(16, 7))

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
            
            figname = f'Images/Errors_and_efficiency/Errors_and_efficiency_{self.training_set_selection}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_ref_parameter_space_input)}_{max(self.ecc_ref_parameter_space_input)}_Ni={len(self.ecc_ref_parameter_space_input)}]_No={len(self.ecc_ref_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_basis_vecs_phase}_Nga={self.N_basis_vecs_amp}_ms={self.minimum_spacing_greedy}.png'
            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Images/Errors_and_efficiency', exist_ok=True)
            plt.savefig(figname, dpi=300)

            print(self.colored_text(f'Figure is saved in {figname}', 'blue'))

        return mismatches_waveform, computation_times_surrogate
    
    def update_hyperparameters(self, amount_input_wfs, amount_output_wfs, N_amp, N_phase, minimum_spacing):
        """Update surrogate hyperparameters and regenerate parameter spaces."""
        self.amount_input_wfs = amount_input_wfs
        self.amount_output_wfs = amount_output_wfs
        self.N_basis_vecs_amp = N_amp
        self.N_basis_vecs_phase = N_phase
        self.minimum_spacing_greedy=minimum_spacing

        # Rebuild eccentricity grids
        self.ecc_ref_parameter_space_input = np.linspace(
            self.ecc_ref_parameterspace_range[0],
            self.ecc_ref_parameterspace_range[1],
            amount_input_wfs
        )
        self.ecc_ref_parameter_space_output = np.linspace(
            self.ecc_ref_parameterspace_range[0],
            self.ecc_ref_parameterspace_range[1],
            amount_output_wfs
        )

    def hyperparameter_optimisation(self, M_input_waveforms_PS, M_output_waveforms_PS, N_basis_vecs_amp_PS, N_basis_vecs_phase_PS, minimum_spacing_list, training_set_selection='GPR_opt', plot_distributions=False):
        
        if training_set_selection is not self.training_set_selection:
            self.training_set_selection = training_set_selection
            print(f"WARNING: Class object training_set_selection is different from specified function variable. Updated training set selection of class self object to {self.training_set_selection} for hyperparameter optimisation.")

        param_grid = list(itertools.product(
            M_input_waveforms_PS, 
            M_output_waveforms_PS, 
            N_basis_vecs_amp_PS, 
            N_basis_vecs_phase_PS,
            minimum_spacing_list
        ))

        # print(param_grid, M_input_waveforms_PS, 
        #     M_output_waveforms_PS, 
        #     N_basis_vecs_amp_PS, 
        #     N_basis_vecs_phase_PS,
        #     minimum_spacing_list)

        try:
            df_sorted = pd.read_csv(f"Straindata/Hyperparameter_opt/"
            f"results_hyperparameter_opt_{self.training_set_selection}_"
            f"Ngp=[{min(N_basis_vecs_phase_PS)}, {max(N_basis_vecs_phase_PS)}, n={len(N_basis_vecs_phase_PS)}]_"
            f"Nga=[{min(N_basis_vecs_amp_PS)}, {max(N_basis_vecs_amp_PS)}, n={len(N_basis_vecs_amp_PS)}]_"
            f"Mi=[{min(M_input_waveforms_PS)}, {max(M_input_waveforms_PS)}, n={len(M_input_waveforms_PS)}]_"
            f"Mo=[{min(M_output_waveforms_PS)}, {max(M_output_waveforms_PS)}, n={len(M_output_waveforms_PS)}]_"
            f"ms=[{min(minimum_spacing_list)}, {max(minimum_spacing_list)}, n={len(minimum_spacing_list)}].csv")

            print("Loaded existing hyperparameter optimisation results.")

        except:
            print("No existing results found. Starting hyperparameter optimisation...")
            
            results = []
            mean_mismatches = []
            mean_computation_times = []

            for amount_input_wfs, amount_output_wfs, N_amp, N_phase, minimum_spacing in param_grid:
                print(f"Testing combination: Input WFs: {amount_input_wfs}, Output WFs: {amount_output_wfs}, N_amp: {N_amp}, N_phase: {N_phase}, min_spacing: {minimum_spacing}")
                
                self.update_hyperparameters(amount_input_wfs, amount_output_wfs, N_amp, N_phase, minimum_spacing)
                hyperparam_mismatches, hyperparam_computational_times = self.analyse_eccentric_mass_indepedent_accuracy(training_set_selection=training_set_selection, save_fig_fits=True, plot_GRP_fit=True, plot_errors_and_efficiency=True)
                
                mismatches_waveform =  np.zeros((len(self.ecc_ref_parameter_space_output)))
                for ecc_idx, ecc in enumerate(self.ecc_ref_parameter_space_output):
                    surrogate_amp, surrogate_phase, computation_time = self.generate_PhenomTE_surrogate(
                                ecc_ref=ecc,
                                geometric_units=True,
                                get_computation_time=True
                            )
                    
                    surrogate_h = surrogate_amp * np.exp(1j * surrogate_phase)
                    # Calculate mismatches
                    mismatch, idx, phase_shift = self.calculate_mismatch(surrogate_h)
                    mismatches_waveform[ecc_idx] = mismatch
            # sys.exit(1) 
            # Save mean mismatch of the hyperparameter combination
                mean_mismatch = np.mean(hyperparam_mismatches)
                mean_mismatches.append(mean_mismatch)
                mean_computation_times.append(np.mean(hyperparam_computational_times))

                entry = {
                        'amount_input_wfs': amount_input_wfs,
                        'amount_output_wfs': amount_output_wfs,
                        'N_basis_vecs_amp': N_amp,
                        'N_basis_vecs_phase': N_phase,
                        'mismatch': mean_mismatch,
                        'minimum_spacing': minimum_spacing,
                        'computation time': np.mean(hyperparam_computational_times)
                    }

                results.append(entry)

                if plot_distributions is True:
                    mismatch_distribution = plt.figure()
                    plt.hist(hyperparam_mismatches, bins=20)
                    plt.xlabel('Mismatch')
                    plt.ylabel('Count')

                    figname = f'Mismatch_distributions_{self.training_set_selection}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_ref_parameter_space_input)}_{max(self.ecc_ref_parameter_space_input)}_Ni={len(self.ecc_ref_parameter_space_input)}]_No={len(self.ecc_ref_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_basis_vecs_phase}_Nga={self.N_basis_vecs_amp}_ms={self.minimum_spacing_greedy}.png'
                    # Ensure the directory exists, creating it if necessary and save
                    os.makedirs('Images/Mismatch_distributions', exist_ok=True)
                    mismatch_distribution.savefig(f'Images/Mismatch_distributions/{figname}', dpi=300)



            # convert to DataFrame once
            df = pd.DataFrame(results)

            # sort DataFrame by mismatch ascending
            df_sorted = df.sort_values(by="mismatch", ascending=True)

            # save to CSV (good for portability) 
            os.makedirs('Straindata/Hyperparameter_opt', exist_ok=True)
            df_sorted.to_csv(
                f"Straindata/Hyperparameter_opt/"
                f"results_hyperparameter_opt_{self.training_set_selection}_"
                f"Ngp=[{min(N_basis_vecs_phase_PS)}, {max(N_basis_vecs_phase_PS)}, n={len(N_basis_vecs_phase_PS)}]_"
                f"Nga=[{min(N_basis_vecs_amp_PS)}, {max(N_basis_vecs_amp_PS)}, n={len(N_basis_vecs_amp_PS)}]_"
                f"Mi=[{min(M_input_waveforms_PS)}, {max(M_input_waveforms_PS)}, n={len(M_input_waveforms_PS)}]_"
                f"Mo=[{min(M_output_waveforms_PS)}, {max(M_output_waveforms_PS)}, n={len(M_output_waveforms_PS)}]_"
                f"ms=[{min(minimum_spacing_list)}, {max(minimum_spacing_list)}, n={len(minimum_spacing_list)}].csv",
                index=False
            )

        # # OR save to Pickle (keeps Python/Pandas types exactly)
        # df_sorted.to_pickle("results_mismatch_all.pkl")

        # Later you can reload easily:
        # df = pd.read_csv("results_mismatch_all.csv")
        # df = pd.read_pickle("results_mismatch_all.pkl")

        # for preview
        print(df_sorted.head(30))

        self.plot_3d_hyperopt_results(df_sorted)




        # figure = corner.corner(
        #     data,
        #     labels=["amount_input_wfs", "amount_output_wfs", 
        #             "N_basis_vecs_amp", "N_basis_vecs_phase"],
        #     show_titles=True
        # )

        # # overlay mismatch colors
        # axes = np.array(figure.axes).reshape((4, 4))
        # mismatches = df['mismatch'].values

        # for yi in range(4):
        #     for xi in range(yi):
        #         ax = axes[yi, xi]
        #         ax.scatter(data[:, xi], data[:, yi], c=mismatches, cmap="viridis", s=5)
        
        # log_mismatches = np.log10(df['mismatch'].values)
        # ax.scatter(data[:, xi], data[:, yi], c=log_mismatches, cmap="viridis", s=5)

    def plot_3d_hyperopt_results(self, df_sorted):
        # Ensure correct data types
        df_sorted = df_sorted.copy()
        df_sorted['minimum_spacing'] = df_sorted['minimum_spacing'].astype(float)
        df_sorted['N_basis_vecs_phase'] = df_sorted['N_basis_vecs_phase'].astype(float)
        df_sorted['N_basis_vecs_amp'] = df_sorted['N_basis_vecs_amp'].astype(float)
        df_sorted['mismatch'] = df_sorted['mismatch'].astype(float)
        df_sorted['computation time'] = df_sorted['computation time'].astype(float)

        # Normalise computation time for colormap scaling
        comp_time_norm = (df_sorted['computation time'] - df_sorted['computation time'].min()) / \
                        (df_sorted['computation time'].max() - df_sorted['computation time'].min())
        df_sorted['comp_time_norm'] = comp_time_norm

        # Iterate over all minimum_spacing values
        for ms in sorted(df_sorted['minimum_spacing'].unique()):
            subset = df_sorted[df_sorted['minimum_spacing'] == ms]

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

            # Create scatter plot
            p = ax.scatter(
                subset['N_basis_vecs_phase'],
                subset['N_basis_vecs_amp'],
                subset['mismatch'],
                c=subset['comp_time_norm'],
                cmap='viridis',
                s=80,
                edgecolor='k',
                alpha=0.8
            )

            # Labels and colorbar
            ax.set_xlabel('N_basis_vecs_phase', labelpad=10)
            ax.set_ylabel('N_basis_vecs_amp', labelpad=10)
            ax.set_zlabel('Mean mismatch', labelpad=10)
            ax.set_zscale('log')
            ax.set_title(f'Hyperparameter optimisation (minimum_spacing={ms})')
            cbar = fig.colorbar(p, ax=ax, shrink=0.6, pad=0.1)
            cbar.set_label('Normalized computation time', rotation=270, labelpad=15)

            # Improve layout and perspective
            ax.view_init(elev=25, azim=45)
            plt.tight_layout()

            # Save the figure
            os.makedirs("Images/Hyperparameter_3D", exist_ok=True)
            figname = f'Hyperparameter_3D_opt_{self.training_set_selection}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_ref_parameter_space_input)}_{max(self.ecc_ref_parameter_space_input)}_Ni={len(self.ecc_ref_parameter_space_input)}]_No={len(self.ecc_ref_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_basis_vecs_phase}_Nga={self.N_basis_vecs_amp}_ms={self.minimum_spacing_greedy}.png'

            plt.savefig(f'Images/Hyperparameter_3D/{figname}', dpi=300)
            print(f"Saved 3D plot for minimum_spacing={ms} to Images/Hyperparameter_3D/{figname}")




sampling_frequency = 2048 # or 4096
duration = 4 # seconds
time_array = np.linspace(-duration, 0, int(sampling_frequency * duration))  # time in seconds

analyse_surrogate_greedy = AnalyseSurrogate(
    time_array=time_array,
    eccentric_paramspace = np.linspace(0.0, 0.3, 100),  
    mass_ratio_paramspace = np.linspace(1, 3, 3),      
    f_ref = 20,
    f_lower = 10,
    training_set_selection='greedy',
)

analyse_surrogate_gpr = AnalyseSurrogate(
    time_array=time_array,
    eccentric_paramspace = np.linspace(0.0, 0.3, 100),  
    mass_ratio_paramspace = np.linspace(1, 3, 3),      
    f_ref = 20,
    f_lower = 10,
    training_set_selection='GPR_opt',
)


# analyse_surrogate.analyse_eccentric_mass_indepedent_accuracy(
#     plot_GRP_fit = True,
#     save_fig_fits = True,
#     plot_surr_datapieces = True,
#     save_fig_datapieces = True,
#     plot_surr_wf = True,
#     save_fig_surr = True
# )

# analyse_surrogate_greedy.hyperparameter_optimisation(M_input_waveforms_PS=[40], M_output_waveforms_PS=[500], N_basis_vecs_amp_PS=np.arange(20, 40, step=5), N_basis_vecs_phase_PS=np.arange(20, 40, step=5), minimum_spacing_list=[0.003, 0.005], plot_distributions=True)
# analyse_surrogate_greedy.hyperparameter_optimisation(M_input_waveforms_PS=[50], M_output_waveforms_PS=[500], N_basis_vecs_amp_PS=np.arange(25, 50, step=5), N_basis_vecs_phase_PS=np.arange(25, 50, step=5), minimum_spacing_list=[0.003, 0.005], plot_distributions=True)
# analyse_surrogate_greedy.hyperparameter_optimisation(M_input_waveforms_PS=[60], M_output_waveforms_PS=[500], N_basis_vecs_amp_PS=np.arange(25, 60, step=5), N_basis_vecs_phase_PS=np.arange(25, 60, step=5), minimum_spacing_list=[0.003, 0.005], plot_distributions=True)

# analyse_surrogate_gpr.hyperparameter_optimisation(M_input_waveforms_PS=[40], M_output_waveforms_PS=[500], N_basis_vecs_amp_PS=np.arange(20, 40, step=5), N_basis_vecs_phase_PS=np.arange(25, 40, step=5), minimum_spacing_list=[0.003, 0.005], plot_distributions=True)
analyse_surrogate_gpr.hyperparameter_optimisation(training_set_selection='greedy', M_input_waveforms_PS=[40], M_output_waveforms_PS=[500], N_basis_vecs_amp_PS=[30], N_basis_vecs_phase_PS=np.arange(20, 40, step=5), minimum_spacing_list=[0.003], plot_distributions=True)
analyse_surrogate_gpr.hyperparameter_optimisation(training_set_selection='greedy', M_input_waveforms_PS=[40], M_output_waveforms_PS=[500], N_basis_vecs_amp_PS=np.arange(20, 40, step=5), N_basis_vecs_phase_PS=[30], minimum_spacing_list=[0.003], plot_distributions=True)

# analyse_surrogate_gpr.hyperparameter_optimisation(training_set_selection='greedy', M_input_waveforms_PS=[50], M_output_waveforms_PS=[500], N_basis_vecs_amp_PS=np.arange(25, 45, step=5), N_basis_vecs_phase_PS=np.arange(25, 45, step=5), minimum_spacing_list=[0.003], plot_distributions=True)

# analyse_surrogate_gpr.hyperparameter_optimisation(M_input_waveforms_PS=[50], M_output_waveforms_PS=[500], N_basis_vecs_amp_PS=np.arange(25, 50, step=5), N_basis_vecs_phase_PS=np.arange(25, 50, step=5), minimum_spacing_list=[0.003, 0.005], plot_distributions=True)
# analyse_surrogate_gpr.hyperparameter_optimisation(M_input_waveforms_PS=[60], M_output_waveforms_PS=[500], N_basis_vecs_amp_PS=np.arange(25, 60, step=5), N_basis_vecs_phase_PS=np.arange(25, 60, step=5), minimum_spacing_list=[0.003, 0.005], plot_distributions=True)

# analyse_surrogate.hyperparameter_optimisation(M_input_waveforms_PS=[40], M_output_waveforms_PS=[500], N_basis_vecs_amp_PS=[20,25], N_basis_vecs_phase_PS=[20,25], minimum_spacing_list=[0.004, 0.005, 0.006, 0.007, 0.008], plot_distributions=True)

# plt.show()