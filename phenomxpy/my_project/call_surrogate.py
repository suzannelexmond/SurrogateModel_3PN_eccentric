from generate_phenom_surrogate_Singlewf import *

import numpy as np
from pathlib import Path

class Generate_Surrogate_Offline(Generate_Surrogate):
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
        self.time_array = time_array
        self.chi1 = chi1
        self.chi2 = chi2
        self.phiRef = phiRef
        self.rel_anomaly = rel_anomaly
        self.inclination = inclination
        self.f_lower = f_lower
        self.f_ref = f_ref
        self.reference_total_mass = reference_total_mass
        self.reference_luminosity_distance = reference_luminosity_distance
        self.N_greedy_vecs_amp = N_greedy_vecs_amp
        self.N_greedy_vecs_phase = N_greedy_vecs_phase
        self.min_greedy_error_amp = min_greedy_error_amp
        self.min_greedy_error_phase = min_greedy_error_phase
        self.ecc_ref_parameterspace_range = ecc_ref_parameterspace_range
        self.amount_input_wfs = amount_input_wfs
        self.amount_output_wfs = amount_output_wfs
        self.truncate_at_ISCO = truncate_at_ISCO
        self.truncate_at_tmin = truncate_at_tmin
        self.waveforms_in_geom_units = waveforms_in_geom_units
        
      
        # Initialize surrogate generation
        self.surrogate = Generate_Surrogate(
            self.time_array,
            self.ecc_ref_parameterspace_range,
            self.amount_input_wfs,
            self.amount_output_wfs,
            self.reference_total_mass,
            self.reference_luminosity_distance,
            self.N_greedy_vecs_amp,
            self.N_greedy_vecs_phase,
            self.min_greedy_error_amp,
            self.min_greedy_error_phase,
            self.f_lower,
            self.f_ref,
            self.chi1,
            self.chi2,
            self.phiRef,
            self.rel_anomaly,
            self.inclination,
            self.truncate_at_ISCO,
            self.truncate_at_tmin,
            self.waveforms_in_geom_units
        )

        Generate_Surrogate.__init__(self, time_array=time_array, ecc_ref_parameterspace_range=ecc_ref_parameterspace_range, reference_total_mass=self.reference_total_mass, reference_luminosity_distance=self.reference_luminosity_distance, amount_input_wfs=amount_input_wfs, amount_output_wfs=amount_output_wfs, N_greedy_vecs_amp=N_greedy_vecs_amp, N_greedy_vecs_phase=N_greedy_vecs_phase, min_greedy_error_amp=min_greedy_error_amp, min_greedy_error_phase=min_greedy_error_phase, f_lower=f_lower, f_ref=f_ref, chi1=chi1, chi2=chi2, phiRef=phiRef, rel_anomaly=rel_anomaly, inclination=inclination, truncate_at_ISCO=truncate_at_ISCO, truncate_at_tmin=truncate_at_tmin, geometric_units=self.waveforms_in_geom_units)

        # self._fit_and_save()

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
            self.surrogate.fit_to_training_set(
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
            self.surrogate.residual_greedy_basis = data['residual_greedy_basis']
            self.surrogate.empirical_nodes_idx = data['empirical_nodes']

            self.surrogate.compute_B_matrix(
                    property=property,
                    save_matrix_to_file=True
                )
            data = np.load(filename)
        
        return {
            'B_matrix': data['B_matrix']
        }





class Generate_Surrogate_Online(Generate_Surrogate_Offline):

    def __init__(
        self,
        time_array,
        ecc_ref=None,
        total_mass=None,
        luminosity_distance=None,
        f_lower=10,
        f_ref=20,
        chi1=0,
        chi2=0,
        phiRef=0.,
        rel_anomaly=0.,
        inclination=0.,
        truncate_at_ISCO=True,
        truncate_at_tmin=True,
    ):

        self.total_mass = total_mass
        self.ecc_ref = ecc_ref
        self.luminosity_distance = luminosity_distance
        self.f_lower = f_lower
        self.f_ref = f_ref
        self.chi1 = chi1
        self.chi2 = chi2
        self.phiRef = phiRef
        self.rel_anomaly = rel_anomaly
        self.inclination = inclination  
        self.truncate_at_ISCO = truncate_at_ISCO
        self.truncate_at_tmin = truncate_at_tmin

        self.surrogate_amp = None
        self.surrogate_phase = None

        Generate_Surrogate_Offline.__init__(
            self,
            time_array=time_array,
            ecc_ref_parameterspace_range=[0.0, 0.3],
            reference_total_mass=60,
            reference_luminosity_distance=200,
            amount_input_wfs=60,
            amount_output_wfs=500,
            N_greedy_vecs_amp=40,
            N_greedy_vecs_phase=40,
            f_lower=self.f_lower,
            f_ref=self.f_ref,
            chi1=self.chi1,
            chi2=self.chi2,
            phiRef=self.phiRef,
            rel_anomaly=self.rel_anomaly,
            inclination=self.inclination,
            truncate_at_ISCO=True,
            truncate_at_tmin=True,

        )


        
    def load_offline_surrogate(self, plot_GPRfit=False, save_fig_fits=False):
        """Load precomputed surrogate data and assign it to self.surrogate."""

        try:
            start = time.time()
            # Load the precomputed surrogate data
            data = np.load( 
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
            f"Nga={self.N_greedy_vecs_amp}.npz", allow_pickle=True)


            # Amplitude data
            self.surrogate.gaussian_fit_amp = data['gaussian_fit_amp']
            self.surrogate.empirical_nodes_idx_amp = data['empirical_nodes_idx_amp']
            self.surrogate.residual_greedy_basis_amp = data['residual_greedy_basis_amp']
            self.surrogate.B_matrix_amp = data['B_matrix_amp']

            # Phase data
            self.surrogate.gaussian_fit_phase = data['gaussian_fit_phase']
            self.surrogate.empirical_nodes_idx_phase = data['empirical_nodes_idx_phase']
            self.surrogate.residual_greedy_basis_phase = data['residual_greedy_basis_phase']
            self.surrogate.B_matrix_phase = data['B_matrix_phase']

            # Indices and time array
            self.surrogate.greedy_parameters_idx_amp = data['greedy_parameters_idx_amp']
            self.surrogate.greedy_parameters_idx_phase = data['greedy_parameters_idx_phase']
            self.surrogate.time = data['time']


            # Circular waveform properties
            self.surrogate.amp_circ = data['amp_circ']
            self.surrogate.phase_circ = data['phase_circ']

            print("Surrogate model loaded successfully. Time taken:", time.time() - start)

            if plot_GPRfit is True:
                self._plot_GPR_fits('phase', save_fig_fits=save_fig_fits)
                self._plot_GPR_fits('amplitude', save_fig_fits=save_fig_fits)


        except:
            print("Surrogate model not found. Generating new surrogate data...")
            # Generate surrogate data and save it
            self.create_offline_surrogate()
            # Try loading again
            self.load_offline_surrogate(plot_GPRfit=plot_GPRfit, save_fig_fits=save_fig_fits)



    
    def generate_PhenomTE_surrogate(self, ecc_ref=None, plot_surr_datapiece=None, save_fig_datapiece=False, plot_surr_wf=None, save_fig_surr=False, plot_GPRfit=False, save_fig_fits=False):
        """
        Call the surrogate model with the given output eccentricity reference.
        If plot_surr_wf is True, it will plot the surrogate waveform against the real waveform.
        If plot_surr_datapiece is True, it will plot the surrogate datapiece against the real datapiece.
        """
        # Either set in class object or in function specific
        if ecc_ref is None:
            ecc_ref = self.ecc_ref
        else:
            self.ecc_ref = ecc_ref

        if self.surrogate.surrogate_amp is None or self.surrogate.surrogate_phase is None:
            print('Load surrogate ...')
            start = time.time()
            self.load_offline_surrogate()
            print('Load offline surrogate. Time taken:', time.time() - start)
        else:
            print('Surrogate already loaded, skipping loading step.')
        
        start = time.time()
        self.surrogate_amp, self.surrogate_phase = self.surrogate.generate_surrogate_waveform(
            output_ecc_ref=self.ecc_ref,
            plot_surr_datapiece=plot_surr_datapiece,
            save_fig_datapiece=save_fig_datapiece,
            plot_surr_wf=plot_surr_wf,
            save_fig_surr=save_fig_surr,
            plot_GPRfit=plot_GPRfit,
            save_fits_to_file=False,
            save_fig_fits=save_fig_fits
        )
        print('Surrogate calculated. Time taken:', time.time() - start)

        return self.surrogate_amp, self.surrogate_phase

    def get_surrogate_polarisations(self, total_mass=None, luminosity_distance=None, geometric_units=False, plot_polarisations=False, save_fig=False):
        """ Get the polarisation amplitudes for the surrogate waveform.
        If geometric_units is True, it will return the polarisation amplitudes in geometric units.
        If geometric_units is False, it will return the polarisation amplitudes in SI units.
        """ 
        if (geometric_units is False) and ((total_mass is None) or (luminosity_distance is None)):
            raise ValueError("For SI units, please provide total_mass and luminosity_distance.")
        elif geometric_units is True:
            total_mass = None
            luminosity_distance = None
            raise ValueError("For geometric units, please do not provide total_mass and luminosity_distance. Parameters are automatically set to total_mass=None and luminosity_distance=None.")
        
        if geometric_units is False:
            # If SI units are requested, ensure total_mass and luminosity_distance are set. 
            # Use class attributes if not provided in function call.
            if total_mass is None:
                total_mass = self.total_mass
            if luminosity_distance is None:
                luminosity_distance = self.luminosity_distance
            else:
                # Update class attributes if provided in function call
                self.luminosity_distance = luminosity_distance
                self.total_mass = total_mass
        print(total_mass, luminosity_distance, self.surrogate_amp)
        # Convert the surrogate amplitude and phase to polarisation amplitudes
        self.hplus, self.hcross = self.polarisations(phase=self.surrogate_phase, amplitude=self.surrogate_amp, geometric_units=geometric_units, distance=luminosity_distance, total_mass=total_mass, plot_polarisations=plot_polarisations, save_fig=save_fig)
        
        return self.hplus, self.hcross
    
# sampling_frequency = 2048 # or 4096
# duration = 4 # seconds
# time_array = np.linspace(-duration, 0, int(sampling_frequency * duration))  # time in seconds


# online = Generate_Surrogate_Online(
#         time_array,
#         total_mass=50,
#         luminosity_distance=200,
#         f_lower=11,
#         f_ref=20,
#         chi1=0,
#         chi2=0,
#         phiRef=0.,
#         rel_anomaly=0.,
#         inclination=0.,
#         truncate_at_ISCO=True,
#         truncate_at_tmin=True,
#     )


# Generate the surrogate model with the specified output eccentricity reference
# start1 = time.time()

# for ecc in np.linspace(0.01, 0.2, 5):
#     if ecc==0.01:
#         online.generate_PhenomTE_surrogate(ecc_ref=ecc, plot_GPRfit=True, plot_surr_datapiece=True, plot_surr_wf=True, save_fig_datapiece=True, save_fig_fits=True, save_fig_surr=True)
#     else:
#         online.generate_PhenomTE_surrogate(ecc_ref=ecc, plot_surr_datapiece=True, plot_surr_wf=True, save_fig_datapiece=True, save_fig_fits=True, save_fig_surr=True)

# plt.show()
# print(1, online.time)
# hp, hc = online.get_surrogate_polarisations(geometric_units=False)
# print(2, len(online.time), len(online.hplus))

# online_true = Generate_Surrogate_Online(
#         time_array,
#         total_mass=50,
#         luminosity_distance=200,
#         f_lower=10,
#         f_ref=20,
#         chi1=0,
#         chi2=0,
#         phiRef=0.,
#         rel_anomaly=0.,
#         inclination=0.,
#         truncate_at_ISCO=True,
#         truncate_at_tmin=True,
#     )

# hp_true, hc_true, time_true = online_true.simulate_inspiral_mass_independent(ecc_ref=0.1, custom_time_array=SecondtoMass(time_array, 50))
# print(20, time_true, hp_true)
# print(21, online.time[-len(hp):], hp)
# fig_20, ax = plt.subplots()
# ax.plot(online.time[-len(hp):], hp, label='surr')
# # ax.plot(MasstoSecond(time_true, 50), AmpNRtoSI(hp_true, 200, 50), label='true')
# ax.legend()
# plt.show()


# print(f"Surrogate loading took {time.time() - start1:.4f} seconds.")
# offline = Generate_Surrogate_Offline(time_array=time_array,
#             ecc_ref_parameterspace_range=[0.0, 0.3],
#             total_mass_range=[60, 100],
#             luminosity_distance_range=[200, 500],
#             amount_input_wfs=60,
#             amount_output_wfs=500,
#             N_greedy_vecs_amp=30,
#             N_greedy_vecs_phase=30,
#             f_lower=10,
#             f_ref=20,
#             chi1=0,
#             chi2=0,
#             phiRef=0,
#             rel_anomaly=0,
#             inclination=0,
#             truncate_at_ISCO=True,
#             truncate_at_tmin=True)

# offline.fit_to_training_set(property='amplitude', plot_fits=True, save_fig_fits=True, save_fits_to_file=True, N_greedy_vecs=30)
# offline.fit_to_training_set(property='phase', plot_fits=True, save_fig_fits=True, save_fits_to_file=True, N_greedy_vecs=30)

# online.get_training_set('phase', N_greedy_vecs=40, plot_emp_nodes_at_ecc=0.2, plot_training_set=True, save_fig_emp_nodes=True, save_fig_training_set=True, plot_greedy_vecs=True)
# online.get_training_set('amplitude', N_greedy_vecs=40, plot_emp_nodes_at_ecc=0.2, plot_training_set=True, save_fig_emp_nodes=True, save_fig_training_set=True, plot_greedy_vecs=True)
# start3 = time.time()
# online.generate_PhenomTE_surrogate(output_ecc_ref=0.1, geometric_units=True, plot_GPRfit=True)
# end3 = time.time()
# print(f"Surrogate generation took {time.time() - start3:.4f} seconds.")
# plt.show()



# online.generate_PhenomTE_surrogate(output_ecc_ref=0.1, plot_surr_datapiece=True, save_fig_datapiece=True, plot_surr_wf=True, save_fig_surr=True, plot_GPRfit=True, save_fig_fits=True, geometric_units=True)
# hplus, hcross = online.get_surrogate_polarisations(geometric_units=True)


# phen = phenomt.PhenomTE(
#             mode=[2,2],
#             times=time_array,
#             eccentricity=0.14,                
#             f_ref=20,                   
#             f_lower=10,
#             phiRef=0,
#             inclination=0)
        
# phen.compute_polarizations(times=time_array)
# start2 = time.time()
# phen = phenomt.PhenomTE(
#             mode=[2,2],
#             times=time_array,
#             eccentricity=0.1,                
#             f_ref=20,                   
#             f_lower=10,
#             phiRef=0,
#             inclination=0)
        
# phen.compute_polarizations(times=time_array)

# end2 = time.time()
# print(f"Simulation took {end2 - start2:.4f} seconds.")

# print(f'Total surrogate improvement speed: {(end2 - start2)/(end3 - start3)}')

# sp = Simulate_Inspiral(
#     time_array=time_array,
#     luminosity_distance=300,
#     total_mass=80,
#     ecc_ref=0.1)
# sp.simulate_inspiral_mass_independent(ecc_ref=0.1)
