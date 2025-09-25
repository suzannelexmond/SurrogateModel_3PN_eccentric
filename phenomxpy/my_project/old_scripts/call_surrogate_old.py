from generate_phenom_surrogate_Singlewf import *

import numpy as np



class Generate_Online_Surrogate(Load_Offline_Surrogate):
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
        **kwargs
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

        Load_Offline_Surrogate.__init__(
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
            truncate_at_ISCO=truncate_at_ISCO,
            truncate_at_tmin=truncate_at_tmin,

        )

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

    def reconstruct_surrogate_datapiece(self, property, B_matrix, fit_matrix, plot_surr_datapiece=False, save_fig_datapiece=False):
        """
        Reconstructs the surrogate model for a given parameter using different empirical nodes for amplitude and phase.
        
        Parameters:
        ------------------
        B_matrix (numpy.ndarray), shape (m, time_samples): Empricial interpolant matrix
        fit_matrix (numpy.ndarray), shape (m, lambda): Array of fitted greedy parameters at time nodes with lambda as the number of parameters in parameter_space.
        time_samples (numpy.ndarray), shape (time_samples, 1): Array representing the time-domain samples.
        plot_surr_datapiece (bool) : Set this to True for plot of surrogate datapiece as comparison with real estimated value at given ecc_ref.
        
        Returns:
        ------------------
        surrogate_datapiece (numpy.ndarray), shape (time_samples, lambda): Array representing the reconstructed surrogate waveform datapiece (amplitude or phase).
        """
        
        computation_time = None
        m, _ = B_matrix.shape

        # fit_vector2 = np.zeros(m)

        # for i in range(m):
        #     fit_vector2[i] = fit_matrix[i].predict(self.ecc_parameter_space_output)[0]


        fit_vector = fit_matrix.T[self.ecc_ref_idx]  # Get the fit vector for the specific output eccentricity reference
        reconstructed_residual = np.sum(B_matrix * fit_vector[:, None], axis=0)

        # Change back from residual to original (+ circulair)
        surrogate_datapiece = self.residual_to_original(residual_waveform=reconstructed_residual, property=property)

        if plot_surr_datapiece is True:
            self._plot_surr_datapieces(property, surrogate_datapiece, save_fig_datapiece)

        return surrogate_datapiece, computation_time
    
    def _plot_surr_datapieces(self, property, surrogate_datapiece, save_fig_datapiece=False, geometric_units=True, total_mass=None, luminosity_distance=None):
            
        # Create a 2x1 subplot grid with height ratios 3:1
        fig_surrogate_datapieces, axs = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1}, sharex=True)

        if geometric_units is True:
            # Simulate the real waveform datapiece
            real_hp, real_hc = self.simulate_inspiral_mass_independent(self.ecc_ref)
        else:
            real_hp, real_hc = self.simulate_inspiral_mass_dependent(total_mass=total_mass, distance=luminosity_distance, ecc_ref=self.ecc_ref)

        if property == 'amplitude':
            true_datapiece = self.amplitude(real_hp, real_hc)
            units = ''
        elif property == 'phase':
            true_datapiece = self.phase(real_hp, real_hc)
            units = ' [radians]'

        # Plot Surrogate and Real Amplitude (Top Left)
        # axs[0].plot(self.ecc_parameter_space_output, surrogate_datapiece[index_ecc_ref], label='surr')
        axs[0].plot(self.time, surrogate_datapiece, linewidth=0.6, label=f'surrogate e = {self.ecc_ref}')
        # axs[0].plot(self.time, true_phase[index_ecc_ref], linewidth=0.6, label=f'Surrogate: e = {plot_surr_datapiece}')
        axs[0].plot(self.time, true_datapiece, linewidth=0.6, linestyle='dashed', label=f'true {property} e = {self.ecc_ref}')
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
        relative_error = abs(surrogate_datapiece - true_datapiece) / abs(true_datapiece)
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
            figname = f'Surrogate_{property}_ecc_ref={self.ecc_ref}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_parameter_space_input)}_{max(self.ecc_parameter_space_input)}_Ni={len(self.ecc_parameter_space_input)}]_No={len(self.ecc_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}.png'
            
            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Images/Surrogate_datapieces_Single', exist_ok=True)
            fig_surrogate_datapieces.savefig('Images/Surrogate_datapieces_Single/' + figname)

            print('Figure is saved in Images/Surrogate_datapieces_Single/' + figname)

        return surrogate_datapiece, true_datapiece, relative_error
        

    def generate_surrogate_waveform(self, ecc_ref, plot_surr_datapiece=False, save_fig_datapiece=False, plot_surr_wf=None, save_fig_surr=False, plot_GPRfit=False, save_fits_to_file=True, save_fig_fits=False, save_matrix_to_file=True):

        if isinstance(ecc_ref, float):
            try:
                self.ecc_ref_idx = np.where(self.ecc_parameter_space_output == ecc_ref)[0][0]
                self.ecc_ref = ecc_ref
            except:
                ecc_ref_asked = ecc_ref
                self.ecc_ref = self.ecc_parameter_space_output[np.abs(self.ecc_parameter_space_output - ecc_ref).argmin()]
                self.ecc_ref_idx = np.where(self.ecc_parameter_space_output == self.ecc_ref)[0][0]
                print(f'Eccentricity value for ecc_ref={ecc_ref_asked} not in ouput parameterspace. Eccentricity calculated for closest existing value at e={self.ecc_ref}.')

        
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
            if plot_GPRfit is True:
                self._plot_GPR_fits('amplitude', save_fig_fits=save_fig_fits)

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
            # Reconstruct phase datapiece
            self.surrogate_phase, computation_time_phase = self.reconstruct_surrogate_datapiece(property='phase', B_matrix=self.B_matrix_phase, fit_matrix=self.gaussian_fit_phase, plot_surr_datapiece=plot_surr_datapiece, save_fig_datapiece=save_fig_datapiece)
        else:
            self.surrogate_phase, computation_time_phase = self.reconstruct_surrogate_datapiece(property='phase', B_matrix=self.B_matrix_phase, fit_matrix=self.gaussian_fit_phase, plot_surr_datapiece=plot_surr_datapiece, save_fig_datapiece=save_fig_datapiece)
            if plot_GPRfit is True:
                self._plot_GPR_fits('phase', save_fig_fits=save_fig_fits)

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

        
        if self.geometric_units is False:
            # Convert mass-independent waveforms to a 3 dimensional mass-dependent grid of (total_mass x ecc_ref x time)
            surrogate_amp_SI, surrogate_phase_SI = self.surrogate_datapieces_from_NR_to_SI()

            h_surrogate = surrogate_amp_SI * np.exp(1j * surrogate_phase_SI)

        else:
            h_surrogate = self.surrogate_amp * np.exp(1j * self.surrogate_phase)


        if plot_surr_wf is True:
            self.plot_surrogate_waveform(h_surrogate, save_fig_surr=save_fig_surr, geometric_units=self.geometric_units)

        return self.surrogate_amp, self.surrogate_phase


    def plot_surrogate_waveform(self, surrogate_h, save_fig_surr=False, geometric_units=True):
            # Plot surrogate waveform
        fig_surrogate, axs = plt.subplots(4, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1, 3, 1], 'hspace': 0.2}, sharex=True)

        if self.geometric_units is True:
            true_hp, true_hc = self.simulate_inspiral_mass_independent(self.ecc_ref)
        else:
            true_hp, true_hc = self.simulate_inspiral_mass_dependent(total_mass=self.total_mass, distance=self.luminosity_distance, ecc_ref=self.ecc_ref)


        phase = self.phase(true_hp, true_hc)
        amp = self.amplitude(true_hp, true_hc)
        true_h = amp * np.exp(1j * phase)


        axs[0].plot(self.time, np.real(true_h), linewidth=0.6, label=f'true waveform e = {self.ecc_ref}')
        axs[0].plot(self.time, np.real(surrogate_h), linewidth=0.6, label=f'surrogate e = {self.ecc_ref}')
        axs[0].set_ylabel('$h_+$')
        axs[0].grid(True)
        axs[0].legend()

        # Calculate and Plot plus polarisation error 
        relative_error_hp = abs(np.real(surrogate_h) - np.real(true_h)) / abs(np.real(true_h))
        relative_error_hp[relative_error_hp > 1] = 0

        axs[1].plot(self.time, abs(np.real(surrogate_h) - np.real(true_h)), linewidth=0.6)
        axs[1].set_ylabel('|$h_{+, S} - h_+$|')
        axs[1].grid(True)
        # axs[1].set_ylim(0, 10)
        # axs[1].set_title('Relative error $h_x$')

        # axs[2].plot(self.time, true_hc[length_diff:], linewidth=0.6, label='True waveform before')
        axs[2].plot(self.time, np.imag(true_h), linewidth=0.6, label=f'true waveform e = {self.ecc_ref}')
        axs[2].plot(self.time, np.imag(surrogate_h), linewidth=0.6, label=f'surrogate e = {self.ecc_ref}')
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
        relative_error_hc = abs(np.imag(surrogate_h) - np.imag(true_h)) / abs(np.imag(true_h))
        relative_error_hc[relative_error_hc > 1] = 0
        axs[3].plot(self.time, abs(np.imag(surrogate_h) - np.imag(true_h)), linewidth=0.6)
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
            figname = f'Surrogate_wf_ecc_ref={self.ecc_ref}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_parameter_space_input)}_{max(self.ecc_parameter_space_input)}_Ni={len(self.ecc_parameter_space_input)}]_No={len(self.ecc_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}.png'
            
            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Images/Surrogate_wf', exist_ok=True)
            fig_surrogate.savefig('Images/Surrogate_wf/' + figname)

            print('Figure is saved in Images/Surrogate_wf/' + figname)

        return surrogate_h, true_h, relative_error_hp, relative_error_hc

        
    
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


class Call_Surrogate(Generate_Online_Surrogate): 

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
        **kwargs
    ):

        Generate_Online_Surrogate.__init__(
            self,
            time_array=time_array,
            ecc_ref=ecc_ref,
            total_mass=total_mass,
            luminosity_distance=luminosity_distance,
            f_lower=f_lower,
            f_ref=f_ref,
            chi1=chi1,
            chi2=chi2,
            phiRef=phiRef,
            rel_anomaly=rel_anomaly,
            inclination=inclination,
            truncate_at_ISCO=truncate_at_ISCO,
            truncate_at_tmin=truncate_at_tmin,
        )

    
        
    def load_offline_surrogate(self, plot_GPRfit=False, save_fig_fits=False):
        """Load precomputed surrogate data and assign it to the class parameters."""

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
            self.gaussian_fit_amp = data['gaussian_fit_amp']
            self.empirical_nodes_idx_amp = data['empirical_nodes_idx_amp']
            self.residual_greedy_basis_amp = data['residual_greedy_basis_amp']
            self.B_matrix_amp = data['B_matrix_amp']

            # Phase data
            self.gaussian_fit_phase = data['gaussian_fit_phase']
            self.empirical_nodes_idx_phase = data['empirical_nodes_idx_phase']
            self.residual_greedy_basis_phase = data['residual_greedy_basis_phase']
            self.B_matrix_phase = data['B_matrix_phase']

            # Indices and time array
            self.greedy_parameters_idx_amp = data['greedy_parameters_idx_amp']
            self.greedy_parameters_idx_phase = data['greedy_parameters_idx_phase']
            self.time = data['time']


            # Circular waveform properties
            self.amp_circ = data['amp_circ']
            self.phase_circ = data['phase_circ']

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

        if self.surrogate_amp is None or self.surrogate_phase is None:
            print('Load surrogate ...')
            start = time.time()
            self.load_offline_surrogate()
            print('Load offline surrogate. Time taken:', time.time() - start)
        else:
            print('Surrogate already loaded, skipping loading step.')
        
        start = time.time()
        self.surrogate_amp, self.surrogate_phase = self.generate_surrogate_waveform(
            ecc_ref=self.ecc_ref,
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
        if (geometric_units is False) and ((total_mass is None) and (self.total_mass is None)) or ((luminosity_distance is None) and (self.luminosity_distance is None)):
            raise ValueError("For SI units, please provide total_mass and luminosity_distance.")
        elif (geometric_units is False) and ((self.total_mass is not None) or (self.luminosity_distance is not None)):
            if total_mass is None:
                total_mass = self.total_mass
            if luminosity_distance is None:
                luminosity_distance = self.luminosity_distance
        elif (geometric_units is False) and ((total_mass is not None) or (luminosity_distance is not None)):
            # Update class attributes if provided in function call
            self.luminosity_distance = luminosity_distance
            self.total_mass = total_mass
        elif (geometric_units is True) and ((total_mass is not None) or (luminosity_distance is not None)):
            total_mass = None
            luminosity_distance = None
            raise ValueError("For geometric units, please do not provide total_mass and luminosity_distance. Parameters are automatically set to total_mass=None and luminosity_distance=None.")
        
     
        print(total_mass, luminosity_distance, self.surrogate_amp)
        # Convert the surrogate amplitude and phase to polarisation amplitudes
        self.hplus, self.hcross = self.polarisations(phase=self.surrogate_phase, amplitude=self.surrogate_amp, geometric_units=geometric_units, distance=luminosity_distance, total_mass=total_mass, plot_polarisations=plot_polarisations, save_fig=save_fig)
        
        return self.hplus, self.hcross
    
sampling_frequency = 2048 # or 4096
duration = 4 # seconds
time_array = np.linspace(-duration, 0, int(sampling_frequency * duration))  # time in seconds


online = Call_Surrogate(
        time_array,
        total_mass=50,
        luminosity_distance=200,
        f_lower=11,
        f_ref=20,
        chi1=0,
        chi2=0,
        phiRef=0.,
        rel_anomaly=0.,
        inclination=0.,
        truncate_at_ISCO=True,
        truncate_at_tmin=True,
    )



for ecc in np.linspace(0.01, 0.2, 5):
    if ecc==0.01:
        online.generate_PhenomTE_surrogate(ecc_ref=ecc, plot_GPRfit=True, plot_surr_datapiece=True, plot_surr_wf=True, save_fig_datapiece=True, save_fig_fits=True, save_fig_surr=True)
    else:
        online.generate_PhenomTE_surrogate(ecc_ref=ecc, plot_surr_datapiece=True, plot_surr_wf=True, save_fig_datapiece=True, save_fig_fits=True, save_fig_surr=True)

plt.show()

hp, hc = online.get_surrogate_polarisations(geometric_units=False, plot_polarisations=True, save_fig=True)
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
# online.generate_PhenomTE_surrogate(ecc_ref=0.1, geometric_units=True, plot_GPRfit=True)
# end3 = time.time()
# print(f"Surrogate generation took {time.time() - start3:.4f} seconds.")
# plt.show()



# online.generate_PhenomTE_surrogate(ecc_ref=0.1, plot_surr_datapiece=True, save_fig_datapiece=True, plot_surr_wf=True, save_fig_surr=True, plot_GPRfit=True, save_fig_fits=True, geometric_units=True)
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
