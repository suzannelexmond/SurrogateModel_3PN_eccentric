from generate_phenom_surrogate import *

import numpy as np
from pathlib import Path

class Generate_Online_Surrogate_Testing(Load_Offline_Surrogate):

    def __init__(
        self,
        time_array,
        ecc_ref_parameterspace_range=[0.0, 0.3],
        amount_input_wfs=60,
        amount_output_wfs=500,
        N_basis_vecs_amp=40,
        N_basis_vecs_phase=40,
        training_set_selection='GPR_opt',
        minimum_spacing_greedy=0.008,
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
        self.output_ecc_ref = ecc_ref
        self.luminosity_distance = luminosity_distance

        self._printed_surrogate_loaded = False

        Load_Offline_Surrogate.__init__(
            self,
            time_array=time_array,
            ecc_ref_parameterspace_range=ecc_ref_parameterspace_range,
            amount_input_wfs=amount_input_wfs,
            amount_output_wfs=amount_output_wfs,
            N_basis_vecs_amp=N_basis_vecs_amp,
            N_basis_vecs_phase=N_basis_vecs_phase,
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
        plot_surr_datapiece (bool) : Set this to True for plot of surrogate datapiece as comparison with real estimated value at given output_ecc_ref.
        
        Returns:
        ------------------
        surrogate_datapiece (numpy.ndarray), shape (time_samples, lambda): Array representing the reconstructed surrogate waveform datapiece (amplitude or phase).
        """
        
        m, _ = B_matrix.shape

        fit_vector = fit_matrix.T[self.output_ecc_ref_idx]  # Get the fit vector for the specific output eccentricity reference
        reconstructed_residual = np.sum(B_matrix * fit_vector[:, None], axis=0)

        # Change back from residual to original (+ circulair)
        surrogate_datapiece = self.residual_to_original(residual_waveform=reconstructed_residual, property=property)
        

        # Surrogate datapieces in specified units
        if (self.geometric_units is False) and (property == 'amplitude'):
            surrogate_datapiece = AmpNRtoSI(surrogate_datapiece, self.luminosity_distance, self.total_mass)

        # Set class objects
        if property == 'phase':
            self.surrogate_phase = surrogate_datapiece
        else:
            self.surrogate_amp = surrogate_datapiece

        # Plotting option
        if plot_surr_datapiece is True:
                self._plot_surr_datapieces(property, save_fig_datapiece)

        return surrogate_datapiece
    

    def calculate_relative_error(self, ecc_ref=None,
                             get_true_and_surrogate_output=False,
                             get_computation_time_PhenomTE=False,
                             property=None):
        """
        Compute relative error for waveform amplitude, phase, and polarizations.
        """
        if self.output_ecc_ref is None:
            self.output_ecc_ref = ecc_ref

        start = time.time()
        true_hp, true_hc = self.simulate_inspiral(
            ecc_ref=self.output_ecc_ref,
            geometric_units=True,
            truncate_at_ISCO=False,
            truncate_at_tmin=False
        )
        computation_time_PhenomTE = time.time() - start

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
        true_h = true_amp * np.exp(1j * true_phase)
        surrogate_h = self.surrogate_amp * np.exp(1j * self.surrogate_phase)

        # Relative error helper
        def relative_err(surrogate_result, true_result):
            relative_error = np.abs(surrogate_result - true_result) / np.abs(true_result)
            return np.nan_to_num(relative_error, nan=0.0)

        # Compute all relative errors
        rel_amp = relative_err(self.surrogate_amp, true_amp)
        rel_phase = np.abs(self.surrogate_phase - true_phase)
        rel_h = relative_err(surrogate_h, true_h)

        # Map properties
        prop_map = {
            'amplitude': (rel_amp, self.surrogate_amp, true_amp),
            'phase':     (rel_phase, self.surrogate_phase, true_phase),
            'waveform':  (rel_h, surrogate_h, true_h)
        }

        # Determine which property to return
        if property is None:
            # Return all relative errors as a tuple or dict
            result = (rel_amp, rel_phase, rel_h)
            if get_true_and_surrogate_output:
                result = {
                    'amplitude': (self.surrogate_amp, true_amp, rel_amp),
                    'phase':     (self.surrogate_phase, true_phase, rel_phase),
                    'waveform':  (surrogate_h, true_h, rel_h)
                }
        else:
            # Property-specific return
            base = prop_map[property]
            if get_true_and_surrogate_output:
                result = base  # full tuple: (rel, surrogate, true)
            else:
                result = base[0]  # just relative error

        # Optionally append computation time
        if get_computation_time_PhenomTE:
            if isinstance(result, tuple):
                result = (*result, computation_time_PhenomTE)
            elif isinstance(result, dict):
                result['computation_time'] = computation_time_PhenomTE
            else:
                # single value, wrap in tuple
                result = (result, computation_time_PhenomTE)

        return result


    def _plot_surr_datapieces(self, property, save_fig_datapiece=False, ecc_ref=None):

        relative_error, surrogate_datapiece, true_datapiece  = self.calculate_relative_error(ecc_ref=ecc_ref, property=property)
        
        # Plot difference between surrogate and real datapiece at given eccentricity reference
        fig_surrogate_datapieces, axs = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1}, sharex=True)

        axs[0].plot(self.time, surrogate_datapiece, linewidth=0.6, label=f'surrogate e = {self.output_ecc_ref}')
        axs[0].plot(self.time, true_datapiece, linewidth=0.6, linestyle='dashed', label=f'true {property} e = {self.output_ecc_ref}')
        
        if property == 'phase':
            axs[0].set_ylabel('$\phi$ [radians]' )
        else:
            axs[0].set_ylabel('A')
        axs[0].grid(True)
        # axs[0].set_title(f'Surrogate vs Real {property}, ga={self.min_greedy_error_amp}, gp={self.min_greedy_error_phase}')
        axs[0].legend(loc='upper left')

        # Calculate and plot relative error between surrogate and real datapiece
        
        axs[1].plot(self.time, relative_error, linewidth=0.6)
        if property == 'phase':
            axs[1].set_ylabel('|($\phi_S$ - $\phi$)|')
        else:
            axs[1].set_ylabel('|($A_S$ - A) / A|')
        axs[1].set_xlabel('t [M]')
        axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
        axs[1].grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        if save_fig_datapiece is True:
            figname = f'Images/Surrogate_datapiece_Single/Surrogate_{property}_ecc_ref={self.output_ecc_ref}_M={self.total_mass}_l_dist={self.luminosity_distance}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_ref_parameter_space_input)}_{max(self.ecc_ref_parameter_space_input)}_Ni={len(self.ecc_ref_parameter_space_input)}]_No={len(self.ecc_ref_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_basis_vecs_phase}_Nga={self.N_basis_vecs_amp}.png'
            
            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Images/Surrogate_datapieces_Single', exist_ok=True)
            fig_surrogate_datapieces.savefig(figname)

            print(self.colored_text(f'Figure is saved in {figname}', 'blue'))

        

    def generate_surrogate_waveform(self, output_ecc_ref, plot_surr_datapiece=False, save_fig_datapiece=False, plot_surr_wf=None, save_fig_surr=False, plot_GPRfit=False, save_fits_to_file=True, save_fig_fits=False, save_matrix_to_file=True):

        if isinstance(output_ecc_ref, float):
            try:
                self.output_ecc_ref_idx = np.where(np.round(self.ecc_ref_parameter_space_output, 4) == np.round(output_ecc_ref, 4))[0][0]
                self.output_ecc_ref = output_ecc_ref
            except:
                
                output_ecc_ref_asked = output_ecc_ref
                self.output_ecc_ref = self.ecc_ref_parameter_space_output[np.abs(self.ecc_ref_parameter_space_output - output_ecc_ref).argmin()]
                self.output_ecc_ref_idx = np.where(self.ecc_ref_parameter_space_output == self.output_ecc_ref)[0][0]
                # print(f'Eccentricity value for output_ecc_ref={np.round(output_ecc_ref_asked, 4)} not in output parameterspace. Eccentricity calculated for closest existing value at e={np.round(self.output_ecc_ref, 4)}.')

        if self.gaussian_fit_amp is None:
            print('Loading surrogate amplitude...')
            # Set timer for computational time of the surrogate model
            # start_time_amp = time.time()

            # Get matrix with interpolated fits and B_matrix
            if self.training_set_selection == 'GPR_opt':
                self.gaussian_fit_amp = self.fit_to_training_set_GPR_opt(N_basis_vecs=self.N_basis_vecs_amp, property='amplitude', plot_GPR_fits=plot_GPRfit, save_fig_GPR_fits=save_fig_fits, save_fits_to_file=save_fits_to_file)[0]
            elif self.training_set_selection == 'greedy':
                self.gaussian_fit_amp = self.fit_to_training_set(min_greedy_error=self.min_greedy_error_amp, N_basis_vecs=self.N_basis_vecs_amp, property='amplitude', plot_GPR_fits=plot_GPRfit, save_fig_GPR_fits=save_fig_fits, save_fits_to_file=save_fits_to_file)[0]
            # Get empirical nodes for amplitude
            self.empirical_nodes_idx_amp = self.empirical_nodes_idx
            # Get residual greedy basis of amplitude
            self.residual_reduced_basis_amp = self.residual_reduced_basis
            
        if self.B_matrix_amp is None:
            # Get B_matrix for amplitude
            self.B_matrix_amp = self.compute_B_matrix(property='amplitude', save_matrix_to_file=save_matrix_to_file)
            # Reconstruct amplitude datapiece
            self.surrogate_amp = self.reconstruct_surrogate_datapiece(property='amplitude', B_matrix=self.B_matrix_amp, fit_matrix=self.gaussian_fit_amp, plot_surr_datapiece=plot_surr_datapiece, save_fig_datapiece=save_fig_datapiece)

        else:
            # print('Reconstruct surrogate datapiece...')
            self.surrogate_amp = self.reconstruct_surrogate_datapiece(property='amplitude', B_matrix=self.B_matrix_amp, fit_matrix=self.gaussian_fit_amp, plot_surr_datapiece=plot_surr_datapiece, save_fig_datapiece=save_fig_datapiece)
            if plot_GPRfit is True:
                self._plot_GPR_fits('amplitude', save_fig_fits=save_fig_fits)


        if self.gaussian_fit_phase is None:
            print('Loading surrogate phase...')
            # Set timer for computational time of the surrogate model
            # start_time_phase = time.time()

            # Get matrix with interpolated fits and B_matrix
            start1 = time.time()

            if self.training_set_selection == 'GPR_opt':
                self.surrogate_phase = self.fit_to_training_set_GPR_opt(N_basis_vecs=self.N_basis_vecs_phase, property='phase', plot_fits=plot_GPRfit, save_fig_fits=save_fig_fits, save_fits_to_file=save_fits_to_file)[0]
            elif self.training_set_selection == 'greedy':
                self.gaussian_fit_phase = self.fit_to_training_set(min_greedy_error=self.min_greedy_error_phase, N_basis_vecs=self.N_basis_vecs_phase, property='phase', plot_fits=plot_GPRfit, save_fig_fits=save_fig_fits, save_fits_to_file=save_fits_to_file)[0]
            
            print(f'GPR fit phase took {time.time() - start1:.4f}s')
            # Get empirical nodes of phase
            start2 = time.time()
            self.empirical_nodes_idx_phase = self.empirical_nodes_idx
            # Get residual greedy basis of phase
            self.residual_reduced_basis_phase = self.residual_reduced_basis
            
            print(f'Setting self took {time.time() - start2:.4f}s')
            start3 = time.time()

        if self.B_matrix_phase is None:
            # Get B_matrix for phase
            self.B_matrix_phase = self.compute_B_matrix(property='phase', save_matrix_to_file=save_matrix_to_file)
            print(f'B_matrix took {time.time() - start3:.4f}s')
            # Reconstruct phase datapiece
            self.surrogate_phase = self.reconstruct_surrogate_datapiece(property='phase', B_matrix=self.B_matrix_phase, fit_matrix=self.gaussian_fit_phase, plot_surr_datapiece=plot_surr_datapiece, save_fig_datapiece=save_fig_datapiece)
        else:
            self.surrogate_phase = self.reconstruct_surrogate_datapiece(property='phase', B_matrix=self.B_matrix_phase, fit_matrix=self.gaussian_fit_phase, plot_surr_datapiece=plot_surr_datapiece, save_fig_datapiece=save_fig_datapiece)
            
            
            if plot_GPRfit is True:
                self._plot_GPR_fits('phase', save_fig_fits=save_fig_fits)

        # # End timer for computation of surrogate model
        # end_time_phase = time.time()

        # # Compute total computational time of the surrogate datapieces
        # if computation_time_phase is None:
        #     computation_time_amp = end_time_amp - start_time_amp
        #     computation_time_phase = end_time_phase - start_time_phase

        # filename = f'Straindata/Surrogate_datapieces/Surrogate_datapieces_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_ref_parameter_space_input)}_{max(self.ecc_ref_parameter_space_input)}_N={len(self.ecc_ref_parameter_space_input)}]_No={len(self.ecc_ref_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_basis_vecs_phase}_Nga={self.N_basis_vecs_amp}.npz'
        # if save_surr_to_file is True and not os.path.isfile(filename):
        #     # Ensure the directory exists, creating it if necessary and save
        #     os.makedirs('Straindata/Surrogate_datapieces', exist_ok=True)
        #     np.savez(filename, surrogate_amp=self.surrogate_amp, surrogate_phase=self.surrogate_phase, computation_t_amp=computation_time_amp, computation_t_phase=computation_time_phase, time=self.time)
        #     print('Surrogate datapieces saved in ' + filename)


        if plot_surr_wf is True:
            self.plot_surrogate_waveform(save_fig_surr=save_fig_surr, geometric_units=self.waveforms_in_geom_units)

        return self.surrogate_amp, self.surrogate_phase


    def plot_surrogate_waveform(self, save_fig_surr=False, geometric_units=True):
            # Plot surrogate waveform

        surrogate_h, true_h, relative_error = self.calculate_relative_error('waveform')

        if self.waveforms_in_geom_units is True:
            # true_hp, true_hc = self.simulate_inspiral(self.output_ecc_ref)
            surrogate_h = AmpNRtoSI(surrogate_h, self.luminosity_distance, self.total_mass)
            true_h = AmpNRtoSI(true_h, self.luminosity_distance, self.total_mass)
        # else:
            # true_hp, true_hc = self.simulate_inspiral(total_mass=self.total_mass, distance=self.luminosity_distance, ecc_ref=self.output_ecc_ref, geometric_units=False)


        # phase = self.phase(true_hp, true_hc)
        # amp = self.amplitude(true_hp, true_hc)
        # true_h = amp * np.exp(1j * phase)

        fig_surrogate, axs = plt.subplots(4, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1, 3, 1], 'hspace': 0.2}, sharex=True)

        axs[0].plot(self.time, np.real(true_h), linewidth=0.6, label=f'true waveform e = {self.output_ecc_ref}')
        axs[0].plot(self.time, np.real(surrogate_h), linewidth=0.6, label=f'surrogate e = {self.output_ecc_ref}')
        axs[0].set_ylabel('$h_+$')
        axs[0].grid(True)
        axs[0].legend()

        # Calculate and Plot plus polarisation error 
        # relative_error_hp = abs(np.real(surrogate_h) - np.real(true_h)) / abs(np.real(true_h))
        # relative_error_hp[relative_error_hp > 1] = 0

        axs[1].plot(self.time, abs(np.real(surrogate_h) - np.real(true_h)), linewidth=0.6)
        axs[1].set_ylabel('|$h_{+, S} - h_+$|')
        axs[1].grid(True)
        # axs[1].set_ylim(0, 10)
        # axs[1].set_title('Relative error $h_x$')

        # axs[2].plot(self.time, true_hc[length_diff:], linewidth=0.6, label='True waveform before')
        axs[2].plot(self.time, np.imag(true_h), linewidth=0.6, label=f'true waveform e = {self.output_ecc_ref}')
        axs[2].plot(self.time, np.imag(surrogate_h), linewidth=0.6, label=f'surrogate e = {self.output_ecc_ref}')
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
        # relative_error_hc = abs(np.imag(surrogate_h) - np.imag(true_h)) / abs(np.imag(true_h))
        # relative_error_hc[relative_error_hc > 1] = 0
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
            figname = f'Images/Surrogate_wf/Surrogate_wf_ecc_ref={self.output_ecc_ref}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_ref_parameter_space_input)}_{max(self.ecc_ref_parameter_space_input)}_Ni={len(self.ecc_ref_parameter_space_input)}]_No={len(self.ecc_ref_parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_basis_vecs_phase}_Nga={self.N_basis_vecs_amp}.png'
            
            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Images/Surrogate_wf', exist_ok=True)
            fig_surrogate.savefig(figname)

            print(self.colored_text(f'Figure is saved in {figname}', 'blue'))

        return surrogate_h, true_h, relative_error

    
    # def surrogate_datapieces_from_NR_to_SI(self):
    #     # Phase is already unitless so doesn't need converting
    #     surrogate_amp_SI = np.zeros((len(self.total_mass_range), len(self.ecc_ref_parameter_space_output), len(self.time))) #
    #     surrogate_phase_SI = np.zeros((len(self.total_mass_range), len(self.ecc_ref_parameter_space_output), len(self.time))) #

    #     for total_mass, distance in zip(self.total_mass_range, self.luminosity_distance_range):
    #         self.time = MasstoSecond(self.time, total_mass)
    #         for ecc_ref in self.ecc_ref_parameter_space_output:
    #             surrogate_amp_SI[total_mass, ecc_ref, :] = AmpNRtoSI(self.surrogate_amp.T[ecc_ref], distance, total_mass)
    #             surrogate_phase_SI[total_mass, ecc_ref, :] = self.surrogate_phase.t[ecc_ref]
        
    #     return surrogate_amp_SI, surrogate_phase_SI

    def geometric_units_warning(self, geometric_units, total_mass, luminosity_distance):
        # Warning for inconsistency in geometric_units parameter with class object
        if geometric_units != self.geometric_units:
            self.geometric_units = geometric_units
            print(f"Warning: geometric_units parameter is different from the class attribute. Using geometric_units ={geometric_units} and setting class object.")

        if (geometric_units is False) and (((total_mass is None) and (self.total_mass is None)) or ((luminosity_distance is None) and (self.luminosity_distance is None))):
            # For SI units: SI units but no distance or mass provided
            raise ValueError("For SI units, please provide total_mass and luminosity_distance in function call or class object.")
        
        elif (geometric_units is False) and (((self.total_mass is not None) and (total_mass is None)) or ((self.luminosity_distance is not None) and (luminosity_distance is None))):
            # For SI units: Use class attributes for total mass and distance if not provided in function call
            if total_mass is None:
                total_mass = self.total_mass
            if luminosity_distance is None:
                luminosity_distance = self.luminosity_distance
        
        elif (geometric_units is False) and (((total_mass is not None) and (self.total_mass is None)) or ((luminosity_distance is not None) and (self.luminosity_distance is None))):
            # For SI units: Update class attributes for total mass and distance if provided in function call
            if self.luminosity_distance is None:
                self.luminosity_distance = luminosity_distance
            if self.total_mass is None:
                self.total_mass = total_mass
        
        elif (geometric_units is True) and ((total_mass is not None) or (luminosity_distance is not None)):
            # For geometric units: Raise error if geometric units is True but distance or mass is provided
            total_mass = None
            luminosity_distance = None
            raise ValueError("For geometric units, please do not provide total_mass and luminosity_distance. Parameters are automatically set to total_mass=None and luminosity_distance=None.")
        
        return geometric_units, total_mass, luminosity_distance
    
        















class Call_Surrogate_Testing(Generate_Online_Surrogate_Testing):

    def __init__(
        self,
        time_array,
        ecc_ref_parameterspace_range=[0.0, 0.3],
        amount_input_wfs=60,
        amount_output_wfs=500,
        N_basis_vecs_amp=40,
        N_basis_vecs_phase=40,
        training_set_selection='GPR_opt',
        minimum_spacing_greedy=0.008,
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
        geometric_units=True
    ):
        
        self.geometric_units = geometric_units

        Generate_Online_Surrogate_Testing.__init__(
            self,
            time_array=time_array,
            ecc_ref_parameterspace_range=ecc_ref_parameterspace_range,
            amount_input_wfs=amount_input_wfs,
            amount_output_wfs=amount_output_wfs,
            N_basis_vecs_amp=N_basis_vecs_amp,
            N_basis_vecs_phase=N_basis_vecs_phase,
            training_set_selection=training_set_selection,
            minimum_spacing_greedy=minimum_spacing_greedy,
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

        
        
    def load_offline_surrogate(self, plot_GPRfit=False, save_fig_fits=False, plot_empirical_nodes_at_ecc=None, save_fig_empirical_nodes=False, plot_greedy_vectors=False, save_fig_greedy_vectors=False):
        """Load precomputed surrogate data and assign it to class object."""
        try:
            start = time.time()
            # Load the precomputed surrogate data
            data = np.load( 
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
            f"min_s={self.minimum_spacing_greedy}.npz", allow_pickle=True)


            # Amplitude data
            self.gaussian_fit_amp = data['gaussian_fit_amp']
            self.empirical_nodes_idx_amp = data['empirical_nodes_idx_amp']
            self.residual_reduced_basis_amp = data['residual_reduced_basis_amp']
            self.B_matrix_amp = data['B_matrix_amp']

            # Phase data
            self.gaussian_fit_phase = data['gaussian_fit_phase']
            self.empirical_nodes_idx_phase = data['empirical_nodes_idx_phase']
            self.residual_reduced_basis_phase = data['residual_reduced_basis_phase']
            self.B_matrix_phase = data['B_matrix_phase']

            # Indices and time array
            self.best_rep_parameters_idx_amp = data['best_rep_parameters_idx_amp']
            self.best_rep_parameters_idx_phase = data['best_rep_parameters_idx_phase']
            self.best_rep_parameters_amp = data['best_rep_parameters_amp']
            self.best_rep_parameters_phase = data['best_rep_parameters_phase']

            self.time = data['time']


            # Circular waveform properties
            self.amp_circ = data['amp_circ']
            self.phase_circ = data['phase_circ']

            print("Surrogate model loaded successfully. Time taken:", time.time() - start, self.gaussian_fit_amp.shape, self.gaussian_fit_phase.shape, self.B_matrix_amp.shape, self.B_matrix_phase.shape)

            if plot_GPRfit is True:
                self._plot_GPR_fits('phase', save_fig_fits=save_fig_fits)
                self._plot_GPR_fits('amplitude', save_fig_fits=save_fig_fits)
            
            if plot_empirical_nodes_at_ecc is not None:
                self._plot_empirical_nodes(self.empirical_nodes_idx_amp, 'amplitude', eccentricity=plot_empirical_nodes_at_ecc, save_fig=save_fig_empirical_nodes)
                self._plot_empirical_nodes(self.empirical_nodes_idx_phase, 'phase', eccentricity=plot_empirical_nodes_at_ecc, save_fig=save_fig_empirical_nodes)
            
            if plot_greedy_vectors is True:
                self._plot_greedy_vectors(greedy_basis=self.residual_reduced_basis_amp, greedy_parameters_idx=self.best_rep_parameters_idx_amp, property='amplitude', save_basis_vecs_fig=save_fig_greedy_vectors)
                self._plot_greedy_vectors(greedy_basis=self.residual_reduced_basis_phase, greedy_parameters_idx=self.best_rep_parameters_idx_phase, property='phase', save_basis_vecs_fig=save_fig_greedy_vectors)

        except:
            print("Surrogate model not found. Generating new surrogate data...")
            # Generate surrogate data and save it
            self.create_offline_surrogate()
            # Try loading again
            self.load_offline_surrogate(plot_GPRfit=plot_GPRfit, save_fig_fits=save_fig_fits, plot_empirical_nodes_at_ecc=None, save_fig_empirical_nodes=False, plot_greedy_vectors=False, save_fig_greedy_vectors=False)



    
    def generate_PhenomTE_surrogate(self, ecc_ref=None, geometric_units=True, total_mass=None, luminosity_distance=None, get_computation_time=False, plot_surr_datapiece=None, save_fig_datapiece=False, plot_surr_wf=None, save_fig_surr=False, plot_GPRfit=False, save_fig_fits=False):
        """
        Call the surrogate model with the given output eccentricity reference.
        If plot_surr_wf is True, it will plot the surrogate waveform against the real waveform.
        If plot_surr_datapiece is True, it will plot the surrogate datapiece against the real datapiece.
        """
        # Start timer
        start = time.time()

        # If ecc_ref is provided in function call, update class attribute. Otherwise use class attribute.
        if ecc_ref is None:
            ecc_ref = self.output_ecc_ref
        else:
            self.output_ecc_ref = ecc_ref

        # Warning for inconsistency in geometric_units parameter with class object
        geometric_units, total_mass, luminosity_distance = self.geometric_units_warning(geometric_units, total_mass, luminosity_distance)

        # Check if surrogate is already loaded, if not load it
        if self.surrogate_amp is None or self.surrogate_phase is None:
            print('Load surrogate ...')
            start = time.time()
            self.load_offline_surrogate()
            print('Load offline surrogate. Time taken:', time.time() - start)
        else:
            if not self._printed_surrogate_loaded:
                self._printed_surrogate_loaded = True
                print('Surrogate already loaded, skipping loading step.')
        

        self.surrogate_amp, self.surrogate_phase = self.generate_surrogate_waveform(
            output_ecc_ref=self.output_ecc_ref,
            plot_surr_datapiece=plot_surr_datapiece,
            save_fig_datapiece=save_fig_datapiece,
            plot_surr_wf=plot_surr_wf,
            save_fig_surr=save_fig_surr,
            plot_GPRfit=plot_GPRfit,
            save_fits_to_file=False,
            save_fig_fits=save_fig_fits
        )

        if get_computation_time is False:
            return self.surrogate_amp, self.surrogate_phase
        else:
            return self.surrogate_amp, self.surrogate_phase, time.time() - start

    def get_surrogate_polarisations(self, total_mass=None, luminosity_distance=None, geometric_units=False, plot_polarisations=False, save_fig=False):
        """ Get the polarisation amplitudes for the surrogate waveform.
        If geometric_units is True, it will return the polarisation amplitudes in geometric units.
        If geometric_units is False, it will return the polarisation amplitudes in SI units.
        """ 

        # Warning for inconsistency in geometric_units parameter with class object
        geometric_units, total_mass, luminosity_distance = self.geometric_units_warning(geometric_units, total_mass, luminosity_distance)

        # Convert the surrogate amplitude and phase to polarisation amplitudes
        self.hplus, self.hcross = self.polarisations(phase=self.surrogate_phase, amplitude=self.surrogate_amp, geometric_units=geometric_units, distance=luminosity_distance, total_mass=total_mass, plot_polarisations=plot_polarisations, save_fig=save_fig)
        
        return self.hplus, self.hcross
    