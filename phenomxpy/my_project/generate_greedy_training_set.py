from generate_PhenomTE import *

from sklearn.preprocessing import normalize
from scipy.linalg import orth

plt.switch_backend('WebAgg')

class Generate_TrainingSet(Waveform_Properties, Simulate_Inspiral):
    """
    Class to generate a training dataset for gravitational waveform simulations using a greedy algorithm and empirical interpolation.
    Inherits from WaveformProperties and SimulateInspiral to leverage methods for waveform 
    property calculations and waveform generation.

    """

    def __init__(self, time_array, ecc_ref_parameterspace, mean_ano_parameterspace, N_basis_vecs_amp=None, N_basis_vecs_phase=None, min_greedy_error_amp=None, min_greedy_error_phase=None, minimum_spacing_greedy=0.005, f_ref=20, f_lower=10, chi1=0, chi2=0, phiRef=0., rel_anomaly=0., inclination=0., truncate_at_ISCO=True, truncate_at_tmin=True):
        """
        Parameters:
        ----------------
        time_array [s], np.array : Time array in seconds.
        ecc_ref [dimensionless], float: Eccentricity of binary at start f_lower
        total_mass [M_sun], int : Total mass of the binary in solar masses
        f_lower [Hz], float: Start frequency of the waveform
        f_ref [Hz], float: Reference frequency at which the waveform parameters (eccentricity, spin, ...) are defined.
        chi1 [dimensionless], float, ndarray : Spin of primary. If float, interpreted as z component
        chi2 [dimensionless], float, ndarray : Spin of secondary. If float, interpreted as z component
        rel_anomaly [rad], float : Relativistic anomaly. Radial phase which parametrizes the orbit within the Keplerian (relativistic) parametrization. Defaults to 0 (periastron).
        inclination [rad], float : Inclination angle of the binary system. Defaults to 0 (face-on).
        luminosity_distance [Mpc], float : Luminosity distance of the binary in megap
        parameter_space_input [dimensionless], float, ndarray : Numpy array of eccentric values used to calculate training set. MORE VALUES RESULTS IN LARGER COMPUTATIONAL TIME!
        residual_greedy_basis [dimenionless OR rad], narray, float: Stores the residual greedy basis, set later during get_greedy_parameters().
        greedy_parameters_idx [dimensionless], narray, float: Holds indices of greedy parameters, set later during get_greedy_parameters().
        empirical_nodes_idx [dimenionless], narray, int: Stores indices of empirical nodes, to be set during get_empirical_nodes().

        """
        self.ecc_ref_parameter_space_input = ecc_ref_parameterspace
        self.minimum_spacing_greedy = minimum_spacing_greedy

        self.min_greedy_error_amp = min_greedy_error_amp
        self.min_greedy_error_phase = min_greedy_error_phase
        self.N_basis_vecs_amp = N_basis_vecs_amp
        self.N_basis_vecs_phase = N_basis_vecs_phase

        # To be stored parameters
        self.residual_reduced_basis = None
        self.best_rep_parameters_idx = None
        self.best_rep_parameters = None
        self.empirical_nodes_idx = None

        self.highest_tmin_value = None
        # Inherit parameters from all previously defined classes
        Waveform_Properties.__init__(self, time_array=time_array, ecc_ref=None, total_mass=None, luminosity_distance=None, f_lower=f_lower, f_ref=f_ref, chi1=chi1, chi2=chi2, phiRef=phiRef, rel_anomaly=rel_anomaly, inclination=inclination, truncate_at_ISCO=truncate_at_ISCO, truncate_at_tmin=truncate_at_tmin)

    def generate_property_dataset(self, ecc_list, property, save_dataset_to_file=None, plot_residuals_time_evolv=False, plot_residuals_eccentric_evolv=False, save_fig_eccentric_evolv=False, save_fig_time_evolve=False, show_legend=True):
        """
        Generates a dataset of waveform residuals based on the specified property for a certain range of eccentricities (ecc).

        Parameters:
        ----------
        ecc_list : list of floats
            List of reference eccentricities for which to calculate residuals.
        property : str
            Specifies which property to calculate ('phase' or 'amplitude').
        save_dataset_to_file : bool, optional
            If True, saves the generated dataset to a file.
        plot_residuals : bool, optional
            If True, plots the residuals for each eccentricity.
        save_fig : bool, optional
            If True, saves the residual plot to Images/Residuals.

        Returns:
        -------
        residual_dataset : ndarray
            Array of residuals for each eccentricity.

        """

        try: 
            # Attempt to load existing residual dataset
            filename = f'Straindata/Residuals/residuals_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(ecc_list)}_{max(ecc_list)}_N={len(ecc_list)}].npz'
            load_residuals = np.load(filename)
            
            residual_dataset = load_residuals['residual']
            self.time = load_residuals['time']
            
            print(f'Residual parameterspace dataset found for {property}')
            
        except Exception as e:
            print(e)

            # If attempt to load residuals failed, generate polarisations and calculate residuals
            hp_dataset, hc_dataset = self._generate_polarisation_data(ecc_list)
            residual_dataset = self._calculate_residuals(ecc_list, hp_dataset, hc_dataset, property)
            del hp_dataset, hc_dataset  # Free memory

            print(f'Generated residual parameterspace dataset for {property} ', len(ecc_list), ' waveforms')
            # If save_dataset_to_file is True save the residuals to file in Straindata/Residuals
            if save_dataset_to_file is True and not os.path.isfile(f'Straindata/Residuals/residuals_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(ecc_list)}_{max(ecc_list)}_N={len(ecc_list)}].npz'):
                self._save_residual_dataset(ecc_list, property, residual_dataset)

        # If plot_residuals is True, plot whole residual dataset
        if (plot_residuals_eccentric_evolv is True) or (plot_residuals_time_evolv is True):
            self._plot_residuals(residual_dataset, ecc_list, property, plot_residuals_eccentric_evolv, plot_residuals_time_evolv, save_fig_eccentric_evolv, save_fig_time_evolve, show_legend=show_legend )
        
        return residual_dataset
    
    def _generate_polarisation_data(self, ecc_list):
        """
        Helper function to generate polarisation data for a list of eccentricities.

        Parameters:
        ----------
        ecc_list : list of floats
            List of minimum eccentricities.

        Returns:
        -------
        hp_dataset : ndarray
            Plus polarisation data.
        hc_dataset : ndarray
            Cross polarisation data.

        """
        try:
            # Attempt to load existing polarisation dataset
            load_polarisations = np.load(f'Straindata/Polarisations/polarisations_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(ecc_list)}_{max(ecc_list)}_N={len(ecc_list)}]_t_lower={int(self.time[0])}.npz', allow_pickle=True)
            hp_dataset = load_polarisations['hp']
            hc_dataset = load_polarisations['hc']
            self.time = load_polarisations['time']

            print('Loaded polarisations')

        except:
            # Get waveform size for truncated ISCO waveform of smallest waveform
            sorted_ecc_list = np.sort(ecc_list)

            ISCO_ecc = sorted_ecc_list[-1] # Highest eccentricity in the list --> earliest ISCO cut-off
            hp_ISCO, hc_ISCO = self.simulate_inspiral(ecc_ref=ISCO_ecc, truncate_at_ISCO=True, truncate_at_tmin=True)
            
            hp_dataset = np.zeros((len(ecc_list), len(self.time))) 
            hc_dataset = np.zeros((len(ecc_list), len(self.time)))

            for i, ecc in enumerate(ecc_list):
                if ecc == ISCO_ecc:
                    # Store first waveform in dataset
                    hp_dataset[i] = hp_ISCO
                    hc_dataset[i] = hc_ISCO
                else:
                    hp, hc = self.simulate_inspiral(ecc_ref=ecc, truncate_at_ISCO=False, truncate_at_tmin=False)

                    hp_dataset[i] = hp[:len(hp_ISCO)]  # Ensure the waveform length matches the shortest time array
                    hc_dataset[i] = hc[:len(hp_ISCO)]  # Ensure the waveform length matches the shortest time array

                    del hp, hc  # Explicit cleanup

                self.time = self.time[:len(hp_ISCO)]  # Ensure time array matches the dataset waveform length

            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Straindata/Polarisations', exist_ok=True)
            np.savez(f'Straindata/Polarisations/polarisations_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(ecc_list)}_{max(ecc_list)}_N={len(ecc_list)}]_t_lower={int(self.time[0])}.npz', hp=hp_dataset, hc=hc_dataset, time=self.time)

        return hp_dataset, hc_dataset

    def _calculate_residuals(self, ecc_list, hp_dataset, hc_dataset, property):
        """
        Helper function to calculate residuals for a property given polarisation data.

        Parameters:
        ----------
        ecc_list : list of floats
            List of minimum eccentricities.
        hp_dataset : ndarray
            Plus polarisation data.
        hc_dataset : ndarray
            Cross polarisation data.
        property : str
            Specifies which property to calculate ('phase' or 'amplitude').

        Returns:
        -------
        residual_dataset : ndarray
            Array of residuals for each eccentricity.
            
        """
        self.circulair_wf()

        # Create empty residual dataset
        residual_dataset = np.zeros((len(ecc_list), len(self.time)))

        # Fill residual dataset with residuals of chosen property for given eccentric parameter space
        for i, ecc in enumerate(ecc_list):
            residual = self.calculate_residual(hp_dataset[i], hc_dataset[i], ecc, property)
            residual_dataset[i] = residual

            del residual
        return residual_dataset
    
    def _plot_residuals(self, residual_dataset, ecc_list, property, plot_eccentric_evolv=False, plot_time_evolve=False, save_fig_eccentric_evolve=False, save_fig_time_evolve=False, show_legend=True):
        """Function to plot residuals dataset including save figure option."""
        if plot_eccentric_evolv is True:
            fig_residuals_ecc = plt.figure()
            for i in range(0, len(self.time), 100):
                plt.plot(ecc_list, residual_dataset.T[i], label='t/M = ' + f'{round(self.time[i], 1)}', linewidth=0.6)
                
            plt.xlabel('eccentricity')
            if property == 'phase':
                plt.ylabel(' $\Delta \phi_{22}$ [radians]')
            elif property == 'amplitude':
                plt.ylabel('$\Delta A_{22}$')
            else:
                print('Choose property = "phase", "amplitude"', property)
                sys.exit(1)

            plt.title(f'Residuals {property}')
            plt.grid(True)

            if show_legend:
                plt.legend(loc='upper right', fontsize='small', ncol=2)

            plt.tight_layout()

            if save_fig_eccentric_evolve is True:
                ISCO = '' if self.truncate_at_ISCO else 'NO_ISCO_'
                tmin = '' if self.truncate_at_tmin else 'NO_tmin_'

                figname = f'Images/Residuals/Residuals_eccentric_evolv_{property}_{ISCO}_{tmin}_M={self.total_mass}_ecc_list=[{round(min(ecc_list), 2)}_{round(max(ecc_list), 2)}_N={len(self.ecc_ref_parameter_space_input)}]_f_lower={self.f_lower}_f_ref={self.f_ref}.png'
               
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Residuals', exist_ok=True)
                fig_residuals_ecc.savefig(figname)

                print(self.colored_text(f'Figure is saved in {figname}', 'blue'))
                # plt.close('all') # Clean up plot

        if plot_time_evolve is True:
            fig_residuals_t = plt.figure()

            for i in range(len(residual_dataset)):
                plt.plot(self.time, residual_dataset[i], label='e$_{ref}$' + f' = {round(ecc_list[i], 3)}', linewidth=0.6)
               
            plt.xlabel('t [M]')
            if property == 'phase':
                plt.ylabel(' $\Delta \phi_{22}$ [radians]')
            elif property == 'amplitude':
                plt.ylabel('$\Delta A_{22}$')
            else:
                print('Choose property = "phase", "amplitude"', property)
                sys.exit(1)

            plt.title(f'Residuals {property}')
            plt.grid(True)

            if show_legend:
                plt.legend(loc='upper right', fontsize='small', ncol=2)

            plt.tight_layout()

            if save_fig_time_evolve is True:
                ISCO = '' if self.truncate_at_ISCO else 'NO_ISCO'
                tmin = '' if self.truncate_at_tmin else 'NO_tmin'



                figname = f'Images/Residuals/Residuals_time_evolv_{property}_{ISCO}_{tmin}_M={self.total_mass}_ecc_list=[{min(ecc_list)}_{max(ecc_list)}_N={len(self.ecc_ref_parameter_space_input)}]_f_lower={self.f_lower}_f_ref={self.f_ref}.png'

                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Residuals', exist_ok=True)
                fig_residuals_t.savefig(figname)

                print(self.colored_text(f'Figure is saved in {figname}', 'blue'))
                # plt.close('all')
    
    def _save_residual_dataset(self, ecc_list, property, residual_dataset):
        """Function to save residual dataset to file."""

        os.makedirs('Straindata/Residuals', exist_ok=True)
        file_path = f'Straindata/Residuals/residuals_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(ecc_list)}_{max(ecc_list)}_N={len(ecc_list)}].npz'
        np.savez(file_path, residual=residual_dataset, time=self.time, eccentricities=ecc_list)
        print('Residuals saved to Straindata/Residuals')


    def get_greedy_parameters(self, U, property, min_greedy_error=None, N_greedy_vecs=None, reg=1e-6, plot_greedy_error=False, save_greedy_error_fig=False, plot_greedy_vectors=False, save_greedy_vecs_fig=False, plot_greedy_basis_formation=False, minimum_spacing=None):
        """
        Greedy algorithm to select representative vectors from U using an orthonormal basis.

        Parameters
        ----------
        U : ndarray, shape (num_vectors, vector_length)
            Dataset of vectors to build the greedy basis from.
        N_greedy_vecs : int, optional
            Maximum number of vectors to include in the greedy basis.
        tol : float, optional
            Stop when the maximum residual norm falls below this tolerance.
        minimum_spacing : float, optional
            If not None, greedy points are picked with a minimum spacing in between points.

        Returns
        -------
        greedy_basis : ndarray
            Orthonormal greedy basis vectors.
        greedy_indices : list
            Indices of the vectors chosen from U.
        residuals : list
            Maximum residual norm at each iteration.
        """

        if minimum_spacing is None:
            minimum_spacing = self.minimum_spacing_greedy
            print('minimum spacing: ', minimum_spacing)

        # Make a copy of U and normalize each vector to avoid scale issues
        U = U.copy()
        time_array = self.time
        folder_img = f'test_{property}_{N_greedy_vecs}'

        U_normalized = normalize(U, axis=1)[1:] # Skip the first row (zero vector of ecc=0) to prevent false uniqueness due to inner product of zero vectors
        num_vectors = U.shape[0]

        greedy_basis_orthonormal = []
        greedy_basis = []
        greedy_indices = []
        greedy_errors = []
        
        # Get delta ecc_ref for application of minimum_spacing
        delta_ecc_ref = self.ecc_ref_parameter_space_input[1] - self.ecc_ref_parameter_space_input[0]
        minimum_spacing_idx = int(minimum_spacing / delta_ecc_ref)
        
        # Mask to track valid vectors (True = can be picked)
        valid_mask = np.ones(U_normalized.shape[0], dtype=bool)

        for step in range(num_vectors):
            # Stop if no more valid vectors due to minimum spacing
            if not valid_mask.any():
                print(self.colored_text(f'WARNING: break in get_greedy_params at N_greedy_vecs = {len(greedy_indices)}. No more valid vectors due to minimum spacing constraint of {minimum_spacing}.', 'red'))
                if property == 'phase':
                    self.N_basis_vecs_phase = len(greedy_indices)
                elif property == 'amplitude':
                    self.N_basis_vecs_amp = len(greedy_indices)

                break

            # Compute residuals: h - sum_i <h, e_i> e_i for all vectors h in U
            if len(greedy_basis_orthonormal) == 0:
                # First iteration: residuals are just the norms of U
                residual_norms = np.linalg.norm(U_normalized, axis=1)

            else:
                

                # Stack basis for matrix operations
                B = np.vstack(greedy_basis_orthonormal)  # shape: (m, vector_length)
                # Compute inner products <h, e_i> for all h in U
                coeffs = U_normalized @ B.T      # shape: (num_vectors, m)
                # Reconstruct projections
                U_proj = coeffs @ B              # shape: (num_vectors, vector_length)
                # Residuals
                residual_norms = np.linalg.norm(U_normalized - U_proj, axis=1)

                # Accumulated rounding error
                # orth_err = np.linalg.norm(B @ B.T - np.eye(len(B)), 'fro')
                # print("orthogonal error:", orth_err)
                
                if plot_greedy_basis_formation:
                    max_idx = np.argmax(residual_norms)

                    fig_comp_greedy, axs = plt.subplots(4, 1, figsize=(15, 15))

                    print('shapes: ', coeffs.shape, U_proj.shape, residual_norms.shape)
                    axs[0].scatter(self.ecc_ref_parameter_space_input[greedy_indices], np.zeros(len(greedy_indices)), label=f'current greedy indices, i={step}')
                    axs[0].scatter(self.ecc_ref_parameter_space_input[0], 0, c='orange', label='minimum spacing')
                    axs[0].scatter(self.ecc_ref_parameter_space_input[minimum_spacing_idx], 0, c='orange')
                    axs[0].plot(self.ecc_ref_parameter_space_input[1:], residual_norms,  label=f'step = {step}')
                    axs[0].set_xlabel('ecc')
                    axs[0].set_ylabel('residuals norm')
                    axs[0].legend()

                    axs[1].scatter(self.ecc_ref_parameter_space_input[greedy_indices], np.zeros(len(greedy_indices)))
                    for i in range(len(U_normalized)):
                        axs[1].plot(time_array, U_normalized.T, color='grey')
                    for j in range(step):
                        axs[1].plot(time_array, U_normalized[greedy_indices[j] - 1], color='red')
                    axs[1].plot(time_array, U_normalized[greedy_indices[-1] - 1], label='last added vec', color='blue')
                    # axs[1].plot(self.ecc_ref_parameter_space_input[1:], residual_norms,  label=f'step = {step}')
                    axs[1].set_ylabel('residuals diff')
                    axs[1].legend()


                    axs[2].scatter(self.ecc_ref_parameter_space_input[greedy_indices], np.zeros(len(greedy_indices)), label=f'current greedy indices, i={step}')
                    axs[2].plot(self.ecc_ref_parameter_space_input[1:], coeffs)
                    axs[2].set_ylabel('coeffs')
                    axs[2].legend()

                    colors = plt.cm.tab10.colors[:7]
                    for i,c in zip([1, 2, 3, 4, 5, 6, max_idx], colors):
                        # axs[3].plot(U_normalized[i], label="U_normalized", c=c)
                        # axs[3].plot(U_proj[i], label="U_proj", c=c)
                        axs[3].plot(U_normalized[i] - U_proj[i], label=f"residual {str(i)}", c=c)
                    axs[3].legend()
                    axs[3].set_ylabel('residuals vec')
                    axs[3].set_xlabel('time')

                    os.makedirs(f'Images/{folder_img}/', exist_ok=True)
                    fig_comp_greedy.savefig(f'Images/{folder_img}/test_greedy_{step}.png')

                    plt.close('all')



            # Apply mask: exclude already-picked vectors + minimum spacing. Set non pick to -infinity so they are never picked.
            masked_residuals = np.where(valid_mask, residual_norms, -np.inf)

            # Find the vector with the largest residual
            max_idx = np.argmax(masked_residuals)
            max_res = residual_norms[max_idx]

            # Save highest residual --> greedy error
            greedy_errors.append(round(float(max_res), 6)) 
            greedy_indices.append(int(max_idx + 1))  # +1 to account for the zero vector at index 0. int for clearer show of greedy incidces
            greedy_basis.append(U[max_idx + 1])  # Store the original vector from U

            # Add new vector to the orthonormal basis
            new_vec = U_normalized[max_idx].copy()

            # Modified Gram-Schmidt: single pass
            for b in greedy_basis_orthonormal:   # use the orthonormal vectors
                new_vec -= np.dot(new_vec, b) * b

            # second pass (re-orthogonalize to remove numerical residue)
            for b in greedy_basis_orthonormal:
                new_vec -= np.dot(new_vec, b) * b

            # Normalise vector
            norm = np.linalg.norm(new_vec)
            new_vec /= norm

            greedy_basis_orthonormal.append(new_vec)

            # Update mask to enforce minimum spacing around this index
            start = max(0, max_idx - minimum_spacing_idx)
            end = min(valid_mask.size, max_idx + minimum_spacing_idx + 1)
            valid_mask[start:end] = False

            # --- Check stopping conditions ---
            if min_greedy_error is not None and (max_res <= min_greedy_error or len(greedy_basis) == len(U)):
                break
            if N_greedy_vecs is not None and len(greedy_basis) >= N_greedy_vecs:
                break
            

        # Stack basis for convenience
        greedy_basis = np.vstack(greedy_basis)
        greedy_basis_orthonormal = np.vstack(greedy_basis_orthonormal)

        #  Plot greedy errors if requested
        if plot_greedy_error:
            self._plot_greedy_errors(greedy_errors, property, save_greedy_error_fig)

        if plot_greedy_vectors:
            # self._plot_greedy_vectors(U, greedy_basis_orthonormal, greedy_indices, property, save_greedy_vecs_fig)
            self._plot_greedy_vectors(U=U, greedy_basis=greedy_basis, greedy_parameters_idx=greedy_indices, property=property, save_greedy_vecs_fig=save_greedy_vecs_fig)

        print(f'Highest error of best approximation of the basis: {round(np.min(greedy_errors), 5)} | {len(greedy_basis)} basis vectors')
        print(greedy_indices, greedy_errors)

        return greedy_indices, greedy_basis_orthonormal


    def _plot_greedy_vectors(self, greedy_basis, greedy_parameters_idx, property, save_greedy_vecs_fig, U=None):
        """Function to plot and option to save the greedy basis vectors. If U is specified, also plot complete dataset before greedy algorithm."""

        fig_greedy_vecs, (ax_main, ax_bottom) = plt.subplots(
            2, 1, figsize=(12, 6),
            gridspec_kw={'height_ratios': [4, 0.5]}
        )

        # --- Top plot: dataset and greedy basis vectors ---
        if U is not None:
            for i, vec in enumerate(U):
                ax_main.plot(vec, color='grey', alpha=0.3, label='Vector dataset' if i == 0 else None)

        for i, vec in enumerate(greedy_basis):
            ax_main.plot(vec, color='red', linewidth=0.6,
                        label=f'{self.ecc_ref_parameter_space_input[greedy_parameters_idx[i]]}')

        

        ax_main.set_ylabel('Vector Value')
        ax_main.set_title(f'Greedy Basis Vectors ({len(greedy_basis)} vectors) for {property}')
        ax_main.legend()
        ax_main.grid(True)

        # --- Bottom plot: eccentric points and greedy indices ---
        ecc_values = self.ecc_ref_parameter_space_input # assuming first component = eccentricity
        y = np.zeros(len(self.ecc_ref_parameter_space_input))

        # All dataset points
        ax_bottom.plot(ecc_values, y, color='grey', alpha=0.6, label='Dataset points')

        # Greedy-selected points
        ax_bottom.scatter(ecc_values[greedy_parameters_idx], y[greedy_parameters_idx], color='red', s=50, label='Greedy parameters')

        ax_bottom.set_yticks([])
        ax_bottom.set_xlabel('Vector index / parameter')
        ax_bottom.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax_main.legend(loc='best', ncol=3, fontsize='small')


        plt.tight_layout()
        fig_greedy_vecs.show()


        if save_greedy_vecs_fig:
            os.makedirs('Images/Greedy_vectors', exist_ok=True)
            plt.savefig(f'Images/Greedy_vectors/Greedy_vectors_{property}_M={self.total_mass}_ecc=[{min(self.ecc_ref_parameter_space_input)}_{max(self.ecc_ref_parameter_space_input)}]_f_lower={self.f_lower}_f_ref={self.f_ref}_iN={len(self.ecc_ref_parameter_space_input)}_ms={self.minimum_spacing_greedy}.png')
            print('Greedy vectors fig saved to Images/Greedy_vectors')
            # plt.close('all')

    def _plot_greedy_errors(self, greedy_errors, property, save_greedy_fig):
        """Function to plot and option to save the greedy errors."""
        N_basis_vectors = np.arange(1, len(greedy_errors) + 1)

        fig_greedy_errors = plt.figure(figsize=(7, 5))
        plt.plot(N_basis_vectors, greedy_errors, label='Greedy Errors')
        plt.scatter(N_basis_vectors, greedy_errors, s=4)
        # for i, label in enumerate(self.ecc_ref_parameter_space_input[greedy_parameters_idx]):
        #     plt.annotate(label, (N_basis_vectors[i], greedy_errors[i]), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=5.5)
        plt.xlabel('Number of Waveforms')
        if property == 'phase':
            plt.ylabel(f'Greedy error $\Delta \phi$')
        elif property == 'amplitude':
            plt.ylabel(f'Greedy error $\Delta A$')
        plt.yscale('log')
        # plt.title('Greedy errors of residual {} for N = {}'.format(property, len(greedy_errors)-1))
        plt.grid(True)
        fig_greedy_errors.show()

        if save_greedy_fig:
            os.makedirs('Images/Greedy_errors', exist_ok=True)
            plt.savefig(f'Images/Greedy_errors/Greedy_error_{property}_M={self.total_mass}_ecc=[{min(self.ecc_ref_parameter_space_input)}_{max(self.ecc_ref_parameter_space_input)}]_f_lower={self.f_lower}_f_ref={self.f_ref}_iN={len(self.ecc_ref_parameter_space_input)}_gerr={min(greedy_errors)}_ms={self.minimum_spacing_greedy}.png')
            
            print('Greedy error fig saved to Images/Greedy_errors')
            # plt.close('all')
        

    def _plot_validation_errors(self, validation_vecs, greedy_basis, trivial_basis, property, save_validation_fig):
        """Function to plot and option to save validation errors."""
        
        def compute_proj_errors(basis, V, reg=1e-6):
            """Computes the projection errors when approximating target vectors V using the basis."""
            
            normalized_basis = basis / np.linalg.norm(basis, axis=1, keepdims=True)
            normalized_V = V / np.linalg.norm(V, axis=1, keepdims=True)
            G = np.dot(normalized_basis, normalized_basis.T) + reg * np.eye(normalized_basis.shape[0])
            R = np.dot(normalized_basis, normalized_V.T)
            errors = []
            
            for N in range(len(normalized_basis) + 1):
                V_proj = np.dot(np.linalg.solve(G[:N, :N], R[:N, :]).T, normalized_basis[:N, :]) if N > 0 else np.zeros_like(V)
                errors.append(np.max(np.linalg.norm(normalized_V - V_proj, axis=1)).round(6))
            return errors
        
        greedy_validation_errors = compute_proj_errors(greedy_basis, validation_vecs)
        trivial_validation_errors = compute_proj_errors(trivial_basis, validation_vecs)
        N_basis_vectors = np.arange(1, len(greedy_validation_errors) + 1)

        plt.figure(figsize=(7, 5))
        plt.plot(N_basis_vectors, greedy_validation_errors, label='Greedy Basis Errors')
        plt.plot(N_basis_vectors, trivial_validation_errors, label='Trivial Basis Errors')
        plt.scatter(N_basis_vectors, greedy_validation_errors, s=4)
        plt.scatter(N_basis_vectors, trivial_validation_errors, s=4)
        plt.xlabel('Number of Waveforms')
        plt.ylabel('Validation Error')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)

        if save_validation_fig:
            os.makedirs('Images/Validation_errors', exist_ok=True)
            plt.savefig(f'Images/Validation_errors/Validation_error_{property}_M={self.total_mass}_ecc=[{min(self.ecc_ref_parameter_space_input)}_{max(self.ecc_ref_parameter_space_input)}]_f_lower={self.f_lower}_f_ref={self.f_ref}_iN={len(self.ecc_ref_parameter_space_input)}.png')
        
        # plt.close('all')

    # def get_empirical_nodes(self, reduced_basis, property, plot_emp_nodes_at_ecc=True, save_fig=True):
    #     """
    #     Perform the Empirical Interpolation Method (EIM).
        
    #     Parameters
    #     ----------
    #     reduced_basis : ndarray, shape (m, L)
    #         The reduced basis vectors (m basis functions, each of length L).
    #     grid_points : ndarray, shape (L,)
    #         The discrete grid points (t_l values).

    #     Returns
    #     -------
    #     emp_nodes_idx : list of int
    #         Indices of empirical nodes in the grid.
    #     emp_nodes : list of float
    #         The actual grid point locations.
    #     """
    #     m, L = reduced_basis.shape
    #     emp_nodes_idx = []
    #     emp_nodes = []

    #     U, S, VT = np.linalg.svd(reduced_basis)
    #     plt.semilogy(S, 'o-')
    #     plt.title("Singular Values of Reduced Basis")
    #     plt.show()

    #     # Step 1 — first node: pick the max abs value from first basis vector
    #     i = np.argmax(np.abs(reduced_basis[0]))
    #     emp_nodes_idx.append(i)

    #     # Loop for j = 2 ... m
    #     for j in range(1, m):
    #         # Build V matrix (j-1 x j-1) from previous basis vectors at previous nodes
    #         V = reduced_basis[:j, emp_nodes_idx]
    #         print(f"Iteration {j}:")
    #         print(f"Rank of V:", np.linalg.matrix_rank(V), ' should be smaller than {j}')
    #         print("Condition number of V:", np.linalg.cond(V), ' should be > 1e10')
    #         # print('V: ', V)
    #         # # Solve for coefficients that interpolate e_j at previous nodes
    #         # coeffs = np.linalg.solve(V, reduced_basis[j, emp_nodes_idx])

    #         coeffs = np.linalg.pinv(V) @ reduced_basis[j, emp_nodes_idx]
    #         # Build interpolant on whole grid
    #         interpolant = np.dot(coeffs, reduced_basis[:j])

    #         # Compute residual
    #         residual = reduced_basis[j] - interpolant

    #         # Pick next node as location of max abs residual
    #         i = np.argmax(np.abs(residual))
    #         emp_nodes_idx.append(i)

    #     emp_nodes = self.time[emp_nodes_idx]
    #     # print(emp_nodes_idx, emp_nodes)

    #     # Optional: Plot the empirical nodes if plot_emp_nodes_at_ecc is set
    #     if plot_emp_nodes_at_ecc:
    #         self._plot_empirical_nodes(emp_nodes_idx, property, plot_emp_nodes_at_ecc, save_fig)

    #     return emp_nodes_idx, emp_nodes
    
    # def get_empirical_nodes(self, reduced_basis, property, plot_emp_nodes_at_ecc=True, save_fig=True):
    #     """
    #     Reduced basis needs to be orthonormal!
    #     """
        
    #     m, L = reduced_basis.shape

    #     emp_nodes_idx = []
    #     emp_nodes = []
    #     time_grid = self.time  # Assuming this is your grid points

    #     # # Initial SVD check
    #     # U, S, VT = np.linalg.svd(reduced_basis)
    #     # plt.semilogy(S, 'o-')
    #     # plt.title("Singular Values of Reduced Basis")
    #     # plt.show()

    #     # First node selection
    #     i = np.argmax(np.abs(reduced_basis[0]))
    #     emp_nodes_idx.append(int(i))

    #     for j in range(1, reduced_basis.shape[0]):
    #         V = reduced_basis[:j, emp_nodes_idx]
    #         coeffs = np.linalg.pinv(V) @ reduced_basis[j, emp_nodes_idx]
    #         interpolant = np.dot(coeffs, reduced_basis[:j])
    #         residual = reduced_basis[j] - interpolant

    #         # # --- Enhanced Residual Visualization ---
    #         # plt.figure(figsize=(15, 10))
            
    #         # # Plot 1: Current basis vector being approximated
    #         # plt.subplot(3, 1, 1)
    #         # plt.plot(time_grid, reduced_basis[j], 'b-', label=f'Basis Vector {j}')
    #         # plt.scatter(time_grid[emp_nodes_idx], reduced_basis[j, emp_nodes_idx], 
    #         #         c='blue', marker='o', label='Node Values')
    #         # plt.title(f'Target Basis Vector {j} to Approximate')
    #         # plt.legend()
            
    #         # Plot 2: Interpolant construction
    #         # plt.subplot(3, 1, 2)
    #         # for k in range(j):
    #         #     plt.plot(time_grid, coeffs[k] * reduced_basis[k], '--', alpha=0.5, 
    #         #             label=f'{coeffs[k]:.2f}×Basis{k}')
    #         # plt.plot(time_grid, interpolant, 'r-', linewidth=2, label='Interpolant Sum')
    #         # plt.scatter(time_grid[emp_nodes_idx], interpolant[emp_nodes_idx], 
    #         #         c='red', marker='x', label='Interpolant at Nodes')
    #         # plt.title('Interpolant Construction (Weighted Sum of Previous Basis)')
    #         # plt.legend()
            
    #         # # Plot 3: Residual calculation
    #         # plt.subplot(3, 1, 3)
    #         # plt.plot(time_grid, reduced_basis[j], 'b-', label='Original Vector')
    #         # plt.plot(time_grid, interpolant, 'r-', label='Interpolant')
    #         # plt.plot(time_grid, residual, 'g-', label='Residual')
    #         # plt.scatter(time_grid[emp_nodes_idx], np.zeros_like(emp_nodes_idx),
    #         #         c='black', marker='x', label='Existing Nodes')
    #         # new_node = np.argmax(np.abs(residual))
    #         # plt.scatter(time_grid[new_node], residual[new_node], 
    #         #         c='magenta', s=100, label='New Node Candidate')
    #         # plt.title(f'Residual Calculation (Max at {time_grid[new_node]:.2f})')
    #         # plt.legend()
            
    #         # plt.tight_layout()
    #         # plt.show()

    #         i = np.argmax(np.abs(residual))
    #         emp_nodes_idx.append(int(i))

    #     # --- Final Plots ---
    #     emp_nodes = self.time[emp_nodes_idx]
    #     print(emp_nodes_idx, emp_nodes)
        
    #     # Plot 5: All Basis Vectors with Final Nodes
    #     plt.figure(figsize=(12, 6))
    #     # for k in range(m):
    #         # plt.plot(self.time, reduced_basis[k], alpha=0.5, label=f"Basis {k}" if k < 5 else None)
    #     plt.scatter(emp_nodes, np.zeros_like(emp_nodes), c='red', marker='x', s=100, label="EIM Nodes")
    #     plt.title(f"Final Basis Vectors and Selected Nodes for {property}")
    #     plt.xlabel("Time")
    #     plt.ylabel("Basis Value")
    #     plt.legend(ncol=2)
    #     plt.grid(True)
    #     plt.show()

    #     # Plot 6: Node Distribution Histogram
    #     # if len(emp_nodes_idx) > 1:
    #         # distances = np.diff(np.sort(emp_nodes_idx))
    #         # plt.figure(figsize=(8, 4))
    #         # plt.hist(distances, bins=20)
    #         # plt.title("Distance Between Consecutive Nodes")
    #         # plt.xlabel("Grid Points")
    #         # plt.ylabel("Frequency")
    #         # plt.show()

    #     if plot_emp_nodes_at_ecc:
    #         self._plot_empirical_nodes(emp_nodes_idx, property, plot_emp_nodes_at_ecc, save_fig)

    #     return emp_nodes_idx

    def gram_schmidt_rows(self, V, tol=1e-12, verbose=True):
        """
        Orthonormalize a set of row vectors using a numerically stable 
        Modified Gram–Schmidt process with reorthogonalization.

        Parameters
        ----------
        V : np.ndarray
            Shape (n_vec, n_dim). Each row is a vector to orthonormalize.
        tol : float, optional
            Threshold below which a vector is considered linearly dependent and skipped.
        verbose : bool, optional
            If True, prints diagnostic messages for dependent rows.

        Returns
        -------
        Q : np.ndarray
            Orthonormalized rows (shape (n_valid_vecs, n_dim)).
        """

        V = np.asarray(V, dtype=float).copy()
        n_vec, n_dim = V.shape
        Q_list = []

        for i in range(n_vec):
            v = V[i].copy()

            # First orthogonalization pass
            for q in Q_list:
                v -= np.dot(q, v) * q

            # Second orthogonalization pass (reorthogonalization)
            for q in Q_list:
                v -= np.dot(q, v) * q

            norm = np.linalg.norm(v)

            if norm < tol:
                if verbose:
                    print(f"⚠️  Skipping row {i} (norm={norm:.2e}) — linearly dependent or near-zero.")
                continue  # skip nearly dependent vector

            Q_list.append(v / norm)

        if not Q_list:
            raise ValueError("All input vectors were linearly dependent or zero.")

        Q = np.vstack(Q_list)
        return Q

    
    def empirical_interpolation_from_dataset(self, waveforms_dataset, property, plot_emp_nodes_at_ecc=True, save_fig=True):
        """
        Compute empirical interpolation nodes directly from a dataset (not from a reduced basis).

        Parameters
        ----------
        property_matrix : np.ndarray
            2D array with shape (num_waveforms, num_points).
            Each row is a waveform (or function sampled at discrete points).
        N_nodes : int
            Desired number of empirical nodes to select.

        Returns
        -------
        emp_nodes_idx : list
            List of indices of selected empirical nodes.
        """

        if property == 'phase':
            N_nodes = self.N_basis_vecs_phase
        if property == 'amplitude':
            N_nodes = self.N_basis_vecs_amp

        
        def calc_empirical_interpolant(property_array, reduced_basis, emp_nodes_idx):
            """
            Calculates the empirical interpolant for a specific waveform property using a reduced basis.
            
            Parameters:
            ----------------
            - property_array (numpy.ndarray): The waveform property values (e.g., phase or amplitude).
            - reduced_basis (numpy.ndarray): Reduced basis of residual properties.
            - emp_nodes_idx (list): Indices of empirical nodes.

            Returns:
            ----------------
            - empirical_interpolant (numpy.ndarray): The computed empirical interpolant of the waveform property.
            """
            
            empirical_interpolant = np.zeros_like(property_array)
            m = len(emp_nodes_idx)
            
            # Prepare interpolation coefficients
            B_j_vec = np.zeros((reduced_basis.shape[1], m))
            V = np.array([[reduced_basis[i][emp_nodes_idx[j]] for i in range(m)] for j in range(m)])
            V_inv = np.linalg.pinv(V)  # pseudo-inverse for stability

            # Calculate B_j interpolation vector
            for t in range(reduced_basis.shape[1]):
                for i in range(m):
                    B_j_vec[t, i] = np.dot(reduced_basis[:, t], V_inv[:, i])

            # Compute the empirical interpolant
            for j in range(reduced_basis.shape[0]):
                empirical_interpolant += B_j_vec[:, j] * property_array[emp_nodes_idx[j]]

            return empirical_interpolant
        
        reduced_basis = self.gram_schmidt_rows(waveforms_dataset)
        
        # Initialize with the index of the maximum value in the first basis vector
        i = np.argmax(reduced_basis[0])
        emp_nodes_idx = [i]
        EI_error = []

        # Loop through the reduced basis to calculate interpolants
        for j in range(1, N_nodes):
            empirical_interpolant = calc_empirical_interpolant(reduced_basis[j], reduced_basis[:j], emp_nodes_idx)
            residuals = empirical_interpolant - reduced_basis[j][:, np.newaxis].T
            EI_error.append(np.linalg.norm(residuals))

            # Identify the next empirical node based on the maximum residual
            next_idx = np.argmax(np.abs(residuals))
            emp_nodes_idx.append(next_idx)

        #     # Inside the loop
        #     fig_residuals = plt.figure(figsize=(8, 4))
        #     plt.plot(np.abs(residuals).flatten(), label=f"Step {j}")
        #     plt.axvline(next_idx, color='r', linestyle='--', label=f"New node {next_idx}")
        #     plt.title(f"Residual at Step {j}")
        #     plt.xlabel("Sample index")
        #     plt.ylabel("|Residual|")
        #     plt.legend()
        #     plt.tight_layout()
        #     fig_residuals.savefig(f'Images/Empirical_nodes/test_residuals_{j}.png')

        # fig_error = plt.figure(figsize=(6, 4))
        # plt.semilogy(EI_error, marker='o')
        # plt.title("Empirical Interpolation Error Convergence")
        # plt.xlabel("Iteration")
        # plt.ylabel("‖Residual‖₂")
        # plt.grid(True, which='both', ls='--')
        # fig_error.savefig('Images/Empirical_nodes/test_error.png')

        # # Example: Compare the 3rd waveform’s true vs interpolated
        # j = 2
        # wf_true = reduced_basis[j]
        # wf_interp = calc_empirical_interpolant(wf_true, reduced_basis[:j], emp_nodes_idx[:j])

        # fig_compare = plt.figure(figsize=(8, 4))
        # plt.plot(wf_true, label="True waveform", lw=2)
        # plt.plot(wf_interp, '--', label="Interpolated", lw=2)
        # plt.scatter(emp_nodes_idx[:j], wf_true[emp_nodes_idx[:j]], color='red', label="Empirical nodes")
        # plt.title(f"Empirical Interpolation at Step {j}")
        # plt.legend()
        # plt.tight_layout()
        # fig_compare.savefig('Images/Empirical_nodes/test_compare.png')


        # Optional: Plot the empirical nodes if plot_emp_nodes_at_ecc is set
        if plot_emp_nodes_at_ecc:
            self._plot_empirical_nodes_from_dataset(emp_nodes_idx, property, plot_emp_nodes_at_ecc, save_fig)
        
        return emp_nodes_idx

    def get_empirical_nodes_test(self, reduced_basis, property, plot_emp_nodes_at_ecc=True, save_fig=True):
        """
        Calculate the empirical nodes for a given dataset based on a reduced basis of residual properties.

        Parameters:
        ----------------
        - reduced_basis (numpy.ndarray): Reduced basis of residual properties (phase or amplitude).
        - property (str): Waveform property to evaluate, options are "phase" or "amplitude".
        - plot_emp_nodes_at_ecc (float, optional): If set, plots the empirical nodes at a specified eccentricity value.
        - save_fig (bool, optional): Saves the empirical nodes plot if set to True.

        Returns:
        ----------------
        - emp_nodes_idx (list): Indices of empirical nodes for the given dataset.
        """


        def calc_empirical_interpolant(property_array, reduced_basis, emp_nodes_idx):
            """
            Compute empirical interpolant for a waveform property using reduced basis.
            """
            m = len(emp_nodes_idx)
            empirical_interpolant = np.zeros_like(property_array)

            m = len(emp_nodes_idx)
            V = reduced_basis[:m, emp_nodes_idx].T
            rhs = property_array[emp_nodes_idx]


            # Solve for coefficients (α)
            try:
                alpha = np.linalg.solve(V, rhs)
            except np.linalg.LinAlgError:
                alpha = np.linalg.pinv(V) @ rhs

            # Compute interpolant
            empirical_interpolant = np.sum(alpha[:, None] * reduced_basis[:m, :], axis=0)

            return empirical_interpolant
        
        if property == 'phase':
            N_nodes = self.N_basis_vecs_phase
        if property == 'amplitude':
            N_nodes = self.N_basis_vecs_amp

        if reduced_basis.shape[0] < reduced_basis.shape[1]:
            # assume rows = functions (waveforms)
            reduced_basis = reduced_basis.T

        # Each row = a basis vector
        reduced_basis = np.array(reduced_basis, dtype=float)
        reduced_basis /= np.linalg.norm(reduced_basis, axis=1)[:, None]

        
        # Initialize
        i0 = np.argmax(np.abs(reduced_basis[0]))
        emp_nodes_idx = [i0]
        EI_error = []

        for j in range(1, N_nodes):
            property_array = reduced_basis[j]
            empirical_interpolant = calc_empirical_interpolant(property_array, reduced_basis[:j], emp_nodes_idx)   


            residuals = property_array - empirical_interpolant
            print(f'EI error: {np.linalg.norm(residuals)}')
            EI_error.append(np.linalg.norm(residuals))

            next_idx = np.argmax(np.abs(residuals))
            emp_nodes_idx.append(next_idx)


            # Inside the loop
            if j%100 == 0:
                fig_residuals = plt.figure(figsize=(8, 4))
                plt.plot(np.abs(residuals).flatten(), label=f"Step {j}")
                plt.axvline(next_idx, color='r', linestyle='--', label=f"New node {next_idx}")
                plt.title(f"Residual at Step {j}")
                plt.xlabel("Sample index")
                plt.ylabel("|Residual|")
                plt.legend()
                plt.tight_layout()
                fig_residuals.savefig(f'Images/Empirical_nodes/test_residuals_{j}.png')

                plt.close('all')

        fig_error = plt.figure(figsize=(6, 4))
        plt.semilogy(EI_error, marker='o')
        plt.title("Empirical Interpolation Error Convergence")
        plt.xlabel("Iteration")
        plt.ylabel("‖Residual‖₂")
        plt.grid(True, which='both', ls='--')
        fig_error.savefig('Images/Empirical_nodes/test_error.png')

        # Example: Compare the 3rd waveform’s true vs interpolated
        j = 2
        wf_true = reduced_basis[j]
        wf_interp = calc_empirical_interpolant(wf_true, reduced_basis[:j], emp_nodes_idx[:j])

        fig_compare = plt.figure(figsize=(8, 4))
        plt.plot(wf_true, label="True waveform", lw=2)
        plt.plot(wf_interp, '--', label="Interpolated", lw=2)
        plt.scatter(emp_nodes_idx[:j], wf_true[emp_nodes_idx[:j]], color='red', label="Empirical nodes")
        plt.title(f"Empirical Interpolation at Step {j}")
        plt.legend()
        plt.tight_layout()
        fig_compare.savefig('Images/Empirical_nodes/test_compare.png')

        # Optional: Plot the empirical nodes if plot_emp_nodes_at_ecc is set
        if plot_emp_nodes_at_ecc:
            self._plot_empirical_nodes(emp_nodes_idx, property, plot_emp_nodes_at_ecc, save_fig)
        
        return emp_nodes_idx


    # def get_empirical_nodes(self, reduced_basis, property, plot_emp_nodes_at_ecc=True, save_fig=True):
    #     """
    #     Calculate the empirical nodes for a given dataset based on a reduced basis of residual properties.

    #     Parameters:
    #     ----------------
    #     - reduced_basis (numpy.ndarray): Reduced basis of residual properties (phase or amplitude).
    #     - property (str): Waveform property to evaluate, options are "phase" or "amplitude".
    #     - plot_emp_nodes_at_ecc (float, optional): If set, plots the empirical nodes at a specified eccentricity value.
    #     - save_fig (bool, optional): Saves the empirical nodes plot if set to True.

    #     Returns:
    #     ----------------
    #     - emp_nodes_idx (list): Indices of empirical nodes for the given dataset.
    #     """

    #     if property == 'phase':
    #         N_nodes = self.N_basis_vecs_phase
    #     if property == 'amplitude':
    #         N_nodes = self.N_basis_vecs_amp

    #     # def calc_empirical_interpolant(property_array, reduced_basis, emp_nodes_idx):
    #     #     """
    #     #     Calculates the empirical interpolant for a specific waveform property using a reduced basis.
            
    #     #     Parameters:
    #     #     ----------------
    #     #     - property_array (numpy.ndarray): The waveform property values (e.g., phase or amplitude).
    #     #     - reduced_basis (numpy.ndarray): Reduced basis of residual properties.
    #     #     - emp_nodes_idx (list): Indices of empirical nodes.

    #     #     Returns:
    #     #     ----------------
    #     #     - empirical_interpolant (numpy.ndarray): The computed empirical interpolant of the waveform property.
    #     #     """
    #     #     empirical_interpolant = np.zeros_like(property_array)
    #     #     m = len(emp_nodes_idx)
            
    #     #     # Prepare interpolation coefficients
    #     #     B_j_vec = np.zeros((reduced_basis.shape[1], m))
    #     #     # V = np.array([[reduced_basis[i][emp_nodes_idx[j]] for i in range(m)] for j in range(m)])
    #     #     V = np.array([[reduced_basis[i][emp_nodes_idx[j]] for j in range(m)] for i in range(m)])
    #     #     m = len(emp_nodes_idx[:j])
    #     #     print(f"Step {j}: reduced_basis[:j].shape = {reduced_basis[:j].shape}")
    #     #     print(f"Step {j}: property_array.shape = {property_array.shape}")
    #     #     print(f"Step {j}: emp_nodes_idx[:j] = {emp_nodes_idx[:j]}")
    #     #     V = np.array([[reduced_basis[i][emp_nodes_idx[k]] for k in range(m)] for i in range(m)])
    #     #     print(f"Step {j}: V.shape = {V.shape}, V_inv.norm() = {np.linalg.norm(np.linalg.pinv(V)):.2e}")
    #     #     B_j_vec = reduced_basis[:j].T @ np.linalg.pinv(V)
    #     #     print(f"Step {j}: B_j_vec.shape = {B_j_vec.shape}, max(B_j_vec) = {np.max(B_j_vec):.2e}")
    #     #     empirical_interpolant = B_j_vec @ property_array[emp_nodes_idx[:j]]
    #     #     print(f"Step {j}: empirical_interpolant norm = {np.linalg.norm(empirical_interpolant):.2e}, max = {np.max(empirical_interpolant):.2e}")

            
    #     #     # cond_V = np.linalg.cond(V)
    #     #     # print(f"Step {j}: condition number of V = {cond_V:.2e}")
            
    #     #     V_inv = np.linalg.pinv(V)  # pseudo-inverse for stability

    #     #     # Calculate B_j interpolation vector
    #     #     # for t in range(reduced_basis.shape[1]):
    #     #     #     for i in range(m):
    #     #             # B_j_vec[t, i] = np.dot(reduced_basis[:, t], V_inv[:, i])
    #     #     B_j_vec = reduced_basis.T @ V_inv   # shape: (n_samples, m)
    #     #     empirical_interpolant = B_j_vec @ property_array[emp_nodes_idx]


    #     #     # Compute the empirical interpolant
    #     #     # for j in range(reduced_basis.shape[0]):
    #     #     #     empirical_interpolant += B_j_vec[:, j] * property_array[emp_nodes_idx[j]]
    #     #     #     print(f"Step {j}: shapes -> B_j_vec {B_j_vec.shape}, property_array {property_array.shape}")

    #     #     return empirical_interpolant

    #     def calc_empirical_interpolant(property_array, reduced_basis, emp_nodes_idx):
    #         """
    #         Calculates the empirical interpolant for a specific waveform property using a reduced basis.
            
    #         Parameters:
    #         - property_array (np.ndarray): The waveform property values (e.g., phase or amplitude).
    #         - reduced_basis (np.ndarray): Reduced basis of residual properties (shape: n_basis_vectors x n_samples).
    #         - emp_nodes_idx (list): Indices of empirical nodes to use (length must match number of basis vectors).

    #         Returns:
    #         - empirical_interpolant (np.ndarray): The computed empirical interpolant (shape: n_samples).
    #         """
    #         m = len(emp_nodes_idx)
            
    #         # Construct interpolation matrix V (m x m)
    #         V = np.array([[reduced_basis[i, emp_nodes_idx[k]] for k in range(m)] for i in range(m)])
    #         cond_V = np.linalg.cond(V)
    #         print(f"Step m={m}: cond(V) = {cond_V:.2e}")
    #         V_inv = np.linalg.pinv(V)  # pseudo-inverse for stability

    #         # Compute B_j_vec: (n_samples x m)
    #         B_j_vec = reduced_basis[:m, :].T @ V_inv

    #         # Interpolant: multiply by the property values at the nodes
    #         property_values_at_nodes = property_array[emp_nodes_idx[:m]]
    #         empirical_interpolant = B_j_vec @ property_values_at_nodes

    #         # Debug prints (optional)
    #         print(f"calc_empirical_interpolant: m={m}, V.shape={V.shape}, B_j_vec.shape={B_j_vec.shape}, interpolant norm={np.linalg.norm(empirical_interpolant):.2e}")

    #         return empirical_interpolant
        
    #     # 1️⃣ Remove zero vectors
    #     nonzero_idx = [k for k, vec in enumerate(reduced_basis) if np.linalg.norm(vec) > 1e-14]
    #     reduced_basis = reduced_basis[nonzero_idx]

    #     # 2️⃣ Normalize remaining vectors
    #     # reduced_basis = reduced_basis / np.linalg.norm(reduced_basis, axis=1, keepdims=True)
    #     from scipy.linalg import orth
    #     reduced_basis = orth(reduced_basis.T).T

    #     for k in range(reduced_basis.shape[0]):
    #         print(f"Basis {k} norm: {np.linalg.norm(reduced_basis[k]):.2e}, max: {np.max(np.abs(reduced_basis[k])):.2e}")

       


    #     # Use this vector to start
    #     i = np.argmax(np.abs(reduced_basis[0]))
    #     emp_nodes_idx = [i]

    #     # i = np.argmax(reduced_basis[0])
    #     # emp_nodes_idx = [i]
    #     EI_error = []

    #     # Loop through the reduced basis to calculate interpolants
    #     for j in range(1, N_nodes):
    #         # Before calculating empirical_interpolant
    #         # Before calculating empirical_interpolant
    #         fig_basis = plt.figure(figsize=(8, 4))
    #         for k in range(j + 1):
    #             plt.plot(reduced_basis[k], label=f'Basis {k}')
    #         plt.title(f"Reduced Basis Vectors up to Step {j}")
    #         plt.xlabel("Sample index")
    #         plt.ylabel("Basis value")
    #         plt.legend()
    #         plt.tight_layout()
    #         fig_basis.savefig(f'Images/Empirical_nodes/basis_vectors_step_{j}.png')
    #         plt.close(fig_basis)

    #         fig_nodes = plt.figure(figsize=(8, 4))
    #         for k in range(j + 1):
    #             plt.plot(reduced_basis[k], label=f'Basis {k}')
    #         plt.scatter(emp_nodes_idx, [reduced_basis[0][idx] for idx in emp_nodes_idx],
    #                     color='red', marker='o', s=50, label='Empirical nodes')
    #         plt.title(f"Empirical Nodes Selected up to Step {j}")
    #         plt.xlabel("Sample index")
    #         plt.ylabel("Basis value")
    #         plt.legend()
    #         plt.tight_layout()
    #         fig_nodes.savefig(f'Images/Empirical_nodes/nodes_up_to_step_{j}.png')
    #         plt.close(fig_nodes)


    #         # empirical_interpolant = calc_empirical_interpolant(reduced_basis[j], reduced_basis[:j], emp_nodes_idx)
    #         empirical_interpolant = calc_empirical_interpolant(
    #             reduced_basis[j], 
    #             reduced_basis[:j], 
    #             emp_nodes_idx[:j]   # only previous nodes
    #         )



    #         fig_interp = plt.figure(figsize=(8, 4))
    #         plt.plot(reduced_basis[j], label="Target (true)", lw=2)
    #         plt.plot(empirical_interpolant, '--', label="Interpolant", lw=2)
    #         plt.scatter(emp_nodes_idx, reduced_basis[j][emp_nodes_idx],
    #                     color='red', label="Empirical nodes")
    #         plt.title(f"Interpolation Fit at Step {j}")
    #         plt.xlabel("Sample index")
    #         plt.ylabel("Value")
    #         plt.legend()
    #         plt.tight_layout()
    #         fig_interp.savefig(f'Images/Empirical_nodes/interp_fit_step_{j}.png')
    #         plt.close(fig_interp)

    #         residuals = reduced_basis[j] - empirical_interpolant  # shape: (n_samples,)
    #         # residuals = empirical_interpolant - reduced_basis[j][:, np.newaxis].T
    #         EI_error.append(np.linalg.norm(residuals))

    #         # Identify the next empirical node based on the maximum residual
    #         next_idx = np.argmax(np.abs(residuals))
    #         emp_nodes_idx.append(next_idx)

    #         # Inside the loop
    #         fig_residuals = plt.figure(figsize=(8, 4))
    #         for k in range(1, j + 1):
    #             if k < len(EI_error):
    #                 prev_residual = np.load(f'Images/Empirical_nodes/residual_values_step_{k}.npy')
    #                 plt.plot(np.abs(prev_residual).flatten(), alpha=0.4, label=f'Step {k}')
    #         plt.plot(np.abs(residuals).flatten(), label=f"Step {j} (current)", lw=2)
    #         plt.axvline(next_idx, color='r', linestyle='--', label=f"New node {next_idx}")
    #         plt.title(f"Residual Evolution up to Step {j}")
    #         plt.xlabel("Sample index")
    #         plt.ylabel("|Residual|")
    #         plt.legend()
    #         plt.tight_layout()
    #         fig_residuals.savefig(f'Images/Empirical_nodes/residuals_up_to_step_{j}.png')
    #         np.save(f'Images/Empirical_nodes/residual_values_step_{j}.npy', residuals)
    #         plt.close(fig_residuals)


    #     fig_error = plt.figure(figsize=(6, 4))
    #     plt.semilogy(EI_error, marker='o')
    #     plt.title("Empirical Interpolation Error Convergence")
    #     plt.xlabel("Iteration")
    #     plt.ylabel("‖Residual‖₂")
    #     plt.grid(True, which='both', ls='--')
    #     fig_error.savefig('Images/Empirical_nodes/test_error.png')

    #     # Example: Compare the 3rd waveform’s true vs interpolated
    #     j = 2
    #     wf_true = reduced_basis[j]
    #     wf_interp = calc_empirical_interpolant(wf_true, reduced_basis[:j], emp_nodes_idx[:j])

    #     fig_compare = plt.figure(figsize=(8, 4))
    #     plt.plot(wf_true, label="True waveform", lw=2)
    #     plt.plot(wf_interp, '--', label="Interpolated", lw=2)
    #     plt.scatter(emp_nodes_idx[:j], wf_true[emp_nodes_idx[:j]], color='red', label="Empirical nodes")
    #     plt.title(f"Empirical Interpolation at Step {j}")
    #     plt.legend()
    #     plt.tight_layout()
    #     fig_compare.savefig('Images/Empirical_nodes/test_compare.png')

    #     # Optional: Plot the empirical nodes if plot_emp_nodes_at_ecc is set
    #     if plot_emp_nodes_at_ecc:
    #         self._plot_empirical_nodes(emp_nodes_idx, property, plot_emp_nodes_at_ecc, save_fig)
        
    #     return emp_nodes_idx

    def get_empirical_nodes(self, reduced_basis, property, plot_emp_nodes_at_ecc=True, save_fig=True):
        """
        Calculate the empirical nodes for a given dataset based on a reduced basis of residual properties.

        Parameters:
        ----------------
        - reduced_basis (numpy.ndarray): Reduced basis of residual properties (phase or amplitude).
        - property (str): Waveform property to evaluate, options are "phase" or "amplitude".
        - plot_emp_nodes_at_ecc (float, optional): If set, plots the empirical nodes at a specified eccentricity value.
        - save_fig (bool, optional): Saves the empirical nodes plot if set to True.

        Returns:
        ----------------
        - emp_nodes_idx (list): Indices of empirical nodes for the given dataset.
        """

        # Set number of nodes based on property
        if property == 'phase':
            N_nodes = self.N_basis_vecs_phase
        elif property == 'amplitude':
            N_nodes = self.N_basis_vecs_amp
        else:
            raise ValueError("Property must be 'phase' or 'amplitude'.")

        # -------------------------------
        # Preprocess the basis
        # -------------------------------

        # 1️⃣ Remove zero or near-zero vectors
        nonzero_idx = [k for k, vec in enumerate(reduced_basis) if np.linalg.norm(vec) > 1e-14]
        reduced_basis = reduced_basis[nonzero_idx]

        # 2️⃣ Orthonormalize with scipy.linalg.orth
        reduced_basis = orth(reduced_basis.T).T

        for k in range(reduced_basis.shape[0]):
            print(f"Basis {k} norm: {np.linalg.norm(reduced_basis[k]):.2e}, max: {np.max(np.abs(reduced_basis[k])):.2e}")

        # -------------------------------
        # Helper: empirical interpolant
        # -------------------------------
        def calc_empirical_interpolant(property_array, reduced_basis, emp_nodes_idx):
            m = len(emp_nodes_idx)
            V = np.array([[reduced_basis[i, emp_nodes_idx[k]] for k in range(m)] for i in range(m)])
            cond_V = np.linalg.cond(V)
            print(f"Step m={m}: cond(V) = {cond_V:.2e}")
            V_inv = np.linalg.pinv(V)
            B_j_vec = reduced_basis[:m, :].T @ V_inv
            empirical_interpolant = B_j_vec @ property_array[emp_nodes_idx[:m]]
            print(f"calc_empirical_interpolant: m={m}, V.shape={V.shape}, B_j_vec.shape={B_j_vec.shape}, interpolant norm={np.linalg.norm(empirical_interpolant):.2e}")
            return empirical_interpolant

        # -------------------------------
        # Initialize empirical nodes
        # -------------------------------
        i = np.argmax(np.abs(reduced_basis[0]))
        emp_nodes_idx = [i]
        EI_error = []

        # -------------------------------
        # Main loop for empirical nodes
        # -------------------------------
        print(f"Starting empirical interpolation with {N_nodes} nodes...", "reduced_basis shape:", reduced_basis.shape)
        for j in range(1, N_nodes):
        #     # Plot basis vectors
        #     fig_basis = plt.figure(figsize=(8, 4))
        #     for k in range(j + 1):
        #         plt.plot(reduced_basis[k], label=f'Basis {k}')
        #     plt.title(f"Reduced Basis Vectors up to Step {j}")
        #     plt.xlabel("Sample index")
        #     plt.ylabel("Basis value")
        #     plt.legend()
        #     plt.tight_layout()
        #     fig_basis.savefig(f'Images/Empirical_nodes/basis_vectors_step_{j}.png')
        #     plt.close(fig_basis)

        #     # Plot current empirical nodes
        #     fig_nodes = plt.figure(figsize=(8, 4))
        #     for k in range(j + 1):
        #         plt.plot(reduced_basis[k], label=f'Basis {k}')
        #     plt.scatter(emp_nodes_idx, [reduced_basis[0][idx] for idx in emp_nodes_idx],
        #                 color='red', marker='o', s=50, label='Empirical nodes')
        #     plt.title(f"Empirical Nodes Selected up to Step {j}")
        #     plt.xlabel("Sample index")
        #     plt.ylabel("Basis value")
        #     plt.legend()
        #     plt.tight_layout()
        #     fig_nodes.savefig(f'Images/Empirical_nodes/nodes_up_to_step_{j}.png')
        #     plt.close(fig_nodes)

            # Compute empirical interpolant
            empirical_interpolant = calc_empirical_interpolant(
                reduced_basis[j],
                reduced_basis[:j],
                emp_nodes_idx[:j]
            )

            # Plot interpolation fit
            # fig_interp = plt.figure(figsize=(8, 4))
            # plt.plot(reduced_basis[j], label="Target (true)", lw=2)
            # plt.plot(empirical_interpolant, '--', label="Interpolant", lw=2)
            # plt.scatter(emp_nodes_idx, reduced_basis[j][emp_nodes_idx], color='red', label="Empirical nodes")
            # plt.title(f"Interpolation Fit at Step {j}")
            # plt.xlabel("Sample index")
            # plt.ylabel("Value")
            # plt.legend()
            # plt.tight_layout()
            # fig_interp.savefig(f'Images/Empirical_nodes/interp_fit_step_{j}.png')
            # plt.close(fig_interp)

            # Compute residuals
            residuals = reduced_basis[j] - empirical_interpolant
            EI_error.append(np.linalg.norm(residuals))

            # Identify next empirical node
            next_idx = np.argmax(np.abs(residuals))
            emp_nodes_idx.append(next_idx)

            # Plot residual evolution
            # fig_residuals = plt.figure(figsize=(8, 4))
            for k in range(1, j + 1):
                if k < len(EI_error):
                    try:
                        prev_residual = np.load(f'Images/Empirical_nodes/residual_values_step_{k}.npy')
                        # plt.plot(np.abs(prev_residual).flatten(), alpha=0.4, label=f'Step {k}')
                    except FileNotFoundError:
                        pass
        #     plt.plot(np.abs(residuals).flatten(), label=f"Step {j} (current)", lw=2)
        #     plt.axvline(next_idx, color='r', linestyle='--', label=f"New node {next_idx}")
        #     plt.title(f"Residual Evolution up to Step {j}")
        #     plt.xlabel("Sample index")
        #     plt.ylabel("|Residual|")
        #     plt.legend()
        #     plt.tight_layout()
        #     fig_residuals.savefig(f'Images/Empirical_nodes/residuals_up_to_step_{j}.png')
        #     np.save(f'Images/Empirical_nodes/residual_values_step_{j}.npy', residuals)
        #     plt.close(fig_residuals)

        # # Plot EI error convergence
        # fig_error = plt.figure(figsize=(6, 4))
        # plt.semilogy(EI_error, marker='o')
        # plt.title("Empirical Interpolation Error Convergence")
        # plt.xlabel("Iteration")
        # plt.ylabel("‖Residual‖₂")
        # plt.grid(True, which='both', ls='--')
        # fig_error.savefig('Images/Empirical_nodes/test_error.png')

        # # Compare 3rd waveform as example
        # j = 2
        # wf_true = reduced_basis[j]
        # wf_interp = calc_empirical_interpolant(wf_true, reduced_basis[:j], emp_nodes_idx[:j])
        # fig_compare = plt.figure(figsize=(8, 4))
        # plt.plot(wf_true, label="True waveform", lw=2)
        # plt.plot(wf_interp, '--', label="Interpolated", lw=2)
        # plt.scatter(emp_nodes_idx[:j], wf_true[emp_nodes_idx[:j]], color='red', label="Empirical nodes")
        # plt.title(f"Empirical Interpolation at Step {j}")
        # plt.legend()
        # plt.tight_layout()
        # fig_compare.savefig('Images/Empirical_nodes/test_compare.png')

        # Optional: plot empirical nodes at eccentricity
        if plot_emp_nodes_at_ecc:
            self._plot_empirical_nodes(emp_nodes_idx, property, plot_emp_nodes_at_ecc, save_fig)

        return emp_nodes_idx


    def _plot_empirical_nodes(self, emp_nodes_idx, property, eccentricity, save_fig):
        """
        Helper function to plot empirical nodes for a given eccentricity.

        Parameters:
        ----------------
        - emp_nodes_idx (list): Indices of the empirical nodes.
        - property (str): Waveform property being plotted (e.g., "phase" or "amplitude").
        - eccentricity (float): Eccentricity value for the plot.
        - save_fig (bool): If True, saves the plot to a file.
        """
        # Get the waveform at the requested eccentricity
        hp, hc = self.simulate_inspiral(ecc_ref=eccentricity)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the waveform and the empirical nodes together for reference
        ax.plot(self.time, hp, linewidth=0.2, color='black', label=f'$h_+$: ecc = {eccentricity}')
        ax.plot(self.time, hc, linewidth=0.2, linestyle='dashed', color='black', label=f'$h_\times$: ecc = {eccentricity}')
        ax.scatter(self.time[emp_nodes_idx], np.zeros(len(emp_nodes_idx)), color='red', s=8)

        ax.set_ylabel(f'$h_{{22}}$')
        ax.set_xlabel('t [M]')
        ax.legend(loc='upper left')

        # IF savefig is set to True, save the figure
        if save_fig:
            fig_path = f'Images/Empirical_nodes/EIM_{property}_e={eccentricity}_M={self.total_mass}_f_lower={self.f_lower}_f_ref={self.f_ref}_iN={len(self.ecc_ref_parameter_space_input)}_gN={len(self.best_rep_parameters_idx)}_ms={self.minimum_spacing_greedy}.png'
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            fig.savefig(fig_path)
            print(self.colored_text(f'Figure is saved in {fig_path}', 'blue'))

    def _plot_empirical_nodes_from_dataset(self, emp_nodes_idx, property, eccentricity, save_fig):
        """
        Helper function to plot empirical nodes for a given eccentricity.

        Parameters:
        ----------------
        - emp_nodes_idx (list): Indices of the empirical nodes.
        - property (str): Waveform property being plotted (e.g., "phase" or "amplitude").
        - eccentricity (float): Eccentricity value for the plot.
        - save_fig (bool): If True, saves the plot to a file.
        """
        # Get the waveform at the requested eccentricity
        hp, hc = self.simulate_inspiral(ecc_ref=eccentricity)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the waveform and the empirical nodes together for reference
        ax.plot(self.time, hp, linewidth=0.2, color='black', label=f'$h_+$: ecc = {eccentricity}')
        ax.plot(self.time, hc, linewidth=0.2, linestyle='dashed', color='black', label=f'$h_\times$: ecc = {eccentricity}')
        ax.scatter(self.time[emp_nodes_idx], np.zeros(len(emp_nodes_idx)), color='red', s=8)

        ax.set_ylabel(f'$h_{{22}}$')
        ax.set_xlabel('t [M]')
        ax.legend(loc='upper left')

        # IF savefig is set to True, save the figure
        if save_fig:
            fig_path = f'Images/Empirical_nodes/EIM_dataset_{property}_e={eccentricity}_M={self.total_mass}_f_lower={self.f_lower}_f_ref={self.f_ref}_iN={len(self.ecc_ref_parameter_space_input)}_gN={len(self.best_rep_parameters_idx)}_ms={self.minimum_spacing_greedy}.png'
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            fig.savefig(fig_path)
            print(self.colored_text(f'Figure is saved in {fig_path}', 'blue'))

  
    
    def get_training_set_greedy(self, property, emp_nodes_of_full_dataset=False, min_greedy_error=None, N_greedy_vecs=None, plot_training_set=False, 
                        plot_greedy_error=False, save_fig_greedy_error=False, plot_emp_nodes_at_ecc=False, save_fig_emp_nodes=False, save_fig_training_set=False, 
                        save_dataset_to_file=True, save_fig_residuals_eccentric=False, save_fig_residuals_time=False, plot_greedy_vecs=False, save_fig_greedy_vecs=False):
        """
        Generate a training set for the surrogate model by calculating residuals, selecting greedy parameters, and determining empirical nodes.
        
        Parameters:
        ----------------
        - property (str): Waveform property (e.g., 'phase' or 'amplitude') for generating the dataset.
        - min_greedy_error (float, optional): Minimum greedy error threshold for stopping criterion in greedy selection.
        - N_greedy_vecs (int, optional): Number of greedy vectors to select.
        - plot_training_set (bool, optional): If True, plots the training set.
        - plot_greedy_error (bool, optional): If True, plots greedy error.
        - plot_emp_nodes_at_ecc (float or bool, optional): If True, plots empirical nodes at specified eccentricity.
        - save_fig (bool, optional): If True, saves the plot of the training set.
        - save_dataset_to_file (bool, optional): If True, saves the generated dataset.

        Returns:
        ----------------
        - residual_training_set (numpy.ndarray): Training set of residuals at empirical nodes.
        """
        # Import from class object if min_greedy and N_greedy vecs are not specified
        if (min_greedy_error is None) and (N_greedy_vecs is None):
            if property == 'phase':
                min_greedy_error = self.min_greedy_error_phase
                N_greedy_vecs = self.N_basis_vecs_phase
            elif property == 'amplitude':
                min_greedy_error = self.min_greedy_error_amp
                N_greedy_vecs = self.N_basis_vecs_amp

        
        # Step 1: Generate residuals for the full parameter space
        residual_parameterspace_input = self.generate_property_dataset(
            ecc_list=self.ecc_ref_parameter_space_input,
            property=property,
            save_dataset_to_file=save_dataset_to_file,
            plot_residuals_eccentric_evolv=True,
            plot_residuals_time_evolv=True,
            save_fig_eccentric_evolv=save_fig_residuals_eccentric,
            save_fig_time_evolve=save_fig_residuals_time
        )
        
        # Step 2: Select the best representative parameters using a greedy algorithm
        print('Calculating greedy parameters...')
        self.best_rep_parameters_idx, residual_greedy_basis_orthonormal = self.get_greedy_parameters(
            U=residual_parameterspace_input,
            property=property,
            N_greedy_vecs=N_greedy_vecs,
            min_greedy_error=min_greedy_error,
            plot_greedy_error=plot_greedy_error,
            save_greedy_error_fig=save_fig_greedy_error,
            plot_greedy_vectors=plot_greedy_vecs,
            save_greedy_vecs_fig=save_fig_greedy_vecs
        )

        self.residual_reduced_basis = residual_parameterspace_input[self.best_rep_parameters_idx]
        self.best_rep_parameters = list(self.ecc_ref_parameter_space_input[self.best_rep_parameters_idx])
        # self.best_rep_parameters_idx, self.residual_reduced_basis = self.get_greedy_parameters(
        #     U=residual_parameterspace_input,
        #     min_greedy_error=min_greedy_error,
        #     N_greedy_vecs=N_greedy_vecs,
        #     property=property,
        #     plot_greedy_error=plot_greedy_error,
        #     save_greedy_fig=save_fig_greedy_error,
        # )
        
        # Step 3: Calculate empirical nodes of the greedy basis
        print('Calculating empirical nodes...')
        if emp_nodes_of_full_dataset:
            self.empirical_nodes_idx = self.empirical_interpolation_from_dataset(
            waveforms_dataset=residual_parameterspace_input,
            property=property,
            plot_emp_nodes_at_ecc=plot_emp_nodes_at_ecc,
            save_fig=save_fig_emp_nodes
        )
        else:
            self.empirical_nodes_idx = self.get_empirical_nodes(
                reduced_basis=residual_greedy_basis_orthonormal,
                property=property,
                plot_emp_nodes_at_ecc=plot_emp_nodes_at_ecc,
                save_fig=save_fig_emp_nodes
            )

        # self.empirical_nodes_idx = self.get_empirical_nodes_test(
        #     reduced_basis=residual_greedy_basis_orthonormal,
        #     property=property,
        #     plot_emp_nodes_at_ecc=plot_emp_nodes_at_ecc,
        #     save_fig=save_fig_emp_nodes
        # )


        # print(f'emp nodes: {self.empirical_nodes_idx}, {empirical_nodes_idx}')
        
        # Step 4: Generate the training set at empirical nodes
        residual_training_set = self.residual_reduced_basis[:, self.empirical_nodes_idx]
        self.time_training = self.time[self.empirical_nodes_idx]

        # Optionally plot the training set
        if plot_training_set:
            self._plot_training_set(property, save_fig_training_set)
        print('vecs:', self.N_basis_vecs_phase, self.N_basis_vecs_amp)
        return residual_training_set

    def _plot_training_set(self, property, save_fig):
        """
        Helper function to plot and optionally save the training set of residuals.

        Parameters:
        ----------------
        - residual_training_set (numpy.ndarray): Training set of residuals at empirical nodes.
        - property (str): The waveform property ('phase' or 'amplitude').
        - save_fig (bool): If True, saves the plot to a file.
        """
        fig, ax = plt.subplots()

        for i, idx in enumerate(self.best_rep_parameters_idx):
            ax.plot(self.time, self.residual_reduced_basis[i], label=f'e={round(self.ecc_ref_parameter_space_input[idx], 3)}', linewidth=0.6)
            ax.scatter(self.time[self.empirical_nodes_idx], self.residual_reduced_basis[i][self.empirical_nodes_idx])

        ax.set_xlabel('t [M]')
        ax.set_ylabel('greedy residual')
        ax.legend()
        ax.set_title('Residual Training Set')
        ax.grid(True)

        if save_fig:
            figname = f'Images/Trainingset/Training_set_{property}_M={self.total_mass}_ecc_list=[{min(self.ecc_ref_parameter_space_input)}_{max(self.ecc_ref_parameter_space_input)}]_f_lower={self.f_lower}_f_ref={self.f_ref}_gN={len(self.best_rep_parameters_idx)}.png'
            os.makedirs('Images/TrainingSet', exist_ok=True)
            fig.savefig(figname)
            print(self.colored_text(f'Figure is saved in {figname}', 'blue'))

sampling_frequency = 2048 # or 4096
duration = 4 # seconds
time_array = np.linspace(-duration, 0, int(sampling_frequency * duration))  # time in seconds

gt = Generate_TrainingSet(time_array=time_array, ecc_ref_parameterspace=np.linspace(0, 0.3, num=10), mean_ano_parameterspace=[0], N_basis_vecs_amp=20, N_basis_vecs_phase=20,
                          minimum_spacing_greedy=0.003 )
res_ds_phase = gt.generate_property_dataset(np.linspace(0, 0.3, num=10), 'phase', plot_residuals_time_evolv=True, save_fig_time_evolve=True, plot_residuals_eccentric_evolv=True, save_fig_eccentric_evolv=True)
res_ds_amp = gt.generate_property_dataset(np.linspace(0, 0.3, num=10), 'amplitude', plot_residuals_time_evolv=True, save_fig_time_evolve=True, plot_residuals_eccentric_evolv=True, save_fig_eccentric_evolv=True)
# hp, hc = gt.simulate_inspiral(0.3, geometric_units=True)
# res_ds_amp = gt.calculate_residual(hp, hc, 0.3, 'amplitude', plot_residual=True, save_fig=True)
# gt.get_greedy_parameters(U=res_ds_phase, property='phase', N_greedy_vecs=21)
# gt.get_greedy_parameters(U=res_ds_amp, property='amplitude', N_greedy_vecs=21)
# gt.get_empirical_nodes(res_ds_phase, 'phase', plot_emp_nodes_at_ecc=0.1, save_fig=True)
# gt.get_empirical_nodes(res_ds_amp, 'amplitude', plot_emp_nodes_at_ecc=0.1, save_fig=True)

# hp, hc = gt.simulate_inspiral(0.0)

# N_vecs = [35, 30, 25, 20]
# residual = gt.calculate_residual(hp, hc, 0.0, 'phase', plot_residual=True, save_fig=True)
# gt.calculate_residual(hp, hc, 0.0, 'amplitude', plot_residual=True, save_fig=True)
# for vecs in N_vecs:
#     gt.get_greedy_parameters(res_ds_phase, 'phase', N_greedy_vecs=vecs)
# gt.get_greedy_parameters(res_ds_amp, 'amplitude', N_greedy_vecs=15, plot_greedy_vectors=True, save_greedy_vecs_fig=True)

# for ecc in np.linspace(0, 0.2, num=20)[10:]:
# #     print(ecc)
#     hp, hc = gt.simulate_inspiral(ecc, plot_polarisations=True, save_fig=True)
#     gt.calculate_residual(hp, hc, ecc, 'phase', plot_residual=True, save_fig=True)
# print(np.linspace(0, 0.2, num=100)[-10:])
# gt._generate_polarisation_data(np.linspace(0.01, 0.5, num=20))
# gt.get_training_set_greedy(property='phase', N_greedy_vecs=21)
# gt.get_training_set_greedy(property='amplitude', N_greedy_vecs=21)
# gt.get_training_set_greedy_test(property='phase', N_greedy_vecs=20, plot_emp_nodes_at_ecc=0.1, save_fig_emp_nodes=True)
# gt.generate_property_dataset(ecc_list=np.linspace(0.01, 0.2, num=5), property='phase', plot_residuals_eccentric_evolv=True, plot_residuals_time_evolv=True)
# gt.generate_property_dataset(ecc_list=np.linspace(0.01, 0.2, num=5), property='amplitude', plot_residuals_eccentric_evolv=True, plot_residuals_time_evolv=True)
# gt.get_training_set_greedy(property='phase', plot_emp_nodes_at_ecc=0.1, plot_greedy_error=True, plot_residuals_eccentric_evolve=True, plot_residuals_time_evolve=True, plot_training_set=True)
# # gt.get_training_set_greedy(property='amplitude', plot_emp_nodes_at_ecc=0.1, plot_greedy_error=True, plot_residuals_eccentric_evolve=True, plot_residuals_time_evolve=True, plot_training_set=True,  save_fig_residuals_eccentric=True, save_fig_residuals_time=True, save_fig_training_set=True)
# gt.get_training_set_greedy(property='phase', plot_emp_nodes_at_ecc=0.1, save_fig_emp_nodes=True)
# gt.get_training_set_greedy(property='amplitude', plot_emp_nodes_at_ecc=0.11, save_fig_emp_nodes=True)
# plt.show()


