from generate_PhenomTE import *
import random
from sklearn.preprocessing import normalize

plt.switch_backend('WebAgg')

class Generate_TrainingSet(Waveform_Properties, Simulate_Inspiral):
    """
    Class to generate a training dataset for gravitational waveform simulations using a greedy algorithm and empirical interpolation.
    Inherits from WaveformProperties and SimulateInspiral to leverage methods for waveform 
    property calculations and waveform generation.

    """

    def __init__(self, time_array, ecc_ref_parameterspace, total_mass, luminosity_distance, f_lower=10, f_ref=20, chi1=0, chi2=0, phiRef=0., rel_anomaly=0., inclination=0., truncate_at_ISCO=True, truncate_at_tmin=True):
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
        self.parameter_space_input = ecc_ref_parameterspace
    
        # To be stored parameters
        self.time = None
        self.residual_greedy_basis = None
        self.greedy_parameters_idx = None
        self.empirical_nodes_idx = None
        self.highest_tmin_value = None

        # Inherit parameters from all previously defined classes
        super().__init__(time_array, None, total_mass, luminosity_distance, f_lower, f_ref, chi1, chi2, phiRef, rel_anomaly, inclination, truncate_at_ISCO, truncate_at_tmin)
    
    def generate_property_dataset(self, ecc_list, property, save_dataset_to_file=None, plot_residuals_time_evolv=False, plot_residuals_eccentric_evolv=False, save_fig_eccentric_evolv=False, save_fig_time_evolve=False):
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

            # If save_dataset_to_file is True save the residuals to file in Straindata/Residuals
            if save_dataset_to_file is True and not os.path.isfile(f'Straindata/Residuals/residuals_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(ecc_list)}_{max(ecc_list)}_N={len(ecc_list)}].npz'):
                self._save_residual_dataset(ecc_list, property, residual_dataset)
       
        # If plot_residuals is True, plot whole residual dataset
        if (plot_residuals_eccentric_evolv is True) or (plot_residuals_time_evolv is True):
            self._plot_residuals(residual_dataset, ecc_list, property, plot_residuals_eccentric_evolv, plot_residuals_time_evolv, save_fig_eccentric_evolv, save_fig_time_evolve )
        
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
            hp_ISCO, hc_ISCO = self.simulate_inspiral_mass_independent(ISCO_ecc, truncate_at_ISCO=True)


            hp_dataset = np.zeros((len(ecc_list), len(self.time)), dtype=np.float32)  # float32 to save memory storage
            hc_dataset = np.zeros((len(ecc_list), len(self.time)), dtype=np.float32)

            for i, ecc in enumerate(ecc_list):
                if ecc == ISCO_ecc:
                    # Store first waveform in dataset
                    hp_dataset[i] = hp_ISCO
                    hc_dataset[i] = hc_ISCO
                else:
                    hp, hc = self.simulate_inspiral_mass_independent(ecc, truncate_at_ISCO=False)

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
    
    def _plot_residuals(self, residual_dataset, ecc_list, property, plot_eccentric_evolv=False, plot_time_evolve=False, save_fig_eccentric_evolve=False, save_fig_time_evolve=False):
        """Function to plot residuals dataset including save figure option."""
        if plot_eccentric_evolv is True:
            fig_residuals_ecc = plt.figure()

            for i in range(len(residual_dataset)):
                plt.plot(ecc_list, residual_dataset.T[i])
                
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
            # plt.legend()

            plt.tight_layout()

            if save_fig_eccentric_evolve is True:
                ISCO = '' if self.truncate_at_ISCO else 'NO_ISCO_'
                tmin = '' if self.truncate_at_tmin else 'NO_tmin_'

                figname = f'Residuals_eccentric_evolv_{property}_{ISCO}{tmin}M={self.total_mass}_ecc_list=[{round(min(ecc_list), 2)}_{round(max(ecc_list), 2)}_N={len(self.parameter_space_input)}]_f_lower={self.f_lower}_f_ref={self.f_ref}.png'
               
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Residuals', exist_ok=True)
                fig_residuals_ecc.savefig('Images/Residuals/' + figname)

                print('Figure is saved in Images/Residuals' + figname)
                plt.close('all') # Clean up plot

        if plot_time_evolve is True:
            fig_residuals_t = plt.figure()

            for i in range(len(residual_dataset)):
                plt.plot(self.time, residual_dataset[i], label='e$_{min}$' + f' = {round(ecc_list[i], 3)}', linewidth=0.6)
               
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
            # plt.legend()

            plt.tight_layout()

            if save_fig_time_evolve is True:
                ISCO = '' if self.truncate_at_ISCO else 'NO_ISCO'
                tmin = '' if self.truncate_at_tmin else 'NO_tmin'

           
                figname = f'Residuals_time_evolv_{property}_{ISCO}_{tmin}_M={self.total_mass}_ecc_list=[{min(ecc_list)}_{max(ecc_list)}_N={len(self.parameter_space_input)}]_f_lower={self.f_lower}_f_ref={self.f_ref}.png'
           
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Residuals', exist_ok=True)
                fig_residuals_t.savefig('Images/Residuals/' + figname)

                print('Figure is saved in Images/Residuals' + figname)
                plt.close('all')
    
    def _save_residual_dataset(self, ecc_list, property, residual_dataset):
        """Function to save residual dataset to file."""

        os.makedirs('Straindata/Residuals', exist_ok=True)
        file_path = f'Straindata/Residuals/residuals_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(ecc_list)}_{max(ecc_list)}_N={len(ecc_list)}].npz'
        np.savez(file_path, residual=residual_dataset, time=self.time, eccentricities=ecc_list)
        print('Residuals saved to Straindata/Residuals')


    def get_greedy_parameters(self, U, property, min_greedy_error=None, N_greedy_vecs=None, reg=1e-6, 
                            plot_greedy_error=True, plot_validation_errors=False, save_validation_fig=False, 
                            save_greedy_fig=True):
        """
        Perform strong greedy algorithm to select the basis vectors with highest uniqueness. 
        The process stops when either convergence is reached or when a specified number of 
        basis vectors is reached.

        Parameters:
        ----------
        U : numpy.ndarray
            Non-normalized training set where each row represents a data point.
        property : str
            Specifies which property ('phase', 'amplitude') to compute.
        min_greedy_error : float, optional
            Stop the algorithm once the minimum greedy error is reached.
        N_greedy_vecs : int, optional
            Stop the algorithm once a specified number of basis vectors is reached.
        reg : float, optional
            Regularization parameter to stabilize computation, default is 1e-6.
        plot_greedy_error : bool, optional
            If True, plots the greedy error for each added basis vector.
        plot_validation_errors : bool, optional
            If True, plots the validation errors comparing greedy and trivial bases.
        save_validation_fig : bool, optional
            If True, saves the validation error plot to file.
        save_greedy_fig : bool, optional
            If True, saves the greedy error plot to file.

        Returns:
        -------
        greedy_parameters_idx : list
            Indices of the selected greedy basis vectors.
        greedy_basis : numpy.ndarray
            Selected basis vectors based on highest uniqueness.
        """

        def calc_validation_vectors(num_vectors):
            """Randomly samples validation vectors from parameter space."""
            parameter_space = np.linspace(min(self.parameter_space_input), max(self.parameter_space_input), num=5000).round(4)
            validation_set = random.sample(list(parameter_space), num_vectors)
            validation_vecs = self.generate_property_dataset(property=property, ecc_list=validation_set)
            return validation_vecs

        # Argument checks and initial setup
        if (min_greedy_error is None) == (N_greedy_vecs is None):
            raise ValueError("Specify either min_greedy_error (float) or N_greedy_vecs (int), not both.")

       # Normalize the dataset U
        U_copy = U.copy()
        U_normalised = normalize(U_copy, axis=1) 
        U_copy = None # Free memory

        normalised_basis = np.empty((0, U.shape[1]))  # Initialize an empty array for the basis
        greedy_basis = np.empty((0, U.shape[1]))  # Initialize an empty array for the basis
        
        greedy_parameters_idx = []
        errors = [1]
        greedy_errors=[1]

        while True:
            # Either break the loop when minimum greedy error is reached or when the specified amount of vectors in the greedy basis is reached.
            if min_greedy_error is not None:
                if np.max(errors) <= min_greedy_error or len(greedy_basis) == len(U):
                    break
            if N_greedy_vecs is not None or len(greedy_basis) == len(U):
                if len(greedy_basis) >= N_greedy_vecs:
                    break

            # Compute projection errors using normalized U
            G = np.dot(normalised_basis, normalised_basis.T) + reg * np.eye(normalised_basis.shape[0]) if normalised_basis.size > 0 else np.zeros((0, 0))  # Compute Gramian
            R = np.dot(normalised_basis, U_normalised.T)  # Compute inner product
            lambdas = np.linalg.lstsq(G, R, rcond=None)[0] if normalised_basis.size > 0 else np.zeros((0, U_normalised.shape[0]))  # Use pseudoinverse
            U_proj = np.dot(lambdas.T, normalised_basis) if normalised_basis.size > 0 else np.zeros_like(U_normalised)  # Compute projection
            
            errors = np.linalg.norm(U_normalised - U_proj, axis=1)  # Calculate errors
            max_error_idx = np.argmax(errors)

            # Extend basis with non-normalized U
            normalised_basis = np.vstack([normalised_basis, U_normalised[max_error_idx]])
            greedy_basis = np.vstack([greedy_basis, U[max_error_idx]])

            greedy_parameters_idx.append(int(max_error_idx))
            greedy_errors.append(np.max(errors))

        # Plot greedy errors if requested
        if plot_greedy_error:
            self._plot_greedy_errors(greedy_errors, property, save_greedy_fig)

        # Plot validation errors
        if plot_validation_errors:
            validation_vecs = calc_validation_vectors(15)
            trivial_basis = U[:len(greedy_basis)]
            self._plot_validation_errors(validation_vecs, greedy_basis, trivial_basis, property, save_validation_fig)

        print(f'Highest error of best approximation of the basis: {round(np.min(greedy_errors), 5)} | {len(greedy_basis)} basis vectors')
        print(greedy_parameters_idx)
        return greedy_parameters_idx, greedy_basis

    def _plot_greedy_errors(self, greedy_errors, property, save_greedy_fig):
        """Function to plot and option to save the greedy errors."""
        N_basis_vectors = np.arange(1, len(greedy_errors) + 1)

        plt.figure(figsize=(7, 5))
        plt.plot(N_basis_vectors, greedy_errors, label='Greedy Errors')
        plt.scatter(N_basis_vectors, greedy_errors, s=4)
        # for i, label in enumerate(self.parameter_space_input[greedy_parameters_idx]):
        #     plt.annotate(label, (N_basis_vectors[i], greedy_errors[i]), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=5.5)
        plt.xlabel('Number of Waveforms')
        if property == 'phase':
            plt.ylabel(f'Greedy error $\Delta \phi$')
        elif property == 'amplitude':
            plt.ylabel(f'Greedy error $\Delta A$')
        plt.yscale('log')
        # plt.title('Greedy errors of residual {} for N = {}'.format(property, len(greedy_errors)-1))
        plt.grid(True)

        if save_greedy_fig:
            os.makedirs('Images/Greedy_errors', exist_ok=True)
            plt.savefig(f'Images/Greedy_errors/Greedy_error_{property}_M={self.total_mass}_ecc=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}]_f_lower={self.f_lower}_f_ref={self.f_ref}_iN={len(self.parameter_space_input)}_gerr={min(greedy_errors).round(4)}.png')
            
            print('Greedy error fig saved to Images/Greedy_errors')
            plt.close('all')

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
            plt.savefig(f'Images/Validation_errors/Validation_error_{property}_M={self.total_mass}_ecc=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}]_f_lower={self.f_lower}_f_ref={self.f_ref}_iN={len(self.parameter_space_input)}.png')
            plt.close('all')

    
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
        
        def calc_empirical_interpolant(waveform_property, reduced_basis, emp_nodes_idx):
            """
            Calculates the empirical interpolant for a specific waveform property using a reduced basis.
            
            Parameters:
            ----------------
            - waveform_property (numpy.ndarray): The waveform property values (e.g., phase or amplitude).
            - reduced_basis (numpy.ndarray): Reduced basis of residual properties.
            - emp_nodes_idx (list): Indices of empirical nodes.

            Returns:
            ----------------
            - empirical_interpolant (numpy.ndarray): The computed empirical interpolant of the waveform property.
            """
            empirical_interpolant = np.zeros_like(waveform_property)
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
                empirical_interpolant += B_j_vec[:, j] * waveform_property[emp_nodes_idx[j]]

            return empirical_interpolant

        # Initialize with the index of the maximum value in the first basis vector
        i = np.argmax(reduced_basis[0])
        emp_nodes_idx = [i]
        EI_error = []

        # Loop through the reduced basis to calculate interpolants
        for j in range(1, reduced_basis.shape[0]):
            empirical_interpolant = calc_empirical_interpolant(reduced_basis[j], reduced_basis[:j], emp_nodes_idx)
            residuals = empirical_interpolant - reduced_basis[j][:, np.newaxis].T
            EI_error.append(np.linalg.norm(residuals))

            # Identify the next empirical node based on the maximum residual
            next_idx = np.argmax(np.abs(residuals))
            emp_nodes_idx.append(next_idx)

        # Optional: Plot the empirical nodes if plot_emp_nodes_at_ecc is set
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
        hp, hc= self.simulate_inspiral_mass_independent(eccentricity)
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
            fig_path = f'Images/Empirical_nodes/EIM_{property}_e={eccentricity}_M={self.total_mass}_f_lower={self.f_lower}_f_ref={self.f_ref}_iN={len(self.parameter_space_input)}_gN={len(self.greedy_parameters_idx)}.png'
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            fig.savefig(fig_path)
            print(f'Figure is saved in {fig_path}')

    # def get_training_set_test(self, property, min_greedy_error=None, N_greedy_vecs=None, plot_training_set=True, 
    #                         plot_greedy_error=False, save_fig_greedy_error=False, plot_emp_nodes_at_ecc=False, save_fig_emp_nodes=False, save_fig_training_set=False, 
    #                         save_dataset_to_file=True, plot_residuals_eccentric_evolve=False, plot_residuals_time_evolve=True, save_fig_residuals_eccentric=False, save_fig_residuals_time=False):
    #         """
    #         Generate a training set for the surrogate model by calculating residuals, selecting greedy parameters, and determining empirical nodes.
            
    #         Parameters:
    #         ----------------
    #         - property (str): Waveform property (e.g., 'phase' or 'amplitude') for generating the dataset.
    #         - min_greedy_error (float, optional): Minimum greedy error threshold for stopping criterion in greedy selection.
    #         - N_greedy_vecs (int, optional): Number of greedy vectors to select.
    #         - plot_training_set (bool, optional): If True, plots the training set.
    #         - plot_greedy_error (bool, optional): If True, plots greedy error.
    #         - plot_emp_nodes_at_ecc (float or bool, optional): If True, plots empirical nodes at specified eccentricity.
    #         - save_fig (bool, optional): If True, saves the plot of the training set.
    #         - save_dataset_to_file (bool, optional): If True, saves the generated dataset.

    #         Returns:
    #         ----------------
    #         - residual_training_set (numpy.ndarray): Training set of residuals at empirical nodes.
    #         """
    #         # Step 1: Generate residuals for the full parameter space
    #         residual_parameterspace_input = self.generate_property_dataset(
    #             ecc_list=self.parameter_space_input,
    #             property=property,
    #             save_dataset_to_file=save_dataset_to_file,
    #             plot_residuals_eccentric_evolv=plot_residuals_eccentric_evolve,
    #             plot_residuals_time_evolv=plot_residuals_time_evolve,
    #             save_fig_eccentric_evolv=save_fig_residuals_eccentric,
    #             save_fig_time_evolve=save_fig_residuals_time
    #         )
            
    #         # Step 2: Select the best representative parameters using a greedy algorithm
    #         print('Calculating greedy parameters...')
    #         self.greedy_parameters_idx, self.residual_greedy_basis = self.get_greedy_parameters(
    #             U=residual_parameterspace_input,
    #             min_greedy_error=min_greedy_error,
    #             N_greedy_vecs=N_greedy_vecs,
    #             property=property,
    #             plot_greedy_error=plot_greedy_error,
    #             save_greedy_fig=save_fig_greedy_error
    #         )
            
    #         # Step 3: Calculate empirical nodes of the greedy basis
    #         print('Calculating empirical nodes...')
    #         self.empirical_nodes_idx = self.get_empirical_nodes_test(
    #             reduced_basis=self.residual_greedy_basis,
    #             property=property,
    #             plot_emp_nodes_at_ecc=plot_emp_nodes_at_ecc,
    #             save_fig=save_fig_emp_nodes
    #         )
            
    #         # Step 4: Generate the training set at empirical nodes
    #         residual_training_set = self.residual_greedy_basis[:, self.empirical_nodes_idx]
    #         self.time_training = self.time[self.empirical_nodes_idx]

    #         # Optionally plot the training set
    #         if plot_training_set:
    #             self._plot_training_set(residual_training_set, property, save_fig_training_set)

    #         return residual_training_set
    
    def get_training_set(self, property, min_greedy_error=None, N_greedy_vecs=None, plot_training_set=False, 
                        plot_greedy_error=False, save_fig_greedy_error=False, plot_emp_nodes_at_ecc=False, save_fig_emp_nodes=False, save_fig_training_set=False, 
                        save_dataset_to_file=True, plot_residuals_eccentric_evolve=False, plot_residuals_time_evolve=False, save_fig_residuals_eccentric=False, save_fig_residuals_time=False):
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
        # Step 1: Generate residuals for the full parameter space
        residual_parameterspace_input = self.generate_property_dataset(
            ecc_list=self.parameter_space_input,
            property=property,
            save_dataset_to_file=save_dataset_to_file,
            plot_residuals_eccentric_evolv=plot_residuals_eccentric_evolve,
            plot_residuals_time_evolv=plot_residuals_time_evolve,
            save_fig_eccentric_evolv=save_fig_residuals_eccentric,
            save_fig_time_evolve=save_fig_residuals_time
        )
        
        # Step 2: Select the best representative parameters using a greedy algorithm
        print('Calculating greedy parameters...')
        self.greedy_parameters_idx, self.residual_greedy_basis = self.get_greedy_parameters(
            U=residual_parameterspace_input,
            min_greedy_error=min_greedy_error,
            N_greedy_vecs=N_greedy_vecs,
            property=property,
            plot_greedy_error=plot_greedy_error,
            save_greedy_fig=save_fig_greedy_error
        )
        
        # Step 3: Calculate empirical nodes of the greedy basis
        print('Calculating empirical nodes...')
        self.empirical_nodes_idx = self.get_empirical_nodes(
            reduced_basis=self.residual_greedy_basis,
            property=property,
            plot_emp_nodes_at_ecc=plot_emp_nodes_at_ecc,
            save_fig=save_fig_emp_nodes
        )
        
        # Step 4: Generate the training set at empirical nodes
        residual_training_set = self.residual_greedy_basis[:, self.empirical_nodes_idx]
        self.time_training = self.time[self.empirical_nodes_idx]

        # Optionally plot the training set
        if plot_training_set:
            self._plot_training_set(property, save_fig_training_set)

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

        for i, idx in enumerate(self.greedy_parameters_idx):
            ax.plot(self.time, self.residual_greedy_basis[i], label=f'e={round(self.parameter_space_input[idx], 3)}', linewidth=0.6)
            ax.scatter(self.time[self.empirical_nodes_idx], self.residual_greedy_basis[i][self.empirical_nodes_idx])

        ax.set_xlabel('t [M]')
        ax.set_ylabel('greedy residual')
        ax.legend()
        ax.set_title('Residual Training Set')
        ax.grid(True)

        if save_fig:
            figname = f'Training_set_{property}_M={self.total_mass}_ecc_list=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}]_f_lower={self.f_lower}_f_ref={self.f_ref}_gN={len(self.greedy_parameters_idx)}.png'
            os.makedirs('Images/TrainingSet', exist_ok=True)
            fig.savefig(f'Images/TrainingSet/{figname}')
            print(f'Figure is saved in Images/TrainingSet/{figname}')

# sampling_frequency = 2048 # or 4096
# duration = 4 # seconds
# time_array = np.linspace(-duration, 0, int(sampling_frequency * duration))  # time in seconds

# gt = Generate_TrainingSet(time_array=time_array, total_mass=60, luminosity_distance=200, ecc_ref_parameterspace=np.linspace(0, 0.2, num=100))
# for ecc in np.linspace(0, 0.2, num=20)[10:]:
# #     print(ecc)
#     hp, hc = gt.simulate_inspiral_mass_independent(ecc, plot_polarisations=True, save_fig=True)
#     gt.calculate_residual(hp, hc, ecc, 'phase', plot_residual=True, save_fig=True)
# print(np.linspace(0, 0.2, num=100)[-10:])
# gt._generate_polarisation_data(np.linspace(0.01, 0.5, num=20))
# gt.get_training_set(property='phase', N_greedy_vecs=20, plot_emp_nodes_at_ecc=0.1, save_fig_emp_nodes=True, save_fig_greedy_error=True, plot_residuals_eccentric_evolve=True, plot_residuals_time_evolve=True, save_fig_residuals_eccentric=True, save_fig_residuals_time=True, save_fig_training_set=True)
# gt.get_training_set_test(property='phase', N_greedy_vecs=20, plot_emp_nodes_at_ecc=0.1, save_fig_emp_nodes=True)
# gt.generate_property_dataset(ecc_list=np.linspace(0.01, 0.2, num=5), property='phase', plot_residuals_eccentric_evolv=True, plot_residuals_time_evolv=True)
# gt.generate_property_dataset(ecc_list=np.linspace(0.01, 0.2, num=5), property='amplitude', plot_residuals_eccentric_evolv=True, plot_residuals_time_evolv=True)
# gt.get_training_set(property='phase', N_greedy_vecs=20, plot_emp_nodes_at_ecc=0.1, plot_greedy_error=True, plot_residuals_eccentric_evolve=True, plot_residuals_time_evolve=True, plot_training_set=True)
# gt.get_training_set(property='amplitude', N_greedy_vecs=20, plot_emp_nodes_at_ecc=0.1, plot_greedy_error=True, plot_residuals_eccentric_evolve=True, plot_residuals_time_evolve=True, plot_training_set=True,  save_fig_residuals_eccentric=True, save_fig_residuals_time=True, save_fig_training_set=True)
# plt.show()


