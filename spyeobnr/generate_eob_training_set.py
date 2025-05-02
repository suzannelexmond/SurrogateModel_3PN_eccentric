from generate_eccentric_eob import *
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from pycbc.types import TimeSeries

plt.switch_backend('WebAgg')

class Generate_TrainingSet(Waveform_Properties, Simulate_Inspiral):
    """
    Class to generate a training dataset for gravitational waveform simulations.
    Inherits from WaveformProperties and SimulateInspiral to leverage methods for waveform 
    property calculations and waveform generation.

    Parameters:
    ----------
    parameter_space_input : array-like
        Array of parameter values defining the waveform parameter space.
    waveform_size : int, optional
        Size of the waveform (number of indices before merger).
    total_mass : float, default=50
        Total mass of the binary black hole system in solar masses.
    mass_ratio : float, default=1
        Mass ratio of the binary system (0 < q < 1).
    freqmin : float, default=18
        Minimum frequency to start the waveform simulation.
    """

    def __init__(self, parameter_space_input, waveform_size=None, mass_ratio=1, freqmin=650):
        
        self.parameter_space_input = parameter_space_input
        
        self.TS = None
        self.residual_greedy_basis = None
        self.greedy_parameters_idx = None
        self.empirical_nodes_idx = None

        super().__init__(eccmin=None, mass_ratio=mass_ratio, freqmin=freqmin, waveform_size=waveform_size)
    
    def generate_property_dataset(self, eccmin_list, property, save_dataset_to_file=None, plot_residuals_time_evolv=False, plot_residuals_eccentric_evolv=False, save_fig_eccentric_evolv=False, save_fig_time_evolve=False):
        """
        Generates a dataset of waveform residuals based on the specified property for a certain range of eccentricities (eccmin).

        Parameters:
        ----------
        eccmin_list : list of floats
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
        phase_shift_eccminlist : ndarray
            Phase shift for each eccentricity in eccmin_list.
        """

        try: 
            # Attempt to load existing residual dataset
            load_residuals = np.load(f'Straindata/Residuals/residuals_{property}_ecc=[{min(eccmin_list)}_{max(eccmin_list)}_N={len(eccmin_list)}].npz')
            
            residual_dataset = load_residuals['residual']
            self.TS = load_residuals['TS'][-self.waveform_size:]
            
            print(f'Residual parameterspace dataset found for {property}')
            
        except Exception as e:
            print(e)

            # If attempt to load residuals failed, generate polarisations and calculate residuals
            hp_dataset, hc_dataset, self.TS = self._generate_polarisation_data(eccmin_list)
            residual_dataset = self._calculate_residuals(eccmin_list, hp_dataset, hc_dataset, property)
            self.TS = self.TS[-self.waveform_size:]

            # IF save_dataset_to_file is True save the residuals to file in Straindata/Residuals
            if save_dataset_to_file is True and not os.path.isfile(f'Straindata/Residuals/residuals_{property}_ecc=[{min(eccmin_list)}_{max(eccmin_list)}_N={len(eccmin_list)}].npz'):
                self._save_residual_dataset(eccmin_list, property, residual_dataset)
       
        # If plot_residuals is True, plot whole residual dataset
        if (plot_residuals_eccentric_evolv is True) or (plot_residuals_time_evolv is True):
            self._plot_residuals(residual_dataset, eccmin_list, property, plot_residuals_eccentric_evolv, plot_residuals_time_evolv, save_fig_eccentric_evolv, save_fig_time_evolve )
        
        return residual_dataset
    
    def _generate_polarisation_data(self, eccmin_list):
        """
        Helper function to generate polarisation data for a list of eccentricities.

        Parameters:
        ----------
        eccmin_list : list of floats
            List of minimum eccentricities.

        Returns:
        -------
        hp_dataset : ndarray
            Plus polarisation data.
        hc_dataset : ndarray
            Cross polarisation data.
        TS : ndarray
            Time series data.
        """
        try:
            # Attempt to load existing polarisation dataset
            load_polarisations = np.load(f'Straindata/Polarisations/polarisations_e=[{min(eccmin_list)}_{max(eccmin_list)}_N={len(eccmin_list)}]_fmin={self.freqmin}_q={self.mass_ratio}.npz', allow_pickle=True)
            hp_dataset = load_polarisations['hp']
            hc_dataset = load_polarisations['hc']
            self.TS = load_polarisations['TS']

            print('Loaded polarisations')

        except:
            hp_dataset, hc_dataset = [], []
            for eccmin in eccmin_list:
                hp, hc, TS = self.simulate_inspiral_mass_independent(eccmin)
                hp_dataset.append(hp)
                hc_dataset.append(hc)
                self.TS = TS
            

            # Save hp and hc for unequal waveform lengths
            hp_dataset, hc_dataset = np.array(hp_dataset, dtype=object), np.array(hc_dataset, dtype=object)

            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Straindata/Polarisations', exist_ok=True)
            np.savez(f'Straindata/Polarisations/polarisations_e=[{min(eccmin_list)}_{max(eccmin_list)}_N={len(eccmin_list)}]_fmin={self.freqmin}_q={self.mass_ratio}.npz', hp=hp_dataset, hc=hc_dataset, TS=self.TS)

        return hp_dataset, hc_dataset, self.TS

    def _calculate_residuals(self, eccmin_list, hp_dataset, hc_dataset, property):
        """
        Helper function to calculate residuals for a property given polarisation data.

        Parameters:
        ----------
        eccmin_list : list of floats
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
        phase_shift_eccminlist : ndarray
            Phase shift for each eccentricity in eccmin_list.
        """
        
        self.circulair_wf()

        # Choose future waveform length of all waveforms to be either self set waveform length or the shortest waveform in the dataset
        limit_waveform_size = min(len(self.TS), len(hp_dataset[-1])) - 20 # Shortest waveform: hp_dataset[-1]
        if self.waveform_size is None:
            self.waveform_size = limit_waveform_size
        elif (self.waveform_size is not None) and (self.waveform_size > limit_waveform_size):
            self.waveform_size = limit_waveform_size

        # Create residual dataset
        residual_dataset = np.zeros((len(eccmin_list), self.waveform_size))

        for i, eccmin in enumerate(eccmin_list):
            residual = self.calculate_residual(hp_dataset[i], hc_dataset[i], self.TS, property)
            residual_dataset[i] = residual[-self.waveform_size:]

        return residual_dataset
    
    def _plot_residuals(self, residual_dataset, eccmin_list, property, plot_eccentric_evolv=True, plot_time_evolve=True, save_fig_eccentric_evolve=False, save_fig_time_evolve=False):
        """Function to plot and option for saving residuals."""
        print(f'ecc evolve ={plot_eccentric_evolv}, time evolve={plot_time_evolve}')
        if plot_eccentric_evolv is True:
            fig_residuals_ecc = plt.figure()

            for i in range(len(residual_dataset)):
                plt.plot(eccmin_list, residual_dataset.T[i])

            plt.xlabel('eccentricity')
            if property == 'phase':
                plt.ylabel(' $\Delta \phi_{22}$ [radians]')
            elif property == 'amplitude':
                plt.ylabel('$\Delta A_{22}$')
            else:
                print('Choose property = "phase", "amplitude"', property, 1)
                sys.exit(1)

            plt.title(f'Residuals {property}')
            plt.grid(True)
            # plt.legend()

            plt.tight_layout()

            if save_fig_eccentric_evolve is True:
                figname = f'Residuals_eccentric_evolv_{property}_q={self.mass_ratio}_ecc_list=[{min(eccmin_list)}_{max(eccmin_list)}_N={len(self.parameter_space_input)}]_fmin={self.freqmin}.png'
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Residuals', exist_ok=True)
                fig_residuals_ecc.savefig('Images/Residuals/' + figname)

                print('Figure is saved in Images/Residuals')

        if plot_time_evolve is True:
            fig_residuals_t = plt.figure()

            for i in range(len(residual_dataset)):
                plt.plot(self.TS[-self.waveform_size:], residual_dataset[i], label='e$_{min}$' + f' = {eccmin_list[i]}', linewidth=0.6)

            plt.xlabel('t [M]')
            if property == 'phase':
                plt.ylabel(' $\Delta \phi_{22}$ [radians]')
            elif property == 'amplitude':
                plt.ylabel('$\Delta A_{22}$')
            else:
                print('Choose property = "phase", "amplitude"', property, 1)
                sys.exit(1)

            plt.title(f'Residuals {property}')
            plt.grid(True)
            # plt.legend()

            plt.tight_layout()

            if save_fig_time_evolve is True:
                figname = f'Residuals_time_evolv_{property}_q={self.mass_ratio}_ecc_list=[{min(eccmin_list)}_{max(eccmin_list)}_N={len(self.parameter_space_input)}]_fmin={self.freqmin}.png'
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Residuals', exist_ok=True)
                fig_residuals_t.savefig('Images/Residuals/' + figname)

                print('Figure is saved in Images/Residuals')
    
    def _save_residual_dataset(self, eccmin_list, property, residual_dataset):
        """Function to save residual dataset to file."""

        os.makedirs('Straindata/Residuals', exist_ok=True)
        file_path = f'Straindata/Residuals/residuals_{property}_ecc=[{min(eccmin_list)}_{max(eccmin_list)}_N={len(eccmin_list)}].npz'
        np.savez(file_path, residual=residual_dataset, TS=self.TS[-self.waveform_size:], eccentricities=eccmin_list)
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
            validation_vecs = self.generate_property_dataset(property=property, eccmin_list=validation_set)
            return validation_vecs

        # Argument checks and initial setup
        if (min_greedy_error is None) == (N_greedy_vecs is None):
            raise ValueError("Specify either min_greedy_error (float) or N_greedy_vecs (int), not both.")

       # Normalize the dataset U
        U_copy = U.copy()
        U_normalised = U_copy / np.linalg.norm(U_copy, axis=1, keepdims=True)

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

            greedy_parameters_idx.append(max_error_idx)
            greedy_errors.append(np.max(errors))

        # Plot greedy errors if requested
        if plot_greedy_error:
            self._plot_greedy_errors(greedy_errors, greedy_parameters_idx, property, save_greedy_fig)

        # Validation error plot
        if plot_validation_errors:
            validation_vecs = calc_validation_vectors(15)
            trivial_basis = U[:len(greedy_basis)]
            self._plot_validation_errors(validation_vecs, greedy_basis, trivial_basis, property, save_validation_fig)

        print(f'Highest error of best approximation of the basis: {round(np.min(greedy_errors), 5)} | {len(greedy_basis)} basis vectors')
        print(greedy_parameters_idx)
        return greedy_parameters_idx, greedy_basis

    def _plot_greedy_errors(self, greedy_errors, greedy_parameters_idx, property, save_greedy_fig):
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
            plt.savefig(f'Images/Greedy_errors/Greedy_error_{property}_q={self.mass_ratio}_ecc=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}]_fmin={self.freqmin}_iN={len(self.parameter_space_input)}_gerr={min(greedy_errors).round(4)}.png')
            print('Greedy error fig saved to Images/Greedy_errors')

    def _plot_validation_errors(self, validation_vecs, greedy_basis, trivial_basis, property, save_validation_fig):
        """Function to plot and option to save validation errors."""
        
        def compute_proj_errors(basis, V):
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
            plt.savefig(f'Images/Validation_errors/Validation_error_{property}_q={self.mass_ratio}_ecc=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}]_fmin={self.freqmin}_iN={len(self.parameter_space_input)}.png')


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

        hp, hc, TS = self.simulate_inspiral_mass_independent(eccentricity)
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(TS[-self.waveform_size:], hp[-self.waveform_size:], linewidth=0.2, color='black', label=f'$h_+$: ecc = {eccentricity}')
        ax.plot(TS[-self.waveform_size:], hc[-self.waveform_size:], linewidth=0.2, linestyle='dashed', color='black', label=f'$h_\times$: ecc = {eccentricity}')
        ax.scatter(self.TS[emp_nodes_idx], np.zeros(len(emp_nodes_idx)), color='red', s=8)

        ax.set_ylabel(f'$h_{{22}}$')
        ax.set_xlabel('t [M]')
        ax.legend(loc='upper left')

        if save_fig:
            fig_path = f'Images/Empirical_nodes/EIM_{property}_e={eccentricity}_q={self.mass_ratio}_fmin={self.freqmin}_iN={len(self.parameter_space_input)}_gN={len(self.greedy_parameters_idx)}.png'
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            fig.savefig(fig_path)
            print(f'Figure is saved in {fig_path}')

    
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
            eccmin_list=self.parameter_space_input,
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
        self.TS_training = self.TS[self.empirical_nodes_idx]

        # Step 5: Optionally plot the training set
        if plot_training_set:
            self._plot_training_set(residual_training_set, property, save_fig_training_set)

        return residual_training_set

    def _plot_training_set(self, residual_training_set, property, save_fig):
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
            ax.plot(self.TS, self.residual_greedy_basis[i], label=f'e={round(self.parameter_space_input[idx], 2)}', linewidth=0.6)
            ax.scatter(self.TS[self.empirical_nodes_idx], self.residual_greedy_basis[i][self.empirical_nodes_idx])
            ax.scatter(self.greedy_parameters_idx, residual_training_set.T[i], s=3)

        ax.set_xlabel('t [M]')
        ax.set_ylabel('greedy residual')
        ax.legend()
        ax.set_title('Residual Training Set')
        ax.grid(True)

        if save_fig:
            figname = f'Training_set_{property}_q={self.mass_ratio}_ecc_list=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}]_fmin={self.freqmin}_gN={len(self.greedy_parameters_idx)}.png'
            os.makedirs('Images/TrainingSet', exist_ok=True)
            fig.savefig(f'Images/TrainingSet/{figname}')
            print('Figure is saved in Images/TrainingSet')


gt = Generate_TrainingSet(parameter_space_input=np.linspace(0.01, 0.3, num=40), freqmin=650)
# gt.generate_property_dataset(eccmin_list=np.linspace(0.01, 0.2, num=100), property='phase', plot_residuals_eccentric_evolv=True, plot_residuals_time_evolv=True)
# gt.generate_property_dataset(eccmin_list=np.linspace(0.01, 0.2, num=100), property='amplitude', plot_residuals_eccentric_evolv=True, plot_residuals_time_evolv=True)
gt.get_training_set(property='phase', N_greedy_vecs=20, plot_emp_nodes_at_ecc=0.1, plot_greedy_error=True, plot_residuals_eccentric_evolve=True, plot_residuals_time_evolve=True, plot_training_set=True)
gt.get_training_set(property='amplitude', N_greedy_vecs=20, plot_emp_nodes_at_ecc=0.1, plot_greedy_error=True, plot_residuals_eccentric_evolve=True, plot_residuals_time_evolve=True, plot_training_set=True)
plt.show()


