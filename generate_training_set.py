from generate_eccentric_wf import *
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

    def __init__(self, parameter_space_input, waveform_size=None, total_mass=50, mass_ratio=1, freqmin=18):
        
        self.parameter_space_input = parameter_space_input
        
        self.residual_greedy_basis = None
        self.greedy_parameters_idx = None
        self.empirical_nodes_idx = None

        self.phase_shift_total_input = np.zeros(len(self.parameter_space_input))

        super().__init__(eccmin=None, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin, waveform_size=waveform_size)
    
    def generate_property_dataset(self, eccmin_list, property, save_dataset_to_file=None, plot_residuals=False, save_fig=False):
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
            load_residuals = np.load(f'Straindata/Residuals/residuals_{property}_e=[{min(eccmin_list)}_{max(eccmin_list)}]_N={len(eccmin_list)}.npz_')
            residual_dataset = load_residuals['residual']
            self.TS_M = load_residuals['TS_M'][-self.waveform_size:]
            phase_shift_eccminlist = load_residuals['total_phase_shift']

            print(f'Residual parameterspace dataset found for {property}')
            
        except Exception as e:
            print(e)

            # If attempt to load residuals failed, generate polarisations and calculate residuals
            hp_dataset, hc_dataset, self.TS_M = self._generate_polarisation_data(eccmin_list)
            residual_dataset, phase_shift_eccminlist = self._calculate_residuals(eccmin_list, hp_dataset, hc_dataset, property)

            # IF save_dataset_to_file is True save the residuals to file in Straindata/Residuals
            if save_dataset_to_file is True and not os.path.isfile(f'Straindata/Residuals/residuals_{property}_e=[{min(eccmin_list)}_{max(eccmin_list)}]_N={len(eccmin_list)}.npz'):
                self._save_residual_dataset(eccmin_list, property, residual_dataset, phase_shift_eccminlist)
       
        # If plot_residuals is True, plot whole residual dataset
        if plot_residuals is True:
            self._plot_residuals(residual_dataset, eccmin_list, property, save_fig)
        
        self.TS_M = self.TS_M[-self.waveform_size:]
        return residual_dataset, phase_shift_eccminlist
    
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
            load_polarisations = np.load(f'Straindata/Polarisations/polarisations_e=[{min(eccmin_list)}_{max(eccmin_list)}]_N={len(eccmin_list)}.npz', allow_pickle=True)
            hp_dataset = load_polarisations['hp']
            hc_dataset = load_polarisations['hc']
            self.TS_M = load_polarisations['TS']

            print('Loaded polarisations')

        except:
            hp_dataset, hc_dataset = [], []
            for eccmin in eccmin_list:
                hp, hc, TS = self.simulate_inspiral_mass_independent(eccmin)
                hp_dataset.append(hp)
                hc_dataset.append(hc)
                self.TS_M = TS
            
            # Determine waveform size based on t_ref
            # self.waveform_size = len(self.TS_M) - np.where(int(self.TS_M) == self.t_ref)[0]

            # Save hp and hc for unequal lengths
            hp_dataset, hc_dataset = np.array(hp_dataset, dtype=object), np.array(hc_dataset, dtype=object)

            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Straindata/Polarisations', exist_ok=True)
            np.savez(f'Straindata/Polarisations/polarisations_e=[{min(eccmin_list)}_{max(eccmin_list)}]_N={len(eccmin_list)}.npz', hp=hp_dataset, hc=hc_dataset, TS=self.TS_M)

        return hp_dataset, hc_dataset, self.TS_M

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
        # Create residual dataset
        residual_dataset = np.zeros((len(eccmin_list), self.waveform_size))
        
        for i, eccmin in enumerate(eccmin_list):
            # Convert numpy arrays back to Timeseries object for residual calculation
            hp, hc = TimeSeries(hp_dataset[i], delta_t=self.DeltaT), TimeSeries(hc_dataset[i], delta_t=self.DeltaT)
            residual = self.calculate_residual(hp, hc, property)
            try :
                residual_dataset[i] = residual[-self.waveform_size:]
            except:
                print('Waveform size if too short. Waveform is by default set to length of shortest waveform.')
                self.waveform_size = len(hp_dataset[-1]) - 20 # Cut waveform to smallest waveform size + 20 for rough start cutoff
                residual_dataset = residual_dataset[:, :self.waveform_size]
                residual_dataset[i] = residual[-self.waveform_size:]
            

        
        # Create phaseshift dataset 
        phase_shift_eccminlist = np.zeros(len(eccmin_list))
        if property == 'phase':

            # # File with very largly extended parameterspace to accurately predict the phase shifts
            # load_phase_shifts = np.load(f'Straindata/Phaseshift/estimated_phase_shift_{self.freqmin}Hz.npz')
            # loaded_phase_shift = load_phase_shifts['phase_shift']
            # loaded_parameter_space = load_phase_shifts['parameter_space']
                
            # # Only use eccentricities for requested surrogate model
            # total_phase_shift_cut = loaded_phase_shift[loaded_parameter_space <= max(eccmin_list)]
            # old_size = len(total_phase_shift_cut)
            # new_size = len(eccmin_list)

            # # Generate the old and new indices
            # old_indices = np.linspace(0, old_size - 1, old_size)
            # new_indices = np.linspace(0, old_size - 1, new_size)

            # # Interpolate the values at the new indices
            # phase_shift_eccminlist = np.interp(new_indices, old_indices, total_phase_shift_cut)
            phase_shift_eccminlist = residual_dataset[:, 0]
            self._save_phase_shifts(phase_shift_eccminlist)
            # Adjust residual dataset to cancel out the phase sudden shifts
            residual_dataset = (residual_dataset.T + phase_shift_eccminlist).T

        return residual_dataset, phase_shift_eccminlist
    
    def _plot_residuals(self, residual_dataset, eccmin_list, property, save_fig):
        """Function to plot and option for saving residuals."""
        fig_residuals = plt.figure()

        for i in range(len(residual_dataset)):
            plt.plot(self.TS_M[-self.waveform_size:], residual_dataset[i], label='e$_{min}$' + f' = {eccmin_list[i]}', linewidth=0.6)
        
        plt.xlabel('t [M]')
        if property == 'phase':
            plt.ylabel(' $\Delta \phi_{22}$ [radians]')
        elif property == 'amplitude':
            plt.ylabel('$\Delta A_{22}$')
        else:
            print('Choose property = "phase", "amplitude", "frequency"', property, 1)
            sys.exit(1)
        plt.title(f'Residuals {property}')
        plt.grid(True)
        plt.legend(fontsize='small')

        plt.tight_layout()

        if save_fig is True:
            figname = f'Residuals M={self.total_mass}, q={self.mass_ratio}, ecc_list=[{min(eccmin_list)}_{max(eccmin_list)}].png'
            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Images/Residuals', exist_ok=True)
            fig_residuals.savefig('Images/Residuals/' + figname)

            print('Figure is saved in Images/Residuals')
    
    def _save_residual_dataset(self, eccmin_list, property, residual_dataset, phase_shift_eccminlist):
        """Function to save residual dataset to file."""
        os.makedirs('Straindata/Residuals', exist_ok=True)
        file_path = f'Straindata/Residuals/residuals_{property}_e=[{min(eccmin_list)}_{max(eccmin_list)}]_N={len(eccmin_list)}.npz'
        np.savez(file_path, residual=residual_dataset, TS_M=self.TS_M[-self.waveform_size:], eccentricities=eccmin_list, total_phase_shift=phase_shift_eccminlist)
        print('Residuals saved to Straindata/Residuals')

    def _save_phase_shifts(self, phase_shift_eccminlist):
        """Function to save full parameter space phaseshift to file."""
        print(self.freqmin, 'Hz')
        os.makedirs('Straindata/Phaseshift', exist_ok=True)
        file_path = f'Straindata/Phaseshift/estimated_phase_shift_{self.freqmin}Hz'
        np.savez(file_path, total_phase_shift=phase_shift_eccminlist, parameter_space=self.parameter_space_input)
        print('Phaseshift saved to Straindata/Phaseshift')


    def get_greedy_parameters(self, U, property, min_greedy_error=None, N_greedy_vecs=None, reg=1e-6, 
                            plot_greedy_error=True, plot_validation_errors=False, save_validation_fig=False, 
                            save_greedy_fig=False):
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
            validation_vecs, _ = self.generate_property_dataset(property=property, eccmin_list=validation_set)
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
        for i, label in enumerate(self.parameter_space_input[greedy_parameters_idx]):
            plt.annotate(label, (N_basis_vectors[i], greedy_errors[i]), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=5.5)
        plt.xlabel('Number of Waveforms')
        plt.ylabel('Greedy Error')
        plt.yscale('log')
        plt.title('Greedy errors of residual {} for N = {}'.format(property, len(greedy_errors)-1))
        plt.grid(True)

        if save_greedy_fig:
            os.makedirs('Images/Greedy_errors', exist_ok=True)
            plt.savefig(f'Images/Greedy_errors/Greedy_error_{property}_{self.total_mass}.png')

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
            plt.savefig(f'Images/Validation_errors/Validation_error_{property}_{self.total_mass}.png')


    def get_empirical_nodes(self, reduced_basis, property, plot_emp_nodes=False, save_fig=False):
        """
        Calculate the empirical nodes for a given dataset based on a reduced basis of residual properties.

        Parameters:
        ----------------
        - reduced_basis (numpy.ndarray): Reduced basis of residual properties (phase or amplitude).
        - property (str): Waveform property to evaluate, options are "phase" or "amplitude".
        - plot_emp_nodes (float, optional): If set, plots the empirical nodes at a specified eccentricity value.
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

        # Optional: Plot the empirical nodes if plot_emp_nodes is set
        if plot_emp_nodes:
            self._plot_empirical_nodes(emp_nodes_idx, property, plot_emp_nodes, save_fig)

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
        hp, hc, TS_M = self.simulate_inspiral_mass_independent(eccentricity)
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(TS_M, hp, linewidth=0.2, color='black', label=f'$h_+$: ecc = {eccentricity}')
        ax.plot(TS_M, hc, linewidth=0.2, linestyle='dashed', color='black', label=f'$h_x$: ecc = {eccentricity}')
        ax.scatter(self.TS_M[emp_nodes_idx], np.zeros(len(emp_nodes_idx)), color='red', s=8)

        ax.set_ylabel(f'$h_{{22}}$')
        ax.set_xlabel('t [M]')
        ax.legend(loc='upper left')

        if save_fig:
            fig_path = f'Images/Empirical_nodes/EIM_{property}_e={eccentricity}.png'
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            fig.savefig(fig_path)
            print(f'Figure is saved in {fig_path}')

    
    def get_training_set(self, property, min_greedy_error=None, N_greedy_vecs=None, plot_training_set=False, 
                        plot_greedy_error=True, save_greedy_fig=False, plot_emp_nodes=False, save_fig=False, save_dataset_to_file=True):
        """
        Generate a training set for the surrogate model by calculating residuals, selecting greedy parameters, and determining empirical nodes.
        
        Parameters:
        ----------------
        - property (str): Waveform property (e.g., 'phase' or 'amplitude') for generating the dataset.
        - min_greedy_error (float, optional): Minimum greedy error threshold for stopping criterion in greedy selection.
        - N_greedy_vecs (int, optional): Number of greedy vectors to select.
        - plot_training_set (bool, optional): If True, plots the training set.
        - plot_greedy_error (bool, optional): If True, plots greedy error.
        - plot_emp_nodes (bool, optional): If True, plots empirical nodes.
        - save_fig (bool, optional): If True, saves the plot of the training set.
        - save_dataset_to_file (bool, optional): If True, saves the generated dataset.

        Returns:
        ----------------
        - residual_training_set (numpy.ndarray): Training set of residuals at empirical nodes.
        """
        # Step 1: Generate residuals for the full parameter space
        residual_parameterspace_input, self.phase_shift_total_input = self.generate_property_dataset(
            eccmin_list=self.parameter_space_input,
            property=property,
            save_dataset_to_file=save_dataset_to_file
        )
        
        # Step 2: Select the best representative parameters using a greedy algorithm
        print('Calculating greedy parameters...')
        self.greedy_parameters_idx, self.residual_greedy_basis = self.get_greedy_parameters(
            U=residual_parameterspace_input,
            min_greedy_error=min_greedy_error,
            N_greedy_vecs=N_greedy_vecs,
            property=property,
            plot_greedy_error=plot_greedy_error,
            save_greedy_fig=save_greedy_fig
        )
        
        # Step 3: Calculate empirical nodes of the greedy basis
        print('Calculating empirical nodes...')
        self.empirical_nodes_idx = self.get_empirical_nodes(
            reduced_basis=self.residual_greedy_basis,
            property=property,
            plot_emp_nodes=plot_emp_nodes
        )
        
        # Step 4: Generate the training set at empirical nodes
        residual_training_set = self.residual_greedy_basis[:, self.empirical_nodes_idx]
        self.TS_training = self.TS_M[self.empirical_nodes_idx]

        # Step 5: Optionally plot the training set
        if plot_training_set:
            self._plot_training_set(residual_training_set, property, save_fig)

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
            ax.plot(self.TS_M, self.residual_greedy_basis[i], label=f'e={self.parameter_space_input[idx]}', linewidth=0.6)
            ax.scatter(self.TS_M[self.empirical_nodes_idx], self.residual_greedy_basis[i][self.empirical_nodes_idx])
            ax.scatter(self.greedy_parameters_idx, residual_training_set.T[i], s=3)

        ax.legend()
        ax.set_title('Residual Training Set')
        ax.grid(True)

        if save_fig:
            figname = f'Training set {property} M={self.total_mass}, q={self.mass_ratio}, ecc_list=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}].png'
            os.makedirs('Images/TrainingSet', exist_ok=True)
            fig.savefig(f'Images/TrainingSet/{figname}')
            print('Figure is saved in Images/TrainingSet')


gt = Generate_TrainingSet(np.linspace(0.01, 0.4, num=5000), 30000, freqmin=10)
gt.generate_property_dataset(np.linspace(0.01, 0.4, num=5000), 'phase', save_dataset_to_file=True)
gt = Generate_TrainingSet(np.linspace(0.01, 0.4, num=5000), 30000, freqmin=20)
gt.generate_property_dataset(np.linspace(0.01, 0.4, num=5000), 'phase', save_dataset_to_file=True)

