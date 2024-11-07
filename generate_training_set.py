from generate_eccentric_wf import *
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from pycbc.types import TimeSeries

plt.switch_backend('WebAgg')

<<<<<<< HEAD
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
        
=======
class Generate_TrainingSet(Waveform_properties, Simulate_Inspiral):

    def __init__(self, parameter_space_input, waveform_size=None, total_mass=50, mass_ratio=1, freqmin=18):
        self.parameter_space_input = parameter_space_input
        self.waveform_size = waveform_size

>>>>>>> 878d76609a262e58255aee4c9589b4c93410c3b3
        self.residual_greedy_basis = None
        self.greedy_parameters_idx = None
        self.empirical_nodes_idx = None

        self.phase_shift_total_input = np.zeros(len(self.parameter_space_input))

<<<<<<< HEAD
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
            load_residuals = np.load(f'Straindata/Residuals_/residuals_{property}_e=[{min(eccmin_list)}_{max(eccmin_list)}]_N={len(eccmin_list)}.npz_')
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
            self.plot_residuals(residual_dataset, eccmin_list, property, save_fig)
        
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
            residual_dataset[i] = residual[-self.waveform_size:]
        
        # Create phaseshift dataset 
        phase_shift_eccminlist = np.zeros(len(eccmin_list))
        if property == 'phase':

            # File with very largly extended parameterspace to accurately predict the phase shifts
            load_phase_shifts = np.load('Straindata/Phaseshift/estimated_phase_shift.npz')
            loaded_phase_shift = load_phase_shifts['phase_shift']
            loaded_parameter_space = load_phase_shifts['parameter_space']
            
            # Only use eccentricitie for requested surrogate model
            total_phase_shift_cut = loaded_phase_shift[loaded_parameter_space <= max(eccmin_list)]
            old_size = len(total_phase_shift_cut)
            new_size = len(eccmin_list)

            # Generate the old and new indices
            old_indices = np.linspace(0, old_size - 1, old_size)
            new_indices = np.linspace(0, old_size - 1, new_size)

            # Interpolate the values at the new indices
            phase_shift_eccminlist = np.interp(new_indices, old_indices, total_phase_shift_cut)
            
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


    def get_greedy_parameters(self, U, property, min_greedy_error=None, N_greedy_vecs=None, reg=1e-6, 
                            plot_greedy_error=False, plot_validation_errors=False, save_validation_fig=False, 
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
=======
        Simulate_Inspiral.__init__(self, eccmin=None, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin, waveform_size=waveform_size)
        Waveform_properties.__init__(self, eccmin=None, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin, waveform_size=waveform_size)
    
    
    def generate_property_dataset(self, eccmin_list, property, save_dataset_to_file=None, plot_residuals=False, save_fig=False):

        if plot_residuals is True:
            fig_residuals = plt.figure()

        try: 
            load_residuals = np.load(f'Straindata/Residuals/{save_dataset_to_file}')
            residual_dataset = load_residuals['residual']
            self.TS_M = load_residuals['TS_M'][-self.waveform_size:]
            total_phase_shift = load_residuals['total_phase_shift']
            print(f'Residual parameterspace dataset found for {property}')
            
        except:
            try:
                load_polarisations = np.load(f'Straindata/Polarisations/polarizations_{min(eccmin_list)}_{max(eccmin_list)}_{len(eccmin_list)}wfs.npz', allow_pickle=True)
                hp_dataset = load_polarisations['hp']
                hc_dataset = load_polarisations['hc']
                self.TS_M = load_polarisations['TS']
                
                print('Loaded polarisations')
            except:
                print(f'No excisting residuals or polarisations found for {property}')
                
                hp_dataset = []
                hc_dataset = []

                for i, eccentricity in enumerate(eccmin_list):
                    hp, hc, TS_M = self.simulate_inspiral_mass_independent(eccentricity)
                    
                    hp_dataset.append(hp)
                    hc_dataset.append(hc)
                    self.TS_M = TS_M

                hp_dataset = np.array(hp_dataset, dtype=object)
                hc_dataset = np.array(hc_dataset, dtype=object)

                header = f'amount of waveforms: {len(self.parameter_space_input)}'
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Straindata/Polarisations', exist_ok=True)
                np.savez(f'Straindata/Polarisations/polarizations_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_{len(eccmin_list)}wfs.npz', hp=hp_dataset, hc=hc_dataset, TS=self.TS_M)

            residual_dataset = np.zeros((len(eccmin_list), self.waveform_size))

            for i, eccentricity in enumerate(eccmin_list):
                # Start new for-loop to calculate residuals in case polarisations are already saved.
                self.eccmin = eccentricity
                hp, hc = TimeSeries(hp_dataset[i], delta_t=self.DeltaT), TimeSeries(hc_dataset[i], delta_t=self.DeltaT)

                residual = self.calculate_residual(hp, hc, property)
                residual_dataset[i] = residual[-self.waveform_size:]

            total_phase_shift = np.zeros(len(eccmin_list))
            if property == 'phase':
                total_phase_shift = -residual_dataset[:, 0]
                residual_dataset = (residual_dataset.T + total_phase_shift).T


        if plot_residuals is True:
            print(total_phase_shift)
            for i in range(len(residual_dataset)):
                # plt.plot(eccmin_list, residual_dataset.T[i])
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

        if save_dataset_to_file is not None and not os.path.isfile(f'Straindata/Residuals/{save_dataset_to_file}'):

            header = str(eccmin_list)
            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Straindata/Residuals', exist_ok=True)
            np.savez('Straindata/Residuals/' + save_dataset_to_file, residual=residual_dataset, TS_M=self.TS_M[-self.waveform_size:], eccentricities=eccmin_list, total_phase_shift=total_phase_shift)
            print('Residuals saved to Straindata/Residuals/')

        self.TS_M = self.TS_M[-self.waveform_size:]
        return residual_dataset, total_phase_shift
    
    def get_greedy_parameters(self, U, min_greedy_error, property, reg=1e-6, plot_greedy_error=False, plot_validation_errors=False, save_validation_fig=False, save_greedy_fig=False):
        """
        Perform strong greedy algorithm that selects greedy basis based on highest uniqueness. 
        When convergence is reached, other added waveforms no longer contribute to the basis.

        Parameters:
        ----------------
        - U (numpy.ndarray) : Non-normalized training set, each row represents a data point.
        - reg (float, optional) : Regularization parameter to stabilize computation (default is 1e-6).
        
        Returns:
        ----------------
        - greedy_basis (numpy.ndarray) : Selected basis vectors based on highest uniqueness (lowest projection errors).
        - greedy_parameters (list) : Parameters corresponding to the selected greedy basis.
        - greedy_errors (list) : List of projection (greedy) errors which represent the worst error of the best approximation by the basis, each time a vector gets added to the basis.
        """

        def calc_validation_vectors(num_vectors, property):
            print('Calculate validation vectors...')

            parameter_space = np.linspace(min(self.parameter_space_input), max(self.parameter_space_input), num=5000).round(4)
            validation_set = random.sample(list(parameter_space), num_vectors)

            validation_vecs, _ = self.generate_property_dataset(property=property, eccmin_list=validation_set)
            print('Calculated validation vectors')

            return validation_vecs


        def compute_proj_errors(basis, V, reg=1e-6):
            """
            Computes the projection errors when approximating target vectors V 
            using a given basis.

            Parameters:
            ----------------
            - basis (numpy.ndarray): The NORMALIZED basis vectors used for projection.
            - V (numpy.ndarray): The NORMALIZED target vectors to be approximated.
            - reg (float, optional): Regularization parameter to stabilize the computation
            (default is 1e-6).

            Returns:
            ----------------
            - errors (list): List of projection errors for each number of basis vectors
            """
            basis = basis / np.linalg.norm(basis, axis=1, keepdims=True)
            V = V / np.linalg.norm(V, axis=1, keepdims=True)
            
            
            G = np.dot(basis, basis.T) + reg * np.eye(basis.shape[0]) # The gramian matrix of the inner product with itself 
            # In some cases this is a singular matrix and will cause computational problems. To prevent this, I added a small regulation to the diagonal terms of the matrix.
            R = np.dot(basis, V.T)
            errors = []
            
            for N in range(basis.shape[0] + 1):
                if N > 0:
                    v = np.linalg.solve(G[:N, :N], R[:N, :])
                    V_proj = np.dot(v.T, basis[:N, :])
                else:
                    V_proj = np.zeros_like(V)
                errors.append(np.max(np.linalg.norm(V - V_proj, axis=1, ord=2)).round(6))
            
            return errors
        
        # Normalize the dataset U
>>>>>>> 878d76609a262e58255aee4c9589b4c93410c3b3
        U_copy = U.copy()
        U_normalised = U_copy / np.linalg.norm(U_copy, axis=1, keepdims=True)

        normalised_basis = np.empty((0, U.shape[1]))  # Initialize an empty array for the basis
        greedy_basis = np.empty((0, U.shape[1]))  # Initialize an empty array for the basis
        
        greedy_parameters_idx = []
        errors = [1]
        greedy_errors=[1]

<<<<<<< HEAD
        while True:
            # Either break the loop when minimum greedy error is reached or when the specified amount of vectors in the greedy basis is reached.
            if min_greedy_error is not None:
                if np.max(errors) <= min_greedy_error:
                    break
            if N_greedy_vecs is not None:
                if len(greedy_basis) >= N_greedy_vecs:
                    break

=======
        while np.max(errors) >= min_greedy_error:
>>>>>>> 878d76609a262e58255aee4c9589b4c93410c3b3
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

<<<<<<< HEAD
        # Plot greedy errors if requested
        if plot_greedy_error:
            self._plot_greedy_errors(greedy_errors, greedy_parameters_idx, property, save_greedy_fig)

        # Validation error plot
        if plot_validation_errors:
            validation_vecs = calc_validation_vectors(15)
            trivial_basis = U[:len(greedy_basis)]
            self._plot_validation_errors(validation_vecs, greedy_basis, trivial_basis, property, save_validation_fig)

        print('Highest error of best approximation of the basis:', round(np.min(greedy_errors), 5))
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
        plt.title(f'Greedy errors of residual {property} for M={self.total_mass}')
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
=======
        N_basis_vectors = np.linspace(1, len(greedy_errors), num=len(greedy_errors))

        if plot_greedy_error is True:
            """
            Plot greedy errors of residual dataset.
            """
            fig_greedy_error = plt.figure(figsize=(7,5))

            plt.scatter(N_basis_vectors, greedy_errors, s=4)
            plt.plot(N_basis_vectors, greedy_errors)
            
            # Annotate each point with its label
            for i, label in enumerate(self.parameter_space_input[greedy_parameters_idx]):
                plt.annotate(label, (N_basis_vectors[i], greedy_errors[i]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=5.5)
            
            plt.title(f'greedy errors of residual {property} {min(self.parameter_space_input)} - {max(self.parameter_space_input)}' )
            plt.xlabel('Number of waveforms')
            plt.ylabel('greedy error')
            plt.yscale('log')
            plt.grid(True)
           

            if save_greedy_fig is True:
                figname = f'Greedy_error_{property}_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_{len(U)}_wfs.png'

                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Greedy_errors', exist_ok=True)
                fig_greedy_error.savefig('Images/Greedy_errors/' + figname)

        if plot_validation_errors is True:
            """ Compare projection errors of the greedy basis and 
            trivial-basis (unordered-basis) to some randomly (from parameterspace) sampled validation vectors.
            """
            validation_vecs = calc_validation_vectors(num_vectors=15, property=property)
            trivial_basis = U.copy()
 
            greedy_validation_errors = compute_proj_errors(greedy_basis, validation_vecs)
            trivial_validation_errors = compute_proj_errors(trivial_basis[:len(greedy_basis)], validation_vecs)


            fig_validation_error = plt.figure(figsize=(7,5))

            plt.scatter(N_basis_vectors, greedy_validation_errors, label='greedy', s=4)
            plt.plot(N_basis_vectors, greedy_validation_errors)
            plt.scatter(N_basis_vectors, trivial_validation_errors, label='trivial', s=4)
            plt.plot(N_basis_vectors, trivial_validation_errors)
            
            
            # Annotate each point with its label
            for i, label in enumerate(self.parameter_space_input[greedy_parameters_idx]):
                plt.annotate(label, (N_basis_vectors[i], greedy_validation_errors[i]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=5.5)
                plt.annotate(self.parameter_space_input[i], (N_basis_vectors[i], trivial_validation_errors[i]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=5.5)
            
            plt.title(f'greedy error of validation set {property} {min(self.parameter_space_input)} - {max(self.parameter_space_input)}' )
            plt.xlabel('Number of waveforms')
            plt.ylabel('validation error')
            plt.yscale('log')
            plt.legend()
            plt.grid()

            if save_validation_fig is True:
                figname = f'Validation_error_{property}_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_{len(U)}_wfs.png'

                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Greedy_errors', exist_ok=True)
                fig_validation_error.savefig('Images/Greedy_errors/' + figname)

                print('Figure is saved in Images/Greedy_errors')


        print('Highest error of best approximation of the basis: ', np.min(greedy_errors).round(5))

        return greedy_parameters_idx, greedy_basis
>>>>>>> 878d76609a262e58255aee4c9589b4c93410c3b3


    def get_empirical_nodes(self, reduced_basis, property, plot_emp_nodes=False, save_fig=False):
        """
<<<<<<< HEAD
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
=======
        Calculate the empirical nodes for a given dataset. 
        
        Parameters:
        ----------------
        - reduced_basis (numpy.ndarray): Reduced basis of residual properties (phase or amplitude)
        - plot_emp_nodes (float) : For plot of the empirical nodes at a certain eccentric polarisation, 
            set plot_emp_nodes to float eccentric value (plot_emp_nodes=0.2). For no display of plot, set to False.
        - property (str) : Choose waveform property == "phase" OR "amplitude"
        
        Returns:
        ----------------
        - emp_nodes_idx (numpy.ndarray): The empirical nodes for a given dataset.
        """

        def calc_empirical_interpolant(waveform_property, reduced_basis, emp_nodes_idx):
            """
            Calculate the empirical interpolant for a given waveform datapiece.
            
            Parameters:
            ----------------
            - waveform_property (numpy.ndarray): Residual waveform property.
            - reduced_basis (numpy.ndarray): Reduced basis of residual properties (phase or amplitude)
            - emp_nodes_idx (numpy.ndarray) : Indices of empirical nodes
            
            Returns:
            ----------------
            - empirical_interpolant (numpy.ndarray): The empirical interpolant of the waveform property.
            """

            empirical_interpolant = np.zeros_like(waveform_property)

            m = len(emp_nodes_idx)
            B_j_vec = np.zeros((reduced_basis.shape[1], m))  # Ensure complex dtype

            V = np.zeros((m, m))
            for j in range(m):
                for i in range(m):
                    V[j][i] = reduced_basis[i][emp_nodes_idx[j]]

            V_inv = np.linalg.pinv(V)

            for t in range(reduced_basis.shape[1]):
                B_j = 0
                for i in range(m):
                    B_j_vec[t, i] = np.dot(reduced_basis[:, t], V_inv[:, i])

            # Calculate the empirical interpolant
>>>>>>> 878d76609a262e58255aee4c9589b4c93410c3b3
            for j in range(reduced_basis.shape[0]):
                empirical_interpolant += B_j_vec[:, j] * waveform_property[emp_nodes_idx[j]]

            return empirical_interpolant
<<<<<<< HEAD

        # Initialize with the index of the maximum value in the first basis vector
=======
            
    
>>>>>>> 878d76609a262e58255aee4c9589b4c93410c3b3
        i = np.argmax(reduced_basis[0])
        emp_nodes_idx = [i]
        EI_error = []

<<<<<<< HEAD
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
                        plot_greedy_error=False, plot_emp_nodes=False, save_fig=False, save_dataset_to_file=True):
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
            plot_greedy_error=plot_greedy_error
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

=======
        for j in range(1, reduced_basis.shape[0]):
            empirical_interpolant = calc_empirical_interpolant(reduced_basis[j], reduced_basis[:j], emp_nodes_idx)

            r = empirical_interpolant - reduced_basis[j][:, np.newaxis].T
            EI_error.append(np.linalg.norm(r))

            idx = np.argmax(np.abs(r))
            emp_nodes_idx.append(idx) 

        if plot_emp_nodes is not False:
            hp, hc, TS_M = self.simulate_inspiral_mass_independent(plot_emp_nodes)
            
            fig_EIM = plt.figure(figsize=(12, 6))

            plt.plot(TS_M, hp, linewidth=0.2, color='black', label = f'$h_+$: ecc = {plot_emp_nodes}')
            plt.plot(TS_M, hc, linewidth=0.2, linestyle='dashed', color='black', label=f'$h_x$: ecc = {plot_emp_nodes}')
            plt.scatter(self.TS_M[emp_nodes_idx], np.zeros(len(emp_nodes_idx)), color='red', s=8)
            plt.ylabel(f'$h_{22}$')
            plt.xlabel('t [M]')
            plt.legend(loc = 'upper left')  

            if save_fig is True:
            
                figname = f'EIM_{property}_e={plot_emp_nodes}.png'
                fig_EIM.savefig('Images/Empirical_nodes/' + figname)
                print('Figure is saved in Images/Emoirical_nodes')

        return emp_nodes_idx
    

    
    def get_training_set(self, property, min_greedy_error, plot_training_set=False, save_fig=False):
        # Get waveform property residuals for ful parameterspace
        # try:
        #     load_parameterspace_input = np.load(f'Straindata/Residuals/residual_{property}_full_parameterspace_input_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}.npz')
        #     residual_parameterspace_input = load_parameterspace_input['residual']
        #     self.TS_M = load_parameterspace_input['TS_M'][-self.waveform_size:]
        #     self.phase_shift_total_input = load_parameterspace_input['total_phase_shift']
        #     print(f'No residual parameterspace dataset found for {property}')

        residual_parameterspace_input, self.phase_shift_total_input = self.generate_property_dataset(eccmin_list=self.parameter_space_input, property=property, save_dataset_to_file=f'residual_{property}_full_parameterspace_input_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_{len(self.parameter_space_input)}wfs.npz')

        # Get best representative greedy parameters of the full parameterspace
        print('Calculating greedy parameters...')
        self.greedy_parameters_idx, self.residual_greedy_basis = self.get_greedy_parameters(U = residual_parameterspace_input, min_greedy_error=min_greedy_error, property=property)
        # Get empirical nodes of greedy basis
        print('Calculating empirical nodes...')
        self.empirical_nodes_idx = self.get_empirical_nodes(reduced_basis=self.residual_greedy_basis, property=property)

        # Generate training set of greedy basis at empirical nodes
        residual_training_set = self.residual_greedy_basis[:, self.empirical_nodes_idx]
        self.TS_training = self.TS_M[self.empirical_nodes_idx]

        if plot_training_set is True:

            fig_trainingset = plt.figure()
         
            for i in range(len(self.greedy_parameters_idx)):
                plt.plot(self.TS_M, self.residual_greedy_basis[i], label=f'e={self.parameter_space_input[self.greedy_parameters_idx[i]]}', linewidth=0.6)
                plt.scatter(self.TS_M[self.empirical_nodes_idx], self.residual_greedy_basis[i][self.empirical_nodes_idx])
                plt.scatter(self.greedy_parameters_idx, residual_training_set.T[i], s=3)
            plt.legend()
            plt.title('residual training set')
            plt.grid(True)

            if save_fig is True:
                figname = f'Training set {property} M={self.total_mass}, q={self.mass_ratio}, ecc_list=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}].png'
                
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/TrainingSet', exist_ok=True)
                fig_trainingset.savefig('Images/TrainingSet/' + figname)

                print('Figure is saved in Images/TrainingSet')

        return residual_training_set
    


    

    # def normalize_matrix(self, matrix):
    #     """ Normalize each row of a matrix. """
    #     return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    # def compute_projection(self, basis, targets, reg=1e-6):
    #     """
    #     Compute the projection of targets using a given basis and calculate errors.
        
    #     Parameters:
    #     - basis (numpy.ndarray): Basis vectors used for projection (assumed normalized).
    #     - targets (numpy.ndarray): Target vectors to be approximated (assumed normalized).
    #     - reg (float, optional): Regularization parameter (default is 1e-6).
        
    #     Returns:
    #     - projection (numpy.ndarray): The projection of targets onto the basis.
    #     - errors (numpy.ndarray): Errors of the projection for each target vector.
    #     """
    #     if basis.size == 0:
    #         # Return zero projections and the full norm of targets if basis is empty
    #         return np.zeros_like(targets), np.linalg.norm(targets, axis=1)
        
    #     G = np.dot(basis, basis.T) + reg * np.eye(basis.shape[0])  # Gram matrix with regularization
    #     R = np.dot(basis, targets.T)  # Inner product matrix
    #     lambdas = np.linalg.lstsq(G, R, rcond=None)[0]  # Solve the linear system
    #     projection = np.dot(lambdas.T, basis)  # Calculate the projection
    #     errors = np.linalg.norm(targets - projection, axis=1)  # Compute the errors
        
    #     return projection, errors

    # def compute_projection_errors(self, basis, targets, reg=1e-6):
    #     """
    #     Compute the maximum projection errors of the targets using a given basis.
        
    #     Parameters:
    #     - basis (numpy.ndarray): Basis vectors used for projection (assumed normalized).
    #     - targets (numpy.ndarray): Target vectors to be approximated (assumed normalized).
    #     - reg (float, optional): Regularization parameter (default is 1e-6).
        
    #     Returns:
    #     - errors (list): List of maximum projection errors for each basis size.
    #     """
    #     basis, targets = normalize(basis), normalize(targets)
    #     errors = [1]
    #     for N in range(1, basis.shape[0] + 1):
    #         _, projection_errors = self.compute_projection(basis[:N, :], targets, reg=reg)
    #         errors.append(np.max(projection_errors).round(6))
        
    #     return errors

    # def get_greedy_parameters(self, U, property, min_greedy_error=5e-1, reg=1e-6, 
    #                         plot_greedy_error=False, plot_validation_errors=False, 
    #                         save_validation_fig=False, save_greedy_fig=False):
    #     """
    #     Perform strong greedy algorithm to select basis vectors based on highest uniqueness.
        
    #     Parameters:
    #     ----------------
    #     - U (numpy.ndarray): Non-normalized training set, each row represents a data point.
    #     - reg (float, optional): Regularization parameter to stabilize computation (default is 1e-6).
    #     - min_greedy_error (float, optional): Minimum greedy error for stopping criterion (default is 5e-1).
        
    #     Returns:
    #     ----------------
    #     - greedy_basis (numpy.ndarray): Selected basis vectors based on highest uniqueness.
    #     - greedy_parameters (list): Parameters corresponding to the selected greedy basis.
    #     - greedy_errors (list): List of projection (greedy) errors.
    #     """
    #     U_normalised = self.normalize_matrix(U)
    #     normalised_basis, greedy_basis = np.empty((0, U.shape[1])), np.empty((0, U.shape[1]))
    #     greedy_parameters_idx, greedy_errors = [], []
    #     errors = [1]

    #     while np.max(errors) >= min_greedy_error:
    #         _, errors = self.compute_projection(normalised_basis, U_normalised, reg)
    #         max_error_idx = np.argmax(errors)

    #         normalised_basis = np.vstack([normalised_basis, U_normalised[max_error_idx]])
    #         greedy_basis = np.vstack([greedy_basis, U[max_error_idx]])
    #         greedy_parameters_idx.append(max_error_idx)
    #         greedy_errors.append(np.max(errors).round(3))

    #     N_basis_vectors = np.arange(1, len(greedy_errors) + 1)

    #     # Plot greedy errors
    #     if plot_greedy_error:
    #         self.plot_errors(N_basis_vectors, [greedy_errors], ['greedy'], 
    #                     f'Greedy Errors of Residual {property}', 'Greedy Error',
    #                     save_path=f'Images/Greedy_errors/Greedy_error_{property}.png' if save_greedy_fig else None)

    #     # Plot validation errors
    #     if plot_validation_errors:
    #         validation_vecs = self.calc_validation_vectors(num_vectors=15, property=property)
    #         trivial_basis = U.copy()
            
    #         greedy_validation_errors = self.compute_projection_errors(greedy_basis, validation_vecs)
    #         trivial_validation_errors = self.compute_projection_errors(trivial_basis[:len(greedy_basis)], validation_vecs)
    #         print(greedy_errors, greedy_validation_errors, trivial_validation_errors, len(greedy_errors), len(greedy_validation_errors), len(trivial_validation_errors))
    #         self.plot_errors(N_basis_vectors, [trivial_validation_errors, greedy_validation_errors], 
    #                     ['trivial', 'greedy'], 
    #                     f'Validation Error of {property}', 'Validation Error',
    #                     save_path=f'Images/Greedy_errors/Validation_error_{property}.png' if save_validation_fig else None)

    #     print('Highest error of best approximation of the basis: ', np.min(greedy_errors))
    #     return greedy_basis, greedy_parameters_idx, greedy_errors

    # def plot_errors(self, N_basis_vectors, errors, labels, title, ylabel, save_path=None):
    #     """ Utility function to plot and optionally save errors. """
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     for err, label in zip(errors, labels):
    #         ax.plot(N_basis_vectors, err, label=label)
    #         ax.scatter(N_basis_vectors, err, s=4)
    #     ax.set_title(title)
    #     ax.set_xlabel('Number of waveforms')
    #     ax.set_ylabel(ylabel)
    #     ax.set_yscale('log')
    #     ax.legend()
    #     if save_path:
    #         os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #         fig.savefig(save_path)
    #         print(f'Figure saved at {save_path}')
    #     plt.show()

        

# gds = Generate_TrainingSet(parameter_space=np.linspace(0.1, 0.2, num=500).round(4), waveform_size=3500, freqmin=18)
# gds.generate_property_dataset(eccmin_list=np.linspace(0.01, 0.2, num=500).round(4), property='phase', save_dataset_to_file='residual_phase_full_parameterspace_0.01_0.2.npz')
# gds.generate_property_dataset(eccmin_list=np.linspace(0.01, 0.2, num=500).round(4), property='amplitude', save_dataset_to_file='residual_amplitude_full_parameterspace_0.01_0.2.npz')
# gds.generate_property_dataset(eccmin_list=np.linspace(0.01, 0.2, num=10).round(4), property='phase', plot_residuals=True)
# gds.generate_property_dataset(eccmin_list=np.linspace(0.01, 0.1, num=500).round(4), property='amplitude', save_dataset_to_file='residual_amplitude_full_parameterspace_0.01_0.2.npz')
# parameter_space=np.linspace(0.01, 0.2, num=100).round(4)
# load = np.load('Straindata/Residuals/residual_phase_full_parameterspace_0.01_0.2_noshift.npz')
# residuals = load['residual']
# print(residuals.shape)

# fig1 = plt.figure()
# plt.plot(parameter_space, residuals[:, 100])
# plt.plot(parameter_space, residuals[:, 200])
# plt.plot(parameter_space, residuals[:, 300])
# plt.show()

# ecc_list = np.linspace(0.01, 0.01, num=10).round(4)
# gds.generate_property_dataset(eccmin_list=ecc_list, property='phase', plot_residuals=True)
# plt.show()
# sim_fig = plt.figure()
# for ecc in ecc_list:
#     # gds.simulate_inspiral_mass_independent(ecc, plot_polarisations=True)
#     si = Simulate_Inspiral(eccmin=ecc, freqmin=10)
#     si.simulate_inspiral()

# training_set = np.load('Straindata/Residuals/residual_phase_full_parameterspace.npz')['residual']
# gds.get_greedy_parameters(U=training_set, property='phase', min_greedy_error=1e-3, plot_greedy_error=True, save_greedy_fig=True, plot_validation_errors=True, save_validation_fig=True)

# training_set = np.load('Straindata/Residuals/residual_amplitude_full_parameterspace.npz')['residual']
# gds.get_greedy_parameters(U=training_set, property='amplitude', min_greedy_error=1e-3, plot_greedy_error=True, save_greedy_fig=True, plot_validation_errors=True, save_validation_fig=True)
# gds.get_training_set(property='phase', plot_training_set=True, save_fig=True)
# gds.get_training_set(property='amplitude', plot_training_set=True, save_fig=True)
# plt.show()
>>>>>>> 878d76609a262e58255aee4c9589b4c93410c3b3
