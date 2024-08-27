from Generate_eccentric import *
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

plt.switch_backend('WebAgg')

class Reduced_basis(Dataset):
    def __init__(self, eccmin_list, waveform_size = None, total_mass=50, mass_ratio=1, freqmin=20):
        
        self.res_amp = None
        self.res_phase = None
        self.val_vecs_prop = None
        self.val_vecs_pol = None

        Simulate_Inspiral.__init__(self, eccmin=None, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin)
        Waveform_properties.__init__(self, eccmin=None, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin)
        Dataset.__init__(self, eccmin_list=eccmin_list, waveform_size=waveform_size, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin)

        self.parameter_space = np.linspace(min(self.eccmin_list), max(self.eccmin_list), 500).round(4)

    def import_polarisations(self, save_dataset=False, eccmin_list=None, training_set=False, val_vecs=False):
        try:

            if training_set is True:
                eccmin_list = self.parameter_space

                hp_DS = np.load(f'Straindata/Training_Hp_{min(eccmin_list)}_{max(eccmin_list)}.npz')['hp_dataset']
                hc_DS = np.load(f'Straindata/Training_Hc_{min(eccmin_list)}_{max(eccmin_list)}.npz')['hc_dataset']
                self.TS_M = np.load(f'Straindata/Training_TS_{min(eccmin_list)}_{max(eccmin_list)}.npz')['time'][-self.waveform_size:]
            
            else:
                hp_DS = np.load(f'Straindata/Hp_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')['hp_dataset']
                hc_DS = np.load(f'Straindata/Hc_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')['hc_dataset']
                self.TS_M = np.load(f'Straindata/TS_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')['time'][-self.waveform_size:]
            
            print('Dataset imported.')
        except:
            print('Dataset is not available. Generating new dataset...')
            hp_DS, hc_DS, self.TS_M = self.generate_dataset_polarisations(save_dataset, eccmin_list, val_vecs)


        return hp_DS[:, -self.waveform_size:], hc_DS[:, -self.waveform_size:], self.TS_M
    
    def import_waveform_property(self, property='Phase', save_dataset=False, eccmin_list=None, training_set=False, val_vecs=False):
        
        try:
            if property == 'Amplitude' or property == 'Phase':
                if training_set is True:
                    eccmin_list = self.parameter_space

                    # Always generate new dataset for validation vectors
                    residual_dataset = np.load(f'Straindata/Training_Res_{property}_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')['Residual_dataset']
                    self.TS_M = np.load(f'Straindata/Training_TS_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')['time'][-self.waveform_size:]
                
                else:
                    residual_dataset = np.load(f'Straindata/Res_{property}_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')['Residual_dataset']
                    self.TS_M = np.load(f'Straindata/TS_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')['time'][-self.waveform_size:]
                    
                    print('Dataset {} imported.'.format(property))

                return residual_dataset[:, -self.waveform_size:], self.TS_M 
            
            else:
                print('Choose property = "Phase" or "amplitude"')
                sys.exit(1)
        except: 
            print('Dataset is not available. Generating new dataset...')
            
            if property == 'Amplitude' or property == 'Phase':
                residual_dataset = self.generate_dataset_property(property, save_dataset, eccmin_list, training_set, val_vecs)
            else:
                print('Choose property = "Phase" or "Amplitude"')

        self.TS_M = self.TS_M[-self.waveform_size:]

        return residual_dataset[:, -self.waveform_size:], self.TS_M
    
    def reduced_basis(self, basis):
        num_vectors = basis.shape[0]
        ortho_basis = np.zeros_like(basis)
        
        for i in range(num_vectors):
            vector = basis[i]
            projection = np.zeros_like(vector)
            for j in range(i):
                # vdot for complex vector
                projection += np.dot(basis[i], ortho_basis[j]) * ortho_basis[j]
            ortho_vector = vector - projection
            ortho_basis[i] = ortho_vector / np.linalg.norm(ortho_vector)
        
        return ortho_basis

    def calc_validation_vectors(self, num_vectors, save_dataset=False, property = None, polarisation=None):

        if ((self.val_vecs_prop is None) and (property is not None)) or ((self.val_vecs_pol is None) and (polarisation is not None)):
            print('Calculate validation vectors...')

            parameter_space = np.linspace(min(self.eccmin_list), max(self.eccmin_list), num=1000).round(4)
            validation_set = random.sample(list(parameter_space), num_vectors)

            if (property is not None) and (polarisation is None):
                validation_vecs = self.generate_dataset_property(property=property, save_dataset=False, eccmin_list=validation_set, val_vecs=True)

            elif (property is None) and (polarisation is not None):
                hp, hc, TS_M = self.generate_dataset_polarisations(save_dataset=False, eccmin_list=validation_set)
                if polarisation == 'plus':
                    validation_vecs = hp
                else:
                    validation_vecs = hc
            else:

                print('Choose either polarisation = "plus" or "cross" for polarisation comparison OR property = "Phase" or "Amplitude" for property comparison.')
                sys.exit(1)

            print('Calculated validation vectors')

            return validation_vecs
        
        else:
            if property is not None:
                return self.val_vecs_prop
            else:
                return self.val_vecs_pol

    def compute_proj_errors(self, basis, V, reg=1e-6):
        """
        Computes the projection errors when approximating target vectors V 
        using a given basis.

        Parameters:
        - basis (numpy.ndarray): The NORMALIZED basis vectors used for projection.
        - V (numpy.ndarray): The NORMALIZED target vectors to be approximated.
        - reg (float, optional): Regularization parameter to stabilize the computation
        (default is 1e-6).

        Returns:
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


    def strong_greedy(self, U, parameter_list, reg=1e-6, min_greedy_error=5e-1):
        """
        Perform strong greedy selection to arrange the training set from least similar to most similar. 

        Parameters:
        - U (numpy.ndarray): Non-normalized training set, each row represents a data point.
        - parameter_list (list): Parameters corresponding to each data point in U.
        - reg (float, optional): Regularization parameter to stabilize computation (default is 1e-6).
        - min_greedy_error (float, optional): Minimum greedy error for stopping criterion (default is 5e-1).

        Returns:
        - ordered_basis (numpy.ndarray): Selected basis vectors arranged from least to most similar to the training set.
        - parameters (list): Parameters corresponding to the selected basis vectors.
        """
        # Normalize the dataset U
        U_copy = U.copy()
        U_normalised = U_copy / np.linalg.norm(U_copy, axis=1, keepdims=True)

        ordered_basis = np.empty((0, U.shape[1]))  # Initialize an empty array for the basis
        greedy_basis = np.empty((0, U.shape[1]))  # Initialize an empty array for the basis
        parameters = []  # Initialize an empty array for the parameters of reduced basis
        parameters_idx = []
        errors = [1]

        while np.max(errors) >= min_greedy_error:
            # Compute projection errors using normalized U
            G = np.dot(ordered_basis, ordered_basis.T) + reg * np.eye(ordered_basis.shape[0]) if ordered_basis.size > 0 else np.zeros((0, 0))  # Compute Gramian
            R = np.dot(ordered_basis, U_normalised.T)  # Compute inner product
            lambdas = np.linalg.lstsq(G, R, rcond=None)[0] if ordered_basis.size > 0 else np.zeros((0, U_normalised.shape[0]))  # Use pseudoinverse
            U_proj = np.dot(lambdas.T, ordered_basis) if ordered_basis.size > 0 else np.zeros_like(U_normalised)  # Compute projection
            errors = np.linalg.norm(U_normalised - U_proj, axis=1)  # Calculate errors

            # Extend basis with non-normalized U
            max_error_idx = np.argmax(errors)
            ordered_basis = np.vstack([ordered_basis, U_normalised[max_error_idx]])
            greedy_basis = np.vstack([greedy_basis, U[max_error_idx]])

            parameters.append(parameter_list[max_error_idx])
            parameters_idx.append(max_error_idx)
            print('error strong greedy', np.max(errors.round(6)))

        return greedy_basis, parameters, parameters_idx




    def plot_greedy_error(self, min_greedy_error, training_set, property=None, polarisation=None, plot_greedy_error=True):

        validation_vecs = self.calc_validation_vectors(num_vectors=10, save_dataset=False, property=property, polarisation=polarisation)
        print(training_set.shape, validation_vecs.shape)

        trivial_basis = training_set.copy()
        greedy_basis, parameters_gb, parameters_gb_idx = self.strong_greedy(U=training_set, parameter_list=self.parameter_space, min_greedy_error=min_greedy_error)
        # print(f'greedy_basis {property}', greedy_basis[0])

        greedy_errors = self.compute_proj_errors(greedy_basis, validation_vecs)
        print('plot errors', greedy_errors)
        trivial_errors = self.compute_proj_errors(trivial_basis, validation_vecs)

        if plot_greedy_error is True:

            fig_greedy_error = plt.figure(figsize=(8,6))

            N_basis_vectors = np.linspace(1, len(greedy_errors), num=len(greedy_errors))

            plt.scatter(N_basis_vectors, trivial_errors[:len(N_basis_vectors)], label='trivial', s=4)
            plt.plot(N_basis_vectors, trivial_errors[:len(N_basis_vectors)])
            plt.scatter(N_basis_vectors, greedy_errors, label='greedy', s=4)

            # Annotate each point with its label
            for i, label in enumerate(parameters_gb):
                plt.annotate(label, (N_basis_vectors[i], greedy_errors[i]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=5.5)
                plt.annotate(self.parameter_space[i], (N_basis_vectors[i], trivial_errors[i]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=5.5)
                
            plt.plot(N_basis_vectors, greedy_errors)

            if polarisation is not None:
                plt.title(f'greedy error {polarisation} {min(self.eccmin_list)} - {max(self.eccmin_list)}' )
            else:
                plt.title(f'greedy error {property} {min(self.eccmin_list)} - {max(self.eccmin_list)}' )
            
            plt.xlabel('Number of waveforms')
            plt.ylabel('error')
            plt.yscale('log')
            plt.legend()
            plt.show()
            
            if (polarisation is not None) and (property is None):
                figname = f'Greedy_error_{polarisation}_{min(self.eccmin_list)}_{max(self.eccmin_list)}_{len(training_set)}_wfs.png'
            else:
                figname = f'Greedy_error_{property}_{min(self.eccmin_list)}_{max(self.eccmin_list)}_{len(training_set)}_wfs.png'


        print('Highest error of best approximation of the basis: ', np.min(greedy_errors))

        return greedy_basis, greedy_errors, parameters_gb, parameters_gb_idx

    def greedy_error_efficiency(self, property=None, polarisation=None, save_dataset=False):
        """
        Compare the greedy error for a dataset of 20 ordered (after greedy algorithm) waveforms 
        with picking the first 20 waveforms of an ordered dataset of size 500.

        """
        if property is not None and polarisation is None:
            set_500_wfs = self.import_waveform_property(property=property, save_dataset=save_dataset, training_set=True)[0]
            set_20_wfs = self.import_waveform_property(property=property, save_dataset=save_dataset)[0]

        elif polarisation is not None and property is None:
            hp_500, hc_500, TS_M = self.import_polarisations(save_dataset, self.parameter_space, training_set=True)
            hp_20, hc_20, TS_M = self.import_polarisations(save_dataset, self.eccmin_list)

            if polarisation == 'plus':
                set_500_wfs = hp_500
                set_20_wfs = hp_20
            else:
                set_500_wfs = hc_500
                set_20_wfs = hc_20 

        else:
            print('Choose either polarisation = "plus" or "cross" for polarisation comparison OR property = "Phase" or "Amplitude" for property comparison.')

        greedy_basis_500, greedy_errors_500, parameters_500, parameters_500_idx = self.plot_greedy_error(training_set=set_500_wfs, property=property, polarisation=polarisation)
        greedy_basis_20, greedy_errors_20, parameters_20, parameters_20_idx = self.plot_greedy_error(training_set=set_20_wfs, property=property, polarisation=polarisation)

        N_basis_vectors = np.linspace(1, len(greedy_errors_20), num=len(greedy_errors_20))

        fig_greedy_error_efficiency = plt.figure(figsize=(8,6))

        # Annotate each point with its label
        for i, label in enumerate(parameters_20):
            plt.annotate(label, (N_basis_vectors[i], greedy_errors_20[i]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=5.5)
            plt.annotate(parameters_500[i], (N_basis_vectors[i], greedy_errors_500[i]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=5.5)
        
        plt.scatter(N_basis_vectors, greedy_errors_20, label='20 original waveforms', s=4)
        plt.plot(N_basis_vectors, greedy_errors_20)

        plt.scatter(N_basis_vectors, greedy_errors_500[:len(N_basis_vectors)], label='500 original waveforms', s=4)
        plt.plot(N_basis_vectors, greedy_errors_500[:len(N_basis_vectors)])

        if polarisation is not None:
            plt.title(f'greedy error {polarisation} {min(self.eccmin_list)} - {max(self.eccmin_list)}' )
        else:
            plt.title(f'greedy error {property} {min(self.eccmin_list)} - {max(self.eccmin_list)}' )
        
        plt.xlabel('Number of waveforms')
        plt.ylabel('error')
        plt.yscale('log')
        plt.legend()
        plt.show()

        if polarisation is not None:
            figname = f'Greedy_error_efficiency_{polarisation}_{min(self.eccmin_list)}_{max(self.eccmin_list)}.png'
        if property is not None:
            figname = f'Greedy_error_efficiency_{property}_{min(self.eccmin_list)}_{max(self.eccmin_list)}.png'
        fig_greedy_error_efficiency.savefig('Images/Greedy_error/' + figname)

        print('Highest error of best approximation of the basis for 20 waveform set: ', np.min(greedy_errors_20))
        print('Highest error of best approximation of the basis for first 20 of 500 waveform set: ', np.min(greedy_errors_500))



    def plot_reduced_basis(self, reduced_basis, dataset):

        fig_reduced_basis, axs = plt.subplots(2, figsize=(10, 6))
        plt.subplots_adjust(hspace=0.4)
                                                

        for i in range(len(reduced_basis)):
            axs[0].plot(self.TS_M, reduced_basis[i], label = 'e = {}'.format(self.eccmin_list[i]), linewidth=0.6)
            axs[0].set_xlabel('t [M]')
            # axs[0].set_ylabel('$\Delta\phi_{22}$ [radians]')
            axs[0].set_ylabel('$\Delta A_{22}$')
            axs[0].grid()
            # axs[0].set_xlim(-7.6e6, -7.4e6)
            
            legend = axs[0].legend(loc='lower left', ncol=7)
            
            for text in legend.get_texts():
                text.set_fontsize(8)
            
            axs[1].plot(self.TS_M, dataset[i], label = 'e = {}'.format(self.eccmin_list[i]), linewidth=0.6)
            axs[1].set_ylabel('$\Delta A_{22}$')
            
            # axs[1].set_xlim(-50000, 0)
            axs[1].set_xlabel('t [M]')
            axs[1].grid()
            
            # Get the legend of the first subplot
            legend = axs[1].legend(loc='lower left', ncol=7)

            for text in legend.get_texts():
                text.set_fontsize(8)

        plt.show()

    def plot_dataset_polarisations(self, save_dataset=False):
        hp_DS, hc_DS, TS = self.import_polarisations(save_dataset)

        fig_dataset_hphc, axs = plt.subplots(2, figsize=(12, 8))
        plt.subplots_adjust(hspace=0.4)
                                                
        for i in range(16, 20):
            axs[0].plot(TS, hp_DS[i], label = 'e = {}'.format(self.eccmin_list[i]), linewidth=0.6)
            axs[0].set_xlabel('t [M]')
            axs[0].set_ylabel('$h_+$')
            axs[0].grid()
            # axs[0].set_xlim(-20000, 0)
            # axs[0].set_ylim(0, 0.00075)
            axs[0].legend(loc = 'lower left', fontsize=8)
            
            axs[1].plot(TS, hc_DS[i], label = 'e = {}'.format(self.eccmin_list[i]), linewidth=0.6)
            axs[1].set_ylabel('$h_x$')
            # axs[1].set_xlim(-20000, 0)
            # axs[1].set_ylim(-0.025, 0.025)
            axs[1].set_xlabel('t [M]')
            axs[1].grid()
            axs[1].legend(loc = 'lower left', fontsize=8)
            
            # figname = 'hp_hc_e={}_{}'.format(self.eccmin_list.min, self.eccmin_list.max)
            # fig_dataset_hphc.savefig('Images/' + figname)

        plt.show()

    def plot_dataset_properties(self, property='Phase', save_dataset=False):
        if property == 'Frequency':
            units = ' [Hz]'
            quantity = '$\Delta$f = f$_{22}$ - f$_{circ}$'
        elif property == 'Amplitude':
            units = ''
            quantity = '$\Delta$A = A$_{22}$ - A$_{circ}$'
        elif property == 'Phase':
            units = ' [radians]'
            quantity = '$\Delta\phi$ = $\phi_{circ}$ - $\phi_{22}$'
        

        residual_dataset, TS_M = self.import_waveform_property(property, save_dataset)

        fig_dataset_property = plt.figure(figsize=(12, 8))

        for i in range(len(residual_dataset)):
            plt.plot(TS_M, residual_dataset[i], label = 'e = {}'.format(self.eccmin_list[i]), linewidth=0.6)
            plt.xlabel('t [M]')
            plt.ylabel(quantity + units)
            plt.legend()
            plt.grid(True)

        figname = f'{property}_{min(self.eccmin_list)}_{max(self.eccmin_list)}.png'
        fig_dataset_property.savefig('Images/Dataset_properties/' + figname)
        print('fig is saved')
        # plt.show()






class Empirical_Interpolation_Method(Reduced_basis, Dataset):

    def __init__(self, eccmin_list, waveform_size = None, total_mass=50, mass_ratio=1, freqmin=5):
        Simulate_Inspiral.__init__(self, eccmin=None, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin)
        Waveform_properties.__init__(self, eccmin=None, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin)
        Dataset.__init__(self, eccmin_list, waveform_size, total_mass, mass_ratio, freqmin)
        Reduced_basis.__init__(self, eccmin_list, waveform_size=waveform_size, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin)
        # super().__init__(eccmin_list, waveform_size, total_mass, mass_ratio, freqmin)


    def calc_empirical_interpolant(self, waveform, reduced_basis, emp_nodes_idx):
        """
        Calculate the empirical interpolant for a given waveform and reduced basis.
        
        Parameters:
        - waveform: The complex waveform to interpolate.
        - reduced_basis: The complex reduced basis.
        - T: Indices of the empirical nodes.
        
        Returns:
        - empirical_interpolant: The empirical interpolant of the waveform.
        """
        
        # # Initialize the matrix for the interpolation coefficients
        # B_j_vec = np.zeros((reduced_basis.shape[1], reduced_basis.shape[0]), dtype=complex)
        empirical_interpolant = np.zeros_like(waveform, dtype=complex)

        # # Construct the Vandermonde matrix V
        # V = np.zeros((len(reduced_basis), len(reduced_basis)), dtype=complex)
        # for j in range(len(reduced_basis)):
        #     for i in range(len(T)):
        #         V[j][i] = reduced_basis[i][T[j]]

        # # Compute the interpolation coefficients
        # V_inv = np.linalg.inv(V)
        # for j in range(V.shape[1]): 
        #     B_j = np.zeros_like(reduced_basis[0], dtype=complex)
        #     for i in range(len(reduced_basis)):
        #         B_j += reduced_basis[i].T * V_inv[i][j]
        #     B_j_vec[:, j] = B_j

        m = len(emp_nodes_idx)
        B_j_vec = np.zeros((reduced_basis.shape[1], m), dtype=complex)  # Ensure complex dtype

        V = np.zeros((m, m), dtype=complex)
        for j in range(m):
            for i in range(m):
                V[j][i] = reduced_basis[i][emp_nodes_idx[j]]

        V_inv = np.linalg.inv(V)

        for t in range(reduced_basis.shape[1]):
            B_j = 0
            for i in range(m):
                # B_j += reduced_basis[i].conj().T * V_inv[i][j]  # Use conjugate transpose for complex numbers
                B_j_vec[t, i] = np.dot(reduced_basis[:, t], V_inv[:, i])
            # B_j_vec[:, j] = B_j

        # Calculate the empirical interpolant
        for j in range(reduced_basis.shape[0]):
            empirical_interpolant += B_j_vec[:, j] * waveform[emp_nodes_idx[j]]

        return empirical_interpolant

    # def calc_empirical_interpolant(self, waveform, reduced_basis, T):

    #     B_j_vec = np.zeros((reduced_basis.shape[1], reduced_basis.shape[0]))
    #     empirical_interpolant = 0

    #     V = np.zeros((len(reduced_basis), len(reduced_basis)))
    #     for j in range(len(reduced_basis)):
    #         for i in range(len(T)):
    #             V[j][i] = reduced_basis[i][T[j]]

    #     for j in range(V.shape[1]): 
    #         B_j = 0
    #         for i in range(len(reduced_basis)):
    #             B_j += reduced_basis[i].T * np.linalg.inv(V)[i][j]

    #         B_j_vec[:, j] = B_j
        
    #     for j in range(reduced_basis.shape[0]):
    #         empirical_interpolant += B_j_vec[:, j]*waveform[T[j]]

    #     return empirical_interpolant
    

    def calc_empirical_nodes(self, reduced_basis, time_array):
        
        i = np.argmax(reduced_basis[0])
        emp_nodes = [time_array[i]]
        emp_nodes_idx = [i]
        EI_error = []

        for j in range(1, reduced_basis.shape[0]):
            empirical_interpolant = self.calc_empirical_interpolant(reduced_basis[j], reduced_basis[:j], emp_nodes_idx,)
            
            # EI_error.append(np.linalg.norm(reduced_basis[j] - empirical_interpolant))
            
            r = empirical_interpolant - reduced_basis[j][:, np.newaxis].T
            EI_error.append(np.linalg.norm(r))
            idx = np.argmax(np.abs(r))
            emp_nodes.append(time_array[idx]) 
            emp_nodes_idx.append(idx) 

        return emp_nodes_idx

    def plot_empirical_nodes(self, min_greedy_error, property='Phase', save_dataset = False, plot_greedy_error=False):
         
        try:
            loaded_residual = np.load(f'Straindata/Greedy_Res_{property}_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')

            loaded_h = np.load(f'Straindata/Greedy_h_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')
            greedy_parameters = loaded_residual['eccentricity']

            training_set = loaded_residual['residual']
            h = loaded_h['greedy_basis']

            self.TS_M = np.load(f'Straindata/Training_TS_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')['time'][-self.waveform_size:]
            
            print('Loaded greedy dataset')

        except:
            print('Generate greedy dataset')

            training_set, self.TS_M = self.import_waveform_property(property=property, save_dataset=save_dataset, training_set=True)
            greedy_basis, greedy_errors, greedy_parameters, greedy_parameters_idx = self.plot_greedy_error(min_greedy_error=min_greedy_error, training_set=training_set, property=property, plot_greedy_error=plot_greedy_error)

            np.savez(f'Straindata/Greedy_Res_{property}_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz', eccentricity=greedy_parameters, residual=greedy_basis)


        def compute_waveform(h_plus, h_cross):
            return h_plus + 1j * h_cross
        
        hp, hc, self.TS_M = self.generate_dataset_polarisations(save_dataset=False, eccmin_list=greedy_parameters)
        # Complex waveform
        h = compute_waveform(hp, hc)

        reduced_basis = self.reduced_basis(h)
        np.savez(f'Straindata/Greedy_h_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz', eccentricity=greedy_parameters, greedy_basis=reduced_basis)

        print('greedy parameters: ', greedy_parameters)
        emp_nodes_idx = self.calc_empirical_nodes(greedy_basis, self.TS_M)

        nodes_time = []
        nodes_polarisation = []

        for node in emp_nodes_idx:
            nodes_time.append(self.TS_M[node])
            nodes_polarisation.append(0)

        fig_EIM = plt.figure(figsize=(12, 6))

        plt.plot(self.TS_M, np.real(h[0]), linewidth=0.2, label = 'e = {}'.format(greedy_parameters[0]))
        plt.scatter(nodes_time, nodes_polarisation, color='black', s=8)
        plt.ylabel(f'$h_+$')
        # axs[1].set_ylim(-0.5e-23, 0.5e-23)
        # axs[1].set_xlim(-0.4e6, 1000)
        plt.xlabel('t [M]')
        plt.legend(loc = 'upper left')  

        figname = f'EIM_{property}_e={greedy_parameters[0]}.png'
        fig_EIM.savefig('Images/Empirical_nodes/' + figname)
        print('fig is saved')

        plt.show()

