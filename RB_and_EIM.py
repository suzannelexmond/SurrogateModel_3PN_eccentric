from Generate_eccentric import *
import random

plt.switch_backend('WebAgg')

class Reduced_basis(Dataset):
    def __init__(self, eccmin_list, waveform_size = None, total_mass=10, mass_ratio=1, freqmin=5):
        
        self.res_amp = None
        self.res_phase = None
        self.val_vecs = None
        
        Simulate_Inspiral.__init__(self, eccmin=None, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin)
        Waveform_properties.__init__(self, eccmin=None, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin)
        Dataset.__init__(self, eccmin_list=eccmin_list, waveform_size=waveform_size, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin)

    def import_polarisations(self, save_dataset=False, eccmin_list=None):
        try:
            hp_DS = np.loadtxt(f'Straindata/Hp_{min(self.eccmin_list)}_{max(self.eccmin_list)}.txt', skiprows=1)
            hc_DS = np.loadtxt(f'Straindata/Hc_{min(self.eccmin_list)}_{max(self.eccmin_list)}.txt', skiprows=1)
            self.TS_M = np.loadtxt(f'Straindata/TS_{min(self.eccmin_list)}_{max(self.eccmin_list)}.txt', skiprows=1)
        
            print('Dataset imported.')
        except:
            print('Dataset is not available. Generating new dataset...')
            hp_DS, hc_DS, self.TS_M = self.generate_dataset_polarisations(save_dataset, eccmin_list)
        
        # hp_DS = hp_DS[:, -self.waveform_size:]
        # hc_DS = hc_DS[:, -self.waveform_size:]
        # self.TS_M = self.TS_M[-self.waveform_size:]

        return hp_DS, hc_DS, self.TS_M
    
    def import_waveform_property(self, property='Phase', save_dataset=False, eccmin_list=None):
        try:
            if property == 'Amplitude' or property == 'Phase':
                residual_dataset = np.loadtxt(f'Straindata/Res_{property}_{min(self.eccmin_list)}_{max(self.eccmin_list)}.txt', skiprows=1)
                self.TS_M = np.loadtxt(f'Straindata/TS_{min(self.eccmin_list)}_{max(self.eccmin_list)}.txt', skiprows=1)
                
                print('Dataset {} imported.'.format(property))
        except: 
            print('Dataset is not available. Generating new dataset...')
            
            if property == 'Amplitude' or property == 'Phase':
                residual_dataset = self.generate_dataset_property(property, save_dataset, eccmin_list)

        # length_diff = residual_dataset.shape[1] - self.waveform_size
        # residual_dataset = residual_dataset[:, length_diff:]
        # self.TS_M = self.TS_M[length_diff:]

        return residual_dataset, self.TS_M
    
    def reduced_basis(self, basis):
        num_vectors = basis.shape[0]
        ortho_basis = np.zeros_like(basis)
        
        for i in range(num_vectors):
            vector = basis[i]
            projection = np.zeros_like(vector)
            for j in range(i):
                projection += np.dot(basis[i], ortho_basis[j]) * ortho_basis[j]
            ortho_vector = vector - projection
            ortho_basis[i] = ortho_vector / np.linalg.norm(ortho_vector)
        
        return ortho_basis

    def calc_validation_vectors(self, num_vectors, save_dataset, property = 'Phase'):
        
        if (self.val_vecs is None) or ((self.val_vecs is not None) and (len(self.val_vecs) != num_vectors)):
            print('Calculate validation vectors...')

            parameter_space = np.linspace(min(self.eccmin_list), max(self.eccmin_list), num=100)
            validation_set = random.sample(list(parameter_space), num_vectors)

            validation_vecs = self.generate_dataset_property(property=property, save_dataset=False, eccmin_list=validation_set, val_vecs=True)
            
            self.val_vecs = validation_vecs
            
            print('Calculated validation vectors')

            if save_dataset == True:
                header = str(validation_set)
                np.savetxt(f'Straindata/Valvecs_Res_{property}_{min(self.eccmin_list)}_{max(self.eccmin_list)}.txt', validation_vecs, header=header)
                print('Validation vectors saved')

            return validation_vecs
        
        else:
            return self.val_vecs
        
    def compute_proj_errors(self, basis, V, reg=1e-6):
        """
        Computes the projection errors when approximating target vectors V 
        using a given basis.

        Parameters:
        - basis (numpy.ndarray): The basis vectors used for projection.
        - V (numpy.ndarray): The target vectors to be approximated.
        - reg (float, optional): Regularization parameter to stabilize the computation
        (default is 1e-6).

        Returns:
        - errors (list): List of projection errors for each number of basis vectors
        """

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
            errors.append(np.max(np.linalg.norm(V - V_proj, axis=1, ord=2)))
        return errors

    def strong_greedy(self, U, N, reg=1e-6):
        """
        Perform strong greedy selection to arrange the training set from least similar to most similar.

        Parameters:
        - U (numpy.ndarray): Training set, each row represents a data point.
        - N (int): Number of basis vectors to select.
        - param_space (numpy.ndarray): Parameters corresponding to each data point in U.
        - reg (float, optional): Regularization parameter to stabilize computation (default is 1e-6).

        Returns:
        - basis (numpy.ndarray): Selected basis vectors arranged from least to most similar to the training set.
        - parameters (list): Parameters corresponding to the selected basis vectors.
        """
        U = U / np.linalg.norm(U, axis=1, keepdims=True)

        ordered_basis = np.empty((0, U.shape[1]))  # Initialize an empty array for the basis
        parameters = []  # Initialize an empty array for the parameters of reduced basis

        for n in range(N):
            # Compute projection errors. # The gramian matrix of the inner product with itself 
        # In some cases this is a singular matrix and will cause computational problems. To prevent this, I added a small regulation to the diagonal terms of the matrix.
            G = np.dot(ordered_basis, ordered_basis.T) + reg * np.eye(ordered_basis.shape[0]) if ordered_basis.size > 0 else np.zeros((0, 0))  # Compute Gramian
            R = np.dot(ordered_basis, U.T)  # Compute inner product
            lambdas = np.linalg.lstsq(G, R, rcond=None)[0] if ordered_basis.size > 0 else np.zeros((0, U.shape[0]))  # Use pseudoinverse
            U_proj = np.dot(lambdas.T, ordered_basis) if ordered_basis.size > 0 else np.zeros_like(U)  # Compute projection
            errors = np.linalg.norm(U - U_proj, axis=1)  # Calculate errors
        
            # Extend basis
            ordered_basis = np.vstack([ordered_basis, U[np.argmax(errors)]])
            parameters.append(self.eccmin_list[np.argmax(errors)])
        
        return ordered_basis, parameters

    def plot_greedy_error(self, num_validation_vecs, property='Phase', save_dataset='False'):

        training_set, TS_M = self.import_waveform_property(property=property, save_dataset=True)
        validation_vecs = self.calc_validation_vectors(num_validation_vecs, property)

        greedy_basis, parameters_gb = self.strong_greedy(training_set, len(training_set))

        greedy_errors = self.compute_proj_errors(greedy_basis, validation_vecs)
        trivial_errors = self.compute_proj_errors(training_set, validation_vecs)
        print(greedy_errors, trivial_errors)
        fig_greedy_error = plt.figure(figsize=(8,6))


        N_basis_vectors = np.linspace(0, len(trivial_errors), num=(len(trivial_errors)+1))

        plt.scatter(N_basis_vectors, trivial_errors, label='trivial', s=4)
        plt.plot(N_basis_vectors, trivial_errors)
        plt.scatter(N_basis_vectors, greedy_errors, label='greedy', s=4)

        # Annotate each point with its label
        for i, label in enumerate(parameters_gb):
            plt.annotate(label, (N_basis_vectors[i], greedy_errors[i]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=5.5)

            
        plt.plot(N_basis_vectors, greedy_errors)
        plt.title(f'greedy error {property} {min(self.eccmin_list)} - {max(self.eccmin_list)}' )
        plt.xlabel('Number of waveforms')
        plt.ylabel('error')
        plt.yscale('log')
        plt.legend()
        plt.show()
        
        figname = f'Greedy_error_{property}_{min(self.eccmin_list)}_{max(self.eccmin_list)}.png'
        fig_greedy_error.savefig('Images/Greedy_error/' + figname)

        print('Highest error of best approximation of the basis: ', np.min(greedy_errors))
    
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
            axs[0].set_ylim(-1, 1)
            
            legend = axs[0].legend(loc='lower left', ncol=7)
            
            for text in legend.get_texts():
                text.set_fontsize(8)
            
            axs[1].plot(self.TS_M, dataset[i], label = 'e = {}'.format(self.eccmin_list[i]), linewidth=0.6)
            axs[1].set_ylabel('$\Delta A_{22}$')
            axs[1].set_ylim(-1, 1)
            
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
                                                    
            for i in range(len(hp_DS)):
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

    def __init__(self, eccmin_list, waveform_size = None, total_mass=10, mass_ratio=1, freqmin=5):
        Simulate_Inspiral.__init__(self, eccmin=None, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin)
        Waveform_properties.__init__(self, eccmin=None, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin)
        Dataset.__init__(self, eccmin_list, waveform_size, total_mass, mass_ratio, freqmin)
        Reduced_basis.__init__(self, eccmin_list, waveform_size=waveform_size, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin)
        # super().__init__(eccmin_list, waveform_size, total_mass, mass_ratio, freqmin)



    def calc_empirical_interpolant(self, waveform, reduced_basis, T):

        B_j_vec = np.zeros((reduced_basis.shape[1], reduced_basis.shape[0]))
        empirical_interpolant = 0

        V = np.zeros((len(reduced_basis), len(reduced_basis)))
        for j in range(len(reduced_basis)):
            for i in range(len(T)):
                V[j][i] = reduced_basis[i][T[j]]

        for j in range(V.shape[1]): 
            B_j = 0
            for i in range(len(reduced_basis)):
                B_j += reduced_basis[i].T * np.linalg.inv(V)[i][j]

            B_j_vec[:, j] = B_j
        
        for j in range(reduced_basis.shape[0]):
            empirical_interpolant += B_j_vec[:, j]*waveform[T[j]]

        return empirical_interpolant
    

    def calc_empirical_nodes(self, reduced_basis, time_samples):
        i = np.argmax(reduced_basis[0])
        emp_nodes = [time_samples[i]]
        emp_nodes_idx = [i]
        EI_error = []

        for j in range(1, reduced_basis.shape[0]):
            empirical_interpolant = self.calc_empirical_interpolant(reduced_basis[j], reduced_basis[:j], emp_nodes_idx)
            
            # EI_error.append(np.linalg.norm(reduced_basis[j] - empirical_interpolant))
            
            r = empirical_interpolant - reduced_basis[j][:, np.newaxis].T
            EI_error.append(np.linalg.norm(r))
            idx = np.argmax(np.abs(r))
            emp_nodes.append(time_samples[idx]) 
            emp_nodes_idx.append(idx) 

        return emp_nodes_idx

    def plot_empirical_nodes(self, polarisation='plus', save_dataset = False):
        
        hp_DS, hc_DS, TS_M = self.import_polarisations(save_dataset)
        if polarisation == 'plus':
            reduced_basis = self.reduced_basis(hp_DS)
            y_label = '+'
        elif polarisation == 'cross':
            reduced_basis = self.reduced_basis(hc_DS)
            y_label = 'x'
        else:
            print('Choose polarisation = "plus" or "cross"')
        print(self.waveform_size)
        emp_nodes_idx = self.calc_empirical_nodes(reduced_basis, self.TS_M)

        nodes_time = []
        nodes_polarisation = []

        for node in emp_nodes_idx:
            nodes_time.append(self.TS_M[node])
            nodes_polarisation.append(0)

        fig_EIM = plt.figure(figsize=(12, 6))

        plt.plot(self.TS_M, hp_DS[-1], linewidth=0.2, label = 'e = {}'.format(self.eccmin_list[-1]))
        plt.scatter(nodes_time, nodes_polarisation, color='black', s=8)
        plt.ylabel(f'$h_{y_label}$')
        # axs[1].set_ylim(-0.5e-23, 0.5e-23)
        # axs[1].set_xlim(-0.4e6, 1000)
        plt.xlabel('t [M]')
        plt.legend(loc = 'upper left')  

        figname = 'EIM_{}_e={}.png'.format(polarisation, self.eccmin_list[-1])
        fig_EIM.savefig('Images/Empirical_nodes/' + figname)
        print('fig is saved')

        plt.show()

