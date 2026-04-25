from tracemalloc import start

from matplotlib import gridspec

from generate_greedy_training_set import *

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Product, ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel, ExpSineSquared, DotProduct, ConstantKernel as C
import time
import seaborn as sns
from matplotlib.lines import Line2D
from inspect import getframeinfo
from sklearn.preprocessing import StandardScaler
from fileinput import filename
from inspect import currentframe
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

f = currentframe()


@dataclass
class GPRFitResults(Warnings):
    # Initialization parameters
    property: str = "phase"

    ecc_ref_space_input: Any = None
    ecc_ref_space_output: Any = None

    mean_ano_ref_space_input: Any = None
    mean_ano_ref_space_output: Any = None

    mass_ratio_space_input: Any = None
    mass_ratio_space_output: Any = None

    chi1_space_input: Any = None
    chi1_space_output: Any = None

    chi2_space_input: Any = None
    chi2_space_output: Any = None

    time: Any = None

    N_basis_vecs: Optional[int] = None
    min_greedy_error: Optional[float] = None

    f_ref: float = None
    f_lower: float = None
    phiRef: float = 0.0
    inclination: float = 0.0
    truncate_at_ISCO: bool = True
    truncate_at_tmin: bool = True
    luminosity_distance: Optional[float] = None

    # Results fields for loading after initialization
    circ: Any = None
    residuals: Any = None
    basis_indices: Any = field(default_factory=list)
    empirical_indices: Any = field(default_factory=list)
    residual_basis: Any = None
    training_set: Any = None

    best_lmls: Any = None
    best_guess_kernels: Any = None
    optimized_kernels: Any = None
    mean_predictions: Any = None
    confidence_95_preds: Any = None

    def __post_init__(self):
        """Check if initial parameter spaces are valid and round them to 4 decimal places for consistency in filenames."""
        for name in [
            "ecc_ref_space_input",
            "mean_ano_ref_space_input",
            "mass_ratio_space_input",
            "chi1_space_input",
            "chi2_space_input",
        ]:
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, np.round(np.asarray(value, dtype=float), 4))
    
    def update_results(self, **kwargs):
        """Update the results fields of the GPRFitResults object with new values provided as keyword arguments. 
        Only existing attributes will be updated; any keys that do not correspond to existing attributes will raise an error."""
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"{type(self).__name__} has no attribute '{key}'")
            setattr(self, key, value)
            
    @staticmethod
    def _range_block(name, values):
        """Create block of [min_max_N] for each parameter space. Only include properties that are not None."""
        values = np.asarray(values, dtype=float)
        return f"{name}=[{values.min():g}_{values.max():g}_N={len(values)}]"

    @staticmethod
    def _scalar_block(name, value):
        """Formats a scalar property into a string block for filenames. If the value is None, it returns an empty string."""
        return f"{name}={value:g}"

    @staticmethod
    def _kernel_to_string(kernel):
        """Converts a kernel object to a string representation suitable for filenames by removing spaces and special characters."""
        text = str(kernel)
        text = text.replace(" ", "")
        text = text.replace("\n", "")
        text = text.replace("**", "^")
        text = text.replace("*", "x")
        text = text.replace("(", "")
        text = text.replace(")", "")
        return text

    def name_blocks(self):
        """Constructs a list of strings representing the properties of the GPRFitResults object for use in filenames."""
        blocks = [
            self.property,
            self._range_block("e", self.ecc_ref_space_input),
            self._range_block("l", self.mean_ano_ref_space_input),
            self._range_block("q", self.mass_ratio_space_input),
            self._range_block("x1", self.chi1_space_input),
            self._range_block("x2", self.chi2_space_input),
            self._scalar_block("fr", self.f_ref),
            self._scalar_block("fl", self.f_lower),
        ]

        if self.phiRef != 0:
            blocks.append(self._scalar_block("phi", self.phiRef))
        if self.inclination != 0:
            blocks.append(self._scalar_block("incl", self.inclination))
        if self.N_basis_vecs is not None:
            blocks.append(f"Nb={self.N_basis_vecs}")
        if self.min_greedy_error is not None:
            blocks.append(f"gerr={self.min_greedy_error}")
        if not self.truncate_at_ISCO:
            blocks.append("noISCO")
        if not self.truncate_at_tmin:
            blocks.append("notmin")
        if self.luminosity_distance is not None:
            blocks.append("SI")
        if self.optimized_kernels is not None:
            blocks.append(f"kernel={self._kernel_to_string(self.optimized_kernels)}")

        return blocks

    def filename(self, prefix="GPRFitResults", ext="pkl", directory=None):
        """Constructs a filename based on the properties of the GPRFitResults object."""
        name = f"{prefix}_{'_'.join(self.name_blocks())}.{ext}"
        if directory is not None:
            return f"{directory.rstrip('/')}/{name}"
        return name
    
    def figname(self, prefix="fig", ext="png", directory=None):
        # Ensure the directory exists, creating it if necessary and save
        if directory is not None:
            os.makedirs(directory, exist_ok=True)

        figname = self.filename(prefix=prefix, ext=ext, directory=directory)

        # Figure saving confirmation
        print(self.colored_text(f"Figure is saved in {figname}", 'blue'))

        return figname

    def save(self, prefix="GPRFitResults", directory="Straindata/GPRFitResults"):
        """Saves the GPRFitResults object to a file."""
        if directory is not None:
            os.makedirs(directory, exist_ok=True)

        filepath = self.filename(prefix=prefix, ext="pkl", directory=directory)

        with open(filepath, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Figure saving confirmation
        print(self.colored_text(f'GPRFitResults saved to {filepath}', 'blue'))

        return filepath

    def load(self, prefix="GPRFitResults", directory="Straindata/GPRFitResults"):
        """Loads the saved GPRFitResults object from a file."""

        filepath = self.filename(prefix=prefix, ext="pkl", directory=directory)
        with open(filepath, 'rb') as f:
            loaded_obj = pickle.load(f)

        if not isinstance(loaded_obj, GPRFitResults):
            raise ValueError(f"Loaded object is not of type GPRFitResults. Got {type(loaded_obj)} instead.")

        # Figure loading confirmation
        print(self.colored_text(f"GPRFitResults object loaded from {filepath}", 'blue'))

        return loaded_obj


class Generate_Offline_Surrogate(Generate_TrainingSet):

    def __init__(self, time_array, 
                 mass_ratio_range=[1, 20], 
                 ecc_ref_range=[0.0, 0.3], 
                 mean_ano_ref_range=[0, 2*np.pi], 
                 chi1_range=[-0.995, 0.995], 
                 chi2_range=[-0.995, 0.995], 
                 amount_input_wfs=100, 
                 amount_output_wfs=500, 
                 N_basis_vecs_amp=None, 
                 N_basis_vecs_phase=None, 
                 min_greedy_error_amp=None, 
                 min_greedy_error_phase=None, 
                 training_set_selection='GPR_opt', 
                 minimum_spacing_greedy=0.008, 
                 f_lower=10, 
                 f_ref=20, 
                 phiRef=0., 
                 inclination=0., 
                 truncate_at_ISCO=True, 
                 truncate_at_tmin=True):
        
        
        if (N_basis_vecs_amp is None and N_basis_vecs_phase is None) and \
            (min_greedy_error_amp is None and min_greedy_error_phase is None):
                print('Choose either settings for the amount of basis_vecs OR the minimum greedy error.')
                sys.exit(1)

        # Input refers to the dataset used to pick the training set, output refers to the parameterspace of the computed surrogate
        self.ecc_ref_parameterspace_range = ecc_ref_range
        self.ecc_ref_space_input = self.allowed_eccentricity_warning(np.linspace(*ecc_ref_range, amount_input_wfs).round(4))
        self.ecc_ref_space_output = self.allowed_eccentricity_warning(np.linspace(*ecc_ref_range, amount_output_wfs).round(4))

        self.mean_ano_ref_space_input = self.allowed_mean_anomaly_warning(np.linspace(*mean_ano_ref_range, amount_input_wfs).round(4))
        self.mean_ano_ref_space_output = self.allowed_mean_anomaly_warning(np.linspace(*mean_ano_ref_range, amount_output_wfs).round(4))

        self.mass_ratio_space_input = self.allowed_mass_ratio_warning(np.linspace(*mass_ratio_range, amount_input_wfs).round(4))
        self.mass_ratio_space_output = self.allowed_mass_ratio_warning(np.linspace(*mass_ratio_range, amount_output_wfs).round(4))

        self.chi1_space_input = self.allowed_chispin_warning(np.linspace(*chi1_range, amount_input_wfs).round(4))
        self.chi1_space_output = self.allowed_chispin_warning(np.linspace(*chi1_range, amount_output_wfs).round(4))

        self.chi2_space_input = self.allowed_chispin_warning(np.linspace(*chi2_range, amount_input_wfs).round(4))
        self.chi2_space_output = self.allowed_chispin_warning(np.linspace(*chi2_range, amount_output_wfs).round(4))

        self.amount_input_wfs = amount_input_wfs
        self.amount_output_wfs = amount_output_wfs

        self.training_set_selection = training_set_selection

        self.surrogate_amp = None
        self.surrogate_phase = None

        self.gaussian_fit_amp = None
        self.gaussian_fit_phase = None
        self.indices_basis_amp = None
        self.indices_basis_phase = None

        self.gpr_amp = None
        self.gpr_phase = None

        super().__init__(time_array=time_array, 
                 ecc_ref_parameterspace=self.ecc_ref_space_input, 
                 mean_ano_parameterspace=self.mean_ano_ref_space_input, 
                 mass_ratio_parameterspace=self.mass_ratio_space_input,
                 chi1_parameterspace=self.chi1_space_input,
                 chi2_parameterspace=self.chi2_space_input,
                 N_basis_vecs_amp=N_basis_vecs_amp, 
                 N_basis_vecs_phase=N_basis_vecs_phase, 
                 min_greedy_error_amp=min_greedy_error_amp, 
                 min_greedy_error_phase=min_greedy_error_phase, 
                 minimum_spacing_greedy=minimum_spacing_greedy, 
                 f_ref=f_ref, 
                 f_lower=f_lower, 
                 phiRef=phiRef, 
                 inclination=inclination, 
                 truncate_at_ISCO=truncate_at_ISCO, 
                 truncate_at_tmin=truncate_at_tmin)
        
    def result_kwargs_gpr(self, property):
        """Collect all parameters needed for saving results.
        Stores both input and output parameter spaces.
        Only property-dependent fields are switched."""
        
        if property == "phase":
            N_basis_vecs = self.N_basis_vecs_phase
            min_greedy_error = self.min_greedy_error_phase
        elif property == "amplitude":
            N_basis_vecs = self.N_basis_vecs_amp
            min_greedy_error = self.min_greedy_error_amp
        else:
            raise ValueError("property must be 'phase' or 'amplitude'")

        return dict(
            # property-specific
            property=property,
            N_basis_vecs=N_basis_vecs,
            min_greedy_error=min_greedy_error,

            # parameter spaces (store BOTH input and output)
            ecc_ref_space_input=getattr(self, "ecc_ref_space_input", None),
            ecc_ref_space_output=getattr(self, "ecc_ref_space_output", None),

            mean_ano_ref_space_input=getattr(self, "mean_ano_ref_space_input", None),
            mean_ano_ref_space_output=getattr(self, "mean_ano_ref_space_output", None),

            mass_ratio_space_input=getattr(self, "mass_ratio_space_input", None),
            mass_ratio_space_output=getattr(self, "mass_ratio_space_output", None),

            chi1_space_input=getattr(self, "chi1_space_input", None),
            chi1_space_output=getattr(self, "chi1_space_output", None),

            chi2_space_input=getattr(self, "chi2_space_input", None),
            chi2_space_output=getattr(self, "chi2_space_output", None),

            # waveform / global settings
            time=self.time,
            f_ref=self.f_ref,
            f_lower=self.f_lower,
            phiRef=self.phiRef,
            inclination=self.inclination,
            truncate_at_ISCO=self.truncate_at_ISCO,
            truncate_at_tmin=self.truncate_at_tmin,
        )

    def _diagnose_gpr_issues(self, gaussian_process, X_train_scaled, y_train_scaled):
            issues = []
            
            # 1. Check if length scale is hitting bounds
            kernel = gaussian_process.kernel_
            if hasattr(kernel, 'length_scale_bounds'):
                length_scale = kernel.length_scale
                lb, ub = kernel.length_scale_bounds
                if np.abs(length_scale - lb) < 1e-6 or np.abs(length_scale - ub) < 1e-6:
                    issues.append("Length scale hitting optimization bounds")
            
            # 2. Check training fit quality
            y_pred, std_pred = gaussian_process.predict(X_train_scaled, return_std=True)
            residuals = y_pred - y_train_scaled
            if np.max(np.abs(residuals)) > 2.0:  # Large residuals in scaled space
                issues.append("Poor fit to training data")
            
            # 3. Check uncertainty calibration
            z_scores = residuals / std_pred
            if np.mean(z_scores**2) > 2.0:  # Poor uncertainty calibration
                issues.append("Poor uncertainty calibration")
            
            # 4. Check kernel matrix conditioning
            K = gaussian_process.kernel_(X_train_scaled)
            cond_num = np.linalg.cond(K + 1e-8 * np.eye(K.shape[0]))
            if cond_num > 1e10:
                issues.append(f"Severe ill-conditioning (cond={cond_num:.2e})")
            
            return issues
    

    def _gaussian_process_regression_test(self, time_node, train_obj:TrainingSetResults, optimized_kernel=None, plot_kernels=False, save_fig_kernels=False):
        """
        fit_to_params: choose 'greedy' for fitting to greedy paramaters, or choors 'GPR_opt' for fitting to GPR optimized chosen parameters
        """

        # Extract X and training data
        X = self.ecc_ref_space_output[:, np.newaxis]
        X_train = np.array(self.ecc_ref_space_input[train_obj.basis_indices]).reshape(-1, 1)
        y_train = np.squeeze(train_obj.training_set.T[time_node])

        # Scale X_train
        scaler_x = StandardScaler()
        X_train_scaled = scaler_x.fit_transform(X_train)

        # Scale X (for predictions)
        X_scaled = scaler_x.transform(X)

        # Scale y_train
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

        # Define kernels to try

        # Median distance between nearest neighbors 
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(X_train_scaled)
        distances, indices = nn.kneighbors(X_train_scaled)
        median_nn_distance = np.median(distances[:, 1])  # distance to nearest neighbor
        base_ls = max(0.1, median_nn_distance * 2)  # Scale up from nearest neighbor distance
        print('base_ls = ', base_ls)

        # Length scale multipliers and upper bounds
        ls_multipliers = [0.3, 1.0, 3.0, 10.0] # length scale multipliers
        ls_upper_bounds = [0.5, 1.0, 2.0, 5.0] # length scale upper bounds
        smoothness_params = [1.0, 1.5, 2.0, 2.5]  # Matern nu values

        ls_multipliers = [1.0]
        ls_upper_bounds = [1.0]
        smoothness_params = [1.0]

        # Different kernel configurations
        kernels = []

        

        # Matern based kernels without noise
        for nu in smoothness_params:
            for ls_mult in ls_multipliers:
                for ls_ub in ls_upper_bounds:
                    if base_ls * ls_mult <= ls_ub:
                        kernel = Matern(length_scale=base_ls * ls_mult, length_scale_bounds=(0.1, ls_ub), nu=nu)
                        kernels.append([kernel, ls_ub])
        
        # # Matern based kernels with noise terms
        # noise_levels = [1e-4, 1e-3, 1e-2, 1e-1]
        # for nu in smoothness_params:
        #     for ls_mult in ls_multipliers:
        #         for ls_ub in ls_upper_bounds:
        #             for noise in noise_levels:
        #                 kernel = Matern(length_scale=base_ls * ls_mult, length_scale_bounds=(1, ls_ub), nu=nu) + WhiteKernel(noise_level=noise, noise_level_bounds=(1e-6, 1))
        #                 kernels.append(kernel)

        # # RBF based kernels
        # for ls_mult in ls_multipliers:
        #     for ls_ub in ls_upper_bounds:
        #         kernel = RBF(length_scale=base_ls * ls_mult, length_scale_bounds=(1, ls_ub))
        #         kernels.append(kernel)

        # # Rational quadratic based kernels
        # alphas = [0.1, 1.0, 10.0]
        # for alpha in alphas:
        #     for ls_mult in ls_multipliers:
        #         for ls_ub in ls_upper_bounds:
        #             kernel = RationalQuadratic(length_scale=base_ls * ls_mult, alpha=alpha, length_scale_bounds=(1e-2, ls_ub), alpha_bounds=(1e-2, 100))
        #             kernels.append(kernel)

        # kernels = [
        #     Matern(length_scale=0.1, length_scale_bounds=(1e-1, 1), nu=1.5)  # <= 0.3 eccentricity,

        # ]

        y_predict_kernels = []
        y_predict_std_kernels = []
        lml_kernels = []
        
        best_lml = -np.inf # log marginal likelihood

        for kernel in kernels:
            try:
                start = time.time()
                # if optimized_kernel is None:
                #     gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
                # else:
                #     gaussian_process = GaussianProcessRegressor(kernel=optimized_kernel, optimizer=None)
                # print(self.colored_text(f'Trying kernel: {kernel[0]} with ls_bounds=[0.1, {kernel[1]}]', 'blue'))
                gaussian_process = GaussianProcessRegressor(kernel=kernel[0], n_restarts_optimizer=20, random_state=42)
                
                # Fit the GP model on scaled data
                gaussian_process.fit(X_train_scaled, y_train_scaled)
                optimized_kernel = gaussian_process.kernel_

                end = time.time()

                # Log-Marginal Likelihood
                lml = gaussian_process.log_marginal_likelihood_value_

                # Calculate training score (RMSE on training data)
                y_pred_train, std_train = gaussian_process.predict(X_train_scaled, return_std=True)
                train_rmse = np.sqrt(np.mean((y_pred_train - y_train_scaled)**2))
                
                # Print the optimized kernel and hyperparameters
                # print(f"kernel = {kernel[0]}, ls_bounds=[0.1, {kernel[1]}]; Optimized kernel: {optimized_kernel} | time = {end - start:.2f}s | LML = {lml:.4f}, training_score = {train_rmse}")

                # Make predictions on scaled X
                y_predict_scaled, std_prediction_scaled = gaussian_process.predict(X_scaled, return_std=True)
                y_predict = scaler_y.inverse_transform(y_predict_scaled.reshape(-1, 1)).flatten()
                y_predict_std = std_prediction_scaled * scaler_y.scale_[0]

                y_predict_kernels.append(y_predict)
                y_predict_std_kernels.append(y_predict_std)

                self._diagnose_gpr_issues(gaussian_process, X_train_scaled, y_train_scaled)


                if lml > best_lml:
                    best_lml = lml
                    best_guess_kernel = kernel
                    best_optimized_kernel = optimized_kernel
                    best_y_predict = y_predict
                    best_y_predict_std = y_predict_std

            except Exception as e:
                print(self.colored_text(f'GPR failed for kernel {kernel}: {e}', 'red'))
                continue

        print(self.colored_text(f"Best guess kernel = {best_guess_kernel}; Optimized kernel: {best_optimized_kernel} | time = {end - start:.2f}s | LML = {best_lml:.4f} | X_train_scaled = {X_train_scaled[:10]} | Y_train_scaled = {y_train_scaled[:10]}", 'green'))
        lml_kernels.append(best_lml)


        if plot_kernels is True:
            GPR_fit = plt.figure()

            for i in range(len(y_predict_kernels)):
                plt.scatter(X_train, y_train, color='red', label="Observations", s=10)
                plt.plot(X, y_predict_kernels[i], label='Mean prediction', linewidth=0.8)
                plt.fill_between(
                    X.ravel(),
                (y_predict_kernels[i] - 1.96 * y_predict_std_kernels[i]), 
                (y_predict_kernels[i] + 1.96 * y_predict_std_kernels[i]),
                    alpha=0.5,
                    label=r"95% confidence interval",
                )
            plt.legend(loc = 'upper left')
            plt.xlabel("$e$")
            if property == 'amplitude':
                plt.ylabel("$f_A(e)$")
            elif property == 'phase':
                plt.ylabel("$f_{\phi}(e)$")
            plt.title(f"GPR {property} at T_{time_node}, best optimized kernel: {best_optimized_kernel[0]} with ls_bounds=[0.1, {best_optimized_kernel[1]}]")
            # plt.show()

            if save_fig_kernels is True:
                figname = f'Images/Gaussian_kernels/Gaussian_kernels_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_ref_space)}_{max(self.ecc_ref_space)}_Ni={len(self.ecc_ref_space)}]_No={len(self.ecc_ref_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_basis_vecs_phase}_Nga={self.N_basis_vecs_amp}.png'
                
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Gaussian_kernels', exist_ok=True)
                GPR_fit.savefig(figname)

                print(self.colored_text(f'Figure is saved in {figname}', 'blue'))

        # return gpr_obj
        return best_y_predict, [(best_y_predict - 1.96 * best_y_predict_std), (best_y_predict + 1.96 * best_y_predict_std)], best_optimized_kernel, best_lml


    def _gaussian_process_regression(self, time_node, training_set, optimized_kernel=None, plot_kernels=False, save_fig_kernels=False):
        # Extract X and training data
        X = self.ecc_ref_space_output[:, np.newaxis]
        X_train = np.array(self.best_rep_parameters).reshape(-1, 1)
        y_train = np.squeeze(training_set.T[time_node])

        # Scale X_train
        scaler_x = StandardScaler()
        X_train_scaled = scaler_x.fit_transform(X_train)

        # Scale X (for predictions)
        X_scaled = scaler_x.transform(X)

        # Scale y_train
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

        kernels = [
            Matern(length_scale=0.1, length_scale_bounds=(1e-1, 1), nu=1.5)  # <= 0.3 eccentricity
        ]

        mean_prediction_per_kernel = []
        std_predictions_per_kernel = []
        lml_per_kernel = []

        for kernel in kernels:
            start = time.time()
            if optimized_kernel is None:
                gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
            else:
                gaussian_process = GaussianProcessRegressor(kernel=optimized_kernel, optimizer=None)

            # Fit the GP model on scaled data
            gaussian_process.fit(X_train_scaled, y_train_scaled)
            optimized_kernel = gaussian_process.kernel_

            end = time.time()

            # Log-Marginal Likelihood
            lml = gaussian_process.log_marginal_likelihood_value_
            lml_per_kernel.append(lml)

            # Print the optimized kernel and hyperparameters
            print(f"kernel = {kernel}; Optimized kernel: {optimized_kernel} | time = {end - start:.2f}s | LML = {lml:.4f}")

            # Make predictions on scaled X
            mean_prediction_scaled, std_prediction_scaled = gaussian_process.predict(X_scaled, return_std=True)
            mean_prediction = scaler_y.inverse_transform(mean_prediction_scaled.reshape(-1, 1)).flatten()
            std_prediction = std_prediction_scaled * scaler_y.scale_[0]

            mean_prediction_per_kernel.append(mean_prediction)
            std_predictions_per_kernel.append(std_prediction)
        
        if plot_kernels is True:
            GPR_fit = plt.figure()

            for i in range(len(mean_prediction_per_kernel)):
                plt.scatter(X_train, y_train, color='red', label="Observations", s=10)
                plt.plot(X, mean_prediction_per_kernel[i], label='Mean prediction', linewidth=0.8)
                plt.fill_between(
                    X.ravel(),
                (mean_prediction_per_kernel[i] - 1.96 * std_predictions_per_kernel[i]), 
                (mean_prediction_per_kernel[i] + 1.96 * std_predictions_per_kernel[i]),
                    alpha=0.5,
                    label=r"95% confidence interval",
                )
            plt.legend(loc = 'upper left')
            plt.xlabel("$e$")
            if property == 'amplitude':
                plt.ylabel("$f_A(e)$")
            elif property == 'phase':
                plt.ylabel("$f_{\phi}(e)$")
            # plt.title(f"GPR {property} at T_{time_node}")
            # plt.show()

            if save_fig_kernels is True:
                figname = f'Gaussian_kernels_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_ref_space)}_{max(self.ecc_ref_space)}_Ni={len(self.ecc_ref_space)}]_No={len(self.ecc_ref_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}.png'
                
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Gaussian_kernels', exist_ok=True)
                GPR_fit.savefig('Images/Gaussian_kernels/' + figname)

                print('Figure is saved in Images/Gaussian_kernels/' + figname)

        return mean_prediction, [(mean_prediction - 1.96 * std_prediction), (mean_prediction + 1.96 * std_prediction)], optimized_kernel, lml_per_kernel

    def _get_gpr_obj(self, property):
        """Helper method to get or create the GPRFitResults object for the specified property."""
        if property == "amplitude":
            if self.gpr_amp is None:
                self.gpr_amp = GPRFitResults(**self.result_kwargs_gpr(property="amplitude"))
            return self.gpr_amp
        
        elif property == "phase":
            if self.gpr_phase is None:
                self.gpr_phase = GPRFitResults(**self.result_kwargs_gpr(property="phase"))
            return self.gpr_phase
        
        else:
            raise ValueError(f"Unknown property: {property}")
        
        

    def fit_to_training_set(self, 
                            property, 
                            min_greedy_error=None, N_basis_vecs=None, 
                            training_set=None, 
                            X_train=None, 
                            save_fits_to_file=True, 
                            plot_kernels=False, save_fig_kernels=False,
                            plot_GPR_fits=False, save_fig_GPR_fits=False, 
                            plot_residuals_ecc_evolve=False, save_fig_ecc_evolve=False, 
                            plot_residuals_time_evolve=False, save_fig_time_evolve=False, 
                            no_file_load=False
                            ):

        # training object for the chosen property (phase or amplitude)
        gpr_obj = self._get_gpr_obj(property)
        train_obj = self._get_training_obj(property)

        # Resolve the number of basis vectors and minimum greedy error based on the training set object if not provided as arguments
        if N_basis_vecs is None and min_greedy_error is None:
            N_basis_vecs = self.resolve_property(N_basis_vecs, gpr_obj.N_basis_vecs)
            min_greedy_error = self.resolve_property(min_greedy_error, gpr_obj.min_greedy_error)

            if N_basis_vecs is None or min_greedy_error is None:
                print('Choose either settings for the amount of basis_vecs OR the minimum greedy error.')
                sys.exit(1)

        try:
            if no_file_load is True:
                raise FileNotFoundError
          
            gpr_obj = gpr_obj.load()
            train_obj = train_obj.load()

        except Exception as e:
            print(f'line {getframeinfo(f).lineno}: {e}')

            if training_set is None:
                try:
                    train_obj = train_obj.load()
                except Exception as e:
                    print(f'line {getframeinfo(f).lineno}: {e}')
                    # Generate the training set of greedy parameters at empirical nodes
                    train_obj = self.get_training_set_greedy(property=property, min_greedy_error=min_greedy_error, N_greedy_vecs=N_basis_vecs)

            # Create empty arrays to save fitvalues
            gpr_obj.mean_predictions = np.zeros((train_obj.training_set.shape[1], len(self.ecc_ref_space_output)))
            gpr_obj.confidence_95_preds = np.zeros((train_obj.training_set.shape[1], 2, len(self.ecc_ref_space_output))) # 95% confidence interval
            gpr_obj.best_lmls = np.zeros(train_obj.training_set.shape[1])

            print(f'Interpolate {property}...')

            start2 = time.time()
            optimized_kernel = None

            for node_i in range(len(train_obj.empirical_indices)):
                
                best_y_predict, confidence_95, optimized_kernel, best_lml = self._gaussian_process_regression_test(node_i, train_obj, optimized_kernel, X_train, plot_kernels)

                gpr_obj.mean_predictions[node_i] = best_y_predict # Best prediction 
                gpr_obj.confidence_95_preds[node_i] = confidence_95 # 95% confidence level
                gpr_obj.best_lmls[node_i] = best_lml # Log-Marginal likelihood

            end2 = time.time()
            print(f'time full GPR = {end2 - start2}')


        # If plot_fits is True, plot the GPR fits
        if plot_GPR_fits:
            self._plot_GPR_fits(train_obj=train_obj, gpr_obj=gpr_obj, gaussian_fit=None, training_set=None, lml_fits=None, save_fig_fits=save_fig_GPR_fits)

        # If save_fits_to_file is True, save the GPR fits to a file
        if save_fits_to_file:
            # Save the GPR fits to a file
            gpr_obj.save()
        
        # If plot_residuals_ecc_evolve or plot_residuals_time_evolve is True, plot the residuals
        if (plot_residuals_ecc_evolve is True) or (plot_residuals_time_evolve is True):
            # Plot residuals
            self._plot_residuals(train_obj=train_obj, plot_eccentric_evolv=plot_residuals_ecc_evolve, save_fig_eccentric_evolve=save_fig_ecc_evolve, plot_time_evolve=plot_residuals_time_evolve, save_fig_time_evolve=save_fig_time_evolve)

        return gpr_obj
    

    def fit_to_training_set_GPR_opt(self, 
                                    property, 
                                    N_basis_vecs=None, 
                                    save_fits_to_file=True, 
                                    plot_kernels=False, save_fig_kernels=False,
                                    plot_GPR_fits=False, save_fig_GPR_fits=False, 
                                    plot_residuals_ecc_evolve=False, save_fig_ecc_evolve=False, 
                                    plot_residuals_time_evolve=False, save_fig_time_evolve=False,
                                    plot_greedy_vecs=False, save_fig_greedy_vecs=False, 
                                    plot_greedy_error=False, save_fig_greedy_error=False, 
                                    plot_emp_nodes_at_ecc=False, save_fig_emp_nodes=False, 
                                    plot_training_set=False, save_fig_training_set=False,
                                    save_fits_to_file_iter = False
                                    ):
        

        # Get first 3 points to produce a start for GPR
        train_obj = self.get_training_set_greedy(property, N_greedy_vecs=3, emp_nodes_of_full_dataset=True, plot_greedy_error=plot_greedy_error, save_fig_greedy_error=save_fig_greedy_error, plot_emp_nodes_at_ecc=plot_emp_nodes_at_ecc, save_fig_emp_nodes=save_fig_emp_nodes, plot_training_set=plot_training_set, save_fig_training_set=save_fig_training_set, 
                        save_dataset_to_file=True, plot_greedy_vecs=plot_greedy_vecs, save_fig_greedy_vecs=save_fig_greedy_vecs)


        while len(train_obj.basis_indices) <= N_basis_vecs:

            # Update the length of the basis for every iteration

            train_obj.Nb = len(train_obj.residual_basis)


            # Save fits only for last iteration 
            if len(train_obj.basis_indices) == N_basis_vecs:
                save_fits_to_file_iter = save_fits_to_file

            # Fit the basis vecs with GPR and save fits file at last iteration
            gpr_obj = self.fit_to_training_set(property, N_basis_vecs=len(train_obj.basis_indices), training_set=train_obj.residual_basis, no_file_load=True, save_fits_to_file=save_fits_to_file_iter, plot_kernels=plot_kernels, plot_GPR_fits=plot_GPR_fits, save_fig_kernels=save_fig_kernels, save_fig_GPR_fits=save_fig_GPR_fits, plot_residuals_ecc_evolve=plot_residuals_ecc_evolve, save_fig_ecc_evolve=save_fig_ecc_evolve, plot_residuals_time_evolve=plot_residuals_time_evolve, save_fig_time_evolve=save_fig_time_evolve)

            # Load in property residuals of full parameter space dataset
            try:
                filename = f'Straindata/Residuals/residuals_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_ref_space_output)}_{max(self.ecc_ref_space_output)}_N={len(self.ecc_ref_space_output)}].npz'
                with np.load(filename) as data:
                    residual_parameterspace_output = data['residual']
                    self.time = data['time']

            except Exception as e:
                print(f'line {getframeinfo(f).lineno}: {e}')
                train_obj_output = TrainingSetResults(**self.result_kwargs_training(property=property, ecc_ref_space=self.ecc_ref_space_output))
                residual_parameterspace_output = self.generate_property_dataset(train_obj_output, save_dataset_to_file=True)

            # Calculate the relative errors of GPR fits vs property dataset
            combined_gaussian_error = np.zeros(len(self.ecc_ref_space_output))
            for i in range(len(gpr_obj.mean_predictions)):
                combined_gaussian_error += abs(residual_parameterspace_output[:, self.empirical_nodes_idx[i]] - gpr_obj.mean_predictions.T[:, i])

            # Add new training set point at the place of worst fit error
            worst_relative_GPR_error_idx = np.argmax(combined_gaussian_error)

            # Add time-domain vec of worst fit parameter to the residual basis
            opt_residual_basis_vector = residual_parameterspace_output[worst_relative_GPR_error_idx]
            self.residual_reduced_basis = np.vstack([self.residual_reduced_basis, opt_residual_basis_vector])

            # Update the best pick parameters 
            self.indices_basis = np.append(list(self.indices_basis), worst_relative_GPR_error_idx)
            self.best_rep_parameters = np.append(list(self.best_rep_parameters), self.ecc_ref_space_output[worst_relative_GPR_error_idx])

            #Generate the training set at empirical nodes for next GPR iteration
            residual_training_set = self.residual_reduced_basis[:, self.empirical_nodes_idx]

        return gpr_obj


    def _plot_GPR_fits(self, train_obj:TrainingSetResults, gpr_obj:GPRFitResults, gaussian_fit=None, training_set=None, lml_fits=None, save_fig_fits=False):

        if gpr_obj.mean_predictions is None:
            try:
                gpr_obj = gpr_obj.load()
            except Exception as e:
                print(f'line {getframeinfo(f).lineno}: {e}. \n Make sure to run fit_to_training_set() before plotting')

        if train_obj.training_set is None:
            try:
                train_obj = train_obj.load()
            except Exception as e:
                print(f'line {getframeinfo(f).lineno}: {e}. \n Make sure to run fit_to_training_set before plotting')

        train_output_param_space = TrainingSetResults(**self.result_kwargs_training(property=train_obj.property, ecc_ref_space=self.ecc_ref_space_output))
        train_output_param_space = self.generate_property_dataset(train_obj=train_output_param_space, save_residuals=True, save_polarizations=True)

        best_rep_parameters = train_obj.ecc_ref_space[train_obj.basis_indices]

        fig_residual_training_fit, axs = plt.subplots(
            3, 1,
            figsize=(11, 6),
            gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.2}
        )

        axs[1].sharex(axs[0])

        # Increase only the gap between axs[1] and axs[2]
        pos = axs[2].get_position()
        axs[2].set_position([
            pos.x0,
            pos.y0 - 0.04,   # move down; tune this
            pos.width,
            pos.height
        ])
        # # Create a colormap from Viridis
        # color_palette = sns.color_palette("tab10", as_cmap=True)
        # # Number of distinct colors needed
        # num_colors = len(gpr_obj.mean_predictions)  # Replace with the actual number of datasets (e.g., len(gaussian_fit))
        # # Evenly sample the colormap
        # colors = [color_palette(i / (num_colors - 1)) for i in range(num_colors)]

        # Define custom legend elements for training points and true phase/amplitude
        custom_legend_elements = [
            Line2D([0], [0], linestyle='dashed', color='black', linewidth=1, label=f'true {train_obj.property}'),
            Line2D([0], [0], marker='o', linestyle='None', color='black', label='training Points')
        ]

        # Create a list to hold dynamic handles and labels
        dynamic_handles = []
        dynamic_labels = []

        # Sort gaussian fits by worst likelihood 
        lml_array = np.array(gpr_obj.best_lmls).flatten()
        sorted_lml_fits = np.argsort(lml_array)[::-1] # Highest to lowest

        # Plotting data
        combined_rel_error = np.zeros(len(self.ecc_ref_space_output))
        # Plot 10 worst fits based on highest log-marginal likelihood
        for i in sorted_lml_fits[:10]:
            ##################################################
            # GPR Fit
            ##################################################
            # Plot mean predictions of GPR fit at empirical nodes
            line_fit, = axs[0].plot(self.ecc_ref_space_output, gpr_obj.mean_predictions.T[:, i], linewidth=0.6)
            
            # Scatter plot for training points
            axs[0].scatter(best_rep_parameters, train_obj.training_set[:, i], s=6)

            # Plot residuals (true phase/amplitude at empirical nodes)
            axs[0].plot(self.ecc_ref_space_output, train_output_param_space.residuals[:, train_obj.empirical_indices[i]], 
                        linestyle='dashed', linewidth=0.6)
            

            # Collect handles and labels for the dynamic fits
            dynamic_handles.append(line_fit)
            dynamic_labels.append(f't={int(self.time[train_obj.empirical_indices[i]])} [M]')

            ###################################################
            # Relative error
            ###################################################
            # Plot relative error of GPR fit vs true property at empirical nodes
            relative_error = abs(train_output_param_space.residuals[:, train_obj.empirical_indices[i]] - gpr_obj.mean_predictions.T[:, i])
            combined_rel_error += relative_error

            axs[1].plot(self.ecc_ref_space_output, 
                        relative_error, 
                        linewidth=0.6)
            
            ################################################
            # Empirical nodes
            ################################################
            axs[2].scatter(self.time[train_obj.empirical_indices], np.zeros(len(train_obj.empirical_indices)), s=3, label='Empirical nodes')

        # Combine custom and dynamic legend elements
        combined_handles = custom_legend_elements + dynamic_handles
        combined_labels = [handle.get_label() for handle in custom_legend_elements] + dynamic_labels

        # Add the combined legend to the top-left subplot
        axs[0].legend(combined_handles, combined_labels, ncol=2)

        # Set labels and titles
        if train_obj.property == 'phase':
            axs[0].set_ylabel('$\Delta \phi$')
            # axs[0].set_title(f'GPRfit $\phi$; greedy error = {min_greedy_error},N={len(self.indices_basis)}')
        else:
            axs[0].set_ylabel('$\Delta$ A')
            # axs[0].set_title(f'GPRfit A; greedy error = {min_greedy_error}, N={len(self.indices_basis)}')
        axs[0].grid()

        axs[1].set_xlabel('eccentricity')
        if train_obj.property == 'phase':
            axs[1].set_ylabel('|$\Delta \phi_{S} - \Delta \phi|$')
        else:
            axs[1].set_ylabel('|$\Delta A_{S} - \Delta A|$')
        axs[1].grid()

        axs[2].set_xlabel('time [M]')

        plt.setp(axs[0].get_xticklabels(), visible=False)
        plt.tight_layout()

        # Save the figure if requested
        if save_fig_fits is True:
            figname = gpr_obj.figname(prefix="GPR_fits", ext="png", directory="Images/Gaussian_fits")
            fig_residual_training_fit.savefig(figname)

    

    def compute_B_matrix(self, train_obj:TrainingSetResults, save_matrix_to_file=True):
        """
        Computes the B matrix for all empirical nodes and basis functions.
        
        e_matrix: Array of shape (m, time_samples) representing the reduced basis functions evaluated at different time samples.
        V_inv: Inverse of the interpolation matrix of shape (m, m).
        
        Returns:
        B_matrix: Array of shape (m, time_samples) where each row represents B_j(t) for j=1,2,...,m
        """
        filename = train_obj.filename(prefix='B_matrix', directory='Straindata/B_matrix')

        try:
            load_B_matrix = np.load(filename)
            B_matrix = load_B_matrix['B_matrix']
            print(f'B_matrix {property} load succeeded: {filename}', B_matrix.shape)

            load_B_matrix.close()

        except Exception as e:
            print(f'line {getframeinfo(f).lineno}: {e}')
            
            m, time_length = train_obj.residual_basis.shape
            B_matrix = np.zeros((m, time_length))

            V = np.zeros((m, m))
            for j in range(m):
                for i in range(m):
                    # print(i, j)
                    V[j][i] = train_obj.residual_basis[i][train_obj.empirical_indices[j]]

            V_inv = np.linalg.pinv(V)

            
            # Compute each B_j(t) for j = 1, 2, ..., m
            for j in range(m):
                # Compute B_j(t) as a linear combination of all e_i(t) with weights from V_inv[:, j]
                for i in range(m):
                    B_matrix[j] += train_obj.residual_basis[i] * V_inv[i, j]
            
        
            if save_matrix_to_file:

                # Ensure the directory exists, creating it if necessary and save
                # os.makedirs('Straindata/B_matrix', exist_ok=True)
                np.savez(filename, B_matrix=B_matrix)
                # print('B_matrix fits saved in Straindata/B_matrix/' + filename)

        return B_matrix
    

sampling_frequency = 2048 # or 4096
duration = 4 # seconds
time_array = np.linspace(-duration, 0, int(sampling_frequency * duration))  # time in seconds

gt = Generate_Offline_Surrogate(time_array=time_array, 
                                ecc_ref_range=[0.0, 0.1], 
                                mean_ano_ref_range=[0, 0],
                                mass_ratio_range=[1, 1],
                                chi1_range=[0, 0],
                                chi2_range=[0, 0],
                                amount_input_wfs=100, 
                                amount_output_wfs=500, 
                                min_greedy_error_amp=1e-8,
                                min_greedy_error_phase=1e-6,
                                minimum_spacing_greedy=0.003, 
                                training_set_selection='greedy')
train_obj_p = gt._get_training_obj('phase')
# gt.generate_property_dataset(train_obj=train_obj_p, 
#                              plot_residuals_eccentric_evolv=True,
#                              plot_residuals_time_evolv=True
#                              )
# gt.get_training_set_greedy(property='phase', 
#                            min_greedy_error=1e-8, 
#                            plot_greedy_error=True,
#                            plot_emp_nodes_at_ecc=True, 
#                            plot_training_set=True,
#                            plot_greedy_vecs=True,
#                            plot_emp_nodes_on_basis=True)

gt.fit_to_training_set('amplitude', min_greedy_error=1e-8, save_fits_to_file=True, 
                           plot_GPR_fits=True, save_fig_GPR_fits=True, 
                        #    plot_residuals_ecc_evolve=True, save_fig_ecc_evolve=True, 
                        #    plot_residuals_time_evolve=True, save_fig_time_evolve=True,
                           )
    # gt.fit_to_training_set('amplitude', min_greedy_error=1e-6, save_fits_to_file=True, plot_GPR_fits=True, save_fig_GPR_fits=True, plot_residuals_ecc_evolve=True, save_fig_ecc_evolve=True, plot_residuals_time_evolve=True, save_fig_time_evolve=True)

plt.show()
# gt.fit_to_training_set('amplitude', N_basis_vecs=21, save_fits_to_file=True)