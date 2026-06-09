from tracemalloc import start

from matplotlib import axes, gridspec, path
from matplotlib.backends.backend_pdf import PdfPages

from generate_greedy_training_set import *

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Product, ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel, ExpSineSquared, DotProduct, ConstantKernel as C
import time
import traceback
import seaborn as sns
from matplotlib.lines import Line2D
from inspect import getframeinfo
from sklearn.preprocessing import StandardScaler
from fileinput import filename
from inspect import currentframe
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import ConvergenceWarning

import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec


warnings.simplefilter("ignore", ConvergenceWarning)

f = currentframe()

import plotly.io as pio
# default for script-style plotting
pio.renderers.default = "browser"

import plotly.express as px
from sklearn.decomposition import PCA

@dataclass
class GPRFitResults(Warnings):
    
    # ------------------------------------------------------------
    # WAVEFORM SYSTEMATICS
    # ------------------------------------------------------------
    
    property: str = "phase"
    time: Any = None

    f_ref: float = None
    f_lower: float = None
    phiRef: float = 0.0
    inclination: float = 0.0
    truncate_at_ISCO: bool = True
    truncate_at_tmin: bool = True
    luminosity_distance: Optional[float] = None

    # ------------------------------------------------------------
    # INPUT/ OUTPUT SPACES
    # ------------------------------------------------------------
    
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

    parameter_grid: Any = None

    # ------------------------------------------------------------
    # TRAINING SYSTEMATICS
    # ------------------------------------------------------------
    N_basis_vecs: Optional[int] = None
    min_greedy_error: Optional[float] = None

    # ------------------------------------------------------------
    # TRAINING STRUCTURE
    # ------------------------------------------------------------
    residuals: Any = None
    basis_indices: list = field(default_factory=list)
    empirical_indices: list = field(default_factory=list)
    training_set: Any = None

    # ------------------------------------------------------------
    # FULL GP RESULTS (offline use)
    # ------------------------------------------------------------
    gp_models: list = field(default_factory=list)   # ONE GP per time node
    kernels: list = field(default_factory=list)

    mean_predictions: Any = None
    confidence_95_preds: Any = None
    best_lmls: Any = None
    best_train_rmses: Any = None
    best_scores: Any = None

    # ------------------------------------------------------------
    # SCALERS (CRITICAL FOR ONLINE USE)
    # ------------------------------------------------------------
    scaler_x: Any = None
    scaler_y: Any = None


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

    def save_gpr_obj(self, prefix="GPRFitResults", directory="Straindata/GPRFitResults"):
        """Saves the GPRFitResults object to a file."""
        if directory is not None:
            os.makedirs(directory, exist_ok=True)
        filepath = self.filename(prefix=prefix, ext="pkl", directory=directory)

        with open(filepath, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Figure saving confirmation
        print(self.colored_text(f'GPRFitResults saved to {filepath}', 'blue'))
        return filepath

    def load_gpr_obj(self, prefix="GPRFitResults", directory="Straindata/GPRFitResults"):
        """Loads the saved GPRFitResults object from a file."""

        filepath = self.filename(prefix=prefix, ext="pkl", directory=directory)
        with open(filepath, 'rb') as f:
            loaded_obj = pickle.load(f)

        if not isinstance(loaded_obj, GPRFitResults):
            raise ValueError(f"Loaded object is not of type GPRFitResults. Got {type(loaded_obj)} instead.")

        # Figure loading confirmation
        print(self.colored_text(f"GPRFitResults object loaded from {filepath}", 'blue'))

        return loaded_obj
    
    def save_offline_surrogate(self, prefix="Offline_GP", directory="Straindata/Offline_GP_Models"):
        """
        Only what is needed for ONLINE inference.
        """
        if directory is not None:
            os.makedirs(directory, exist_ok=True)
        filepath = self.filename(prefix=prefix, ext="pkl", directory=directory)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        bundle = {
            "property": self.property,
            "time": self.time,

            # IMPORTANT: GP models OR kernels+training data
            "gp_models": self.gp_models,
            "kernels": self.kernels,
            "scaler_x": self.scaler_x,
            "scaler_y": self.scaler_y,

            "parameter_grid": self.parameter_grid,
        }

        with open(filepath, "wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_offline_surrogate(self, 
                               prefix="Offline_GP", 
                               directory="Straindata/Offline_GP_Models"):
        """
        Load the surrogate bundle saved by save_offline_surrogate.
        """
        filepath = self.filename(prefix=prefix, ext="pkl", directory=directory)
        with open(filepath, "rb") as f:
            bundle = pickle.load(f)

        self.property = bundle.get("property", None)
        self.time = bundle.get("time", None)
        self.gp_models = bundle.get("gp_models", None)
        self.scaler_x = bundle.get("scaler_x", None)
        self.scaler_y = bundle.get("scaler_y", None)
        self.parameter_grid = bundle.get("parameter_grid", None)


class Generate_Offline_Surrogate(Generate_TrainingSet):

    def __init__(self, time_array, 
                 ecc_ref_parameterspace=np.linspace(0.0, 0.3, num=50), 
                 mean_ano_parameterspace=np.linspace(0.0, 2*np.pi, num=50), 
                 mass_ratio_parameterspace=np.linspace(1, 20, num=50),
                 chi1_parameterspace=np.linspace(-0.995, 0.995, num=50),
                 chi2_parameterspace=np.linspace(-0.995, 0.995, num=50),
                 M_output_wfs_per_dimension=500, 
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
        # Check if property is valid and adjust settings accordingly
        self.ecc_ref_space_input = self.allowed_eccentricity_warning(ecc_ref_parameterspace)
        self.mass_ratio_space_input = self.allowed_mass_ratio_warning(mass_ratio_parameterspace)
        self.mean_ano_ref_space_input = self.allowed_mean_anomaly_warning(mean_ano_parameterspace)
        self.chi1_space_input = self.allowed_chispin_warning(chi1_parameterspace)
        self.chi2_space_input = self.allowed_chispin_warning(chi2_parameterspace)

        self.ecc_ref_parameterspace_range = min(ecc_ref_parameterspace), max(ecc_ref_parameterspace)
        self.ecc_ref_space_output = self.allowed_eccentricity_warning(np.linspace(*self.ecc_ref_parameterspace_range, M_output_wfs_per_dimension).round(4))

        self.mean_ano_ref_parameterspace_range = min(mean_ano_parameterspace), max(mean_ano_parameterspace)
        self.mean_ano_ref_space_output = self.allowed_mean_anomaly_warning(np.linspace(*self.mean_ano_ref_parameterspace_range, M_output_wfs_per_dimension).round(4))

        self.mass_ratio_parameterspace_range = min(mass_ratio_parameterspace), max(mass_ratio_parameterspace)
        self.mass_ratio_space_output = self.allowed_mass_ratio_warning(np.linspace(*self.mass_ratio_parameterspace_range, M_output_wfs_per_dimension).round(4))

        self.chi1_parameterspace_range = min(chi1_parameterspace), max(chi1_parameterspace)
        self.chi1_space_output = self.allowed_chispin_warning(np.linspace(*self.chi1_parameterspace_range, M_output_wfs_per_dimension).round(4))

        self.chi2_parameterspace_range = min(chi2_parameterspace), max(chi2_parameterspace)
        self.chi2_space_output = self.allowed_chispin_warning(np.linspace(*self.chi2_parameterspace_range, M_output_wfs_per_dimension).round(4))


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
    
    # def _gaussian_process_regression_t(
    #     self,
    #     time_node,
    #     train_obj: TrainingSetResults,
    #     optimized_kernel=None,
    #     plot_kernels=False,
    #     save_fig_kernels=False,
    #     time_coupled=False
    # ):
    #     """
    #     Gaussian Process Regression for one empirical time node.

    #     ------------------------------------------------------------------
    #     WHAT THIS FUNCTION DOES
    #     ------------------------------------------------------------------

    #     We want to learn:

    #         f(parameters) -> waveform quantity

    #     where the waveform quantity is either:
    #         - phase residual
    #         - amplitude residual

    #     at ONE empirical time node.

    #     ------------------------------------------------------------
    #     TWO MODES
    #     ------------------------------------------------------------

    #     1) time_coupled = False
    #     --------------------------------
    #     Each time node gets its own independent GP.

    #     Input dimensions:
    #         [e, l, q, chi1, chi2]

    #     This is:
    #         - simpler
    #         - faster
    #         - usually more stable
    #         - easier to optimize

    #     Recommended FIRST approach.


    #     2) time_coupled = True
    #     --------------------------------
    #     Time is added as another GP dimension.

    #     Input dimensions:
    #         [e, l, q, chi1, chi2, t]

    #     The GP then learns correlations across time.

    #     Advantages:
    #         - smoother evolution in time
    #         - fewer discontinuities between nodes

    #     Disadvantages:
    #         - MUCH harder optimization
    #         - higher dimensionality
    #         - requires more training data
    #         - more prone to over-smoothing

    #     Recommended only after the uncoupled model works well.

    #     ------------------------------------------------------------------
    #     KERNEL SEARCH STRATEGY
    #     ------------------------------------------------------------------

    #     We test MANY kernels automatically.

    #     We vary:
    #         - kernel smoothness (nu)
    #         - length scales
    #         - isotropic vs anisotropic kernels
    #         - noise levels

    #     Then we select the BEST kernel using:

    #         score = LML - alpha * train_rmse

    #     because:

    #         LML alone often prefers overly smooth kernels.

    #     Adding RMSE penalizes kernels that fit poorly.

    #     ------------------------------------------------------------------
    #     IMPORTANT KERNEL CONCEPTS
    #     ------------------------------------------------------------------

    #     Matern kernel:
    #         Controls smoothness of interpolation.

    #     nu:
    #         0.5  -> rough / jagged
    #         1.5  -> moderately smooth
    #         2.5  -> very smooth

    #     length_scale:
    #         Controls correlation distance.

    #         small:
    #             rapid variation

    #         large:
    #             smooth variation

    #     isotropic:
    #         one length scale for ALL dimensions

    #     anisotropic:
    #         separate length scale per dimension

    #         VERY important in high dimensions because:
    #             eccentricity may vary rapidly
    #             while spin dependence may vary slowly

    #     WhiteKernel:
    #         models numerical noise.

    #     ------------------------------------------------------------------
    #     HOW TO TEST EFFICIENTLY
    #     ------------------------------------------------------------------

    #     STEP 1:
    #         Start SIMPLE:

    #             time_coupled = False

    #             smoothness_params = [1.5]
    #             ls_multipliers = [1.0]
    #             noise_levels = [1e-6]

    #     STEP 2:
    #         Inspect:
    #             - train_rmse
    #             - plots
    #             - prediction smoothness

    #     STEP 3:
    #         Add anisotropic kernels.

    #         This is usually the BIGGEST improvement
    #         for multidimensional waveform spaces.

    #     STEP 4:
    #         Add more:
    #             smoothness_params
    #             ls_multipliers

    #     STEP 5:
    #         ONLY THEN try:
    #             time_coupled=True

    #     ------------------------------------------------------------------
    #     PRACTICAL ADVICE
    #     ------------------------------------------------------------------

    #     In high dimensions:

    #         anisotropic kernels > isotropic kernels

    #     almost always.

    #     Also:

    #         too many optimizer restarts
    #         can dominate runtime.

    #     Start with:
    #         n_restarts_optimizer=5

    #     before using:
    #         20

    #     ------------------------------------------------------------------
    #     """

        # # ============================================================
        # # BUILD INPUT MATRICES
        # # ============================================================

        # if time_coupled is False:

        #     # --------------------------------------------------------
        #     # STANDARD MODE:
        #     # one GP per time node
        #     # --------------------------------------------------------
        #     X = np.asarray(self.parameter_grid)
        #     X_train = X[train_obj.basis_indices]

        # else:

        #     # --------------------------------------------------------
        #     # TIME-COUPLED MODE:
        #     # append time as extra dimension
        #     # --------------------------------------------------------

        #     X_param = np.asarray(self.parameter_grid)

        #     t = np.full((X_param.shape[0], 1), self.time[time_node])

        #     X_train = np.hstack([
        #         X_param[train_obj.basis_indices],
        #         t[train_obj.basis_indices]
        #     ])

        #     X = np.hstack([
        #         X_param,
        #         t
        #     ])

        # # ============================================================
        # # TRAINING TARGETS
        # # ============================================================

        # y_train = np.squeeze(train_obj.training_set.T[time_node])

        # # ============================================================
        # # SCALE INPUTS
        # # ============================================================

        # # ------------------------------------------------------------
        # # WHY SCALE?
        # #
        # # GP kernels are VERY sensitive to feature scales.
        # #
        # # Example:
        # #
        # #     eccentricity ~ 0.01
        # #     q            ~ 10
        # #
        # # Without scaling:
        # #     q dominates distance calculations.
        # # ------------------------------------------------------------

        # scaler_x = StandardScaler()

        # X_train_scaled = scaler_x.fit_transform(X_train)

        # X_scaled = scaler_x.transform(X)

        # scaler_y = StandardScaler()

        # y_train_scaled = scaler_y.fit_transform(
        #     y_train.reshape(-1, 1)
        # ).flatten()

        # # ============================================================
        # # ESTIMATE CHARACTERISTIC LENGTH SCALE
        # # ============================================================

        # # ------------------------------------------------------------
        # # We estimate a reasonable initial kernel size from
        # # nearest-neighbor distances.
        # #
        # # This gives the optimizer a MUCH better starting point.
        # # ------------------------------------------------------------

        # nn = NearestNeighbors(n_neighbors=2)

        # nn.fit(X_train_scaled)

        # distances, indices = nn.kneighbors(X_train_scaled)

        # median_nn_distance = np.median(distances[:, 1])

        # base_ls = median_nn_distance * 2

        # print("base_ls =", base_ls)

        # # ============================================================
        # # INPUT DIMENSION
        # # ============================================================

        # dim = X_train_scaled.shape[1]

        # # ============================================================
        # # KERNEL SEARCH PARAMETERS
        # # ============================================================

        # # ------------------------------------------------------------
        # # Multipliers applied to base_ls.
        # #
        # # small:
        # #     rapid local variation
        # #
        # # large:
        # #     smoother interpolation
        # # ------------------------------------------------------------

        # # ls_multipliers = [0.3, 1.0, 3.0, 10.0]
        # # ls_multipliers = [1.0]

        # # ------------------------------------------------------------
        # # Upper optimization bounds for length scales.
        # #
        # # Prevents optimizer from going to absurdly smooth kernels.
        # # ------------------------------------------------------------

        # # ls_upper_bounds = [0.5, 1.0, 2.0, 5.0]

        # # ------------------------------------------------------------
        # # Matern smoothness values.
        # #
        # # lower:
        # #     rougher functions
        # #
        # # higher:
        # #     smoother functions
        # # ------------------------------------------------------------

        # smoothness_params = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0]
        # # smoothness_params = [1.5]

        # kernels = []

        # # ============================================================
        # # 1. ISOTROPIC MATERN KERNELS
        # # ============================================================

        # # ------------------------------------------------------------
        # # One single length scale shared across ALL dimensions.
        # #
        # # Simpler.
        # # Faster.
        # # Often insufficient in high dimensions.
        # # ------------------------------------------------------------

        # for nu in smoothness_params:

        #     kernel = Matern(
        #         length_scale=base_ls,
        #         length_scale_bounds=(0.1 * base_ls, 10 * base_ls),
        #         nu=nu
        #     )

        #     kernels.append({
        #         "kernel": kernel,
        #         "type": "isotropic",
        #         "nu": nu,
        #         "base_ls": base_ls
        #     })

        # # ============================================================
        # # 2. ANISOTROPIC MATERN KERNELS
        # # ============================================================

        # # ------------------------------------------------------------
        # # Separate length scale per dimension.
        # #
        # # MUCH more powerful for:
        # #     [e, l, q, chi1, chi2]
        # #
        # # because every parameter may vary differently.
        # # ------------------------------------------------------------

        # for nu in smoothness_params:

        #     kernel = Matern(
        #         length_scale=np.ones(dim) * base_ls,
        #         length_scale_bounds=(0.1 * base_ls, 10 * base_ls),
        #         nu=nu
        #     )

        #     kernels.append({
        #         "kernel": kernel,
        #         "type": "anisotropic",
        #         "nu": nu,
        #         "base_ls": base_ls
        #     })

        # # ============================================================
        # # 3. ADD WHITE NOISE TERMS
        # # ============================================================

        # # ------------------------------------------------------------
        # # Models numerical noise / imperfect data.
        # #
        # # Prevents overfitting.
        # # ------------------------------------------------------------

        # # noise_levels = [1e-8, 1e-6, 1e-4]
        # noise_levels = [1e-6]

        # extra_kernels = []

        # for entry in kernels:

        #     for noise in noise_levels:

        #         noisy_kernel = (
        #             entry["kernel"]
        #             + WhiteKernel(
        #                 noise_level=noise,
        #                 noise_level_bounds=(1e-10, 1e-1)
        #             )
        #         )

        #         extra_kernels.append({
        #             "kernel": noisy_kernel,
        #             "type": entry["type"] + "_noise",
        #             "nu": entry["nu"],
        #             "base_ls": entry["base_ls"],
        #             "noise": noise
        #         })

        # kernels.extend(extra_kernels)

        # print(f"Generated {len(kernels)} kernels")

        # # ------------------------------------------------------------
        # # Warm-start from previous time node kernel
        # # ------------------------------------------------------------

        # if optimized_kernel is not None:

        #     kernels.insert(
        #         0,
        #         {
        #             "kernel": optimized_kernel,
        #             "type": "warm_start",
        #             "nu": None,
        #             "base_ls": None
        #         }
        #     )

        # # ============================================================
        # # STORAGE
        # # ============================================================

        # y_predict_kernels = []
        # y_predict_std_kernels = []

        # kernel_diagnostics = []

        # kernel_fit_times = []

        # best_score = -np.inf
        # best_lml = -np.inf
        # best_train_rmse = np.inf

        # # ------------------------------------------------------------
        # # IMPORTANT:
        # #
        # # LML alone tends to prefer over-smoothed kernels.
        # #
        # # We therefore penalize high training error.
        # #
        # # Larger alpha:
        # #     prioritize fit quality more
        # #
        # # Smaller alpha:
        # #     prioritize smoothness / generalization more
        # # ------------------------------------------------------------

        # alpha = 1.0 / np.std(y_train_scaled)

        

        # # ============================================================
        # # KERNEL SEARCH
        # # ============================================================

        # for entry in kernels:

        #     try:

        #         start = time.time()

        #         kernel_label = {
        #             "isotropic": "Iso",
        #             "anisotropic": "Aniso",
        #             "isotropic_noise": "Iso+W",
        #             "anisotropic_noise": "Aniso+W",
        #             "warm_start": "Warm"
        #         }[entry["type"]]

        #         kernel_label += f" ν={entry['nu']}"

        #         if "noise" in entry:
        #             kernel_label += f", noise={entry['noise']:.0e}"

        #         start_kernel_time = time.time()
        #         gaussian_process = GaussianProcessRegressor(
        #             kernel=entry["kernel"],
        #             n_restarts_optimizer=5,
        #             random_state=42
        #         )

        #         # ----------------------------------------------------
        #         # TRAIN GP
        #         # ----------------------------------------------------

        #         gaussian_process.fit(
        #             X_train_scaled,
        #             y_train_scaled
        #         )

        #         optimized_kernel = gaussian_process.kernel_

        #         end_kernel_time = time.time()
        #         kernel_fit_time = end_kernel_time - start_kernel_time
        #         kernel_fit_times.append(kernel_fit_time)

        #         end = time.time()

        #         # ----------------------------------------------------
        #         # LOG MARGINAL LIKELIHOOD
        #         # ----------------------------------------------------

        #         lml = gaussian_process.log_marginal_likelihood_value_

        #         # ----------------------------------------------------
        #         # TRAINING ERROR
        #         # ----------------------------------------------------

        #         y_pred_train, std_train = gaussian_process.predict(
        #             X_train_scaled,
        #             return_std=True
        #         )

        #         train_rmse = np.sqrt(
        #             np.mean(
        #                 (y_pred_train - y_train_scaled) ** 2
        #             )
        #         )

        #         # ----------------------------------------------------
        #         # COMBINED SCORE
        #         # ----------------------------------------------------

        #         score = lml - alpha * train_rmse

        #         # ----------------------------------------------------
        #         # PREDICT FULL GRID
        #         # ----------------------------------------------------

        #         y_predict_scaled, std_prediction_scaled = (
        #             gaussian_process.predict(
        #                 X_scaled,
        #                 return_std=True
        #             )
        #         )

        #         y_predict = scaler_y.inverse_transform(
        #             y_predict_scaled.reshape(-1, 1)
        #         ).flatten()

        #         y_predict_std = (
        #             std_prediction_scaled
        #             * scaler_y.scale_[0]
        #         )

        #         y_predict_kernels.append(y_predict)

        #         y_predict_std_kernels.append(y_predict_std)

                
        #         kernel_diagnostics.append({
        #             "kernel": gaussian_process.kernel_,
        #             "score": score,
        #             "lml": lml,
        #             "rmse": train_rmse,
        #             "label": kernel_label,
        #         })

        #         self._diagnose_gpr_issues(
        #             gaussian_process,
        #             X_train_scaled,
        #             y_train_scaled
        #         )

        #         # ----------------------------------------------------
        #         # KEEP BEST KERNEL
        #         # ----------------------------------------------------

        #         if score > best_score:

        #             best_score = score

        #             best_lml = lml

        #             best_train_rmse = train_rmse

        #             best_optimized_kernel = gaussian_process.kernel_

        #             best_gaussian_process = gaussian_process

        #             # gp prediction at training locations
        #             best_y_train_pred = scaler_y.inverse_transform(
        #                 y_pred_train.reshape(-1,1)
        #             ).flatten()

        #             # uncertainty at training locations
        #             best_y_train_std = (
        #                 std_train * scaler_y.scale_[0]
        #             )

        #             # gp prediction on entire predicted grid 
        #             best_y_predict = y_predict

        #             # uncertainty on entire predicted grid
        #             best_y_predict_std = y_predict_std

        #     except Exception as e:

        #         print(
        #             self.colored_text(
        #                 f"GPR failed for kernel {entry}: {e}",
        #                 "red"
        #             )
        #         )

        #         traceback.print_exc()

        #         continue

        # # ============================================================
        # # FINAL REPORT
        # # ============================================================

        # print(
        #     self.colored_text(
        #         f"Best kernel = {best_optimized_kernel}\n"
        #         f"LML = {best_lml:.4f}\n"
        #         f"RMSE = {best_train_rmse:.6e}\n"
        #         f"Score = {best_score:.4f}",
        #         "green"
        #     )
        # )

        # # ============================================================
        # # BUILD RESULT DICTIONARY
        # # ============================================================

        # result_dict = {

        #     # --------------------------------------------------------
        #     # Trained GP
        #     # --------------------------------------------------------

        #     "gp": best_gaussian_process,

        #     # --------------------------------------------------------
        #     # Optimized kernel
        #     # --------------------------------------------------------

        #     "kernel": best_optimized_kernel,

        #     # --------------------------------------------------------
        #     # Scalers
        #     # --------------------------------------------------------

        #     "scaler_x": scaler_x,
        #     "scaler_y": scaler_y,

        #     # --------------------------------------------------------
        #     # Predictions
        #     # --------------------------------------------------------

        #     "mean_prediction": best_y_predict,

        #     "confidence_95": np.array([
        #         best_y_predict - 1.96 * best_y_predict_std,
        #         best_y_predict + 1.96 * best_y_predict_std
        #     ]),

        #     "std_prediction": best_y_predict_std,

        #     "train_prediction": best_y_train_pred,
        #     "train_std": best_y_train_std,

        #     # --------------------------------------------------------
        #     # Diagnostics
        #     # --------------------------------------------------------

        #     "lml": best_lml,
        #     "train_rmse": best_train_rmse,
        #     "score": best_score,

        #     # --------------------------------------------------------
        #     # Metadata
        #     # --------------------------------------------------------

        #     "time_node": time_node,
        #     "time_value": self.time[time_node],

        #     "basis_indices": train_obj.basis_indices,

        #     "parameter_dim": dim,

        #     "time_coupled": time_coupled
        # }

    def _gaussian_process_regression_t(
        self,
        time_node,
        train_obj: TrainingSetResults,
        optimized_kernel=None,
        plot_kernel_predictions=False,
        plot_kernel_errors=False,
        save_fig_kernels=False,
        time_coupled=False,
        screening=True,
        refinement=True
    ):

        """
        Gaussian Process Regression for one empirical time node.
        (comments unchanged)
        """

        # ============================================================
        # BUILD INPUT MATRICES
        # ============================================================
        train_obj.load_residuals()

        if time_coupled is False:

            X = np.asarray(train_obj.parameter_grid)
            X_train = X[train_obj.basis_indices]

        else:

            X_param = np.asarray(train_obj.parameter_grid)

            t = np.full((X_param.shape[0], 1), self.time[train_obj.empirical_indices[time_node]])

            X_train = np.hstack([
                X_param[train_obj.basis_indices],
                t[train_obj.basis_indices]
            ])

            X = np.hstack([
                X_param,
                t
            ])

        # ============================================================
        # TRAINING TARGETS
        # ============================================================

        y_train = np.squeeze(train_obj.training_set.T[time_node])

        # ============================================================
        # SCALE INPUTS
        # ============================================================

        scaler_x = StandardScaler()
        X_train_scaled = scaler_x.fit_transform(X_train)
        X_scaled = scaler_x.transform(X)

        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(
            y_train.reshape(-1, 1)
        ).flatten()

        # ============================================================
        # ESTIMATE CHARACTERISTIC LENGTH SCALE
        # ============================================================

        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(X_train_scaled)

        distances, _ = nn.kneighbors(X_train_scaled)

        median_nn_distance = np.median(distances[:, 1])
        base_ls = median_nn_distance * 2

        print("base_ls =", base_ls)

        # ============================================================
        # INPUT DIMENSION
        # ============================================================

        dim = X_train_scaled.shape[1]

        # ============================================================
        # KERNEL SEARCH PARAMETERS
        # ============================================================

        smoothness_params = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0]

        kernels = []

        # ============================================================
        # 1. ISOTROPIC MATERN KERNELS
        # ============================================================

        for nu in smoothness_params:

            kernel = Matern(
                length_scale=base_ls,
                length_scale_bounds=(0.1 * base_ls, 10 * base_ls),
                nu=nu
            )

            kernels.append({
                "kernel": kernel,
                "type": "isotropic",
                "nu": nu,
                "base_ls": base_ls
            })

        # ============================================================
        # 2. ANISOTROPIC MATERN KERNELS
        # ============================================================

        for nu in smoothness_params:

            kernel = Matern(
                length_scale=np.ones(dim) * base_ls,
                length_scale_bounds=(0.1 * base_ls, 10 * base_ls),
                nu=nu
            )

            kernels.append({
                "kernel": kernel,
                "type": "anisotropic",
                "nu": nu,
                "base_ls": base_ls
            })

        # ============================================================
        # 3. ADD WHITE NOISE TERMS
        # ============================================================

        noise_levels = [1e-6]

        extra_kernels = []

        for entry in kernels:

            for noise in noise_levels:

                noisy_kernel = (
                    entry["kernel"]
                    + WhiteKernel(
                        noise_level=noise,
                        noise_level_bounds=(1e-10, 1e-1)
                    )
                )

                extra_kernels.append({
                    "kernel": noisy_kernel,
                    "type": entry["type"] + "_noise",
                    "nu": entry["nu"],
                    "base_ls": entry["base_ls"],
                    "noise": noise
                })

        kernels.extend(extra_kernels)

        print(f"Generated {len(kernels)} kernels")

        # ============================================================
        # WARM START
        # ============================================================

        if optimized_kernel is not None:

            kernels.insert(
                0,
                {
                    "kernel": optimized_kernel,
                    "type": "warm_start",
                    "nu": None,
                    "base_ls": None
                }
            )

        # ============================================================
        # STORAGE
        # ============================================================

        y_predict_kernels = []
        y_predict_std_kernels = []
        kernel_diagnostics = []

        best_score = -np.inf
        best_lml = -np.inf
        best_train_rmse = np.inf

        lml_list = []
        rmse_list = []

        # ============================================================
        # MODE CONTROL
        # ============================================================

        if screening and not refinement:
            mode = "screening"
        elif refinement and not screening:
            mode = "refinement"
        elif screening and refinement:
            mode = "full"
        else:
            raise ValueError("Enable screening or refinement")

        # ============================================================
        # KERNEL SEARCH
        # ============================================================

        for i, entry in enumerate(kernels):

            success = False

            try:

                # ----------------------------------------------------
                # restart policy
                # ----------------------------------------------------
                if mode == "screening":
                    n_restarts = 2
                elif mode == "refinement":
                    n_restarts = 8
                else:
                    n_restarts = 5

                # ----------------------------------------------------
                # label
                # ----------------------------------------------------
                kernel_label = {
                    "isotropic": "Iso",
                    "anisotropic": "Aniso",
                    "isotropic_noise": "Iso+W",
                    "anisotropic_noise": "Aniso+W",
                    "warm_start": "Warm"
                }[entry["type"]]

                kernel_label += f" ν={entry['nu']}"
                if "noise" in entry:
                    kernel_label += f", noise={entry['noise']:.0e}"

                # ----------------------------------------------------
                # timing
                # ----------------------------------------------------
                t0 = time.perf_counter()

                gp = GaussianProcessRegressor(
                    kernel=entry["kernel"],
                    n_restarts_optimizer=n_restarts,
                    random_state=42
                )

                gp.fit(X_train_scaled, y_train_scaled)

                kernel_fit_time = time.perf_counter() - t0

                # ----------------------------------------------------
                # metrics
                # ----------------------------------------------------
                lml = gp.log_marginal_likelihood_value_

                y_pred_train, std_train = gp.predict(
                    X_train_scaled,
                    return_std=True
                )

                train_rmse = np.sqrt(
                    np.mean((y_pred_train - y_train_scaled) ** 2)
                )

                lml_list.append(lml)
                rmse_list.append(train_rmse)

                # ----------------------------------------------------
                # screening prune
                # ----------------------------------------------------
                if mode == "screening" and len(lml_list) > 5:

                    if lml < np.mean(lml_list) - 5 * np.std(lml_list):
                        continue

                    if train_rmse > np.mean(rmse_list) + 3 * np.std(rmse_list):
                        continue

                # ----------------------------------------------------
                # prediction
                # ----------------------------------------------------
                y_pred_scaled, std_pred_scaled = gp.predict(
                    X_scaled,
                    return_std=True
                )

                y_pred = scaler_y.inverse_transform(
                    y_pred_scaled.reshape(-1, 1)
                ).flatten()

                y_pred_std = std_pred_scaled * scaler_y.scale_[0]

                y_predict_kernels.append(y_pred)
                y_predict_std_kernels.append(y_pred_std)

                success = True


            except Exception as e:

                print(
                    self.colored_text(
                        f"GPR failed for kernel {entry}: {e}",
                        "red"
                    )
                )

                traceback.print_exc()

                continue


            # ============================================================
            # ONLY STORE IF SUCCESSFUL
            # ============================================================

            if success:

                kernel_diagnostics.append({
                    # time node
                    "time_node": time_node,

                    # fitted kernel
                    "kernel": gp.kernel_,

                    # raw metrics
                    "lml": lml,
                    "rmse": train_rmse,

                    # metadata
                    "label": kernel_label,
                    "fit_time": kernel_fit_time,

                    # store objects needed later
                    "gp": gp,

                    # training predictions
                    "y_train_pred": y_pred_train,
                    "std_train": std_train,

                    # full-grid predictions
                    "y_pred": y_pred,
                    "y_pred_std": y_pred_std,
                })


        # ============================================================
        # COMPUTE FINAL SCORES
        # ============================================================

        all_lmls = np.array([
            d["lml"] for d in kernel_diagnostics
        ])

        all_rmses = np.array([
            d["rmse"] for d in kernel_diagnostics
        ])

        mean_lml = np.mean(all_lmls)
        std_lml = np.std(all_lmls) + 1e-12

        mean_rmse = np.mean(all_rmses)
        std_rmse = np.std(all_rmses) + 1e-12

        for d in kernel_diagnostics:

            # Use Normalized LML - Normalized RMSE as final score
            lml_normalized = (d["lml"] - mean_lml
            ) / std_lml

            rmse_normalized = (
                d["rmse"] - mean_rmse
            ) / std_rmse

            d["score"] = (
                lml_normalized
                - 0.5 * rmse_normalized
            )

        best_idx = np.argmax(
            [d["score"] for d in kernel_diagnostics]
        )

        best_info = kernel_diagnostics[best_idx]

        best_score = best_info["score"]
        best_lml = best_info["lml"]
        best_train_rmse = best_info["rmse"]

        best_gaussian_process = best_info["gp"]
        best_optimized_kernel = best_info["kernel"]

        best_y_train_pred = scaler_y.inverse_transform(
            best_info["y_train_pred"].reshape(-1, 1)
        ).flatten()

        best_y_train_std = (
            best_info["std_train"]
            * scaler_y.scale_[0]
        )

        best_y_predict = best_info["y_pred"]
        best_y_predict_std = best_info["y_pred_std"]

        # ============================================================
        # FINAL REPORT
        # ============================================================

        print(
            self.colored_text(
                f"Best kernel = {best_optimized_kernel}\n"
                f"LML = {best_lml:.4f}\n"
                f"RMSE = {best_train_rmse:.6e}\n"
                f"Score = {best_score:.4f}",
                "green"
            )
        )

        # ============================================================
        # RESULT DICTIONARY (UNCHANGED STRUCTURE)
        # ============================================================

        result_dict = {
            "gp": best_gaussian_process,
            "kernel": best_optimized_kernel,
            "scaler_x": scaler_x,
            "scaler_y": scaler_y,
            "mean_prediction": best_y_predict,
            "confidence_95": np.array([
                best_y_predict - 1.96 * best_y_predict_std,
                best_y_predict + 1.96 * best_y_predict_std
            ]),
            "std_prediction": best_y_predict_std,
            "train_prediction": best_y_train_pred,
            "train_std": best_y_train_std,
            "lml": best_lml,
            "train_rmse": best_train_rmse,
            "score": best_score,
            "time_node": time_node,
            "time_value": self.time[time_node],
            "basis_indices": train_obj.basis_indices,
            "parameter_dim": dim,
            "time_coupled": time_coupled
        }


        # ============================================================
        # PLOT KERNEL OPTIMIZATION RESULTS
        # ============================================================


        if plot_kernel_errors:

            # Plot scores, rmse, lml, fit times for all kernels
            self.plot_kernel_diagnostics(
                kernel_diagnostics,
                train_obj=train_obj,
                best_score=best_score,
                save_fig=save_fig_kernels,
            )

            # Plot error heatmaps for top k kernels
            self.plot_best_kernel_error_heatmaps(
                kernel_diagnostics=kernel_diagnostics,
                train_obj=train_obj,
                time_node=time_node,
                n_q_slices=5,
                top_k=3,
            )

        if plot_kernel_predictions:

            # ------------------------------------------------------------
            # Find worst kernels (lowest score → highest)
            # ------------------------------------------------------------

            worst_indices = np.argsort(
                [d["score"] for d in kernel_diagnostics]
            )[:5]

            worst_indices = worst_indices[np.argsort(
                [kernel_diagnostics[i]["score"] for i in worst_indices]
            )]

            # ------------------------------------------------------------
            # TRAINING ORDERING
            # ------------------------------------------------------------

            order = np.argsort(train_obj.basis_indices)

            idx_train = np.array(train_obj.basis_indices)
            y_train_plot = y_train[order]

            # ------------------------------------------------------------
            # TRUTH ON FULL GRID
            # ------------------------------------------------------------

            truth = train_obj.residuals[:, train_obj.empirical_indices[time_node]]

            # ------------------------------------------------------------
            # DIAGNOSTICS
            # ------------------------------------------------------------

            full_rmse = np.sqrt(
                np.mean((best_y_predict - truth) ** 2)
            )

            train_rmse_check = np.sqrt(
                np.mean(
                    (
                        best_y_predict[idx_train]
                        - truth[idx_train]
                    ) ** 2
                )
            )

            # ------------------------------------------------------------
            # PLOT SETUP
            # ------------------------------------------------------------

            fig_kernel_comparison, (ax1, ax2) = plt.subplots(
                2, 1,
                figsize=(12, 8),
                gridspec_kw={"height_ratios": [3, 1]},
                sharex=True
            )

            x_axis = np.arange(len(best_y_predict))

            # ------------------------------------------------------------
            # TRUE RESIDUALS (FULL GRID)
            # ------------------------------------------------------------

            ax1.plot(
                x_axis,
                truth,
                color="blue",
                linewidth=2.5,
                label="True residuals (full grid)"
            )

            # ------------------------------------------------------------
            # TRAINING VALUES
            # ------------------------------------------------------------

            ax1.scatter(
                idx_train,
                truth[idx_train],
                color="red",
                s=40,
                zorder=10,
                label="Training points"
            )

            # ------------------------------------------------------------
            # WORST KERNELS
            # ------------------------------------------------------------
            
            for rank, idx in enumerate(worst_indices):

                y_pred = y_predict_kernels[idx]
                y_std = y_predict_std_kernels[idx]
                info = kernel_diagnostics[idx]

                ax1.plot(
                    x_axis,
                    y_pred,
                    linewidth=1,
                    alpha=0.6,
                    label=(
                        f"Worst {rank+1}: "
                        f"{info['label']} "
                        f"| score={info['score']:.2e}"
                    )
                )

                ax1.fill_between(
                    x_axis,
                    y_pred - 1.96 * y_std,
                    y_pred + 1.96 * y_std,
                    alpha=0.08
                )

                error = y_pred - truth
                ax2.plot(x_axis, error, label=f"Worst {rank+1}: {info['label']}")


            # ------------------------------------------------------------
            # BEST MODEL
            # ------------------------------------------------------------

            best_idx = int(np.argmax(
                [d["score"] for d in kernel_diagnostics]
            ))

            best_info = kernel_diagnostics[best_idx]

            ax1.plot(
                x_axis,
                best_y_predict,
                color="black",
                linewidth=2.5,
                label=(
                    f"Best: {best_info['label']} "
                    f"| score={best_info['score']:.2e}"
                )
            )

            ax1.fill_between(
                x_axis,
                best_y_predict - 1.96 * best_y_predict_std,
                best_y_predict + 1.96 * best_y_predict_std,
                color="black",
                alpha=0.12
            )

            # ------------------------------------------------------------
            # GP PREDICTION AT TRAINING LOCATIONS
            # ------------------------------------------------------------

            ax1.scatter(
                idx_train,
                best_y_predict[idx_train],  # ← Use this instead!
                marker="x",
                color="limegreen",
                s=80,
                linewidths=2,
                zorder=20,
                label="GP prediction @ training points"
            )

            error_best = best_y_predict - truth
            ax2.plot(x_axis, error_best, label=f"Best: {best_info['label']}", color="limegreen", linewidth=2.5)

            # ------------------------------------------------------------
            # LABELS
            # ------------------------------------------------------------

            ax1.set_ylabel("Residual")
            ax2.set_ylabel("Error")
            ax2.set_xlabel("Parameter grid index")

            ax1.legend(fontsize=8)
            ax2.legend(fontsize=8)

            ax1.set_title(
                f"GPR kernel comparison\n"
                f"node={time_node}\n"
                f"full RMSE={full_rmse:.3e}, train RMSE={train_rmse_check:.3e}"
            )

            plt.tight_layout()

            

            # ------------------------------------------------------------
            # SAVE
            # ------------------------------------------------------------

            if save_fig_kernels:
                figname = train_obj.figname(prefix="gp_kernel_comparison",
                                    directory="Images/Gaussian_kernels/Kernel_comparison",
                                  include_extra=f"T{time_node}")
                fig_kernel_comparison.savefig(figname, dpi=300)


            # ------------------------------------------------------------
            # ------------------------------------------------------------

            fig_parity, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(6, 8),
                gridspec_kw={"height_ratios": [3, 1]},
                sharex=False
            )

            # ============================================================
            # PARITY PLOT (TOP)
            # ============================================================

            y_true = y_train
            y_pred = result_dict["train_prediction"]

            ax1.scatter(
                y_true,
                y_pred,
                s=10,
                alpha=0.6
            )

            # diagonal reference line
            minv = min(y_true.min(), y_pred.min())
            maxv = max(y_true.max(), y_pred.max())

            ax1.plot([minv, maxv], [minv, maxv], 'k--', linewidth=1)

            ax1.set_xlabel("True training values (physical)")
            ax1.set_ylabel("GP prediction (physical)")
            ax1.grid(True)

            ax1.set_title(
                f"GP Fit for best kernel T{time_node}: {best_info['label']}\n"
                f"RMSE = {result_dict['train_rmse']:.3e} | lml = {best_info['lml']:.3e} | score = {best_info['score']:.3e}"
            )

            # ============================================================
            # RESIDUAL PLOT (BOTTOM)
            # ============================================================

            residuals = y_pred - y_true

            ax2.bar(
                np.arange(len(residuals)),
                residuals,
                color="steelblue",
                alpha=0.8
            )

            ax2.axhline(0, color="black", linewidth=1)

            ax2.set_xlabel("Training sample index")
            ax2.set_ylabel("Residual (pred - true)")
            ax2.grid(True)

            # optional: tighten view if needed
            ylim = np.percentile(np.abs(residuals), 95)
            ax2.set_ylim(-ylim, ylim)

            plt.tight_layout()

            # ------------------------------------------------------------
            # SAVE
            # ------------------------------------------------------------

            if save_fig_kernels:
                figname = train_obj.figname(prefix="gp_parity_residual",
                                  directory="Images/Gaussian_kernels/Parity_check",
                                  include_extra=f"T{time_node}")
                fig_parity.savefig(figname, dpi=300)

        # ============================================================
        # RETURN
        # ============================================================

        return result_dict


    def plot_kernel_diagnostics(
        self,
        kernel_diagnostics,
        train_obj: TrainingSetResults,
        best_score=None,
        save_fig=False
    ):
        """
        KERNEL DIAGNOSTICS PLOTTING MODES

        This function can display kernel performance in two different ways:

        ----------------------------------------------------------------------
        1. RAW MODE ("raw")
        ----------------------------------------------------------------------

        In raw mode, the plot shows the actual values produced by the model:

        - Log Marginal Likelihood (LML)
        - Train RMSE (prediction error)
        - Score (combined selection metric)

        These values are shown exactly as computed by the Gaussian Process.

        What this means:
        - LML is the true model evidence from the GP
        - RMSE is the actual prediction error on training data
        - Score is computed directly from these raw values

        Use this mode when:
        - You want physical or numerical interpretation
        - You are debugging model behavior
        - You want to understand absolute performance

        ----------------------------------------------------------------------

        2. Z-SCORE MODE ("zscore")
        ----------------------------------------------------------------------

        In z-score mode, values are transformed into a standardized scale
        so they can be compared fairly across different magnitudes.

        A z-score means:
            how far a value is from the average, measured in standard deviations.

        For example:
            z = 0   → exactly average performance
            z = +2  → much better than average
            z = -2  → much worse than average

        In this mode:

        - LML is converted into lml_z
        - RMSE is converted into rmse_z
        - Score is computed using these normalized values

        This makes different metrics comparable on the same scale.

        Use this mode when:
        - You are selecting the best kernel
        - You want fair comparison between metrics with different units
        - You care about ranking rather than physical meaning

        ----------------------------------------------------------------------

        SUMMARY
        ----------------------------------------------------------------------

        - "raw" mode = actual physical / numerical values
        - "zscore" mode = normalized values showing relative performance
        """

        labels = [k["label"] for k in kernel_diagnostics]

        x = np.arange(len(labels))

        # ============================================================
        # SELECT DATA MODE
        # ============================================================

        lmls = np.array([k["lml"] for k in kernel_diagnostics])
        rmses = np.array([k["rmse"] for k in kernel_diagnostics])
        scores = np.array([k["score"] for k in kernel_diagnostics])

        lml_label = "Log Marginal Likelihood"
        rmse_label = "Train RMSE"
        score_label = "Score (Norm[LML] - 0.5 * Norm[RMSE])"
                
        kernel_fit_times = np.array([k["fit_time"] for k in kernel_diagnostics])

        n_plots = 4

        fig_kernel_scores, axes = plt.subplots(
            n_plots,
            1,
            figsize=(14, 4.5 * n_plots),
            sharex=True,
            constrained_layout=True
        )

        # ============================================================
        # 1. LML
        # ============================================================
        ax = axes[0]
        ax.plot(x, lmls, marker="o", color="tab:blue")
        ax.set_ylabel(lml_label)
        ax.set_title(f"Kernel diagnostics T{kernel_diagnostics[0]['time_node']}")

        ax.axhline(np.mean(lmls), linestyle="--", color="gray", label="mean")
        ax.axhline(np.max(lmls), linestyle=":", color="green", label="best")

        ax.legend()

        # ============================================================
        # 2. RMSE
        # ============================================================
        ax = axes[1]
        ax.plot(x, rmses, marker="o", color="tab:orange")
        ax.set_ylabel(rmse_label)

        ax.axhline(np.mean(rmses), linestyle="--", color="gray", label="mean")
        ax.axhline(np.min(rmses), linestyle=":", color="green", label="best")

        ax.legend()

        # ============================================================
        # 3. SCORE
        # ============================================================
        ax = axes[2]
        ax.plot(x, scores, marker="o", color="tab:purple")
        ax.set_ylabel(score_label)

        ax.axhline(np.mean(scores), linestyle="--", color="gray", label="mean")
        ax.axhline(np.max(scores), linestyle=":", color="green", label="best")

        if best_score is not None:
            ax.axhline(best_score, linestyle="--", color="black", label="selected best")

        ax.legend()

        # ============================================================
        # 4. FIT TIME
        # ============================================================
        ax = axes[3]

        times = kernel_fit_times

        ax.plot(x, times, marker="o", color="tab:red")
        ax.set_ylabel("Fit time (s)")
        ax.set_xlabel("Kernel")

        ax.set_yscale("log")

        ax.axhline(np.median(times), linestyle="--", color="gray", label="median")
        ax.axhline(np.min(times), linestyle=":", color="green", label="fastest")

        ax.legend()

        # ============================================================
        # X LABELS
        # ============================================================
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(labels, rotation=45, ha="right")

        plt.tight_layout()

        if save_fig:
                figname = train_obj.figname(
                    prefix="kernel_diagnostics",
                    ext="png",
                    directory="Images/Gaussian_kernels/Diagnostics",
                    include_greedy=True,
                    include_extra=f"node={kernel_diagnostics[0]['time_node']}"
                )
                fig_kernel_scores.savefig(figname, dpi=300)

    def plot_best_kernel_error_heatmaps(
    self,
    kernel_diagnostics,
    time_node,
    train_obj,
    n_q_slices=3,
    top_k=2
    ):
        """
        Heatmaps of relative prediction error for best kernels.

        Layout:
        - rows = q-slices
        - columns = kernels
        - 1 colorbar per row (no overlap)
        """

        train_obj.load_residuals()

        # ============================================================
        # DATA
        # ============================================================
        X = np.asarray(self.parameter_grid)
        y_true = np.asarray(train_obj.residuals[:, time_node])

        ecc = X[:, 0]
        l = X[:, 1]
        q = X[:, 2]

        eps = 1e-12

        # ============================================================
        # BEST KERNELS
        # ============================================================
        best_indices = np.argsort(
            [d["score"] for d in kernel_diagnostics]
        )[-top_k:][::-1]

        best_kernels = [kernel_diagnostics[i] for i in best_indices]

        # ============================================================
        # Q SLICES
        # ============================================================
        q_vals = np.unique(q)
        if n_q_slices > len(q_vals):
            n_q_slices = len(q_vals)
        q_indices = np.linspace(0, len(q_vals) - 1, n_q_slices).astype(int)
        q_slices = q_vals[q_indices]

        print(f"Selected q slices: {q_slices}")

        # ============================================================
        # OUTPUT FILE
        # ============================================================
        filepath = train_obj.filename(
            prefix="kernel_error_heatmaps",
            ext="pdf",
            directory="Images/Gaussian_kernels/Heatmaps",
            include_greedy=True,
            include_extra=f"node={time_node}"
        )

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        pdf = PdfPages(filepath)

        n_kernels = len(best_kernels)
        n_q = len(q_slices)

        # ============================================================
        # FIGURE + GRID SPEC
        # ============================================================
        fig = plt.figure(
            figsize=(5 * n_kernels + 1.2, 4 * n_q)
        )

        gs = gridspec.GridSpec(
            n_q,
            n_kernels + 1,  # extra column for colorbar
            width_ratios=[1] * n_kernels + [0.05],
            wspace=0.25,
            hspace=0.35
        )

        axes = np.empty((n_q, n_kernels), dtype=object)
        cbar_axes = []

        for qi in range(n_q):
            for k_id in range(n_kernels):
                axes[qi, k_id] = fig.add_subplot(gs[qi, k_id])

            cbar_axes.append(fig.add_subplot(gs[qi, -1]))

        # ============================================================
        # LOOP OVER Q SLICES
        # ============================================================
        for qi, qv in enumerate(q_slices):

            mask_q = np.isclose(q, qv)

            # --------------------------------------------------------
            # compute row normalization (shared across kernels)
            # --------------------------------------------------------
            all_errors = []

            for k in best_kernels:
                y_pred = k["y_pred"]
                rel_error = (y_pred - y_true) / (np.abs(y_true) + eps)
                all_errors.append(rel_error[mask_q])

            all_errors = np.concatenate(all_errors)
            vmin, vmax = np.nanmin(all_errors), np.nanmax(all_errors)
            norm = Normalize(vmin=vmin, vmax=vmax)

            # --------------------------------------------------------
            # LOOP KERNELS
            # --------------------------------------------------------
            for k_id, k in enumerate(best_kernels):

                ax = axes[qi, k_id]

                y_pred = k["y_pred"]
                rel_error = (y_pred - y_true) / (np.abs(y_true) + eps)

                e_vals = ecc[mask_q]
                l_vals = l[mask_q]
                err_vals = rel_error[mask_q]

                e_unique = np.unique(e_vals)
                l_unique = np.unique(l_vals)

                Z = np.full((len(e_unique), len(l_unique)), np.nan)

                for i in range(len(e_vals)):
                    ei = np.where(e_unique == e_vals[i])[0][0]
                    li = np.where(l_unique == l_vals[i])[0][0]
                    Z[ei, li] = err_vals[i]

                im = ax.imshow(
                    Z,
                    aspect="auto",
                    origin="lower",
                    norm=norm,
                    extent=[
                        l_unique.min() if len(l_unique) > 1 else l_unique.min() - 1e-3,
                        l_unique.max() if len(l_unique) > 1 else l_unique.max() + 1e-3,
                        e_unique.min() if len(e_unique) > 1 else e_unique.min() - 1e-3,
                        e_unique.max() if len(e_unique) > 1 else e_unique.max() + 1e-3
                    ],
                )

                ax.set_title(f"k={kernel_diagnostics[k_id]['label']}, q={qv:.3g}")

                ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

                if qi == n_q - 1:
                    ax.set_xlabel("l")
                else:
                    ax.set_xticklabels([])

                if k_id == 0:
                    ax.set_ylabel("e")
                else:
                    ax.set_yticklabels([])

            # ========================================================
            # COLORBAR (PER ROW, NO OVERLAP)
            # ========================================================
            cbar = fig.colorbar(
                axes[qi, 0].images[0],
                cax=cbar_axes[qi]
            )
            cbar.set_label("Relative error")

        # ============================================================
        # FINALIZE
        # ============================================================
        fig.tight_layout()

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        pdf.close()

        print(self.colored_text(f"Saved kernel error PDF → {filepath}", "blue"))

    # def _gaussian_process_regression(self, time_node, training_set, optimized_kernel=None, plot_kernels=False, save_fig_kernels=False):
    #     # Extract X and training data
    #     X = self.ecc_ref_space_output[:, np.newaxis]
    #     X_train = np.array(self.best_rep_parameters).reshape(-1, 1)
    #     y_train = np.squeeze(training_set.T[time_node])

    #     # Scale X_train
    #     scaler_x = StandardScaler()
    #     X_train_scaled = scaler_x.fit_transform(X_train)

    #     # Scale X (for predictions)
    #     X_scaled = scaler_x.transform(X)

    #     # Scale y_train
    #     scaler_y = StandardScaler()
    #     y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

    #     kernels = [
    #         Matern(length_scale=0.1, length_scale_bounds=(1e-1, 1), nu=1.5)  # <= 0.3 eccentricity
    #     ]

    #     mean_prediction_per_kernel = []
    #     std_predictions_per_kernel = []
    #     lml_per_kernel = []

    #     for kernel in kernels:
    #         start = time.time()
    #         if optimized_kernel is None:
    #             gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
    #         else:
    #             gaussian_process = GaussianProcessRegressor(kernel=optimized_kernel, optimizer=None)

    #         # Fit the GP model on scaled data
    #         gaussian_process.fit(X_train_scaled, y_train_scaled)
    #         optimized_kernel = gaussian_process.kernel_

    #         end = time.time()

    #         # Log-Marginal Likelihood
    #         lml = gaussian_process.log_marginal_likelihood_value_
    #         lml_per_kernel.append(lml)

    #         # Print the optimized kernel and hyperparameters
    #         print(f"kernel = {kernel}; Optimized kernel: {optimized_kernel} | time = {end - start:.2f}s | LML = {lml:.4f}")

    #         # Make predictions on scaled X
    #         mean_prediction_scaled, std_prediction_scaled = gaussian_process.predict(X_scaled, return_std=True)
    #         mean_prediction = scaler_y.inverse_transform(mean_prediction_scaled.reshape(-1, 1)).flatten()
    #         std_prediction = std_prediction_scaled * scaler_y.scale_[0]

    #         mean_prediction_per_kernel.append(mean_prediction)
    #         std_predictions_per_kernel.append(std_prediction)
        
    #     if plot_kernels is True:
    #         GPR_fit = plt.figure()

    #         for i in range(len(mean_prediction_per_kernel)):
    #             plt.scatter(X_train, y_train, color='red', label="Observations", s=10)
    #             plt.plot(X, mean_prediction_per_kernel[i], label='Mean prediction', linewidth=0.8)
    #             plt.fill_between(
    #                 X.ravel(),
    #             (mean_prediction_per_kernel[i] - 1.96 * std_predictions_per_kernel[i]), 
    #             (mean_prediction_per_kernel[i] + 1.96 * std_predictions_per_kernel[i]),
    #                 alpha=0.5,
    #                 label=r"95% confidence interval",
    #             )
    #         plt.legend(loc = 'upper left')
    #         plt.xlabel("$e$")
    #         if train_obj.property == 'amplitude':
    #             plt.ylabel("$f_A(e)$")
    #         elif train_obj.property == 'phase':
    #             plt.ylabel("$f_{\phi}(e)$")
    #         # plt.title(f"GPR {train_obj.property} at T_{time_node}")
    #         # plt.show()

    #         if save_fig_kernels is True:
    #             figname = f'Gaussian_kernels_{train_obj.property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(self.ecc_ref_space)}_{max(self.ecc_ref_space)}_Ni={len(self.ecc_ref_space)}]_No={len(self.ecc_ref_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}.png'
                
    #             # Ensure the directory exists, creating it if necessary and save
    #             os.makedirs('Images/Gaussian_kernels', exist_ok=True)
    #             GPR_fit.savefig('Images/Gaussian_kernels/' + figname)

    #             print('Figure is saved in Images/Gaussian_kernels/' + figname)

    #     return mean_prediction, [(mean_prediction - 1.96 * std_prediction), (mean_prediction + 1.96 * std_prediction)], optimized_kernel, lml_per_kernel

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
                            time_coupled=False, 
                            save_fits_to_file=True, 
                            plot_kernel_errors=False, plot_kernel_predictions=False, save_fig_kernels=False,
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
          
            gpr_obj = gpr_obj.load_gpr_obj()
            train_obj = train_obj.load()

        except Exception as e:
            print(e)
            traceback.print_exc()

            if train_obj.training_set is None:
                try:
                    train_obj = train_obj.load()
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    # Generate the training set of greedy parameters at empirical nodes
                    train_obj = self.get_training_set_greedy(property=property, min_greedy_error=min_greedy_error, N_greedy_vecs=N_basis_vecs)
            
            print(f"Training set for {property} has {len(train_obj.basis_indices)} basis vectors and {len(train_obj.empirical_indices)} empirical nodes.")

            # Create empty arrays to save fitvalues
            N_nodes = len(train_obj.empirical_indices)

            gpr_obj.gp_models = []
            gpr_obj.kernels = []

            gpr_obj.scaler_x = []
            gpr_obj.scaler_y = []

            gpr_obj.best_lmls = np.zeros(N_nodes)
            gpr_obj.best_train_rmses = np.zeros(N_nodes)
            gpr_obj.best_scores = np.zeros(N_nodes)
            
            print(f'Interpolate {property}...')

            start2 = time.time()
            optimized_kernel = None

            for node_i in range(N_nodes):

                result = self._gaussian_process_regression_t(
                    time_node=node_i,
                    train_obj=train_obj,
                    optimized_kernel=optimized_kernel,
                    plot_kernel_errors=plot_kernel_errors,
                    plot_kernel_predictions=plot_kernel_predictions,
                    save_fig_kernels=save_fig_kernels,
                    time_coupled=time_coupled
                )

                # --------------------------------------------
                # Store GP information
                # --------------------------------------------

                gpr_obj.gp_models.append(
                    result["gp"]
                )

                gpr_obj.kernels.append(
                    result["kernel"]
                )

                gpr_obj.scaler_x.append(
                    result["scaler_x"]
                )

                gpr_obj.scaler_y.append(
                    result["scaler_y"]
                )

                # --------------------------------------------
                # Diagnostics
                # --------------------------------------------
                # Log-Marginal likelihood
                gpr_obj.best_lmls[node_i] = (
                    result["lml"]
                )

                gpr_obj.best_train_rmses[node_i] = (
                    result["train_rmse"]
                )

                gpr_obj.best_scores[node_i] = (
                    result["score"]
                )

                # --------------------------------------------
                # Warm-start next empirical node
                # --------------------------------------------
                optimized_kernel = result["kernel"]

            end2 = time.time()
            print(f'time full GPR = {end2 - start2}')


        # If plot_fits is True, plot the GPR fits
        if plot_GPR_fits:
            self._plot_GPR_fits(train_obj=train_obj, gpr_obj=gpr_obj, gaussian_fit=None, training_set=None, lml_fits=None, save_fig_fits=save_fig_GPR_fits)

        # If save_fits_to_file is True, save the GPR fits to a file
        if save_fits_to_file:
            # Save the GPR fits to a file
            gpr_obj.save_gpr_obj()
        
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
                print(e)
                traceback.print_exc()
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
                gpr_obj = gpr_obj.load_gpr_obj()
            except Exception as e:
                print(e, '\n Make sure to run fit_to_training_set() before plotting')
                traceback.print_exc()

        if train_obj.training_set is None:
            try:
                train_obj = train_obj.load()
            except Exception as e:
                print(e, '\n Make sure to run fit_to_training_set before plotting')
                traceback.print_exc()

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
            print(e)
            traceback.print_exc()
            
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
                                ecc_ref_parameterspace=np.linspace(0.001, 0.1, num=80),
                                mean_ano_parameterspace=[0],
                                mass_ratio_parameterspace=[1],
                                chi1_parameterspace=[0],
                                chi2_parameterspace=[0],
                                M_output_wfs_per_dimension=150, 
                                min_greedy_error_amp=1e-8,
                                min_greedy_error_phase=1e-6,
                                minimum_spacing_greedy=0.003, 
                                training_set_selection='greedy')

"""
CHECK MY NOTES ON WHAST TO DO NEXT! TODO.txt file!
"""
# train_obj_p = gt._get_training_obj('phase')

# gt.generate_property_dataset(train_obj=train_obj_p, 
#                             #  plot_residuals_eccentric_evolve=True,
#                             #  plot_residuals_time_evolve=True,
#                              )
# gt.get_training_set_greedy(property='phase', 
#                            min_greedy_error=1e-8, 
#                         #    plot_greedy_error=True,
#                         #    plot_training_set=True,
#                         #    plot_emp_nodes_on_basis=True
#                         )

# plt.show()
gt.fit_to_training_set('amplitude', 
                       min_greedy_error=1e-8, 
                       save_fits_to_file=False, 
                       plot_kernel_errors=True,
                       plot_kernel_predictions=True,
                       save_fig_kernels=True,
                       time_coupled=True,
                    #    plot_GPR_fits=True, save_fig_GPR_fits=True, 
                    #    plot_residuals_ecc_evolve=True, save_fig_ecc_evolve=True, 
                    #    plot_residuals_time_evolve=True, save_fig_time_evolve=True,
                    )
plt.show()
#     # gt.fit_to_training_set('amplitude', min_greedy_error=1e-6, save_fits_to_file=True, plot_GPR_fits=True, save_fig_GPR_fits=True, plot_residuals_ecc_evolve=True, save_fig_ecc_evolve=True, plot_residuals_time_evolve=True, save_fig_time_evolve=True)

# plt.show()
# # gt.fit_to_training_set('amplitude', N_basis_vecs=21, save_fits_to_file=True)