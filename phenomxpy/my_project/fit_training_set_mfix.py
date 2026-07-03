# =====================================================================
# [OPTIMIZED VERSION] Memory improvements for fit_training_set.py
# Applied Safe Quick Wins #1-5 (comments marked with #[OPTIMIZED])
# =====================================================================
import datetime

from generate_greedy_training_set import *

from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, WhiteKernel, 
    ExpSineSquared, DotProduct, ConstantKernel as C, Product
)
import time
import traceback
from sklearn.preprocessing import StandardScaler
from inspect import currentframe
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import ConvergenceWarning

from matplotlib.colors import Normalize, LogNorm
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

from collections import Counter

import pickle

warnings.simplefilter("ignore", ConvergenceWarning)

f = currentframe()

import plotly.io as pio
pio.renderers.default = "browser"

@dataclass
class GPRFitResults(Warnings):
    
    property: str = "phase"
    time: Any = None

    f_ref: float = None
    f_lower: float = None
    phiRef: float = 0.0
    inclination: float = 0.0
    truncate_at_ISCO: bool = True
    truncate_at_tmin: bool = True
    luminosity_distance: Optional[float] = None

    ecc_ref_space_sampl: Any = None
    ecc_ref_space_val: Any = None

    mean_ano_ref_space_sampl: Any = None
    mean_ano_ref_space_val: Any = None

    mass_ratio_space_sampl: Any = None
    mass_ratio_space_val: Any = None

    chi1_space_sampl: Any = None
    chi1_space_val: Any = None

    chi2_space_sampl: Any = None
    chi2_space_val: Any = None

    parameter_grid: Any = None

    N_basis_vecs: Optional[int] = None
    min_greedy_error: Optional[float] = None

    residuals: Any = None
    basis_indices: list = field(default_factory=list)
    empirical_indices: list = field(default_factory=list)
    training_set: Any = None

    gp_models: list = field(default_factory=list)
    kernels: list = field(default_factory=list)
    labels: list = field(default_factory=list)

    best_lmls: Any = None
    best_train_rmses: Any = None
    best_scores: Any = None
    validation_errors: Any = None
    fit_times: Any = None

    scaler_x: list = field(default_factory=list)
    scaler_y: list = field(default_factory=list)

    def __post_init__(self):
        for name in [
            "ecc_ref_space_sampl",
            "mean_ano_ref_space_sampl",
            "mass_ratio_space_sampl",
            "chi1_space_sampl",
            "chi2_space_sampl",
        ]:
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, np.round(np.asarray(value, dtype=float), 4))
    
    def update_results(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"{type(self).__name__} has no attribute '{key}'")
            setattr(self, key, value)
            
    @staticmethod
    def _range_block(name, values):
        values = np.asarray(values, dtype=float)
        return f"{name}=[{values.min():g}_{values.max():g}_N={len(values)}]"

    @staticmethod
    def _scalar_block(name, value):
        return f"{name}={value:g}"

    @staticmethod
    def _kernel_to_string(kernel):
        text = str(kernel)
        text = text.replace(" ", "")
        text = text.replace("\n", "")
        text = text.replace("**", "^")
        text = text.replace("*", "x")
        text = text.replace("(", "")
        text = text.replace(")", "")
        return text

    def name_blocks(self):
        blocks = [
            self.property,
            self._range_block("e", self.ecc_ref_space_sampl),
            self._range_block("l", self.mean_ano_ref_space_sampl),
            self._range_block("q", self.mass_ratio_space_sampl),
            self._range_block("x1", self.chi1_space_sampl),
            self._range_block("x2", self.chi2_space_sampl),
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
        name = f"{prefix}_{'_'.join(self.name_blocks())}.{ext}"
        if directory is not None:
            os.makedirs(directory, exist_ok=True)
            return f"{directory.rstrip('/')}/{name}"
        return name
    
    def figname(self, prefix="fig", ext="png", directory=None):
        if directory is not None:
            os.makedirs(directory, exist_ok=True)
        figname = self.filename(prefix=prefix, ext=ext, directory=directory)
        print(self.colored_text(f"Figure is saved in {figname}", 'blue'))
        return figname
    
    def save_gpr_obj(self, prefix="GPRFitResults", directory="Straindata/GPRFitResults"):
        if directory is not None:
            os.makedirs(directory, exist_ok=True)
        filepath = self.filename(prefix=prefix, ext="pkl", directory=directory)

        with open(filepath, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(self.colored_text(f'GPRFitResults saved to {filepath}', 'blue'))
        
        # [OPTIMIZED #2]: Force GC after large pickle dump
        if MEMORY_PROFILE:
            check_memory_usage(f"After save_gpr_obj: {filepath}")
        gc.collect()
        
        return filepath
    
    def load_gpr_obj(self, prefix="GPRFitResults", directory="Straindata/GPRFitResults"):
        filepath = self.filename(prefix=prefix, ext="pkl", directory=directory)
        with open(filepath, 'rb') as f:
            loaded_obj = pickle.load(f)

        if not isinstance(loaded_obj, GPRFitResults):
            raise ValueError(f"Loaded object is not of type GPRFitResults. Got {type(loaded_obj)} instead.")

        print(self.colored_text(f"GPRFitResults object loaded from {filepath}", 'blue'))
        return loaded_obj
    
    def save_offline_surrogate(self, prefix="Offline_GP", directory="Straindata/Offline_GP_Models"):
        if directory is not None:
            os.makedirs(directory, exist_ok=True)
        filepath = self.filename(prefix=prefix, ext="pkl", directory=directory)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        bundle = {
            "property": self.property,
            "time": self.time,
            "gp_models": self.gp_models,
            "kernels": self.kernels,
            "scaler_x": self.scaler_x,
            "scaler_y": self.scaler_y,
            "parameter_grid": self.parameter_grid,
        }

        with open(filepath, "wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

        # [OPTIMIZED #2]: Clear memory after saving surrogate
        del bundle
        gc.collect()

    def load_offline_surrogate(self, prefix="Offline_GP", directory="Straindata/Offline_GP_Models"):
        filepath = self.filename(prefix=prefix, ext="pkl", directory=directory)
        with open(filepath, "rb") as f:
            bundle = pickle.load(f)

        self.property = bundle.get("property", None)
        self.time = bundle.get("time", None)
        self.gp_models = bundle.get("gp_models", None)
        self.scaler_x = bundle.get("scaler_x", None)
        self.scaler_y = bundle.get("scaler_y", None)
        self.parameter_grid = bundle.get("parameter_grid", None)
        
        # [OPTIMIZED #2]: Clear unused data
        del bundle
        gc.collect()


class Generate_Offline_Surrogate(Generate_TrainingSet):

    def __init__(self, time_array, 
                 ecc_ref_parameterspace=np.linspace(0.0, 0.3, num=50), 
                 mean_ano_parameterspace=np.linspace(0.0, 2*np.pi, num=50), 
                 mass_ratio_parameterspace=np.linspace(1, 20, num=50),
                 chi1_parameterspace=np.linspace(-0.995, 0.995, num=50),
                 chi2_parameterspace=np.linspace(-0.995, 0.995, num=50),
                 sampling_val_ecc_ref=0.01, 
                 sampling_val_mean_ano=0.1,
                 sampling_val_mass_ratio=0.5,
                 sampling_val_chi1=0.1,
                 sampling_val_chi2=0.1,
                 N_basis_vecs_amp=None, 
                 N_basis_vecs_phase=None, 
                 min_greedy_error_amp=None, 
                 min_greedy_error_phase=None, 
                 training_set_selection='GPR_opt', 
                 f_lower=10, 
                 f_ref=20, 
                 phiRef=0., 
                 inclination=0., 
                 truncate_at_ISCO=True, 
                 truncate_at_tmin=True):
        
        # [OPTIMIZED #2]: Check memory before initialization
        if MEMORY_PROFILE:
            check_memory_usage("START Generate_Offline_Surrogate.__init__")

        if (N_basis_vecs_amp is None and N_basis_vecs_phase is None) and \
            (min_greedy_error_amp is None and min_greedy_error_phase is None):
                print('Choose either settings for the amount of basis_vecs OR the minimum greedy error.')
                sys.exit(1)

        self.ecc_ref_space_sampl = self.allowed_eccentricity_warning(ecc_ref_parameterspace)
        self.mass_ratio_space_sampl = self.allowed_mass_ratio_warning(mass_ratio_parameterspace)
        self.mean_ano_ref_space_sampl = self.allowed_mean_anomaly_warning(mean_ano_parameterspace)
        self.chi1_space_sampl = self.allowed_chispin_warning(chi1_parameterspace)
        self.chi2_space_sampl = self.allowed_chispin_warning(chi2_parameterspace)

        def validation_space(sampl_space, allowed_warning_func, sampling_val_output):
            min_val, max_val = min(sampl_space), max(sampl_space)
            if min_val == max_val:
                return np.array([min_val])
            else:
                n = int((max_val - min_val) / sampling_val_output) + 1
                return allowed_warning_func(
                    np.linspace(min_val, max_val, num=n).round(4)
                )
        # Validation spaces for each parameter
        self.ecc_ref_space_val = validation_space(ecc_ref_parameterspace, self.allowed_eccentricity_warning, sampling_val_ecc_ref)
        self.mean_ano_ref_space_val = validation_space(mean_ano_parameterspace, self.allowed_mean_anomaly_warning, sampling_val_mean_ano)
        self.mass_ratio_space_val = validation_space(mass_ratio_parameterspace, self.allowed_mass_ratio_warning, sampling_val_mass_ratio)
        self.chi1_space_val = validation_space(chi1_parameterspace, self.allowed_chispin_warning, sampling_val_chi1)
        self.chi2_space_val = validation_space(chi2_parameterspace, self.allowed_chispin_warning, sampling_val_chi2)
        
        # Large validation space warning
        validation_space_samples = self.ecc_ref_space_val.size * self.mean_ano_ref_space_val.size * self.mass_ratio_space_val.size * self.chi1_space_val.size * self.chi2_space_val.size
        if validation_space_samples > 1e6:
            print(self.colored_text(f"Warning: The validation space has {validation_space_samples} samples, which may be computationally expensive and will ask for a large memory storage. \
                  Consider increasing the sampling values to reduce the number of samples.", 'yellow'))
        
        # 
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
                 ecc_ref_parameterspace=self.ecc_ref_space_sampl, 
                 mean_ano_parameterspace=self.mean_ano_ref_space_sampl, 
                 mass_ratio_parameterspace=self.mass_ratio_space_sampl,
                 chi1_parameterspace=self.chi1_space_sampl,
                 chi2_parameterspace=self.chi2_space_sampl,
                 N_basis_vecs_amp=N_basis_vecs_amp, 
                 N_basis_vecs_phase=N_basis_vecs_phase, 
                 min_greedy_error_amp=min_greedy_error_amp, 
                 min_greedy_error_phase=min_greedy_error_phase, 
                 f_ref=f_ref, 
                 f_lower=f_lower, 
                 phiRef=phiRef, 
                 inclination=inclination, 
                 truncate_at_ISCO=truncate_at_ISCO, 
                 truncate_at_tmin=truncate_at_tmin)
        
        # [OPTIMIZED #2]: Check memory after init
        if MEMORY_PROFILE:
            check_memory_usage("END Generate_Offline_Surrogate.__init__")

    def result_kwargs_gpr(self, property):
        if property == "phase":
            N_basis_vecs = self.N_basis_vecs_phase
            min_greedy_error = self.min_greedy_error_phase
        elif property == "amplitude":
            N_basis_vecs = self.N_basis_vecs_amp
            min_greedy_error = self.min_greedy_error_amp
        else:
            raise ValueError("property must be 'phase' or 'amplitude'")


        return dict(
            property=property,
            N_basis_vecs=N_basis_vecs,
            min_greedy_error=min_greedy_error,
            ecc_ref_space_sampl=getattr(self, "ecc_ref_space_sampl", None),
            ecc_ref_space_val=getattr(self, "ecc_ref_space_val", None),
            mean_ano_ref_space_sampl=getattr(self, "mean_ano_ref_space_sampl", None),
            mean_ano_ref_space_val=getattr(self, "mean_ano_ref_space_val", None),
            mass_ratio_space_sampl=getattr(self, "mass_ratio_space_sampl", None),
            mass_ratio_space_val=getattr(self, "mass_ratio_space_val", None),
            chi1_space_sampl=getattr(self, "chi1_space_sampl", None),
            chi1_space_val=getattr(self, "chi1_space_val", None),
            chi2_space_sampl=getattr(self, "chi2_space_sampl", None),
            chi2_space_val=getattr(self, "chi2_space_val", None),
            time=self.time,
            f_ref=self.f_ref,
            f_lower=self.f_lower,
            phiRef=self.phiRef,
            inclination=self.inclination,
            truncate_at_ISCO=self.truncate_at_ISCO,
            truncate_at_tmin=self.truncate_at_tmin,
        )

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

    def build_kernels(self, base_ls, dim):

        # # ============================================================
        # # KERNEL SEARCH PARAMETERS
        # # ============================================================

        # # ------------------------------------------------------------
        # # Matern smoothness values.
        # #
        # # lower:
        # #     rougher functions
        # #
        # # higher:
        # #     smoother functions
        # # ------------------------------------------------------------


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

        smoothness_params = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0]
        kernels = []

        for nu in smoothness_params:
            kernels.append({
                "kernel": Matern(length_scale=base_ls, length_scale_bounds=(0.1 * base_ls, 10 * base_ls), nu=nu),
                "type": "isotropic",
                "nu": nu,
                "base_ls": base_ls
            })

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

        # anisotropic
        for nu in smoothness_params:
            kernels.append({
                "kernel": Matern(length_scale=np.ones(dim) * base_ls, length_scale_bounds=(0.1 * base_ls, 10 * base_ls), nu=nu),
                "type": "anisotropic",
                "nu": nu,
                "base_ls": base_ls
            })

        # # ============================================================
        # # 3. ADD WHITE NOISE TERMS
        # # ============================================================

        # # ------------------------------------------------------------
        # # Models numerical noise / imperfect data.
        # #
        # # Prevents overfitting.
        # # ------------------------------------------------------------

        # noise augmentation
        noise_levels = [1e-6]
        extra = []
        for entry in kernels:
            for noise in noise_levels:
                extra.append({
                    "kernel": entry["kernel"] + WhiteKernel(noise_level=noise, noise_level_bounds=(1e-10, 1e-1)),
                    "type": entry["type"] + "_noise",
                    "nu": entry["nu"],
                    "base_ls": entry["base_ls"],
                    "noise": noise
                })

        kernels.extend(extra)
        print(f"Generated {len(kernels)} kernels")
        return kernels

    def _train_gp_kernels(self, time_node, train_obj, optimized_kernel=None,
                        time_coupled=False, screening=True, refinement=True):
        """val
        Pure GP training.

        Fits all candidate kernels for one empirical time node, computes raw
        metrics (LML, RMSE), normalised scores, and returns the full diagnostics
        list plus auxiliary objects needed by the report.

        Returns
        -------
        kernel_diagnostics : list[dict]
            One entry per successfully fitted kernel.
        aux : dict
            Scalers, scaled arrays, y_train, dim, best_idx, etc. — everything
            the report function needs without re-computing.
        """

        # [OPTIMIZED #2]
        if MEMORY_PROFILE:
            check_memory_usage(f"_train_gp_kernels START node {time_node}")

        # ==============================================================
        # BUILD INPUT MATRICES
        # ==============================================================
        X_training_space = train_obj.parameter_grid

        if time_coupled is False:
            X_train = np.asarray(train_obj.parameter_grid)[train_obj.basis_indices]
        else:
            t_train = np.full(
                (train_obj.parameter_grid.shape[0], 1),
                self.time[train_obj.empirical_indices[time_node]]
            )
            X_train = np.hstack([
                np.asarray(train_obj.parameter_grid)[train_obj.basis_indices],
                t_train[train_obj.basis_indices]
            ])
            t_training_space = np.full(
                (X_training_space.shape[0], 1),
                self.time[train_obj.empirical_indices[time_node]]
            )
            X_training_space = np.hstack([X_training_space, t_training_space])

        y_train = np.squeeze(train_obj.training_set.T[time_node])

        # ==============================================================
        # SCALE INPUTS
        # ==============================================================
        scaler_x = StandardScaler()
        X_train_scaled = scaler_x.fit_transform(X_train)
        X_training_space_scaled = scaler_x.transform(X_training_space)

        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

        # ==============================================================
        # ESTIMATE CHARACTERISTIC LENGTH SCALE
        # ==============================================================
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(X_train_scaled)
        distances, _ = nn.kneighbors(X_train_scaled)
        median_nn_distance = np.median(distances[:, 1])
        base_ls = median_nn_distance * 2

        dim = X_train_scaled.shape[1]
        kernels = self.build_kernels(base_ls=base_ls, dim=dim)

        if optimized_kernel is not None:
            kernels.insert(0, {
                "kernel": optimized_kernel[0],
                "type": "warm_start",
                "label": optimized_kernel[1]
            })

        # ==============================================================
        # MODE SELECTION
        # ==============================================================
        if screening and not refinement:
            mode = "screening"
        elif refinement and not screening:
            mode = "refinement"
        else:
            mode = "full"

        # ==============================================================
        # KERNEL SEARCH LOOP
        # ==============================================================
        kernel_diagnostics = []
        lml_list = []
        rmse_list = []

        for entry in kernels:
            try:
                # ---- restart policy ----
                if mode == "screening":
                    n_restarts = 2
                elif mode == "refinement":
                    n_restarts = 8
                else:
                    n_restarts = 5

                # ---- label ----
                if entry["type"] == 'warm_start':
                    kernel_label = entry["label"]
                else:
                    kernel_label = {
                        "isotropic": "Iso",
                        "anisotropic": "Aniso",
                        "isotropic_noise": "Iso+W",
                        "anisotropic_noise": "Aniso+W",
                    }[entry["type"]]
                    kernel_label += f" ν={entry['nu']}"
                    if "noise" in entry:
                        kernel_label += f", noise={entry['noise']:.0e}"

                # ---- fit ----
                t0 = time.perf_counter()
                gp = GaussianProcessRegressor(
                    kernel=entry["kernel"],
                    n_restarts_optimizer=n_restarts,
                    random_state=42
                )
                gp.fit(X_train_scaled, y_train_scaled)
                kernel_fit_time = time.perf_counter() - t0

                # ---- raw metrics ----
                lml = gp.log_marginal_likelihood_value_
                y_pred_train, std_train = gp.predict(X_train_scaled, return_std=True)
                train_rmse = np.sqrt(np.mean((y_pred_train - y_train_scaled) ** 2))

                lml_list.append(lml)
                rmse_list.append(train_rmse)

                # ---- screening prune ----
                if mode == "screening" and len(lml_list) > 5:
                    if lml < np.mean(lml_list) - 5 * np.std(lml_list):
                        continue
                    if train_rmse > np.mean(rmse_list) + 3 * np.std(rmse_list):
                        continue

                # ---- full-grid predictions ----
                y_pred_scaled, std_pred_scaled = gp.predict(
                    X_training_space_scaled, return_std=True
                )
                y_pred_training_space = scaler_y.inverse_transform(
                    y_pred_scaled.reshape(-1, 1)
                ).flatten()
                y_pred_training_space_std = std_pred_scaled * scaler_y.scale_[0]

                kernel_diagnostics.append({
                    "time_node": time_node,
                    "kernel": gp.kernel_,
                    "lml": lml,
                    "rmse": train_rmse,
                    "label": kernel_label,
                    "fit_time": kernel_fit_time,
                    "gp": gp,
                    "y_train_pred": y_pred_train,
                    "std_train": std_train,
                    "y_pred": y_pred_training_space,
                    "y_pred_std": y_pred_training_space_std,
                })

            except Exception as e:
                print(self.colored_text(f"GPR failed for kernel {entry}: {e}", "red"))
                traceback.print_exc()
                continue

        # ==============================================================
        # COMPUTE NORMALISED SCORES
        # ==============================================================
        all_lmls = np.array([d["lml"] for d in kernel_diagnostics])
        all_rmses = np.array([d["rmse"] for d in kernel_diagnostics])

        mean_lml, std_lml = np.mean(all_lmls), np.std(all_lmls) + 1e-12
        mean_rmse, std_rmse = np.mean(all_rmses), np.std(all_rmses) + 1e-12

        for d in kernel_diagnostics:
            lml_norm = (d["lml"] - mean_lml) / std_lml
            rmse_norm = (d["rmse"] - mean_rmse) / std_rmse
            d["score"] = lml_norm - 0.5 * rmse_norm

        best_idx = int(np.argmax([d["score"] for d in kernel_diagnostics]))

        # ---- clear screening lists ----
        del lml_list, rmse_list, all_lmls, all_rmses
        gc.collect()

        # ==============================================================
        # PACKAGE AUXILIARY DATA FOR THE REPORT
        # ==============================================================
        aux = {
            "scaler_x": scaler_x,
            "scaler_y": scaler_y,
            "X_train_scaled": X_train_scaled,
            "X_training_space_scaled": X_training_space_scaled,
            "y_train": y_train,
            "y_train_scaled": y_train_scaled,
            "dim": dim,
            "time_coupled": time_coupled,
            "best_idx": best_idx,
            "base_ls": base_ls,
        }

        # ---- free large scaled arrays that the report doesn't need inline ----
        # (they're passed via aux; the caller decides whether to keep them)
        del nn, distances
        gc.collect()

        if MEMORY_PROFILE:
            check_memory_usage(f"_train_gp_kernels END node {time_node} "
                            f"({len(kernel_diagnostics)} kernels)")

        return kernel_diagnostics, aux

    # def _load_all_kernel_diagnostics(
    #     self,
    #     time_node,
    #     n_nodes,
    # ):
    #     """
    #     Load per-node kernel diagnostics pickle files from disk.
        
    #     Returns:
    #         list: List of length n_nodes, each entry is the diagnostics dict for that node
    #             If file not found for node i, that position is None
    #     """
    #     reports_dir ="Straindata/Gaussian_kernels/Reports"
    #     os.makedirs("Straindata/Gaussian_kernels/Reports", exist_ok=True)


    #     print(self.colored_text("-"*50, "blue"))
    #     print(self.colored_text(f"Processing Time Node {time_node}/{n_nodes-1}", "white"))
        
    #     # =========================================================================
    #     # LOAD PER-NODE KERNEL DIAGNOSTICS
    #     # =========================================================================
    #     node_diag_path = None
    #     if os.path.exists(reports_dir):
    #         prefix = f"kernel_diagnostics_T{time_node}"
    #         for fname in os.listdir(reports_dir):
    #             if fname.startswith(prefix) and fname.endswith(".pkl"):
    #                 node_diag_path = os.path.join(reports_dir, fname)
    #                 break
        
    #     if node_diag_path is None:
    #         print(self.colored_text(f"⚠ Skipping node {time_node} - no diagnostics file found", "yellow"))
    #         return  
                        
    #     with open(node_diag_path, "rb") as f:
    #         kernel_diagnostics_loaded = pickle.load(f)
        
    #     # Reconstruct to match original format
    #     kernel_diagnostics = []
    #     for d in kernel_diagnostics_loaded:
    #         kernel_diagnostics.append({
    #             "time_node": d["time_node"],
    #             "kernel_str": d["kernel_str"],
    #             "label": d["kernel_type"],
    #             "lml": d["lml"],
    #             "rmse": d["rmse"],
    #             "score": d["score"],
    #             "fit_time": d["fit_time"],
    #             "y_train_pred": np.array(d["y_train_pred"]),
    #             "y_pred": np.array(d["y_pred"]),
    #             "y_pred_std": np.array(d["y_pred_std"]),
    #             "std_train": np.array(d["std_train"]),
    #             "gp": None,  # Not in pickle file
    #             "kernel": None,  # String only
    #         })
        
    #     if len(kernel_diagnostics) == 0:
    #         print(self.colored_text(f"⚠ Skipping node {time_node} - empty diagnostics file", "yellow"))
    #         return
                        
    #     print(self.colored_text(f"Loaded {len(kernel_diagnostics)} kernels for node {time_node}", "green"))

        
    #     return kernel_diagnostics

    def _load_all_kernel_diagnostics(self, time_node, n_nodes):
        """
        Load per-node kernel diagnostics pickle files from disk.
        
        Returns:
            tuple: (kernel_diagnostics_list, aux_dict)
                - kernel_diagnostics_list: List of kernel diagnostic dicts for that node
                - aux_dict: Contains scaler_x, scaler_y, X_train_scaled, etc.
                If file not found for node i, returns (None, None)
        """
        reports_dir ="Straindata/Gaussian_kernels/Reports"
        os.makedirs("Straindata/Gaussian_kernels/Reports", exist_ok=True)


        print(self.colored_text("-"*50, "blue"))
        print(self.colored_text(f"Processing Time Node {time_node}/{n_nodes-1}", "white"))
        
        # =========================================================================
        # LOAD PER-NODE KERNEL DIAGNOSTICS WITH AUX DATA
        # =========================================================================
        node_diag_path = None
        if os.path.exists(reports_dir):
            prefix = f"kernel_diagnostics_T{time_node}"
            for fname in os.listdir(reports_dir):
                if fname.startswith(prefix) and fname.endswith(".pkl"):
                    node_diag_path = os.path.join(reports_dir, fname)
                    break
        
        if node_diag_path is None:
            print(self.colored_text(f"⚠ Skipping node {time_node} - no diagnostics file found", "yellow"))
            return None, None
                            
        with open(node_diag_path, "rb") as f:
            diagnostics_bundle = pickle.load(f)
        
        # Extract auxiliary data from bundle
        aux = diagnostics_bundle.get("aux", {})
        
        # Get original kernel_diagnostics list directly
        kernel_diagnostics = diagnostics_bundle.get("diagnostics", [])
        
        if len(kernel_diagnostics) == 0:
            print(self.colored_text(f"⚠ Skipping node {time_node} - empty diagnostics file", "yellow"))
            return None, None
                            
        print(self.colored_text(f"Loaded {len(kernel_diagnostics)} kernels for node {time_node} from {node_diag_path}", "blue"))

        
        return kernel_diagnostics, aux

    # def _kernel_analysis_report(
    #     self,
    #     gpr_obj:GPRFitResults,
    #     train_obj:TrainingSetResults,
    #     time_node,
    #     time_coupled=False,
    #     n_best_pred=5,
    #     n_q_slices=3,
    # ):
    #     """
    #     Generate SEPARATE PDF reports for EACH time node.
        
    #     Each node gets its OWN PDF file containing:
    #     - Page 1: Title & Node Statistics
    #     - Page 2: Kernel Diagnostics (LML/RMSE/Scores/Fit Times)
    #     - Page 3: Kernel Predictions Comparison (Best vs Others)
    #     - Page 4: Parity Plot (True vs Pred)
    #     - Page 5: Validation Heatmaps
        
    #     All reports stored in reports_dir/ subfolder named via gpr_obj filename logic.
        
    #     Prerequisites:
    #     - gpr_obj.load_gpr_obj() already called
    #     - train_obj.load() already called (optional but recommended)
    #     - kernel_diagnostics_T*.pkl files exist for each node
    #     """
    #     reports_dir="Images/Gaussian_kernels/PerNodeReports"
    #     os.makedirs(reports_dir, exist_ok=True)

    #     # =========================================================================
    #     # LOOP THROUGH ALL TIME NODES
    #     # =========================================================================
    #     successful_reports = 0
    #     n_nodes = len(train_obj.empirical_indices)

    #     try:
    #         kernel_diagnostics = self._load_all_kernel_diagnostics(
    #             time_node=time_node,
    #             n_nodes=n_nodes
    #         )

    #         # =========================================================================
    #         # GET BEST INFO FROM THIS NODE'S DIANOSTICS
    #         # =========================================================================
    #         best_idx = int(np.argmax([k["score"] for k in kernel_diagnostics]))
    #         best_info = kernel_diagnostics[best_idx]
            
    #         # Get scalers from gpr_obj for this node
    #         scaler_x = kernel_diagnostics[time_node]["scaler_x"]
    #         scaler_y = kernel_diagnostics[time_node]["scaler_y"]
    #         y_train = train_obj.training_set.T[time_node] if train_obj else None
            
    #         # =========================================================================
    #         # DEFINE OUTPUT PATH FOR THIS NODE'S PDF
    #         # =========================================================================
    #         report_filename = gpr_obj.figname(
    #             prefix=f"kernel_analysis_report_T{time_node}",
    #             ext="pdf",
    #             directory=reports_dir,
    #             include_greedy=True,
    #             include_extra=""
    #         )
            
    #         pdf = PdfPages(report_filename)
            
    #         # =========================================================================
    #         # PAGE 1: TITLE & NODE STATISTICS
    #         # =========================================================================
    #         fig_title, ax = plt.subplots(figsize=(10, 8))
    #         ax.axis('off')
    #         title_text = f'''KERNEL ANALYSIS REPORT - TIME NODE {time_node}
    #         ========================================================

    #         Property: {property}
    #         Total Kernels Tested: {len(kernel_diagnostics)}

    #         BEST KERNEL:
    #         • Label: {best_info['label']}
    #         • LML: {best_info['lml']:.4f}
    #         • Train RMSE: {best_info['rmse']:.6e}
    #         • Score: {best_info['score']:.4f}
    #         • Fit Time: {best_info['fit_time']:.3f}s

    #         MODEL STATS ACROSS ALL KERNELS:
    #         • Best LML: {np.max([k["lml"] for k in kernel_diagnostics]):.4f}
    #         • Worst LML: {np.min([k["lml"] for k in kernel_diagnostics]):.4f}
    #         • Best RMSE: {np.min([k["rmse"] for k in kernel_diagnostics]):.6e}
    #         • Mean Score: {np.mean([k["score"] for k in kernel_diagnostics]):.4f}

    #         '''
    #         ax.text(0.5, 0.5, title_text, fontsize=11, va='center', ha='center',
    #                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    #         plt.tight_layout()
            
    #         pdf.savefig(fig_title, bbox_inches="tight")
    #         plt.close(fig_title)
    #         gc.collect()
            
    #         # =========================================================================
    #         # PAGE 2: KERNEL DIAGNOSTICS PLOTS
    #         # =========================================================================
    #         result = self._plot_kernel_diagnostics(
    #             kernel_diagnostics=kernel_diagnostics,
    #             train_obj=train_obj,
    #             save_fig=False
    #         )
            
    #         if result:
    #             figs = result if isinstance(result, (tuple, list)) else [result]
    #             for fig in figs:
    #                 pdf.savefig(fig, bbox_inches="tight")
    #                 plt.close(fig)
            
    #         gc.collect()
            
    #         # =========================================================================
    #         # PAGE 3: KERNEL PREDICTIONS COMPARISON
    #         # =========================================================================

    #         result = self._plot_kernel_predictions(
    #             kernel_diagnostics=kernel_diagnostics,
    #             train_obj=train_obj,
    #             scaler_y=scaler_y,
    #             y_train=y_train,
    #             n_best=n_best_pred,
    #             save_fig=False
    #         )
            
    #         if result:
    #             figs = result if isinstance(result, (tuple, list)) else [result]
    #             for fig in figs:
    #                 pdf.savefig(fig, bbox_inches="tight")
    #                 plt.close(fig)
            
    #         gc.collect()
            
    #         # =========================================================================
    #         # PAGE 4: PARITY PLOT
    #         # =========================================================================
    #         best_result_dict = {
    #             "gp": None,  # Not in pickle, skip or would need gp from gpr_obj
    #             "scaler_x": scaler_x,
    #             "scaler_y": scaler_y,
    #             "train_prediction": scaler_y.inverse_transform(
    #                 best_info["y_train_pred"].reshape(-1, 1)
    #             ).flatten(),
    #             "y_train": y_train,
    #             "time_node": time_node,
    #             "basis_indices": train_obj.basis_indices,
    #             "label": best_info["label"],
    #             "train_rmse": best_info["rmse"],
    #             "lml": best_info["lml"],
    #             "score": best_info["score"],
    #         }
            
    #         result = self._plot_parity(
    #             best_result_dict=best_result_dict,
    #             train_obj=train_obj,
    #             save_fig=False
    #         )
                
    #         if result:
    #             figs = result if isinstance(result, (tuple, list)) else [result]
    #             for fig in figs:
    #                 pdf.savefig(fig, bbox_inches="tight")
    #                 plt.close(fig)
            
    #         gc.collect()
            
    #         # =========================================================================
    #         # PAGE 5: VALIDATION HEATMAPS
    #         # =========================================================================
    #         result = self._plot_validation_heatmaps_kernels(
    #             time_coupled=time_coupled,
    #             kernel_diagnostics=kernel_diagnostics,
    #             scaler_x=scaler_x,
    #             scaler_y=scaler_y,
    #             time_node=time_node,
    #             train_obj=train_obj,
    #             n_q_slices=n_q_slices,
    #             top_k=3
    #         )
            
    #         if result:
    #             figs = result if isinstance(result, (tuple, list)) else [result]
    #             for fig in figs:
    #                 pdf.savefig(fig, bbox_inches="tight")
    #                 plt.close(fig)
        
    #         gc.collect()
            
    #         # =========================================================================
    #         # FINALIZE PDF
    #         # =========================================================================

    #         pdf.close()
    #         plt.close('all')
    #         gc.collect()
            
    #         print(self.colored_text(f"Report saved in: {os.path.basename(report_filename)}", "green"))
    #         successful_reports += 1
            
    #     except Exception as e:
    #         print(self.colored_text(f"Failed node diagnostics report for {time_node}: {e}", "red"))
    #         traceback.print_exc()
    #         return     
        
    #     # =========================================================================
    #     # SUMMARY
    #     # =========================================================================
    #     print(self.colored_text("="*70, "cyan"))
    #     print(self.colored_text(f"COMPLETED: {successful_reports}/{n_nodes} per-node reports generated", "green"))
    #     print(self.colored_text(f"All reports saved to: {reports_dir}", "green"))
    #     print(self.colored_text("="*70, "cyan"))
        
    #     return successful_reports

    def _kernel_analysis_report(
        self,
        gpr_obj:GPRFitResults,
        train_obj:TrainingSetResults,
        time_node,
        time_coupled=False,
        n_best_pred=5,
        n_q_slices=3,
    ):
        """
        Generate SEPARATE PDF reports for EACH time node.
        
        Each node gets its OWN PDF file containing:
        - Page 1: Title & Node Statistics
        - Page 2: Kernel Diagnostics (LML/RMSE/Scores/Fit Times)
        - Page 3: Kernel Predictions Comparison (Best vs Others)
        - Page 4: Parity Plot (True vs Pred)
        - Page 5: Validation Heatmaps
        
        All reports stored in reports_dir/ subfolder named via gpr_obj filename logic.
        
        Prerequisites:
        - gpr_obj.load_gpr_obj() already called
        - train_obj.load() already called (optional but recommended)
        - kernel_diagnostics_T*.pkl files exist for each node (NOW CONTAINS AUX DATA!)
        """
        reports_dir="Images/Gaussian_kernels/PerNodeReports"
        os.makedirs(reports_dir, exist_ok=True)

        # =========================================================================
        # LOOP THROUGH ALL TIME NODES
        # =========================================================================
        successful_reports = 0
        n_nodes = len(train_obj.empirical_indices)


        # =========================================================================
        # LOAD PER-NODE KERNEL DIAGNOSTICS WITH AUX DATA (CHANGED TO USE HELPER)
        # =========================================================================
        kernel_diagnostics, aux = self._load_all_kernel_diagnostics(
            time_node=time_node,
            n_nodes=n_nodes,
        )
        # Load scalers from aux dictionary
        scaler_x, scaler_y = aux["scaler_x"], aux["scaler_y"]  

        if kernel_diagnostics is None:
            print(self.colored_text(f"⚠ Skipping node {time_node} - load failed", "yellow"))
            return
        
        y_train = train_obj.training_set.T[time_node] if train_obj else None
            
        # =========================================================================
        # GET BEST INFO FROM THIS NODE'S DIANOSTICS
        # =========================================================================
        best_idx = int(np.argmax([k["score"] for k in kernel_diagnostics]))
        best_info = kernel_diagnostics[best_idx]
            
        # =========================================================================
        # DEFINE OUTPUT PATH FOR THIS NODE'S PDF
        # =========================================================================
        report_filename = gpr_obj.figname(
            prefix=f"kernel_analysis_report_T{time_node}",
            ext="pdf",
            directory=reports_dir,
        )
        
        pdf = PdfPages(report_filename)
        
        # =========================================================================
        # PAGE 1: TITLE & NODE STATISTICS
        # =========================================================================
        try:
            fig_title, ax = plt.subplots(figsize=(10, 8))
            ax.axis('off')
            title_text = f'''KERNEL ANALYSIS REPORT - TIME NODE {time_node}
            ========================================================

            Property: {gpr_obj.property}
            Total Kernels Tested: {len(kernel_diagnostics)}

            BEST KERNEL:
            • Label: {best_info['label']}
            • LML: {best_info['lml']:.4f}
            • Train RMSE: {best_info['rmse']:.6e}
            • Score: {best_info['score']:.4f}
            • Fit Time: {best_info['fit_time']:.3f}s

            MODEL STATS ACROSS ALL KERNELS:
            • Best LML: {np.max([k["lml"] for k in kernel_diagnostics]):.4f}
            • Worst LML: {np.min([k["lml"] for k in kernel_diagnostics]):.4f}
            • Best RMSE: {np.min([k["rmse"] for k in kernel_diagnostics]):.6e}
            • Mean Score: {np.mean([k["score"] for k in kernel_diagnostics]):.4f}

            '''
            ax.text(0.5, 0.5, title_text, fontsize=11, va='center', ha='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            plt.tight_layout()
            
            pdf.savefig(fig_title, bbox_inches="tight")
            plt.close(fig_title)
            gc.collect()
        except Exception as e:
            print(self.colored_text(f"EXCEPTION in title page for node {time_node}: {e}", 'red'))
            traceback.print_exc()
            
        # =========================================================================
        # PAGE 2: KERNEL DIAGNOSTICS PLOTS
        # =========================================================================
        try:
            result = self._plot_kernel_diagnostics(
                kernel_diagnostics=kernel_diagnostics,
                train_obj=train_obj,
                save_fig=False
            )
            
            if result:
                figs = result if isinstance(result, (tuple, list)) else [result]
                for fig in figs:
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
            
            gc.collect()
        except Exception as e:
            print(self.colored_text(f"EXCEPTION in _plot_kernel_diagnostics for node {time_node}: {e}", 'red'))
            traceback.print_exc()
        # =========================================================================
        # PAGE 3: KERNEL PREDICTIONS COMPARISON
        # =========================================================================
        try:
            result = self._plot_kernel_predictions(
                kernel_diagnostics=kernel_diagnostics,
                train_obj=train_obj,
                scaler_y=scaler_y,  # NOW FROM LOADED AUX
                y_train=y_train,
                n_best=n_best_pred,
                save_fig=False
            )
            
            if result:
                figs = result if isinstance(result, (tuple, list)) else [result]
                for fig in figs:
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
            
            gc.collect()
        except Exception as e:
            print(self.colored_text(f"EXCEPTION in _plot_kernel_predictions for node {time_node}: {e}", 'red'))
            traceback.print_exc()
        # =========================================================================
        # PAGE 4: PARITY PLOT
        # =========================================================================
        try:
            best_result_dict = {
                "gp": None,  # Not in pickle, skip or would need gp from gpr_obj
                "scaler_x": scaler_x,  # NOW FROM LOADED AUX
                "scaler_y": scaler_y,  # NOW FROM LOADED AUX
                "train_prediction": scaler_y.inverse_transform(
                    best_info["y_train_pred"].reshape(-1, 1)
                ).flatten(),
                "y_train": y_train,
                "time_node": time_node,
                "basis_indices": train_obj.basis_indices,
                "label": best_info["label"],
                "train_rmse": best_info["rmse"],
                "lml": best_info["lml"],
                "score": best_info["score"],
            }
            
            result = self._plot_parity(
                best_result_dict=best_result_dict,
                train_obj=train_obj,
                save_fig=False
            )
                    
            if result:
                figs = result if isinstance(result, (tuple, list)) else [result]
                for fig in figs:
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
            
            gc.collect()
        except Exception as e:
            print(self.colored_text(f"EXCEPTION in _plot_parity for node {time_node}: {e}", 'red'))
            traceback.print_exc()
        
        # =========================================================================
        # PAGE 5: VALIDATION HEATMAPS
        # =========================================================================
        try:
            result = self._plot_validation_heatmaps_kernels(
                time_coupled=time_coupled,
                kernel_diagnostics=kernel_diagnostics,
                scaler_x=scaler_x,  
                scaler_y=scaler_y,  
                time_node=time_node,
                train_obj=train_obj,
                n_q_slices=n_q_slices,
                top_k=3
            )
            
            if result:
                figs = result if isinstance(result, (tuple, list)) else [result]
                for fig in figs:
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
        
            gc.collect()
        except Exception as e:
            print(self.colored_text(f"EXCEPTION in _plot_validation_heatmaps_kernels for node {time_node}: {e}", 'red'))
            traceback.print_exc()
        
        # =========================================================================
        # FINALIZE PDF
        # =========================================================================

        pdf.close()
        
        plt.close('all')
        gc.collect()
        
        return reports_dir

    # def _kernel_analysis_report(self, 
    #                             kernel_diagnostics, 
    #                             aux, 
    #                             train_obj:TrainingSetResults,
    #                             n_wfs_pred=5,
    #                             save_dir="Images/Gaussian_kernels/Reports"):
    #     """
    #     Generate a comprehensive kernel analysis report for one time node.

    #     Calls existing plotting functions with save_fig=False, captures their
    #     returned figures (single or tuple), writes them to a single PDF,
    #     and closes everything immediately.

    #     Also saves a lightweight diagnostics pickle (metrics only, no arrays).
    #     """

    #     best_idx = aux["best_idx"]
    #     best_info = kernel_diagnostics[best_idx]
    #     time_node = best_info["time_node"]
    #     scaler_x = aux["scaler_x"]
    #     scaler_y = aux["scaler_y"]
    #     y_train = aux["y_train"]
    #     time_coupled = aux["time_coupled"]

    #     os.makedirs(save_dir, exist_ok=True)
    #     report_path = train_obj.figname(
    #         prefix="kernel_report",
    #         ext="pdf",
    #         directory=save_dir,
    #         include_greedy=True,
    #         include_extra=f"node={time_node}"
    #     )

    #     # Load true residuals for prediction comparison & parity
    #     train_obj.load_residuals()
    #     y_true = train_obj.residuals[:, train_obj.empirical_indices[time_node]]

    #     with PdfPages(report_path) as pdf:

    #         # ============================================================
    #         # FIG 1+2 — KERNEL DIAGNOSTICS (metrics panel + LML vs RMSE scatter)
    #         # ============================================================
    #         try:
    #             result = self._plot_kernel_diagnostics(
    #                 kernel_diagnostics=kernel_diagnostics,
    #                 train_obj=train_obj,
    #                 best_score=best_info["score"],
    #                 save_fig=False,
    #             )
    #             self._handle_figure_result(result, pdf)
    #             print(self.colored_text("Kernel diagnostics added to report", "blue"))
    #         except Exception as e:
    #             print(self.colored_text(f"Kernel diagnostics failed: {e}", "yellow"))
    #             traceback.print_exc()

    #         gc.collect()

    #         # ============================================================
    #         # FIG 3 — KERNEL PREDICTION COMPARISON (best vs worst)
    #         # ============================================================
    #         try:
    #             result = self._plot_kernel_predictions(
    #                 kernel_diagnostics=kernel_diagnostics,
    #                 train_obj=train_obj,
    #                 scaler_y=scaler_y,
    #                 y_train=y_train,
    #                 n_wfs=n_wfs_pred,
    #                 save_fig=False,
    #             )
    #             self._handle_figure_result(result, pdf)
    #             print(self.colored_text("Kernel predictions added to report", "blue"))
    #         except Exception as e:
    #             print(self.colored_text(f"Kernel predictions failed: {e}", "yellow"))
    #             traceback.print_exc()

    #         gc.collect()

    #         # ============================================================
    #         # FIG 4 — PARITY PLOT
    #         # ============================================================
    #         try:
    #             result = self._plot_parity(
    #                 best_result_dict={
    #                     "gp": best_info["gp"],
    #                     "scaler_x": scaler_x,
    #                     "scaler_y": scaler_y,
    #                     "train_prediction": scaler_y.inverse_transform(
    #                         best_info["y_train_pred"].reshape(-1, 1)
    #                     ).flatten(),
    #                     "y_train": y_train,
    #                     "time_node": time_node,
    #                     "basis_indices": train_obj.basis_indices,
    #                     "label": best_info["label"],
    #                     "train_rmse": best_info["rmse"],
    #                     "lml": best_info["lml"],
    #                     "score": best_info["score"],
    #                 },
    #                 train_obj=train_obj,
    #                 save_fig=False,
    #             )
    #             self._handle_figure_result(result, pdf)
    #             print(self.colored_text("Parity plot added to report", "blue"))
    #         except Exception as e:
    #             print(self.colored_text(f"Parity plot failed: {e}", "yellow"))
    #             traceback.print_exc()

    #         gc.collect()

    #         # ============================================================
    #         # FIG 5 — VALIDATION HEATMAPS
    #         # ============================================================
    #         try:
    #             result = self._plot_validation_heatmaps_kernels(
    #                 time_coupled=time_coupled,
    #                 kernel_diagnostics=kernel_diagnostics,
    #                 scaler_x=scaler_x,
    #                 scaler_y=scaler_y,
    #                 train_obj=train_obj,
    #                 time_node=time_node,
    #                 n_q_slices=5,
    #                 top_k=3,
    #             )
    #             self._handle_figure_result(result, pdf)
    #             print(self.colored_text("Validation heatmaps added to report", "blue"))
    #         except Exception as e:
    #             print(self.colored_text(f"Heatmap generation skipped: {e}", "yellow"))
    #             traceback.print_exc()

    #         gc.collect()

    #     print(self.colored_text(f"Kernel report PDF saved: {report_path}", "blue"))

    #     # ================================================================
    #     # SAVE LIGHTWEIGHT DIAGNOSTICS PICKLE
    #     # ================================================================
    #     diag_path = train_obj.filename(
    #         prefix="kernel_diagnostics",
    #         ext="pkl",
    #         directory=save_dir,
    #         include_greedy=True,
    #         include_extra=f"node={time_node}"
    #     )

    #     # Only save scalar metrics — prediction arrays are in the PDF
    #     diag_save = []
    #     for d in kernel_diagnostics:
    #         diag_save.append({
    #             "time_node": d["time_node"],
    #             "kernel_str": str(d["kernel"]),
    #             "lml": d["lml"],
    #             "rmse": d["rmse"],
    #             "score": d["score"],
    #             "label": d["label"],
    #             "fit_time": d["fit_time"],
    #         })

    #     with open(diag_path, "wb") as f:
    #         pickle.dump(diag_save, f, protocol=pickle.HIGHEST_PROTOCOL)

    #     print(self.colored_text(f"Kernel diagnostics saved: {diag_path}", "blue"))

    #     # ================================================================
    #     # CLEANUP
    #     # ================================================================
    #     del diag_save, y_true
    #     gc.collect()

        
    #     if MEMORY_PROFILE:
    #         check_memory_usage(f"After kernel report node {time_node}")

    # def _gaussian_process_regression_t(self, 
    #                                 time_node, 
    #                                 train_obj:TrainingSetResults,
    #                                 optimized_kernel=None,
    #                                 time_coupled=False,
    #                                 screening=True,
    #                                 refinement=True,
    #                                 save_diagnostics=True): 
    #     """
    #     Gaussian Process Regression for one empirical time node.
        
    #     Now ALSO saves per-node kernel_diagnostics as pickle file for later report regeneration.
    #     """
        
    #     kernel_diagnostics, aux = self._train_gp_kernels(
    #         time_node=time_node,
    #         train_obj=train_obj,
    #         optimized_kernel=optimized_kernel,
    #         time_coupled=time_coupled,
    #         screening=screening,
    #         refinement=refinement,
    #     )
        
    #     best_idx = aux["best_idx"]
    #     best_info = kernel_diagnostics[best_idx]
        
    #     print(self.colored_text(
    #         f"Best kernel = {best_info['kernel']}\n"
    #         f"LML = {best_info['lml']:.4f}\n"
    #         f"RMSE = {best_info['rmse']:.6e}\n"
    #         f"Score = {best_info['score']:.4f}",
    #         "green"
    #     ))
        
    #     # Save kernel diagnostics
    #     if save_diagnostics:
    #         diag_path = train_obj.filename(
    #             prefix=f"kernel_diagnostics_T{time_node}",
    #             ext="pkl",
    #             directory="Straindata/Gaussian_kernels/Reports",
    #             include_greedy=True,
    #             include_extra=""
    #         )
            
    #         diagnostics = []
    #         for d in kernel_diagnostics:
    #             diagnostics.append({
    #                 "time_node": d["time_node"],
    #                 "kernel_str": str(d["kernel"]),
    #                 "kernel_type": d["label"],
    #                 "lml": d["lml"],
    #                 "rmse": d["rmse"],
    #                 "score": d["score"],
    #                 "fit_time": d["fit_time"],
    #                 # Keep prediction arrays too (needed for some plots)
    #                 "y_train_pred": d["y_train_pred"].tolist(),  # Convert numpy to list
    #                 "y_pred": d["y_pred"].tolist(),
    #                 "y_pred_std": d["y_pred_std"].tolist(),
    #                 "std_train": d["std_train"].tolist(),
    #             })
            
    #         with open(diag_path, "wb") as f:
    #             pickle.dump(diagnostics, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    #         print(self.colored_text(f"✓ Kernel diagnostics saved for node {time_node}: {os.path.basename(diag_path)}", "cyan"))

    #     result_dict = {
    #         "gp":             best_info["gp"],
    #         "kernel":         best_info["kernel"],
    #         "label":          best_info["label"],
    #         "scaler_x":       aux["scaler_x"],
    #         "scaler_y":       aux["scaler_y"],
    #         "lml":            best_info["lml"],
    #         "train_rmse":     best_info["rmse"],
    #         "score":          best_info["score"],
    #         "time_node":      time_node,
    #         "parameter_dim":  aux["dim"],
    #         "time_coupled":   time_coupled,
    #         "fit_time":       best_info["fit_time"],
    #     }
        
    #     # Discard heavy data
    #     del kernel_diagnostics, aux
    #     gc.collect()
        
    #     if MEMORY_PROFILE:
    #         check_memory_usage(f"_gaussian_process_regression_t DONE node {time_node}")
        
    #     return result_dict

    def _gaussian_process_regression_t(self, 
                                    time_node, 
                                    train_obj:TrainingSetResults,
                                    optimized_kernel=None,
                                    time_coupled=False,
                                    screening=True,
                                    refinement=True,
                                    save_diagnostics=True): 
        """
        Gaussian Process Regression for one empirical time node.
        
        Now ALSO saves per-node kernel_diagnostics + aux as pickle file for later report regeneration.
        """
        
        kernel_diagnostics, aux = self._train_gp_kernels(
            time_node=time_node,
            train_obj=train_obj,
            optimized_kernel=optimized_kernel,
            time_coupled=time_coupled,
            screening=screening,
            refinement=refinement,
        )
        
        best_idx = aux["best_idx"]
        best_info = kernel_diagnostics[best_idx]
        
        print(self.colored_text(
            f"Best kernel = {best_info['kernel']}\n"
            f"LML = {best_info['lml']:.4f}\n"
            f"RMSE = {best_info['rmse']:.6e}\n"
            f"Score = {best_info['score']:.4f}",
            "green"
        ))
        
        # Save kernel diagnostics AND aux together
        if save_diagnostics:
            diag_path = train_obj.filename(
                prefix=f"kernel_diagnostics_T{time_node}",
                ext="pkl",
                directory="Straindata/Gaussian_kernels/Reports",
                include_greedy=True,
                include_extra=""
            )
            
            diagnostics_bundle = {
            "node_i": time_node,
            "time_node": time_node,
            "aux": aux,  # Full aux dict with scalers!
            "diagnostics": kernel_diagnostics,  # Save entire list directly
        }
        
        with open(diag_path, "wb") as f:
            pickle.dump(diagnostics_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(self.colored_text(f"Kernel diagnostics saved for node {time_node}: {os.path.basename(diag_path)}", "blue"))

        result_dict = {
            "gp":             best_info["gp"],
            "kernel":         best_info["kernel"],
            "label":          best_info["label"],
            "scaler_x":       aux["scaler_x"],
            "scaler_y":       aux["scaler_y"],
            "lml":            best_info["lml"],
            "train_rmse":     best_info["rmse"],
            "score":          best_info["score"],
            "time_node":      time_node,
            "parameter_dim":  aux["dim"],
            "time_coupled":   time_coupled,
            "fit_time":       best_info["fit_time"],
        }
        
        # Discard heavy data
        del kernel_diagnostics, aux
        gc.collect()
        
        if MEMORY_PROFILE:
            check_memory_usage(f"_gaussian_process_regression_t DONE node {time_node}")
        
        return result_dict

    
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

    
    # def fit_to_training_set(self, property, min_greedy_error=None, N_basis_vecs=None,
    #                         time_coupled=False, save_fits_to_file=True,
    #                         kernel_report=False,
    #                         gpr_fit_report=False,
    #                         no_file_load=False,
    #                         screening=True, refinement=True,
    #                         n_wfs_pred=5, n_q_slices=5,
    #                         error_metric="rms",
    #                         save_diagnostics=True
    #                         ):
    #     """
    #     Fit GP models for all empirical time nodes.

    #     Parameters
    #     ----------
    #     kernel_report : bool
    #         Per-node kernel analysis: PDF + diagnostics pickle for EACH time node.
    #     gpr_fit_report : bool
    #         Overall fit report: single PDF with diagnostics, worst predictions,
    #         and validation heatmaps covering ALL nodes at once.
    #         Generated AFTER the main loop finishes.
    #     report_dir : str
    #         Directory for per-node kernel reports.
    #     fit_report_dir : str
    #         Directory for the overall GPR fit report PDF.
    #     n_worst : int
    #         Number of worst nodes to highlight in the fit report.
    #     n_q_slices : int
    #         Mass-ratio slice count for validation heatmaps.
    #     error_metric : str
    #         Error metric for heatmaps ("rms", "max", "mean", etc.).
    #     """

    #     # [OPTIMIZED #2]
    #     if MEMORY_PROFILE:
    #         check_memory_usage(f"fit_to_training_set START for {property}")

    #     gpr_obj = self._get_gpr_obj(property)
    #     train_obj = self._get_training_obj(property)


    #     if N_basis_vecs is None and min_greedy_error is None:
    #         N_basis_vecs = self.resolve_property(N_basis_vecs, gpr_obj.N_basis_vecs)
    #         min_greedy_error = self.resolve_property(min_greedy_error, gpr_obj.min_greedy_error)
    #         if N_basis_vecs is None and min_greedy_error is None:
    #             print('Choose either settings for the amount of basis_vecs OR the minimum greedy error.')
    #             sys.exit(1)

    #     try:
    #         if no_file_load:
    #             raise FileNotFoundError
    #         gpr_obj = gpr_obj.load_gpr_obj()
    #         train_obj = train_obj.load()

    #         if kernel_report:
    #             for time_node in range(len(train_obj.empirical_indices)):
    #                 self._kernel_analysis_report(
    #                     gpr_obj=gpr_obj,
    #                     train_obj=train_obj,
    #                     kernel_diagnostics=None,  # Will load from disk
    #                     time_node=time_node,
    #                     time_coupled=time_coupled,
    #                     n_best_pred=n_wfs_pred,
    #                     n_q_slices=n_q_slices,
    #                 )


    #     except Exception as e:
    #         print(e)
    #         traceback.print_exc()
    #         if train_obj.training_set is None:
    #             try:
    #                 train_obj = train_obj.load()
    #             except Exception as e2:
    #                 print(e2)
    #                 train_obj = self.get_training_set_greedy(
    #                     property=property,
    #                     min_greedy_error=min_greedy_error,
    #                     N_greedy_vecs=N_basis_vecs
    #                 )

    #         print(f"Training set for {property} has {len(train_obj.basis_indices)} basis vectors")

    #         N_nodes = len(train_obj.empirical_indices)
    #         gpr_obj.gp_models = []
    #         gpr_obj.kernels = []
    #         gpr_obj.labels = []
    #         gpr_obj.scaler_x = []
    #         gpr_obj.scaler_y = []
    #         gpr_obj.best_lmls = np.zeros(N_nodes)
    #         gpr_obj.best_train_rmses = np.zeros(N_nodes)
    #         gpr_obj.best_scores = np.zeros(N_nodes)
    #         gpr_obj.fit_times = np.zeros(N_nodes)

    #         print(f'Interpolate {property}...')
    #         start2 = time.time()
    #         optimized_kernel = None

    #         for node_i in range(N_nodes):
    #             # [OPTIMIZED #2]: Periodic memory checkpoint
    #             if node_i % 10 == 0 and MEMORY_PROFILE:
    #                 check_memory_usage(f"Processing node {node_i}/{N_nodes}")

    #             best_result = self._gaussian_process_regression_t(
    #                 time_node=node_i,
    #                 train_obj=train_obj,
    #                 optimized_kernel=optimized_kernel,
    #                 time_coupled=time_coupled,
    #                 screening=screening,
    #                 refinement=refinement,
    #                 save_diagnostics=save_diagnostics
    #             )

    #             if kernel_report:
    #                 self._kernel_analysis_report(
    #                     gpr_obj=gpr_obj,
    #                     train_obj=train_obj,
    #                     time_node=node_i,
    #                     time_coupled=time_coupled,
    #                     n_best_pred=n_wfs_pred,
    #                     n_q_slices=n_q_slices,
    #                 )

    #             # Extract warm-start BEFORE cleanup
    #             optimized_kernel = [best_result["kernel"], best_result["label"]]

    #             # Store results
    #             gpr_obj.gp_models.append(best_result["gp"])
    #             gpr_obj.kernels.append(best_result["kernel"])
    #             gpr_obj.labels.append(best_result["label"])
    #             gpr_obj.scaler_x.append(best_result["scaler_x"])
    #             gpr_obj.scaler_y.append(best_result["scaler_y"])
    #             gpr_obj.best_lmls[node_i] = best_result["lml"]
    #             gpr_obj.best_train_rmses[node_i] = best_result["train_rmse"]
    #             gpr_obj.best_scores[node_i] = best_result["score"]
    #             gpr_obj.fit_times[node_i] = best_result["fit_time"]

    #             # [OPTIMIZED #2]: Free memory periodically
    #             if node_i % 5 == 0:
    #                 del best_result
    #                 gc.collect()

    #         end2 = time.time()
    #         print(f'time full GPR = {end2 - start2}')

    #         if save_fits_to_file:
    #             gpr_obj.save_gpr_obj()

    #     # ================================================================
    #     # OVERALL GPR FIT REPORT (after loop)
    #     # ================================================================
    #     if gpr_fit_report:
    #         self._gpr_fit_report(
    #             gpr_obj=gpr_obj,
    #             train_obj=train_obj,
    #             time_coupled=time_coupled,
    #             n_wfs_pred=n_wfs_pred,
    #             n_q_slices=n_q_slices,
    #             error_metric=error_metric,
    #         )

    #     if MEMORY_PROFILE:
    #         check_memory_usage(f"fit_to_training_set END for {property}")

    #     return gpr_obj

    def fit_to_training_set(self, property, min_greedy_error=None, N_basis_vecs=None,
                            time_coupled=False, save_fits_to_file=True,
                            kernel_report=False,
                            gpr_fit_report=False,
                            no_file_load=False,
                            screening=True, refinement=True,
                            n_wfs_pred=5, n_q_slices=5,
                            error_metric="rms",
                            save_diagnostics=True
                            ):
        """
        Fit GP models for all empirical time nodes.

        Parameters
        ----------
        kernel_report : bool
            Per-node kernel analysis: PDF + diagnostics pickle for EACH time node.
        gpr_fit_report : bool
            Overall fit report: single PDF with diagnostics, worst predictions,
            and validation heatmaps covering ALL nodes at once.
            Generated AFTER the main loop finishes.
        """

        # [OPTIMIZED #2]
        if MEMORY_PROFILE:
            check_memory_usage(f"fit_to_training_set START for {property}")

        gpr_obj = self._get_gpr_obj(property)
        train_obj = self._get_training_obj(property)


        if N_basis_vecs is None and min_greedy_error is None:
            N_basis_vecs = self.resolve_property(N_basis_vecs, gpr_obj.N_basis_vecs)
            min_greedy_error = self.resolve_property(min_greedy_error, gpr_obj.min_greedy_error)
            if N_basis_vecs is None and min_greedy_error is None:
                print('Choose either settings for the amount of basis_vecs OR the minimum greedy error.')
                sys.exit(1)

        try:
            if no_file_load:
                raise FileNotFoundError
            gpr_obj = gpr_obj.load_gpr_obj()
            train_obj = train_obj.load()

            if kernel_report:
                for time_node in range(len(train_obj.empirical_indices)):
                    self._kernel_analysis_report(
                        gpr_obj=gpr_obj,
                        train_obj=train_obj,
                        time_node=time_node,
                        time_coupled=time_coupled,
                        n_best_pred=n_wfs_pred,
                        n_q_slices=n_q_slices,
                    )


        except Exception as e:
            print(e)
            traceback.print_exc()
            if train_obj.training_set is None:
                try:
                    train_obj = train_obj.load()
                except Exception as e2:
                    print(e2)
                    train_obj = self.get_training_set_greedy(
                        property=property,
                        min_greedy_error=min_greedy_error,
                        N_greedy_vecs=N_basis_vecs
                    )

            print(f"Training set for {property} has {len(train_obj.basis_indices)} basis vectors")

            N_nodes = len(train_obj.empirical_indices)
            gpr_obj.gp_models = []
            gpr_obj.kernels = []
            gpr_obj.labels = []
            gpr_obj.scaler_x = []
            gpr_obj.scaler_y = []
            gpr_obj.best_lmls = np.zeros(N_nodes)
            gpr_obj.best_train_rmses = np.zeros(N_nodes)
            gpr_obj.best_scores = np.zeros(N_nodes)
            gpr_obj.fit_times = np.zeros(N_nodes)

            print(f'Interpolate {property}...')
            start2 = time.time()
            optimized_kernel = None

            for node_i in range(N_nodes):
                # [OPTIMIZED #2]: Periodic memory checkpoint
                if node_i % 10 == 0 and MEMORY_PROFILE:
                    check_memory_usage(f"Processing node {node_i}/{N_nodes}")

                best_result = self._gaussian_process_regression_t(
                    time_node=node_i,
                    train_obj=train_obj,
                    optimized_kernel=optimized_kernel,
                    time_coupled=time_coupled,
                    screening=screening,
                    refinement=refinement,
                    save_diagnostics=save_diagnostics
                )

                if kernel_report:
                    self._kernel_analysis_report(
                        gpr_obj=gpr_obj,
                        train_obj=train_obj,
                        time_node=node_i,
                        time_coupled=time_coupled,
                        n_best_pred=n_wfs_pred,
                        n_q_slices=n_q_slices,
                    )

                # Extract warm-start BEFORE cleanup
                optimized_kernel = [best_result["kernel"], best_result["label"]]

                # Store results
                gpr_obj.gp_models.append(best_result["gp"])
                gpr_obj.kernels.append(best_result["kernel"])
                gpr_obj.labels.append(best_result["label"])
                gpr_obj.scaler_x.append(best_result["scaler_x"])
                gpr_obj.scaler_y.append(best_result["scaler_y"])
                gpr_obj.best_lmls[node_i] = best_result["lml"]
                gpr_obj.best_train_rmses[node_i] = best_result["train_rmse"]
                gpr_obj.best_scores[node_i] = best_result["score"]
                gpr_obj.fit_times[node_i] = best_result["fit_time"]

                # [OPTIMIZED #2]: Free memory periodically
                if node_i % 5 == 0:
                    del best_result
                    gc.collect()

            end2 = time.time()
            print(f'time full GPR = {end2 - start2}')

            if save_fits_to_file:
                gpr_obj.save_gpr_obj()

        # ================================================================
        # OVERALL GPR FIT REPORT (after loop)
        # ================================================================
        if gpr_fit_report:
            self._gpr_fit_report(
                gpr_obj=gpr_obj,
                train_obj=train_obj,
                time_coupled=time_coupled,
                n_wfs_pred=n_wfs_pred,
                n_q_slices=n_q_slices,
                error_metric=error_metric,
            )

        if MEMORY_PROFILE:
            check_memory_usage(f"fit_to_training_set END for {property}")

        return gpr_obj
    
    def _gpr_fit_report(self, 
                        gpr_obj:GPRFitResults, 
                        train_obj:TrainingSetResults,
                        time_coupled=False,
                        n_wfs_pred=5,
                        n_q_slices=5,
                        error_metric="rms",
                        ):
        """Generate a comprehensive GPR fit report covering ALL time nodes."""

        report_dir = "Images/Gaussian_kernels/FitReports"
        os.makedirs(report_dir, exist_ok=True)

        report_path = gpr_obj.figname(
            prefix="gpr_fit_report",
            ext="pdf",
            directory=report_dir,
        )

        if MEMORY_PROFILE:
            check_memory_usage("_gpr_fit_report START")

        with PdfPages(report_path) as pdf:

            # ============================================================
            # SECTION 1 — FIT DIAGNOSTICS ACROSS ALL NODES
            # ============================================================
            try:
                result = self._plot_fit_diagnostics(
                    gpr_obj=gpr_obj,
                    save_fig=False,
                )

                # HANDLE EITHER Single Figure OR Tuple/List of Figures
                if result is not None:
                    figs_to_save = result if isinstance(result, (tuple, list)) else [result]
                    for fig in figs_to_save:
                        pdf.savefig(fig, bbox_inches="tight")
                        plt.close(fig)
                else:
                    # Fallback: capture any open figures the function created
                    for fig_num in plt.get_fignums():
                        fig = plt.figure(fig_num)
                        pdf.savefig(fig, bbox_inches="tight")
                        plt.close(fig)

                print(self.colored_text("Fit diagnostics added to report", "blue"))

            except Exception as e:
                print(self.colored_text(f"Fit diagnostics failed: {e}", "yellow"))
                traceback.print_exc()

            gc.collect()

            # ============================================================
            # SECTION 2 — FIT PREDICTIONS
            # ============================================================
            try:
                result = self._plot_fit_predictions(
                    gpr_obj=gpr_obj,
                    train_obj=train_obj,
                    n_wfs=n_wfs_pred,
                    save_fig=False,
                )

                if result is not None:
                    figs_to_save = result if isinstance(result, (tuple, list)) else [result]
                    for fig in figs_to_save:
                        pdf.savefig(fig, bbox_inches="tight")
                        plt.close(fig)
                else:
                    for fig_num in plt.get_fignums():
                        fig = plt.figure(fig_num)
                        pdf.savefig(fig, bbox_inches="tight")
                        plt.close(fig)

                print(self.colored_text("Fit predictions added to report", "blue"))

            except Exception as e:
                print(self.colored_text(f"Fit predictions plot failed: {e}", "yellow"))
                traceback.print_exc()

            gc.collect()

            # ============================================================
            # SECTION 3 — VALIDATION HEATMAPS
            # ============================================================
            try:
                result = self._plot_validation_heatmaps(
                    time_coupled=time_coupled,
                    gpr_obj=gpr_obj,
                    train_obj=train_obj,
                    n_q_slices=n_q_slices,
                    error_metric=error_metric,
                    save_fig=False,
                )

                if result is not None:
                    figs_to_save = result if isinstance(result, (tuple, list)) else [result]
                    for fig in figs_to_save:
                        pdf.savefig(fig, bbox_inches="tight")
                        plt.close(fig)
                else:
                    for fig_num in plt.get_fignums():
                        fig = plt.figure(fig_num)
                        pdf.savefig(fig, bbox_inches="tight")
                        plt.close(fig)

                print(self.colored_text("Validation heatmaps added to report", "blue"))

            except Exception as e:
                print(self.colored_text(f"Validation heatmaps failed: {e}", "yellow"))
                traceback.print_exc()

            gc.collect()

        # ================================================================
        # ENSURE ALL FIGURES ARE CLOSED
        # ================================================================
        plt.close("all")
        gc.collect()

        print(self.colored_text(f"GPR fit report saved: {report_path}", "blue"))

        if MEMORY_PROFILE:
            check_memory_usage("_gpr_fit_report END")

#################################################################################
# PLOTTING FUNCTIONS #
#################################################################################

    def _handle_figure_result(self, result, pdf):
        """
        Utility to handle plotting function returns (single or multiple figures).
        Saves all to PDF and closes them immediately to keep memory flat.
        
        Parameters
        ----------
        result : Figure | tuple|list | None
            Return value from a plotting function - may be single figure, 
            tuple/list of figures, or None.
        pdf : PdfPages object
            The PDF Pages writer to which figures should be saved.
        
        Returns
        -------
        bool
            True if at least one figure was processed successfully.
        """
        if result is None:
            # No explicit figures returned — fallback to capturing any open figures
            fig_nums = plt.get_fignums().copy()
            for fig_num in fig_nums:
                try:
                    fig = plt.figure(fig_num)
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                except Exception as e:
                    print(self.colored_text(
                        f"Failed to capture fallback figure #{fig_num}: {e}", "yellow"
                    ))
                    continue
            return len(fig_nums) > 0
        
        # Result might be single figure or tuple/list of figures
        figs_to_save = result if isinstance(result, (tuple, list)) else [result]
        
        success_count = 0
        for fig in figs_to_save:
            try:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                success_count += 1
            except Exception as e:
                print(self.colored_text(f"Failed to save/close figure: {e}", "yellow"))
                continue
        
        return success_count > 0

    def _plot_kernel_diagnostics(self, 
                                 kernel_diagnostics, 
                                 train_obj:TrainingSetResults, 
                                 save_fig=False
                                 ):
        """Plot kernel diagnostics with automatic figure cleanup"""
        
        labels = [k["label"] for k in kernel_diagnostics]
        x = np.arange(len(labels))

        lmls = np.array([k["lml"] for k in kernel_diagnostics])
        rmses = np.array([k["rmse"] for k in kernel_diagnostics])
        scores = np.array([k["score"] for k in kernel_diagnostics])
        kernel_fit_times = np.array([k["fit_time"] for k in kernel_diagnostics])

        n_plots = 4
        fig_kernel_scores, axes = plt.subplots(n_plots, 1, figsize=(14, 4.5 * n_plots), sharex=True, constrained_layout=True)

        ax = axes[0]
        ax.plot(x, lmls, marker="o", color="tab:blue")
        ax.set_ylabel("Log Marginal Likelihood")
        ax.axhline(np.mean(lmls), linestyle="--", color="gray", label="mean")
        ax.axhline(np.max(lmls), linestyle=":", color="green", label="best")
        ax.legend()

        ax = axes[1]
        ax.plot(x, rmses, marker="o", color="tab:orange")
        ax.set_ylabel("Train RMSE")
        ax.axhline(np.mean(rmses), linestyle="--", color="gray", label="mean")
        ax.axhline(np.min(rmses), linestyle=":", color="green", label="best")
        ax.legend()

        ax = axes[2]
        ax.plot(x, scores, marker="o", color="tab:purple")
        ax.set_ylabel("Score")
        ax.axhline(np.mean(scores), linestyle="--", color="gray", label="mean")
        ax.axhline(np.max(scores), linestyle=":", color="green", label="best")
        ax.legend()

        ax = axes[3]
        ax.plot(x, kernel_fit_times, marker="o", color="tab:red")
        ax.set_ylabel("Fit time (s)")
        ax.set_yscale("log")
        ax.legend()

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(labels, rotation=45, ha="right")
        plt.tight_layout()

        fig_lml_rmse, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(lmls, rmses, c=scores, cmap="viridis", s=80, edgecolor="k")
        ax.set_xlabel("Log Marginal Likelihood")
        ax.set_ylabel("Train RMSE")
        plt.colorbar(sc, ax=ax).set_label("Score")
        plt.tight_layout()

        if save_fig:
            figname = train_obj.figname(
                prefix="kernel_diagnostics", 
                ext="pdf", 
                directory="Images/Gaussian_kernels/Diagnostics", 
                include_greedy=True, 
                include_extra=f"node={kernel_diagnostics[0]['time_node']}"
                )
            with PdfPages(figname) as pdf:
                pdf.savefig(fig_kernel_scores, bbox_inches="tight")
                pdf.savefig(fig_lml_rmse, bbox_inches="tight")
            
            # Close figures after saving
            plt.close(fig_kernel_scores)
            plt.close(fig_lml_rmse)
            gc.collect()
            
            return None  # Caller doesn't need them—they went to file
        
        else:
            # Caller wants to handle figures themselves (e.g., embed in report PDF)
            return fig_kernel_scores, fig_lml_rmse  # Don't close—caller decides when


    def _plot_kernel_predictions(self,
        kernel_diagnostics,
        train_obj,
        scaler_y,
        y_train,
        n_best=5,
        save_fig=False,
    ):
        """
        Plots kernel comparison for GP models and computes best/worst kernels.

        Parameters
        ----------
        kernel_diagnostics : list[dict]
            Output container with kernel results.
        y_predict_kernels : list[np.ndarray]
            Full-grid predictions for each kernel.
        y_predict_std_kernels : list[np.ndarray]
            Full-grid std predictions for each kernel.
        train_obj : object
            Must implement:
                - load_residuals()
                - residuals
                - empirical_indices
                - basis_indices
        scaler_y : sklearn-like scaler
            Used for inverse transform / scaling.
        y_train : np.ndarray
            Training targets in original scale.
        time_node : int / str
            Identifier for the current time node.
        n_best : int
            Number of best kernels to display.

        Returns
        -------
        fig : matplotlib.figure.Figure
        best_info : dict
        """

        # ============================================================
        # LOAD TRUE VALUES
        # ============================================================
        train_obj.load_residuals()
        time_node = kernel_diagnostics[0]["time_node"]
        y_true_training_space = train_obj.residuals[:, train_obj.empirical_indices[time_node]]

        # ============================================================
        # COMPUTE BEST / WORST INDICES
        # ============================================================
        scores = np.array([d["score"] for d in kernel_diagnostics])

        best_idx = int(np.argmax(scores))

        best_indices = np.argsort(scores)[:n_best]
        best_indices = best_indices[np.argsort(scores[best_indices])]

        best_info = kernel_diagnostics[best_idx]

        # ============================================================
        # BEST MODEL PREPARATION
        # ============================================================
        best_y_predict = kernel_diagnostics[best_idx]["y_pred"]
        best_y_predict_std = kernel_diagnostics[best_idx]["y_pred_std"]

        best_y_train_pred = scaler_y.inverse_transform(
            best_info["y_train_pred"].reshape(-1, 1)
        ).flatten()


        # ============================================================
        # DIAGNOSTICS
        # ============================================================
        full_rmse = np.sqrt(np.mean((best_y_predict - y_true_training_space) ** 2))

        train_rmse_check = np.sqrt(
            np.mean(
                (best_y_train_pred - y_true_training_space[train_obj.basis_indices]) ** 2
            )
        )

        # ============================================================
        # PLOT SETUP
        # ============================================================
        fig_kernel_predictions, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(12, 10),
            gridspec_kw={"height_ratios": [3, 2]},
            sharex=True
        )

        x_axis = np.arange(len(best_y_predict))

        # ============================================================
        # TRUE VALUES
        # ============================================================
        ax1.plot(
            x_axis,
            y_true_training_space,
            color="blue",
            linewidth=2.5,
            label="True residuals"
        )

        # ============================================================
        # TRAINING POINTS
        # ============================================================
        ax1.scatter(
            x_axis[train_obj.basis_indices],
            y_train,
            color="red",
            s=40,
            zorder=10,
            label="Training points"
        )

        # ============================================================
        # BEST MODELS
        # ============================================================
        for rank, idx in enumerate(best_indices):
            y_pred = kernel_diagnostics[idx]["y_pred"]
            y_std = kernel_diagnostics[idx]["y_pred_std"]
            info = kernel_diagnostics[idx]

            ax1.plot(
                x_axis,
                y_pred,
                linewidth=1,
                alpha=0.6,
                label=f"Best {rank+1}: {info['label']} | score={info['score']:.2e}"
            )

            ax1.fill_between(
                x_axis,
                y_pred - 1.96 * y_std,
                y_pred + 1.96 * y_std,
                alpha=0.08
            )

            error = y_pred - y_true_training_space
            ax2.plot(x_axis, error, label=f"Best {rank+1}: {info['label']}")

        # ============================================================
        # BEST MODEL
        # ============================================================
        ax1.plot(
            x_axis,
            best_y_predict,
            color="black",
            linewidth=2.5,
            label=f"Best: {best_info['label']} | score={best_info['score']:.2e}"
        )

        ax1.fill_between(
            x_axis,
            best_y_predict - 1.96 * best_y_predict_std,
            best_y_predict + 1.96 * best_y_predict_std,
            color="black",
            alpha=0.12
        )

        error_best = best_y_predict - y_true_training_space
        ax2.plot(
            x_axis,
            error_best,
            color="limegreen",
            linewidth=2.5,
            label="Best model error"
        )

        # ============================================================
        # FINAL LABELS
        # ============================================================
        ax1.set_ylabel("Residual")
        ax2.set_ylabel(r"$y_{pred}$ - $y_{true}$")
        ax2.set_xlabel("Parameter grid index")

        ax1.set_title(
            f"GPR kernel comparison\n"
            f"node={time_node}\n"
            f"RMSE={full_rmse:.3e}, train RMSE={train_rmse_check:.3e}"
        )

        ax1.legend(fontsize=8)
        ax2.legend(fontsize=8)

        plt.tight_layout()

        if save_fig:
            figname = train_obj.figname(prefix="gp_kernel_comparison", directory="Images/Gaussian_kernels/Kernel_comparison", include_extra=f"T{time_node}")
            fig_kernel_predictions.savefig(figname, dpi=300)
        
            # [OPTIMIZED #5]: Close figure
            plt.close(fig_kernel_predictions)
            gc.collect()

        else:
            return fig_kernel_predictions
            
    


    def _plot_validation_heatmaps_kernels(
        self,
        time_coupled,
        kernel_diagnostics,
        scaler_x,
        scaler_y,
        time_node,
        train_obj: TrainingSetResults,
        n_q_slices=3,
        top_k=2
    ):
        """
        Heatmaps of relative prediction error for best kernels.

        Supports dynamic parameter selection:
        - max 3 effective varying dimensions
        - x1 + x2 → x_eff
        - remaining dims become slices
        """

        # ============================================================
        # VALIDATION DATA
        # ============================================================
        train_obj_truth_space = self.generate_truth_residuals(
            property=train_obj.property
        )

        X_val = train_obj_truth_space.parameter_grid

        t = np.full((X_val.shape[0], 1),
                    self.time[train_obj.empirical_indices[time_node]])

        if time_coupled:
            X_val = np.hstack([X_val, t])

        X_scaled = scaler_x.transform(X_val)

        truth_residuals = train_obj_truth_space.residuals
        y_true = truth_residuals[:, train_obj.empirical_indices[time_node]]

        eps = 1e-12

        # ============================================================
        # PARAMETER DETECTION (ORDERED)
        # ============================================================
        param_names = ["e", "l", "q", "x1", "x2"]
        X_base = X_val[:, :len(param_names)]

        varying = []
        for i, name in enumerate(param_names):
            if np.unique(X_base[:, i]).size > 1:
                varying.append(name)

        # ---- x1 + x2 compression ----
        if "x1" in varying and "x2" in varying:
            varying = [v for v in varying if v not in ["x1", "x2"]]
            varying.append("x_eff")

        if len(varying) > 3:
            raise ValueError(
                f"Too many varying parameters {varying}. "
                "Max supported is 3 (after x_eff compression)."
            )

        # ============================================================
        # BEST KERNELS
        # ============================================================
        best_indices = np.argsort(
            [d["score"] for d in kernel_diagnostics]
        )[-top_k:][::-1]

        best_kernels = [kernel_diagnostics[i] for i in best_indices]

        # ============================================================
        # Q SLICES (OR GENERIC LAST DIM SLICES)
        # ============================================================
        slice_dim = varying[-1] if len(varying) > 2 else None

        if slice_dim is not None:
            if slice_dim == "x_eff":
                x_eff = X_base[:, 3] + X_base[:, 4]
                slice_vals = np.unique(x_eff)
            else:
                idx = param_names.index(slice_dim)
                slice_vals = np.unique(X_base[:, idx])

            n_slices = min(n_q_slices, len(slice_vals))
            slice_indices = np.linspace(0, len(slice_vals) - 1, n_slices).astype(int)
            slices = slice_vals[slice_indices]
        else:
            slices = [None]

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
        n_slices = len(slices)

        # ============================================================
        # FIGURE LAYOUT
        # ============================================================
        fig = plt.figure(
            figsize=(5 * n_kernels + 1.2, 4 * n_slices)
        )

        gs = gridspec.GridSpec(
            n_slices,
            n_kernels + 1,
            width_ratios=[1] * n_kernels + [0.05],
            wspace=0.25,
            hspace=0.35
        )

        axes = np.empty((n_slices, n_kernels), dtype=object)
        cbar_axes = []

        for si in range(n_slices):
            for ki in range(n_kernels):
                axes[si, ki] = fig.add_subplot(gs[si, ki])
            cbar_axes.append(fig.add_subplot(gs[si, -1]))

        # ============================================================
        # PRECOMPUTE PREDICTIONS (UNCHANGED LOGIC)
        # ============================================================
        all_errors_per_kernel = []

        for k in best_kernels:
            gp = k["gp"]
            y_pred_scaled, _ = gp.predict(X_scaled, return_std=True)

            y_pred = scaler_y.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).flatten()

            rel_error = np.abs(y_pred - y_true) / (np.abs(y_true) + eps)
            all_errors_per_kernel.append(rel_error)

        all_errors_per_kernel = np.array(all_errors_per_kernel)

        vmin = np.nanmin(all_errors_per_kernel)
        vmax = np.nanmax(all_errors_per_kernel)
        norm = Normalize(vmin=vmin, vmax=vmax)

        # ============================================================
        # PLOTTING
        # ============================================================
        for si, s in enumerate(slices):

            if slice_dim is None:
                mask = np.ones(len(X_val), dtype=bool)

            else:
                if slice_dim == "x_eff":
                    x_eff = X_base[:, 3] + X_base[:, 4]
                    mask = np.isclose(x_eff, s)
                else:
                    idx = param_names.index(slice_dim)
                    mask = np.isclose(X_base[:, idx], s)

            for ki, k in enumerate(best_kernels):

                ax = axes[si, ki]

                err_vals = all_errors_per_kernel[ki][mask]

                X_sub = X_base[mask]

                # ====================================================
                # DETERMINE 2D AXES (FIRST TWO VARYING)
                # ====================================================
                active_dims = []
                for i, name in enumerate(param_names):
                    if name in varying and name != slice_dim:
                        active_dims.append(i)

                if len(active_dims) < 2:
                    raise ValueError("Need at least 2 varying dims for heatmap")

                d0, d1 = active_dims[:2]

                x_vals = X_sub[:, d1]
                y_vals = X_sub[:, d0]

                x_unique = np.unique(x_vals)
                y_unique = np.unique(y_vals)

                Z = np.full((len(y_unique), len(x_unique)), np.nan)

                for i in range(len(x_vals)):
                    yi = np.where(y_unique == y_vals[i])[0][0]
                    xi = np.where(x_unique == x_vals[i])[0][0]
                    Z[yi, xi] = err_vals[i]

                im = ax.imshow(
                    Z,
                    aspect="auto",
                    origin="lower",
                    norm=norm,
                    extent=[
                        x_unique.min(), x_unique.max(),
                        y_unique.min(), y_unique.max()
                    ],
                )

                title_extra = f"k={k['label']}"
                if slice_dim is not None:
                    title_extra += f", {slice_dim}={s:.3g}"

                ax.set_title(title_extra)

                ax.set_xlabel(param_names[d1])
                ax.set_ylabel(param_names[d0])

            cbar = fig.colorbar(
                axes[si, 0].images[0],
                cax=cbar_axes[si]
            )
            cbar.set_label(r"$|y_{pred} - y_{true}|$")

        # ============================================================
        # FINALIZE PDF
        # ============================================================
        fig.tight_layout()

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        pdf.close()

        print(self.colored_text(
            f"Saved kernel error PDF → {filepath}",
            "blue"
        ))

        return fig

    def _plot_parity(self,
                    best_result_dict,
                    train_obj:TrainingSetResults,
                    save_fig=False,
                    ):
        
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

        y_true = best_result_dict["y_train"]
        y_pred = best_result_dict["train_prediction"]

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
            f"GP Fit for best kernel T{best_result_dict['time_node']}: {best_result_dict['label']}\n"
            f"RMSE = {best_result_dict['train_rmse']:.3e} | lml = {best_result_dict['lml']:.3e} | score = {best_result_dict['score']:.3e}"
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
        ax2.set_ylabel(r"$y_{pred} - y_{true}$")
        ax2.grid(True)

        # optional: tighten view if needed
        ylim = np.percentile(np.abs(residuals), 95)
        ax2.set_ylim(-ylim, ylim)

        plt.tight_layout()

        # ------------------------------------------------------------
        # SAVE
        # ------------------------------------------------------------

        if save_fig:
            figname = train_obj.figname(prefix="gp_parity_residual",
                            directory="Images/Gaussian_kernels/Parity_check",
                            include_extra=f"T{best_result_dict['time_node']}")
            fig_parity.savefig(figname, dpi=300)
        else:
            return fig_parity

    
    def mismatch(self, h1, h2):
        h1_n = h1 / np.linalg.norm(h1)
        h2_n = h2 / np.linalg.norm(h2)
        pointwise_overlap = (h1_n * h2_n)
        return 1 - pointwise_overlap

    def generate_truth_residuals(self,
                            property,
                            ):
        # ============================================================
        # VALIDATION GRID
        # ============================================================
        train_obj_truth_space = TrainingSetResults(
            **self.result_kwargs_training(
                property=property,
                time=self.time,
                ecc_ref_space=self.ecc_ref_space_val,
                mean_ano_ref_space=self.mean_ano_ref_space_val,
                mass_ratio_space=self.mass_ratio_space_val,
                chi1_space=self.chi1_space_val,
                chi2_space=self.chi2_space_val,
                truncate_at_ISCO=False,
                truncate_at_tmin=False,
            )
        )
        
        # ============================================================
        # GET TRUTH RESIDUALS
        # ============================================================

        try:
            train_obj_truth_space.load_residuals()
        except Exception:
            self._calculate_residuals(
                train_obj=train_obj_truth_space,
            )

        return train_obj_truth_space
    
    def get_true_vs_pred(self,
                            time_coupled,
                            train_obj:TrainingSetResults,
                            gpr_obj:GPRFitResults,
                            get_mismatch=False,
                            get_diff=False,
                            get_relative_diff=False
                            ):
        if gpr_obj.validation_errors is None:

            train_obj_truth_space = self.generate_truth_residuals(
                property=gpr_obj.property
            )
            truth_residuals = train_obj_truth_space.residuals
            y_true = truth_residuals.T[train_obj.empirical_indices]

            X_val_base = train_obj_truth_space.parameter_grid

            
            
            y_pred = []

            for t_node, gp in enumerate(gpr_obj.gp_models):

                X_node = X_val_base.copy()

                if time_coupled:
                    t = np.full(
                        (X_val_base.shape[0], 1),
                        self.time[train_obj.empirical_indices[t_node]],
                    )

                    X_val = np.hstack([X_val_base, t])
                else:
                    X_val = X_val_base

                # Scale x
                X_scaled = gpr_obj.scaler_x[t_node].transform(X_node)

                # Predict y_pred with scaled x and y
                gp = gpr_obj.gp_models[t_node]

                y_pred_scaled, _ = gp.predict(
                    X_scaled,
                    return_std=True
                )

                # Unscale y
                y_pred_node = gpr_obj.scaler_y[t_node].inverse_transform(
                            y_pred_scaled.reshape(-1, 1)
                        ).flatten()

                y_pred.append(y_pred_node)

            if time_coupled:
                # collapse time dimension (or store full tensor if needed)
                y_pred = np.mean(y_pred, axis=0)

            y_pred = np.array(y_pred)

            results = {
                "y_true": y_true,
                "y_pred": y_pred,
                "X_val": X_val
            }

            def mismatch(h1, h2):
                    h1_n = h1 / np.linalg.norm(h1)
                    h2_n = h2 / np.linalg.norm(h2)
                    pointwise_overlap = (h1_n * h2_n)
                    return 1 - pointwise_overlap

            if get_mismatch:
                results["mismatch"] = mismatch(y_pred, y_true)
            
            if get_diff:
                results["diff"] = y_pred - y_true

            if get_relative_diff:
                eps = 1e-12
                results["relative_diff"] = np.abs((y_pred - y_true) / (y_true + eps))
            
            gpr_obj.validation_errors = results
        
        return gpr_obj.validation_errors


    def _plot_validation_heatmaps(
        self,
        time_coupled,
        gpr_obj:GPRFitResults,
        train_obj:TrainingSetResults,
        n_q_slices=3,
        error_metric='mismatch', # "mismatch" vs "difference"
        save_fig=False,
    ):
        """
        Heatmaps of relative prediction error for kernels.

        Layout:
        - rows = kernels (gp models)
        - columns = q-slices
        - 1 colorbar per row
        """

        if error_metric == "mismatch":
            get_mismatch = True
            get_diff = False
        elif error_metric == "difference":
            get_mismatch = False
            get_diff = True
        else:
            print(self.colored_text("Choose either 'mismatch' or 'difference' as error metric", 'red'))
            sys.exit(1)

        err_results = self.get_true_vs_pred(
            time_coupled=time_coupled,
            train_obj=train_obj,
            gpr_obj=gpr_obj,
            get_diff=get_diff,
            get_mismatch=get_mismatch,
        )

        if error_metric == "mismatch":
            errors = err_results["mismatch"]
        else:
            errors = np.abs(err_results["diff"])


        X_val = err_results["X_val"]

        ecc = X_val[:, 0]
        l = X_val[:, 1]
        q = X_val[:, 2]

        # ============================================================
        # Q SLICES
        # ============================================================
        q_vals = np.unique(q)
        n_q_slices = min(n_q_slices, len(q_vals))
        q_slices = np.linspace(0, len(q_vals) - 1, n_q_slices).astype(int)
        q_slices = q_vals[q_slices]

        print(f"Selected q slices: {q_slices}")

        # ============================================================
        # PLOTTING
        # ============================================================
        
        n_fits = len(gpr_obj.gp_models)

        norm = LogNorm(
            vmin=np.nanmin(errors),
            vmax=np.nanmax(errors),
        )

        # ============================================================
        # GRID
        # ============================================================
        fig = plt.figure(
            figsize=(5 * n_q_slices + 1.2, 4 * n_fits)
        )

        gs = gridspec.GridSpec(
            n_fits,
            n_q_slices + 1,
            width_ratios=[1] * n_q_slices + [0.05],
            wspace=0.25,
            hspace=0.35,
        )

        axes = np.empty((n_fits, n_q_slices), dtype=object)
        cbar_axes = []

        for i in range(n_fits):
            for j in range(n_q_slices):
                axes[i, j] = fig.add_subplot(gs[i, j])

            cbar_axes.append(fig.add_subplot(gs[i, -1]))

        # ============================================================
        # PLOTTING
        # ============================================================
        for qi, qv in enumerate(q_slices):

            mask_q = np.isclose(q, qv)

            for gp_id in range(n_fits):

                ax = axes[gp_id, qi]

                rel_error = errors[gp_id]
                # print(f'in err ; T{gp_id}', gpr_obj.labels[gp_id])
                err_vals = rel_error[mask_q]

                e_vals = ecc[mask_q]
                l_vals = l[mask_q]

                e_unique = np.unique(e_vals)
                l_unique = np.unique(l_vals)

                Z = np.full((len(e_unique), len(l_unique)), np.nan)

                for k in range(len(e_vals)):
                    ei = np.where(e_unique == e_vals[k])[0][0]
                    li = np.where(l_unique == l_vals[k])[0][0]
                    Z[ei, li] = err_vals[k]

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
                ax.set_title(f"T{gp_id}: k={gpr_obj.labels[gp_id]}, q={qv:.3g}")

                ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
                ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(style='sci', axis='x', scilimits=(-3, 3))

                if qi == n_q_slices - 1:
                    ax.set_xlabel("l")
                else:
                    ax.set_xlabel("")

                ax.set_ylabel("e")

        # ============================================================
        # COLORBARS (FIXED: ONE PER ROW)
        # ============================================================
        for i in range(n_fits):

            cbar = fig.colorbar(
                axes[i, 0].images[0],
                cax=cbar_axes[i],
            )

            if error_metric == 'difference':
                cbar.set_label(r"$|y_{\rm pred}-y_{\rm true}|$")
            else:
                cbar.set_label(r"$1 - \hat{y}_{\mathrm{pred}}(x)\,\hat{y}_{\mathrm{true}}(x)$")

        # ============================================================
        # FINALIZE
        # ============================================================
        fig.tight_layout()

        if save_fig:
            filepath = gpr_obj.filename(
                    prefix="kernel_error_heatmaps",
                    ext="pdf",
                    directory="Images/Fit_Diagnostics/Heatmaps",
                )
            pdf = PdfPages(filepath)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pdf.close()

            print(self.colored_text(f"Saved kernel error PDF → {filepath}", "blue"))
        else:
            return fig
    
    def _plot_fit_predictions(self,
        gpr_obj:GPRFitResults,
        train_obj:TrainingSetResults,
        n_wfs=5,
        save_fig=False,
        ):
        """
        Plots kernel comparison for GP models and computes best/worst kernels.

        Parameters
        ----------
        kernel_diagnostics : list[dict]
            Output container with kernel results.
        y_predict_kernels : list[np.ndarray]
            Full-grid predictions for each kernel.
        y_predict_std_kernels : list[np.ndarray]
            Full-grid std predictions for each kernel.
        train_obj : object
            Must implement:
                - load_residuals()
                - residuals
                - empirical_indices
                - basis_indices
        scaler_y : sklearn-like scaler
            Used for inverse transform / scaling.
        y_train : np.ndarray
            Training targets in original scale.
        time_node : int / str
            Identifier for the current time node.
        n_worst : int
            Number of worst kernels to display.

        Returns
        -------
        fig : matplotlib.figure.Figure
        best_info : dict
        """

        # ============================================================
        # VALIDATION GRID
        # ============================================================

        err_results = self.get_true_vs_pred(
            time_coupled=False,
            train_obj=train_obj,
            gpr_obj=gpr_obj,
            get_diff=True,
            get_mismatch=True,
        )

        X_val = err_results["X_val"]

        # ============================================================
        # PLOT SETUP
        # ============================================================
        fig_worst_predictions, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(
            6, 1,
            figsize=(12, 12),
            gridspec_kw={"height_ratios": [3, 1, 1, 3, 1, 1]},
            sharex=True
        )

        # Plot colors
        colors = plt.cm.tab10(np.linspace(0, 1, n_wfs*2))
        # Pick n_wfs based on lowest score
        sorted_indices = np.argsort(gpr_obj.best_scores)
        best_gp_indices = sorted_indices[:n_wfs]
        worst_gp_indices = sorted_indices[-n_wfs:][::-1]
        indices_to_plot = np.concatenate([best_gp_indices, worst_gp_indices])
        print(indices_to_plot, best_gp_indices, worst_gp_indices)

        for idx, time_node in enumerate(indices_to_plot):
            print(f"Plotting T{time_node}, {idx}")

            color = colors[idx]

            y_pred = err_results["y_pred"][time_node]
            y_true = err_results["y_true"][time_node]
            mismatch = err_results["mismatch"][time_node]
            error_diff = err_results["diff"][time_node]
            
            # ============================================================
            # KEEP TRACK OF VARYING AND CONSTANT PARAMETERS FOR LABELING
            # ============================================================


            # Parameter label
            param_names = ["e", "l", "q", "x1", "x2"]
            param_fmts  = [".3f", ".2f", ".1f", ".2f", ".2f"]

            values = [X_val[:, j] for j in range(X_val.shape[1])]
            vary = [len(np.unique(v)) > 1 for v in values]

            varying_params = []

            for i, (name, fmt) in enumerate(zip(param_names, param_fmts)):
                if vary[i]:
                    varying_params.append(f"{name}={X_val[time_node, i]:{fmt}}")

            x_label = "(" + ", ".join(varying_params) + ")"

            fixed_params = []

            for i, (name, fmt) in enumerate(zip(param_names, param_fmts)):
                if not vary[i]:
                    fixed_params.append(f"{name}={X_val[0, i]:{fmt}}")

            fixed_title = ""
            if fixed_params:
                fixed_title = " | fixed: " + ", ".join(fixed_params)
            
            # ============================================================
            # PLOT N-WORST
            # ============================================================


            x_axis = np.arange(len(y_pred))

            if time_node in worst_gp_indices:
                print(f"Plotting worst kernel T{time_node}, {idx}")
                # Truth (only first gets generic legend entry)
                ax1.plot(
                    x_axis,
                    y_true,
                    color=color,
                    linestyle="--",
                    linewidth=2
                )

                # Prediction
                ax1.plot(
                    x_axis,
                    y_pred,
                    color=color,
                    linewidth=1.5
                )

                # Invisible line to create one legend entry per parameter set
                ax1.plot(
                    [],
                    [],
                    color=color,
                    linewidth=4,
                    label=x_label,
                )

                ax2.plot(
                    x_axis,
                    error_diff,
                    color=color,
                    linewidth=2,
                )

                ax3.plot(
                    x_axis,
                    mismatch,
                    color=color,
                    linewidth=2,
                )
            
            if time_node in best_gp_indices:
                print(f"Plotting best kernel T{time_node}, {idx}")
                # Truth (only first gets generic legend entry)
                ax4.plot(
                    x_axis,
                    y_true,
                    color=color,
                    linestyle="--",
                    linewidth=2
                )

                # Prediction
                ax4.plot(
                    x_axis,
                    y_pred,
                    color=color,
                    linewidth=1.5
                )

                # Invisible line to create one legend entry per parameter set
                ax4.plot(
                    [],
                    [],
                    color=color,
                    linewidth=4,
                    label=x_label,
                )

                ax5.plot(
                    x_axis,
                    error_diff,
                    color=color,
                    linewidth=2,
                )

                ax6.plot(
                    x_axis,
                    mismatch,
                    color=color,
                    linewidth=2,
                )
        # ============================================================
        # FINAL LABELS
        # ============================================================

        # Add generic legend entries once
        ax1.plot(
            [], [],
            color="black",
            linestyle="--",
            linewidth=2,
            label="Truth ",
        )

        ax1.plot(
            [], [],
            color="black",
            linestyle="-",
            linewidth=2,
            label="Prediction",
        )

        # Add generic legend entries once
        ax2.plot(
            [], [],
            color="black",
            linewidth=2,
            label="Prediction error",
        )

        # Add generic legend entries once
        ax3.plot(
            [], [],
            color="black",
            linewidth=2,
            label="Mismatch",
        )

        if train_obj.property == "amplitude":
            ax1.set_ylabel(r"$\Delta A$")
        else:
            ax1.set_ylabel(r"$\Delta \phi$")
        ax2.set_ylabel(r"$y_{\rm pred}-y_{\rm true}$")
        ax3.set_ylabel(r"$1 - \hat{y}_{\mathrm{pred}}(x)\,\hat{y}_{\mathrm{true}}(x)$")
        ax2.set_xlabel("Parameter grid index")

        ax1.set_title(
            f"Worst {n_wfs} predictions and their true residuals{fixed_title}"
        )

        ax4.set_title(
            f"Best {n_wfs} predictions and their true residuals{fixed_title}"
        )

        ax1.legend()
        ax2.legend()
        ax3.legend()

        plt.tight_layout()

        if save_fig:
            figname = train_obj.figname(prefix=f"worst_{n_wfs}_gp",
                                directory="Images/Fit_Diagnostics/Worst_predictions",
            )
            fig_worst_predictions.savefig(figname, dpi=300)
        else:
            return fig_worst_predictions
    

    def _plot_fit_diagnostics(
        self,
        gpr_obj:GPRFitResults,
        save_fig=False
    ):
        """
        KERNEL DIAGNOSTICS PLOTTING MODES FOR BEST FIT KERNELS

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

        labels = []
        for i, l in enumerate(gpr_obj.labels):
            labels.append(f"{l} (T{i})")

        x = np.arange(len(labels))

        # ============================================================
        # SELECT DATA MODE
        # ============================================================

        lmls = gpr_obj.best_lmls
        rmses = gpr_obj.best_train_rmses
        scores = gpr_obj.best_scores


        lml_label = "Log Marginal Likelihood"
        rmse_label = "Train RMSE"
        score_label = r"$N_{\mathrm{LML}} - \frac{1}{2}N_{\mathrm{RMSE}}$"
                
        kernel_fit_times = gpr_obj.fit_times
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
        ax.set_title(f"Kernel diagnostics for all nodes")

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

        plt.tight_layout()

        # ============================================================
        # X LABELS
        # ============================================================
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(labels, rotation=45, ha="right")



        # ============================================================
        # HISTOGRAM KERNELS PLOT
        # ============================================================

        counts = Counter(gpr_obj.labels)

        fig_kernel_freq, ax = plt.subplots(
            figsize=(10, 4),
            constrained_layout=True
        )

        ax.bar(counts.keys(), counts.values())

        ax.set_ylabel("Wins")
        ax.set_title("Kernel selection frequency")

        ax.tick_params(axis="x", rotation=30)

        plt.tight_layout()


        if save_fig:

            figname = gpr_obj.figname(
                prefix="fit_diagnostics",
                ext="pdf",
                directory="Images/Fit_Diagnostics/Errors",
            )

            with PdfPages(figname) as pdf:

                # ============================================================
                # 1. MAIN FIT DIAGNOSTICS
                # ============================================================
                pdf.savefig(fig_kernel_scores, bbox_inches="tight")

                # ============================================================
                # 2. KERNEL SELECTION FREQUENCY
                # ============================================================
                pdf.savefig(fig_kernel_freq, bbox_inches="tight")
        
        else:
            return fig_kernel_freq, fig_kernel_scores


sampling_frequency = 2048 # or 4096
duration = 4 # seconds
time_array = np.linspace(-duration, 0, int(sampling_frequency * duration))  # time in seconds

gt = Generate_Offline_Surrogate(time_array=time_array, 
                                ecc_ref_parameterspace=np.linspace(0.001, 0.3, num=60),
                                mean_ano_parameterspace=[0],
                                mass_ratio_parameterspace=[1],
                                chi1_parameterspace=[0],
                                chi2_parameterspace=[0],
                                sampling_val_ecc_ref=0.01,
                                sampling_val_mean_ano=0.1,
                                sampling_val_mass_ratio=0.5,
                                min_greedy_error_amp=1e-6,
                                min_greedy_error_phase=1e-6,
                                # N_basis_vecs_amp=60,
                                # N_basis_vecs_phase=60,
                                training_set_selection='greedy')

# train_obj_p = gt._get_training_obj('phase')

# gt.generate_property_dataset(train_obj=train_obj_p, 
#                             #  plot_residuals_eccentric_evolve=True,
#                             #  plot_residuals_time_evolve=True,
#                              )

# train_amp = gt.get_training_set_greedy(
#     property='amplitude', 
#     max_tree_depth=0,
#     save_greedy_errors=False,
#     save_orthonormal_basis=False,
#     save_train_obj=False,
#     # plot_training_set=True, 
#     # plot_residuals_time=True,
#     plot_greedy_error=True,
#     # save_fig_training_set=True,
#     # plot_emp_nodes_on_basis=True,
#     # save_fig_emp_nodes_on_basis=True,
#     plot_basis_indices=True,
#     show_legend_ts=True
#     )

# train_phase = gt.get_training_set_greedy(
#     property='phase', 
#     max_tree_depth=0,
#     save_greedy_errors=False,
#     save_orthonormal_basis=False,
#     save_train_obj=False,
#     # plot_training_set=True, 
#     # plot_residuals_time=True,
#     plot_greedy_error=True,
#     # save_fig_training_set=True,
#     # plot_emp_nodes_on_basis=True,
#     # save_fig_emp_nodes_on_basis=True,
#     plot_basis_indices=True,
#     show_legend_ts=True
#     )

# plt.show()
gt.fit_to_training_set('amplitude',
                       time_coupled=False,
                       save_fits_to_file=True,
                       kernel_report=True,
                       gpr_fit_report=True,
                       screening=True,
                       refinement=True,
                       error_metric='mismatch',
                       n_wfs_pred=5,
                    )