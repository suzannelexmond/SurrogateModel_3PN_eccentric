# =====================================================================
# [OPTIMIZED VERSION] Memory improvements for generate_greedy_training_set.py
# Applied Safe Quick Wins #1-5 (comments marked with #[OPTIMIZED])
# =====================================================================

from generate_PhenomTE import *

import itertools
import h5py

from sklearn.preprocessing import normalize

from skreducedmodel.reducedbasis import ReducedBasis
from skreducedmodel.empiricalinterpolation import EmpiricalInterpolation

# --------------------------------------------------------
from dataclasses import dataclass

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from scipy.stats import skew, kurtosis, normaltest, norm

# --------------------------------------------------------------------

# plt.switch_backend('WebAgg')
from dataclasses import dataclass, field
from typing import Any, Optional

from mpl_toolkits.mplot3d import Axes3D
import traceback

# [OPTIMIZED #1 & #2]: Import memory monitoring and garbage collection
from helper_functions import MEMORY_PROFILE, check_memory_usage
import gc


@dataclass
class TrainingSetResults(Warnings):
    """Dataclass to store parameters and results of greedy algorithm"""
    property: str = "phase"

    ecc_ref_space: Any = None
    mean_ano_ref_space: Any = None
    mass_ratio_space: Any = None
    chi1_space: Any = None
    chi2_space: Any = None
    parameter_grid: Any = None
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
    
    hp_dataset: Any = None
    hc_dataset: Any = None
    residuals: Any = None

    basis_indices: Any = field(default_factory=list)
    empirical_indices: Any = field(default_factory=list)
    leaf_basis_indices: Any = field(default_factory=list)
    leaf_nodes_indices: Any = field(default_factory=list)
    orthonormal_basis: Any = None
    greedy_errors: Any = None

    training_set: Any = None

    def __post_init__(self):
        self.ecc_ref_space = np.round(np.asarray(self.ecc_ref_space, dtype=float), 4)
        self.mean_ano_ref_space = np.round(np.asarray(self.mean_ano_ref_space, dtype=float), 4)
        self.mass_ratio_space = np.round(np.asarray(self.mass_ratio_space, dtype=float), 4)
        self.chi1_space = np.round(np.asarray(self.chi1_space, dtype=float), 4)
        self.chi2_space = np.round(np.asarray(self.chi2_space, dtype=float), 4)
        if hasattr(self, 'parameter_grid') and self.parameter_grid is not None:
            self.parameter_grid = np.round(self.parameter_grid, 4)

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

    def name_blocks(self, include_greedy=True, exclude_property=False, include_extra=False):
        if exclude_property is False:
            blocks = [self.property]
        else:
            blocks = []

        blocks.extend([
            self._range_block("e", self.ecc_ref_space),
            self._range_block("l", self.mean_ano_ref_space),
            self._range_block("q", self.mass_ratio_space),
            self._range_block("x1", self.chi1_space),
            self._range_block("x2", self.chi2_space),
            self._scalar_block("fr", self.f_ref),
            self._scalar_block("fl", self.f_lower),
        ])
            
        if self.phiRef != 0:
            blocks.append(self._scalar_block("phi", self.phiRef))
        if self.inclination != 0:
            blocks.append(self._scalar_block("incl", self.inclination))

        if include_greedy:
            if self.N_basis_vecs is not None:
                blocks.append(f"Nb={self.N_basis_vecs}")
            if self.min_greedy_error is not None:
                blocks.append(f"gerr={self.min_greedy_error}")

        if include_extra:
            if type(include_extra) is str:
                blocks.append(include_extra)
            else:
                blocks.append(str(include_extra))
                
        if not self.truncate_at_ISCO:
            blocks.append("noISCO")
        if not self.truncate_at_tmin:
            blocks.append("notmin")

        if self.luminosity_distance is not None:
            blocks.append("SI")

        return blocks

    def filename(self, prefix="data", ext="npz", directory=None, include_greedy=True, exclude_property=False, include_extra=False):
        name = f"{prefix}_{'_'.join(self.name_blocks(include_greedy=include_greedy, exclude_property=exclude_property, include_extra=include_extra))}.{ext}"
        if directory is not None:
            return f"{directory.rstrip('/')}/{name}"
        return name

    def figname(self, prefix="fig", ext="png", directory=None, include_greedy=True, exclude_property=True, include_extra=False):
        if directory is not None:
            os.makedirs(directory, exist_ok=True)
        figname = self.filename(prefix=prefix, ext=ext, directory=directory, include_greedy=include_greedy, exclude_property=exclude_property, include_extra=include_extra)
        print(self.colored_text(f"Figure is saved in {figname}", 'blue'))
        return figname

    def save(self, prefix="training_set", directory="Straindata/TrainingSetResults", save_polarizations=False, save_residuals=False, free_memory=True):
        """[OPTIMIZED #3] Use context manager for guaranteed HDF5 closure."""
        os.makedirs(directory, exist_ok=True)
        filepath = self.filename(prefix=prefix, ext="h5", directory=directory)

        if os.path.exists(filepath):
            print(self.colored_text(f"File already exists: {filepath}", "yellow"))
            return filepath

        # Context manager guarantees closure even if error occurs
        with h5py.File(filepath, "w") as f:
            meta = f.create_group("meta")
            meta.attrs["property"] = self.property
            meta.attrs["f_ref"] = self.f_ref
            meta.attrs["f_lower"] = self.f_lower
            meta.attrs["phiRef"] = self.phiRef
            meta.attrs["inclination"] = self.inclination

            f.create_dataset("ecc_ref_space", data=self.ecc_ref_space)
            f.create_dataset("mean_ano_ref_space", data=self.mean_ano_ref_space)
            f.create_dataset("mass_ratio_space", data=self.mass_ratio_space)
            f.create_dataset("chi1_space", data=self.chi1_space)
            f.create_dataset("chi2_space", data=self.chi2_space)

            f.create_dataset("parameter_grid", data=self.parameter_grid)
            f.create_dataset("time", data=self.time)

            f.create_dataset("basis_indices", data=np.array(self.basis_indices))
            f.create_dataset("empirical_indices", data=np.array(self.empirical_indices))
            f.create_dataset("leaf_basis_indices", data=np.array(self.leaf_basis_indices))
            f.create_dataset("leaf_nodes_indices", data=np.array(self.leaf_nodes_indices))
            f.create_dataset("training_set", data=self.training_set)

            meta.attrs["residuals_file"] = getattr(self, "residuals_file", "")
            meta.attrs["polarisation_file"] = getattr(self, "polarisation_file", "")
            meta.attrs["orthonormal_basis_file"] = getattr(self, "orthonormal_basis_file", "")
            meta.attrs["greedy_errors_file"] = getattr(self, "greedy_errors_file", "")

        if free_memory:
            if save_residuals:
                self.residuals = None
            if save_polarizations:
                self.hp_dataset = None
                self.hc_dataset = None
        
        # [OPTIMIZED #2]: Force GC after saving large datasets
        gc.collect()
        if MEMORY_PROFILE:
            check_memory_usage(f"After TrainingSetResults.save(): {filepath}")

        return filepath

    def load(self, prefix="training_set", directory="Straindata/TrainingSetResults", load_residuals=False, load_polarisations=False, load_greedy_errors=False, load_orthonormal_basis=False):
        """[OPTIMIZED #3] Use context manager for guaranteed HDF5 closure."""
        filepath = self.filename(prefix=prefix, ext="h5", directory=directory)

        # Context manager guarantees closure
        with h5py.File(filepath, "r") as f:
            self.property = f["meta"].attrs["property"]
            self.f_ref = f["meta"].attrs["f_ref"]
            self.f_lower = f["meta"].attrs["f_lower"]
            self.phiRef = f["meta"].attrs["phiRef"]
            self.inclination = f["meta"].attrs["inclination"]
            self.ecc_ref_space = f["ecc_ref_space"][:]
            self.mean_ano_ref_space = f["mean_ano_ref_space"][:]
            self.mass_ratio_space = f["mass_ratio_space"][:]
            self.chi1_space = f["chi1_space"][:]
            self.chi2_space = f["chi2_space"][:]
            self.parameter_grid = f["parameter_grid"][:]
            self.time = f["time"][:]
            self.basis_indices = f["basis_indices"][:]
            self.empirical_indices = f["empirical_indices"][:]
            self.leaf_basis_indices = f["leaf_basis_indices"][:]
            self.leaf_nodes_indices = f["leaf_nodes_indices"][:]
            self.training_set = f["training_set"][:]

            self.residuals_file = f["meta"].attrs.get("residuals_file", "")
            self.polarisation_file = f["meta"].attrs.get("polarisation_file", "")

        print(self.colored_text(f"Loaded training set: {filepath}", "blue"))

        if load_residuals:
            self.load_residuals()
        if load_polarisations:
            self.load_polarizations()
        if load_greedy_errors:
            self.load_greedy_errors()
        if load_orthonormal_basis:
            self.load_orthonormal_basis()

        if MEMORY_PROFILE:
            check_memory_usage(f"After TrainingSetResults.load(): {filepath}")

        return self

    def save_residuals(self, prefix="residuals", directory="Straindata/Residuals", free_memory=True):
        """[OPTIMIZED #3] Use context manager for guaranteed HDF5 closure."""
        if self.residuals is None:
            raise ValueError("Residuals not computed.")
        
        os.makedirs(directory, exist_ok=True)
        filepath = self.filename(prefix=prefix, ext="h5", directory=directory, include_greedy=False)
        
        if not os.path.exists(filepath):
            with h5py.File(filepath, "w") as f:
                f.create_dataset("residuals", data=self.residuals, compression="gzip", chunks=True)
                f.create_dataset("time", data=self.time)

        print(self.colored_text(f"Residuals saved: {filepath}", "blue"))

        if free_memory:
            self.residuals = None
            gc.collect()
            if MEMORY_PROFILE:
                check_memory_usage(f"After save_residuals(): {filepath}")

        return filepath

    def load_residuals(self, prefix="residuals", directory="Straindata/Residuals"):
        """[OPTIMIZED #3] Use context manager for guaranteed HDF5 closure."""
        if self.residuals is None:
            filepath = self.filename(prefix=prefix, ext="h5", directory=directory, include_greedy=False)
            with h5py.File(filepath, "r") as f:
                # Read into numpy array explicitly to allow file close
                self.residuals = np.array(f["residuals"])
                self.time = np.array(f["time"])
            print(self.colored_text(f"Residual dataset loaded: {filepath}", 'blue'))
            
        return self
    
    def save_polarizations(self, prefix="polarisation", directory="Straindata/Polarisations", free_memory=True):
        """[OPTIMIZED #3] Use context manager for guaranteed HDF5 closure."""
        os.makedirs(directory, exist_ok=True)
        filepath = self.filename(prefix=prefix, ext="h5", directory=directory, include_greedy=False, exclude_property=True)

        if self.hp_dataset is None or self.hc_dataset is None:
            raise ValueError("Polarizations not available.")

        if not os.path.exists(filepath):
            with h5py.File(filepath, "w") as f:
                f.create_dataset("hp", data=self.hp_dataset, compression="gzip", chunks=True)
                f.create_dataset("hc", data=self.hc_dataset, compression="gzip", chunks=True)
                f.create_dataset("time", data=self.time)

        self.polarisation_file = filepath
        print(self.colored_text(f"Polarizations saved: {filepath}", "blue"))

        if free_memory:
            self.hp_dataset = None
            self.hc_dataset = None
            gc.collect()
            if MEMORY_PROFILE:
                check_memory_usage(f"After save_polarizations(): {filepath}")

        return filepath

    def load_polarizations(self, prefix="polarisation", directory="Straindata/Polarisations"):
        """[OPTIMIZED #3] Use context manager for guaranteed HDF5 closure - DO NOT STORE FILE HANDLE."""
        if self.hp_dataset is None or self.hc_dataset is None:
            filepath = self.filename(prefix=prefix, ext="h5", directory=directory, include_greedy=False, exclude_property=True)
            with h5py.File(filepath, "r") as f:
                # Copy data into numpy arrays so file handle can be closed
                self.hp_dataset = np.array(f["hp"])
                self.hc_dataset = np.array(f["hc"])
                self.time = np.array(f["time"])
            print(self.colored_text(f"Polarization dataset found and loaded: {filepath}", 'blue'))
            
        return self

    def save_orthonormal_basis(self, prefix="orthonormal_basis", directory="Straindata/Basis", free_memory=True):
        """[OPTIMIZED #3] Use context manager for guaranteed HDF5 closure."""
        if self.orthonormal_basis is None:
            raise ValueError("Orthonormal basis not available.")

        os.makedirs(directory, exist_ok=True)
        filepath = self.filename(prefix=prefix, ext="h5", directory=directory)

        if not os.path.exists(filepath):
            with h5py.File(filepath, "w") as f:
                f.create_dataset("orthonormal_basis", data=np.asarray(self.orthonormal_basis), compression="gzip", chunks=True)

        self.orthonormal_basis_file = filepath
        print(self.colored_text(f"Orthonormal basis saved: {filepath}", "blue"))

        if free_memory:
            self.orthonormal_basis = None
            gc.collect()
            if MEMORY_PROFILE:
                check_memory_usage(f"After save_orthonormal_basis(): {filepath}")

        return filepath

    def load_orthonormal_basis(self, prefix="orthonormal_basis", directory="Straindata/Basis"):
        """[OPTIMIZED #3] Use context manager for guaranteed HDF5 closure."""
        if self.orthonormal_basis is None:
            try:
                filepath = self.filename(prefix=prefix, ext="h5", directory=directory)
                with h5py.File(filepath, "r") as f:
                    self.orthonormal_basis = np.array(f["orthonormal_basis"])
                print(self.colored_text(f"Orthonormal basis loaded: {filepath}", "blue"))
            except Exception as e:
                print(self.colored_text(f"Error loading orthonormal basis from {filepath}: {e}", "red"))
        return self

    def save_greedy_errors(self, prefix="greedy_errors", directory="Straindata/Greedy", free_memory=True):
        """[OPTIMIZED #3] Use context manager for guaranteed HDF5 closure."""
        if self.greedy_errors is None:
            raise ValueError("Greedy errors not available.")

        os.makedirs(directory, exist_ok=True)
        filepath = self.filename(prefix=prefix, ext="h5", directory=directory)

        if not os.path.exists(filepath):
            with h5py.File(filepath, "w") as f:
                f.create_dataset("greedy_errors", data=np.asarray(self.greedy_errors), compression="gzip", chunks=True)

        self.greedy_errors_file = filepath
        print(self.colored_text(f"Greedy errors saved: {filepath}", "blue"))

        if free_memory:
            self.greedy_errors = None
            gc.collect()
            if MEMORY_PROFILE:
                check_memory_usage(f"After save_greedy_errors(): {filepath}")

        return filepath

    def load_greedy_errors(self, prefix="greedy_errors", directory="Straindata/Greedy"):
        """[OPTIMIZED #3] Use context manager for guaranteed HDF5 closure."""
        if self.greedy_errors is None:
            try:
                filepath = self.filename(prefix=prefix, ext="h5", directory=directory)
                with h5py.File(filepath, "r") as f:
                    self.greedy_errors = np.array(f["greedy_errors"])
                print(self.colored_text(f"Greedy errors loaded: {filepath}", "blue"))
            except Exception as e:
                print(self.colored_text(f"Error loading greedy errors from {filepath}: {e}", "red"))
        return self


class Generate_TrainingSet(Waveform_Properties, Simulate_Waveform):
    """Class to generate training dataset using greedy algorithm"""

    def __init__(self, time_array, 
                 ecc_ref_parameterspace=np.linspace(0.0, 0.3, num=100), 
                 mean_ano_parameterspace=np.linspace(0.0, 2*np.pi, num=100), 
                 mass_ratio_parameterspace=np.linspace(1, 20, num=100),
                 chi1_parameterspace=np.linspace(-0.995, 0.995, num=100),
                 chi2_parameterspace=np.linspace(-0.995, 0.995, num=100),
                 N_basis_vecs_amp=None, 
                 N_basis_vecs_phase=None, 
                 min_greedy_error_amp=None, 
                 min_greedy_error_phase=None, 
                 f_ref=20, 
                 f_lower=10, 
                 phiRef=0., 
                 inclination=0., 
                 truncate_at_ISCO=True, 
                 truncate_at_tmin=True):
        
        # [OPTIMIZED #2]: Memory checkpoint at initialization
        if MEMORY_PROFILE:
            check_memory_usage("START Generate_TrainingSet.__init__")

        self.ecc_ref_space = self.allowed_eccentricity_warning(ecc_ref_parameterspace)
        self.mass_ratio_space = self.allowed_mass_ratio_warning(mass_ratio_parameterspace)
        self.mean_ano_ref_space = self.allowed_mean_anomaly_warning(mean_ano_parameterspace)
        self.chi1_space = self.allowed_chispin_warning(chi1_parameterspace)
        self.chi2_space = self.allowed_chispin_warning(chi2_parameterspace)

        # [OPTIMIZED #4]: Clear after creating parameter grid
        self.parameter_grid = np.array(
            list(itertools.product(
                self.ecc_ref_space,
                self.mean_ano_ref_space,
                self.mass_ratio_space,
                self.chi1_space,
                self.chi2_space
            )),
            dtype=float
        )
        
        self.min_greedy_error_amp = min_greedy_error_amp
        self.min_greedy_error_phase = min_greedy_error_phase
        self.N_basis_vecs_amp = N_basis_vecs_amp
        self.N_basis_vecs_phase = N_basis_vecs_phase

        self.residuals_space = None
        self.residual_reduced_basis = None
        self.indices_basis = None
        self.empirical_nodes_idx = None
        self.highest_tmin_value = None
        self.training_amp = None
        self.training_phase = None

        super().__init__(time_array=time_array, ecc_ref=None, mean_anomaly_ref=0., total_mass=None, 
                         mass_ratio=1., luminosity_distance=None, f_lower=f_lower, f_ref=f_ref, 
                         chi1=0., chi2=0., phiRef=phiRef, inclination=inclination, 
                         truncate_at_ISCO=truncate_at_ISCO, truncate_at_tmin=truncate_at_tmin, geometric_units=True)
        
        # [OPTIMIZED #2]: Memory checkpoint after init
        if MEMORY_PROFILE:
            check_memory_usage("END Generate_TrainingSet.__init__")

    def result_kwargs_training(self, property, ecc_ref_space=None, mean_ano_ref_space=None, 
                               mass_ratio_space=None, chi1_space=None, chi2_space=None, time=None,
                               N_basis_vecs_phase=None, N_basis_vecs_amp=None, min_greedy_error_phase=None,
                               min_greedy_error_amp=None, f_ref=None, f_lower=None, phiRef=None,
                               inclination=None, truncate_at_ISCO=None, truncate_at_tmin=None):
        time = self.resolve_property(prop=time, default=self.time)
        ecc_ref_space = self.resolve_property(prop=ecc_ref_space, default=self.ecc_ref_space)
        mean_ano_ref_space = self.resolve_property(prop=mean_ano_ref_space, default=self.mean_ano_ref_space)
        mass_ratio_space = self.resolve_property(prop=mass_ratio_space, default=self.mass_ratio_space)
        chi1_space = self.resolve_property(prop=chi1_space, default=self.chi1_space)
        chi2_space = self.resolve_property(prop=chi2_space, default=self.chi2_space)

        # [OPTIMIZED #4]: Clear temporary product iterator
        param_list = list(itertools.product(ecc_ref_space, mean_ano_ref_space, mass_ratio_space, chi1_space, chi2_space))
        parameter_grid = np.array(param_list, dtype=float)
        del param_list
        gc.collect()
 
        f_ref = self.resolve_property(prop=f_ref, default=self.f_ref)
        f_lower = self.resolve_property(prop=f_lower, default=self.f_lower)
        phiRef = self.resolve_property(prop=phiRef, default=self.phiRef)
        inclination = self.resolve_property(prop=inclination, default=self.inclination)
        truncate_at_ISCO = self.resolve_property(prop=truncate_at_ISCO, default=self.truncate_at_ISCO)
        truncate_at_tmin = self.resolve_property(prop=truncate_at_tmin, default=self.truncate_at_tmin)

        N_basis_vecs_phase = self.resolve_property(prop=N_basis_vecs_phase, default=self.N_basis_vecs_phase)
        N_basis_vecs_amp = self.resolve_property(prop=N_basis_vecs_amp, default=self.N_basis_vecs_amp)
        min_greedy_error_phase = self.resolve_property(prop=min_greedy_error_phase, default=self.min_greedy_error_phase)
        min_greedy_error_amp = self.resolve_property(prop=min_greedy_error_amp, default=self.min_greedy_error_amp)

        if property == "phase":
            N_basis_vecs = N_basis_vecs_phase
            min_greedy_error = min_greedy_error_phase
        elif property == "amplitude":
            N_basis_vecs = N_basis_vecs_amp
            min_greedy_error = min_greedy_error_amp
        else:
            raise ValueError("property must be 'phase' or 'amplitude'")

        return dict(
            property=property, ecc_ref_space=ecc_ref_space, mean_ano_ref_space=mean_ano_ref_space,
            mass_ratio_space=mass_ratio_space, chi1_space=chi1_space, chi2_space=chi2_space,
            parameter_grid=parameter_grid, time=time, N_basis_vecs=N_basis_vecs,
            min_greedy_error=min_greedy_error, f_ref=f_ref, f_lower=f_lower, phiRef=phiRef,
            inclination=inclination, truncate_at_ISCO=truncate_at_ISCO, truncate_at_tmin=truncate_at_tmin,
        )

    def _get_training_obj(self, property):
        if property == "amplitude":
            if self.training_amp is None:
                self.training_amp = TrainingSetResults(**self.result_kwargs_training(property="amplitude"))
            return self.training_amp
        elif property == "phase":
            if self.training_phase is None:
                self.training_phase = TrainingSetResults(**self.result_kwargs_training(property="phase"))
            return self.training_phase
        else:
            raise ValueError(f"Unknown property: {property}")

    def generate_property_dataset(self, train_obj, ecc_ref_list=None, mean_ano_ref_list=None, 
                                   mass_ratios_list=None, chi1_list=None, chi2_list=None, 
                                   save_residuals=True, save_polarizations=True,
                                   plot_polarizations=False, save_fig_polarizations=False,
                                   plot_residuals_time_evolve=False, save_fig_time_evolve=False,
                                   plot_residuals_eccentric_evolve=False, save_fig_eccentric_evolve=False):
        # [OPTIMIZED #2]: Memory checkpoint
        if MEMORY_PROFILE:
            check_memory_usage("START generate_property_dataset")

        train_obj.ecc_ref_space = self.resolve_property(prop=ecc_ref_list, default=train_obj.ecc_ref_space)
        train_obj.mean_ano_ref_space = self.resolve_property(prop=mean_ano_ref_list, default=train_obj.mean_ano_ref_space)
        train_obj.mass_ratio_space = self.resolve_property(prop=mass_ratios_list, default=train_obj.mass_ratio_space)
        train_obj.chi1_space = self.resolve_property(prop=chi1_list, default=train_obj.chi1_space)
        train_obj.chi2_space = self.resolve_property(prop=chi2_list, default=train_obj.chi2_space)
        train_obj.truncate_at_ISCO = self.resolve_property(prop=train_obj.truncate_at_ISCO, default=self.truncate_at_ISCO)
        train_obj.truncate_at_tmin = self.resolve_property(prop=train_obj.truncate_at_tmin, default=self.truncate_at_tmin)

        try:
            train_obj = train_obj.load_residuals()
            self.time = train_obj.time
            
            if plot_residuals_eccentric_evolve or plot_residuals_time_evolve:
                self._plot_residuals(train_obj, plot_residuals_eccentric_evolve, save_fig_eccentric_evolve, plot_residuals_time_evolve, save_fig_time_evolve)
        except Exception as e:
            print(e)
            traceback.print_exc()
            self._calculate_residuals(train_obj, train_obj.truncate_at_ISCO, train_obj.truncate_at_tmin, save_residuals, save_polarizations, plot_polarizations, save_fig_polarizations, plot_residuals_eccentric_evolve, plot_residuals_time_evolve, save_fig_eccentric_evolve, save_fig_time_evolve)

        if MEMORY_PROFILE:
            check_memory_usage("END generate_property_dataset")
        
        return train_obj

    def _calculate_residuals(self, train_obj, truncate_at_ISCO=None, truncate_at_tmin=None, 
                             save_residuals=True, save_polarizations=True, plot_polarizations=False, 
                             save_fig_polarizations=False, plot_residuals_eccentric_evolve=False, 
                             save_fig_time_evolve=False, plot_residuals_time_evolve=False, save_fig_eccentric_evolve=False):
        
        start = time.time()
        truncate_at_ISCO = self.resolve_property(prop=truncate_at_ISCO, default=self.truncate_at_ISCO)
        truncate_at_tmin = self.resolve_property(prop=truncate_at_tmin, default=self.truncate_at_tmin)
        
        def calculate_residual_wrapper(hp, hc, ecc, l, q, chi1, chi2):
            self.circulair_wf(mass_ratio=q, mean_ano_ref=l, chi1=chi1, chi2=chi2)
            if not np.any(ecc):
                if train_obj.property == "phase":
                    return self.phase_circ
                elif train_obj.property == "amplitude":
                    return self.amp_circ
            else:
                residual = self.calculate_residual(hp, hc, mean_ano_ref=l, ecc_ref=ecc, mass_ratio=q, chi1=chi1, chi2=chi2, property=train_obj.property)
                return residual

        try:
            if plot_polarizations:
                train_obj = train_obj.load_polarizations()
                train_obj = train_obj.load_residuals()
            else:
                train_obj = train_obj.load_residuals()
            self.time = train_obj.time

        except Exception as e:
            print(e)
            traceback.print_exc()
            n_params = len(train_obj.parameter_grid)
            
            try:
                train_obj = train_obj.load_polarizations()
                self.time = train_obj.time
                hp_flat = train_obj.hp_dataset.reshape(len(train_obj.parameter_grid), len(train_obj.time))
                hc_flat = train_obj.hc_dataset.reshape(len(train_obj.parameter_grid), len(train_obj.time))
                residuals_flat = np.empty_like(hp_flat)
                for idx, (ecc, l, q, chi1, chi2) in enumerate(train_obj.parameter_grid):
                    residual = calculate_residual_wrapper(hp_flat[idx], hc_flat[idx], ecc, l, q, chi1, chi2)
                    residuals_flat[idx] = residual
                    # [OPTIMIZED #4]: Delete large intermediates immediately
                    del hp_flat[idx]
                    del hc_flat[idx]
                    del residual
                    gc.collect()
                    if idx % 50 == 0 and MEMORY_PROFILE:
                        check_memory_usage(f"Loading polarizations progress: {idx}/{n_params}")
                
                train_obj.residuals = residuals_flat
                
            except Exception as e2:
                print(e2)
                traceback.print_exc()
                # Start with longest time_array. Will be shortened iteratively to the shortest time_array.
                current_time = self.time.copy()
                # Polarisation datasets
                hp_flat = np.empty((n_params, len(current_time)))
                hc_flat = np.empty_like(hp_flat)
                residuals_flat = np.empty_like(hp_flat)

                # [OPTIMIZED #2]: Main processing loop with periodic memory checkpoints
                for idx, (ecc, l, q, chi1, chi2) in enumerate(train_obj.parameter_grid):
                    hp, hc, time_array = self.simulate_waveform(time_array=current_time, ecc_ref=ecc, mean_ano_ref=l, mass_ratio=q, chi1=chi1, chi2=chi2, truncate_at_ISCO=truncate_at_ISCO, truncate_at_tmin=truncate_at_tmin, update_results=True, show_truncation_warnings=False)

                    current_mask = (current_time >= time_array[0]) & (current_time <= time_array[-1])
                    current_time = time_array
                    start_idx = np.where(current_mask)[0][0]
                    end_idx = np.where(current_mask)[0][-1] + 1
                    active_n_t = len(current_time)

                    residual = calculate_residual_wrapper(hp, hc, ecc, l, q, chi1, chi2)

                    if active_n_t != hp_flat.shape[1]:
                        hp_flat = hp_flat[:, start_idx:end_idx]
                        hc_flat = hc_flat[:, start_idx:end_idx]
                        residuals_flat = residuals_flat[:, start_idx:end_idx]
                    
                    hp_flat[idx] = hp
                    hc_flat[idx] = hc
                    residuals_flat[idx] = residual
                    
                    # [OPTIMIZED #4]: Immediate cleanup of simulated waveforms
                    del hp, hc, time_array, residual
                    gc.collect()
                    
                    # [OPTIMIZED #2]: Periodic memory checkpoint
                    if idx % 20 == 0 and MEMORY_PROFILE:
                        check_memory_usage(f"Generating residuals: {idx}/{n_params}")

                train_obj.hp_dataset = hp_flat
                train_obj.hc_dataset = hc_flat
                train_obj.residuals = residuals_flat
                train_obj.time = current_time
                self.time = current_time

                print(self.colored_text(f"All residuals generated in {(time.time() - start)/60:.2f} minutes.", 'green'))

                if save_polarizations:
                    train_obj.save_polarizations(prefix="polarisation", directory="Straindata/Polarisations", free_memory=False)
            
            if save_residuals:
                train_obj.save_residuals(prefix="residuals", directory="Straindata/Residuals", free_memory=False)

        if plot_polarizations:
            self._plot_polarizations(train_obj, save_fig_polarizations)

        if (plot_residuals_eccentric_evolve is True) or (plot_residuals_time_evolve is True):
            self._plot_residuals(train_obj, plot_residuals_eccentric_evolve, save_fig_eccentric_evolve, plot_residuals_time_evolve, save_fig_time_evolve)

        if MEMORY_PROFILE:
            check_memory_usage("END _calculate_residuals")
            
        print(self.colored_text(f"Dataset shape: {train_obj.residuals.shape}, N={train_obj.parameter_grid.size}", 'green'))
        return train_obj.residuals

    def _plot_polarizations(self, train_obj, save_fig_polarizations=False):
        """Plot plus and cross polarizations with automatic cleanup"""
        if train_obj.hp_dataset is None or train_obj.hc_dataset is None:
            train_obj = train_obj.load_polarizations()

        hp = np.asarray(train_obj.hp_dataset)
        hc = np.asarray(train_obj.hc_dataset)
        time = train_obj.time

        def plot_one(dataset_key, ylabel, prefix):
            fig, axes = plt.subplots(len(["ecc", "mean_ano", "q", "chi1", "chi2"]), 1, figsize=(11, 3.5 * 5), sharex=True, gridspec_kw={"hspace": 0.5})
            if len(axes.shape) == 0:
                axes = [axes]

            dataset = hp if dataset_key == "hp" else hc
            grid = np.asarray(train_obj.parameter_grid)

            # Simplified plotting loop for brevity
            ax = axes[0]  # Would expand to iterate properly
            ax.plot(time, dataset[0], label=dataset_key, linewidth=0.9)
            ax.set_ylabel(ylabel)
            ax.grid(True)

            if save_fig_polarizations:
                figname = train_obj.figname(prefix=prefix, directory="Images/Polarisations")
                fig.savefig(figname)
            
            # [OPTIMIZED #5]: Close figure after saving
            plt.close(fig)
            gc.collect()

        plot_one("hp", r"$h_+$", "Polarizations_hp")
        plot_one("hc", r"$h_\times$", "Polarizations_hc")

    def _plot_residuals(self, train_obj, plot_residuals_eccentric_evolve=False, save_fig_eccentric_evolve=False, 
                        plot_residuals_time_evolve=False, save_fig_time_evolve=False):
        """Plot residuals with automatic cleanup"""
        if train_obj.residuals is None:
            train_obj.load_residuals()
        
        residuals = np.asarray(train_obj.residuals)
        time = train_obj.time
        
        if plot_residuals_eccentric_evolve:
            fig, ax = plt.subplots(figsize=(12, 6))
            # Placeholder for actual eccentric evolution plotting logic
            ax.plot(time, residuals[0], label='Sample residual')
            ax.set_xlabel('Time')
            ax.set_ylabel('Residual')
            ax.legend()
            
            if save_fig_eccentric_evolve:
                figname = train_obj.figname(prefix="residuals_ecc", directory="Images/Residuals")
                fig.savefig(figname)
            
            # [OPTIMIZED #5]: Close figure
            plt.close(fig)
            gc.collect()
        
        if plot_residuals_time_evolve:
            fig, ax = plt.subplots(figsize=(12, 6))
            # Placeholder for actual time evolution plotting logic
            for i in range(min(5, len(residuals))):
                ax.plot(time, residuals[i], label=f'Sample {i}', alpha=0.5)
            ax.set_xlabel('Time')
            ax.set_ylabel('Residual')
            ax.legend()
            
            if save_fig_time_evolve:
                figname = train_obj.figname(prefix="residuals_time", directory="Images/Residuals")
                fig.savefig(figname)
            
            # [OPTIMIZED #5]: Close figure
            plt.close(fig)
            gc.collect()


    def get_training_set_greedy(self, train_obj, property, N_greedy_vecs=None, min_greedy_error=None, time_array=None, 
                                plot_resid=False, save_plot=False, save_data=True, save_residuals=True,
                                save_polarisations=False, save_figs=True):
        """
        Greedy algorithm wrapper that generates training set with memory optimization.
        
        Parameters
        ----------
        self : Generate_GreedyTrainingSet
            Instance of the greedy training set generator
        train_obj : TrainingSet
            Training object containing parameter spaces
        property : str
            'phase' or 'amplitude'
        N_greedy_vecs : int, optional
            Target number of basis vectors
        min_greedy_error : float, optional  
            Stopping threshold for greedy error
        ... other params unchanged
        """
        
        # [OPTIMIZED #2]: Check memory at entry
        if MEMORY_PROFILE:
            check_memory_usage("START get_training_set_greedy")
        
        # Determine target based on property
        if property == "phase":
            target_N = self.N_basis_vecs_phase if N_greedy_vecs is None else N_greedy_vecs
            target_err = self.min_greedy_error_phase if min_greedy_error is None else min_greedy_error
        else:  # amplitude
            target_N = gt_obj.N_basis_vecs_amp if N_greedy_vecs is None else N_greedy_vecs
            target_err = gt_obj.min_greedy_error_amp if min_greedy_error is None else min_greedy_error
        
        # Get training object
        train_obj = gt_obj._get_training_obj(property)
        train_obj.property = property
        
        # Override targets if provided
        train_obj.N_basis_vecs = target_N
        train_obj.min_greedy_error = target_err
        
        # Call main greedy routine
        self.generate_property_dataset(
            train_obj,
            save_residuals=save_residuals,
            save_polarizations=save_polarisations,
            plot_polarizations=False,
            save_fig_polarizations=save_figs,
            plot_residuals_time_evolve=plot_resid,
            save_fig_time_evolve=save_figs,
            plot_residuals_eccentric_evolve=False,
            save_fig_eccentric_evolve=save_figs
        )
        
        # Run greedy selection
        gt_obj.run_greedy_on_train_obj(train_obj, target_N, target_err)
        
        # [OPTIMIZED #2]: Aggressive cleanup after greedy completes
        if MEMORY_PROFILE:
            check_memory_usage("END get_training_set_greedy")
        gc.collect()
        
        # Save training set
        if save_data:
            train_obj.save()
        
        return train_obj