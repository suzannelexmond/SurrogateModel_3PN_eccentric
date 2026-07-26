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
    mass_ratio_space: Any = None
    chi1_space: Any = None
    chi2_space: Any = None

    parameter_grid: Any = None
    time: Any = None
    mean_anomaly: Any = None
    l_to_t_mapping: Any = None

    parametrization = 'mean_anomaly'

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

    # File paths
    residuals_file: str = ""
    polarisation_file: str = ""
    orthonormal_basis_file: str = ""
    greedy_errors_file: str = ""

    def __post_init__(self):
        self.ecc_ref_space = np.round(np.asarray(self.ecc_ref_space, dtype=float), 4)
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
            'l',
            self._range_block("e", self.ecc_ref_space),
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
            f.create_dataset("mass_ratio_space", data=self.mass_ratio_space)
            f.create_dataset("chi1_space", data=self.chi1_space)
            f.create_dataset("chi2_space", data=self.chi2_space)

            f.create_dataset("parameter_grid", data=self.parameter_grid)
            f.create_dataset("mean_anomaly", data=self.mean_anomaly)
            f.create_dataset("l_to_t_mapping", data=self.l_to_t_mapping)

            self.basis_indices = list(f["basis_indices"][:])
            self.empirical_indices = list(f["empirical_indices"][:])
            self.leaf_basis_indices = list(f["leaf_basis_indices"][:])
            self.leaf_nodes_indices = list(f["leaf_nodes_indices"][:])
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
            self.mass_ratio_space = f["mass_ratio_space"][:]
            self.chi1_space = f["chi1_space"][:]
            self.chi2_space = f["chi2_space"][:]
            self.parameter_grid = f["parameter_grid"][:]
            self.mean_anomaly = f["mean_anomaly"][:]
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
                f.create_dataset("l_to_t_mapping", data=self.l_to_t_mapping)
                f.create_dataset("mean_anomaly", data=self.mean_anomaly)

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
                self.l_to_t_mapping = np.array(f["l_to_t_mapping"])
                self.mean_anomaly = np.array(f["mean_anomaly"])
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
    """
        Parameters:
        ----------------
        time_array [s], np.array : Time array in seconds.
        ecc_ref_parameterspace [dimensionless], np.array : Array of reference eccentricities.
        mean_ano_parameterspace [rad], np.array : Array of reference mean anomalies.
        mass_ratio_parameterspace [dimensionless], np.array : Array of reference mass ratios.
        chi1_parameterspace [dimensionless], np.array : Array of reference primary spins.
        chi2_parameterspace [dimensionless], np.array : Array of reference secondary spins.
        f_lower [Hz], float: Start frequency of the waveform
        f_ref [Hz], float: Reference frequency at which the waveform parameters (eccentricity, spin, ...) are defined.
        inclination [rad], float : Inclination angle of the binary system. Defaults to 0 (face-on).
        luminosity_distance [Mpc], float : Luminosity distance of the binary in megaparsecs. Defaults to 100 Mpc.
        phiRef [rad], float : Reference phase of the waveform at the reference frequency. Defaults to 0.
        truncate_at_ISCO [bool] : Whether to truncate the waveform at the ISCO. Defaults to True.
        truncate_at_tmin [bool] : Whether to truncate the waveform at the minimum time in the time array. Defaults to True.
        
        !Either N_basis_vecs_amp or min_greedy_error_amp can be specified to set the stopping criterion for the greedy algorithm for amplitude residuals. If both are None, no stopping criterion is applied and all basis vectors will be selected.
        !Either N_basis_vecs_phase or min_greedy_error_phase can be specified to set the stopping criterion for the greedy algorithm for phase residuals. If both are None, no stopping criterion is applied and all basis vectors will be selected.

        N_basis_vecs_amp [int] : Maximum number of basis vectors for amplitude residuals. If None, no limit is applied.
        N_basis_vecs_phase [int] : Maximum number of basis vectors for phase residuals. If None, no limit is applied.
        min_greedy_error_amp [float] : Minimum greedy error for amplitude residuals. If None, no minimum error threshold is applied.
        min_greedy_error_phase [float] : Minimum greedy error for phase residuals. If None, no minimum error threshold is applied.
        
        """

    def __init__(self, 
                 time_array, 
                 ecc_ref_parameterspace=np.linspace(0.0, 0.3, num=100), 
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
        self.chi1_space = self.allowed_chispin_warning(chi1_parameterspace)
        self.chi2_space = self.allowed_chispin_warning(chi2_parameterspace)

        # [OPTIMIZED #4]: Clear after creating parameter grid
        self.parameter_grid = np.array(
            list(itertools.product(
                self.ecc_ref_space,
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

        # Inherit parameters from all previously defined classes
        super().__init__(time_array=time_array, 
                         ecc_ref=None, 
                         mean_anomaly_ref=0.,
                         total_mass=None, 
                         mass_ratio=None, 
                         luminosity_distance=None, 
                         f_lower=f_lower, 
                         f_ref=f_ref, 
                         chi1=None, 
                         chi2=None, 
                         phiRef=phiRef, 
                         inclination=inclination, 
                         truncate_at_ISCO=truncate_at_ISCO, 
                         truncate_at_tmin=truncate_at_tmin,
                         geometric_units=True,
                         parametrization='mean_anomaly'
                         )

        # [OPTIMIZED #2]: Memory checkpoint after init
        if MEMORY_PROFILE:
            check_memory_usage("END Generate_TrainingSet.__init__")

    def result_kwargs_training(self, property, time=None,
                               ecc_ref_space=None, mass_ratio_space=None, chi1_space=None, chi2_space=None, 
                               N_basis_vecs_phase=None, N_basis_vecs_amp=None, min_greedy_error_phase=None,
                               min_greedy_error_amp=None, f_ref=None, f_lower=None, phiRef=None,
                               inclination=None, truncate_at_ISCO=None, truncate_at_tmin=None):
        # Mean anomaly domain for the waveforms
        time = self.resolve_property(prop=time, default=self.time)
        
        # Initial parameters of the binary BBH system
        ecc_ref_space = self.resolve_property(prop=ecc_ref_space, default=self.ecc_ref_space)
        mass_ratio_space = self.resolve_property(prop=mass_ratio_space, default=self.mass_ratio_space)
        chi1_space = self.resolve_property(prop=chi1_space, default=self.chi1_space)
        chi2_space = self.resolve_property(prop=chi2_space, default=self.chi2_space)

        # [OPTIMIZED #4]: Clear temporary product iterator
        param_list = list(itertools.product(ecc_ref_space, mass_ratio_space, chi1_space, chi2_space))
        parameter_grid = np.array(param_list, dtype=float)
        del param_list
        gc.collect()
 
        f_ref = self.resolve_property(prop=f_ref, default=self.f_ref)
        f_lower = self.resolve_property(prop=f_lower, default=self.f_lower)
        phiRef = self.resolve_property(prop=phiRef, default=self.phiRef)
        inclination = self.resolve_property(prop=inclination, default=self.inclination)
        
        # Waveform truncation settings
        truncate_at_ISCO = self.resolve_property(prop=truncate_at_ISCO, default=self.truncate_at_ISCO)
        truncate_at_tmin = self.resolve_property(prop=truncate_at_tmin, default=self.truncate_at_tmin)
        
        # Greedy algorithm stopping criteria
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
            property=property, ecc_ref_space=ecc_ref_space,
            mass_ratio_space=mass_ratio_space, chi1_space=chi1_space, chi2_space=chi2_space,
            parameter_grid=parameter_grid, time=time, 
            N_basis_vecs=N_basis_vecs, min_greedy_error=min_greedy_error, 
            f_ref=f_ref, f_lower=f_lower, phiRef=phiRef, inclination=inclination, 
            truncate_at_ISCO=truncate_at_ISCO, truncate_at_tmin=truncate_at_tmin,
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

    def generate_property_dataset(self, train_obj:TrainingSetResults, 
                                   ecc_ref_list=None, mean_ano_ref_list=None,
                                   mass_ratios_list=None, chi1_list=None, chi2_list=None, 
                                   save_residuals=True, save_polarizations=True,
                                   plot_polarizations=False, save_fig_polarizations=False,
                                   plot_residuals_time_evolve=False, save_fig_time_evolve=False,
                                   plot_residuals_eccentric_evolve=False, save_fig_eccentric_evolve=False):
        """
        Generates a dataset of waveform residuals based on the specified property for a certain range of eccentricities (ecc).

        Parameters:
        ----------
        ecc_list : list of floats
            List of reference eccentricities for which to calculate residuals.
        mean_ano_ref_list : list of floats
            List of reference mean anomalies for which to calculate residuals.
        train_obj : TrainingSetResults
            The training set results object for the specified property.
        save_dataset_to_file : bool, optional
            If True, saves the generated dataset to a file.
        plot_residuals : bool, optional
            If True, plots the residuals for each eccentricity.
        save_fig : bool, optional
            If True, saves the residual plot to Images/Residuals.
        show_legend : bool, optional
            If True, displays the legend on the plot.

        Returns:
        -------
        residual_dataset : ndarray
            Array of residuals for each eccentricity.
        """
        
        # Memory checkpoint
        if MEMORY_PROFILE:
            check_memory_usage("START generate_property_dataset")

        train_obj.ecc_ref_space = self.resolve_property(prop=ecc_ref_list, default=train_obj.ecc_ref_space)
        train_obj.mass_ratio_space = self.resolve_property(prop=mass_ratios_list, default=train_obj.mass_ratio_space)
        train_obj.chi1_space = self.resolve_property(prop=chi1_list, default=train_obj.chi1_space)
        train_obj.chi2_space = self.resolve_property(prop=chi2_list, default=train_obj.chi2_space)
        train_obj.truncate_at_ISCO = self.resolve_property(prop=train_obj.truncate_at_ISCO, default=self.truncate_at_ISCO)
        train_obj.truncate_at_tmin = self.resolve_property(prop=train_obj.truncate_at_tmin, default=self.truncate_at_tmin)

        try:
            train_obj = train_obj.load_residuals()
            
            if plot_residuals_eccentric_evolve or plot_residuals_time_evolve:
                self._plot_residuals(train_obj, 
                                     plot_residuals_eccentric_evolve, 
                                     save_fig_eccentric_evolve, 
                                     plot_residuals_time_evolve, 
                                     save_fig_time_evolve)
        except Exception as e:
            print(e)
            traceback.print_exc()
            self._calculate_residuals(train_obj, 
                                      train_obj.truncate_at_ISCO, 
                                      train_obj.truncate_at_tmin, 
                                      save_residuals, 
                                      save_polarizations, 
                                      plot_polarizations, 
                                      save_fig_polarizations, 
                                      plot_residuals_eccentric_evolve, 
                                      plot_residuals_time_evolve, 
                                      save_fig_eccentric_evolve, 
                                      save_fig_time_evolve)

        if MEMORY_PROFILE:
            check_memory_usage("END generate_property_dataset")
        
        return train_obj

    # def _calculate_residuals(self, train_obj:TrainingSetResults, truncate_at_ISCO=None, truncate_at_tmin=None, 
    #                          save_residuals=True, save_polarizations=True, plot_polarizations=False, 
    #                          save_fig_polarizations=False, plot_residuals_param_evolve=False, 
    #                          save_fig_l_evolve=False, plot_residuals_l_evolve=False, save_fig_param_evolve=False):
    #     """
    #     Calculate residuals for a property given polarisation data.

    #     Parameters:
    #     ----------
    #     ecc_list : list of floats
    #         List of minimum eccentricities.
    #     hp_dataset : ndarray
    #         Plus polarisation data.
    #     hc_dataset : ndarray
    #         Cross polarisation data.
    #     property : str
    #         Specifies which property to calculate ('phase' or 'amplitude').

    #     Returns:
    #     -------
    #     residual_dataset : ndarray
    #         Array of residuals for each eccentricity.
            
    #     """

    #     start = time.time()
    #     truncate_at_ISCO = self.resolve_property(prop=truncate_at_ISCO, default=self.truncate_at_ISCO)
    #     truncate_at_tmin = self.resolve_property(prop=truncate_at_tmin, default=self.truncate_at_tmin)
        
    #     def calculate_residual_wrapper(hp, hc, ecc, q, chi1, chi2):
    #         if not np.any(ecc):
    #             if train_obj.property == "phase":
    #                 return self.phase_circ
    #             elif train_obj.property == "amplitude":
    #                 return self.amp_circ
    #         else:
    #             self.ecc_ref, self.mass_ratio, self.chi1, self.chi2 = ecc, q, chi1, chi2
    #             residual_dict = self.calculate_residual(hp=hp, hc=hc, property=train_obj.property, mean_ano_ref=None)

    #             train_obj.mean_anomaly = residual_dict['L_grid']
    #             train_obj.time = residual_dict['t_out']
    #             train_obj.l_to_t_mapping = residual_dict['mapping_dict']

    #             return residual_dict['residual_in_L'], train_obj.time

    #     try:
    #         """ Try loading residuals from previous pre-computation """
    #         if plot_polarizations:
    #             train_obj = train_obj.load_polarizations()
    #             train_obj = train_obj.load_residuals()
    #         else:
    #             train_obj = train_obj.load_residuals()
    #         self.time = train_obj.time

    #     except Exception as e:
    #         """No pre-computed residuals found. Attempting to load polarisations to prevent re-computation of waveforms."""
    #         print(e)
    #         traceback.print_exc()
    #         n_params = len(train_obj.parameter_grid)
            
    #         try:
    #             """ Try loading polarisations to prevent re-computation """
    #             train_obj = train_obj.load_polarizations()
    #             self.time = train_obj.time
    #             hp_flat = train_obj.hp_dataset.reshape(len(train_obj.parameter_grid), len(train_obj.time))
    #             hc_flat = train_obj.hc_dataset.reshape(len(train_obj.parameter_grid), len(train_obj.time))
    #             residuals_flat = np.empty_like(hp_flat)

    #             for idx, (ecc, q, chi1, chi2) in enumerate(train_obj.parameter_grid):
    #                 residual = calculate_residual_wrapper(hp=hp_flat[idx], hc=hc_flat[idx], ecc=ecc, q=q, chi1=chi1, chi2=chi2)
    #                 residuals_flat[idx] = residual
    #                 # Delete large intermediates immediately
    #                 del residual
    #                 gc.collect()
    #                 if idx % 50 == 0 and MEMORY_PROFILE:
    #                     check_memory_usage(f"Loading polarizations progress: {idx}/{n_params}")
                
    #             train_obj.residuals = residuals_flat
                
    #         except Exception as e2:
    #             """No pre-computed polarisations and residuals found. Compute all from scratch."""
                
    #             print(e2)
    #             traceback.print_exc()
                
    #             # Start with longest time_array. Will be shortened iteratively to the shortest time_array.
    #             current_time = self.time.copy()
                
    #             # Polarisation datasets
    #             hp_flat = np.empty((n_params, len(current_time)))
    #             hc_flat = np.empty_like(hp_flat)
    #             residuals_flat = np.empty_like(hp_flat)

    #             # Simulate all polarizations and calculate residuals for every parameter combination in the parameter grid.
    #             # Main processing loop with periodic memory checkpoints
    #             for idx, (ecc, q, chi1, chi2) in enumerate(train_obj.parameter_grid):
    #                 hp, hc, time_array = self.simulate_waveform(time_array=current_time, ecc_ref=ecc, mass_ratio=q, chi1=chi1, chi2=chi2, truncate_at_ISCO=truncate_at_ISCO, truncate_at_tmin=truncate_at_tmin, update_results=True, show_truncation_warnings=False)

    #                 # Calculate circular waveform for the same parameters, but with ecc=0, to use as reference for the residual calculation
    #                 residual, t_out = calculate_residual_wrapper(hp=hp, hc=hc, ecc=ecc, q=q, chi1=chi1, chi2=chi2)
                    
    #                 # Update mask: keep only the part of base_time covered by this waveform
    #                 current_mask = (current_time >= t_out[0]) & (current_time <= t_out[-1])
    #                 current_time = t_out
    #                 start_idx = np.where(current_mask)[0][0]
    #                 end_idx = np.where(current_mask)[0][-1] + 1
    #                 active_n_t = len(current_time)

    #                 if active_n_t != hp_flat.shape[1]:
    #                     hp_flat = hp_flat[:, start_idx:end_idx]
    #                     hc_flat = hc_flat[:, start_idx:end_idx]
    #                     residuals_flat = residuals_flat[:, start_idx:end_idx]
                    
    #                 hp_flat[idx] = hp
    #                 hc_flat[idx] = hc
    #                 residuals_flat[idx] = residual
                    
    #                 # Immediate cleanup of simulated waveforms
    #                 del hp, hc, time_array, residual, t_out
    #                 gc.collect()
                    
    #                 # [OPTIMIZED #2]: Periodic memory checkpoint
    #                 if idx % 20 == 0 and MEMORY_PROFILE:
    #                     check_memory_usage(f"Generating residuals: {idx}/{n_params}")

    #             train_obj.hp_dataset = hp_flat
    #             train_obj.hc_dataset = hc_flat
    #             train_obj.residuals = residuals_flat
    #             train_obj.time = current_time
    #             self.time = current_time

    #             print(self.colored_text(f"All residuals generated in {(time.time() - start)/60:.2f} minutes.", 'green'))

    #             if save_polarizations:
    #                 train_obj.save_polarizations(prefix="polarisation", directory="Straindata/Polarisations", free_memory=False)
            
    #         if save_residuals:
    #             train_obj.save_residuals(prefix="residuals", directory="Straindata/Residuals", free_memory=False)

    #     if plot_polarizations:
    #         self._plot_polarizations(train_obj, save_fig_polarizations)

    #     if (plot_residuals_param_evolve is True) or (plot_residuals_l_evolve is True):
    #         self._plot_residuals(train_obj, plot_residuals_param_evolve, save_fig_param_evolve, plot_residuals_l_evolve, save_fig_l_evolve)

    #     if MEMORY_PROFILE:
    #         check_memory_usage("END _calculate_residuals")
            
    #     print(self.colored_text(f"Dataset shape: {train_obj.residuals.shape}, N={train_obj.parameter_grid.size}", 'green'))
    #     return train_obj.residuals
    
    def _calculate_residuals(self, train_obj:TrainingSetResults, truncate_at_ISCO=None, truncate_at_tmin=None, 
                            save_residuals=True, save_polarizations=True, plot_polarizations=False, 
                            save_fig_polarizations=False, plot_residuals_param_evolve=False, 
                            save_fig_l_evolve=False, plot_residuals_l_evolve=False, save_fig_param_evolve=False):
        """
        Calculate residuals for a property given polarisation data.
        """

        start = time.time()
        truncate_at_ISCO = self.resolve_property(prop=truncate_at_ISCO, default=self.truncate_at_ISCO)
        truncate_at_tmin = self.resolve_property(prop=truncate_at_tmin, default=self.truncate_at_tmin)
        
        def calculate_residual_wrapper(hp, hc, ecc, q, chi1, chi2):
            if not np.any(ecc):
                if train_obj.property == "phase":
                    return self.phase_circ, self.time, np.ones(len(self.time), dtype=bool)  # Full mask for circular
                elif train_obj.property == "amplitude":
                    return self.amp_circ, self.time, np.ones(len(self.time), dtype=bool)
            else:
                self.ecc_ref, self.mass_ratio, self.chi1, self.chi2 = ecc, q, chi1, chi2
                residual_dict = self.calculate_residual(hp=hp, hc=hc, property=train_obj.property, mean_ano_ref=None)

                train_obj.mean_anomaly = residual_dict['L_grid']
                train_obj.time = residual_dict['t_out']
                train_obj.l_to_t_mapping = residual_dict['mapping_dict']

                # FIX: Create a LOCAL mask based on residual length, not original time array
                t_mask = np.ones(len(residual_dict['residual_in_L']), dtype=bool)

                return residual_dict['residual_in_L'], train_obj.time, t_mask

        try:
            """ Try loading residuals from previous pre-computation """
            if plot_polarizations:
                train_obj = train_obj.load_polarizations()
                train_obj = train_obj.load_residuals()
            else:
                train_obj = train_obj.load_residuals()
            self.time = train_obj.time

        except Exception as e:
            """No pre-computed residuals found. Attempting to load polarisations to prevent re-computation of waveforms."""
            print(e)
            traceback.print_exc()
            n_params = len(train_obj.parameter_grid)
            
            try:
                """ Try loading polarisations to prevent re-computation """
                train_obj = train_obj.load_polarizations()
                self.time = train_obj.time
                hp_flat = train_obj.hp_dataset.reshape(len(train_obj.parameter_grid), len(train_obj.time))
                hc_flat = train_obj.hc_dataset.reshape(len(train_obj.parameter_grid), len(train_obj.time))

                # Collect residuals as list — each entry may have different length
                residual_list = []

                for idx, (ecc, q, chi1, chi2) in enumerate(train_obj.parameter_grid):
                    residual, t_out, t_mask = calculate_residual_wrapper(
                        ecc=ecc, q=q, chi1=chi1, chi2=chi2
                    )

                    residual_list.append(residual)

                    del residual, t_out, t_mask
                    gc.collect()
                    if idx % 50 == 0 and MEMORY_PROFILE:
                        check_memory_usage(f"Loading polarizations progress: {idx}/{n_params}")

                # Find the shortest residual length
                min_len = min(len(r) for r in residual_list)
                print(f"Shortest residual length: {min_len} (from {len(residual_list)} waveforms)")

                # Truncate all residuals to the shortest length, then stack
                residuals_flat = np.array([r[:min_len] for r in residual_list])

                # Also truncate hp_flat, hc_flat, and time to match
                hp_flat = hp_flat[:, :min_len]
                hc_flat = hc_flat[:, :min_len]
                train_obj.time = train_obj.time[:min_len]
                self.time = train_obj.time

                # Update the polarisation dataset with trimmed versions
                train_obj.hp_dataset = hp_flat
                train_obj.hc_dataset = hc_flat

                train_obj.residuals = residuals_flat
                
            except Exception as e2:
                """No pre-computed polarisations and residuals found. Compute all from scratch."""
                
                print(e2)
                traceback.print_exc()
                
                # Start with longest time_array. Will be shortened iteratively to the shortest time_array.
                current_time = self.time.copy()
                
                # Polarisation datasets
                hp_flat = np.empty((n_params, len(current_time)))
                hc_flat = np.empty_like(hp_flat)
                residuals_flat = np.empty_like(hp_flat)

                # Simulate all polarizations and calculate residuals for every parameter combination in the parameter grid.
                # Main processing loop with periodic memory checkpoints
                for idx, (ecc, q, chi1, chi2) in enumerate(train_obj.parameter_grid):
                    hp_ecc, hc_ecc, hp_circ, hc_circ, time_array = self.simulate_waveform_l(
                        time_array=current_time, 
                        ecc_ref=ecc, 
                        mass_ratio=q, 
                        chi1=chi1, 
                        chi2=chi2, 
                        truncate_at_ISCO=truncate_at_ISCO, 
                        truncate_at_tmin=truncate_at_tmin, 
                        update_results=True, 
                        show_truncation_warnings=True
                    )

                    # Calculate circular waveform for the same parameters, but with ecc=0, to use as reference for the residual calculation
                    residual, t_out, t_mask = calculate_residual_wrapper(
                        hp=hp, hc=hc, ecc=ecc, q=q, chi1=chi1, chi2=chi2
                    )
                    
                    # ============================================
                    # Find overlap between current_time and t_out
                    # ============================================
                    # t_out could be shorter at both front AND back
                    overlap_mask = (current_time >= t_out[0]) & (current_time <= t_out[-1])
                    overlap_indices = np.where(overlap_mask)[0]
                    
                    if len(overlap_indices) == 0:
                        raise ValueError(f"No time overlap between current_time and t_out for idx={idx}")
                    
                    start_idx = overlap_indices[0]
                    end_idx = overlap_indices[-1] + 1  # +1 for slicing
                    active_n_t = len(overlap_indices)  # Should equal len(t_out) == len(hp_cut)
                    
                    # Shrink current_time to the overlapping region
                    current_time = current_time[start_idx:end_idx]
                    
                    # Shrink all flat arrays to match the new current_time length
                    if active_n_t != hp_flat.shape[1]:
                        hp_flat = hp_flat[:, start_idx:end_idx]
                        hc_flat = hc_flat[:, start_idx:end_idx]
                        residuals_flat = residuals_flat[:, start_idx:end_idx]
                    
                    # FIX: t_mask is now sized correctly for the CURRENT hp length
                    # Since t_mask is all True and matches residual length, just use residual length directly
                    hp_flat[idx] = self.hp_ecc 
                    hc_flat[idx] = self.hc_ecc
                    residuals_flat[idx] = residual
                    
                    # Immediate cleanup of simulated waveforms
                    del hp, hc, time_array, residual, t_out, overlap_mask, overlap_indices
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

        if (plot_residuals_param_evolve is True) or (plot_residuals_l_evolve is True):
            self._plot_residuals(train_obj, plot_residuals_param_evolve, save_fig_param_evolve, plot_residuals_l_evolve, save_fig_l_evolve)

        if MEMORY_PROFILE:
            check_memory_usage("END _calculate_residuals")
            
        print(self.colored_text(f"Dataset shape: {train_obj.residuals.shape}, N={train_obj.parameter_grid.size}", 'green'))
        return train_obj.residuals

    def get_greedy_parameters(self, 
                            train_obj:TrainingSetResults, 
                            min_greedy_error=None, N_greedy_vecs=None, 
                            normalize=True, 
                            max_tree_depth=0, 
                            plot_greedy_error=False, save_greedy_error_fig=False, 
                            plot_greedy_vectors=False, save_greedy_vecs_fig=False, 
                            plot_SVD_matrix=False, save_SVD_matrix_fig=False, 
                            plot_basis_indices=False, save_basis_indices_fig=False,
                            show_legend=False,
                            save_greedy_errors=True,
                            save_orthonormal_basis=True,
                            free_memory=True
                            ):
        """
        Greedy algorithm to select representative vectors from U using an orthonormal basis.

        Parameters
        ----------
        Train_obj : TrainingSetResults
            Object containing the training set results, including the residuals dataset and parameter grid.
        min_greedy_error : float, optional
            Stop the greedy algorithm when the maximum residual norm falls below this value.
        N_greedy_vecs : int, optional
            Maximum number of vectors to include in the greedy basis.
        normalize : bool, optional
            If True, normalize the residuals before applying the greedy algorithm.
        max_tree_depth : int, optional
            Maximum depth of the tree in the greedy algorithm. A value of 0 means no tree (standard greedy), 
            while higher values allow for more complex tree structures that can capture more intricate relationships in the data.
        plot_greedy_error : bool, optional
            If True, plots the greedy error convergence as a function of the number of basis vectors.
        save_greedy_error_fig : bool, optional
            If True, saves the greedy error convergence plot to a file.
        plot_greedy_vectors : bool, optional
            If True, plots the selected greedy vectors in the time domain.
        save_greedy_vecs_fig : bool, optional
            If True, saves the greedy vectors plot to a file.
        plot_SVD_matrix : bool, optional
            If True, plots the singular values of the residuals dataset to analyze its intrinsic dimensionality.
        save_SVD_matrix_fig : bool, optional
            If True, saves the SVD matrix plot to a file.
        plot_basis_indices : bool, optional
            If True, plots the distribution of the selected greedy basis indices in the parameter space.
        save_basis_indices_fig : bool, optional
            If True, saves the basis indices plot to a file.

        Returns
        -------
        reduced_basis_object : ReducedBasis
            ReducedBasis object containing the greedy basis and indices of the selected vectors.
        """
        if train_obj.residuals is None:
            train_obj.load_residuals()
            if train_obj.residuals is None:
                raise ValueError("Residuals dataset is not available in the training object. " \
                "Please run _calculate_residuals() before running the greedy algorithm.")
        
        nan_mask = np.isnan(train_obj.residuals)

        if np.any(nan_mask):
            bad_rows = np.where(np.any(nan_mask, axis=1))[0]

            print(self.colored_text(
                f"Warning: NaN values found in residuals dataset.\n"
                f"Bad row indices: {bad_rows}\n"
                f"These will be ignored in the greedy algorithm, but may affect the results.",
                'yellow'
            ))

        # Resolve properties with defaults
        greedy_tol = self.resolve_property(prop=min_greedy_error, default=-np.inf) 
        nmax = self.resolve_property(prop=N_greedy_vecs, default=train_obj.residuals.shape[0])

        # Print warnings for dataset issues that could affect the greedy algorithm performance
        self._greedy_parameters_warning(dataset=train_obj.residuals, 
                                        parameter_grid=self.parameter_grid,
                                        time_array=train_obj.time)
        
        # Get reduced basis object
        reduced_basis_object = ReducedBasis(greedy_tol=greedy_tol, normalize=normalize, nmax=nmax, lmax=max_tree_depth)
        # Calculate the greedy indices
        reduced_basis_object.fit(training_set = train_obj.residuals,
            parameters = self.parameter_grid,
            physical_points = self.time
            )
        # print(train_obj.basis_indices, type(train_obj.basis_indices))
        train_obj.orthonormal_basis = []
        train_obj.greedy_errors = []
        train_obj.empirical_indices = []
        train_obj.leaf_nodes_indices = []

        for leaf in reduced_basis_object.tree.leaves:
            # flat list of all indices
            train_obj.basis_indices.extend(leaf.indices)

            # nested list: one list per leaf
            train_obj.leaf_basis_indices.append(leaf.indices)

            # basis vectors
            train_obj.orthonormal_basis.extend(leaf.basis)

            # errors
            train_obj.greedy_errors.extend(np.asarray(leaf.errors).ravel())
        
        print(self.colored_text(f'Basis calculated using {len(reduced_basis_object.tree.leaves)} discretized space(s).', 'green'))
        
        # Save large data to file to avoid memory issues
        if save_greedy_errors:
            train_obj.save_greedy_errors()
        
        if save_orthonormal_basis:
            train_obj.save_orthonormal_basis()

        ### PLOTTING ###
        if plot_greedy_error:
            self._plot_greedy_errors(
                                        train_obj=train_obj, 
                                        save_greedy_fig=save_greedy_error_fig)
        
        if plot_basis_indices:
            self._plot_basis_indices(
                                    train_obj=train_obj, 
                                    save_basis_indices_fig=save_basis_indices_fig)

        if plot_greedy_vectors:
            self._plot_greedy_vectors( 
                                        train_obj=train_obj, 
                                        save_greedy_vecs_fig=save_greedy_vecs_fig, 
                                        U=None, 
                                        show_legend=show_legend)
        
        if plot_SVD_matrix:
            self._plot_SVD_matrix(train_obj=train_obj, 
                                    save_SVD_matrix_fig=save_SVD_matrix_fig)

        if free_memory:
            train_obj.greedy_errors = None
            train_obj.orthonormal_basis = None

        return reduced_basis_object
    
    def get_empirical_nodes(self, 
                            reduced_basis_object: ReducedBasis, 
                            train_obj: TrainingSetResults, 
                            plot_first_N_vectors=None,
                            plot_emp_nodes_on_basis=False, save_emp_nodes_on_basis_fig=False, 
                            plot_interpolation_matrix=False, save_interpolation_matrix_fig=False, 
                            plot_proj_vs_eim_error=False, save_proj_vs_eim_error_fig=False):
        """
        Calculate the empirical nodes for a given dataset based on a reduced basis of residual properties.

        Parameters:
        ----------------
        - reduced_basis_object (ReducedBasis): The reduced basis object containing the greedy basis and tree structure.
        - train_obj (TrainingSetResults): The training object containing the dataset and parameter grid.
        - plot_first_N_vectors (int, optional): If specified, only plot the first N reduced basis vectors when visualizing empirical nodes on the basis functions.
            If None, plots all basis vectors.
        - plot_emp_nodes_on_basis (bool, optional): If True, plots the empirical nodes on top of the reduced basis functions for each leaf in the tree.
        - save_emp_nodes_on_basis_fig (bool, optional): If True, saves the figure of empirical nodes on basis functions to a file.
        - plot_interpolation_matrix (bool, optional): If True, plots the interpolation matrix to visualize how the empirical nodes are selected across the time domain.
        - save_interpolation_matrix_fig (bool, optional): If True, saves the interpolation matrix figure to a file.
        - plot_proj_vs_eim_error (bool, optional): If True, plots the projection error of the reduced basis approximation versus the empirical interpolation error to analyze the relationship between the greedy basis construction and the empirical interpolation accuracy.
        - save_proj_vs_eim_error_fig (bool, optional): If True, saves the projection vs EIM error figure to a file.

        Returns:
        ----------------
        - emp_nodes_idx (list): Indices of empirical nodes for the given dataset.
        """

        # if eim_per_leaf:

        # Get empirical nodes for each leaf (greedy parameters section) in the tree
        eim = EmpiricalInterpolation(reduced_basis_object)
        eim.fit()

        # Stack empirical nodes from all leaves into the training object
        for leaf in reduced_basis_object.tree.leaves:
            train_obj.empirical_indices.extend(leaf.empirical_nodes)
            train_obj.leaf_nodes_indices.append(leaf.empirical_nodes)

        # Optional: plot empirical nodes at eccentricity
        if plot_interpolation_matrix:
            self._plot_interpolation_matrix(train_obj=train_obj, 
                                            save_fig=save_interpolation_matrix_fig)

        if plot_proj_vs_eim_error:
            self._plot_projection_vs_eim_error(train_obj=train_obj, 
                                               save_fig=save_proj_vs_eim_error_fig)

        if plot_emp_nodes_on_basis:
            self._plot_emp_nodes_on_basis(train_obj=train_obj, 
                                          first_N_vectors=plot_first_N_vectors, 
                                          save_fig=save_emp_nodes_on_basis_fig)

        return train_obj.empirical_indices


   


    def get_training_set_greedy(self, 
                            property, 
                            min_greedy_error=None, N_greedy_vecs=None, 
                            max_tree_depth=0,
                            plot_training_set=False, save_fig_training_set=False,
                            plot_greedy_error=False, save_fig_greedy_error=False,
                            plot_emp_nodes_on_basis=False, save_fig_emp_nodes_on_basis=False,
                            plot_residuals_eccentric=False, save_fig_residuals_eccentric=False, 
                            plot_residuals_time=False, save_fig_residuals_time=False,  
                            plot_interpolation_matrix=False, save_fig_interpolation_matrix=False,
                            plot_proj_vs_eim_error=False, save_fig_proj_vs_eim_error=False,
                            plot_basis_indices=False, save_fig_basis_indices=False,
                            save_residuals=True, 
                            save_polarizations=True,
                            save_greedy_errors=True,
                            save_orthonormal_basis=True,
                            save_train_obj=True,
                            free_memory=True,
                            show_legend_ts=True
                            ):
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
        - train_obj (TrainingSet): An object containing the generated training set and related information.
        """
        if MEMORY_PROFILE:
            check_memory_usage("START get_training_set_greedy")
        
        # Import from class object if min_greedy and N_greedy vecs are not specified
        if (min_greedy_error is None) and (N_greedy_vecs is None):
            if property == 'phase':
                min_greedy_error = self.min_greedy_error_phase
                N_greedy_vecs = self.N_basis_vecs_phase
            elif property == 'amplitude':
                min_greedy_error = self.min_greedy_error_amp
                N_greedy_vecs = self.N_basis_vecs_amp

        # Either create or get already existing training object for the specified property (phase or amplitude)
        train_obj = self._get_training_obj(property)

        try:
            train_obj = train_obj.load()
            self.time = train_obj.time

            # Plotting after loading
            if plot_residuals_eccentric or plot_residuals_time:
                self._plot_residuals(train_obj=train_obj,
                                        plot_eccentric_evolve=plot_residuals_eccentric, save_fig_eccentric_evolve=save_fig_residuals_eccentric,
                                        plot_time_evolve=plot_residuals_time, save_fig_time_evolve=save_fig_residuals_time
                                        )
            if plot_emp_nodes_on_basis:
                self._plot_emp_nodes_on_basis(train_obj=train_obj, 
                                            save_fig=save_fig_emp_nodes_on_basis
                                            )
            if plot_greedy_error:
                self._plot_greedy_errors(train_obj=train_obj,
                                        save_greedy_fig=save_fig_greedy_error
                                        )

            if plot_interpolation_matrix:
                self._plot_interpolation_matrix(train_obj=train_obj,
                                            save_fig=save_fig_interpolation_matrix
                                            )
            if plot_proj_vs_eim_error:
                self._plot_projection_vs_eim_error(train_obj=train_obj, 
                                                save_fig=save_fig_proj_vs_eim_error
                                                )
            if plot_basis_indices:
                self._plot_basis_indices(train_obj=train_obj, 
                                        save_basis_indices_fig=save_fig_basis_indices
                                        )

        except Exception as e:
            print(e)
            traceback.print_exc()
            # Step 1: Generate residuals for the full parameter space
            train_obj = self.generate_property_dataset(
                train_obj=train_obj,
                save_polarizations=save_polarizations,
                save_residuals=save_residuals,
                plot_residuals_eccentric_evolve=plot_residuals_eccentric,
                plot_residuals_time_evolve=plot_residuals_time,
                save_fig_eccentric_evolve=save_fig_residuals_eccentric,
                save_fig_time_evolve=save_fig_residuals_time
            )

            train_obj.load_residuals()

            # Step 2: Select the best representative parameters using a greedy algorithm
            # print('Calculating greedy parameters...')
            reduced_basis_object = self.get_greedy_parameters(
                train_obj=train_obj,
                N_greedy_vecs=N_greedy_vecs,
                min_greedy_error=min_greedy_error,
                max_tree_depth=max_tree_depth,
                save_greedy_errors=save_greedy_errors,
                save_orthonormal_basis=save_orthonormal_basis,
                plot_greedy_error=plot_greedy_error,
                save_greedy_error_fig=save_fig_greedy_error,
                plot_basis_indices=plot_basis_indices,
                save_basis_indices_fig=save_fig_basis_indices,
                free_memory=False
            )
            # print(f'Greedy parameters {property}: {train_obj.basis_indices}, length: {len(train_obj.basis_indices)}  ')

            # Step 3: Calculate empirical nodes of the greedy basis
            train_obj.empirical_indices = self.get_empirical_nodes(
                reduced_basis_object=reduced_basis_object,
                train_obj=train_obj,
                plot_emp_nodes_on_basis=plot_emp_nodes_on_basis,
                save_emp_nodes_on_basis_fig=save_fig_emp_nodes_on_basis,
                plot_interpolation_matrix=plot_interpolation_matrix,
                save_interpolation_matrix_fig=save_fig_interpolation_matrix,
                plot_proj_vs_eim_error=plot_proj_vs_eim_error,
                save_proj_vs_eim_error_fig=save_fig_proj_vs_eim_error
            )

            residual_basis = train_obj.residuals[train_obj.basis_indices] # shape (n_greedy_vecs, n_time)
            train_obj.training_set = residual_basis[:, train_obj.empirical_indices]
            self.time_training = self.time[train_obj.empirical_indices]

            if save_train_obj:
                train_obj.save()

            del residual_basis

        # Optionally plot the training set
        if plot_training_set:
            self._plot_emp_nodes_on_residuals(train_obj, save_fig_training_set, show_legend=show_legend_ts)

        # Clean memory of objects that are no longer needed
        if free_memory:
            train_obj.residuals = None
            train_obj.orthonormal_basis = None
            train_obj.greedy_errors = None

        if MEMORY_PROFILE:
            check_memory_usage("END get_training_set_greedy")
        gc.collect()

        return train_obj
    
    ################################################
    # PLOTTING FUNCTIONS
    ################################################
    
    def _plot_polarizations(self,
                                    train_obj: TrainingSetResults,
                                    save_fig_polarizations=False):
        """
        Plot plus and cross polarizations using the same structure as residual plots.
        """

        if train_obj.hp_dataset is None or train_obj.hc_dataset is None:
            train_obj = train_obj.load_polarizations()

        hp = np.asarray(train_obj.hp_dataset)
        hc = np.asarray(train_obj.hc_dataset)
        time = train_obj.time

        ecc_space = train_obj.ecc_ref_space
        l_space = train_obj.mean_ano_ref_space
        q_space = train_obj.mass_ratio_space
        chi1_space = train_obj.chi1_space
        chi2_space = train_obj.chi2_space

        n_e = len(ecc_space)
        n_l = len(l_space)
        n_q = len(q_space)
        n_c1 = len(chi1_space)
        n_c2 = len(chi2_space)

        # ------------------------------------------------------------
        # baseline indices (same idea as residuals)
        # ------------------------------------------------------------
        e0 = n_e // 2
        l0 = n_l // 2
        q0 = n_q // 2
        c10 = n_c1 // 2
        c20 = n_c2 // 2

        # ------------------------------------------------------------
        # helper: pick representative parameter values
        # ------------------------------------------------------------
        def five_indices(n):
            if n <= 5:
                return np.arange(n, dtype=int)
            return np.linspace(0, n - 1, 5, dtype=int)

        # ------------------------------------------------------------
        # helper: fixed parameter text
        # ------------------------------------------------------------
        def fixed_text(vary_key):
            parts = []

            if vary_key != "ecc":
                parts.append(f"e={float(ecc_space[e0]):.4g}")
            if vary_key != "mean_ano":
                parts.append(f"l={float(l_space[l0]):.4g}")
            if vary_key != "q":
                parts.append(f"q={float(q_space[q0]):.4g}")
            if vary_key != "chi1":
                parts.append(rf"$\chi_1$={float(chi1_space[c10]):.4g}")
            if vary_key != "chi2":
                parts.append(rf"$\chi_2$={float(chi2_space[c20]):.4g}")

            return "Fixed: " + ", ".join(parts)

        # ------------------------------------------------------------
        # shared parameter slicing definition
        # ------------------------------------------------------------

        parameter_effects = [
            {
                "key": "ecc",
                "name": "eccentricity",
                "symbol": "e",
                "values": ecc_space,
                "column": 0,
            },
            {
                "key": "mean_ano",
                "name": "mean anomaly",
                "symbol": "l",
                "values": l_space,
                "column": 1,
            },
            {
                "key": "q",
                "name": "mass ratio",
                "symbol": "q",
                "values": q_space,
                "column": 2,
            },
            {
                "key": "chi1",
                "name": r"$\chi_1$",
                "symbol": r"$\chi_1$",
                "values": chi1_space,
                "column": 3,
            },
            {
                "key": "chi2",
                "name": r"$\chi_2$",
                "symbol": r"$\chi_2$",
                "values": chi2_space,
                "column": 4,
            },
        ]

        # ------------------------------------------------------------
        # plotting function (single structure for hp/hc)
        # ------------------------------------------------------------
        def plot_one(dataset_key, ylabel, prefix):
           
            fig, axes = plt.subplots(
                len(parameter_effects),
                1,
                figsize=(11, 3.5 * len(parameter_effects)),
                sharex=True,
                gridspec_kw={"hspace": 0.5}
            )

            if len(parameter_effects) == 1:
                axes = [axes]

            dataset = hp if dataset_key == "hp" else hc
            grid = np.asarray(train_obj.parameter_grid)

            for ax, effect in zip(axes, parameter_effects):

                values = effect["values"]

                selected_indices = five_indices(len(values))

                for i in selected_indices:

                    target_value = values[i]

                    # baseline parameter set
                    target = [
                        ecc_space[e0],
                        l_space[l0],
                        q_space[q0],
                        chi1_space[c10],
                        chi2_space[c20],
                    ]

                    # vary ONE parameter
                    target[effect["column"]] = target_value

                    target = np.asarray(target)

                    # find matching waveform
                    matches = np.all(np.isclose(grid, target), axis=1)

                    if not np.any(matches):
                        continue

                    idx = np.where(matches)[0][0]

                    ax.plot(
                        time,
                        dataset[idx],
                        linewidth=0.9,
                        label=f"{effect['symbol']} = {float(target_value):.4g}"
                    )

                ax.set_ylabel(ylabel)
                ax.set_title(
                    f"{dataset_key}: varying {effect['name']}\n"
                    f"{fixed_text(effect['key'])}",
                    fontsize=10
                )
                ax.grid(True)

                ax.legend(
                    title=f"Varying {effect['symbol']}",
                    fontsize="small",
                    loc="best"
                )

            axes[-1].set_xlabel("t [M]")

            fig.suptitle(
                f"{dataset_key} polarization: effect of each parameter",
                y=1.005
            )

            plt.tight_layout()

            if save_fig_polarizations:
                figname = train_obj.figname(
                    prefix=prefix,
                    directory="Images/Polarisations"
                )
                fig.savefig(figname)
            
            plt.close(fig)
            gc.collect()

        # ------------------------------------------------------------
        # run for hp and hc
        # ------------------------------------------------------------
        plot_one("hp", r"$h_+$", "Polarizations_hp")
        plot_one("hc", r"$h_\times$", "Polarizations_hc")


    def _plot_residuals(
            self,
            train_obj,
            plot_eccentric_evolve=False,
            save_fig_eccentric_evolve=False,
            plot_time_evolve=False,
            save_fig_time_evolve=False,
            plot_dims=("ecc", "mean_ano", "q", "chi1"),   # NEW
        ):

        train_obj.load_residuals()

        residuals = train_obj.residuals   # (N, T)
        mean_anomaly = train_obj.mean_anomaly
        params = train_obj.parameter_grid

        n_l = len(mean_anomaly)

        if train_obj.property == "phase":
            ylabel = r"$\Delta \phi_{22}$ [radians]"
        elif train_obj.property == "amplitude":
            ylabel = r"$\Delta A_{22}$"
        else:
            raise ValueError("property must be phase or amplitude")

        COL = {
            "ecc": 0,
            "q": 2,
            "chi1": 3,
            "chi2": 4,
        }

        names = {
            "ecc": "eccentricity",
            "q": "mass ratio",
            "chi1": r"$\chi_1$",
            "chi2": r"$\chi_2$",
        }

        symbols = {
            "ecc": "e",
            "q": "q",
            "chi1": r"$\chi_1$",
            "chi2": r"$\chi_2$",
        }

        # ------------------------------------------------------------
        # NEW: filter dimensions
        # ------------------------------------------------------------
        if plot_dims is None:
            plot_dims = list(COL.keys())
        else:
            plot_dims = list(plot_dims)

        COL = {k: v for k, v in COL.items() if k in plot_dims}

        def five_indices(n):
            if n <= 1:
                return np.array([0], dtype=int)
            if n <= 5:
                return np.arange(n, dtype=int)
            return np.linspace(0, n - 1, 5, dtype=int)

        base_idx = len(residuals) // 2

        def fixed_text(vary_key):
            parts = []

            for k, c in COL.items():
                if k != vary_key:
                    if k == "ecc":
                        parts.append(f"e={params[base_idx, c]:.4g}")
                    elif k == "q":
                        parts.append(f"q={params[base_idx, c]:.4g}")
                    elif k == "chi1":
                        parts.append(rf"$\chi_1$={params[base_idx, c]:.4g}")
                    elif k == "chi2":
                        parts.append(rf"$\chi_2$={params[base_idx, c]:.4g}")

            return "Fixed: " + ", ".join(parts)

        # ------------------------------------------------------------
        # BUILD EFFECTS
        # ------------------------------------------------------------
        parameter_effects = []

        for key, col in COL.items():
            mask = np.ones(len(residuals), dtype=bool)

            for k2, c2 in COL.items():
                if k2 != key:
                    mask &= np.isclose(params[:, c2], params[base_idx, c2])

            idx = np.where(mask)[0]

            parameter_effects.append({
                "key": key,
                "col": col,
                "idx": idx
            })

        # ============================================================
        # TIME EVOLUTION PLOT
        # ============================================================
        if plot_time_evolve:

            fig, axes = plt.subplots(
                len(parameter_effects),
                1,
                figsize=(11, 3.5 * len(parameter_effects)),
                sharex=True,
                gridspec_kw={"hspace": 0.5}
            )

            if len(parameter_effects) == 1:
                axes = [axes]

            for ax, effect in zip(axes, parameter_effects):

                idx = effect["idx"]
                col = effect["col"]

                selected_indices = np.linspace(
                    0,
                    len(idx) - 1,
                    min(5, len(idx)),
                    dtype=int
                )

                for i in selected_indices:
                    global_idx = idx[i]

                    ax.plot(
                        mean_anomaly,
                        residuals[global_idx],
                        linewidth=0.9,
                        linestyle="-",
                        label=f"{symbols[effect['key']]} = {params[global_idx, col]:.4g}"
                    )

                ax.set_ylabel(ylabel)
                ax.set_title(
                    f"Varying {names[effect['key']]} over mean anomaly\n"
                    f"{fixed_text(effect['key'])}",
                    fontsize=10
                )
                ax.grid(True)
                ax.legend()

            axes[-1].set_xlabel("l [rad]")

            if save_fig_time_evolve:
                fig.savefig(train_obj.figname(
                    prefix="Residuals_l_evolve",
                    directory="Images/Residuals"
                ))

            plt.close(fig)
            gc.collect()

        # ============================================================
        # PARAMETER EVOLUTION
        # ============================================================
        if plot_eccentric_evolve:

            fig, axes = plt.subplots(
                len(parameter_effects),
                1,
                figsize=(11, 3.5 * len(parameter_effects)),
                gridspec_kw={"hspace": 0.6}
            )

            l_idx = five_indices(n_l)

            if len(parameter_effects) == 1:
                axes = [axes]

            for ax, effect in zip(axes, parameter_effects):

                idx = effect["idx"]
                col = effect["col"]

                for l in l_idx:
                    ax.plot(
                        params[idx, col],
                        residuals[idx, l],
                        linewidth=0.9,
                        linestyle="-",
                        label=f"l = {mean_anomaly[l]:.4g}"
                    )

                ax.set_title(
                    f"Residual change while varying {names[effect['key']]}\n"
                    f"{fixed_text(effect['key'])}"
                )
                ax.set_xlabel(symbols[effect['key']])
                ax.set_ylabel(ylabel)
                ax.grid(True)
                ax.legend()

            if save_fig_eccentric_evolve:
                fig.savefig(train_obj.figname(
                    prefix="Residuals_param_evolve",
                    directory="Images/Residuals"
                ))

            plt.close(fig)
            gc.collect()

    def _plot_basis_indices(self, 
                           train_obj:TrainingSetResults, 
                           save_basis_indices_fig=False
                           ):
        def compute_chi_eff(params):
            q = params[:, 2]
            chi1 = params[:, 3]
            chi2 = params[:, 4]
            return (q * chi1 + chi2) / (1 + q)

        params = train_obj.parameter_grid
        basis_indices = train_obj.basis_indices
        greedy = params[basis_indices]

        # Extract parameters
        e_all, _, q_all, chi1_all, chi2_all = params.T
        e_g, _, q_g, chi1_g, chi2_g = greedy.T

        chi_eff_all = compute_chi_eff(params)
        chi_eff_g = compute_chi_eff(greedy)

        fig = plt.figure(figsize=(14, 6))

        # --- 3D plot: e vs q vs chi_eff ---
        ax = fig.add_subplot(1, 2, 1, projection='3d')

        sc = ax.scatter(
            e_all,
            q_all,
            chi_eff_all,
            color='gray',
            alpha=0.3,
            s=20,
            label="grid"
        )

        ax.scatter(
            e_g,
            q_g,
            chi_eff_g,
            color='blue',
            marker="x",
            s=60,
            label="selected"
        )

        ax.set_xlabel(f"eccentricity (N={len(greedy[:, 0])})")
        ax.set_ylabel(f"mass ratio q (N={len(greedy[:, 2])})")
        ax.set_zlabel(rf"$\chi_{{\mathrm{{eff}}}}$ (N={len(greedy[:, 3])})")
        ax.set_title(f"3D parameter space {train_obj.property} (e, q, chi_eff)")
        ax.legend()


        # --- 2D spin plot ---
        ax2 = fig.add_subplot(1, 2, 2)

        sc2 = ax2.scatter(
            chi1_all,
            chi2_all,
            c=chi_eff_all,
            cmap="viridis",
            alpha=0.4,
            s=20,
            label="grid"
        )

        ax2.scatter(
            chi1_g,
            chi2_g,
            c=chi_eff_g,
            cmap="viridis",
            marker="x",
            s=60,
            label="selected"
        )

        ax2.set_xlabel(rf"$\chi_1$ (N={len(greedy[:, 3])})")
        ax2.set_ylabel(rf"$\chi_2$ (N={len(greedy[:, 4])})")
        ax2.set_title(
            rf"$\chi_1$ vs $\chi_2$ (color = $\chi_{{\mathrm{{eff}}}}$), {train_obj.property}"
        )        
        ax2.legend()

        cbar2 = fig.colorbar(sc2, ax=ax2)
        cbar2.set_label(r"$\chi_{\mathrm{eff}}$")

        plt.tight_layout()

        if save_basis_indices_fig:
            figname = train_obj.figname(
                prefix="basis_indices",
                directory="Images/Basis_indices"
            )
            fig.savefig(figname, dpi=200, bbox_inches="tight")

    def _plot_greedy_errors(
        self,
        train_obj: TrainingSetResults,
        save_greedy_fig=False,
        free_memory=False
    ):
        train_obj.load_greedy_errors()
        print(f"Loaded greedy errors with {len(train_obj.greedy_errors)} entries.")

        stacked_proj_errors = np.asarray(train_obj.greedy_errors).ravel()

        fig_greedy_errors = plt.figure()
        plt.semilogy(
            np.arange(len(stacked_proj_errors)),
            stacked_proj_errors,
            label="greedy error",
            lw=1.8,
            color="blue",
        )
        plt.yscale("log")
        plt.ylabel(f"Greedy error {train_obj.property}")
        plt.title(
            f"Greedy error {train_obj.property} per section "
            f"tree_leaves = {len(train_obj.leaf_basis_indices)}"
        )
        plt.legend()

        if save_greedy_fig:
            figname = train_obj.figname(
                prefix="Greedy_error",
                directory="Images/Greedy_errors",
            )
            fig_greedy_errors.savefig(figname)

        if free_memory:
            train_obj.greedy_errors = None

        return stacked_proj_errors
    

    def _plot_SVD_matrix(self, 
                         train_obj: TrainingSetResults, 
                         save_SVD_matrix_fig=False,
                         free_memory=True
                         ):
        """Function to plot the SVD matrix of the dataset. Option to save figure in Images/SVD_matrices."""
        # Check if residuals are loaded, if not attempt to load
        train_obj.load_residuals()
        
        # Perform SVD on the dataset
        U, S, Vt = np.linalg.svd(train_obj.residuals, full_matrices=False)


        fig_SVD_matrix, axes = plt.subplots(1, 3, figsize=(18,4))

        # Construct the importance spectrum plot (singular values)
        axes[0].semilogy(S, marker='o')
        axes[0].set_xlabel("Mode index")
        axes[0].set_ylabel("Singular value")
        axes[0].set_title("Singular Value Spectrum")
        axes[0].grid(True)


        n_modes_to_plot = 3

        for i in range(n_modes_to_plot):
            # Plot the 3 vectors corresponding to the largest singular values (most dominant coherent structure)
            axes[1].plot(self.time, Vt[i], label=f"Mode {i+1}") 

            axes[1].set_xlabel("Time")
            axes[1].set_ylabel("Mode amplitude")
            axes[1].set_title("Dominant Time Modes")
            axes[1].legend()
            axes[1].grid(True)


        for i in range(n_modes_to_plot):
            # Plot the coefficients of the 3 most dominant modes as a function of eccentricity
            axes[2].plot(self.ecc_ref_space, U[:, i], label=f"Mode {i+1}")

        axes[2].set_xlabel("Eccentricity")
        axes[2].set_ylabel("Mode coefficient")
        axes[2].set_title("Mode Amplitude vs Eccentricity")
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()


        if save_SVD_matrix_fig:
            figname = train_obj.figname(prefix=f'SVD_matrix', directory='Images/SVD_matrices') # Use the figname method of the training object to generate a consistent filename            
            fig_SVD_matrix.savefig(figname)

            # plt.close('all')

        if free_memory:
            del U, S, Vt
            train_obj.residuals = None

    def _plot_greedy_vectors(
            self,
            train_obj: TrainingSetResults,
            save_greedy_vecs_fig=False,
            U=None,
            show_legend=False,
            free_memory=False
        ):
        """Plot greedy basis vectors (no empirical nodes)."""
        children = len(train_obj.leaf_basis_indices)
        print(f"Number of child nodes: {children}")

        base_width = 30
        base_height = 5
        fig_height = base_height + 1.5 * children if children > 0 else base_height

        fig_greedy_vecs, ax_main = plt.subplots(
            1, 1,
            figsize=(base_width, fig_height),
        )

        # --------------------------------------------------
        # Optional: full dataset overlay
        # --------------------------------------------------
        if U is not None:
            for i, vec in enumerate(U):
                ax_main.plot(
                    self.time,
                    vec,
                    color="grey",
                    alpha=0.3,
                    label="Training vectors" if i == 0 else None,
                )

        # --------------------------------------------------
        # Plot greedy basis vectors grouped by leaf
        # --------------------------------------------------
        if train_obj.orthonormal_basis is None:
            train_obj.load_orthonormal_basis()

        print(f"Loaded orthonormal basis with {len(train_obj.orthonormal_basis)} vectors.")

        basis_array = np.asarray(train_obj.orthonormal_basis)

        for leaf_id, idxs in enumerate(train_obj.leaf_basis_indices):
            idxs = np.asarray(idxs, dtype=int)
            basis = basis_array[idxs]

            for j, vec in enumerate(basis):
                label = (
                    f"Leaf {leaf_id}, e={train_obj.parameter_grid[idxs[j]]}"
                    if show_legend else None
                )

                ax_main.plot(
                    self.time,
                    vec,
                    linewidth=0.8,
                    label=label,
                )

        ax_main.set_ylabel("Basis function amplitude")
        ax_main.set_xlabel("time [M]")
        ax_main.set_title(
            f"Greedy Basis Vectors "
            f"({len(train_obj.basis_indices)} total, "
            f"{children} leaves)"
        )

        if show_legend:
            ax_main.legend(loc="best", ncol=3)

        ax_main.grid(True)
        # plt.tight_layout()

        if save_greedy_vecs_fig:
            figname = train_obj.figname(
                prefix="Greedy_vectors",
                directory="Images/Greedy_vectors",
            )
            fig_greedy_vecs.savefig(figname)

        if free_memory:
            train_obj.orthonormal_basis = None

        return fig_greedy_vecs

    def _plot_emp_nodes_on_basis(self, 
                                train_obj: TrainingSetResults, 
                                first_N_vectors=None,
                                save_fig=False):
        """  
        Plot the empirical nodes on top of the reduced basis functions for each leaf in the tree,
        plus a compact visualization of node locations.
        """
        # Load if needed
        train_obj.load_orthonormal_basis()

        # Plot per leaf: basis functions + nodes
        current_idx=0
        for i in range(len(train_obj.leaf_basis_indices)):
            len_leaf_basis = len(train_obj.leaf_basis_indices[i])
            
            # Get the reduced basis functions for this leaf
            reduced_basis = np.array(train_obj.orthonormal_basis)[current_idx : current_idx + len_leaf_basis]
            current_idx+=len(train_obj.leaf_basis_indices[i])
            
            # Only plot the first N vectors for clarity if specified
            if first_N_vectors is None:
                first_N_vectors = len_leaf_basis
            if len(reduced_basis) > first_N_vectors:
                reduced_basis = reduced_basis[:first_N_vectors]

            eim_nodes = train_obj.leaf_nodes_indices[i]
            # --- Create 2-row layout (3:1 ratio) ---
            fig_emp_nodes, (ax, ax_nodes) = plt.subplots(
                2, 1,
                figsize=(20, 6),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1]}
            )

            # =======================
            # Top plot: basis + nodes
            # =======================
            for k in range(len(reduced_basis)):
                ax.plot(
                    self.time,
                    reduced_basis[k],
                    alpha=0.7,
                    lw=1.5
                )

                # Mark node for that basis function
                ax.scatter(
                    self.time[eim_nodes],
                    reduced_basis[k, eim_nodes],
                    color="red",
                    zorder=5,
                    s=2
                )

            # Vertical lines for all nodes
            for node in eim_nodes:
                ax.axvline(self.time[node], color='red', alpha=0.2)

            ax.set_ylabel("Basis function amplitude")
            ax.set_title(f"Reduced basis functions and empirical interpolation nodes: leaf {i}")

            # =======================
            # Bottom plot: node locations
            # =======================
            y_line = 0.0  # horizontal reference

            # horizontal line
            ax_nodes.hlines(y_line, self.time[0], self.time[-1], color="black", lw=1)

            # scatter nodes on line
            ax_nodes.scatter(
                self.time[eim_nodes],
                [y_line] * len(eim_nodes),
                color="red",
                zorder=5
            )

            # clean look
            ax_nodes.set_yticks([])
            ax_nodes.set_xlabel("time [M]")
            ax_nodes.set_title("Empirical node locations")

            # optional: remove spines for clarity
            ax_nodes.spines["left"].set_visible(False)
            ax_nodes.spines["right"].set_visible(False)
            ax_nodes.spines["top"].set_visible(False)

            fig_emp_nodes.tight_layout()

            if save_fig:
                figname = train_obj.figname(
                    prefix=f"RB_functions_with_emp_nodes_leaf_{i}",
                    directory="Images/Empirical_nodes"
                )
                fig_emp_nodes.savefig(figname)


    def _plot_interpolation_matrix(self, 
                                    train_obj: TrainingSetResults,
                                    save_fig=False):
        # Load saved orthonormal basis 
        train_obj.load_orthonormal_basis()
        
        n_leaves = len(train_obj.leaf_nodes_indices)

        fig, ax = plt.subplots(n_leaves, 1, squeeze=False)
        ax = ax.ravel()

        current_idx=0
        for leaf_id in range(len(train_obj.leaf_basis_indices)):
            len_leaf_basis = len(train_obj.leaf_basis_indices[leaf_id])
            # Get the reduced basis functions for this leaf
            reduced_basis = np.array(train_obj.orthonormal_basis)[current_idx : current_idx + len_leaf_basis]
            current_idx+=len(train_obj.leaf_basis_indices[leaf_id])
            
            eim_nodes = train_obj.leaf_nodes_indices[leaf_id]

            # Assuming basis shape = (n_basis, n_samples)
            V = reduced_basis[:, eim_nodes]

            im = ax[leaf_id].imshow(V, aspect='auto', origin='lower')
            ax[leaf_id].set_title(f"EIM interpolation matrix (leaf {leaf_id})")
            ax[leaf_id].set_xlabel("Empirical node index")
            ax[leaf_id].set_ylabel("Basis index")

        fig.colorbar(im, ax=ax)

        plt.tight_layout()

        if save_fig:
            fig_path = train_obj.figname(
                prefix="EIM_interpolation_matrix",
                directory="Images/Empirical_nodes"
            )

            fig.savefig(fig_path)


    def _plot_projection_vs_eim_error(self, 
                                      train_obj:TrainingSetResults, 
                                      save_fig=False):
        # Check if residuals are loaded, if not attempt to load
        train_obj.load_residuals()
        dataset = np.asarray(train_obj.residuals)

        all_proj_errors = []
        all_eim_errors = []
        all_ratios = []

        current_idx=0
        # Get per leaf variables and calculate errors
        for leaf_id in range(len(train_obj.leaf_basis_indices)):
            len_leaf_basis = len(train_obj.leaf_basis_indices[leaf_id])
            # Get the reduced basis functions for this leaf
            reduced_basis = np.array(train_obj.orthonormal_basis)[current_idx : current_idx + len_leaf_basis]
            current_idx+=len(train_obj.leaf_basis_indices[leaf_id])

            eim_nodes = np.asarray(train_obj.leaf_nodes_indices[leaf_id])
            leaf_dataset = dataset[train_obj.leaf_basis_indices[leaf_id]]            # data belonging to this leaf

            proj_errors = []
            eim_errors = []
            ratios = []

            for k in range(1, len(eim_nodes) + 1):
                basis_k = reduced_basis[:k]                 # (k, n_time)
                nodes_k = eim_nodes[:k]

                # Interpolation matrix
                V_k = basis_k[:, nodes_k]                   # (k, k)

                max_proj_err = 0.0
                max_eim_err = 0.0

                for f in leaf_dataset:
                    norm_f = np.linalg.norm(f)
                    if norm_f == 0:
                        continue

                    # Best projection
                    coeff_proj, *_ = np.linalg.lstsq(basis_k.T, f, rcond=None)
                    f_proj = basis_k.T @ coeff_proj
                    proj_err = np.linalg.norm(f - f_proj) / norm_f
                    max_proj_err = max(max_proj_err, proj_err)

                    # EIM approximation
                    f_nodes = f[nodes_k]
                    coeff_eim = np.linalg.solve(V_k.T, f_nodes)
                    f_eim = basis_k.T @ coeff_eim
                    eim_err = np.linalg.norm(f - f_eim) / norm_f
                    max_eim_err = max(max_eim_err, eim_err)

                proj_errors.append(max_proj_err)
                eim_errors.append(max_eim_err)

                if max_proj_err > 0:
                    ratios.append(max_eim_err / max_proj_err)
                else:
                    ratios.append(np.nan)

            all_proj_errors.append(proj_errors)
            all_eim_errors.append(eim_errors)
            all_ratios.append(ratios)

            # --- Plot for this leaf ---
            fig, axs = plt.subplots(
                2, 1,
                figsize=(8, 6),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1]}
            )

            x = np.arange(1, len(eim_nodes) + 1)

            # Error plot
            axs[0].semilogy(x, proj_errors, marker='o', label='Projection error')
            axs[0].semilogy(x, eim_errors, marker='s', label='EIM error')
            axs[0].set_ylabel("Max relative error")
            axs[0].set_title(f"Projection error vs EIM error — Leaf {leaf_id}")
            axs[0].grid(True)
            axs[0].legend()
   
            # Ratio plot
            axs[1].plot(x, ratios, marker='^', label='EIM / Projection')
            axs[1].set_xlabel("Number of modes / EIM nodes")
            axs[1].set_ylabel("Ratio")
            axs[1].grid(True)
            axs[1].legend()

            fig.tight_layout()

            if save_fig:
                fig_path = train_obj.figname(
                    prefix=f"Projection_vs_EIM_error_leaf_{leaf_id}",
                    directory="Images/Empirical_nodes"
                )
                fig.savefig(fig_path)

        return all_proj_errors, all_eim_errors, all_ratios
  
    

    def _plot_emp_nodes_on_residuals(self, 
                                     train_obj:TrainingSetResults, 
                                     save_fig, 
                                     N_greedy_vecs_to_plot=5,
                                     show_legend=True):
        """
        Helper function to plot and optionally save the training set of residuals.

        Parameters:
        ----------------
        - residual_training_set (numpy.ndarray): Training set of residuals at empirical nodes.
        - property (str): The waveform property ('phase' or 'amplitude').
        - save_fig (bool): If True, saves the plot to a file.
        """
        # Check if residuals are loaded, if not attempt to load
        train_obj.load_residuals()

        basis_indices_sorted = np.sort(train_obj.basis_indices)
        plot_residuals_idx = basis_indices_sorted[
            np.linspace(
                0,
                len(basis_indices_sorted) - 1,
                N_greedy_vecs_to_plot,
                dtype=int
            )
        ]

        fig, ax = plt.subplots()

        for idx in plot_residuals_idx:
            ax.plot(self.time, train_obj.residuals[idx], label=f'e = [{train_obj.parameter_grid[idx][0]}]', linewidth=0.6)
            ax.scatter(self.time[train_obj.empirical_indices], train_obj.residuals[idx][train_obj.empirical_indices])

        ax.set_xlabel('t [M]')
        ax.set_ylabel('greedy residual')
        if show_legend:
            ax.legend()
        ax.set_title('Residual Training Set')
        ax.grid(True)

        if save_fig:
            figname = train_obj.figname(prefix='Training_set', directory='Images/TrainingSet')
            fig.savefig(figname)

    def _build_default_gp(self, n_input_dims):
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(
                length_scale=np.ones(n_input_dims),
                length_scale_bounds=(1e-3, 1e3),
                nu=2.5,
            )
            + WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-12, 1e-2))
        )

        return GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=1,
            random_state=0,
        )



# Sampling parameters
sampling_frequency = 2048 # or 4096
duration = 4 # seconds
time_array = np.linspace(-duration, 0, int(sampling_frequency * duration))  # time in seconds

gt = Generate_TrainingSet(time_array=time_array,
                          ecc_ref_parameterspace=np.linspace(0.001, 0.3, num=2),
                          mass_ratio_parameterspace=np.linspace(1, 20, num=2),
                          chi1_parameterspace=np.linspace(-0.9, 0.9, num=2),
                          chi2_parameterspace=np.linspace(-0.9, 0.9, num=2),
                          min_greedy_error_amp=1e-6,
                          min_greedy_error_phase=1e-6,
                          truncate_at_tmin=True,
                          truncate_at_ISCO=True,
                          f_lower=10)

train_obj_a = gt._get_training_obj('amplitude')
gt._calculate_residuals(train_obj_a,
                        plot_polarizations=True,
                        plot_residuals_param_evolve=True,
                        plot_residuals_l_evolve=True)

train_obj_p = gt._get_training_obj('phase')
gt._calculate_residuals(train_obj_p,
                        plot_polarizations=True,
                        plot_residuals_param_evolve=True,
                        plot_residuals_l_evolve=True)

# gt.get_training_set_greedy(property="amplitude", 
#                             plot_interpolation_matrix=True, save_fig_interpolation_matrix=True,
#                             plot_proj_vs_eim_error=True, save_fig_proj_vs_eim_error=True,
#                             plot_residuals_time=True, save_fig_residuals_time=True,
#                             plot_emp_nodes_on_basis=True, save_fig_emp_nodes_on_basis=True,
#                             plot_training_set=True, save_fig_training_set=True,
#                             plot_residuals_eccentric=True, save_fig_residuals_eccentric=True,
#                             plot_greedy_error=True,save_fig_greedy_error=True
#                         )

# gt.get_training_set_greedy(property="phase", 
#                             plot_interpolation_matrix=True, save_fig_interpolation_matrix=True,
#                             plot_proj_vs_eim_error=True, save_fig_proj_vs_eim_error=True,
#                             plot_residuals_time=True, save_fig_residuals_time=True,
#                             plot_emp_nodes_on_basis=True, save_fig_emp_nodes_on_basis=True,
#                             plot_training_set=True, save_fig_training_set=True,
#                             plot_residuals_eccentric=True, save_fig_residuals_eccentric=True,
#                             plot_greedy_error=True, save_fig_greedy_error=True,
#                             plot_basis_indices=True, save_fig_basis_indices=True,
#                             )

plt.show()
plt.close("all")
