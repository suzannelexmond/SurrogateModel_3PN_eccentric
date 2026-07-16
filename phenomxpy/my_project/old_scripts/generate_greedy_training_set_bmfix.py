# =====================================================================
# [OPTIMIZED VERSION] Memory improvements for generate_greedy_training_set.py
# Applied Safe Quick Wins #1-5 (comments marked with #[OPTIMIZED])
# =====================================================================


from generate_PhenomTE import *

import itertools
import h5py

from skreducedmodel.reducedbasis import ReducedBasis
from skreducedmodel.empiricalinterpolation import EmpiricalInterpolation

# --------------------------------------------------------
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from scipy.stats import skew, kurtosis, normaltest, norm

# --------------------------------------------------------------------
# plt.switch_backend('WebAgg')
from dataclasses import dataclass, field
from typing import Any, Optional

import traceback


@dataclass
class TrainingSetResults(Warnings):
    property: str = "phase"

    # Put ALL fields with ANY kind of default together at the end
    ecc_ref_space: Any = None
    mean_ano_ref_space: Any = None
    mass_ratio_space: Any = None
    chi1_space: Any = None
    chi2_space: Any = None
    parameter_grid: Any = None
    time: Any = None

    N_basis_vecs: Optional[int] = None
    min_greedy_error: Optional[float] = None

    f_ref: float = 20.0
    f_lower: float = 10.0
    phiRef: float = 0.0
    inclination: float = 0.0
    truncate_at_ISCO: bool = True
    truncate_at_tmin: bool = True
    luminosity_distance: Optional[float] = None
    
    hp_dataset: Any = None
    hc_dataset: Any = None
    residuals: Any = None
    training_set: Any = None
    orthonormal_basis: Any = None
    greedy_errors: Any = None
    
    # Lists MUST use field(default_factory=list) if mixed with None defaults above
    basis_indices: Any = field(default_factory=list)
    empirical_indices: Any = field(default_factory=list)
    leaf_basis_indices: Any = field(default_factory=list)
    leaf_nodes_indices: Any = field(default_factory=list)
    
    # File paths
    residuals_file: str = ""
    polarisation_file: str = ""
    orthonormal_basis_file: str = ""
    greedy_errors_file: str = ""


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
            os.makedirs(directory, exist_ok=True)
            return f"{directory.rstrip('/')}/{name}"
        return name

    def figname(self, prefix="fig", ext="png", directory=None, include_greedy=True, exclude_property=True, include_extra=False):
        if directory is not None:
            os.makedirs(directory, exist_ok=True)
        figname = self.filename(prefix=prefix, ext=ext, directory=directory, include_greedy=include_greedy, exclude_property=exclude_property, include_extra=include_extra)
        print(self.colored_text(f"Figure is saved in {figname}", 'blue'))
        return figname
    
    def save(self,
            prefix="training_set",
            directory="Straindata/TrainingSetResults",
            save_residuals=False,
            save_polarizations=False,
            save_orthonormal_basis=False,
            save_greedy_errors=False,
            ):
        """Main save function - delegates to specialized save methods."""
        os.makedirs(directory, exist_ok=True)
        filepath = self.filename(prefix=prefix, ext="h5", directory=directory)
        
        if os.path.exists(filepath):
            print(self.colored_text(f"File already exists: {filepath}", "yellow"))
            return filepath
        
        # === CORE DATA TO MAIN H5 FILE ===
        with h5py.File(filepath, "w") as f:
            meta = f.create_group("meta")
            meta.attrs["property"] = self.property
            meta.attrs["f_ref"] = self.f_ref
            meta.attrs["f_lower"] = self.f_lower
            meta.attrs["phiRef"] = self.phiRef
            meta.attrs["inclination"] = self.inclination
            
            # Save scalar truncation flags
            meta.attrs["truncate_at_ISCO"] = self.truncate_at_ISCO
            meta.attrs["truncate_at_tmin"] = self.truncate_at_tmin
            
            # PARAMETER SPACE ARRAYS AS DATASETS (CRITICAL - don't rely on metadata strings!)
            if self.ecc_ref_space is not None:
                f.create_dataset("ecc_ref_space", data=self.ecc_ref_space)
            else:
                f.create_dataset("ecc_ref_space", data=np.array([]))
            
            if self.mean_ano_ref_space is not None:
                f.create_dataset("mean_ano_ref_space", data=self.mean_ano_ref_space)
            else:
                f.create_dataset("mean_ano_ref_space", data=np.array([]))
                
            if self.mass_ratio_space is not None:
                f.create_dataset("mass_ratio_space", data=self.mass_ratio_space)
            else:
                f.create_dataset("mass_ratio_space", data=np.array([]))
                
            if self.chi1_space is not None:
                f.create_dataset("chi1_space", data=self.chi1_space)
            else:
                f.create_dataset("chi1_space", data=np.array([]))
                
            if self.chi2_space is not None:
                f.create_dataset("chi2_space", data=self.chi2_space)
            else:
                f.create_dataset("chi2_space", data=np.array([]))
            
            # Optional: keep metadata strings as BACKWARD COMPATIBILITY reference only
            meta.attrs["ecc_range"] = self._range_block("e", self.ecc_ref_space) if self.ecc_ref_space is not None else ""
            meta.attrs["mean_anomaly_range"] = self._range_block("l", self.mean_ano_ref_space) if self.mean_ano_ref_space is not None else ""
            meta.attrs["mass_ratio_range"] = self._range_block("q", self.mass_ratio_space) if self.mass_ratio_space is not None else ""
            meta.attrs["spin1_range"] = self._range_block("x1", self.chi1_space) if self.chi1_space is not None else ""
            meta.attrs["spin2_range"] = self._range_block("x2", self.chi2_space) if self.chi2_space is not None else ""
            
            # Grid & temporal info
            f.create_dataset("parameter_grid", data=self.parameter_grid, compression="gzip")
            f.create_dataset("time", data=self.time, compression="gzip")
            
            # Greedy selection indices
            f.create_dataset("basis_indices", data=np.array(self.basis_indices))
            f.create_dataset("empirical_indices", data=np.array(self.empirical_indices))
            f.create_dataset("leaf_basis_indices", data=np.array(self.leaf_basis_indices))
            f.create_dataset("leaf_nodes_indices", data=np.array(self.leaf_nodes_indices))
            
            # Primary training set data
            f.create_dataset("training_set", data=self.training_set, compression="gzip", chunks=True)
        
        # === DELEGATE LARGE ARRAY SAVING ===
        
        if save_residuals:
            if self.residuals_file == "":
                if self.residuals is None:
                    raise ValueError("Residuals not available. Cannot save.")
                
                res_path = self.save_residuals(
                    prefix="residuals",
                    directory=f"{directory}/../Residuals",
                    free_memory=True
                )
                with h5py.File(filepath, "a") as f:
                    f["meta"].attrs["residuals_file"] = res_path
                self.residuals_file = res_path
            else:
                print(self.colored_text("Residuals already saved.", "yellow"))
        
        if save_polarizations:
            if self.polarisation_file == "":
                if self.hp_dataset is None or self.hc_dataset is None:
                    raise ValueError("Polarizations not available. Cannot save.")
                
                pol_path = self.save_polarizations(
                    prefix="polarisation", 
                    directory=f"{directory}/../Polarisations",
                    free_memory=True
                )
                with h5py.File(filepath, "a") as f:
                    f["meta"].attrs["polarisation_file"] = pol_path
                self.polarisation_file = pol_path
            else:
                print(self.colored_text("Polarizations already saved.", "yellow"))
        
        if save_orthonormal_basis:
            if self.orthonormal_basis_file == "":
                if self.orthonormal_basis is None:
                    raise ValueError("Orthonormal basis not available. Cannot save.")
                
                basis_path = self.save_orthonormal_basis(
                    prefix="orthonormal_basis",
                    directory=f"{directory}/../Basis",
                    free_memory=False
                )
                with h5py.File(filepath, "a") as f:
                    f["meta"].attrs["orthonormal_basis_file"] = basis_path
                self.orthonormal_basis_file = basis_path
            else:
                print(self.colored_text("Orthonormal basis already saved.", "yellow"))
        
        if save_greedy_errors:
            if self.greedy_errors_file == "":
                if self.greedy_errors is None:
                    raise ValueError("Greedy errors not available. Cannot save.")
                
                err_path = self.save_greedy_errors(
                    prefix="greedy_errors",
                    directory=f"{directory}/../Greedy",
                    free_memory=True
                )
                with h5py.File(filepath, "a") as f:
                    f["meta"].attrs["greedy_errors_file"] = err_path
                self.greedy_errors_file = err_path
            else:
                print(self.colored_text("Greedy errors already saved.", "yellow"))
        
        gc.collect()
        if MEMORY_PROFILE:
            check_memory_usage(f"After TrainingSetResults.save(): {filepath}")
        
        return filepath
    

    def load(self,
            prefix="training_set",
            directory="Straindata/TrainingSetResults",
            load_residuals=False,
            load_polarisations=False,
            load_orthonormal_basis=False,
            load_greedy_errors=False):
        """Load core data plus optionally load separately-stored large arrays."""
        
        filepath = self.filename(prefix=prefix, ext="h5", directory=directory)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Training set file not found: {filepath}")
        
        with h5py.File(filepath, "r") as f:
            # Meta attributes (SCALARS only - don't overwrite arrays!)
            self.property = f["meta"].attrs.get("property", "phase")
            self.f_ref = f["meta"].attrs.get("f_ref", 20.0)
            self.f_lower = f["meta"].attrs.get("f_lower", 10.0)
            self.phiRef = f["meta"].attrs.get("phiRef", 0.0)
            self.inclination = f["meta"].attrs.get("inclination", 0.0)
            
            # TRUNCATE FLAGS - parse from attrs if needed
            self.truncate_at_ISCO = f["meta"].attrs.get("truncate_at_ISCO", True)
            self.truncate_at_tmin = f["meta"].attrs.get("truncate_at_tmin", True)
            
            # LOAD ARRAYS DIRECTLY FROM DATASETS (don't recreate from metadata strings!)
            self.parameter_grid = f["parameter_grid"][:]
            self.time = f["time"][:]
            
            # These should also exist in datasets - load them!
            if "ecc_ref_space" in f:
                self.ecc_ref_space = f["ecc_ref_space"][:]
            if "mean_ano_ref_space" in f:
                self.mean_ano_ref_space = f["mean_ano_ref_space"][:]
            if "mass_ratio_space" in f:
                self.mass_ratio_space = f["mass_ratio_space"][:]
            if "chi1_space" in f:
                self.chi1_space = f["chi1_space"][:]
            if "chi2_space" in f:
                self.chi2_space = f["chi2_space"][:]
            
            self.basis_indices = list(f["basis_indices"][:])
            self.empirical_indices = list(f["empirical_indices"][:])
            self.leaf_basis_indices = [list(leaf) for leaf in f["leaf_basis_indices"]] if "leaf_basis_indices" in f else []
            self.leaf_nodes_indices = [list(nodes) for nodes in f["leaf_nodes_indices"]] if "leaf_nodes_indices" in f else []
            self.training_set = f["training_set"][:]
            
            # Store file paths from metadata
            self.residuals_file = f["meta"].attrs.get("residuals_file", "")
            self.polarisation_file = f["meta"].attrs.get("polarisation_file", "")
            self.orthonormal_basis_file = f["meta"].attrs.get("orthonormal_basis_file", "")
            self.greedy_errors_file = f["meta"].attrs.get("greedy_errors_file", "")
        
        print(self.colored_text(f"Loaded training set: {filepath}", "blue"))
        
        # Delegate to specialized loaders if requested
        if load_residuals and self.residuals_file:
            self.load_residuals()
        if load_polarisations and self.polarisation_file:
            self.load_polarizations()
        if load_orthonormal_basis and self.orthonormal_basis_file:
            self.load_orthonormal_basis()
        if load_greedy_errors and self.greedy_errors_file:
            self.load_greedy_errors()
        
        if MEMORY_PROFILE:
            check_memory_usage(f"After TrainingSetResults.load(): {filepath}")
        
        return self
    

    # ========== SPECIALIZED SAVE METHODS ==========
    
    def save_residuals(self, prefix="residuals", directory="Straindata/Residuals", free_memory=True):
        """Save residuals to external HDF5 file. Always free_memory=True recommended."""
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
    

    def save_polarizations(self, prefix="polarisation", directory="Straindata/Polarisations", free_memory=True):
        """Save polarizations to external HDF5 file. Always free_memory=True recommended."""
        if self.hp_dataset is None or self.hc_dataset is None:
            raise ValueError("Polarizations not available.")
        
        os.makedirs(directory, exist_ok=True)
        filepath = self.filename(prefix=prefix, ext="h5", directory=directory, include_greedy=False, exclude_property=True)
        
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
    

    def save_orthonormal_basis(self, prefix="orthonormal_basis", directory="Straindata/Basis", free_memory=False):
        """Save orthonormal basis to external HDF5 file. Memory kept by default for later use."""
        if self.orthonormal_basis is None:
            raise ValueError("Orthonormal basis not available.")
        
        os.makedirs(directory, exist_ok=True)
        filepath = self.filename(prefix=prefix, ext="h5", directory=directory)
        
        if not os.path.exists(filepath):
            with h5py.File(filepath, "w") as f:
                f.create_dataset("orthonormal_basis", data=np.asarray(self.orthonormal_basis), compression="gzip", chunks=True)
        
        self.orthonormal_basis_file = filepath
        print(self.colored_text(f"Orthonormal basis saved: {filepath}", "blue"))
        
        # NOTE: We deliberately DO NOT free orthonormal_basis here
        # It's needed for subsequent interpolation/surrogate building
        
        if free_memory:
            # This would only be called explicitly elsewhere
            self.orthonormal_basis = None
            gc.collect()
            if MEMORY_PROFILE:
                check_memory_usage(f"After save_orthonormal_basis(): {filepath}")
        
        return filepath
    

    def save_greedy_errors(self, prefix="greedy_errors", directory="Straindata/Greedy", free_memory=True):
        """Save greedy errors to external HDF5 file. Always free_memory=True recommended."""
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
    

    # ========== SPECIALIZED LOAD METHODS ==========
    
    def load_residuals(self, prefix="residuals", directory="Straindata/Residuals"):
        filepath = self.filename(prefix=prefix, ext="h5", directory=directory, include_greedy=False)
        
        if not os.path.exists(filepath):
            # FILE DOESN'T EXIST - RAISE ERROR NOW
            raise FileNotFoundError(f"Cannot load residuals: {filepath} does not exist")
        
        if self.residuals is None:
            with h5py.File(filepath, "r") as f:
                self.residuals = np.array(f["residuals"])
                self.time = np.array(f["time"])
            print(self.colored_text(f"Residuals loaded: {filepath}", 'blue'))
        
        return self
    

    def load_polarizations(self, prefix="polarisation", directory="Straindata/Polarisations"):
        filepath = self.filename(prefix=prefix, ext="h5", directory=directory, include_greedy=False, exclude_property=True)

        if (self.hp_dataset is None or self.hc_dataset is None) and filepath:
            with h5py.File(filepath, "r") as f:
                self.hp_dataset = np.array(f["hp"])
                self.hc_dataset = np.array(f["hc"])
                self.time = np.array(f["time"])
            print(self.colored_text(f"Polarizations loaded: {filepath}", 'blue'))
        
        return self
    

    def load_orthonormal_basis(self, prefix="orthonormal_basis", directory="Straindata/Basis"):
        filepath = self.filename(prefix=prefix, ext="h5", directory=directory)
        
        if self.orthonormal_basis is None and filepath:
            with h5py.File(filepath, "r") as f:
                self.orthonormal_basis = np.array(f["orthonormal_basis"])
            print(self.colored_text(f"Orthonormal basis loaded: {filepath}", 'blue'))
        
        return self
    

    def load_greedy_errors(self, prefix="greedy_errors", directory="Straindata/Greedy"):
        filepath = self.filename(prefix=prefix, ext="h5", directory=directory)
        
        if self.greedy_errors is None and filepath:
            with h5py.File(filepath, "r") as f:
                self.greedy_errors = np.array(f["greedy_errors"])
            print(self.colored_text(f"Greedy errors loaded: {filepath}", 'blue'))
        
        return self

@dataclass
class FlowDiagnosticResult:
    mean_abs_skew: float
    mean_abs_excess_kurtosis: float
    frac_non_normal_nodes: float
    mean_abs_offdiag_corr: float
    max_abs_offdiag_corr: float
    coverage_1sigma: float
    coverage_2sigma: float
    recommendation: str

class Generate_TrainingSet(Waveform_Properties, Simulate_Waveform):
    """
    Class to generate a training dataset for gravitational waveform simulations using a greedy algorithm and empirical interpolation.
    Inherits from WaveformProperties and SimulateInspiral to leverage methods for waveform 
    property calculations and waveform generation.

    """

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
        if MEMORY_PROFILE:
            check_memory_usage("START Generate_TrainingSet.__init__")

        # Check if property is valid and adjust settings accordingly
        self.ecc_ref_space = self.allowed_eccentricity_warning(ecc_ref_parameterspace)
        self.mass_ratio_space = self.allowed_mass_ratio_warning(mass_ratio_parameterspace)
        self.mean_ano_ref_space = self.allowed_mean_anomaly_warning(mean_ano_parameterspace)
        self.chi1_space = self.allowed_chispin_warning(chi1_parameterspace)
        self.chi2_space = self.allowed_chispin_warning(chi2_parameterspace)

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

        # To be stored parameters
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
                         mass_ratio=1., 
                         luminosity_distance=None, 
                         f_lower=f_lower, 
                         f_ref=f_ref, 
                         chi1=0., 
                         chi2=0., 
                         phiRef=phiRef, 
                         inclination=inclination, 
                         truncate_at_ISCO=truncate_at_ISCO, 
                         truncate_at_tmin=truncate_at_tmin,
                         geometric_units=True)
        
        if MEMORY_PROFILE:
            check_memory_usage("END Generate_TrainingSet.__init__")

    def result_kwargs_training(self, property,
                               ecc_ref_space=None,
                               mean_ano_ref_space=None,
                               mass_ratio_space=None,
                               chi1_space=None,
                               chi2_space=None,
                               time=None,
                               N_basis_vecs_phase=None,
                               N_basis_vecs_amp=None,
                               min_greedy_error_phase=None,
                               min_greedy_error_amp=None,
                               f_ref=None,
                               f_lower=None,
                               phiRef=None,
                               inclination=None,
                               truncate_at_ISCO=None,
                               truncate_at_tmin=None):
        """Helper function to resolve the parameters for the TrainingSetResults object based on the provided arguments or default class attributes."""
        # Resolve the parameter spaces and other parameters, using the provided arguments or default class attributes

        # Time domain for the waveforms
        time = self.resolve_property(prop=time, default=self.time)

        # Initial parameters of the binary BBH system
        ecc_ref_space = self.resolve_property(prop=ecc_ref_space, default=self.ecc_ref_space)
        mean_ano_ref_space = self.resolve_property(prop=mean_ano_ref_space, default=self.mean_ano_ref_space)
        mass_ratio_space = self.resolve_property(prop=mass_ratio_space, default=self.mass_ratio_space)
        chi1_space = self.resolve_property(prop=chi1_space, default=self.chi1_space)
        chi2_space = self.resolve_property(prop=chi2_space, default=self.chi2_space)

        # Build full parameter grid
        param_list = list(itertools.product(
            ecc_ref_space, 
            mean_ano_ref_space, 
            mass_ratio_space, 
            chi1_space, 
            chi2_space
            ))
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
            property=property,
            ecc_ref_space=ecc_ref_space,
            mean_ano_ref_space=mean_ano_ref_space,
            mass_ratio_space=mass_ratio_space,
            chi1_space=chi1_space,
            chi2_space=chi2_space,
            parameter_grid=parameter_grid,
            time=time,
            N_basis_vecs=N_basis_vecs,
            min_greedy_error=min_greedy_error,
            f_ref=f_ref,
            f_lower=f_lower,
            phiRef=phiRef,
            inclination=inclination,
            truncate_at_ISCO=truncate_at_ISCO,
            truncate_at_tmin=truncate_at_tmin,
        )


    def _get_training_obj(self, property):
        if property == "amplitude":
            if self.training_amp is None:
                self.training_amp = TrainingSetResults(
                    **self.result_kwargs_training(property="amplitude")
                )
            return self.training_amp
        elif property == "phase":
            if self.training_phase is None:
                self.training_phase = TrainingSetResults(
                    **self.result_kwargs_training(property="phase")
                )
            return self.training_phase
        else:
            raise ValueError(f"Unknown property: {property}")
        

    def generate_property_dataset(self, train_obj: TrainingSetResults, 
                                  ecc_ref_list=None, 
                                  mean_ano_ref_list=None, 
                                  mass_ratios_list=None, 
                                  chi1_list=None, 
                                  chi2_list=None, 
                                  save_residuals=True, 
                                  save_polarizations=True,
                                  plot_polarizations=False, save_fig_polarizations=False,
                                  plot_residuals_time_evolve=False, save_fig_time_evolve=False,
                                  plot_residuals_eccentric_evolve=False, save_fig_eccentric_evolve=False,  
                                  ):
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
        if MEMORY_PROFILE:
            check_memory_usage("START generate_property_dataset")

        # Resolve the parameter space for eccentricities and mass ratios, using the provided lists or default spaces
        train_obj.ecc_ref_space = self.resolve_property(prop=ecc_ref_list, default=train_obj.ecc_ref_space) 
        train_obj.mean_ano_ref_space = self.resolve_property(prop=mean_ano_ref_list, default=train_obj.mean_ano_ref_space)
        train_obj.mass_ratio_space = self.resolve_property(prop=mass_ratios_list, default=train_obj.mass_ratio_space) 
        train_obj.chi1_space = self.resolve_property(prop=chi1_list, default=train_obj.chi1_space)
        train_obj.chi2_space = self.resolve_property(prop=chi2_list, default=train_obj.chi2_space)
        train_obj.truncate_at_ISCO = self.resolve_property(prop=train_obj.truncate_at_ISCO, default=self.truncate_at_ISCO)
        train_obj.truncate_at_tmin = self.resolve_property(prop=train_obj.truncate_at_tmin, default=self.truncate_at_tmin)

        try:
        # Attempt to load existing residual dataset
            train_obj = train_obj.load_residuals()
            self.time = train_obj.time

            if plot_residuals_eccentric_evolve or plot_residuals_time_evolve:
                self._plot_residuals(train_obj, 
                                     plot_residuals_eccentric_evolve, save_fig_eccentric_evolve, 
                                     plot_residuals_time_evolve, save_fig_time_evolve
                                     )

        except Exception as e:
            print(e)
            traceback.print_exc()

            # # If attempt to load residuals failed, generate polarisations and calculate residuals
            # hp_dataset, hc_dataset = self._generate_polarisation_data(train_obj=train_obj, save_polarizations=save_polarizations)
            self._calculate_residuals(train_obj=train_obj, 
                                      truncate_at_ISCO=train_obj.truncate_at_ISCO,
                                      truncate_at_tmin=train_obj.truncate_at_tmin,
                                      save_residuals=save_residuals, 
                                      save_polarizations=save_polarizations,
                                      plot_polarizations=plot_polarizations, save_fig_polarizations=save_fig_polarizations,
                                      plot_residuals_eccentric_evolve=plot_residuals_eccentric_evolve, 
                                      plot_residuals_time_evolve=plot_residuals_time_evolve, 
                                      save_fig_eccentric_evolve=save_fig_eccentric_evolve, 
                                      save_fig_time_evolve=save_fig_time_evolve,
                                      )
            # del hp_dataset, hc_dataset  # Free memory
        return train_obj
   

    def _calculate_residuals(self, 
                             train_obj:TrainingSetResults, 
                             truncate_at_ISCO=None,
                             truncate_at_tmin=None,
                             save_residuals=True, 
                             save_polarizations=True,
                             plot_polarizations=False, save_fig_polarizations=False,
                             plot_residuals_eccentric_evolve=False, save_fig_time_evolve=False,
                             plot_residuals_time_evolve=False, save_fig_eccentric_evolve=False,  
                             ):
        """
        Helper function to calculate residuals for a property given polarisation data.

        Parameters:
        ----------
        ecc_list : list of floats
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
            
        """
        # Start timer to track how long the residual calculation takes
        start = time.time()

        truncate_at_ISCO = self.resolve_property(prop=truncate_at_ISCO, default=self.truncate_at_ISCO)
        truncate_at_tmin = self.resolve_property(prop=truncate_at_tmin, default=self.truncate_at_tmin)
        
        def calculate_residual_wrapper(hp, hc, ecc, l, q, chi1, chi2):
            # Calculate circular waveform for the same parameters, but with ecc=0, to use as reference for the residual calculation
            self.circulair_wf(mass_ratio=q,
                            mean_ano_ref=l,
                            chi1=chi1,
                            chi2=chi2)
            
            if not np.any(ecc):   # in case of no eccentric values
                if train_obj.property == "phase":
                    return self.phase_circ
                elif train_obj.property == "amplitude":
                    return self.amp_circ
                else:
                    raise ValueError("property must be 'phase' or 'amplitude'")
            else:
                residual = self.calculate_residual(
                                                hp, 
                                                hc, 
                                                mean_ano_ref=l,
                                                ecc_ref=ecc, 
                                                mass_ratio=q, 
                                                chi1=chi1, 
                                                chi2=chi2, 
                                                property=train_obj.property
                                                )
                
                return residual


        try:
            """ Try loading residuals from previous pre-computation """
            if plot_polarizations:
                train_obj = train_obj.load_polarizations(prefix="polarisation", directory="Straindata/Polarisations")
                train_obj = train_obj.load_residuals(prefix="residuals", directory="Straindata/Residuals")
            else:
                train_obj = train_obj.load_residuals(prefix="residuals", directory="Straindata/Residuals")
            self.time = train_obj.time

        except Exception as e:
            """No pre-computed residuals found. Attempting to load polarisations to prevent re-computation of waveforms."""
            print(e)
            traceback.print_exc()
            n_params = len(train_obj.parameter_grid)
            
            try:
                """ Try loading polarisations to prevent re-computation """
                train_obj = train_obj.load_polarizations(prefix="polarisation", directory="Straindata/Polarisations")
                self.time = train_obj.time

                hp_flat = train_obj.hp_dataset.reshape(len(train_obj.parameter_grid), len(train_obj.time))
                hc_flat = train_obj.hc_dataset.reshape(len(train_obj.parameter_grid), len(train_obj.time))

                residuals_flat = np.empty_like(hp_flat)
                for idx, (ecc, l, q, chi1, chi2) in enumerate(train_obj.parameter_grid):

                    residual = calculate_residual_wrapper(hp_flat[idx], hc_flat[idx], ecc, l, q, chi1, chi2)
                    residuals_flat[idx] = residual
                    # memory cleanup
                    del hp_flat[idx]
                    del hc_flat[idx]
                    del residual
                    gc.collect()
                    if idx % 50 == 0 and MEMORY_PROFILE:
                        check_memory_usage(f"Loading polarizations progress: {idx}/{n_params}")
                
                print("[DEBUG] Before return:")
                print(f"  train_obj type: {type(train_obj)}")
                print(f"  train_obj.residuals is None: {train_obj.residuals is None}")
                if hasattr(train_obj, 'residuals') and train_obj.residuals is not None:
                    print(f"  train_obj.residuals.shape: {train_obj.residuals.shape}")
                    
                return train_obj
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
                # Residuals dataset
                residuals_flat = np.empty_like(hp_flat)

                # Simulate all polarizations and calculate residuals for every parameter combination in the parameter grid.
                for idx, (ecc, l, q, chi1, chi2) in enumerate(train_obj.parameter_grid):
                    hp, hc, time_array = self.simulate_waveform(
                        time_array=current_time,
                        ecc_ref=ecc,
                        mean_ano_ref=l,
                        mass_ratio=q,
                        chi1=chi1,
                        chi2=chi2,
                        truncate_at_ISCO=truncate_at_ISCO,
                        truncate_at_tmin=truncate_at_tmin,
                        update_results=True,
                        show_truncation_warnings=False
                    )

                    # Update mask: keep only the part of base_time covered by this waveform
                    current_mask = (current_time >= time_array[0]) & (current_time <= time_array[-1])
                    current_time = time_array

                    # idxs of the current time array in the previous time array
                    start_idx = np.where(current_mask)[0][0]
                    end_idx = np.where(current_mask)[0][-1] + 1  # +
                    active_n_t = len(current_time)

                    # Calculate circular waveform for the same parameters, but with ecc=0, to use as reference for the residual calculation
                    residual = calculate_residual_wrapper(hp, hc, ecc, l, q, chi1, chi2)

                    # Adjust lengths of the datasets
                    if active_n_t != hp_flat.shape[1]:
                        hp_flat = hp_flat[:, start_idx:end_idx]
                        hc_flat = hc_flat[:, start_idx:end_idx]
                        residuals_flat = residuals_flat[:, start_idx:end_idx]
                    
                    # Load the simulations into the flattened datasets 
                    hp_flat[idx] = hp
                    hc_flat[idx] = hc
                    residuals_flat[idx] = residual

                    del hp, hc, time_array, residual
                    gc.collect()

                    if idx % 20 == 0 and MEMORY_PROFILE:
                        check_memory_usage(f"Generating residuals: {idx}/{n_params}")


                train_obj.hp_dataset = hp_flat
                train_obj.hc_dataset = hc_flat
                train_obj.residuals = residuals_flat
                print()

                # Update time arrays
                train_obj.time = current_time
                self.time = current_time

                print(self.colored_text(f"All residuals generated in {(time.time() - start)/60:.2f} minutes.", 'green'))

                # Save polarisation and residual datasets to file
                if save_polarizations:
                    train_obj.save_polarizations(prefix="polarisation", directory="Straindata/Polarisations", free_memory=True)
            
            if save_residuals:
                train_obj.save_residuals(prefix="residuals", directory="Straindata/Residuals", free_memory=False)
                # train_obj.save(save_polarizations=save_polarizations, 
                #                 save_residuals=save_residuals, 
                #                 free_memory=False)

        # Plotting functions
        if plot_polarizations:
            self._plot_polarizations(train_obj, save_fig_polarizations=save_fig_polarizations)

        # If plot_residuals is True, plot whole residual dataset
        if (plot_residuals_eccentric_evolve is True) or (plot_residuals_time_evolve is True):
            self._plot_residuals(train_obj=train_obj, 
                                 plot_eccentric_evolve=plot_residuals_eccentric_evolve, save_fig_eccentric_evolve=save_fig_eccentric_evolve, 
                                 plot_time_evolve=plot_residuals_time_evolve, save_fig_time_evolve=save_fig_time_evolve
                                 )
                
        print(self.colored_text(f"Dataset shape: {train_obj.residuals.shape}, N={train_obj.parameter_grid.size} | time_array: [{int(train_obj.time[0])}, {int(train_obj.time[-1])}]", 'green'))
        
        return train_obj
    

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

        # columns:
        # 0 = ecc
        # 1 = mean anomaly
        # 2 = q
        # 3 = chi1
        # 4 = chi2
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

        residuals = np.asarray(train_obj.residuals)   # (N, T)
        time = np.asarray(train_obj.time)
        params = np.asarray(train_obj.parameter_grid)

        n_t = len(time)

        if train_obj.property == "phase":
            ylabel = r"$\Delta \phi_{22}$ [radians]"
        elif train_obj.property == "amplitude":
            ylabel = r"$\Delta A_{22}$"
        else:
            raise ValueError("property must be phase or amplitude")

        COL = {
            "ecc": 0,
            "mean_ano": 1,
            "q": 2,
            "chi1": 3,
            "chi2": 4,
        }

        names = {
            "ecc": "eccentricity",
            "mean_ano": "mean anomaly",
            "q": "mass ratio",
            "chi1": r"$\chi_1$",
            "chi2": r"$\chi_2$",
        }

        symbols = {
            "ecc": "e",
            "mean_ano": "l",
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
                    elif k == "mean_ano":
                        parts.append(f"l={params[base_idx, c]:.4g}")
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
                        time,
                        residuals[global_idx],
                        linewidth=0.9,
                        linestyle="-",
                        label=f"{symbols[effect['key']]} = {params[global_idx, col]:.4g}"
                    )

                ax.set_ylabel(ylabel)
                ax.set_title(
                    f"Varying {names[effect['key']]} over time\n"
                    f"{fixed_text(effect['key'])}",
                    fontsize=10
                )
                ax.grid(True)
                ax.legend()

            axes[-1].set_xlabel("t [M]")

            if save_fig_time_evolve:
                fig.savefig(train_obj.figname(
                    prefix="Residuals_time_evolve",
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

            time_idx = five_indices(n_t)

            if len(parameter_effects) == 1:
                axes = [axes]

            for ax, effect in zip(axes, parameter_effects):

                idx = effect["idx"]
                col = effect["col"]

                for t in time_idx:
                    ax.plot(
                        params[idx, col],
                        residuals[idx, t],
                        linewidth=0.9,
                        linestyle="-",
                        label=f"t/M = {time[t]:.4g}"
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

    # def _plot_residuals(
    #     self,
    #     train_obj: TrainingSetResults,
    #     plot_eccentric_evolve=False,
    #     save_fig_eccentric_evolve=False,
    #     plot_time_evolve=False,
    #     save_fig_time_evolve=False
    # ):
    #     # Check if residuals are loaded, if not attempt to load
    #     train_obj.load_residuals()

    #     residuals = np.asarray(train_obj.residuals)  # (N, T)
    #     time = np.asarray(train_obj.time)

    #     params = np.asarray(train_obj.parameter_grid)  # (N, 5)

    #     ecc_space = train_obj.ecc_ref_space
    #     l_space = train_obj.mean_ano_ref_space
    #     q_space = train_obj.mass_ratio_space
    #     chi1_space = train_obj.chi1_space
    #     chi2_space = train_obj.chi2_space

    #     n_t = len(time)

    #     if train_obj.property == "phase":
    #         ylabel = r"$\Delta \phi_{22}$ [radians]"
    #     elif train_obj.property == "amplitude":
    #         ylabel = r"$\Delta A_{22}$"
    #     else:
    #         raise ValueError("property must be phase or amplitude")

    #     # parameter columns
    #     COL = {
    #         "ecc": 0,
    #         "mean_ano": 1,
    #         "q": 2,
    #         "chi1": 3,
    #         "chi2": 4,
    #     }

    #     param_spaces = {
    #         "ecc": ecc_space,
    #         "mean_ano": l_space,
    #         "q": q_space,
    #         "chi1": chi1_space,
    #         "chi2": chi2_space,
    #     }

    #     names = {
    #         "ecc": "eccentricity",
    #         "mean_ano": "mean anomaly",
    #         "q": "mass ratio",
    #         "chi1": r"$\chi_1$",
    #         "chi2": r"$\chi_2$",
    #     }

    #     symbols = {
    #         "ecc": "e",
    #         "mean_ano": "l",
    #         "q": "q",
    #         "chi1": r"$\chi_1$",
    #         "chi2": r"$\chi_2$",
    #     }

    #     def five_indices(n):
    #         if n <= 5:
    #             return np.arange(n)
    #         return np.linspace(0, n - 1, 5, dtype=int)

    #     # ------------------------------------------------------------
    #     # helper: fixed parameter text (median of remaining params)
    #     # ------------------------------------------------------------
    #     def fixed_text(vary_key, base_idx):
    #         parts = []
    #         for key, col in COL.items():
    #             if key == vary_key:
    #                 continue
    #             v = params[base_idx, col]
    #             parts.append(f"{symbols[key]}={v:.4g}")
    #         return "Fixed: " + ", ".join(parts)

    #     # pick a representative baseline sample
    #     base_idx = len(residuals) // 2

    #     param_effects = []
    #     for key in COL.keys():
    #         col = COL[key]

    #         # filter samples where all OTHER parameters are fixed to baseline
    #         mask = np.ones(len(residuals), dtype=bool)

    #         for k2, c2 in COL.items():
    #             if k2 != key:
    #                 mask &= np.isclose(params[:, c2], params[base_idx, c2])

    #         idx = np.where(mask)[0]

    #         param_effects.append({
    #             "key": key,
    #             "values": params[idx][:, col] if len(idx) > 0 else param_spaces[key],
    #             "indices": idx,
    #             "data": residuals[idx],
    #             "name": names[key],
    #             "symbol": symbols[key],
    #         })

    #     # ------------------------------------------------------------
    #     # TIME EVOLUTION PLOT
    #     # ------------------------------------------------------------
    #     if plot_time_evolve:

    #         fig, axes = plt.subplots(
    #             len(param_effects),
    #             1,
    #             figsize=(11, 3.5 * len(param_effects)),
    #             sharex=True,
    #             gridspec_kw={"hspace": 0.5}
    #         )

    #         for ax, effect in zip(axes, param_effects):

    #             idx = effect["indices"]
    #             values = effect["values"]

    #             selected_idx = five_indices(len(idx))
    #             print(self.colored_text(f"Selected indices for {effect['name']}: {selected_idx}", "green"))

    #             for i in selected_idx:
    #                 ax.plot(
    #                     time,
    #                     residuals[idx[i]],
    #                     label=f"{effect['symbol']}={values[i]:.3g}",
    #                     linewidth=0.9
    #                 )

    #             ax.set_title(f"Varying {effect['name']}\n{fixed_text(effect['key'], idx[0])}")
    #             ax.set_ylabel(ylabel)
    #             ax.grid(True)
    #             ax.legend()

    #         axes[-1].set_xlabel("t [M]")

    #         if save_fig_time_evolve:
    #             fig.savefig(train_obj.figname(
    #                 prefix="Residuals_time_evolve",
    #                 directory="Images/Residuals"
    #             ))

    #     # ------------------------------------------------------------
    #     # PARAMETER EVOLUTION (value vs residual snapshot)
    #     # ------------------------------------------------------------
    #     if plot_eccentric_evolve:

    #         fig, axes = plt.subplots(
    #             len(param_effects),
    #             1,
    #             figsize=(11, 3.5 * len(param_effects)),
    #             gridspec_kw={"hspace": 0.5}
    #         )

    #         time_idx = five_indices(n_t)

    #         for ax, effect in zip(axes, param_effects):

    #             idx = effect["indices"]
    #             values = effect["values"]

    #             for t in time_idx:
    #                 ax.plot(
    #                     values,
    #                     residuals[idx, t],
    #                     label=f"t={time[t]:.3g}"
    #                 )

    #             ax.set_title(f"{effect['name']} dependence")
    #             ax.set_xlabel(effect["symbol"])
    #             ax.set_ylabel(ylabel)
    #             ax.grid(True)
    #             ax.legend()

    #         if save_fig_eccentric_evolve:
    #             fig.savefig(train_obj.figname(
    #                 prefix="Residuals_param_evolve",
    #                 directory="Images/Residuals"
    #             ))

#####################################################################################3

    # def _plot_residuals(self, 
    #                     train_obj: TrainingSetResults, 
    #                     plot_eccentric_evolve=False, save_fig_eccentric_evolve=False,
    #                     plot_time_evolve=False, save_fig_time_evolve=False
    #                     ):
    #     """
    #     Plot residuals from multidimensional residual dataset.

    #     Expected shape:
    #         residuals[ecc, mean_ano, q, chi1, chi2, time]
    #     """

    #     # residuals = np.reshape(train_obj.residuals, (
    #     #     len(train_obj.ecc_ref_space),
    #     #     len(train_obj.mean_ano_ref_space),
    #     #     len(train_obj.mass_ratio_space),
    #     #     len(train_obj.chi1_space),
    #     #     len(train_obj.chi2_space),
    #     #     len(train_obj.time)
    #     # ))

    #     time = train_obj.time

    #     ecc_space = train_obj.ecc_ref_space
    #     l_space = train_obj.mean_ano_ref_space
    #     q_space = train_obj.mass_ratio_space
    #     chi1_space = train_obj.chi1_space
    #     chi2_space = train_obj.chi2_space

    #     n_e = len(ecc_space)
    #     n_l = len(l_space)
    #     n_q = len(q_space)
    #     n_c1 = len(chi1_space)
    #     n_c2 = len(chi2_space)
    #     n_t = len(time)

    #     # Set units for plot labels
    #     if train_obj.property == "phase":
    #         ylabel = r"$\Delta \phi_{22}$ [radians]"
    #     elif train_obj.property == "amplitude":
    #         ylabel = r"$\Delta A_{22}$"
    #     else:
    #         raise ValueError(
    #             f'Choose property = "phase" or "amplitude", got {train_obj.property}'
    #         )

    #     # Create fixed middle-index baseline for each parameter to visualize the effect of varying one parameter at a time while keeping others fixed.
    #     # Plot residuals for every varied parameter in subplot.

    #     # Fixed middle-index baseline for non-varied parameters
    #     e0 = n_e // 2
    #     l0 = n_l // 2
    #     q0 = n_q // 2
    #     c10 = n_c1 // 2
    #     c20 = n_c2 // 2

    #     # For every parameter, select 5 values across the parameter space to vary the effect.
    #     def five_indices(n):
    #         if n <= 5:
    #             return np.arange(n, dtype=int)
    #         return np.linspace(0, n - 1, 5, dtype=int)

    #     # Helper function to generate text for fixed parameters in plot titles
    #     def fixed_text(vary_key):
    #         parts = []

    #         if vary_key != "ecc":
    #             parts.append(f"e={float(ecc_space[e0]):.4g}")
    #         if vary_key != "mean_ano":
    #             parts.append(f"l={float(l_space[l0]):.4g}")
    #         if vary_key != "q":
    #             parts.append(f"q={float(q_space[q0]):.4g}")
    #         if vary_key != "chi1":
    #             parts.append(rf"$\chi_1$={float(chi1_space[c10]):.4g}")
    #         if vary_key != "chi2":
    #             parts.append(rf"$\chi_2$={float(chi2_space[c20]):.4g}")

    #         return "Fixed: " + ", ".join(parts)

    #     parameter_effects = [
    #         {
    #             "key": "ecc",
    #             "name": "eccentricity",
    #             "symbol": "e",
    #             "values": ecc_space,
    #             "data": residuals[:, l0, q0, c10, c20, :],
    #         },
    #         {
    #             "key": "mean_ano",
    #             "name": "mean anomaly",
    #             "symbol": "l",
    #             "values": l_space,
    #             "data": residuals[e0, :, q0, c10, c20, :],
    #         },
    #         {
    #             "key": "q",
    #             "name": "mass ratio",
    #             "symbol": "q",
    #             "values": q_space,
    #             "data": residuals[e0, l0, :, c10, c20, :],
    #         },
    #         {
    #             "key": "chi1",
    #             "name": r"$\chi_1$",
    #             "symbol": r"$\chi_1$",
    #             "values": chi1_space,
    #             "data": residuals[e0, l0, q0, :, c20, :],
    #         },
    #         {
    #             "key": "chi2",
    #             "name": r"$\chi_2$",
    #             "symbol": r"$\chi_2$",
    #             "values": chi2_space,
    #             "data": residuals[e0, l0, q0, c10, :, :],
    #         },
    #     ]

    #     # ------------------------------------------------------------
    #     # Residual vs time, one subplot per varied parameter
    #     # 5 waveforms per parameter
    #     # ------------------------------------------------------------
    #     if plot_time_evolve is True:
    #         fig, axes = plt.subplots(
    #             len(parameter_effects),
    #             1,
    #             figsize=(11, 3.5 * len(parameter_effects)),
    #             sharex=True,
    #             gridspec_kw={"hspace": 0.5}
    #         )

    #         if len(parameter_effects) == 1:
    #             axes = [axes]

    #         for ax, effect in zip(axes, parameter_effects):
    #             values = effect["values"]
    #             data = effect["data"]

    #             selected_indices = five_indices(len(values))

    #             for i in selected_indices:
    #                 value = values[i]

    #                 ax.plot(
    #                     time,
    #                     data[i],
    #                     linewidth=0.9,
    #                     linestyle="-",
    #                     label=f"{effect['symbol']} = {float(value):.4g}"
    #                 )

    #             ax.set_ylabel(ylabel)
    #             ax.set_title(
    #                 f"Varying {effect['name']} over time\n"
    #                 f"{fixed_text(effect['key'])}",
    #                 fontsize=10
    #             )
    #             ax.grid(True)

    #             ax.legend(
    #                 title=f"Varying {effect['symbol']}",
    #                 fontsize="small",
    #                 title_fontsize="small",
    #                 loc="best",
    #                 ncol=1
    #             )

    #         axes[-1].set_xlabel("t [M]")

    #         fig.suptitle(
    #             f"Residual {train_obj.property}: effect of each parameter",
    #             y=1.005
    #         )

    #         plt.tight_layout()

    #         if save_fig_time_evolve is True:
    #             figname = train_obj.figname(
    #                 prefix="Residuals_time_evolve",
    #                 directory="Images/Residuals"
    #             )
    #             fig.savefig(figname)

    #     # ------------------------------------------------------------
    #     # Residual vs parameter value, one subplot per varied parameter
    #     # 5 selected times
    #     # ------------------------------------------------------------
    #     if plot_eccentric_evolve is True:
    #         fig, axes = plt.subplots(
    #             len(parameter_effects),
    #             1,
    #             figsize=(11, 3.5 * len(parameter_effects)),
    #             sharex=False,
    #             gridspec_kw={"hspace": 0.6}
    #         )

    #         if len(parameter_effects) == 1:
    #             axes = [axes]

    #         time_indices = five_indices(n_t)

    #         for i, (ax, effect) in enumerate(zip(axes, parameter_effects)):
    #             values = effect["values"]
    #             data = effect["data"]

    #             for tidx in time_indices:
    #                 ax.plot(
    #                     values,
    #                     data[:, tidx],
    #                     linewidth=0.9,
    #                     linestyle="-",
    #                     label=f"t/M = {float(time[tidx]):.4g}"
    #                 )

    #             ax.set_xlabel(effect["symbol"])
    #             ax.set_ylabel(ylabel)
    #             ax.set_title(
    #                 f"Residual change while varying {effect['name']}: "
    #                 f"{fixed_text(effect['key'])}",
    #                 fontsize=10
    #             )
    #             ax.grid(True)
    #             if i == 0:
    #                 ax.legend(
    #                     title="Selected times",
    #                     fontsize="small",
    #                     title_fontsize="small",
    #                     loc="best",
    #                     ncol=1
    #                 )

    #         fig.suptitle(
    #             f"Residual {train_obj.property}: parameter dependence at selected times",
    #             y=1.005
    #         )

    #         plt.tight_layout()

    #         if save_fig_eccentric_evolve is True:
    #             figname = train_obj.figname(
    #                 prefix="Residuals_eccentric_evolve",
    #                 directory="Images/Residuals"
    #             )
    #             fig.savefig(figname)


    def _save_residual_dataset(self, train_obj:TrainingSetResults):
        """Function to save residual dataset to file."""

        ecc_list = train_obj.ecc_ref_space
        residual_dataset = train_obj.residuals

        os.makedirs('Straindata/Residuals', exist_ok=True)
        # train_obj.filename(prefix=f"residuals", directory="Straindata/Residuals")
        file_path = f'Straindata/Residuals/residuals_{train_obj.property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(ecc_list)}_{max(ecc_list)}_N={len(ecc_list)}].npz'
        np.savez(file_path, residual=residual_dataset, time=self.time, eccentricities=ecc_list)
        print('Residuals saved to Straindata/Residuals')


  

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
            # print(0, train_obj.basis_indices, train_obj.leaf_basis_indices, train_obj.orthonormal_basis)
            
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


    def _get_regression_parameter_matrix(self):
        """
        Build parameter matrix X with one row per waveform.
        For your current class this is just eccentricity.
        """
        X = np.asarray(self.ecc_ref_space, dtype=float).reshape(-1, 1)
        return X

    def test_GPR_quality(
        self,
        U,
        property,
        min_greedy_error=None,
        N_greedy_vecs=None,
        normalize=True,
        max_tree_depth=1,
        n_splits=5,
        max_nodes=None,
        random_state=0,
        verbose=True,
        make_plots=True,
        save_figs=False,
        fig_dir="flow_diagnostic_figures",
    ):
        U = np.asarray(U, dtype=float)
        X = self._get_regression_parameter_matrix()

        if U.shape[0] != X.shape[0]:
            raise ValueError(
                f"Mismatch: U has {U.shape[0]} rows but parameter matrix has {X.shape[0]} rows."
            )

        if verbose:
            print(f"U shape = {U.shape}")
            print(f"X shape = {X.shape}")
            print(f"Running diagnostic for property = {property}")

        rb_obj = self.get_greedy_parameters(
            U=U,
            property=property,
            min_greedy_error=min_greedy_error,
            N_greedy_vecs=N_greedy_vecs,
            normalize=normalize,
            max_tree_depth=max_tree_depth,
            plot_greedy_error=False,
            save_greedy_error_fig=False,
            plot_greedy_vectors=False,
            save_greedy_vecs_fig=False,
            plot_SVD_matrix=False,
            save_SVD_matrix_fig=False,
            show_legend=False,
        )

        emp_nodes_idx = self.get_empirical_nodes(
            reduced_basis_object=rb_obj,
            property=property,
            plot_emp_nodes_at_ecc=False,
            save_fig=False,
            plot_interpolation_matrix=False,
            save_interpolation_matrix_fig=False,
            plot_proj_vs_eim_error=False,
            save_proj_vs_eim_error_fig=False,
        )

        emp_nodes_idx = np.unique(np.asarray(emp_nodes_idx, dtype=int))

        if max_nodes is not None:
            emp_nodes_idx = emp_nodes_idx[:max_nodes]

        if len(emp_nodes_idx) == 0:
            raise ValueError("No empirical nodes returned.")

        self.empirical_nodes_idx = emp_nodes_idx

        if verbose:
            print(f"Using {len(emp_nodes_idx)} empirical nodes")

        Y_nodes = U[:, emp_nodes_idx]

        n_samples, n_nodes = Y_nodes.shape
        n_params = X.shape[1]

        pred_mean = np.zeros_like(Y_nodes)
        pred_std = np.zeros_like(Y_nodes)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
            if verbose:
                print(f"Fold {fold}/{n_splits}")

            X_train, X_test = X[train_idx], X[test_idx]
            Y_train = Y_nodes[train_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            for j in range(n_nodes):
                gp = self._build_default_gp(n_params)
                gp.fit(X_train_s, Y_train[:, j])

                mu, std = gp.predict(X_test_s, return_std=True)
                pred_mean[test_idx, j] = mu
                pred_std[test_idx, j] = np.maximum(std, 1e-12)

        resid = Y_nodes - pred_mean
        z = resid / pred_std

        # Identify worst time nodes
        worst_node_indices, worst_node_diagnostics = self._get_worst_node_indices(
            z_residuals=z,
            pred_mean=pred_mean,
            Y_nodes=Y_nodes,
            top_k=min(5, Y_nodes.shape[1]),
        )

        # Identify hardest eccentric points
        hardest_points = self._get_hardest_parameter_points(
            X=X,
            residuals=resid,
            z_residuals=z,
            top_k=min(10, X.shape[0]),
        )

        skew_vals = skew(z, axis=0, bias=False, nan_policy="omit")
        kurt_vals = kurtosis(z, axis=0, fisher=True, bias=False, nan_policy="omit")

        pvals = []
        for j in range(n_nodes):
            try:
                _, p = normaltest(z[:, j], nan_policy="omit")
            except Exception:
                p = np.nan
            pvals.append(p)
        pvals = np.array(pvals)

        if n_nodes > 1:
            corr = np.corrcoef(z, rowvar=False)
            offdiag = corr[~np.eye(n_nodes, dtype=bool)]
            mean_abs_offdiag_corr = float(np.nanmean(np.abs(offdiag)))
            max_abs_offdiag_corr = float(np.nanmax(np.abs(offdiag)))
        else:
            mean_abs_offdiag_corr = 0.0
            max_abs_offdiag_corr = 0.0

        coverage_1sigma = float(np.mean(np.abs(z) <= 1.0))
        coverage_2sigma = float(np.mean(np.abs(z) <= 2.0))

        mean_abs_skew = float(np.nanmean(np.abs(skew_vals)))
        mean_abs_excess_kurtosis = float(np.nanmean(np.abs(kurt_vals)))
        frac_non_normal_nodes = float(np.nanmean(pvals < 0.05))

        reasons = []
        if mean_abs_skew > 0.5:
            reasons.append("residual skewness")
        if mean_abs_excess_kurtosis > 1.0:
            reasons.append("heavy tails")
        if frac_non_normal_nodes > 0.3:
            reasons.append("many nodes fail normality")
        if mean_abs_offdiag_corr > 0.1:
            reasons.append("correlated node residuals")
        if not (0.63 <= coverage_1sigma <= 0.73):
            reasons.append("1σ coverage is off")
        if not (0.92 <= coverage_2sigma <= 0.98):
            reasons.append("2σ coverage is off")

        if reasons:
            recommendation = (
                "A conditional flow is worth testing because of: "
                + ", ".join(reasons) + "."
            )
        else:
            recommendation = (
                "A flow probably will not help much: GP residuals already look fairly Gaussian and weakly correlated."
            )

        result = FlowDiagnosticResult(
            mean_abs_skew=mean_abs_skew,
            mean_abs_excess_kurtosis=mean_abs_excess_kurtosis,
            frac_non_normal_nodes=frac_non_normal_nodes,
            mean_abs_offdiag_corr=mean_abs_offdiag_corr,
            max_abs_offdiag_corr=max_abs_offdiag_corr,
            coverage_1sigma=coverage_1sigma,
            coverage_2sigma=coverage_2sigma,
            recommendation=recommendation,
        )

        if verbose:
            print("\n=== Flow diagnostic summary ===")
            print(f"mean |skew|               : {result.mean_abs_skew:.4f}")
            print(f"mean |excess kurtosis|    : {result.mean_abs_excess_kurtosis:.4f}")
            print(f"frac nodes p<0.05         : {result.frac_non_normal_nodes:.4f}")
            print(f"mean |offdiag corr|       : {result.mean_abs_offdiag_corr:.4f}")
            print(f"max  |offdiag corr|       : {result.max_abs_offdiag_corr:.4f}")
            print(f"coverage |z|<=1           : {result.coverage_1sigma:.4f}")
            print(f"coverage |z|<=2           : {result.coverage_2sigma:.4f}")
            print(f"\nRecommendation: {result.recommendation}")

            if make_plots:
                if verbose:
                    print(f"Worst node indices: {worst_node_indices}")

                self._plot_worst_node_summary(
                    node_indices=worst_node_indices,
                    diagnostics=worst_node_diagnostics,
                    property=property,
                    save_fig=save_figs,
                    fig_dir=fig_dir,
                )

                self._plot_gp_fits_for_nodes(
                    X=X,
                    Y_nodes=Y_nodes,
                    pred_mean=pred_mean,
                    pred_std=pred_std,
                    node_indices=worst_node_indices,
                    property=property,
                    save_fig=save_figs,
                    fig_dir=fig_dir,
                )

                self._plot_true_vs_pred_for_nodes(
                    Y_nodes=Y_nodes,
                    pred_mean=pred_mean,
                    pred_std=pred_std,
                    node_indices=worst_node_indices,
                    property=property,
                    save_fig=save_figs,
                    fig_dir=fig_dir,
                )

                self._plot_residual_histograms_for_nodes(
                    z_residuals=z,
                    node_indices=worst_node_indices,
                    property=property,
                    save_fig=save_figs,
                    fig_dir=fig_dir,
                )

                self._plot_residual_qq_for_nodes(
                    z_residuals=z,
                    node_indices=worst_node_indices,
                    property=property,
                    save_fig=save_figs,
                    fig_dir=fig_dir,
                )

                self._plot_residual_corr_heatmap(
                    z_residuals=z,
                    property=property,
                    save_fig=save_figs,
                    fig_dir=fig_dir,
                )

                self._plot_coverage_curve(
                    z_residuals=z,
                    property=property,
                    save_fig=save_figs,
                    fig_dir=fig_dir,
                )

                self._plot_skew_kurtosis_scatter(
                    z_residuals=z,
                    property=property,
                    save_fig=save_figs,
                    fig_dir=fig_dir,
                )

                self._plot_residuals_vs_eccentricity_for_nodes(
                    X=X,
                    residuals=resid,
                    node_indices=worst_node_indices,
                    property=property,
                    standardized=False,
                    save_fig=save_figs,
                    fig_dir=fig_dir,
                )

                self._plot_residuals_vs_eccentricity_for_nodes(
                    X=X,
                    residuals=z,
                    node_indices=worst_node_indices,
                    property=property,
                    standardized=True,
                    save_fig=save_figs,
                    fig_dir=fig_dir,
                )

                self._plot_difficulty_vs_eccentricity(
                    X=X,
                    residuals=resid,
                    z_residuals=z,
                    property=property,
                    save_fig=save_figs,
                    fig_dir=fig_dir,
                )

                self._plot_residual_heatmap_vs_eccentricity(
                    X=X,
                    residuals=resid,
                    property=property,
                    standardized=False,
                    save_fig=save_figs,
                    fig_dir=fig_dir,
                )

                self._plot_residual_heatmap_vs_eccentricity(
                    X=X,
                    residuals=z,
                    property=property,
                    standardized=True,
                    save_fig=save_figs,
                    fig_dir=fig_dir,
                )

                self._plot_gp_fits_with_hard_points(
                    X=X,
                    Y_nodes=Y_nodes,
                    pred_mean=pred_mean,
                    pred_std=pred_std,
                    z_residuals=z,
                    node_indices=worst_node_indices,
                    property=property,
                    n_hard_points=3,
                    save_fig=save_figs,
                    fig_dir=fig_dir,
                )

        return {
            "rb_object": rb_obj,
            "emp_nodes_idx": emp_nodes_idx,
            "Y_nodes": Y_nodes,
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "residuals": resid,
            "z_residuals": z,
            "result": result,
            "worst_node_indices": worst_node_indices,
            "worst_node_diagnostics": worst_node_diagnostics,
            "hardest_points": hardest_points,
        }

    
    def _plot_residual_histograms_for_nodes(
        self,
        z_residuals,
        node_indices,
        property="amplitude",
        save_fig=False,
        fig_dir="Images/flow_diagnostic_figures",
    ):

        n_show = len(node_indices)
        fig, axes = plt.subplots(n_show, 1, figsize=(7, 4 * n_show), squeeze=False)

        xgrid = np.linspace(-5, 5, 400)

        for ax, node_idx in zip(axes[:, 0], node_indices):
            z = z_residuals[:, node_idx]
            z = z[np.isfinite(z)]

            ax.hist(z, bins=30, density=True, alpha=0.7, label="Standardized residuals")
            ax.plot(xgrid, norm.pdf(xgrid), linewidth=2, label="Standard normal")
            ax.set_xlabel("Standardized residual")
            ax.set_ylabel("Density")
            ax.set_title(f"{property.capitalize()} residual histogram at node {node_idx}")
            ax.legend()

        plt.tight_layout()

        if save_fig:
            os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(
                os.path.join(fig_dir, f"{property}_residual_histograms_worst_nodes.png"),
                dpi=200,
                bbox_inches="tight",
            )



    def _plot_gp_fits_for_nodes(
        self,
        X,
        Y_nodes,
        pred_mean,
        pred_std,
        node_indices,
        property="amplitude",
        save_fig=False,
        fig_dir="Images/flow_diagnostic_figures",
    ):


        x = X[:, 0]
        order = np.argsort(x)
        x_sorted = x[order]

        n_show = len(node_indices)
        fig, axes = plt.subplots(n_show, 1, figsize=(8, 4 * n_show), squeeze=False)

        for ax, node_idx in zip(axes[:, 0], node_indices):
            y_true = Y_nodes[order, node_idx]
            y_pred = pred_mean[order, node_idx]
            y_std = pred_std[order, node_idx]

            ax.plot(x_sorted, y_true, "o", markersize=4, alpha=0.8, label="True")
            ax.plot(x_sorted, y_pred, "-", linewidth=2, label="GP mean")
            ax.fill_between(
                x_sorted,
                y_pred - 2.0 * y_std,
                y_pred + 2.0 * y_std,
                alpha=0.25,
                label=r"GP $\pm 2\sigma$",
            )
            ax.set_xlabel("Eccentricity")
            ax.set_ylabel(f"{property} value")
            ax.set_title(f"{property.capitalize()} GP fit at EIM node {node_idx}")
            ax.legend()

        plt.tight_layout()

        if save_fig:
            os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(
                os.path.join(fig_dir, f"{property}_gp_fits_worst_nodes.png"),
                dpi=200,
                bbox_inches="tight",
            )

        # plt.show()

    def _plot_residual_corr_heatmap(
        self,
        z_residuals,
        property="amplitude",
        save_fig=False,
        fig_dir="Images/flow_diagnostic_figures",
    ):
        corr = np.corrcoef(z_residuals, rowvar=False)

        plt.figure(figsize=(7, 6))
        plt.imshow(corr, aspect="auto")
        plt.colorbar(label="Correlation")
        plt.xlabel("Node index")
        plt.ylabel("Node index")
        plt.title(f"{property.capitalize()} residual correlation across EIM nodes")
        plt.tight_layout()

        if save_fig:
            os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(os.path.join(fig_dir, f"{property}_residual_corr_heatmap.png"), dpi=200)

        # plt.show()

    def _plot_residual_qq_for_nodes(
        self,
        z_residuals,
        node_indices,
        property="amplitude",
        save_fig=False,
        fig_dir="flow_diagnostic_figures",
    ):


        n_show = len(node_indices)
        fig, axes = plt.subplots(n_show, 1, figsize=(6, 5 * n_show), squeeze=False)

        for ax, node_idx in zip(axes[:, 0], node_indices):
            z = z_residuals[:, node_idx]
            z = z[np.isfinite(z)]
            z_sorted = np.sort(z)

            n = len(z_sorted)
            probs = (np.arange(1, n + 1) - 0.5) / n
            gaussian_q = norm.ppf(probs)

            zmin = min(gaussian_q.min(), z_sorted.min())
            zmax = max(gaussian_q.max(), z_sorted.max())

            ax.plot(gaussian_q, z_sorted, "o", markersize=4)
            ax.plot([zmin, zmax], [zmin, zmax], "--", linewidth=2)
            ax.set_xlabel("Gaussian quantiles")
            ax.set_ylabel("Residual quantiles")
            ax.set_title(f"{property.capitalize()} Q-Q plot at node {node_idx}")

        plt.tight_layout()

        if save_fig:
            os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(
                os.path.join(fig_dir, f"{property}_qq_worst_nodes.png"),
                dpi=200,
                bbox_inches="tight",
            )

        # plt.show()

    def _plot_coverage_curve(
        self,
        z_residuals,
        property="amplitude",
        save_fig=False,
        fig_dir="Images/flow_diagnostic_figures",
    ):
        thresholds = np.linspace(0.1, 3.0, 60)
        empirical = [np.mean(np.abs(z_residuals) <= t) for t in thresholds]
        ideal = [norm.cdf(t) - norm.cdf(-t) for t in thresholds]

        plt.figure(figsize=(7, 5))
        plt.plot(thresholds, empirical, linewidth=2, label="Empirical")
        plt.plot(thresholds, ideal, "--", linewidth=2, label="Gaussian ideal")
        plt.xlabel(r"Threshold $k$ in $|z| \leq k$")
        plt.ylabel("Coverage probability")
        plt.title(f"{property.capitalize()} coverage curve")
        plt.legend()
        plt.tight_layout()

        if save_fig:
            os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(os.path.join(fig_dir, f"{property}_coverage_curve.png"), dpi=200)

        # plt.show()

    def _plot_skew_kurtosis_scatter(
        self,
        z_residuals,
        property="amplitude",
        save_fig=False,
        fig_dir="Images/flow_diagnostic_figures",
    ):
        skew_vals = skew(z_residuals, axis=0, bias=False, nan_policy="omit")
        kurt_vals = kurtosis(z_residuals, axis=0, fisher=True, bias=False, nan_policy="omit")

        plt.figure(figsize=(7, 5))
        plt.plot(skew_vals, kurt_vals, "o")
        plt.axvline(0.0, linestyle="--")
        plt.axhline(0.0, linestyle="--")
        plt.xlabel("Skewness")
        plt.ylabel("Excess kurtosis")
        plt.title(f"{property.capitalize()} node-by-node Gaussianity")
        plt.tight_layout()

        if save_fig:
            os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(os.path.join(fig_dir, f"{property}_skew_kurtosis_scatter.png"), dpi=200)

        # plt.show()

    def _plot_residual_corr_heatmap(
        self,
        z_residuals,
        property="amplitude",
        save_fig=False,
        fig_dir="Images/flow_diagnostic_figures",
    ):
        corr = np.corrcoef(z_residuals, rowvar=False)

        plt.figure(figsize=(7, 6))
        plt.imshow(corr, aspect="auto")
        plt.colorbar(label="Correlation")
        plt.xlabel("Node index")
        plt.ylabel("Node index")
        plt.title(f"{property.capitalize()} residual correlation across EIM nodes")
        plt.tight_layout()

        if save_fig:
            os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(os.path.join(fig_dir, f"{property}_residual_corr_heatmap.png"), dpi=200)

        # plt.show()

    def _plot_true_vs_pred_for_nodes(
        self,
        Y_nodes,
        pred_mean,
        pred_std,
        node_indices,
        property="amplitude",
        save_fig=False,
        fig_dir="Images/flow_diagnostic_figures",
    ):


        n_show = len(node_indices)
        fig, axes = plt.subplots(n_show, 1, figsize=(6, 5 * n_show), squeeze=False)

        for ax, node_idx in zip(axes[:, 0], node_indices):
            y_true = Y_nodes[:, node_idx]
            y_pred = pred_mean[:, node_idx]
            y_std = pred_std[:, node_idx]

            ax.errorbar(
                y_true,
                y_pred,
                yerr=2.0 * y_std,
                fmt="o",
                markersize=4,
                alpha=0.7,
                capsize=0,
                label=r"Prediction with $2\sigma$",
            )

            lo = min(np.min(y_true), np.min(y_pred))
            hi = max(np.max(y_true), np.max(y_pred))
            ax.plot([lo, hi], [lo, hi], "--", linewidth=2, label="Ideal")

            ax.set_xlabel("True node value")
            ax.set_ylabel("Predicted node value")
            ax.set_title(f"{property.capitalize()} true vs GP prediction at node {node_idx}")
            ax.legend()

        plt.tight_layout()

        if save_fig:
            os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(
                os.path.join(fig_dir, f"{property}_true_vs_pred_worst_nodes.png"),
                dpi=200,
                bbox_inches="tight",
            )

    def _plot_worst_node_summary(
        self,
        node_indices,
        diagnostics,
        property="amplitude",
        save_fig=False,
        fig_dir="Images/flow_diagnostic_figures",
    ):


        skew_vals = diagnostics["skew"][node_indices]
        kurt_vals = diagnostics["kurtosis"][node_indices]
        rmse_vals = diagnostics["rmse"][node_indices]
        pvals = diagnostics["pvals"][node_indices]

        labels = [str(i) for i in node_indices]
        x = np.arange(len(node_indices))
        width = 0.2

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - 1.5 * width, np.abs(skew_vals), width, label="|skew|")
        ax.bar(x - 0.5 * width, np.abs(kurt_vals), width, label="|excess kurtosis|")
        ax.bar(x + 0.5 * width, rmse_vals, width, label="RMSE")
        ax.bar(x + 1.5 * width, -np.log10(np.clip(pvals, 1e-300, 1.0)), width, label="-log10(p)")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Worst node index")
        ax.set_ylabel("Diagnostic value")
        ax.set_title(f"{property.capitalize()} worst-node diagnostics")
        ax.legend()
        plt.tight_layout()

        if save_fig:

            skew_vals = diagnostics["skew"]
            os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(
                os.path.join(fig_dir, f"{property}_worst_node_summary.png"),
                dpi=200,
                bbox_inches="tight",
            )

    def _get_worst_node_indices(
        self,
        z_residuals,
        pred_mean,
        Y_nodes,
        top_k=5,
    ):
        """
        Rank nodes by how badly Gaussian/GP assumptions fail.

        Score combines:
        - absolute skewness
        - absolute excess kurtosis
        - normality failure strength
        - RMSE

        Returns
        -------
        worst_idx : ndarray
            Indices of the worst nodes, worst first.
        diagnostics : dict
            Per-node diagnostic arrays.
        """
        from scipy.stats import skew, kurtosis, normaltest

        z = np.asarray(z_residuals)
        resid = Y_nodes - pred_mean

        n_nodes = z.shape[1]

        skew_vals = skew(z, axis=0, bias=False, nan_policy="omit")
        kurt_vals = kurtosis(z, axis=0, fisher=True, bias=False, nan_policy="omit")

        pvals = np.full(n_nodes, np.nan)
        for j in range(n_nodes):
            try:
                _, pvals[j] = normaltest(z[:, j], nan_policy="omit")
            except Exception:
                pvals[j] = np.nan

        rmse_vals = np.sqrt(np.mean(resid**2, axis=0))

        # Turn p-values into a positive “failure” score
        # smaller p => larger score
        pscore = -np.log10(np.clip(pvals, 1e-300, 1.0))

        # Normalize each component so one term does not dominate only by scale
        def safe_norm(a):
            a = np.asarray(a, dtype=float)
            s = np.nanstd(a)
            if not np.isfinite(s) or s == 0:
                return np.zeros_like(a)
            return (a - np.nanmean(a)) / s

        score = (
            safe_norm(np.abs(skew_vals))
            + safe_norm(np.abs(kurt_vals))
            + safe_norm(pscore)
            + 0.5 * safe_norm(rmse_vals)
        )

        worst_idx = np.argsort(score)[::-1][:top_k]

        diagnostics = {
            "score": score,
            "skew": skew_vals,
            "kurtosis": kurt_vals,
            "pvals": pvals,
            "rmse": rmse_vals,
        }

        return worst_idx, diagnostics

    def _plot_residuals_vs_eccentricity_for_nodes(
        self,
        X,
        residuals,
        node_indices,
        property="amplitude",
        standardized=False,
        save_fig=False,
        fig_dir="Images/flow_diagnostic_figures",
    ):

        x = X[:, 0]
        order = np.argsort(x)
        x_sorted = x[order]

        n_show = len(node_indices)
        fig, axes = plt.subplots(n_show, 1, figsize=(8, 3.8 * n_show), squeeze=False)

        for ax, node_idx in zip(axes[:, 0], node_indices):
            r = residuals[order, node_idx]

            ax.plot(x_sorted, r, "o-", markersize=4, alpha=0.85)
            ax.axhline(0.0, linestyle="--", linewidth=1.5)
            ax.set_xlabel("Eccentricity")
            if standardized:
                ax.set_ylabel("Standardized residual")
                ax.set_title(f"{property.capitalize()} standardized residual vs eccentricity (node {node_idx}, t={self.time[node_idx]:.2f}M)")
            else:
                ax.set_ylabel("Residual")
                ax.set_title(f"{property.capitalize()} residual vs eccentricity (node {node_idx}, t={self.time[node_idx]:.2f}M)")

        plt.tight_layout()

        if save_fig:
            os.makedirs(fig_dir, exist_ok=True)
            name = f"{property}_{'z_' if standardized else ''}residuals_vs_ecc_worst_nodes.png"
            plt.savefig(os.path.join(fig_dir, name), dpi=200, bbox_inches="tight")

        # plt.show()

    def _plot_difficulty_vs_eccentricity(
        self,
        X,
        residuals,
        z_residuals,
        property="amplitude",
        save_fig=False,
        fig_dir="Images/flow_diagnostic_figures",
    ):

        x = X[:, 0]
        order = np.argsort(x)
        x_sorted = x[order]

        resid_sorted = residuals[order]
        z_sorted = z_residuals[order]

        rms_resid = np.sqrt(np.mean(resid_sorted**2, axis=1))
        max_abs_resid = np.max(np.abs(resid_sorted), axis=1)

        rms_z = np.sqrt(np.mean(z_sorted**2, axis=1))
        max_abs_z = np.max(np.abs(z_sorted), axis=1)

        fig, axes = plt.subplots(2, 1, figsize=(8, 8), squeeze=False)

        axes[0, 0].plot(x_sorted, rms_resid, "o-", markersize=4, label="RMS residual across nodes")
        axes[0, 0].plot(x_sorted, max_abs_resid, "o-", markersize=4, label="Max |residual| across nodes")
        axes[0, 0].set_xlabel("Eccentricity")
        axes[0, 0].set_ylabel("Residual scale")
        axes[0, 0].set_title(f"{property.capitalize()} difficulty vs eccentricity")
        axes[0, 0].legend()

        axes[1, 0].plot(x_sorted, rms_z, "o-", markersize=4, label="RMS standardized residual across nodes")
        axes[1, 0].plot(x_sorted, max_abs_z, "o-", markersize=4, label="Max |standardized residual| across nodes")
        axes[1, 0].axhline(1.0, linestyle="--", linewidth=1.5)
        axes[1, 0].axhline(2.0, linestyle="--", linewidth=1.5)
        axes[1, 0].set_xlabel("Eccentricity")
        axes[1, 0].set_ylabel("Standardized residual scale")
        axes[1, 0].set_title(f"{property.capitalize()} GP surprise vs eccentricity")
        axes[1, 0].legend()

        plt.tight_layout()

        if save_fig:
            os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(
                os.path.join(fig_dir, f"{property}_difficulty_vs_eccentricity.png"),
                dpi=200,
                bbox_inches="tight",
            )

        # plt.show()

    def _plot_residual_heatmap_vs_eccentricity(
        self,
        X,
        residuals,
        property="amplitude",
        standardized=False,
        save_fig=False,
        fig_dir="Images/flow_diagnostic_figures",
    ):

        x = X[:, 0]
        order = np.argsort(x)
        x_sorted = x[order]
        R = residuals[order].T  # shape (n_nodes, n_samples)

        plt.figure(figsize=(10, 6))
        extent = [x_sorted.min(), x_sorted.max(), 0, R.shape[0] - 1]
        plt.imshow(R, aspect="auto", origin="lower", extent=extent)
        plt.colorbar(label="Standardized residual" if standardized else "Residual")
        plt.xlabel("Eccentricity")
        plt.ylabel("Node index")
        plt.title(
            f"{property.capitalize()} {'standardized ' if standardized else ''}residual heatmap vs eccentricity"
        )
        plt.tight_layout()

        if save_fig:
            os.makedirs(fig_dir, exist_ok=True)
            name = f"{property}_{'z_' if standardized else ''}residual_heatmap_vs_ecc.png"
            plt.savefig(os.path.join(fig_dir, name), dpi=200, bbox_inches="tight")

        # plt.show()

    def _plot_gp_fits_with_hard_points(
        self,
        X,
        Y_nodes,
        pred_mean,
        pred_std,
        z_residuals,
        node_indices,
        property="amplitude",
        n_hard_points=3,
        save_fig=False,
        fig_dir="Images/flow_diagnostic_figures",
    ):

        x = X[:, 0]
        order = np.argsort(x)
        x_sorted = x[order]

        n_show = len(node_indices)
        fig, axes = plt.subplots(n_show, 1, figsize=(8, 4 * n_show), squeeze=False)

        for ax, node_idx in zip(axes[:, 0], node_indices):
            y_true = Y_nodes[order, node_idx]
            y_pred = pred_mean[order, node_idx]
            y_std = pred_std[order, node_idx]
            z = z_residuals[order, node_idx]

            hard_local = np.argsort(np.abs(z))[-n_hard_points:]

            ax.plot(x_sorted, y_true, "o", markersize=4, alpha=0.8, label="True")
            ax.plot(x_sorted, y_pred, "-", linewidth=2, label="GP mean")
            ax.fill_between(
                x_sorted,
                y_pred - 2.0 * y_std,
                y_pred + 2.0 * y_std,
                alpha=0.25,
                label=r"GP $\pm 2\sigma$",
            )

            ax.plot(
                x_sorted[hard_local],
                y_true[hard_local],
                "o",
                markersize=8,
                markerfacecolor="none",
                markeredgewidth=2,
                label=f"{n_hard_points} hardest points",
            )

            for idx in hard_local:
                ax.annotate(f"e={x_sorted[idx]:.3f}", (x_sorted[idx], y_true[idx]), fontsize=8)

            ax.set_xlabel("Eccentricity")
            ax.set_ylabel(f"{property} value")
            ax.set_title(f"{property.capitalize()} GP fit with hardest points (node {node_idx}, t={self.time[node_idx]:.2f}M)")
            ax.legend()

        plt.tight_layout()

        if save_fig:
            os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(
                os.path.join(fig_dir, f"{property}_gp_fits_with_hard_points.png"),
                dpi=200,
                bbox_inches="tight",
            )

        # plt.show()

    def _get_hardest_parameter_points(
        self,
        X,
        residuals,
        z_residuals,
        top_k=10,
    ):

        rms_resid = np.sqrt(np.mean(residuals**2, axis=1))
        max_abs_resid = np.max(np.abs(residuals), axis=1)

        rms_z = np.sqrt(np.mean(z_residuals**2, axis=1))
        max_abs_z = np.max(np.abs(z_residuals), axis=1)

        score = rms_z + 0.5 * max_abs_z
        hard_idx = np.argsort(score)[::-1][:top_k]

        return {
            "indices": hard_idx,
            "eccentricities": X[hard_idx, 0],
            "rms_residual": rms_resid[hard_idx],
            "max_abs_residual": max_abs_resid[hard_idx],
            "rms_z": rms_z[hard_idx],
            "max_abs_z": max_abs_z[hard_idx],
        }
    

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

# # Sampling parameters
# sampling_frequency = 2048 # or 4096
# duration = 4 # seconds
# time_array = np.linspace(-duration, 0, int(sampling_frequency * duration))  # time in seconds

# gt = Generate_TrainingSet(time_array=time_array,
#                           ecc_ref_parameterspace=np.linspace(0.001, 0.3, num=2),
#                           mean_ano_parameterspace=np.linspace(0, 2*np.pi, num=2),
#                           mass_ratio_parameterspace=np.linspace(1, 20, num=2),
#                           chi1_parameterspace=np.linspace(-0.9, 0.9, num=2),
#                           chi2_parameterspace=np.linspace(-0.9, 0.9, num=2),
#                           min_greedy_error_amp=1e-6,
#                           min_greedy_error_phase=1e-6,
#                           truncate_at_tmin=True,
#                           truncate_at_ISCO=True,
#                           f_lower=10)

# tr_obj_p = gt._get_training_obj('phase')
# tr_obj_a = gt._get_training_obj('amplitude')

# gt._calculate_residuals(tr_obj_p,
#                         plot_polarizations=True,
#                         plot_residuals_time_evolve=True,
#                         plot_residuals_eccentric_evolve=True
#                         )
# print(tr_obj_p.residuals.shape)
# plt.show()

# gt._calculate_residuals(tr_obj_a,
# plot_polarizations=True,
# plot_residuals_time_evolve=True,
# plot_residuals_eccentric_evolve=True
# )

# # plt.show()

# gt.get_greedy_parameters(tr_obj_p,
#                          plot_greedy_error=True,
#                          plot_basis_indices=True,
#                          )

# gt.get_greedy_parameters(tr_obj_a,
# plot_greedy_error=True,
# plot_basis_indices=True,
# )

# gt.get_training_set_greedy(property="phase", 
#     plot_interpolation_matrix=True, save_fig_interpolation_matrix=True,
#     plot_proj_vs_eim_error=True, save_fig_proj_vs_eim_error=True,
#     plot_residuals_time=True, save_fig_residuals_time=True,
#     plot_emp_nodes_on_basis=True, save_fig_emp_nodes_on_basis=True,
#     plot_training_set=True, save_fig_training_set=True,
#     plot_residuals_eccentric=True, save_fig_residuals_eccentric=True,
#     plot_greedy_error=True, save_fig_greedy_error=True,
#     plot_basis_indices=True, save_fig_basis_indices=True,
#     )

############################################333
# Sampling parameters
# sampling_frequency = 2048 # or 4096
# duration = 4 # seconds
# time_array = np.linspace(-duration, 0, int(sampling_frequency * duration))  # time in seconds

# gt = Generate_TrainingSet(
#                           time_array=time_array,
#                           ecc_ref_parameterspace=np.linspace(0.001, 0.3, num=5),
#                           mean_ano_parameterspace=np.linspace(0, 2*np.pi, num=5),
#                           mass_ratio_parameterspace=np.linspace(1, 20, num=5),
#                           chi1_parameterspace=np.linspace(-0.9, 0.9, num=5),
#                           chi2_parameterspace=np.linspace(-0.9, 0.9, num=5),
#                           min_greedy_error_amp=1e-6,
#                           min_greedy_error_phase=1e-6,
#                           truncate_at_tmin=True,
#                           truncate_at_ISCO=True,
#                           f_lower=10
#                           )

# tr_obj_p = gt._get_training_obj('phase')
# tr_obj_a = gt._get_training_obj('amplitude')

# gt._calculate_residuals(tr_obj_p,
#                         # plot_polarizations=True,
#                         # plot_residuals_time_evolve=True,
#                         # plot_residuals_eccentric_evolve=True
#                         )

# rb_obj_p = gt.get_greedy_parameters(tr_obj_p,
#                         #  plot_greedy_error=True,
#                         #  plot_basis_indices=True,
#                          )

# gt.get_empirical_nodes(reduced_basis_object=rb_obj_p,
#                        train_obj=tr_obj_p, 
#                     #    plot_emp_nodes_on_basis=True
#                        )

# gt = Generate_TrainingSet(time_array=time_array,
#                           ecc_ref_parameterspace=np.linspace(0.001, 0.3, num=20),
#                           mean_ano_parameterspace=[0],
#                           mass_ratio_parameterspace=np.linspace(1, 20, num=20),
#                           chi1_parameterspace=[0],
#                           chi2_parameterspace=[0],
#                           min_greedy_error_amp=1e-8,
#                           min_greedy_error_phase=1e-8,
#                           truncate_at_tmin=True,
#                           truncate_at_ISCO=True,
#                           f_lower=10)

# gt = Generate_TrainingSet(time_array=time_array,
#                           ecc_ref_parameterspace=np.linspace(0.001, 0.3, num=2),
#                           mean_ano_parameterspace=[0, np.pi],
#                           mass_ratio_parameterspace=[1, 2],
#                           chi1_parameterspace=np.linspace(-0.9, 0.9, num=10),
#                           chi2_parameterspace=[0],
#                           min_greedy_error_amp=1e-8,
#                           min_greedy_error_phase=1e-8,
#                           f_lower=10)


# train_obj_p = gt._get_training_obj('phase')
# gt._calculate_residuals(train_obj_p)


# train_obj_a = gt._get_training_obj('amplitude')
# gt._calculate_residuals(train_obj_a)
# # gt.get_greedy_parameters(train_obj, N_greedy_vecs=50, plot_greedy_error=True)
# gt.get_greedy_parameters(train_obj_a, min_greedy_error=1e-6, plot_greedy_error=True, plot_basis_indices=True)
# gt._generate_polarisation_data(train_obj)


# gt.get_training_set_greedy(property="amplitude", 
#                             plot_interpolation_matrix=True, save_fig_interpolation_matrix=True,
#                             plot_proj_vs_eim_error=True, save_fig_proj_vs_eim_error=True,
#                             plot_greedy_vecs=True, save_fig_greedy_vecs=True,
#                             plot_residuals_time=True, save_fig_residuals_time=True,
#                             plot_emp_nodes_on_basis=True, save_fig_emp_nodes_on_basis=True,
#                             plot_training_set=True, save_fig_training_set=True,
#                             plot_residuals_eccentric=True, save_fig_residuals_eccentric=True,
#                             plot_greedy_error=True,save_fig_greedy_error=True
#                         )

# gt.get_training_set_greedy(property="phase", 
#                             plot_interpolation_matrix=True, save_fig_interpolation_matrix=True,
#                             plot_proj_vs_eim_error=True, save_fig_proj_vs_eim_error=True,
#                             plot_greedy_vecs=True, save_fig_greedy_vecs=True,
#                             plot_residuals_time=True, save_fig_residuals_time=True,
#                             plot_emp_nodes_on_basis=True, save_fig_emp_nodes_on_basis=True,
#                             plot_training_set=True, save_fig_training_set=True,
#                             plot_residuals_eccentric=True, save_fig_residuals_eccentric=True,
#                             plot_greedy_error=True, save_fig_greedy_error=True,
#                             plot_basis_indices=True, save_fig_basis_indices=True,
#                             )

# plt.show()
# plt.close("all")