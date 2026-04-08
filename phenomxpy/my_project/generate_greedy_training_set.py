from time import time
from tracemalloc import start

from generate_PhenomTE import *

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
from scipy.linalg import orth

# --------------------------------------------------------------------

# plt.switch_backend('WebAgg')
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TrainingSetParameters:
    """
    Dataclass to store the parameters and results of the greedy algorithm for a training set.
    Attributes:
    property [str]: the property for which the greedy algorithm is applied ("phase" or "amplitude")
    e [np.ndarray]: eccentricities of the dataset used for reduced basis construction
    l [np.ndarray]: mean anomalies of the dataset used for reduced basis construction
    q [np.ndarray]: mass ratios of the dataset used for reduced basis construction
    chi1 [np.ndarray]: dimensionless spin of the primary black hole
    chi2 [np.ndarray]: dimensionless spin of the secondary black hole
    Nb [int]: number of greedy basis vectors selected
    gerr [float]: greedy error
    fref [float]: reference frequency
    flow [float]: lower frequency
    phi [float]: phase
    inc [float]: inclination
    isco [bool]: innermost stable circular orbit truncation
    tmin [bool]: minimum time at which waveform computation is considered physical
    luminosity_distance [float]: luminosity distance
    circ [np.ndarray]: circular phase or amplitude
    residuals [np.ndarray]: residuals in the parameter space for the chosen property (phase or amplitude)
    basis_indices [list]: indices of the selected greedy basis vectors in the original parameter space
    empirical_indices [list]: indices of the empirical interpolation nodes in the original time array
    residual_basis [np.ndarray]: the reduced basis of the residuals in the parameter space for the chosen property (phase or amplitude)
    training_set [np.ndarray]: the training set of waveforms for the chosen property (phase or amplitude)
    """
    property: str = "phase"  # or "amplitude"

    e: Any = None
    l: Any = None
    q: Any = None
    chi1: Any = None
    chi2: Any = None

    t: Any = None
    

    Nb: Optional[int] = None
    gerr: Optional[float] = None

    fref: float = None
    flow: float = None
    phi: float = None
    inc: float = None
    isco: bool = None
    tmin: bool = None

    luminosity_distance: Optional[float] = None

    # Calculated properties:
    circ: Any = None # circular phase or amplitude
    residuals: Any = None # residuals in the parameter space for the chosen property (phase or amplitude)
    basis_indices: Any = field(default_factory=list) # indices of the selected greedy basis vectors in the original parameter space
    empirical_indices: Any = field(default_factory=list) # indices of the empirical interpolation nodes in the original time array
    residual_basis: Any = None # the reduced basis of the residuals in the parameter space for the chosen property (phase or amplitude)
    training_set: Any = None # the training set of waveforms for the chosen property (phase or amplitude)

    def __post_init__(self):
        self.e = np.round(np.asarray(self.e, dtype=float), 4)
        self.l = np.round(np.asarray(self.l, dtype=float), 4)
        self.q = np.round(np.asarray(self.q, dtype=float), 4)
        self.chi1 = np.round(np.asarray(self.chi1, dtype=float), 4)
        self.chi2 = np.round(np.asarray(self.chi2, dtype=float), 4)

    @staticmethod
    def _range_block(name, values):
        values = np.asarray(values, dtype=float)
        return f"{name}=[{values.min():g}_{values.max():g}_N={len(values)}]"

    @staticmethod
    def _scalar_block(name, value):
        return f"{name}={value:g}"

    def name_blocks(self):
        blocks = [
            self.property,
            self._range_block("e", self.e),
            self._range_block("l", self.l),
            self._range_block("q", self.q),
            self._range_block("x1", self.chi1),
            self._range_block("x2", self.chi2),
            self._scalar_block("fr", self.fref),
            self._scalar_block("fl", self.flow),
        ]

        if self.phi != 0:
            blocks.append(self._scalar_block("phi", self.phi))
        if self.inc != 0:
            blocks.append(self._scalar_block("incl", self.inc))

        if self.Nb is not None:
            blocks.append(f"Nb={self.Nb}")
        if self.gerr is not None:
            blocks.append(f"gerr={self.gerr}")


        if not self.isco:
            blocks.append("noISCO")
        if not self.tmin:
            blocks.append("notmin")

        if self.luminosity_distance is not None:
            blocks.append("SI")

        return blocks

    def filename(self, prefix="data", ext="npz", directory=None):
        name = f"{prefix}_{'_'.join(self.name_blocks())}.{ext}"
        if directory is not None:
            return f"{directory.rstrip('/')}/{name}"
        return name

    def figname(self, prefix="fig", ext="png", directory=None):
        # Ensure the directory exists, creating it if necessary and save
        if directory is not None:
            os.makedirs(directory, exist_ok=True)

        figname = self.filename(prefix=prefix, ext=ext, directory=directory)
        w = Warnings()
        print(w.colored_text(f"Figure is saved in {figname}", 'blue'))

        return figname
    
    def save_residuals(self, prefix, directory):
        
        os.makedirs(directory, exist_ok=True)

        filepath = self.filename(prefix=prefix, ext="npz", directory=directory)

        if self.residuals is None:
            raise ValueError("Residuals are not calculated yet. Run generate_property_dataset() first.")
        else:
            # If file already exists, do not overwrite
            if not os.path.isfile(filepath):
                np.savez(
                    filepath,
                    residuals = self.residuals,
                    t = self.t,
                    circ = self.circ,
                )

            warnings = Warnings()
            print(warnings.colored_text(f"Residual dataset saved in {filepath}", 'blue'))

        return filepath

    def load_residuals(self, filepath):
        data = np.load(filepath, allow_pickle=True)

        self.residuals = data['residuals']
        self.t = data['t']
        self.circ = data['circ']

        data.close()

        warnings = Warnings()
        print(warnings.colored_text(f"Residual dataset found and loaded: {filepath}", 'blue'))

        return self
    
    def save_polarisations(self, hp_dataset, hc_dataset, prefix, directory):
        os.makedirs(directory, exist_ok=True)

        filepath = self.filename(prefix=prefix, ext="npz", directory=directory)

        if hp_dataset is None or hc_dataset is None:
            raise ValueError("Polarisation datasets are not provided.")
        else:
            # If file already exists, do not overwrite
            if not os.path.isfile(filepath):
                np.savez(
                    filepath,
                    hp = hp_dataset,
                    hc = hc_dataset,
                    time = self.t,
                )

            warnings = Warnings()
            print(warnings.colored_text(f"Polarisation dataset saved in {filepath}", 'blue'))

        return filepath

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
                 minimum_spacing_greedy=0.005, 
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
        
        minimum_spacing_greedy [float] : Minimum spacing in eccentricity for greedy selection. Defaults to 0.005.
        """
        # Check if property is valid and adjust settings accordingly
        self.ecc_ref_space = self.allowed_eccentricity_warning(ecc_ref_parameterspace)
        self.mass_ratio_space = self.allowed_mass_ratio_warning(mass_ratio_parameterspace)
        self.mean_ano_ref_space = self.allowed_mean_anomaly_warning(mean_ano_parameterspace)
        self.chi1_space = self.allowed_chispin_warning(chi1_parameterspace)
        self.chi2_space = self.allowed_chispin_warning(chi2_parameterspace)

        self.minimum_spacing_greedy = minimum_spacing_greedy

        self.min_greedy_error_amp = min_greedy_error_amp
        self.min_greedy_error_phase = min_greedy_error_phase
        self.N_basis_vecs_amp = N_basis_vecs_amp
        self.N_basis_vecs_phase = N_basis_vecs_phase

        # To be stored parameters
        self.residuals_space = None
        self.residual_reduced_basis = None
        self.indices_basis = None
        self.best_rep_parameters = None
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

    def generate_property_dataset(self, property, ecc_ref_list=None, mean_ano_ref_list=None, mass_ratios_list=None, chi1_list=None, chi2_list=None, save_dataset_to_file=None, plot_residuals_time_evolv=False, plot_residuals_eccentric_evolv=False, save_fig_eccentric_evolv=False, save_fig_time_evolve=False, show_legend=False):
        """
        Generates a dataset of waveform residuals based on the specified property for a certain range of eccentricities (ecc).

        Parameters:
        ----------
        ecc_list : list of floats
            List of reference eccentricities for which to calculate residuals.
        mean_ano_ref_list : list of floats
            List of reference mean anomalies for which to calculate residuals.
        property : str
            Specifies which property to calculate ('phase' or 'amplitude').
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
        # Resolve the parameter space for eccentricities and mass ratios, using the provided lists or default spaces
        ecc_ref_list = self.resolve_property(prop=ecc_ref_list, default=self.ecc_ref_space) 
        mean_ano_ref_list = self.resolve_property(prop=mean_ano_ref_list, default=self.mean_ano_ref_space)
        mass_ratios_list = self.resolve_property(prop=mass_ratios_list, default=self.mass_ratio_space) 
        chi1_list = self.resolve_property(prop=chi1_list, default=self.chi1_space)
        chi2_list = self.resolve_property(prop=chi2_list, default=self.chi2_space)

        # Create training set objects 
        common_params = dict(
            e=ecc_ref_list,
            l=mean_ano_ref_list,
            q=mass_ratios_list,
            chi1=chi1_list,
            chi2=chi2_list,
            fref=self.f_ref,
            flow=self.f_lower,
            phi=self.phiRef,
            inc=self.inclination,
            isco=self.truncate_at_ISCO,
            tmin=self.truncate_at_tmin,
        )

        if property == "phase":
            self.training_phase = TrainingSetParameters(**common_params, property="phase", Nb=self.N_basis_vecs_phase, gerr=self.min_greedy_error_phase)
        # amplitude
        else:
            self.training_amp = TrainingSetParameters(**common_params, property="amplitude", Nb=self.N_basis_vecs_amp, gerr=self.min_greedy_error_amp)

        # training object for the chosen property (phase or amplitude)
        train_obj = self._get_training_obj(property)

        try:
            filename = train_obj.filename(
                prefix=f"residuals",
                directory="Straindata/Residuals"
            )
        # Attempt to load existing residual dataset
            # filename = f'Straindata/Residuals/residuals_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(ecc_ref_list)}_{max(ecc_ref_list)}_N={len(ecc_ref_list)}]_l=[{min(mean_ano_ref_list)}_{max(mean_ano_ref_list)}_N={len(mean_ano_ref_list)}].npz'
            train_obj = train_obj.load_residuals(filepath=filename)
            self.time = train_obj.t

            

        # except FileNotFoundError:
        #     print(f"Could not find {filename}")

        # try: 
        #     # Attempt to load existing residual dataset
        #     filename = f'Straindata/Residuals/residuals_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(ecc_ref_list)}_{max(ecc_ref_list)}_N={len(ecc_ref_list)}].npz'
        #     load_residuals = np.load(filename)
            
        #     residual_dataset = load_residuals['residual']
        #     self.time = load_residuals['time']
            
            # print(f'Residual parameterspace dataset found for {property}')
            
        except Exception as e:
            print(e)

            # If attempt to load residuals failed, generate polarisations and calculate residuals
            hp_dataset, hc_dataset = self._generate_polarisation_data(train_obj=train_obj)
            self._calculate_residuals(train_obj=train_obj, 
                                      hp_dataset=hp_dataset, 
                                      hc_dataset=hc_dataset, 
                                      save_dataset_to_file=save_dataset_to_file, 
                                      plot_residuals_eccentric_evolv=plot_residuals_eccentric_evolv, 
                                      plot_residuals_time_evolv=plot_residuals_time_evolv, 
                                      save_fig_eccentric_evolv=save_fig_eccentric_evolv, 
                                      save_fig_time_evolve=save_fig_time_evolve, 
                                      show_legend=show_legend)
            
            del hp_dataset, hc_dataset  # Free memory

        #     print(f'Generated residual parameterspace dataset for {property} ', len(ecc_ref_list), ' waveforms')
        #     # If save_dataset_to_file is True save the residuals to file in Straindata/Residuals
        #     if save_dataset_to_file is True and not os.path.isfile(f'Straindata/Residuals/residuals_{property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(ecc_ref_list)}_{max(ecc_ref_list)}_N={len(ecc_ref_list)}].npz'):
        #         self._save_residual_dataset(ecc_ref_list, property, residual_dataset)

        # # If plot_residuals is True, plot whole residual dataset
        # if (plot_residuals_eccentric_evolv is True) or (plot_residuals_time_evolv is True):
        #     self._plot_residuals(residual_dataset, ecc_ref_list, property, plot_residuals_eccentric_evolv, plot_residuals_time_evolv, save_fig_eccentric_evolv, save_fig_time_evolve, show_legend=show_legend )

        return train_obj
    
    def _generate_polarisation_data(self, train_obj:TrainingSetParameters, truncate_at_ISCO=None, truncate_at_tmin=None):
        """
        Helper function to generate polarisation data for a list of eccentricities.

        Parameters:
        ----------
        ecc_list : list of floats
            List of minimum eccentricities.

        Returns:
        -------
        hp_dataset : ndarray
            Plus polarisation data.
        hc_dataset : ndarray
            Cross polarisation data.

        """

        truncate_at_ISCO = self.resolve_property(prop=truncate_at_ISCO, default=self.truncate_at_ISCO)
        truncate_at_tmin = self.resolve_property(prop=truncate_at_tmin, default=self.truncate_at_tmin)
    
        try:
            # Attempt to load existing polarisation dataset
            # filename = training_obj.filename(prefix=f"polarisation", directory="Straindata/Polarisations")
            filename = f'Straindata/Polarisations/polarisations_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(train_obj.e)}_{max(train_obj.e)}_N={len(train_obj.e)}]_t_lower={int(self.time[0])}.npz'
            load_polarisations = np.load(filename, allow_pickle=True)
            hp_dataset = load_polarisations['hp']
            hc_dataset = load_polarisations['hc']
            self.time = load_polarisations['time']

            print('Loaded polarisations')

        except:
            # Get waveform size for truncated ISCO waveform of smallest waveform
            sorted_ecc_list = np.sort(train_obj.e)

            ISCO_ecc = sorted_ecc_list[-1] # Highest eccentricity in the list --> earliest ISCO cut-off
            # update_results=True will update the instance time domain array which is afterwards used on all later generated waveforms.
            hp_ISCO, hc_ISCO, _ = self.simulate_waveform(ecc_ref=ISCO_ecc, mass_ratio=1, chi1=0, chi2=0, truncate_at_ISCO=truncate_at_ISCO, truncate_at_tmin=truncate_at_tmin, update_results=True)

            hp_dataset = np.zeros((len(train_obj.e), len(self.time))) 
            hc_dataset = np.zeros((len(train_obj.e), len(self.time)))

            for i, ecc in enumerate(train_obj.e):
                if ecc == ISCO_ecc:
                    # Store first waveform in dataset
                    hp_dataset[i] = hp_ISCO
                    hc_dataset[i] = hc_ISCO
                else:
                    # No need to truncate again, since the time array is already truncated to the earliest ISCO cut-off
                    hp_dataset[i], hc_dataset[i], _ = self.simulate_waveform(ecc_ref=ecc, truncate_at_ISCO=False, truncate_at_tmin=False, update_results=False) 

                # self.time = self.time[:len(hp_ISCO)]  # Ensure time array matches the dataset waveform length

            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Straindata/Polarisations', exist_ok=True)
            # filename = train_obj.filename(prefix=f"polarisation", directory="Straindata/Polarisations")
            np.savez(f'Straindata/Polarisations/polarisations_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(train_obj.e)}_{max(train_obj.e)}_N={len(train_obj.e)}]_t_lower={int(self.time[0])}.npz', hp=hp_dataset, hc=hc_dataset, time=self.time)

        return hp_dataset, hc_dataset

    def _calculate_residuals(self, train_obj:TrainingSetParameters, hp_dataset, hc_dataset, save_dataset_to_file=False, plot_residuals_eccentric_evolv=False, plot_residuals_time_evolv=False, save_fig_eccentric_evolv=False, save_fig_time_evolve=False, show_legend=False):
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

        self.circulair_wf()

        # Create empty residual dataset
        residual_dataset = np.zeros((len(train_obj.e), len(self.time)))

        # Fill residual dataset with residuals of chosen property for given eccentric parameter space
        for i, ecc in enumerate(train_obj.e):
            residual = self.calculate_residual(hp_dataset[i], hc_dataset[i], ecc, train_obj.property)
            residual_dataset[i] = residual

            del residual

        # Assign the calculated residual dataset to the corresponding training object
        train_obj.residuals = residual_dataset
        train_obj.t = self.time

        print(f'Generated residual parameterspace dataset for {train_obj.property} ', len(train_obj.e), ' waveforms')

        # If save_dataset_to_file is True save the residuals to file in Straindata/Residuals
        # filename = train_obj.filename(prefix=f"residuals_{train_obj.property}", directory="Straindata/Residuals")
        # filename = f'Straindata/Residuals/residuals_{train_obj.property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(train_obj.e)}_{max(train_obj.e)}_N={len(train_obj.e)}].npz'
        if save_dataset_to_file:
            train_obj.save_residuals(prefix=f"residuals", directory="Straindata/Residuals")

        # If plot_residuals is True, plot whole residual dataset
        if (plot_residuals_eccentric_evolv is True) or (plot_residuals_time_evolv is True):
            self._plot_residuals(train_obj, plot_residuals_eccentric_evolv, plot_residuals_time_evolv, save_fig_eccentric_evolv, save_fig_time_evolve, show_legend=show_legend)
        
        self.residuals_space = residual_dataset # Store the residual dataset as an attribute for later use in greedy algorithm

        return residual_dataset
    

    def _plot_residuals(self, train_obj, plot_eccentric_evolv=False, plot_time_evolve=False, save_fig_eccentric_evolve=False, save_fig_time_evolve=False, show_legend=False):
        """Function to plot residuals dataset including save figure option."""
        ecc_list = train_obj.e
        residual_dataset = train_obj.residuals

        if plot_eccentric_evolv is True:
            fig_residuals_ecc = plt.figure()
            for i in range(0, len(self.time), 100):
                plt.plot(ecc_list, residual_dataset.T[i], label='t/M = ' + f'{round(self.time[i], 1)}', linewidth=0.6)
                
            plt.xlabel('eccentricity')
            if train_obj.property == 'phase':
                plt.ylabel(' $\Delta \phi_{22}$ [radians]')
            elif train_obj.property == 'amplitude':
                plt.ylabel('$\Delta A_{22}$')
            else:
                print('Choose property = "phase", "amplitude"', train_obj.property)
                sys.exit(1)

            plt.title(f'Residuals {train_obj.property}')
            plt.grid(True)

            if show_legend:
                plt.legend(loc='upper right', fontsize='small', ncol=2)

            plt.tight_layout()

            if save_fig_eccentric_evolve is True:
                ISCO = '' if self.truncate_at_ISCO else 'NO_ISCO_'
                tmin = '' if self.truncate_at_tmin else 'NO_tmin_'

                figname = train_obj.figname(prefix=f'Residuals_eccentric_evolv', directory='Images/Residuals') # Use the filename method of the training object to generate a consistent filename
                fig_residuals_ecc.savefig(figname)

                # plt.close('all') # Clean up plot

        if plot_time_evolve is True:
            fig_residuals_t = plt.figure()

            for i in range(len(residual_dataset)):
                plt.plot(self.time, residual_dataset[i], label='e$_{ref}$' + f' = {round(ecc_list[i], 3)}', linewidth=0.6)
               
            plt.xlabel('t [M]')
            if train_obj.property == 'phase':
                plt.ylabel(' $\Delta \phi_{22}$ [radians]')
            elif train_obj.property == 'amplitude':
                plt.ylabel('$\Delta A_{22}$')
            else:
                print('Choose property = "phase", "amplitude"', train_obj.property)
                sys.exit(1)

            plt.title(f'Residuals {train_obj.property}')
            plt.grid(True)

            if show_legend:
                plt.legend(loc='upper right', fontsize='small', ncol=2)

            plt.tight_layout()

            if save_fig_time_evolve is True:
                ISCO = '' if self.truncate_at_ISCO else 'NO_ISCO'
                tmin = '' if self.truncate_at_tmin else 'NO_tmin'

                figname = train_obj.figname(prefix=f'Residuals_time_evolv', directory='Images/Residuals') # Use the filename method of the training object to generate a consistent filename
                fig_residuals_t.savefig(figname)

                # plt.close('all')
    

    def _save_residual_dataset(self, train_obj):
        """Function to save residual dataset to file."""

        ecc_list = train_obj.ecc_list
        residual_dataset = train_obj.residuals

        os.makedirs('Straindata/Residuals', exist_ok=True)
        # train_obj.filename(prefix=f"residuals", directory="Straindata/Residuals")
        file_path = f'Straindata/Residuals/residuals_{train_obj.property}_f_lower={self.f_lower}_f_ref={self.f_ref}_e=[{min(ecc_list)}_{max(ecc_list)}_N={len(ecc_list)}].npz'
        np.savez(file_path, residual=residual_dataset, time=self.time, eccentricities=ecc_list)
        print('Residuals saved to Straindata/Residuals')


    def _get_training_obj(self, property):
        if property == "amplitude":
            return self.training_amp
        elif property == "phase":
            return self.training_phase
        else:
            raise ValueError(f"Unknown property: {property}")
        

    def get_greedy_parameters(self, train_obj: TrainingSetParameters, min_greedy_error=None, N_greedy_vecs=None, normalize=True, max_tree_depth=0, plot_greedy_error=False, save_greedy_error_fig=False, plot_greedy_vectors=False, save_greedy_vecs_fig=False, plot_SVD_matrix=False, save_SVD_matrix_fig=False, show_legend=False):
            """
            Greedy algorithm to select representative vectors from U using an orthonormal basis.

            Parameters
            ----------
            U : ndarray, shape (num_vectors, vector_length)
                Dataset of vectors to build the greedy basis from.
            N_greedy_vecs : int, optional
                Maximum number of vectors to include in the greedy basis.
            tol : float, optional
                Stop when the maximum residual norm falls below this tolerance.
            minimum_spacing : float, optional
                If not None, greedy points are picked with a minimum spacing in between points.

            Returns
            -------
            greedy_basis : ndarray
                Orthonormal greedy basis vectors.
            greedy_indices : list
                Indices of the vectors chosen from U.
            residuals : list
                Maximum residual norm at each iteration.
            """
            greedy_tol = self.resolve_property(prop=min_greedy_error, default=-np.inf) 
            nmax = self.resolve_property(prop=N_greedy_vecs, default=train_obj.residuals.shape[0])
            
            parameters = self.ecc_ref_space[self.ecc_ref_space != 0]
            reduced_basis_object = ReducedBasis(greedy_tol=greedy_tol, normalize=normalize, nmax=nmax, lmax=max_tree_depth)

            reduced_basis_object.fit(training_set = train_obj.residuals,
                parameters = parameters,
                physical_points = self.time
                )
            
            train_obj.basis_indices = []

            for leaf in reduced_basis_object.tree.leaves:
                # print(f'Leaf {i} leaf indices: {leaf.indices}, leaf error: {leaf.errors}')    
                train_obj.basis_indices.extend(int(idx) for idx in leaf.indices)


            if plot_greedy_error:
                self._plot_greedy_errors(reduced_basis_object=reduced_basis_object, train_obj=train_obj, save_greedy_fig=save_greedy_error_fig)
            
            if plot_greedy_vectors:
                self._plot_greedy_vectors(reduced_basis_object=reduced_basis_object, train_obj=train_obj, save_greedy_vecs_fig=save_greedy_vecs_fig, U=None, show_legend=show_legend)
            
            if plot_SVD_matrix:
                self._plot_SVD_matrix(train_obj=train_obj, save_SVD_matrix_fig=save_SVD_matrix_fig)
            
            return reduced_basis_object

    def _plot_greedy_errors(self, reduced_basis_object: ReducedBasis, train_obj: TrainingSetParameters, save_greedy_fig=False):
        """Function to plot greedy errors of the reduced basis. Option to save figure in Images/Greedy_errors.
        
        Properties:
        - reduced_basis_object: ReducedBasis object from which to extract the greedy basis and calculate projection errors.
        - train_obj: object containing the dataset of vectors used to build the greedy basis.
        - save_greedy_fig: bool, if True, saves the figure in Images/Greedy_errors with a name that includes the property and other relevant parameters.

        Returns:
        - proj_errors: list of projection errors for each number of greedy basis vectors.
        """

        reduced_basis = []
        for i in range(len(reduced_basis_object.tree.leaves)):
                # print(f'Leaf {i} leaf indices: {leaf.indices}, leaf error: {leaf.errors}')    
                reduced_basis.extend(reduced_basis_object.tree.leaves[i].basis)
        reduced_basis = np.array(reduced_basis) # (n_greedy_vecs, n_time)

        # # Greedy errors
        # proj_errors = []

        # for k in range(1, len(reduced_basis) + 1):
        #     basis_k = reduced_basis[:k]                        # (k, n_time)
        #     max_proj_err = 0.0

        #     for f in train_obj.residuals: # Loop over all vectors in the original dataset
        #         # --- Best projection onto span(basis_k) ---
        #         # Solve min ||f - c @ basis_k||
        #         coeff_proj, *_ = np.linalg.lstsq(basis_k.T, f, rcond=None)
        #         f_proj = basis_k.T @ coeff_proj

        #         proj_err = np.linalg.norm(f - f_proj) / np.linalg.norm(f)
        #         max_proj_err = max(max_proj_err, proj_err)

        #     proj_errors.append(max_proj_err)

        def stack_greedy_errors(node):
            # if leaf node
            if hasattr(node, "errors"):
                return node.errors
            # otherwise, concatenate the errors of children
            errs = []
            if hasattr(node, "children"):
                for child in node.children:
                    errs = np.hstack([errs, stack_greedy_errors(child)])

            return errs

        stacked_proj_errors = stack_greedy_errors(reduced_basis_object.tree)

        fig_greedy_errors = plt.figure()
        plt.semilogy(np.arange(len(stacked_proj_errors)), stacked_proj_errors, label='greedy error', lw=1.8, color='blue')
        plt.yscale('log')
        plt.ylabel('Greedy error')
        plt.title(f'Greedy error per section for max_tree_depth = {reduced_basis_object.lmax}, tree_leaves = {len(reduced_basis_object.tree.leaves)}')
        plt.legend()

        # ax[1].semilogy(np.arange(len(proj_errors)), proj_errors, label='greedy error', lw=1.8, color='blue')
        # ax[1].set_xlabel('# of greedy vectors')
        # ax[1].set_yscale('log')
        # ax[1].set_ylabel('Greedy error')
        # ax[1].set_title('Greedy error over the full basis')
        # ax[1].legend()

        if save_greedy_fig:
            figname = train_obj.figname(prefix=f'Greedy_error', directory='Images/Greedy_errors') # Use the figname method of the training object to generate a consistent filename
            fig_greedy_errors.savefig(figname)

            # plt.close('all')

        return stacked_proj_errors

    def _plot_SVD_matrix(self, train_obj: TrainingSetParameters, save_SVD_matrix_fig=False):
        """Function to plot the SVD matrix of the dataset. Option to save figure in Images/SVD_matrices."""

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
            axes[2].plot(self.ecc_ref_parameterspace, U[:, i], label=f"Mode {i+1}")

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

    # def _plot_greedy_vectors(self, reduced_basis_object, property, save_greedy_vecs_fig, U=None, show_legend=True):
    #     """Function to plot and option to save the greedy basis vectors. If U is specified, also plot complete dataset before greedy algorithm."""
      
    #     fig_greedy_vecs, (ax_main, ax_bottom) = plt.subplots(
    #         2, 1, figsize=(12, 6),
    #         gridspec_kw={'height_ratios': [4, 0.5]}
    #     )

    #     # --- Top plot: dataset and greedy basis vectors ---
    #     if U is not None:
    #         for i, vec in enumerate(U):
    #             ax_main.plot(vec, color='grey', alpha=0.3, label='Vector dataset' if i == 0 else None)

    #     for j in range(len(reduced_basis_object.tree.leaves)):
    #         for i, vec in enumerate(reduced_basis_object.tree.leaves[j].basis):
    #             ax_main.plot(vec, linewidth=0.6,
    #                         label=f'$e$={round(self.ecc_ref_space[self.indices_basis[i]], 3)}')
    
        

    #     ax_main.set_ylabel('Vector Value')
    #     ax_main.set_title(f'Greedy Basis Vectors ({len(self.indices_basis)} vectors) for {property}')
    #     if show_legend:
    #         ax_main.legend()
    #     ax_main.grid(True)

    #     # --- Bottom plot: eccentric points and greedy indices ---
    #     ecc_values = self.ecc_ref_space # assuming first component = eccentricity
    #     y = np.zeros(len(self.ecc_ref_space))

    #     # All dataset points
    #     ax_bottom.plot(ecc_values, y, color='grey', alpha=0.6, label='Dataset points')

    #     # Greedy-selected points
    #     ax_bottom.scatter(ecc_values[self.indices_basis], y[self.indices_basis], color='red', s=20, label='Greedy parameters')

    #     ax_bottom.set_yticks([])
    #     ax_bottom.set_xlabel('Vector index / parameter')
    #     ax_bottom.grid(True, axis='x', linestyle='--', alpha=0.5)
    #     if show_legend:
    #         ax_main.legend(loc='best', ncol=3, fontsize='small')


    #     plt.tight_layout()
    #     fig_greedy_vecs.show()


    #     if save_greedy_vecs_fig:
    #         os.makedirs('Images/Greedy_vectors', exist_ok=True)
    #         figname = f'Images/Greedy_vectors/Greedy_vectors_{property}_M={self.total_mass}_ecc=[{min(self.ecc_ref_space)}_{max(self.ecc_ref_space)}]_f_lower={self.f_lower}_f_ref={self.f_ref}_iN={len(self.ecc_ref_space)}_ms={self.minimum_spacing_greedy}.png'

    #         plt.savefig(figname)
            
    #         print(self.colored_text(f'Greedy vectors fig saved to {figname}', 'blue'))
    #         # plt.close('all')

    def _plot_greedy_vectors(self, reduced_basis_object: ReducedBasis, train_obj: TrainingSetParameters, save_greedy_vecs_fig=False, U=None, show_legend=True):
        """Function to plot and optionally save the greedy basis vectors.
        If U is specified, also plot complete dataset before greedy algorithm.
        """

        # Detect tree size from number of leaves
        children = len(reduced_basis_object.tree.leaves)
        print(f'Number of child nodes: {children}')

        # Base figure size
        base_width = 30
        base_height = 5

        # Extend height with number of leaves
        fig_height = base_height + 2.5*children if children > 0 else base_height

        fig_greedy_vecs, (ax_main, ax_bottom) = plt.subplots(
            2, 1,
            figsize=(base_width, fig_height),
            gridspec_kw={'height_ratios': [5, 2]}
        )

        # --------------------------------------------------
        # Top plot
        # --------------------------------------------------
        if U is not None:
            for i, vec in enumerate(U):
                ax_main.plot(
                    vec,
                    color='grey',
                    alpha=0.3,
                    label='Vector dataset' if i == 0 else None
                )

        for j in range(len(reduced_basis_object.tree.leaves)):
            leaf = reduced_basis_object.tree.leaves[j]
            for i, vec in enumerate(leaf.basis):
                label = None
                if i < len(self.indices_basis):
                    label = f'$e$={round(self.ecc_ref_space[self.indices_basis[i]], 3)}'
                ax_main.plot(vec, linewidth=0.6, label=label)

        ax_main.set_ylabel('Vector Value')
        ax_main.set_title(
            f'Greedy Basis Vectors ({len(self.indices_basis)} vectors for {train_obj.property})'
        )

        if show_legend:
            ax_main.legend(loc='best', ncol=3)

        ax_main.grid(True)

        # --------------------------------------------------
        # Bottom plot: visualize tree splitting
        # --------------------------------------------------

        # First pass:
        # traverse the tree, assign one unique color per node, and collect counts by depth
        traversal = []
        depth_counts = {}

        color_counter = 0
        stack = [(reduced_basis_object.tree, 0, f"C{color_counter}")]

        while stack:
            node, depth, color = stack.pop()
            traversal.append((node, depth, color))

            if hasattr(node, 'indices') and node.indices is not None:
                n_greedy = len(node.indices)
            else:
                n_greedy = 0

            if depth not in depth_counts:
                depth_counts[depth] = []
            depth_counts[depth].append(n_greedy)

            # Assign globally unique colors to children
            if hasattr(node, 'children') and node.children is not None:
                for child in reversed(node.children):
                    color_counter += 1
                    stack.append((child, depth + 1, f"C{color_counter}"))
            else:
                # Fallback for split flags without explicit child objects
                if hasattr(node, 'idxs_subspace1') and node.idxs_subspace1 is not None:
                    color_counter += 1
                    stack.append((node, depth + 1, f"C{color_counter}"))
                if hasattr(node, 'idxs_subspace0') and node.idxs_subspace0 is not None:
                    color_counter += 1
                    stack.append((node, depth + 1, f"C{color_counter}"))

        # Build one greedy label per depth
        depth_labels = {}
        for d, counts in depth_counts.items():
            counts_str = ", ".join(str(c) for c in counts)
            point_word = "point" if len(counts) == 1 and counts[0] == 1 else "points"
            depth_labels[d] = f"Greedy picks (depth {d}: {counts_str} {point_word})"

        used_depth_labels = set()
        used_dataset_label = False

        # Second pass: actual plotting
        for node, depth, color in traversal:
            x = np.asarray(node.train_parameters).squeeze()

            ax_bottom.scatter(
                x,
                [depth] * len(x),
                s=20,
                alpha=0.3,
                color='grey',
                label=f"Node depth {depth} ({len(x)} pts)" if (depth == 0 and not used_dataset_label) else None
            )

            if depth == 0:
                used_dataset_label = True

            if hasattr(node, 'indices') and node.indices is not None:
                greedy_params = x[node.indices]

                ax_bottom.scatter(
                    greedy_params,
                    [depth] * len(greedy_params),
                    s=80,
                    edgecolor='k',
                    facecolor=color,
                    label=depth_labels[depth] if depth not in used_depth_labels else None
                )

                used_depth_labels.add(depth)

        ax_bottom.set_xlabel("Eccentricity")
        ax_bottom.set_ylabel("Tree depth")
        ax_bottom.grid(True, axis='x', linestyle='--', alpha=0.5)


        ax_bottom.legend(loc='best')

        plt.tight_layout()

        if save_greedy_vecs_fig:
            figname = train_obj.figname(prefix=f'Greedy_vectors', directory='Images/Greedy_vectors') # Use the figname method of the training object to generate a consistent filename
            plt.savefig(figname)


    def get_greedy_parameters_old(self, U, property, min_greedy_error=None, N_greedy_vecs=None, reg=1e-6, plot_greedy_error=False, save_greedy_error_fig=False, plot_greedy_vectors=False, save_greedy_vecs_fig=False, plot_greedy_basis_formation=False, minimum_spacing=None):
        """
        Greedy algorithm to select representative vectors from U using an orthonormal basis.

        Parameters
        ----------
        U : ndarray, shape (num_vectors, vector_length)
            Dataset of vectors to build the greedy basis from.
        N_greedy_vecs : int, optional
            Maximum number of vectors to include in the greedy basis.
        tol : float, optional
            Stop when the maximum residual norm falls below this tolerance.
        minimum_spacing : float, optional
            If not None, greedy points are picked with a minimum spacing in between points.

        Returns
        -------
        greedy_basis : ndarray
            Orthonormal greedy basis vectors.
        greedy_indices : list
            Indices of the vectors chosen from U.
        residuals : list
            Maximum residual norm at each iteration.
        """

        if minimum_spacing is None:
            minimum_spacing = self.minimum_spacing_greedy
            print('minimum spacing: ', minimum_spacing)

        # Make a copy of U and normalize each vector to avoid scale issues
        U = U.copy()
        time_array = self.time
        folder_img = f'test_{property}_{N_greedy_vecs}'

        U_normalized = normalize(U, axis=1)[1:] # Skip the first row (zero vector of ecc=0) to prevent false uniqueness due to inner product of zero vectors
        num_vectors = U.shape[0]

        greedy_basis_orthonormal = []
        greedy_basis = []
        greedy_indices = []
        greedy_errors = []
        
        # Get delta ecc_ref for application of minimum_spacing
        delta_ecc_ref =self.ecc_ref_space[1] -self.ecc_ref_space[0]
        minimum_spacing_idx = int(minimum_spacing / delta_ecc_ref)
        
        # Mask to track valid vectors (True = can be picked)
        valid_mask = np.ones(U_normalized.shape[0], dtype=bool)

        for step in range(num_vectors):
            # Stop if no more valid vectors due to minimum spacing
            if not valid_mask.any():
                print(self.colored_text(f'WARNING: break in get_greedy_params at N_greedy_vecs = {len(greedy_indices)}. No more valid vectors due to minimum spacing constraint of {minimum_spacing}.', 'red'))
                if property == 'phase':
                    self.N_basis_vecs_phase = len(greedy_indices)
                elif property == 'amplitude':
                    self.N_basis_vecs_amp = len(greedy_indices)

                break

            # Compute residuals: h - sum_i <h, e_i> e_i for all vectors h in U
            if len(greedy_basis_orthonormal) == 0:
                # First iteration: residuals are just the norms of U
                residual_norms = np.linalg.norm(U_normalized, axis=1)

            else:
                

                # Stack basis for matrix operations
                B = np.vstack(greedy_basis_orthonormal)  # shape: (m, vector_length)
                # Compute inner products <h, e_i> for all h in U
                coeffs = U_normalized @ B.T      # shape: (num_vectors, m)
                # Reconstruct projections
                U_proj = coeffs @ B              # shape: (num_vectors, vector_length)
                # Residuals
                residual_norms = np.linalg.norm(U_normalized - U_proj, axis=1)

                # Accumulated rounding error
                # orth_err = np.linalg.norm(B @ B.T - np.eye(len(B)), 'fro')
                # print("orthogonal error:", orth_err)
                
                if plot_greedy_basis_formation:
                    max_idx = np.argmax(residual_norms)

                    fig_comp_greedy, axs = plt.subplots(4, 1, figsize=(15, 15))

                    print('shapes: ', coeffs.shape, U_proj.shape, residual_norms.shape)
                    axs[0].scatter(self.ecc_ref_space[greedy_indices], np.zeros(len(greedy_indices)), label=f'current greedy indices, i={step}')
                    axs[0].scatter(self.ecc_ref_space[0], 0, c='orange', label='minimum spacing')
                    axs[0].scatter(self.ecc_ref_space[minimum_spacing_idx], 0, c='orange')
                    axs[0].plot(self.ecc_ref_space[1:], residual_norms,  label=f'step = {step}')
                    axs[0].set_xlabel('ecc')
                    axs[0].set_ylabel('residuals norm')
                    axs[0].legend()

                    axs[1].scatter(self.ecc_ref_space[greedy_indices], np.zeros(len(greedy_indices)))
                    for i in range(len(U_normalized)):
                        axs[1].plot(time_array, U_normalized.T, color='grey')
                    for j in range(step):
                        axs[1].plot(time_array, U_normalized[greedy_indices[j] - 1], color='red')
                    axs[1].plot(time_array, U_normalized[greedy_indices[-1] - 1], label='last added vec', color='blue')
                    # axs[1].plot(self.ecc_ref_space[1:], residual_norms,  label=f'step = {step}')
                    axs[1].set_ylabel('residuals diff')
                    axs[1].legend()


                    axs[2].scatter(self.ecc_ref_space[greedy_indices], np.zeros(len(greedy_indices)), label=f'current greedy indices, i={step}')
                    axs[2].plot(self.ecc_ref_space[1:], coeffs)
                    axs[2].set_ylabel('coeffs')
                    axs[2].legend()

                    colors = plt.cm.tab10.colors[:7]
                    for i,c in zip([1, 2, 3, 4, 5, 6, max_idx], colors):
                        # axs[3].plot(U_normalized[i], label="U_normalized", c=c)
                        # axs[3].plot(U_proj[i], label="U_proj", c=c)
                        axs[3].plot(U_normalized[i] - U_proj[i], label=f"residual {str(i)}", c=c)
                    axs[3].legend()
                    axs[3].set_ylabel('residuals vec')
                    axs[3].set_xlabel('time')

                    os.makedirs(f'Images/{folder_img}/', exist_ok=True)
                    fig_comp_greedy.savefig(f'Images/{folder_img}/test_greedy_{step}.png')

                    plt.close('all')



            # Apply mask: exclude already-picked vectors + minimum spacing. Set non pick to -infinity so they are never picked.
            masked_residuals = np.where(valid_mask, residual_norms, -np.inf)

            # Find the vector with the largest residual
            max_idx = np.argmax(masked_residuals)
            max_res = residual_norms[max_idx]

            # Save highest residual --> greedy error
            greedy_errors.append(round(float(max_res), 6)) 
            greedy_indices.append(int(max_idx + 1))  # +1 to account for the zero vector at index 0. int for clearer show of greedy incidces
            greedy_basis.append(U[max_idx + 1])  # Store the original vector from U

            # Add new vector to the orthonormal basis
            new_vec = U_normalized[max_idx].copy()

            # Modified Gram-Schmidt: single pass
            for b in greedy_basis_orthonormal:   # use the orthonormal vectors
                new_vec -= np.dot(new_vec, b) * b

            # second pass (re-orthogonalize to remove numerical residue)
            for b in greedy_basis_orthonormal:
                new_vec -= np.dot(new_vec, b) * b

            # Normalise vector
            norm = np.linalg.norm(new_vec)
            new_vec /= norm

            greedy_basis_orthonormal.append(new_vec)

            # Update mask to enforce minimum spacing around this index
            start = max(0, max_idx - minimum_spacing_idx)
            end = min(valid_mask.size, max_idx + minimum_spacing_idx + 1)
            valid_mask[start:end] = False

            # --- Check stopping conditions ---
            if min_greedy_error is not None and (max_res <= min_greedy_error or len(greedy_basis) == len(U)):
                break
            if N_greedy_vecs is not None and len(greedy_basis) >= N_greedy_vecs:
                break
            

        # Stack basis for convenience
        greedy_basis = np.vstack(greedy_basis)
        greedy_basis_orthonormal = np.vstack(greedy_basis_orthonormal)

        #  Plot greedy errors if requested
        if plot_greedy_error:
            self._plot_greedy_errors(greedy_errors, property, save_greedy_error_fig)

        if plot_greedy_vectors:
            # self._plot_greedy_vectors(U, greedy_basis_orthonormal, greedy_indices, property, save_greedy_vecs_fig)
            self._plot_greedy_vectors(U=U, greedy_basis=greedy_basis, greedy_parameters_idx=greedy_indices, property=property, save_greedy_vecs_fig=save_greedy_vecs_fig)

        print(f'Highest error of best approximation of the basis: {round(np.min(greedy_errors), 5)} | {len(greedy_basis)} basis vectors')
        print(greedy_indices, greedy_errors)

        return greedy_indices, greedy_basis_orthonormal


    def _plot_greedy_vectors_old(self, greedy_basis, greedy_parameters_idx, property, save_greedy_vecs_fig, U=None):
        """Function to plot and option to save the greedy basis vectors. If U is specified, also plot complete dataset before greedy algorithm."""

        fig_greedy_vecs, (ax_main, ax_bottom) = plt.subplots(
            2, 1, figsize=(12, 6),
            gridspec_kw={'height_ratios': [4, 0.5]}
        )

        # --- Top plot: dataset and greedy basis vectors ---
        if U is not None:
            for i, vec in enumerate(U):
                ax_main.plot(vec, color='grey', alpha=0.3, label='Vector dataset' if i == 0 else None)

        for i, vec in enumerate(greedy_basis):
            ax_main.plot(vec, color='red', linewidth=0.6,
                        label=f'{elf.ecc_ref_space[greedy_parameters_idx[i]]}')

        

        ax_main.set_ylabel('Vector Value')
        ax_main.set_title(f'Greedy Basis Vectors ({len(greedy_basis)} vectors) for {property}')
        ax_main.legend()
        ax_main.grid(True)

        # --- Bottom plot: eccentric points and greedy indices ---
        ecc_values =self.ecc_ref_space # assuming first component = eccentricity
        y = np.zeros(len(self.ecc_ref_space))

        # All dataset points
        ax_bottom.plot(ecc_values, y, color='grey', alpha=0.6, label='Dataset points')

        # Greedy-selected points
        ax_bottom.scatter(ecc_values[greedy_parameters_idx], y[greedy_parameters_idx], color='red', s=50, label='Greedy parameters')

        ax_bottom.set_yticks([])
        ax_bottom.set_xlabel('Vector index / parameter')
        ax_bottom.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax_main.legend(loc='best', ncol=3, fontsize='small')


        plt.tight_layout()
        fig_greedy_vecs.show()


        if save_greedy_vecs_fig:
            os.makedirs('Images/Greedy_vectors', exist_ok=True)
            plt.savefig(f'Images/Greedy_vectors/Greedy_vectors_{property}_M={self.total_mass}_ecc=[{min(self.ecc_ref_space)}_{max(self.ecc_ref_space)}]_f_lower={self.f_lower}_f_ref={self.f_ref}_iN={len(self.ecc_ref_space)}_ms={self.minimum_spacing_greedy}.png')
            print('Greedy vectors fig saved to Images/Greedy_vectors')
            # plt.close('all')

    def _plot_greedy_errors_old(self, greedy_errors, property, save_greedy_fig):
        """Function to plot and option to save the greedy errors."""

        N_basis_vectors = np.arange(1, len(greedy_errors) + 1)

        fig_greedy_errors = plt.figure(figsize=(7, 5))
        plt.plot(N_basis_vectors, greedy_errors, label='Greedy Errors')
        plt.scatter(N_basis_vectors, greedy_errors, s=4)
        # for i, label in enumerate(self.ecc_ref_space[greedy_parameters_idx]):
        #     plt.annotate(label, (N_basis_vectors[i], greedy_errors[i]), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=5.5)
        plt.xlabel('Number of Waveforms')
        if property == 'phase':
            plt.ylabel(f'Greedy error $\Delta \phi$')
        elif property == 'amplitude':
            plt.ylabel(f'Greedy error $\Delta A$')
        plt.yscale('log')
        # plt.title('Greedy errors of residual {} for N = {}'.format(property, len(greedy_errors)-1))
        plt.grid(True)
        fig_greedy_errors.show()

        if save_greedy_fig:
            os.makedirs('Images/Greedy_errors', exist_ok=True)
            plt.savefig(f'Images/Greedy_errors/Greedy_error_{property}_M={self.total_mass}_ecc=[{min(self.ecc_ref_space)}_{max(self.ecc_ref_space)}]_f_lower={self.f_lower}_f_ref={self.f_ref}_iN={len(self.ecc_ref_space)}_gerr={min(greedy_errors)}_ms={self.minimum_spacing_greedy}.png')
            
            print('Greedy error fig saved to Images/Greedy_errors')
            # plt.close('all')
        

    def _plot_validation_errors_old(self, validation_vecs, greedy_basis, trivial_basis, property, save_validation_fig):
        """Function to plot and option to save validation errors."""
        
        def compute_proj_errors(basis, V, reg=1e-6):
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
            plt.savefig(f'Images/Validation_errors/Validation_error_{property}_M={self.total_mass}_ecc=[{min(self.ecc_ref_space)}_{max(self.ecc_ref_space)}]_f_lower={self.f_lower}_f_ref={self.f_ref}_iN={len(self.ecc_ref_space)}.png')
        
        # plt.close('all')

    def get_empirical_nodes_old(self, reduced_basis, property, plot_emp_nodes_at_ecc=True, save_fig=True):
        """
        Perform the Empirical Interpolation Method (EIM).
        
        Parameters
        ----------
        reduced_basis : ndarray, shape (m, L)
            The reduced basis vectors (m basis functions, each of length L).
        grid_points : ndarray, shape (L,)
            The discrete grid points (t_l values).

        Returns
        -------
        emp_nodes_idx : list of int
            Indices of empirical nodes in the grid.
        emp_nodes : list of float
            The actual grid point locations.
        """
        m, L = reduced_basis.shape
        emp_nodes_idx = []
        emp_nodes = []

        U, S, VT = np.linalg.svd(reduced_basis)
        plt.semilogy(S, 'o-')
        plt.title("Singular Values of Reduced Basis")
        plt.show()

        # Step 1 — first node: pick the max abs value from first basis vector
        i = np.argmax(np.abs(reduced_basis[0]))
        emp_nodes_idx.append(i)

        # Loop for j = 2 ... m
        for j in range(1, m):
            # Build V matrix (j-1 x j-1) from previous basis vectors at previous nodes
            V = reduced_basis[:j, emp_nodes_idx]
            print(f"Iteration {j}:")
            print(f"Rank of V:", np.linalg.matrix_rank(V), ' should be smaller than {j}')
            print("Condition number of V:", np.linalg.cond(V), ' should be > 1e10')
            # print('V: ', V)
            # # Solve for coefficients that interpolate e_j at previous nodes
            # coeffs = np.linalg.solve(V, reduced_basis[j, emp_nodes_idx])

            coeffs = np.linalg.pinv(V) @ reduced_basis[j, emp_nodes_idx]
            # Build interpolant on whole grid
            interpolant = np.dot(coeffs, reduced_basis[:j])

            # Compute residual
            residual = reduced_basis[j] - interpolant

            # Pick next node as location of max abs residual
            i = np.argmax(np.abs(residual))
            emp_nodes_idx.append(i)

        emp_nodes = self.time[emp_nodes_idx]
        # print(emp_nodes_idx, emp_nodes)

        # Optional: Plot the empirical nodes if plot_emp_nodes_at_ecc is set
        if plot_emp_nodes_at_ecc:
            self._plot_empirical_nodes(emp_nodes_idx, property, plot_emp_nodes_at_ecc, save_fig)

        return emp_nodes_idx, emp_nodes
    
    def get_empirical_nodes_old2(self, reduced_basis, property, plot_emp_nodes_at_ecc=True, save_fig=True):
        """
        Reduced basis needs to be orthonormal!
        """
        
        m, L = reduced_basis.shape

        emp_nodes_idx = []
        emp_nodes = []
        time_grid = self.time  # Assuming this is your grid points

        # # Initial SVD check
        # U, S, VT = np.linalg.svd(reduced_basis)
        # plt.semilogy(S, 'o-')
        # plt.title("Singular Values of Reduced Basis")
        # plt.show()

        # First node selection
        i = np.argmax(np.abs(reduced_basis[0]))
        emp_nodes_idx.append(int(i))

        for j in range(1, reduced_basis.shape[0]):
            V = reduced_basis[:j, emp_nodes_idx]
            coeffs = np.linalg.pinv(V) @ reduced_basis[j, emp_nodes_idx]
            interpolant = np.dot(coeffs, reduced_basis[:j])
            residual = reduced_basis[j] - interpolant

            # # --- Enhanced Residual Visualization ---
            # plt.figure(figsize=(15, 10))
            
            # # Plot 1: Current basis vector being approximated
            # plt.subplot(3, 1, 1)
            # plt.plot(time_grid, reduced_basis[j], 'b-', label=f'Basis Vector {j}')
            # plt.scatter(time_grid[emp_nodes_idx], reduced_basis[j, emp_nodes_idx], 
            #         c='blue', marker='o', label='Node Values')
            # plt.title(f'Target Basis Vector {j} to Approximate')
            # plt.legend()
            
            # Plot 2: Interpolant construction
            # plt.subplot(3, 1, 2)
            # for k in range(j):
            #     plt.plot(time_grid, coeffs[k] * reduced_basis[k], '--', alpha=0.5, 
            #             label=f'{coeffs[k]:.2f}×Basis{k}')
            # plt.plot(time_grid, interpolant, 'r-', linewidth=2, label='Interpolant Sum')
            # plt.scatter(time_grid[emp_nodes_idx], interpolant[emp_nodes_idx], 
            #         c='red', marker='x', label='Interpolant at Nodes')
            # plt.title('Interpolant Construction (Weighted Sum of Previous Basis)')
            # plt.legend()
            
            # # Plot 3: Residual calculation
            # plt.subplot(3, 1, 3)
            # plt.plot(time_grid, reduced_basis[j], 'b-', label='Original Vector')
            # plt.plot(time_grid, interpolant, 'r-', label='Interpolant')
            # plt.plot(time_grid, residual, 'g-', label='Residual')
            # plt.scatter(time_grid[emp_nodes_idx], np.zeros_like(emp_nodes_idx),
            #         c='black', marker='x', label='Existing Nodes')
            # new_node = np.argmax(np.abs(residual))
            # plt.scatter(time_grid[new_node], residual[new_node], 
            #         c='magenta', s=100, label='New Node Candidate')
            # plt.title(f'Residual Calculation (Max at {time_grid[new_node]:.2f})')
            # plt.legend()
            
            # plt.tight_layout()
            # plt.show()

            i = np.argmax(np.abs(residual))
            emp_nodes_idx.append(int(i))

        # --- Final Plots ---
        emp_nodes = self.time[emp_nodes_idx]
        print(emp_nodes_idx, emp_nodes)
        
        # Plot 5: All Basis Vectors with Final Nodes
        plt.figure(figsize=(12, 6))
        # for k in range(m):
            # plt.plot(self.time, reduced_basis[k], alpha=0.5, label=f"Basis {k}" if k < 5 else None)
        plt.scatter(emp_nodes, np.zeros_like(emp_nodes), c='red', marker='x', s=100, label="EIM Nodes")
        plt.title(f"Final Basis Vectors and Selected Nodes for {property}")
        plt.xlabel("Time")
        plt.ylabel("Basis Value")
        plt.legend(ncol=2)
        plt.grid(True)
        plt.show()

        # Plot 6: Node Distribution Histogram
        # if len(emp_nodes_idx) > 1:
            # distances = np.diff(np.sort(emp_nodes_idx))
            # plt.figure(figsize=(8, 4))
            # plt.hist(distances, bins=20)
            # plt.title("Distance Between Consecutive Nodes")
            # plt.xlabel("Grid Points")
            # plt.ylabel("Frequency")
            # plt.show()

        if plot_emp_nodes_at_ecc:
            self._plot_empirical_nodes(emp_nodes_idx, property, plot_emp_nodes_at_ecc, save_fig)

        return emp_nodes_idx

    def gram_schmidt_rows(self, V, tol=1e-12, verbose=True):
        """
        Orthonormalize a set of row vectors using a numerically stable 
        Modified Gram–Schmidt process with reorthogonalization.

        Parameters
        ----------
        V : np.ndarray
            Shape (n_vec, n_dim). Each row is a vector to orthonormalize.
        tol : float, optional
            Threshold below which a vector is considered linearly dependent and skipped.
        verbose : bool, optional
            If True, prints diagnostic messages for dependent rows.

        Returns
        -------
        Q : np.ndarray
            Orthonormalized rows (shape (n_valid_vecs, n_dim)).
        """

        V = np.asarray(V, dtype=float).copy()
        n_vec, n_dim = V.shape
        Q_list = []

        for i in range(n_vec):
            v = V[i].copy()

            # First orthogonalization pass
            for q in Q_list:
                v -= np.dot(q, v) * q

            # Second orthogonalization pass (reorthogonalization)
            for q in Q_list:
                v -= np.dot(q, v) * q

            norm = np.linalg.norm(v)

            if norm < tol:
                if verbose:
                    print(f"⚠️  Skipping row {i} (norm={norm:.2e}) — linearly dependent or near-zero.")
                continue  # skip nearly dependent vector

            Q_list.append(v / norm)

        if not Q_list:
            raise ValueError("All input vectors were linearly dependent or zero.")

        Q = np.vstack(Q_list)
        return Q

    
    def empirical_interpolation_from_dataset(self, waveforms_dataset, property, plot_emp_nodes_at_ecc=True, save_fig=True):
        """
        Compute empirical interpolation nodes directly from a dataset (not from a reduced basis).

        Parameters
        ----------
        property_matrix : np.ndarray
            2D array with shape (num_waveforms, num_points).
            Each row is a waveform (or function sampled at discrete points).
        N_nodes : int
            Desired number of empirical nodes to select.

        Returns
        -------
        emp_nodes_idx : list
            List of indices of selected empirical nodes.
        """

        if property == 'phase':
            N_nodes = self.N_basis_vecs_phase
        if property == 'amplitude':
            N_nodes = self.N_basis_vecs_amp

        
        def calc_empirical_interpolant(property_array, reduced_basis, emp_nodes_idx):
            """
            Calculates the empirical interpolant for a specific waveform property using a reduced basis.
            
            Parameters:
            ----------------
            - property_array (numpy.ndarray): The waveform property values (e.g., phase or amplitude).
            - reduced_basis (numpy.ndarray): Reduced basis of residual properties.
            - emp_nodes_idx (list): Indices of empirical nodes.

            Returns:
            ----------------
            - empirical_interpolant (numpy.ndarray): The computed empirical interpolant of the waveform property.
            """
            
            empirical_interpolant = np.zeros_like(property_array)
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
                empirical_interpolant += B_j_vec[:, j] * property_array[emp_nodes_idx[j]]

            return empirical_interpolant
        
        reduced_basis = self.gram_schmidt_rows(waveforms_dataset)
        
        # Initialize with the index of the maximum value in the first basis vector
        i = np.argmax(reduced_basis[0])
        emp_nodes_idx = [i]
        EI_error = []

        # Loop through the reduced basis to calculate interpolants
        for j in range(1, N_nodes):
            empirical_interpolant = calc_empirical_interpolant(reduced_basis[j], reduced_basis[:j], emp_nodes_idx)
            residuals = empirical_interpolant - reduced_basis[j][:, np.newaxis].T
            EI_error.append(np.linalg.norm(residuals))

            # Identify the next empirical node based on the maximum residual
            next_idx = np.argmax(np.abs(residuals))
            emp_nodes_idx.append(next_idx)

        #     # Inside the loop
        #     fig_residuals = plt.figure(figsize=(8, 4))
        #     plt.plot(np.abs(residuals).flatten(), label=f"Step {j}")
        #     plt.axvline(next_idx, color='r', linestyle='--', label=f"New node {next_idx}")
        #     plt.title(f"Residual at Step {j}")
        #     plt.xlabel("Sample index")
        #     plt.ylabel("|Residual|")
        #     plt.legend()
        #     plt.tight_layout()
        #     fig_residuals.savefig(f'Images/Empirical_nodes/test_residuals_{j}.png')

        # fig_error = plt.figure(figsize=(6, 4))
        # plt.semilogy(EI_error, marker='o')
        # plt.title("Empirical Interpolation Error Convergence")
        # plt.xlabel("Iteration")
        # plt.ylabel("‖Residual‖₂")
        # plt.grid(True, which='both', ls='--')
        # fig_error.savefig('Images/Empirical_nodes/test_error.png')

        # # Example: Compare the 3rd waveform’s true vs interpolated
        # j = 2
        # wf_true = reduced_basis[j]
        # wf_interp = calc_empirical_interpolant(wf_true, reduced_basis[:j], emp_nodes_idx[:j])

        # fig_compare = plt.figure(figsize=(8, 4))
        # plt.plot(wf_true, label="True waveform", lw=2)
        # plt.plot(wf_interp, '--', label="Interpolated", lw=2)
        # plt.scatter(emp_nodes_idx[:j], wf_true[emp_nodes_idx[:j]], color='red', label="Empirical nodes")
        # plt.title(f"Empirical Interpolation at Step {j}")
        # plt.legend()
        # plt.tight_layout()
        # fig_compare.savefig('Images/Empirical_nodes/test_compare.png')


        # Optional: Plot the empirical nodes if plot_emp_nodes_at_ecc is set
        if plot_emp_nodes_at_ecc:
            self._plot_empirical_nodes_from_dataset(emp_nodes_idx, property, plot_emp_nodes_at_ecc, save_fig)
        
        return emp_nodes_idx

    def get_empirical_nodes_test(self, reduced_basis, property, plot_emp_nodes_at_ecc=True, save_fig=True):
        """
        Calculate the empirical nodes for a given dataset based on a reduced basis of residual properties.

        Parameters:
        ----------------
        - reduced_basis (numpy.ndarray): Reduced basis of residual properties (phase or amplitude).
        - property (str): Waveform property to evaluate, options are "phase" or "amplitude".
        - plot_emp_nodes_at_ecc (float, optional): If set, plots the empirical nodes at a specified eccentricity value.
        - save_fig (bool, optional): Saves the empirical nodes plot if set to True.

        Returns:
        ----------------
        - emp_nodes_idx (list): Indices of empirical nodes for the given dataset.
        """


        def calc_empirical_interpolant(property_array, reduced_basis, emp_nodes_idx):
            """
            Compute empirical interpolant for a waveform property using reduced basis.
            """
            m = len(emp_nodes_idx)
            empirical_interpolant = np.zeros_like(property_array)

            m = len(emp_nodes_idx)
            V = reduced_basis[:m, emp_nodes_idx].T
            rhs = property_array[emp_nodes_idx]


            # Solve for coefficients (α)
            try:
                alpha = np.linalg.solve(V, rhs)
            except np.linalg.LinAlgError:
                alpha = np.linalg.pinv(V) @ rhs

            # Compute interpolant
            empirical_interpolant = np.sum(alpha[:, None] * reduced_basis[:m, :], axis=0)

            return empirical_interpolant
        
        if property == 'phase':
            N_nodes = self.N_basis_vecs_phase
        if property == 'amplitude':
            N_nodes = self.N_basis_vecs_amp

        if reduced_basis.shape[0] < reduced_basis.shape[1]:
            # assume rows = functions (waveforms)
            reduced_basis = reduced_basis.T

        # Each row = a basis vector
        reduced_basis = np.array(reduced_basis, dtype=float)
        reduced_basis /= np.linalg.norm(reduced_basis, axis=1)[:, None]

        
        # Initialize
        i0 = np.argmax(np.abs(reduced_basis[0]))
        emp_nodes_idx = [i0]
        EI_error = []

        for j in range(1, N_nodes):
            property_array = reduced_basis[j]
            empirical_interpolant = calc_empirical_interpolant(property_array, reduced_basis[:j], emp_nodes_idx)   


            residuals = property_array - empirical_interpolant
            print(f'EI error: {np.linalg.norm(residuals)}')
            EI_error.append(np.linalg.norm(residuals))

            next_idx = np.argmax(np.abs(residuals))
            emp_nodes_idx.append(next_idx)


            # Inside the loop
            if j%100 == 0:
                fig_residuals = plt.figure(figsize=(8, 4))
                plt.plot(np.abs(residuals).flatten(), label=f"Step {j}")
                plt.axvline(next_idx, color='r', linestyle='--', label=f"New node {next_idx}")
                plt.title(f"Residual at Step {j}")
                plt.xlabel("Sample index")
                plt.ylabel("|Residual|")
                plt.legend()
                plt.tight_layout()
                fig_residuals.savefig(f'Images/Empirical_nodes/test_residuals_{j}.png')

                plt.close('all')

        fig_error = plt.figure(figsize=(6, 4))
        plt.semilogy(EI_error, marker='o')
        plt.title("Empirical Interpolation Error Convergence")
        plt.xlabel("Iteration")
        plt.ylabel("‖Residual‖₂")
        plt.grid(True, which='both', ls='--')
        fig_error.savefig('Images/Empirical_nodes/test_error.png')

        # Example: Compare the 3rd waveform’s true vs interpolated
        j = 2
        wf_true = reduced_basis[j]
        wf_interp = calc_empirical_interpolant(wf_true, reduced_basis[:j], emp_nodes_idx[:j])

        fig_compare = plt.figure(figsize=(8, 4))
        plt.plot(wf_true, label="True waveform", lw=2)
        plt.plot(wf_interp, '--', label="Interpolated", lw=2)
        plt.scatter(emp_nodes_idx[:j], wf_true[emp_nodes_idx[:j]], color='red', label="Empirical nodes")
        plt.title(f"Empirical Interpolation at Step {j}")
        plt.legend()
        plt.tight_layout()
        fig_compare.savefig('Images/Empirical_nodes/test_compare.png')

        # Optional: Plot the empirical nodes if plot_emp_nodes_at_ecc is set
        if plot_emp_nodes_at_ecc:
            self._plot_empirical_nodes(emp_nodes_idx, property, plot_emp_nodes_at_ecc, save_fig)
        
        return emp_nodes_idx


    def _compute_empirical_nodes(self, orthonormal_basis):
        """
        Compute empirical interpolation nodes for a single global reduced basis.

        Parameters
        ----------
        orthonormal_basis : np.ndarray
            Array of shape (n_basis, n_samples), where each row is one basis vector.
            This should be orthonormalized already.

        Returns
        -------
        empirical_nodes : np.ndarray
            Indices of the empirical interpolation nodes.

        If return_interpolant=True, also returns
        ----------------------------------------
        interpolant : np.ndarray
            Interpolant matrix of shape (n_samples, n_basis).

        """
        basis = np.asarray(orthonormal_basis)

        if basis.ndim != 2:
            raise ValueError("orthonormal_basis must have shape (n_basis, n_samples)")

        n_basis, _ = basis.shape
        if n_basis == 0:
            raise ValueError("orthonormal_basis must contain at least one basis vector")

        empirical_nodes = []

        # First node
        first_node = int(np.argmax(np.abs(basis[0])))
        empirical_nodes.append(first_node)

        # Greedy EIM node selection
        for i in range(1, n_basis):
            prev_basis = basis[:i]  # shape: (i, n_samples)
            node_idx = np.array(empirical_nodes, dtype=int)

            # Matrix of previous basis vectors evaluated at previous nodes
            A = prev_basis[:, node_idx].T  # shape: (i, i)

            # Current basis vector evaluated at previous nodes
            b = basis[i, node_idx]         # shape: (i,)

            # Interpolate current basis vector from previous ones
            coeffs = np.linalg.solve(A, b)
            interpolant = coeffs @ prev_basis
            residual = basis[i] - interpolant

            new_node = int(np.argmax(np.abs(residual)))
            empirical_nodes.append(new_node)

        empirical_nodes = np.asarray(empirical_nodes, dtype=int)

        return empirical_nodes, interpolant


    def get_empirical_nodes(self, reduced_basis_object: ReducedBasis, train_obj: TrainingSetParameters, plot_emp_nodes_on_basis=False, save_emp_nodes_on_basis_fig=False, plot_interpolation_matrix=False, save_interpolation_matrix_fig=False, plot_proj_vs_eim_error=False, save_proj_vs_eim_error_fig=False, eim_per_leaf=True):
        """
        Calculate the empirical nodes for a given dataset based on a reduced basis of residual properties.

        Parameters:
        ----------------
        - reduced_basis (numpy.ndarray): Reduced basis of residual properties (phase or amplitude).
        - property (str): Waveform property to evaluate, options are "phase" or "amplitude".
        - plot_emp_nodes_at_ecc (float, optional): If set, plots the empirical nodes at a specified eccentricity value.
        - save_fig (bool, optional): Saves the empirical nodes plot if set to True.

        Returns:
        ----------------
        - emp_nodes_idx (list): Indices of empirical nodes for the given dataset.
        """

        # if eim_per_leaf:

        # Get empirical nodes for each leaf (greedy parameters section) in the tree
        eim = EmpiricalInterpolation(reduced_basis_object)
        eim.fit()
        # Stack empirical nodes from all leaves into the training object
        for i, leaf in enumerate(reduced_basis_object.tree.leaves):
            # print(f"Leaf {i} | Empirical nodes: {leaf.empirical_nodes}")
            train_obj.empirical_indices.extend(leaf.empirical_nodes)

        # else:
        #     # Orthonormalize the full (combined tree) reduced basis
        #     global_basis = np.vstack([leaf.basis for leaf in reduced_basis_object.tree.leaves])
        #     global_basis_ortho = orth(global_basis.T).T
        #     # Create new ReducedBasis object to store with orthonormalized global basis
        #     empirical_indices, interpolant = self._compute_empirical_nodes(global_basis_ortho)
        #     train_obj.empirical_indices = empirical_indices


        # print('Empirical nodes:', emp_nodes_idx,
        #     '\nlength of empirical nodes:', len(emp_nodes_idx),
        #     '\nlength of reduced basis:', len(reduced_basis_object.tree.leaves[i].indices)) # .indices refers to the indices of the greedy basis vectors
        

        # Optional: plot empirical nodes at eccentricity
        if plot_interpolation_matrix:
            self._plot_interpolation_matrix(reduced_basis_object, save_fig=save_interpolation_matrix_fig)

        if plot_proj_vs_eim_error:
            self._plot_projection_vs_eim_error(reduced_basis_object=reduced_basis_object, train_obj=train_obj, save_fig=save_proj_vs_eim_error_fig)

        if plot_emp_nodes_on_basis:
            self._plot_emp_nodes_on_basis(reduced_basis_object, train_obj, save_fig=save_emp_nodes_on_basis_fig)
        
        # if plot_emp_nodes_at_ecc:
        #     self._plot_empirical_nodes(emp_nodes_idx, property, plot_emp_nodes_at_ecc, save_fig)

        return train_obj.empirical_indices


    def _plot_emp_nodes_on_basis(self, reduced_basis_object : ReducedBasis, train_obj: TrainingSetParameters, save_fig=False):
        """  
        Plot the empirical nodes on top of the reduced basis functions for each leaf in the tree.
        """
        for i, leaf in enumerate(reduced_basis_object.tree.leaves):
            reduced_basis = leaf.basis # ORTHONORMALIZED basis functions
            eim_nodes = leaf.empirical_nodes

            fig_emp_nodes, ax = plt.subplots(figsize=(20,5))

            # Plot basis functions
            for j in range(len(eim_nodes)):
                ax.plot(
                    self.time,
                    reduced_basis[j],
                    alpha=0.7,
                    lw=1.5
                )

                # Mark node for that basis function
                ax.scatter(
                    self.time[eim_nodes[j]],
                    reduced_basis[j, eim_nodes[j]],
                    color="red",
                    zorder=5
                )

            # Vertical lines for all nodes
            for node in eim_nodes:
                ax.axvline(self.time[node], color='red', alpha=0.2)

            ax.set_ylabel("Basis function amplitude")
            ax.set_xlabel("time [M]")
            ax.set_title(f"Reduced basis functions and empirical interpolation nodes: leaf {i}")

            fig_emp_nodes.tight_layout()

            if save_fig:
                figname = train_obj.figname(prefix=f"RB_functions_with_emp_nodes_leaf_{i}", directory="Images/Empirical_nodes")
                # fig_path = f'Images/Empirical_nodes/RB_functions_with_emp_nodes_leaf_{i}_e=[{min(self.ecc_ref_space)}, {max(self.ecc_ref_space)}, N={len(self.ecc_ref_space)}]_f_lower={self.f_lower}_f_ref={self.f_ref}_gN={len(self.indices_basis)}.png'
                # os.makedirs(os.path.dirname(figname), exist_ok=True)
                fig_emp_nodes.savefig(figname)
                # self.colored_text(f'Figure is saved in {figname}', "blue")




    # def _plot_interpolation_matrix(self, reduced_basis_object, save_fig=False):
    #     print(len(reduced_basis_object.tree.leaves))
    #     fig, ax = plt.subplots(len(reduced_basis_object.tree.leaves), 1, figsize=(6,5))

    #     for i in range(len(reduced_basis_object.tree.leaves)):
    #         reduced_basis = reduced_basis_object.tree.leaves[i].basis # ORTHONORMALIZED basis functions 
    #         eim_nodes = reduced_basis_object.tree.leaves[i].empirical_nodes # indices of empirical nodes

    #         # Interpolation matrix: V_{ij} = φ_i(t_j)
    #         V = reduced_basis[:, eim_nodes]

    #         im = ax.imshow(V, aspect='auto', origin='lower')

    #         ax[0].set_title("EIM interpolation matrix")
    #         ax[i].set_xlabel("Empirical node index")
    #         ax[i].set_ylabel("Basis index")

    #     fig.colorbar(im, ax=ax)

    #     plt.tight_layout()

    #     if save_fig:
    #         fig_path = f'Images/Empirical_nodes/EIM_interpolation_matrix_M={self.total_mass}_f_lower={self.f_lower}_f_ref={self.f_ref}_iN={len(self.ecc_ref_space)}_gN={len(self.indices_basis)}_ms={self.minimum_spacing_greedy}.png'
    #         os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    #         fig.savefig(fig_path)
    #         print(f'Figure is saved in {fig_path}')

    #     return V

    def _plot_interpolation_matrix(self, reduced_basis_object, save_fig=False):
        n_leaves = len(reduced_basis_object.tree.leaves)

        fig, ax = plt.subplots(n_leaves, 1, squeeze=False)
        ax = ax.ravel()

        for i, leaf in enumerate(reduced_basis_object.tree.leaves):
            reduced_basis = leaf.basis
            eim_nodes = leaf.empirical_nodes

            # Assuming basis shape = (n_basis, n_samples)
            V = reduced_basis[:, eim_nodes]

            im = ax[i].imshow(V, aspect='auto', origin='lower')
            ax[i].set_title(f"EIM interpolation matrix (leaf {i})")
            ax[i].set_xlabel("Empirical node index")
            ax[i].set_ylabel("Basis index")

        fig.colorbar(im, ax=ax)

        plt.tight_layout()

        if save_fig:
            fig_path = (
                f"Images/Empirical_nodes/Interpolation_matrix/"
                f"EIM_interpolation_matrix_M={self.total_mass}"
                f"_f_lower={self.f_lower}"
                f"_f_ref={self.f_ref}"
                f"_iN={len(self.ecc_ref_space)}"
                f"_gN={len(self.indices_basis)}"
                f"_ms={self.minimum_spacing_greedy}.png"
            )
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            fig.savefig(fig_path)
            print(self.colored_text(f"Figure is saved in {fig_path}", "blue"))

    # def _plot_projection_vs_eim_error(self, reduced_basis_object, dataset, property=None, save_fig=False):
    #     """
    #     Compare best projection error and EIM interpolation error
    #     on the same dataset, leaf by leaf.

    #     Parameters
    #     ----------
    #     reduced_basis_object : ReducedBasis
    #         Object containing leaf bases and empirical nodes.
    #     dataset : ndarray, shape (num_vectors, vector_length)
    #         Dataset used to test the approximation errors.
    #     property : str, optional
    #         'phase' or 'amplitude', only used for labeling/saving.
    #     save_fig : bool, optional
    #         If True, save figure.

    #     Returns
    #     -------
    #     all_proj_errors : list of lists
    #         Projection errors per leaf.
    #     all_eim_errors : list of lists
    #         EIM errors per leaf.
    #     """
    #     self.property_warning(property)

    #     n_leaves = len(reduced_basis_object.tree.leaves)
    #     fig, axs = plt.subplots(n_leaves, 1, figsize=(8, 4 * n_leaves))
    #     axs = np.atleast_1d(axs)

    #     all_proj_errors = []
    #     all_eim_errors = []

    #     for i, leaf in enumerate(reduced_basis_object.tree.leaves):
    #         reduced_basis = np.asarray(leaf.basis)              # (n_basis, n_time)
    #         eim_nodes = np.asarray(leaf.empirical_nodes)
    #         print(np.asarray(dataset), dataset.shape)
    #         leaf_dataset = np.asarray(dataset)[leaf.indices]    # test only on this leaf's data

    #         proj_errors = []
    #         eim_errors = []

    #         for k in range(1, len(eim_nodes) + 1):
    #             basis_k = reduced_basis[:k]                     # (k, n_time)
    #             nodes_k = eim_nodes[:k]

    #             # Interpolation matrix
    #             V_k = basis_k[:, nodes_k]                       # (k, k)

    #             max_proj_err = 0.0
    #             max_eim_err = 0.0

    #             for f in leaf_dataset:
    #                 norm_f = np.linalg.norm(f)
    #                 if norm_f == 0:
    #                     continue

    #                 # Best projection
    #                 coeff_proj, *_ = np.linalg.lstsq(basis_k.T, f, rcond=None)
    #                 f_proj = basis_k.T @ coeff_proj
    #                 proj_err = np.linalg.norm(f - f_proj) / norm_f
    #                 max_proj_err = max(max_proj_err, proj_err)

    #                 # EIM approximation
    #                 f_nodes = f[nodes_k]
    #                 coeff_eim = np.linalg.solve(V_k.T, f_nodes)
    #                 f_eim = basis_k.T @ coeff_eim
    #                 eim_err = np.linalg.norm(f - f_eim) / norm_f
    #                 max_eim_err = max(max_eim_err, eim_err)

    #             proj_errors.append(max_proj_err)
    #             eim_errors.append(max_eim_err)

    #         all_proj_errors.append(proj_errors)
    #         all_eim_errors.append(eim_errors)

    #         axs[i].semilogy(range(1, len(eim_nodes)), proj_errors[:-1], marker='o', label='Projection error')
    #         axs[i].semilogy(range(1, len(eim_nodes)), eim_errors[:-1], marker='s', label='EIM error')
    #         axs[i].set_xlabel("Number of modes / EIM nodes")
    #         axs[i].set_ylabel("Max relative error")
    #         axs[i].set_title(f"Leaf {i} (last dataset point has been ignored for better visualization)")
    #         axs[i].grid(True)
    #         axs[i].legend()

    #     fig.suptitle("Projection error vs EIM error", y=1.02)
    #     fig.tight_layout()

    #     if save_fig:
    #         fig_path = (
    #             f'Images/Empirical_nodes/Errors/'
    #             f'EIM_projection_vs_eim_error_{property}_'
    #             f'M={self.total_mass}_f_lower={self.f_lower}_f_ref={self.f_ref}_'
    #             f'iN={len(self.ecc_ref_space)}_'
    #             f'gN={len(self.indices_basis)}_'
    #             f'ms={self.minimum_spacing_greedy}.png'
    #         )
    #         os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    #         fig.savefig(fig_path)
    #         print(self.colored_text(f'Figure is saved in {fig_path}', 'blue'))

    #     return all_proj_errors, all_eim_errors

    def _plot_projection_vs_eim_error(self, reduced_basis_object, train_obj, save_fig=False):
        self.property_warning(property)

        dataset = np.asarray(train_obj.residuals)

        all_proj_errors = []
        all_eim_errors = []
        all_ratios = []

        for i, leaf in enumerate(reduced_basis_object.tree.leaves):
            reduced_basis = np.asarray(leaf.basis)          # (n_basis, n_time)
            eim_nodes = np.asarray(leaf.empirical_nodes)
            leaf_dataset = dataset[leaf.indices]            # data belonging to this leaf

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
            axs[0].set_title(f"Projection error vs EIM error — Leaf {i}")
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
                fig_path = (
                    f'Images/Empirical_nodes/Errors/'
                    f'EIM_projection_vs_eim_error_leaf={i}_{property}_'
                    f'M={self.total_mass}_f_lower={self.f_lower}_f_ref={self.f_ref}_'
                    f'iN={len(self.ecc_ref_space)}_'
                    f'gN={len(self.indices_basis)}_'
                    f'ms={self.minimum_spacing_greedy}.png'
                )
                os.makedirs(os.path.dirname(fig_path), exist_ok=True)
                fig.savefig(fig_path)
                print(self.colored_text(f'Figure for leaf {i} is saved in {fig_path}', 'blue'))

        return all_proj_errors, all_eim_errors, all_ratios

    # def _plot_projection_vs_eim_error(self, reduced_basis_object, save_fig=False):
    #     fig_proj_vs_eim_error, axs = plt.subplots(len(reduced_basis_object.tree.leaves), 1)
    #     axs = np.atleast_1d(axs) # In case there's only one leaf, ensure axs is always an array for consistent indexing

    #     for i in range(len(reduced_basis_object.tree.leaves)):
    #         reduced_basis = reduced_basis_object.tree.leaves[i].basis # ORTHONORMALIZED basis functions 
    #         eim_nodes = reduced_basis_object.tree.leaves[i].empirical_nodes

    #         proj_errors = []
    #         eim_errors = []

    #         for k in range(1, len(eim_nodes) + 1):
    #             basis_k = reduced_basis[:k]                        # (k, n_time)
    #             nodes_k = eim_nodes[:k]

    #             # EIM interpolation operator, matching your implementation
    #             V_k = np.array([[basis_k[m, t] for t in nodes_k] for m in range(k)])
    #             invVt_k = np.linalg.inv(V_k.T)
    #             interpolant_k = basis_k.T @ invVt_k              # (n_time, k)

    #             max_proj_err = 0.0
    #             max_eim_err = 0.0

    #             for f in reduced_basis:  # Loop over all basis functions as test functions
    #                 # --- Best projection onto span(basis_k) ---
    #                 # Solve min ||f - c @ basis_k||
    #                 coeff_proj, *_ = np.linalg.lstsq(basis_k.T, f, rcond=None)
    #                 f_proj = basis_k.T @ coeff_proj

    #                 proj_err = np.linalg.norm(f - f_proj) / np.linalg.norm(f)
    #                 max_proj_err = max(max_proj_err, proj_err)

    #                 # --- EIM reconstruction from node values only ---
    #                 f_nodes = f[nodes_k]
    #                 f_eim = interpolant_k @ f_nodes

    #                 eim_err = np.linalg.norm(f - f_eim) / np.linalg.norm(f)
    #                 max_eim_err = max(max_eim_err, eim_err)

    #             proj_errors.append(max_proj_err)
    #             eim_errors.append(max_eim_err)
        
    #         axs[i].semilogy(range(1, len(eim_nodes)), proj_errors[:-1], marker='o', label='Projection error')
    #         axs[i].semilogy(range(1, len(eim_nodes)), eim_errors[:-1], marker='s', label='EIM error')
    #         axs[i].set_xlabel("Number of modes / EIM nodes")
    #         axs[i].set_ylabel("Max relative error")

    #     axs[0].set_title("Projection error vs EIM error")
    #     axs[0].grid(True)
    #     plt.legend()

    #     fig_proj_vs_eim_error.tight_layout()

    #     if save_fig:
    #         fig_path = f'Images/Empirical_nodes/Errors/EIM_projection_vs_eim_error_M={self.total_mass}_f_lower={self.f_lower}_f_ref={self.f_ref}_iN={len(self.ecc_ref_space)}_gN={len(self.indices_basis)}_ms={self.minimum_spacing_greedy}.png'
    #         os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    #         fig_proj_vs_eim_error.savefig(fig_path)
    #         print(self.colored_text(f'Figure is saved in {fig_path}', 'blue'))

    #     return proj_errors, eim_errors

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
        # Get the waveform at the requested eccentricity
        hp, hc, _ = self.simulate_waveform(ecc_ref=eccentricity)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the waveform and the empirical nodes together for reference
        ax.plot(self.time, hp, linewidth=0.2, color='black', label=f'$h_+$: ecc = {eccentricity}')
        ax.plot(self.time, hc, linewidth=0.2, linestyle='dashed', color='black', label=f'$h_\times$: ecc = {eccentricity}')
        ax.scatter(self.time[emp_nodes_idx], np.zeros(len(emp_nodes_idx)), color='red', s=8)

        ax.set_ylabel(f'$h_{{22}}$')
        ax.set_xlabel('t [M]')
        ax.legend(loc='upper left')

        # IF savefig is set to True, save the figure
        if save_fig:
            fig_path = f'Images/Empirical_nodes/EIM_{property}_e={eccentricity}_M={self.total_mass}_f_lower={self.f_lower}_f_ref={self.f_ref}_iN={len(self.ecc_ref_space)}_gN={len(self.indices_basis)}_ms={self.minimum_spacing_greedy}.png'
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            fig.savefig(fig_path)
            print(self.colored_text(f'Figure is saved in {fig_path}', 'blue'))

    def _plot_empirical_nodes_from_dataset(self, emp_nodes_idx, property, eccentricity, save_fig):
        """
        Helper function to plot empirical nodes for a given eccentricity.

        Parameters:
        ----------------
        - emp_nodes_idx (list): Indices of the empirical nodes.
        - property (str): Waveform property being plotted (e.g., "phase" or "amplitude").
        - eccentricity (float): Eccentricity value for the plot.
        - save_fig (bool): If True, saves the plot to a file.
        """
        # Get the waveform at the requested eccentricity
        hp, hc, _ = self.simulate_waveform(ecc_ref=eccentricity)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the waveform and the empirical nodes together for reference
        ax.plot(self.time, hp, linewidth=0.2, color='black', label=f'$h_+$: ecc = {eccentricity}')
        ax.plot(self.time, hc, linewidth=0.2, linestyle='dashed', color='black', label=f'$h_\times$: ecc = {eccentricity}')
        ax.scatter(self.time[emp_nodes_idx], np.zeros(len(emp_nodes_idx)), color='red', s=8)

        ax.set_ylabel(f'$h_{{22}}$')
        ax.set_xlabel('t [M]')
        ax.legend(loc='upper left')

        # IF savefig is set to True, save the figure
        if save_fig:
            fig_path = f'Images/Empirical_nodes/EIM_dataset_{property}_e={eccentricity}_M={self.total_mass}_f_lower={self.f_lower}_f_ref={self.f_ref}_iN={len(self.ecc_ref_space)}_gN={len(self.indices_basis)}_ms={self.minimum_spacing_greedy}.png'
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            fig.savefig(fig_path)
            print(self.colored_text(f'Figure is saved in {fig_path}', 'blue'))

  
    
    def get_training_set_greedy(self, property, emp_nodes_of_full_dataset=False, min_greedy_error=None, N_greedy_vecs=None, plot_training_set=False, 
                        plot_greedy_error=False, save_fig_greedy_error=False, plot_emp_nodes_on_basis=False, save_fig_emp_nodes_on_basis=False, plot_emp_nodes_at_ecc=False, save_fig_emp_nodes_at_ecc=False, save_fig_training_set=False, 
                        save_dataset_to_file=True, save_fig_residuals_eccentric=False, save_fig_residuals_time=False, plot_greedy_vecs=False, save_fig_greedy_vecs=False, clean_residuals=False):
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
        - residual_training_set (numpy.ndarray): Training set of residuals at empirical nodes.
        """
        # Import from class object if min_greedy and N_greedy vecs are not specified
        if (min_greedy_error is None) and (N_greedy_vecs is None):
            if property == 'phase':
                min_greedy_error = self.min_greedy_error_phase
                N_greedy_vecs = self.N_basis_vecs_phase
            elif property == 'amplitude':
                min_greedy_error = self.min_greedy_error_amp
                N_greedy_vecs = self.N_basis_vecs_amp

        
        # Step 1: Generate residuals for the full parameter space
        train_obj = self.generate_property_dataset(
            ecc_ref_list=self.ecc_ref_space,
            mass_ratios_list=self.mass_ratio_space,
            property=property,
            save_dataset_to_file=save_dataset_to_file,
            plot_residuals_eccentric_evolv=True,
            plot_residuals_time_evolv=True,
            save_fig_eccentric_evolv=save_fig_residuals_eccentric,
            save_fig_time_evolve=save_fig_residuals_time
        )
        
        # Get the training object for the specified property (phase or amplitude)
        # train_obj = self._get_training_obj(property)

        # Step 2: Select the best representative parameters using a greedy algorithm
        # print('Calculating greedy parameters...')
        reduced_basis_object = self.get_greedy_parameters(
            train_obj=train_obj,
            N_greedy_vecs=N_greedy_vecs,
            min_greedy_error=min_greedy_error,
            plot_greedy_error=plot_greedy_error,
            save_greedy_error_fig=save_fig_greedy_error,
            plot_greedy_vectors=plot_greedy_vecs,
            save_greedy_vecs_fig=save_fig_greedy_vecs
        )
        # print(f'Greedy parameters {property}: {train_obj.basis_indices}, length: {len(train_obj.basis_indices)}  ')

        self.best_rep_parameters = list(self.ecc_ref_space[train_obj.basis_indices])
        
        # Step 3: Calculate empirical nodes of the greedy basis
        # print('Calculating empirical nodes...')
        if emp_nodes_of_full_dataset:
            train_obj.empirical_indices = self.empirical_interpolation_from_dataset(
            waveforms_dataset=train_obj.residuals,
            property=property,
            plot_emp_nodes_at_ecc=plot_emp_nodes_at_ecc,
            save_fig=save_fig_emp_nodes_at_ecc
        )
        else:
            train_obj.empirical_indices = self.get_empirical_nodes(
                reduced_basis_object=reduced_basis_object,
                train_obj=train_obj,
                plot_emp_nodes_on_basis=plot_emp_nodes_on_basis,
                save_emp_nodes_on_basis_fig=save_fig_emp_nodes_on_basis,
                plot_interpolation_matrix=False,
                save_interpolation_matrix_fig=False,
                plot_proj_vs_eim_error=False,
                save_proj_vs_eim_error_fig=False
            )

        # self.empirical_nodes_idx = self.get_empirical_nodes_test(
        #     reduced_basis=residual_greedy_basis_orthonormal,
        #     property=property,
        #     plot_emp_nodes_at_ecc=plot_emp_nodes_at_ecc,
        #     save_fig=save_fig_emp_nodes
        # )


        # print(f'emp nodes {property}: {train_obj.empirical_indices}, length: {len(train_obj.empirical_indices)}  ')
        # Step 4: Generate the training set at empirical nodes
        train_obj.residual_basis = train_obj.residuals[train_obj.basis_indices] # shape (n_greedy_vecs, n_time)
        train_obj.training_set = train_obj.residual_basis[:, train_obj.empirical_indices]
        self.time_training = self.time[train_obj.empirical_indices]

        # Optionally plot the training set
        if plot_training_set:
            self._plot_training_set(property, save_fig_training_set, show_legend=False)

        # Clean memory of objects that are no longer needed
        if clean_residuals:
            del train_obj.residuals, train_obj.residual_basis

        return train_obj

    def _plot_training_set(self, property, save_fig, show_legend=True):
        """
        Helper function to plot and optionally save the training set of residuals.

        Parameters:
        ----------------
        - residual_training_set (numpy.ndarray): Training set of residuals at empirical nodes.
        - property (str): The waveform property ('phase' or 'amplitude').
        - save_fig (bool): If True, saves the plot to a file.
        """
        train_obj = self._get_training_obj(property)
        fig, ax = plt.subplots()

        for i, idx in enumerate(train_obj.basis_indices):
            ax.plot(self.time, train_obj.residual_basis[i], label=f'e={round(self.ecc_ref_space[idx], 3)}', linewidth=0.6)
            ax.scatter(self.time[train_obj.empirical_indices], train_obj.residual_basis[i][train_obj.empirical_indices])

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




