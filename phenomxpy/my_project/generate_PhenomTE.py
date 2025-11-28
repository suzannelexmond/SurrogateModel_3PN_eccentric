import sys
import phenomxpy.phenomt as phenomt
from phenomxpy.common import Waveform
from phenomxpy.utils import SecondtoMass, AmpSItoNR, m1ofq, m2ofq, AmpNRtoSI, HztoMf, MftoHz, MasstoSecond
from pycbc import waveform, types
import numpy as np
import matplotlib.pyplot as plt
import os
from timeit import default_timer as timer
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import warnings
warnings.simplefilter("once")
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")


plt.switch_backend('WebAgg')

class Simulate_Inspiral:
    """ Simulates time-domain (2,2) mode EOB waveform of a binary blackhole merger. Generates time-domain from starting frequency (f_lower) till peak at t=0 for time in geometric units. """
    
    def __init__(self, time_array, ecc_ref, total_mass, luminosity_distance, f_lower=10, f_ref=20, chi1=0, chi2=0, phiRef=0., rel_anomaly=0., inclination=0., mean_anomaly_start=0., truncate_at_ISCO=True, truncate_at_tmin=True):
        """
        Parameters:
        ----------------
        time_array [s], np.array : Time array in seconds.
        ecc_ref [dimensionless], float: Eccentricity of binary at start f_lower
        total_mass [M_sun], int : Total mass of the binary in solar masses
        f_lower [Hz], float: Start frequency of the waveform
        f_ref [Hz], float: Reference frequency at which the waveform parameters (eccentricity, spin, ...) are defined.
        chi1 [dimensionless], float, ndarray : Spin of primary. If float, interpreted as z component
        chi2 [dimensionless], float, ndarray : Spin of secondary. If float, interpreted as z component
        rel_anomaly [rad], float : Relativistic anomaly. Radial phase which parametrizes the orbit within the Keplerian (relativistic) parametrization. Defaults to 0 (periastron).
        inclination [rad], float : Inclination angle of the binary system. Defaults to 0 (face-on).
        luminosity_distance [Mpc], float : Luminosity distance of the binary in megap
        """

        # Initial parameters
        if total_mass is None:
            total_mass = 60 # Reference total mass for geometric frequency calculations [M_sun]

        self.time = SecondtoMass(time_array, total_mass) # Time array in geometric units c=G=M=1
        self.ecc_ref = ecc_ref # eccentricity of binary at start frequency
        self.f_lower =  f_lower# Start frequency [Hz]
        self.f_ref = f_ref # Reference frequency at which the waveform parameters (eccentricity, spin, ...) are defined.
        self.total_mass = total_mass # Total mass of the binary [M]
        self.luminosity_distance = luminosity_distance # Luminosity distance of the binary [Mpc]
        self.chi1 = chi1 # Dimensionless spin of primary [float,ndarray]. If float, interpreted as z component
        self.chi2 = chi2 # Dimensionless spin of secondary [float,ndarray]. If float, interpreted as z component
        self.rel_anomaly= rel_anomaly # Relativistic anomaly. Radial phase which parametrizes the orbit within the Keplerian (relativistic) parametrization. Defaults to 0 (periastron)
        self.inclination = inclination # Inclination angle of the binary system [rad]
        self.phiRef = phiRef # Reference phase of the waveform [rad]
        self.truncate_at_ISCO = truncate_at_ISCO
        self.truncate_at_tmin = truncate_at_tmin
        self.mean_anomaly_start = mean_anomaly_start


    def amplitude(self, hplus_NR, hcross_NR, geometric_units=True, luminosity_distance=None, total_mass=None):
        """
        Calculate the amplitude from the plus and cross polarizations.
        If geometric_units is True, the amplitude is returned in geometric units.
        """
        amp_NR = np.abs(hplus_NR - 1j * hcross_NR)
        
        if geometric_units:
            return amp_NR
        else:
            return AmpNRtoSI(amp_NR, luminosity_distance, total_mass)

    def phase(self, hplus, hcross):
        """
        Calculate the phase from the plus and cross polarizations. Unitless.
        """
        phase = np.unwrap(np.arctan2(hcross, hplus))
        phase -= phase[0] # Normalize phase to start at zero, correcting for the initial phase offset.
        
        return phase


    def polarisations(self, phase, amplitude, geometric_units=True, distance=None, total_mass=None, plot_polarisations=False, save_fig=False):
        """
        Calculate the plus and cross polarizations from the phase and amplitude.
        If geometric_units is True, the polarizations are returned in geometric units.
        """
        if geometric_units:
            hplus = amplitude * np.cos(phase)
            hcross = -amplitude * np.sin(phase)
        else:
            hplus = AmpNRtoSI(amplitude, distance, total_mass) * np.cos(phase)
            hcross = AmpNRtoSI(amplitude, distance, total_mass) * np.sin(phase)

            self.time = MasstoSecond(self.time, self.total_mass)
        
        if plot_polarisations is True:
            self._plot_polarisations(hplus, hcross, save_fig=save_fig)
            
        return hplus, hcross
    
    # def truncate_near_merger(self, phen):
    #     cparams = phen.pWF.cparams
    #     e = phen.pWF.e0
    #     e2 = e**2
    #     tchirp_ecccorr = (1 - e2)**3.5 / (1 + (157/24)*e2 + (605/96)*e2**2)

    #     # Reference inspiral time in code units (e.g., from phen.pWF.tchirp if available)
    #     # Or set a fixed value if not available
    #     tchirp_ref = cparams.get("tchirp", abs(phen.pWF.tmin))  # geometric units

    #     # Base extra time you want (e.g., 1.0 means keep the whole waveform)
    #     base_fraction = cparams.get("extra_time_fraction", 1.0)
    #     textra = base_fraction * tchirp_ref * tchirp_ecccorr + cparams.get("textra", 0.0)

    #     # Optional: truncate waveform to a maximum allowed window (e.g., 1/20 of full time)
    #     tmax = abs(phen.pWF.tmin) / 20.0
    #     textra = min(textra, tmax)

    #     # Now apply your taper or truncate logic
    #     hp, hc = time_array_condition_stage1(
    #         hp,
    #         hc,
    #         delta_t=params['delta_t'],  # geometric units
    #         textra=textra,
    #         f_lower=cparams['original_f_lower']
    #     )

    def truncate_waveform_at_isco(self, phen, time_array, plot_ISCO_cut_off=True):
        """
        Truncate the waveform at the ISCO frequency, which is approximately 0.021 in dimensionless units.
        This is done by finding the point where the instantaneous phase frequency Mf crosses the ISCO frequency
        and truncating the waveform at that point.

        Parameters:
        ----------------
        phen : Phenomt object containing the waveform data (hp, hc) and time array.     
        """

        # Compute instantaneous phase frequency Mf = dϕ/dt / 2π
        phase = self.phase(phen.hp, phen.hc)  # Calculate phase from plus and cross polarizations
        dphi_dt = np.gradient(phase, time_array)
        Mf = dphi_dt / (2 * np.pi)

        # Calculate ISCO frequency (dimensionless): Mf_ISCO = 1 / (6^(3/2) * π) ≈ 0.021
        f_isco = 1 / (6**1.5 * np.pi)  # dimensionless Mf_ISCO

        above_isco = np.where(Mf >= f_isco)[0]

        if len(above_isco) == 0:
            idx_cut = len(time_array) # in case there is no ISCO wihtin the specified time-array range
        else:
            idx_cut = above_isco[0]
            print(f'Waveform time-domain has been cut to match only-inspiral waveforms up till (circulair) ISCO.\nNEW TIME-DOMAIN (in geometric units): [{int(self.time[0])}, {int(self.time[idx_cut])}] M')


        # Truncate waveform
        # Only cut hp, hc, not self.time, because self.time does not recover for a new waveform run. Adjust self.time after shortest waveform has been established.
        phen.hp = phen.hp[:idx_cut]
        phen.hc = phen.hc[:idx_cut]
        if idx_cut == len(time_array):
            plot_ISCO_cut_off = False

        if plot_ISCO_cut_off is True:
            ISCO_vs_Mf_after = plt.figure(figsize=(12,5))

            plt.plot(time_array, Mf, label=f'Mf before ISCO cut: $e$={round(phen.pWF.eccentricity, 3)}')
            plt.axhline(f_isco, color='red', linestyle='--', label='Mf ISCO $e$=0', linewidth=0.6)
            plt.scatter(time_array[idx_cut], Mf[idx_cut], color='r', s=6, label='ISCO cut')
            plt.plot(time_array[:idx_cut], Mf[:idx_cut], label=f'Mf after ISCO cut: $e$={round(phen.pWF.eccentricity, 3)} ')
            plt.legend()

            os.makedirs('Images/ISCO', exist_ok=True)  # Ensure the directory exists

            figname = 'Images/ISCO/truncate_at_ISCO_vs_Mf_M={}_e={}_f_lower={}.svg'.format(self.total_mass, round(phen.pWF.eccentricity, 2), self.f_lower)
            ISCO_vs_Mf_after.savefig(figname, dpi=300, bbox_inches='tight')
            # plt.close('all') 
            print(self.colored_text(f'Figure is saved in {figname}', 'blue'))
            
        # Clean memory
        del phase, dphi_dt, Mf, f_isco, above_isco

        return idx_cut
        
    def simulate_inspiral(self, total_mass=None, luminosity_distance=None, custom_time_array=None, ecc_ref=None, mean_ano = None, truncate_at_ISCO=False, truncate_at_tmin=False, geometric_units=True, plot_polarisations=False, save_fig=False):
        """
        Simulate mass-independent plus and cross polarisations of the eccentric eob waveform (pyseobnr) (2,2) mode from f_start till t0 (waveform peak at t=0).
        
        Parameters:
        ----------------
        ecc_ref [dimensionless], float : For other eccentricity than Class specified ecc_ref, set new value.
        plot_polarisations, True OR False, bool : For a plot of the plus and cross polarisations, set to True.
        save_fig, True Or False, bool : If plot of the polarisations should be saved to a automatically created folder \Images, set to True.
        
        Returns:
        ----------------
        hp [dimensionless], np.array: Time-domain plus polarisation 
        hc [dimensionless], np.array: Time-domain cross polarisation 
        t [M], np.array: Time-domain in mass independent geometric units c=G=M=1
        """

        if custom_time_array is None:
            time_array = self.time
        else:
            time_array = custom_time_array

        # Either set ecc_ref specifically or use the class defined value
        if ecc_ref is None:
            ecc_ref = self.ecc_ref
        else:
            ecc_ref = ecc_ref

        # To compute the runtime for 1 simulated waveform
        start = timer()
        

        if (geometric_units is False) and ((total_mass is not None) or (luminosity_distance is not None)):
            reference_total_mass = 60
            f_ref_geom = HztoMf(self.f_ref, reference_total_mass)
            f_lower_geom = HztoMf(self.f_lower, reference_total_mass)

            f_ref = MftoHz(f_ref_geom, total_mass)
            f_lower = MftoHz(f_lower_geom, total_mass)
        elif (geometric_units is False) and ((total_mass is None) or (luminosity_distance is None)):
            warnings.warn("For geometric_units=False, both total_mass and luminosity_distance must be provided.")
        elif (geometric_units is True) and ((total_mass is not None) or (luminosity_distance is not None)):
            total_mass = None
            luminosity_distance = None
            warnings.warn("total_mass and luminosity_distance are ignored when geometric_units=True.")
        else:
            total_mass = 60 # Reference total mass for geometric frequency calculations [M_sun] in case of geometric units
            f_ref = self.f_ref
            f_lower = self.f_lower

        
        phen = phenomt.PhenomTE(
            mode=[2,2],
            times=time_array,
            eccentricity=ecc_ref,  
            total_mass=total_mass, # Used as reference total_mass to calculate geometric frequency during the waveform generation.           
            f_ref=f_ref,                   
            f_lower=f_lower,
            phiRef=self.phiRef,
            inclination=self.inclination,
            mean_anomaly = mean_ano if mean_ano is not None else 0)
        
        if geometric_units is False:
            phen.compute_polarizations(times=time_array, total_mass=total_mass, distance=luminosity_distance)
        else:
            phen.compute_polarizations(times=time_array)

        if phen.pWF.tmin > time_array[0]:
            warnings.warn(
                "t_min is larger than parts of the specified time-domain, resulting in unphysical waveforms. "
                "Either use the truncate_tmin=True setting to automatically truncate to start from t_min=time_array[0] "
                "or adjust the time-array manually to start at higher values."
            )
            # mask to only include the physical range of the time-domain
            if (self.truncate_at_tmin is True) and (truncate_at_tmin is True):
                mask = time_array >= phen.pWF.tmin

                time_array = time_array[mask]
                phen.hp = phen.hp[mask]
                phen.hc = phen.hc[mask]

                print(f'NEW TIME-DOMAIN (in geometric units): [{int(time_array[0])}, {int(time_array[-1])}] M')
                del mask # clear memory

        # True because it's smallest truncated waveform AND true because the surrogate is called with the ISCO cut-off.
        if (self.truncate_at_ISCO is True) and (truncate_at_ISCO is True):
            # Truncate the waveform at ISCO frequency
            idx_cut = self.truncate_waveform_at_isco(phen, time_array)
            time_array = time_array[:idx_cut]

        # print(f'time : SimInspiral_M_independent ecc = {round(ecc_ref, 3)}, len = {len(phen.hp)}, M = {self.total_mass}, lum_dist={self.luminosity_distance}, t=[{int(time_array[0])}, {int(time_array[-1])}, num={len(time_array)}], f_lower={self.f_lower}, f_ref={self.f_ref} | computation time = {(timer()-start)} seconds')

        if plot_polarisations is True:
            self._plot_polarisations(phen.hp, phen.hc, time_array, save_fig)

        if custom_time_array is None:
            self.time = time_array
            return phen.hp, phen.hc
        else:
            return phen.hp, phen.hc, time_array
        
    
    def _plot_polarisations(self, hp, hc, time_array=None, save_fig=True):
        """
        Plot the plus and cross polarizations of the waveform.
        
        Parameters:
        ----------------
        hp [dimensionless], np.array: Plus polarization of the waveform
        hc [dimensionless], np.array: Cross polarization of the waveform
        plot_polarisations, True OR False, bool: Set to True to include a plot of the polarizations
        save_fig, True OR False, bool: Saves the figure to the directory Images/Polarisations
        
        Returns:
        ----------------
        None
        """
        if time_array is None:
            time_array = self.time
            
        fig = plt.figure(figsize=(12,5))
      
        plt.plot(self.time[-len(hp):], hp, label = f'$h_+$ mass indp', linewidth=0.6)
        # plt.plot(self.time[:len(hc)], hc, label = f'$h_\times$', linewidth=0.6)

        plt.legend(loc = 'upper left')
        plt.xlabel('t [s]')
        plt.ylabel('$h_{22}]$')
        plt.title(f'M={self.total_mass}, e={round(self.ecc_ref, 3)}, f_min={self.f_lower} Hz')
        plt.grid(True)

        plt.tight_layout()
        # plt.show()

        if save_fig is True:
            figname = 'Images/Polarisations/Polarisations_M={}_ecc={}.png'.format(self.total_mass, round(self.ecc_ref, 3))
            
            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Images/Polarisations', exist_ok=True)
            fig.savefig(figname, dpi=300, bbox_inches='tight')

            print(self.colored_text(f'Figure is saved in {figname}', 'blue'))

        # plt.close('all')

    def colored_text(self, text, color):
        """
        Returns colored text for terminal output.
        Parameters:
        ----------------
        text : str : Text to be colored
        color : str : Color name ('red', 'green', 'yellow', 'blue')
        Returns:
        ----------------
        str : Colored text
        """

        colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'reset': '\033[0m'
        }
        return f"{colors.get(color, '')}{text}{colors['reset']}"


    





# class Waveform_Properties_l(Simulate_Inspiral):
#     """
#     Calculates and plots residuals (residual = eccentric - non-eccentric) of waveform properties: amplitude, phase and frequency.
#     """

#     def __init__(self, time_array, ecc_ref, total_mass, luminosity_distance, f_lower=10, f_ref=20, chi1=0, chi2=0, phiRef=0., rel_anomaly=0., inclination=0., truncate_at_ISCO=True, truncate_at_tmin=True):
#         """
#         Parameters:
#         ----------------
#         ecc_ref [dimensionless], float: Eccentricity of binary at start f_lower
#         total_mass [M_sun], float : Total mass of the binary in solar masses
#         f_lower [Hz], float: Start frequency of the waveform
#         t_circ [M], np.array : Time array for non-eccentric inspiral
#         hp_circ [dimensionless], np.array : plus polarisation of non-eccentric inspiral
#         hc_circ [dimensionless], np.array : cross polarisation of non-eccentric inspiral
#         """

#         self.hp_circ = None # TimeSeries object of plus polarisation for non-eccentric inspiral 
#         self.hc_circ = None # TimeSeries object of cross polarisation for non-eccentric inspiral 
#         self.phase_circ = None # Phase of the non-eccentric inspiral waveform
#         self.amp_circ = None # Amplitude of the non-eccentric inspiral waveform
#         self.l_grid = None

#         # Inherit parameters from Simulate_Inspiral class
#         Simulate_Inspiral.__init__(self, time_array, ecc_ref, total_mass, luminosity_distance, f_lower, f_ref, chi1, chi2, phiRef, rel_anomaly, inclination, truncate_at_ISCO, truncate_at_tmin)
    
#     def compute_mean_anomaly(self, hp_t, hc_t):
#         # Compute instantaneous frequency as derivative of phase
#         dt = np.gradient(self.time)
#         phase = self.phase(hp_t, hc_t)
#         omega = np.gradient(phase) / dt  # omega(t) = dphi/dt

#         # Integrate omega over time to get mean anomaly
#         # cumtrapz returns one element less, so prepend initial mean anomaly
#         # Approximate cumulative integral using cumsum:
#         M = self.mean_anomaly_start + np.insert(np.cumsum(0.5 * (omega[1:] + omega[:-1]) * np.diff(self.time)), 0, 0)

#         return M
    
#     def circulair_wf(self):
#         """
#         Simulate plus and cross polarisations of NON-ECCENTRIC waveform Inspiral for t in units [M]. 
#         Also saves the phase and amplitude accordingly.
       
#         Returns:
#         ----------------
#         hp_circ [dimensionless], np.array: Time-domain plus polarisation of NON-ECCENTRIC waveform
#         hc_circ [dimensionless], np.array: Time-domain cross polarisation of NON-ECCENTRIC waveform

#         """
        
#         if self.phase_circ is None or self.amp_circ is None:
#             # Generate arrays
#             self.hp_circ, self.hc_circ = self.simulate_inspiral(ecc_ref=0)


#             self.phase_circ = self.phase(self.hp_circ, self.hc_circ)
#             self.amp_circ = self.amplitude(self.hp_circ, self.hc_circ)

#         elif self.amp_circ is not None and len(self.amp_circ) != len(self.time):
#             # Truncate to match
#             self.phase_circ = self.phase_circ[:len(self.time)]
#             self.amp_circ = self.amp_circ[:len(self.time)]
#         else:
#             pass # self.hp_circ and self.hc_circ are already set, no need to recompute


        

#     def calculate_residual(self, hp, hc, ecc_ref=None, property=None, plot_residual=False, save_fig=False):
#         """
#         Calculate residual (= eccentric - non-eccentric) of Waveform Inspiral property.
#         Possible properties: phase, amplitude or frequency
        
#         Parameters: 
#         ----------------
#         hp [dimensionless], np.array : mass independent plus polarisation
#         hc [dimensionless], np.array : mass independent cross polarisation        property, str: Choose residual for ['phase', 'amplitude']
#         plot_residual, True OR False, bool: Set to True to include a plot of the residual including eccentric and non-eccentric case
#         save_fig, True OR False, bool: Saves the figure to the directory Images/Residuals
        
#         Returns:
#         ----------------
#         residual : residual = eccentric - non-eccentric for chosen property
#         """     

#         if ecc_ref is None:
#             ecc_ref = self.ecc_ref
            

#         # Calculate plus and cross polarizations of circular (non-eccentric) waveform
#         self.circulair_wf()

#         # Calculate phase from plus and cross polarizations
#         if property == 'phase':
#             circ = self.phase_circ# non-eccentric case
#             eccentric = self.phase(hp, hc) # eccentric case

            

#             units = '[radians]'

#             # Residual = circular - eccentric to prevent negative residual values
#             residual = circ - eccentric # to prevent negative values

#             mean_anomaly_time_series = self.compute_mean_anomaly(hp_t=hp, hc_t=hc)

#             test_mean_ano, axs = plt.subplots(2)
#             # plt.plot(self.time, mean_anomaly_time_series)
#             axs[0].plot(mean_anomaly_time_series, residual)
#             axs[1].plot(self.time, residual)
#             test_mean_ano.savefig('Images/mean_ano.png')

#             # Warning for negative residual values

#             if eccentric[1] < 0: # 
#                 warnings.warn("Eccentric phase has negative starting values. This may not be expected for physical waveforms. This usually happens when the eccentric waveformlength is shorter than the chosen time array. Consider decreasing the time array length or decreasing the eccentricity.")

#         # Calculate amplitude from plus and cross polarisations
#         elif property == 'amplitude':
#             circ = self.amp_circ # non-eccentric case
#             eccentric = self.amplitude(hp, hc) # eccentric case
#             units = '' # for plotting 
 
#             residual = eccentric - circ

#         else:
#             print('Choose property = "phase", "amplitude", "frequency"', property, 2)
#             sys.exit(1)

#         if plot_residual is True:
#             fig_residual = plt.figure()
            
#             plt.plot(self.time, eccentric, label= f'Eccentric {property}: $e$={ecc_ref}', linewidth=0.6) # eccentric property
#             plt.plot(self.time, circ, label=f'Circular {property}: $e$=0', linewidth=0.6) # non-eccentric property
#             plt.plot(self.time, residual, label=f'Residual {property}', linewidth=0.6) # residual property
            
#             plt.xlabel('t [M]')
#             plt.ylabel(property + ' ' + units)
#             plt.title('Residual')
#             plt.grid(True)
#             plt.legend()

#             plt.tight_layout()

#             # plt.close('all')  # Clean up plots

#             if save_fig is True:
#                 figname = f'Images/Residuals/Residual {property} M={self.total_mass}, ecc={round(ecc_ref, 3)}.png'
                
#                 # Ensure the directory exists, creating it if necessary and save
#                 os.makedirs('Images/Residuals', exist_ok=True)
#                 fig_residual.savefig(figname, dpi=300, bbox_inches='tight')

#                 print(self.colored_text(f'Figure is saved in {figname}', 'blue'))
        
#         del circ, eccentric # clear memory

#         return residual
    


class Waveform_Properties(Simulate_Inspiral):
    """
    Calculates and plots residuals (residual = eccentric - non-eccentric) of waveform properties: amplitude, phase and frequency.
    """

    def __init__(self, time_array, ecc_ref, total_mass, luminosity_distance, f_lower=10, f_ref=20, chi1=0, chi2=0, phiRef=0., rel_anomaly=0., inclination=0., truncate_at_ISCO=True, truncate_at_tmin=True):
        """
        Parameters:
        ----------------
        ecc_ref [dimensionless], float: Eccentricity of binary at start f_lower
        total_mass [M_sun], float : Total mass of the binary in solar masses
        f_lower [Hz], float: Start frequency of the waveform
        t_circ [M], np.array : Time array for non-eccentric inspiral
        hp_circ [dimensionless], np.array : plus polarisation of non-eccentric inspiral
        hc_circ [dimensionless], np.array : cross polarisation of non-eccentric inspiral
        """

        self.hp_circ = None # TimeSeries object of plus polarisation for non-eccentric inspiral 
        self.hc_circ = None # TimeSeries object of cross polarisation for non-eccentric inspiral 
        self.phase_circ = None # Phase of the non-eccentric inspiral waveform
        self.amp_circ = None # Amplitude of the non-eccentric inspiral waveform

        # Inherit parameters from Simulate_Inspiral class
        Simulate_Inspiral.__init__(self, time_array=time_array, ecc_ref=ecc_ref, total_mass=total_mass, luminosity_distance=luminosity_distance, f_lower=f_lower, f_ref=f_ref, chi1=chi1, chi2=chi2, phiRef=phiRef, rel_anomaly=rel_anomaly, inclination=inclination, truncate_at_ISCO=truncate_at_ISCO, truncate_at_tmin=truncate_at_tmin)

    def circulair_wf(self):
        """
        Simulate plus and cross polarisations of NON-ECCENTRIC waveform Inspiral for t in units [M]. 
        Also saves the phase and amplitude accordingly.
       
        Returns:
        ----------------
        hp_circ [dimensionless], np.array: Time-domain plus polarisation of NON-ECCENTRIC waveform
        hc_circ [dimensionless], np.array: Time-domain cross polarisation of NON-ECCENTRIC waveform

        """
        
        if (self.phase_circ is None) or (self.amp_circ is None):
            # Generate arrays
            self.hp_circ, self.hc_circ = self.simulate_inspiral(ecc_ref=0)
            
            self.phase_circ = self.phase(self.hp_circ, self.hc_circ)
            self.amp_circ = self.amplitude(self.hp_circ, self.hc_circ)

        elif self.amp_circ is not None and len(self.amp_circ) != len(self.time):
            # Truncate to match
            self.phase_circ = self.phase_circ[:len(self.time)]
            self.amp_circ = self.amp_circ[:len(self.time)]
        else:
            pass # self.hp_circ and self.hc_circ are already set, no need to recompute
        
        

    def calculate_residual(self, hp, hc, ecc_ref=None, property=None, plot_residual=False, save_fig=False):
        """
        Calculate residual (= eccentric - non-eccentric) of Waveform Inspiral property.
        Possible properties: phase, amplitude or frequency
        
        Parameters: 
        ----------------
        hp [dimensionless], np.array : mass independent plus polarisation
        hc [dimensionless], np.array : mass independent cross polarisation        property, str: Choose residual for ['phase', 'amplitude']
        plot_residual, True OR False, bool: Set to True to include a plot of the residual including eccentric and non-eccentric case
        save_fig, True OR False, bool: Saves the figure to the directory Images/Residuals
        
        Returns:
        ----------------
        residual : residual = eccentric - non-eccentric for chosen property
        """     
        
        if ecc_ref is None:
            ecc_ref = self.ecc_ref

        # Calculate plus and cross polarizations of circular (non-eccentric) waveform
        self.circulair_wf()
        # Calculate phase from plus and cross polarizations
        if property == 'phase':
            circ = self.phase_circ# non-eccentric case
            eccentric = self.phase(hp, hc) # eccentric case
            units = '[radians]'
            # Residual = circular - eccentric to prevent negative residual values
            residual = circ - eccentric # to prevent negative values
            # Warning for negative residual values

            if eccentric[1] < 0: # 
                warnings.warn("Eccentric phase has negative starting values. This may not be expected for physical waveforms. This usually happens when the eccentric waveformlength is shorter than the chosen time array. Consider decreasing the time array length or decreasing the eccentricity.")
            
        # Calculate amplitude from plus and cross polarisations
        elif property == 'amplitude':
            circ = self.amp_circ # non-eccentric case
            eccentric = self.amplitude(hp, hc) # eccentric case
            units = '' # for plotting 
 
            residual = eccentric - circ

        else:
            print('Choose property = "phase", "amplitude", "frequency"', property, 2)
            sys.exit(1)
        if plot_residual is True:
            fig_residual = plt.figure()
            
            plt.plot(self.time, eccentric, label= f'Eccentric {property}: $e$={ecc_ref}', linewidth=0.6) # eccentric property
            plt.plot(self.time, circ, label=f'Circular {property}: $e$=0', linewidth=0.6) # non-eccentric property
            plt.plot(self.time, residual, label=f'Residual {property}', linewidth=0.6) # residual property
            
            plt.xlabel('t [M]')
            plt.ylabel(property + ' ' + units)
            plt.title('Residual')
            plt.grid(True)
            plt.legend()

            plt.tight_layout()

            # plt.close('all')  # Clean up plots

            if save_fig is True:
                figname = f'Images/Residuals/Residual {property} M={self.total_mass}, ecc={round(ecc_ref, 3)}.png'
                
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Residuals', exist_ok=True)
                fig_residual.savefig(figname, dpi=300, bbox_inches='tight')

                print(self.colored_text(f'Figure is saved in {figname}', 'blue'))
        
        del circ, eccentric # clear memory
        return residual
    

# def apply_mean_anomaly_mapping(A0, phi0, l_ref, j=1, m=2, epsilon=0.05):
#     """
#     Apply the approximate universal mean anomaly mapping to phase and amplitude.
    
#     A0, phi0: reference amplitude and phase (at l_ref=0)
#     l_ref: desired mean anomaly (radians)
#     j: harmonic index (default 1 for dominant)
#     m: mode index (default 2 for (2,2))
#     epsilon: small amplitude modulation factor
#     """
#     phi_new = phi0 + m * j * l_ref
#     A_new = A0 * (1 + epsilon * np.sin(l_ref))
#     return A_new, phi_new



# sampling_frequency = 2048 # or 4096
# duration = 4 # seconds
# time_array = np.linspace(-duration, 0, int(sampling_frequency * duration))  # time in seconds

# eccentricties = [0, 0.1, 0.2, 0.3]

# # circulair waveform for reference
# si_0 = Simulate_Inspiral(time_array=time_array, ecc_ref=0, total_mass=60, luminosity_distance=200)
# hp_0, hc_0 = si_0.simulate_inspiral(mean_ano=0, plot_polarisations=False, save_fig=False, truncate_at_ISCO=False, truncate_at_tmin=False, geometric_units=True)
# phase_0 = si_0.phase(hp_0, hc_0)
# amplitude_0 = si_0.amplitude(hp_0, hc_0)

# for ecc in eccentricties:
#     si = Simulate_Inspiral(time_array=time_array, ecc_ref=ecc, total_mass=60, luminosity_distance=200)

#     mean_anos = np.linspace(0, np.pi/2, 4)  # Mean anomalies from 0 to pi/2


#     pols, axs = plt.subplots(3, 1, figsize=(10,6))
#     # pols = plt.figure(figsize=(10,6))
#     phases = np.zeros((len(mean_anos), len(si.time)))
#     amplitudes = np.zeros((len(mean_anos), len(si.time)))

    

#     for i, mean_ano in enumerate(mean_anos):
#         hp, hc = si.simulate_inspiral(mean_ano=mean_ano, plot_polarisations=False, save_fig=False, truncate_at_ISCO=False, truncate_at_tmin=True, geometric_units=True)

#         phase = si.phase(hp, hc)
#         if np.isnan(phase).any():
#             print(f'ecc= {ecc}, mean_ano = {mean_ano}, phase={phase}')
#         amplitude = si.amplitude(hp, hc)
        

#         # phases[i] = phase
#         # amplitudes[i] = amplitude

#         # phases[i] = phase - phase_0
#         # amplitudes[i] = amplitude - amplitude_0

#         axs[0].plot(si.time, hp, label=f'Mean Anomaly={round(mean_ano,2)} rad', lw=0.6)
#         axs[1].plot(si.time, phase - phase_0[-len(phase):], label=f'Mean Anomaly={round(mean_ano,2)} rad', lw=0.6)
#         axs[2].plot(si.time, amplitude - amplitude_0[-len(phase):], label=f'Mean Anomaly={round(mean_ano,2)} rad', lw=0.6)

#     indices = np.arange(3000, len(si.time), 1000)
    
#     # print(indices, phases.T.shape, amplitudes.T.shape)


#     # for t in indices:
#         # plt.plot(mean_anos, phases.T[t], lw=0.6, label='t={}'.format(round(t)))

#         # axs[0].plot(mean_anos, phases.T[t], lw=0.6, label='t={}'.format(round(t)))
#         # axs[1].plot(mean_anos, amplitudes.T[t], lw=0.6, label='t={}'.format(round(t)))

#     # print('phases:', phases.T[1] - phases.T[0])
#     # print('amplitudes:', amplitudes.T[1] - amplitudes.T[0])

#     # plt.xlabel('Time [s]')
#     # axs[0].set_ylabel('$h_+$')
#     # axs[1].set_ylabel('Phase [rad]')
#     # axs[2].set_ylabel('Amplitude')
#     plt.title('Effect of Mean Anomaly on Plus Polarization')
#     # axs[0].legend()
#     plt.grid(True)
#     plt.tight_layout()
#     figname = f'Images/Polarisations/Mean_Anomaly_Effect_e={ecc}.svg'
#     os.makedirs('Images/Polarisations', exist_ok=True)
#     pols.savefig(figname, dpi=300, bbox_inches='tight')

#     print(si.colored_text(f'Figure is saved in {figname}', 'blue'))
########################################################################################
# plt.show()
# wp = Waveform_Properties_l(time_array=time_array, ecc_ref=0.2, total_mass=60, luminosity_distance=200, truncate_at_ISCO=False)
# hp, hc = wp.simulate_inspiral_mass_independent(ecc_ref=0.2, plot_polarisations=True, save_fig=True, truncate_at_ISCO=True, truncate_at_tmin=True)
# hp, hc = wp.simulate_inspiral(ecc_ref=0.2, plot_polarisations=True, save_fig=True, truncate_at_ISCO=False, truncate_at_tmin=False)

# wp.calculate_residual(hp, hc, property='amplitude', plot_residual=True, save_fig=True)
# wp.calculate_residual(hp, hc, property='phase', plot_residual=True, save_fig=True)

# plt.show()
# --- Universal mean anomaly mapping (data-driven style) ---
def apply_mean_anomaly_mapping(A0, phi0, l_ref, j=1):
    """
    Apply analytical mean anomaly mapping like in data-driven papers.
    Only shifts the phase linearly by j*l_ref, leaves amplitude mostly unchanged.
    """
    phi_new = phi0 + j * l_ref      # linear phase shift per harmonic
    A_new = A0.copy()               # amplitude unchanged
    return A_new, phi_new

# --- Simulation settings ---
sampling_frequency = 2048  # or 4096
duration = 4  # seconds
time_array = np.linspace(-duration, 0, int(sampling_frequency * duration))  # time in seconds

eccentricties = [0, 0.1, 0.2, 0.3]  # grid of eccentricities
mean_anos = np.linspace(0, np.pi/2, 50)  # mean anomalies to test

# --- Reference circular waveform ---
si_0 = Simulate_Inspiral(time_array=time_array, ecc_ref=0, total_mass=60, luminosity_distance=200)
hp_0, hc_0 = si_0.simulate_inspiral(mean_ano=0, plot_polarisations=False, save_fig=False,
                                    truncate_at_ISCO=False, truncate_at_tmin=False, geometric_units=True)
phase_0 = si_0.phase(hp_0, hc_0)
amplitude_0 = si_0.amplitude(hp_0, hc_0)

phases_res = np.zeros((len(mean_anos), 5000))
amplitudes_res = np.zeros((len(mean_anos), 5000))

# --- Loop over eccentricities ---
for ecc in eccentricties:
    si = Simulate_Inspiral(time_array=time_array, ecc_ref=ecc, total_mass=60, luminosity_distance=200)

    # Generate reference waveform at mean_ano=0 for this eccentricity
    # hp_ref, hc_ref = si.simulate_inspiral(mean_ano=0, plot_polarisations=False, save_fig=False,
    #                                       truncate_at_ISCO=False, truncate_at_tmin=True, geometric_units=True)
    # phase_ref = si.phase(hp_ref, hc_ref)
    # amplitude_ref = si.amplitude(hp_ref, hc_ref)

    # --- Plotting ---
    fig, axs = plt.subplots(4, 1, figsize=(10, 8))
    axs[0].set_title(f"Eccentricity = {ecc}")

    axs[0].set_ylabel("$\Delta$ Phase (rad)")
    axs[1].set_ylabel("$\Delta$ Amplitude")
    axs[2].set_ylabel("$\Delta$ Phase (rad)")
    axs[3].set_ylabel("$\Delta$ Amplitude")

    axs[1].set_xlabel("Time [s]")
    axs[3].set_xlabel("Mean Anomaly [rad]")

    # Loop over mean anomalies
    for i, mean_ano in enumerate(mean_anos):
        # --- Generate actual waveform ---
        hp_actual, hc_actual = si.simulate_inspiral(mean_ano=mean_ano, plot_polarisations=False, save_fig=False,
                                                    truncate_at_ISCO=False, truncate_at_tmin=True, geometric_units=True)
        
        phase_actual = si.phase(hp_actual, hc_actual)
        amplitude_actual = si.amplitude(hp_actual, hc_actual)

        amplitude_actual, phase_actual = apply_mean_anomaly_mapping(amplitude_actual, phase_actual, mean_ano, j=1)

        # --- Predict via data-driven universal mapping ---
        # A_pred, phi_pred = apply_mean_anomaly_mapping_dd(amplitude_ref, phase_ref, mean_ano, j=1)

        # --- Compute errors ---
        phase_res = phase_actual - phase_0[-len(phase_actual):]
        amplitude_res = amplitude_actual - amplitude_0[-len(phase_actual):]

        phases_res[i] = phase_res[-5000:]
        amplitudes_res[i] = amplitude_res[-5000:]

        # --- Plot ---
        axs[0].plot(si.time, phase_res, label=f"Mean Anomaly={mean_ano:.2f} rad", lw=0.8)
        axs[1].plot(si.time, amplitude_res, label=f"Mean Anomaly={mean_ano:.2f} rad", lw=0.8)

    # axs[0].legend()
    # axs[1].legend()

    for t in np.arange(0, 5000-1, 1000):
        axs[2].plot(mean_anos, phases_res[:, t], lw=0.6, label='t={}'.format(round(si.time[t])))
        axs[3].plot(mean_anos, amplitudes_res[:, t], lw=0.6, label='t={}'.format(round(si.time[t])))

    axs[2].legend()
    axs[3].legend()

    plt.tight_layout()
    os.makedirs('Images/Mean_anomaly', exist_ok=True)
    figname = f'Images/Mean_anomaly/Mean_Anomaly_New_Mapping_Ecc={ecc}.png'
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    # plt.show()

    print(si.colored_text(f'Figure is saved in {figname}', 'blue'))



    # # --- Optional: compute L2-norm mismatch for each mean anomaly ---
    # for i, mean_ano in enumerate(mean_anos):
    #     h_pred = A_pred * np.exp(1j * phi_pred)
    #     h_actual = amplitude_actual * np.exp(1j * phase_actual)
    #     mismatch = 1 - np.abs(np.vdot(h_pred, h_actual))**2 / (np.vdot(h_pred, h_pred) * np.vdot(h_actual, h_actual))
    #     print(f"Ecc={ecc}, Mean Anomaly={mean_ano:.2f}, Mismatch={mismatch:.3e}")

