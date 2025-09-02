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


plt.switch_backend('Agg')

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

        self.time = SecondtoMass(time_array, total_mass) # Time array in geometric units c=G=M=1
        self.ecc_ref = ecc_ref # eccentricity of binary at start frequency
        self.f_lower =  f_lower# Start frequency [Hz]
        self.f_ref = f_ref # Reference frequency at which the waveform parameters (eccentricity, spin, ...) are defined.
        self.total_mass = total_mass # Total mass of the binary [M]
        self.luminosity_distance = luminosity_distance # Luminosity distance of the binary
        self.chi1 = chi1 # Dimensionless spin of primary [float,ndarray]. If float, interpreted as z component
        self.chi2 = chi2 # Dimensionless spin of secondary [float,ndarray]. If float, interpreted as z component
        self.rel_anomaly= rel_anomaly # Relativistic anomaly. Radial phase which parametrizes the orbit within the Keplerian (relativistic) parametrization. Defaults to 0 (periastron)
        self.inclination = inclination # Inclination angle of the binary system [rad]
        self.phiRef = phiRef # Reference phase of the waveform [rad]
        self.truncate_at_ISCO = truncate_at_ISCO
        self.truncate_at_tmin = truncate_at_tmin
        self.mean_anomaly_start = mean_anomaly_start


    def amplitude(self, hplus_NR, hcross_NR, geometric_units=True, distance=None, total_mass=None):
        """
        Calculate the amplitude from the plus and cross polarizations.
        If geometric_units is True, the amplitude is returned in geometric units.
        """
        amp_NR = np.abs(hplus_NR - 1j * hcross_NR)
        
        if geometric_units:
            return amp_NR
        else:
            return AmpNRtoSI(amp_NR, distance, total_mass)

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
        print(len(phase), len(self.time))
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
            ISCO_vs_Mf_after.savefig('Images/ISCO/truncate_at_ISCO_vs_Mf_M={}_e={}_f_lower={}.svg'.format(self.total_mass, round(phen.pWF.eccentricity, 2), self.f_lower), dpi=300, bbox_inches='tight')
            plt.close('all') 
            print('Figure is saved in Images/ISCO')
            
        # Clean memory
        del phase, dphi_dt, Mf, f_isco, above_isco

        return idx_cut
        
      
    def simulate_inspiral_mass_independent(self, ecc_ref=None, custom_time_array=None, plot_polarisations=False, save_fig=False, truncate_at_ISCO=False):
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

        phen = phenomt.PhenomTE(
            mode=[2,2],
            times=time_array,
            eccentricity=ecc_ref,  
            total_mass=self.total_mass,
            distance=self.luminosity_distance,                
            f_ref=self.f_ref,                   
            f_lower=self.f_lower,
            phiRef=self.phiRef,
            inclination=self.inclination,
            mean_anomaly = self.mean_anomaly_start)
        

        phen.compute_polarizations(times=time_array)

        
        if phen.pWF.tmin > time_array[0]:
            warnings.warn(
                "t_min is larger than parts of the specified time-domain, resulting in unphysical waveforms. "
                "Either use the truncate_tmin=True setting to automatically truncate to start from t_min=time_array[0] "
                "or adjust the time-array manually to start at higher values."
            )
            # mask to only include the physical range of the time-domain
            if self.truncate_at_tmin is True:
                mask = time_array >= phen.pWF.tmin

                time_array = time_array[mask]
                phen.hp = phen.hp[mask]
                phen.hc = phen.hc[mask]

                print(f'NEW TIME-DOMAIN (in geometric units): [{int(time_array[0])}, {int(time_array[-1])}] M')
                del mask # clear memory

        # True because it's smallest truncated waveform AND true because the surrogate is called with the ISCO cut-off.
        if (truncate_at_ISCO is True) and (self.truncate_at_ISCO is True):
            # Truncate the waveform at ISCO frequency
            idx_cut = self.truncate_waveform_at_isco(phen, time_array)
            time_array = time_array[:idx_cut]
            

        print(f'time : SimInspiral_M_independent ecc = {round(ecc_ref, 3)}, M = {self.total_mass}, t=[{int(time_array[0])}, {int(time_array[-1])}, num={len(time_array)}] | computation time = {(timer()-start)} seconds')

        if plot_polarisations is True:
            self._plot_polarisations(phen.hp, phen.hc, time_array, save_fig=save_fig)

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

        plt.plot(self.time[-len(hp):], hp, label = f'$h_+$', linewidth=0.6)
        # plt.plot(self.time[:len(hc)], hc, label = f'$h_\times$', linewidth=0.6)

        plt.legend(loc = 'upper left')
        plt.xlabel('t [s]')
        plt.ylabel('$h_{22}]$')
        plt.title(f'M={self.total_mass}, e={round(self.ecc_ref, 3)}, f_min={self.f_lower} Hz')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        if save_fig is True:
            figname = 'Polarisations_M={}_ecc={}.png'.format(self.total_mass, round(self.ecc_ref, 3))
            
            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Images/Polarisations', exist_ok=True)
            fig.savefig('Images/Polarisations/' + figname, dpi=300, bbox_inches='tight')

            print('Figure is saved in Images/Polarisations')

        plt.close('all')


    





class Waveform_Properties_l(Simulate_Inspiral):
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
        self.l_grid = None

        # Inherit parameters from Simulate_Inspiral class
        Simulate_Inspiral.__init__(self, time_array, ecc_ref, total_mass, luminosity_distance, f_lower, f_ref, chi1, chi2, phiRef, rel_anomaly, inclination, truncate_at_ISCO, truncate_at_tmin)
    
    def compute_mean_anomaly(self, hp_t, hc_t):
        # Compute instantaneous frequency as derivative of phase
        dt = np.gradient(self.time)
        phase = self.phase(hp_t, hc_t)
        omega = np.gradient(phase) / dt  # omega(t) = dphi/dt

        # Integrate omega over time to get mean anomaly
        # cumtrapz returns one element less, so prepend initial mean anomaly
        # Approximate cumulative integral using cumsum:
        M = self.mean_anomaly_start + np.insert(np.cumsum(0.5 * (omega[1:] + omega[:-1]) * np.diff(self.time)), 0, 0)

        return M
    
    def circulair_wf(self):
        """
        Simulate plus and cross polarisations of NON-ECCENTRIC waveform Inspiral for t in units [M]. 
        Also saves the phase and amplitude accordingly.
       
        Returns:
        ----------------
        hp_circ [dimensionless], np.array: Time-domain plus polarisation of NON-ECCENTRIC waveform
        hc_circ [dimensionless], np.array: Time-domain cross polarisation of NON-ECCENTRIC waveform

        """
        
        if self.phase_circ is None or self.amp_circ is None:
            print('checking if hp_circ and hc_circ are set')
            # Generate arrays
            self.hp_circ, self.hc_circ = self.simulate_inspiral_mass_independent(ecc_ref=0)


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

            mean_anomaly_time_series = self.compute_mean_anomaly(hp_t=hp, hc_t=hc)

            test_mean_ano, axs = plt.subplots(2)
            # plt.plot(self.time, mean_anomaly_time_series)
            axs[0].plot(mean_anomaly_time_series, residual)
            axs[1].plot(self.time, residual)
            test_mean_ano.savefig('Images/mean_ano.png')

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

            plt.close('all')  # Clean up plots

            if save_fig is True:
                figname = f'Residual {property} M={self.total_mass}, ecc={round(ecc_ref, 3)}.png'
                
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Residuals', exist_ok=True)
                fig_residual.savefig('Images/Residuals/' + figname, dpi=300, bbox_inches='tight')

                print('Figure is saved in Images/Residuals')
        
        del circ, eccentric # clear memory

        return residual
    


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
        Simulate_Inspiral.__init__(self, time_array, ecc_ref, total_mass, luminosity_distance, f_lower, f_ref, chi1, chi2, phiRef, rel_anomaly, inclination, truncate_at_ISCO, truncate_at_tmin)
        
    def circulair_wf(self):
        """
        Simulate plus and cross polarisations of NON-ECCENTRIC waveform Inspiral for t in units [M]. 
        Also saves the phase and amplitude accordingly.
       
        Returns:
        ----------------
        hp_circ [dimensionless], np.array: Time-domain plus polarisation of NON-ECCENTRIC waveform
        hc_circ [dimensionless], np.array: Time-domain cross polarisation of NON-ECCENTRIC waveform

        """
        
        if self.phase_circ is None or self.amp_circ is None:
            print('checking if hp_circ and hc_circ are set')
            # Generate arrays
            self.hp_circ, self.hc_circ = self.simulate_inspiral_mass_independent(ecc_ref=0)
            
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

            plt.close('all')  # Clean up plots

            if save_fig is True:
                figname = f'Residual {property} M={self.total_mass}, ecc={round(ecc_ref, 3)}.png'
                
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Residuals', exist_ok=True)
                fig_residual.savefig('Images/Residuals/' + figname, dpi=300, bbox_inches='tight')

                print('Figure is saved in Images/Residuals')
        
        del circ, eccentric # clear memory

        return residual


    
# sampling_frequency = 2048 # or 4096
# duration = 4 # seconds
# time_array = np.linspace(-duration, 0, int(sampling_frequency * duration))  # time in seconds
# wp = Waveform_Properties_l(time_array=time_array, ecc_ref=0.2, total_mass=60, luminosity_distance=200, truncate_at_ISCO=False)
# hp, hc = wp.simulate_inspiral_mass_independent()

# wp.calculate_residual(hp, hc, property='amplitude', plot_residual=True, save_fig=True)
# wp.calculate_residual(hp, hc, property='phase', plot_residual=True, save_fig=True)

# plt.show()


