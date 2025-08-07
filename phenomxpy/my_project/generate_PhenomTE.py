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
    
    def __init__(self, time_array, ecc_ref, total_mass, luminosity_distance, f_lower=10, f_ref=20, chi1=0, chi2=0, phiRef=0., rel_anomaly=0., inclination=0., truncate_at_ISCO=True, truncate_at_tmin=True):
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


    def polarisations(self, phase, amplitude, geometric_units=True, distance=None, total_mass=None):
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
        
        return hplus, hcross
    
    def truncate_near_merger(self, phen):
        cparams = phen.pWF.cparams
        e = phen.pWF.e0
        e2 = e**2
        tchirp_ecccorr = (1 - e2)**3.5 / (1 + (157/24)*e2 + (605/96)*e2**2)

        # Reference inspiral time in code units (e.g., from phen.pWF.tchirp if available)
        # Or set a fixed value if not available
        tchirp_ref = cparams.get("tchirp", abs(phen.pWF.tmin))  # geometric units

        # Base extra time you want (e.g., 1.0 means keep the whole waveform)
        base_fraction = cparams.get("extra_time_fraction", 1.0)
        textra = base_fraction * tchirp_ref * tchirp_ecccorr + cparams.get("textra", 0.0)

        # Optional: truncate waveform to a maximum allowed window (e.g., 1/20 of full time)
        tmax = abs(phen.pWF.tmin) / 20.0
        textra = min(textra, tmax)

        # Now apply your taper or truncate logic
        hp, hc = time_array_condition_stage1(
            hp,
            hc,
            delta_t=params['delta_t'],  # geometric units
            textra=textra,
            f_lower=cparams['original_f_lower']
        )

    def truncate_waveform_at_isco(self, phen, plot_ISCO_cut_off=True):
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
        dphi_dt = np.gradient(phase, self.time)
        Mf = dphi_dt / (2 * np.pi)

        # print(phen.hp, dphi_dt, Mf, round(phen.pWF.eccentricity, 2))

        # Calculate ISCO frequency (dimensionless): Mf_ISCO = 1 / (6^(3/2) * π) ≈ 0.021
        f_isco = 1 / (6**1.5 * np.pi)  # dimensionless Mf_ISCO


        # ISCO_fig = plt.figure(figsize=(12, 5))

        # zero_crossings = np.where((-0.001 <= dphi_dt) & (dphi_dt <= 0.001))[0]
        # print(zero_crossings, 'zero crossing')

        # hp_circ, hc_circ = self.simulate_inspiral_mass_independent(ecc_ref=0, truncate_at_ISCO=False)
        # phase_circ = self.phase(hp_circ, hc_circ)

        # phase_diff = phase - phase_circ

        # # Find maxima of phase difference (eccentricity peaks)
        # peaks_max, _ = find_peaks(phase_diff)
        # t_max = self.time[peaks_max]
        # phi_max = phase_diff[peaks_max]
        
        # # Find minima (invert signal to find troughs)
        # peaks_min, _ = find_peaks(-phase_diff)
        # t_min = self.time[peaks_min]
        # phi_min = phase_diff[peaks_min]
        
        # # Fit models -------------------------------------------------
        # def max_model(t, a, n, t_c, b):
        #     """Power-law model for maxima"""
        #     return a * (t_c - t)**n + b
            
        # def min_model(t, a, n, t_c, b):
        #     """Power-law model for minima"""
        #     return -a * (t_c - t)**n + b
        
        # # Fit maxima
        # p0_max = [1, -0.1, max(t_max)+100, 0]  # Initial guesses
        # params_max, _ = curve_fit(max_model, t_max, phi_max, p0=p0_max)
        
        # # Fit minima 
        # p0_min = [1, -0.1, max(t_min)+100, 0]
        # params_min, _ = curve_fit(min_model, t_min, phi_min, p0=p0_min)
        
        # # Generate fitted curves
        # t_fit = np.linspace(min(self.time), max(self.time), 1000)
        # fit_max = max_model(t_fit, *params_max)
        # fit_min = min_model(t_fit, *params_min)



        # # plt.plot(self.time, np.gradient(np.gradient(phase, self.time), self.time), label='Mf', linewidth=0.6)
        # plt.plot(self.time, phase)
        # # plt.plot(self.time, power_law(self.time, *p0))
        # plt.scatter(self.time[peaks_max], phase[peaks_max])
        # plt.scatter(self.time[peaks_min], phase[peaks_min])
        # plt.plot(t_fit, fit_max, "r--", label="Fit max (power law)")
        # plt.plot(t_fit, fit_min, "b--", label="Fit min (power law)")

        # # plt.scatter(self.ti)
        # # print('phase', dphi_dt, round(phen.pWF.eccentricity, 2))
        # # if len(zero_crossings) != 0:
        # #     zero_crossing_median = int(zero_crossings[0] - zero_crossings[-1]/2)
        # #     plt.scatter(self.time[zero_crossings], dphi_dt[zero_crossings])
        # # plt.axhline(f_isco, color='red', linestyle='--', label='Mf_ISCO_circ', linewidth=0.6)
        # plt.xlabel('t [M]')
        # plt.ylabel('Mf')
        # plt.title(f'Mf vs Time for M={self.total_mass}, e={round(phen.pWF.eccentricity, 2)}, f_lower={self.f_lower} Hz')
        # plt.legend()
        # plt.grid()

        # os.makedirs('Images/ISCO', exist_ok=True)  # Ensure the directory exists
        # ISCO_fig.savefig('Images/ISCO/ISCO_gradient_Mf_vs_Time_M={}_e={}_f_lower={}.png'.format(self.total_mass, round(phen.pWF.eccentricity, 2), self.f_lower), dpi=300, bbox_inches='tight')
        # plt.close('all')  # Clean up plots



        above_isco = np.where(Mf >= f_isco)[0]

        # ISCO_vs_Mf = plt.figure()

        # plt.plot(self.time, Mf)
        # plt.axhline(f_isco, color='red', linestyle='--', label='Mf_ISCO_circ', linewidth=0.6)
        # plt.scatter(self.time[above_isco], Mf[above_isco], color='r', s=4)

        # os.makedirs('Images/ISCO', exist_ok=True)  # Ensure the directory exists
        # ISCO_vs_Mf.savefig('Images/ISCO/before_ISCO_vs_Mf_M={}_e={}_f_lower={}.png'.format(self.total_mass, round(phen.pWF.eccentricity, 2), self.f_lower), dpi=300, bbox_inches='tight')
        # plt.close('all')  # Clean up plots

        # print(above_isco, 'above ISCO')
        if len(above_isco) == 0:
            idx_cut = len(self.time) # in case there is no ISCO wihtin the specified time-array range
        else:
            idx_cut = above_isco[0]
            print(f'Waveform time-domain has been cut to match only-inspiral waveforms up till (circulair) ISCO.\nNEW TIME-DOMAIN (in geometric units): [{int(self.time[0])}, {int(self.time[idx_cut])}] M')


        # Truncate waveform
        # Only cut hp, hc, not self.time, because self.time does not recover for a new waveform run. Adjust self.time after shortest waveform has been established.
        phen.hp = phen.hp[:idx_cut]
        phen.hc = phen.hc[:idx_cut]
        if idx_cut == len(self.time):
            plot_ISCO_cut_off = False

        if plot_ISCO_cut_off is True:
            ISCO_vs_Mf_after = plt.figure(figsize=(12,5))

            plt.plot(self.time, Mf, label=f'Mf before ISCO cut: $e$={round(phen.pWF.eccentricity, 3)}')
            plt.axhline(f_isco, color='red', linestyle='--', label='Mf ISCO $e$=0', linewidth=0.6)
            plt.scatter(self.time[idx_cut], Mf[idx_cut], color='r', s=6, label='ISCO cut')
            plt.plot(self.time[:idx_cut], Mf[:idx_cut], label=f'Mf after ISCO cut: $e$={round(phen.pWF.eccentricity, 3)} ')
            plt.legend()

            os.makedirs('Images/ISCO', exist_ok=True)  # Ensure the directory exists
            ISCO_vs_Mf_after.savefig('Images/ISCO/truncate_at_ISCO_vs_Mf_M={}_e={}_f_lower={}.svg'.format(self.total_mass, round(phen.pWF.eccentricity, 2), self.f_lower), dpi=300, bbox_inches='tight')
            plt.close('all') 
            print('Figure is saved in Images/ISCO')
            
        # Clean memory
        del phase, dphi_dt, Mf, f_isco, above_isco

        return idx_cut
        
      
    def simulate_inspiral_mass_independent(self, ecc_ref=None, plot_polarisations=False, save_fig=False, truncate_at_ISCO=False):
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

        # Either set ecc_ref specifically or use the class defined value
        if ecc_ref is None:
            ecc_ref = self.ecc_ref
        else:
            ecc_ref = ecc_ref

        # To compute the runtime for 1 simulated waveform
        start = timer()

        phen = phenomt.PhenomTE(
            mode=[2,2],
            times=self.time,
            eccentricity=ecc_ref,  
            total_mass=self.total_mass,
            distance=self.luminosity_distance,                
            f_ref=self.f_ref,                   
            f_lower=self.f_lower,
            phiRef=self.phiRef,
            inclination=self.inclination)
        
        phen.compute_polarizations(times=self.time)

        
        if phen.pWF.tmin > self.time[0]:
            warnings.warn(
                "t_min is larger than parts of the specified time-domain, resulting in unphysical waveforms. "
                "Either use the truncate_tmin=True setting to automatically truncate to start from t_min=time_array[0] "
                "or adjust the time-array manually to start at higher values."
            )
            # mask to only include the physical range of the time-domain
            if self.truncate_at_tmin is True:
                mask = self.time >= phen.pWF.tmin

                self.time = self.time[mask]
                phen.hp = phen.hp[mask]
                phen.hc = phen.hc[mask]

                print(f'NEW TIME-DOMAIN (in geometric units): [{int(self.time[0])}, {int(self.time[-1])}] M')
                del mask # clear memory

        # True because it's smallest truncated waveform AND true because the surrogate is called with the ISCO cut-off.
        if (truncate_at_ISCO is True) and (self.truncate_at_ISCO is True):
            # Truncate the waveform at ISCO frequency
            idx_cut = self.truncate_waveform_at_isco(phen)
            self.time = self.time[:idx_cut]
            

        print(f'time : SimInspiral_M_independent ecc = {round(ecc_ref, 3)}, M = {self.total_mass}, t=[{int(self.time[0])}, {int(self.time[-1])}, num={len(self.time)}] | computation time = {(timer()-start)} seconds')

        if plot_polarisations is True:

            fig_simulate_inspiral = plt.figure(figsize=(12,5))
            print(len(self.time), len(phen.hp))
            plt.plot(self.time[:len(phen.hp)], phen.hp, label = f'$h_+$', linewidth=0.6)
            plt.plot(self.time[:len(phen.hp)], phen.hc, label = f'$h_\times$', linewidth=0.6)

            plt.legend(loc = 'upper left')
            plt.xlabel('t [s]')
            plt.ylabel('$h_{22}]$')
            plt.title(f'M={self.total_mass}, e={round(ecc_ref, 3)}, f_min={self.f_lower} Hz')
            plt.grid(True)

            plt.tight_layout()

            if save_fig is True:
                figname = 'Polarisations_M={}_ecc={}.png'.format(self.total_mass, round(ecc_ref, 3))
                
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Polarisations', exist_ok=True)
                fig_simulate_inspiral.savefig('Images/Polarisations/' + figname, dpi=300, bbox_inches='tight')

                print('Figure is saved in Images/Polarisations')

            plt.close('all')  # Clean up plots

        return phen.hp, phen.hc








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
        if self.hp_circ is None:
            self.hp_circ, self.hc_circ = self.simulate_inspiral_mass_independent(ecc_ref=0)

            self.phase_circ = self.phase(self.hp_circ, self.hc_circ)
            self.amp_circ = self.amplitude(self.hp_circ, self.hc_circ)

        elif (self.hp_circ is not None) and (len(self.hp_circ) != len(self.time)):
                length_diff = len(self.hp_circ) - len(self.time)
                self.hp_circ = self.hp_circ[length_diff:]
                self.hc_circ = self.hc_circ[length_diff:]

                self.phase_circ = self.phase(self.hp_circ, self.hc_circ)
                self.amp_circ = self.amplitude(self.hp_circ, self.hc_circ)

                del length_diff # clear memory
        else:
            pass # hp_circ and hc_circ already calculated

        

    def calculate_residual(self, hp, hc, ecc_ref, property=None, plot_residual=False, save_fig=False):
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
# wp = Waveform_Properties(time_array=time_array, ecc_ref=0.2, total_mass=60, luminosity_distance=200, truncate_at_ISCO=False)
# hp, hc = wp.simulate_inspiral_mass_independent()
# hp, hc = wp.simulate_inspiral_mass_independent(truncate_at_ISCO=False, plot_polarisations=True, save_fig=True)
# wp.calculate_residual(hp, hc, property='amplitude', plot_residual=True, save_fig=True)
# wp.calculate_residual(hp, hc, property='phase', plot_residual=True, save_fig=True)




