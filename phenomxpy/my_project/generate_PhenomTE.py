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

from gw_eccentricity import measure_eccentricity
from gw_eccentricity import get_available_methods
from gw_eccentricity.plot_settings import use_fancy_plotsettings, labelsDict

import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)
plt.close('all')
plt.style.use('default')


warnings.simplefilter("once")
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")



plt.switch_backend('WebAgg')

class Simulate_Inspiral:
    """ Simulates time-domain (2,2) mode EOB waveform of a binary blackhole merger. Generates time-domain from starting frequency (f_lower) till peak at t=0 for time in geometric units. """
    
    def __init__(self, time_array, ecc_ref, total_mass, luminosity_distance, f_lower=10, f_ref=20, chi1=0, chi2=0, phiRef=0., rel_anomaly=0., inclination=0., mean_anomaly_ref=0., truncate_at_ISCO=True, truncate_at_tmin=True):
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
        self.original_time = np.copy(self.time) # Keep original time array for reference
        
        self.f_lower =  f_lower# Start frequency [Hz]
        self.f_ref = f_ref # Reference frequency at which the waveform parameters (eccentricity, spin, ...) are defined.
        self.total_mass = total_mass # Total mass of the binary [M]
        self.luminosity_distance = luminosity_distance # Luminosity distance of the binary [Mpc]
        self.chi1 = chi1 # Dimensionless spin of primary [float,ndarray]. If float, interpreted as z component
        self.chi2 = chi2 # Dimensionless spin of secondary [float,ndarray]. If float, interpreted as z component
        self.inclination = inclination # Inclination angle of the binary system [rad]
        self.phiRef = phiRef # Reference phase of the waveform [rad]
        self.ecc_ref = ecc_ref # eccentricity of binary at start frequency
        self.mean_anomaly_ref = mean_anomaly_ref # Mean anomaly at reference frequency [rad]

        self.truncate_at_ISCO = truncate_at_ISCO
        self.truncate_at_tmin = truncate_at_tmin

        self.hp_ecc = None # TimeSeries object of plus polarisation
        self.hc_ecc = None # TimeSeries object of cross polarisation
        self.amp_ecc = None # Amplitude of the waveform
        self.phase_ecc = None # Phase of the waveform

        self.hp_circ = None # TimeSeries object of plus polarisation for non-eccentric inspiral 
        self.hc_circ = None # TimeSeries object of cross polarisation for non-eccentric inspiral 
        self.phase_circ = None # Phase of the non-eccentric inspiral waveform
        self.amp_circ = None # Amplitude of the non-eccentric inspiral waveform

        self.mean_anomaly = None # Mean anomaly for self.time [rad]
        self.eccentricity = None # Eccentricity for self.time [dimensionless]
    
    def simulate_inspiral(self, total_mass=None, luminosity_distance=None, custom_time_array=None, ecc_ref=None, mean_ano_ref=None, truncate_at_ISCO=False, truncate_at_tmin=False, geometric_units=True, plot_polarisations=False, save_fig_polarisations=False, plot_ISCO_cut_off=False, save_fig_ISCO_cut_off=False):
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

        start = timer()

        reference_total_mass = 60
        if (geometric_units is False) and ((total_mass is not None) or (luminosity_distance is not None)):
            f_ref_geom = HztoMf(self.f_ref, reference_total_mass)
            f_lower_geom = HztoMf(self.f_lower, reference_total_mass)

            f_ref = MftoHz(f_ref_geom, total_mass)
            f_lower = MftoHz(f_lower_geom, total_mass)
        elif (geometric_units is False) and ((total_mass is None) or (luminosity_distance is None)):
            warnings.warn(self.colored_text("For geometric_units=False, both total_mass and luminosity_distance must be provided."), 'red')
            f_ref = self.f_ref
            f_lower = self.f_lower
        elif (geometric_units is True) and ((total_mass is not None) or (luminosity_distance is not None)):
            total_mass = None
            luminosity_distance = None
            warnings.warn(self.colored_text("total_mass and luminosity_distance are ignored when geometric_units=True.", 'red'))
            f_ref = self.f_ref
            f_lower = self.f_lower
        else:
            total_mass = 60 # Reference total mass for geometric frequency calculations [M_sun] in case of geometric units
            f_ref = self.f_ref
            f_lower = self.f_lower

        phen = phenomt.PhenomTE(
            mode=[2,2],
            times=time_array,
            eccentricity=ecc_ref if ecc_ref is not None else self.ecc_ref,  
            total_mass=reference_total_mass if total_mass is None else total_mass, # Used as reference total_mass to calculate geometric frequency during the waveform generation.           
            f_ref=f_ref,                   
            f_lower=f_lower,
            phiRef=self.phiRef,
            inclination=self.inclination,
            mean_anomaly = mean_ano_ref if mean_ano_ref is not None else self.mean_anomaly_ref)
        
        if geometric_units is False:
            phen.compute_polarizations(times=time_array, total_mass=total_mass, distance=luminosity_distance)
        else:
            phen.compute_polarizations(times=time_array)

        if phen.pWF.tmin > time_array[0]:
            warnings.warn(self.colored_text(
                "t_min is larger than parts of the specified time-domain, resulting in unphysical waveforms. "
                "Either use the truncate_tmin=True setting to automatically truncate to physical start of the time-domain "
                "or adjust the time-array manually to start at higher values."
            , 'red'))
            # mask to only include the physical range of the time-domain
            if (self.truncate_at_tmin is True) and (truncate_at_tmin is True):
                mask = time_array >= phen.pWF.tmin

                time_array = time_array[mask]
                phen.hp = phen.hp[mask]
                phen.hc = phen.hc[mask]

                print(self.colored_text(f'NEW TIME-DOMAIN after truncate at tmin (in geometric units): [{int(time_array[0])}, {int(time_array[-1])}] M', 'green'))
                del mask # clear memory

        # True because it's smallest truncated waveform AND true because the surrogate is called with the ISCO cut-off.
        if (self.truncate_at_ISCO is True) and (truncate_at_ISCO is True):
            # Truncate the waveform at ISCO frequency
            idx_cut = self.truncate_waveform_at_ISCO(phen, time_array, plot_ISCO_cut_off=plot_ISCO_cut_off, save_fig_ISCO_cut_off=save_fig_ISCO_cut_off)
            time_array = time_array[:idx_cut]

        # print(f'time : SimInspiral_M_independent ecc = {round(ecc_ref, 3)}, len = {len(phen.hp)}, M = {self.total_mass}, lum_dist={self.luminosity_distance}, t=[{int(time_array[0])}, {int(time_array[-1])}, num={len(time_array)}], f_lower={self.f_lower}, f_ref={self.f_ref} | computation time = {(timer()-start)} seconds')

        if plot_polarisations is True:
            self._plot_polarisations(phen.hp, phen.hc, time_array, save_fig_polarisations)
        
        

        if custom_time_array is None:
            self.time = time_array
            self.hp_ecc, self.hc_ecc = phen.hp, phen.hc
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
        plt.plot(self.time[-len(hc):], hc, label=r'$h_{\times}$', linewidth=0.6)

        plt.legend(loc = 'upper left')
        plt.xlabel('t [s]')
        plt.ylabel('$h_{22}]$')
        plt.title(f'M={self.total_mass}, e={round(self.ecc_ref, 3)}, f_min={self.f_lower} Hz')
        plt.grid(True)

        plt.tight_layout()
        # plt.show()

        if save_fig is True:
            figname = 'Images/Polarisations/Polarisations_M={}_ecc_ref={}_mean_ano_ref={}.png'.format(self.total_mass, round(self.ecc_ref, 3), round(self.mean_anomaly_ref, 2))            
            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Images/Polarisations', exist_ok=True)
            fig.savefig(figname, dpi=300, bbox_inches='tight')

            print(self.colored_text(f'Figure is saved in {figname}', 'blue'))

        # plt.close('all')
        
    
    def truncate_waveform_at_ISCO(self, phen, time_array, plot_ISCO_cut_off=False, save_fig_ISCO_cut_off=False):
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
            print(self.colored_text(f'NEW TIME-DOMAIN after truncate at ISCO (in geometric units): [{int(self.time[0])}, {int(self.time[idx_cut])}] M', 'green'))


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
            plt.plot(time_array[:idx_cut], np.linspace(min(Mf), max(Mf), num=500), linestyle='--', label=f't_ISCO = {time_array[idx_cut]}', color='black', linewidth=0.6)
            plt.legend()

            if save_fig_ISCO_cut_off is True:
                os.makedirs('Images/ISCO', exist_ok=True)  # Ensure the directory exists

                figname = 'Images/ISCO/truncate_at_ISCO_vs_Mf_M={}_e={}_f_lower={}.svg'.format(self.total_mass, round(phen.pWF.eccentricity, 2), self.f_lower)
                ISCO_vs_Mf_after.savefig(figname, dpi=300, bbox_inches='tight')
                # plt.close('all') 
                print(self.colored_text(f'Figure is saved in {figname}', 'blue'))
            
        # Clean memory
        del phase, dphi_dt, Mf, f_isco, above_isco

        return idx_cut
    

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
        
        
    






class Waveform_Properties(Simulate_Inspiral):
    """
    Calculates and plots residuals (residual = eccentric - non-eccentric) of waveform properties: amplitude, phase and frequency.
    """

    def __init__(self, time_array, ecc_ref, total_mass, luminosity_distance, f_lower=10, f_ref=20, mean_anomaly_ref=0, chi1=0, chi2=0, phiRef=0., rel_anomaly=0., inclination=0., truncate_at_ISCO=True, truncate_at_tmin=True):
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

        # Inherit parameters from Simulate_Inspiral class
        Simulate_Inspiral.__init__(self, time_array=time_array, ecc_ref=ecc_ref, total_mass=total_mass, luminosity_distance=luminosity_distance, f_lower=f_lower, f_ref=f_ref, mean_anomaly_ref=mean_anomaly_ref, chi1=chi1, chi2=chi2, phiRef=phiRef, rel_anomaly=rel_anomaly, inclination=inclination, truncate_at_ISCO=truncate_at_ISCO, truncate_at_tmin=truncate_at_tmin)


    def amplitude(self, hplus_geom, hcross_geom, geometric_units=True, luminosity_distance=None, total_mass=None):
        """
        Calculate the amplitude from the plus and cross polarizations.
        If geometric_units is True, the amplitude is returned in geometric units.

        Parameters:
        hplus_geom and hcross_geom are the geometric units polarizations.
        """
        amp_geom = np.abs(hplus_geom - 1j * hcross_geom)

        if geometric_units:
            self.amp_ecc = amp_geom
            return amp_geom
        else:
            amp_SI = AmpNRtoSI(amp_geom, luminosity_distance, total_mass)
            self.amp_ecc = amp_SI
            return amp_SI

    def phase(self, hplus, hcross):
        """
        Calculate the phase from the plus and cross polarizations. Unitless.
        """
        phase = np.unwrap(np.arctan2(hcross, hplus))
        phase -= phase[0] # Normalize phase to start at zero, correcting for the initial phase offset.
        
        self.phase_ecc = phase
        return phase


    def polarisations(self, phase, amplitude, geometric_units=True, distance=None, total_mass=None, plot_polarisations=False, save_fig=False):
        """
        Calculate the plus and cross polarizations from the phase and amplitude.
        If geometric_units is True, the polarizations are returned in geometric units.
        """
        if geometric_units:
            hp = amplitude * np.cos(phase)
            hc = -amplitude * np.sin(phase)
        else:
            hp = AmpNRtoSI(amplitude, distance, total_mass) * np.cos(phase)
            hc = AmpNRtoSI(amplitude, distance, total_mass) * np.sin(phase)
            self.time = MasstoSecond(self.time, self.total_mass)
        
        if plot_polarisations is True:
            self._plot_polarisations(hp, hc, save_fig=save_fig)

        self.hp_ecc, self.hc_ecc = hp, hc
        return hp, hc
    
    def get_orbital_parameters(self, plot_orbital_parameters=False, save_fig_orbital_parameters=False, make_diagnostic_plots=False):
        """
        Compute mean anomaly from time array and waveform polarizations.
        Parameters:
        ----------------
        time_array [M], np.array : Time array in geometric units
        hp [dimensionless], np.array : Plus polarization of the waveform
        hc [dimensionless], np.array : Cross polarization of the waveform
        
        Returns:
        ----------------
        M [rad], np.array : Mean anomaly corresponding to the time array
        """
        # Circulair waveform with more extended time-domain to prevent gw_eccentricity length errors
        time_circ = np.arange(self.time[0] - 200, self.time[-1] + 200, step=self.time[1]-self.time[0])
        self.circulair_wf(custom_time_array=time_circ)

        if self.hp_ecc is None or self.hc_ecc is None:
            self.simulate_inspiral(truncate_at_tmin=True, truncate_at_ISCO=True)

        # Setup dataDict (Note the required format)
        h22_ecc = self.hp_ecc - 1j * self.hc_ecc
        h22_circ = self.hp_circ - 1j * self.hc_circ

        # fig, ax = plt.subplots(1,1, figsize=(12,5), sharex=True)
        # ax.plot(self.time, h22_ecc, label='Eccentric Amplitude')
        # ax.plot(time_circ, h22_circ, label='Circular Amplitude')
        # ax.set_ylabel('Amplitude')
        # ax.legend()
        # fig.savefig('test_h22_ecc_vs_circ.png', dpi=300, bbox_inches='tight')

        dataDict = {"t": self.time,
           "hlm": {(2, 2): h22_ecc},
           "t_zeroecc": time_circ, # Must be longer than ecc to avoid extrapolation errors in gw_eccentricity package
           "hlm_zeroecc": {(2, 2): h22_circ}}

        # Choose method and set tref_in to full time domain for full evolution computation
        method = "ResidualAmplitude"  # Chosen method to measure eccentricity/mean anomaly (most accurate possibility)
        tref_in = self.time

        return_dict = measure_eccentricity(tref_in=tref_in,
                                            method=method,
                                            dataDict=dataDict)
        
        # Object with all orbital parameters
        gwecc_object = return_dict["gwecc_object"]
        print(dir(gwecc_object))

        # Plots to show estimations by the gw_eccentricity package
        if make_diagnostic_plots is True:
            fig, ax = gwecc_object.make_diagnostic_plots()

        # Save in class object
        self.mean_anomaly, self.eccentricity = gwecc_object.mean_anomaly, gwecc_object.eccentricity

        if plot_orbital_parameters is True:
            if plot_orbital_parameters:

                fig_orbital_parameters, axs = plt.subplots(
                    3, 1,
                    sharex=True,
                    figsize=(8, 7),
                    constrained_layout=True
                )

                # --- Amplitude ---
                A = self.amplitude(self.hp_ecc, self.hc_ecc)
                axs[0].plot(gwecc_object.tref_in, A, lw=1.2)
                axs[0].set_ylabel(r'$A_{22}$')
                axs[0].grid(alpha=0.3)

                # --- Eccentricity ---
                axs[1].plot(
                    gwecc_object.tref_out,
                    gwecc_object.eccentricity,
                    lw=1.2
                )
                axs[1].set_ylabel(r'$e$')
                axs[1].grid(alpha=0.3)

                # --- Mean anomaly ---
                # Wrap to [0, 2π) to avoid visual jumps
                mean_anomaly = np.mod(gwecc_object.mean_anomaly, 2*np.pi)

                axs[2].plot(
                    gwecc_object.tref_out,
                    mean_anomaly,
                    lw=1.2,
                    color='green'
                )
                axs[2].set_ylabel(r'$\ell_{\mathrm{gw}}\;[\mathrm{rad}]$')
                axs[2].set_xlabel(r'Time $[M]$')
                axs[2].grid(alpha=0.3)

                plt.show()

            
            if save_fig_orbital_parameters is True:
                os.makedirs('Images/Orbital_Parameters', exist_ok=True)  # Ensure the directory exists
                figname = 'Images/Orbital_Parameters/Orbital_Parameters_vs_time_e_ref={}_l_ref={}f_lower={}_f_ref={}.png'.format(round(self.ecc_ref, 2), round(self.mean_anomaly_ref,2), self.f_lower, self.f_ref)
                fig_orbital_parameters.savefig(figname, dpi=300, bbox_inches='tight')
                
                print(self.colored_text(f'Figure is saved in {figname}', 'blue'))

        return gwecc_object
    
    def circulair_wf(self, custom_time_array=None):
        """
        Simulate plus and cross polarisations of NON-ECCENTRIC waveform Inspiral for t in units [M]. 
        Also saves the phase and amplitude accordingly.
       
        Returns:
        ----------------
        hp_circ [dimensionless], np.array: Time-domain plus polarisation of NON-ECCENTRIC waveform
        hc_circ [dimensionless], np.array: Time-domain cross polarisation of NON-ECCENTRIC waveform

        """
        time_array = custom_time_array if custom_time_array is not None else self.time

        if (self.phase_circ is None) or (self.amp_circ is None):
            self.hp_circ, self.hc_circ, _ = self.simulate_inspiral(ecc_ref=0, custom_time_array=time_array)
            
            self.phase_circ = self.phase(self.hp_circ, self.hc_circ)
            self.amp_circ = self.amplitude(self.hp_circ, self.hc_circ)

        elif self.amp_circ is not None and len(self.amp_circ) != len(time_array):
            # Truncate to match
            self.phase_circ = self.phase_circ[:len(time_array)]
            self.amp_circ = self.amp_circ[:len(time_array)]
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
            circ = self.phase_circ # non-eccentric case
            eccentric = self.phase(hp, hc) # eccentric case
            units = '[radians]'
            # Residual = circular - eccentric to prevent negative residual values
            residual = circ - eccentric # to prevent negative values
            # Warning for negative residual values

            if eccentric[1] < 0: # 
                warnings.warn(self.colored_text("Eccentric phase has negative starting values. This may not be expected for physical waveforms. This usually happens when the eccentric waveformlength is shorter than the chosen time array. Consider decreasing the time array length or decreasing the eccentricity.", 'red'))
            
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
    

# --- Simulation settings ---
sampling_frequency = 2048  # or 4096
duration = 4  # seconds
time_array = np.linspace(-duration, 0, int(sampling_frequency * duration))  # time in seconds

wp = Waveform_Properties(time_array=time_array, ecc_ref=0.2, total_mass=60, luminosity_distance=200, f_lower=10, f_ref=20, truncate_at_ISCO=True, truncate_at_tmin=True)
# hp, hc = wp.simulate_inspiral(geometric_units=True)
# wp.calculate_residual(hp=hp, hc=hc, property='phase', plot_residual=False, save_fig=False)
wp.get_orbital_parameters(plot_orbital_parameters=True, save_fig_orbital_parameters=True, make_diagnostic_plots=True)

# eccentricties = [0.1, 0.2, 0.3]  # grid of eccentricities
# # mean_anos = np.linspace(0, np.pi/2, 50)  # mean anomalies to test
# mean_anos = [0, np.pi, 2*np.pi]

# # --- Reference circular waveform ---
# si_0 = Simulate_Inspiral(time_array=time_array, ecc_ref=0, total_mass=60, luminosity_distance=200)
# hp_0, hc_0 = si_0.simulate_inspiral(truncate_at_ISCO=True, truncate_at_tmin=True, geometric_units=True)

# phase_0 = si_0.phase(hp_0, hc_0)
# amplitude_0 = si_0.amplitude(hp_0, hc_0)

# phases_res = np.zeros((len(mean_anos), 5000))
# amplitudes_res = np.zeros((len(mean_anos), 5000))

# # --- Loop over eccentricities ---
# for ecc in eccentricties:
#     # Loop over mean anomalies
#     for i, mean_ano in enumerate(mean_anos):
#         si = Simulate_Inspiral(time_array=time_array, ecc_ref=ecc, mean_anomaly_ref=mean_ano, total_mass=60, luminosity_distance=200, truncate_at_tmin=True, truncate_at_ISCO=True)

 
#         # --- Generate actual waveform ---
#         hp_actual, hc_actual = si.simulate_inspiral(geometric_units=True, truncate_at_ISCO=True, truncate_at_tmin=True)
        
#         phase_actual = si.phase(hp_actual, hc_actual)
#         amplitude_actual = si.amplitude(hp_actual, hc_actual)

