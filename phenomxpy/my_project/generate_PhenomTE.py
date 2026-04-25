from helper_functions import *

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("TkAgg")

from dataclasses import dataclass
from typing import Any

import warnings
warnings.simplefilter("once")
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

import phenomxpy.phenomt as phenomt
from phenomxpy.utils import SecondtoMass, AmpSItoNR, m1ofq, m2ofq, AmpNRtoSI, HztoMf, MftoHz, MasstoSecond

from gw_eccentricity import measure_eccentricity




@dataclass
class WaveformResult:
    """
    Stores the results of the latest waveform run without updating the instance objects of Simulate_Inspiral
    """
    hp: Any
    hc: Any
    time: Any
    ecc_ref: float
    f_ref: float
    f_lower: float
    geometric_units: bool
    mean_ano_ref: float
    mass_ratio: float


    mass1: float = None
    mass2: float = None
    total_mass: float = None
    chi1: float = None
    chi2: float = None
    luminosity_distance: float = None

class Simulate_Waveform(Warnings, Automated_Settings):
    """ 
    Simulates time-domain (2,2) mode EOB waveform of a binary blackhole merger. Generates time-domain from starting frequency (f_lower) till peak at t=0 for time in geometric units. 
    """
    
    def __init__(self, 
                 time_array, 
                 f_lower=10, 
                 f_ref=20, 
                 ecc_ref=None, 
                 mean_anomaly_ref=0., 
                 total_mass=None, 
                 luminosity_distance=None, 
                 mass_ratio=1, 
                 chi1=0, 
                 chi2=0, 
                 phiRef=0., 
                 inclination=0., 
                 truncate_at_ISCO=True, 
                 truncate_at_tmin=True, 
                 geometric_units=True):
        """
        Parameters:
        ----------------
        time_array [s], np.array : Time array in seconds.
        ecc_ref [dimensionless], float: Eccentricity of binary at start f_lower
        mean_anomaly_ref [rad], float : Mean anomaly at reference frequency f_ref
        total_mass [M_sun], int : Total mass of the binary in solar masses. =None for geometric units
        luminosity_distance [Mpc], float : Luminosity distance of the binary in megaparsec. =None for geometric units
        mass_ratio [dimensionless], float [1, inf] : Mass ratio of the binary, q >= 1
        f_lower [Hz], float: Start frequency of the waveform
        f_ref [Hz], float: Reference frequency at which the waveform parameters (eccentricity, spin, ...) are defined.
        chi1 [dimensionless], float, ndarray : Spin of primary. If float, interpreted as z component
        chi2 [dimensionless], float, ndarray : Spin of secondary. If float, interpreted as z component
        PhiRef = [rad], float : Reference phase of the waveform. 
        inclination [rad], float : Inclination angle of the binary system. Defaults to 0 (face-on).
        truncate_at_ISCO, True OR False, bool : If the waveform should be truncated at the ISCO frequency, set to True. 
        truncate_at_tmin, True OR False, bool : If the waveform should be truncated at the physical start of the time-domain (tmin), set to True.
        geometric_units, True OR False, bool : If the waveform should be generated in geometric units (c=G=M=1), set to True. For physical units, set to False and provide total_mass and luminosity_distance.
        """

        # Initial parameters
        if total_mass is None:
            total_mass = 60 # Reference total mass for geometric frequency calculations [M_sun]

        self.time = SecondtoMass(time_array, total_mass) # Time array in geometric units c=G=M=1
        self.original_time = np.copy(self.time) # Keep original time array for reference
        
        self.f_lower =  f_lower# Start frequency [Hz]
        self.f_ref = f_ref # Reference frequency at which the waveform parameters (eccentricity, spin, ...) are defined.
        self.total_mass = total_mass # Total mass of the binary [M]
        self.mass_ratio = mass_ratio # Mass ratio of the binary (m1/m2 >= 1)
        self.luminosity_distance = luminosity_distance # Luminosity distance of the binary [Mpc]
        self.chi1 = chi1 # Dimensionless spin of primary [float,ndarray]. If float, interpreted as z component
        self.chi2 = chi2 # Dimensionless spin of secondary [float,ndarray]. If float, interpreted as z component
        self.inclination = inclination # Inclination angle of the binary system [rad]
        self.phiRef = phiRef # Reference phase of the waveform [rad]
        self.ecc_ref = ecc_ref # eccentricity of binary at start frequency
        self.mean_ano_ref = mean_anomaly_ref # Mean anomaly at reference frequency [rad]

        self.truncate_at_ISCO = truncate_at_ISCO # If the waveform should be truncated at the ISCO frequency, set to True.
        self.truncate_at_tmin = truncate_at_tmin # If the waveform should be truncated at the physical start of the time-domain (tmin), set to True.

        self.geomtric_units = geometric_units   # Waveform in geometric units for True and SI units for False

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
    
        super().__init__() # Initialize Inherited classes

    def simulate_waveform(self, 
                          time_array=None,
                          f_ref=None, 
                          f_lower=None, 
                          ecc_ref=None, 
                          mean_ano_ref=None,
                          total_mass=None,
                          luminosity_distance=None,  
                          mass_ratio=None, 
                          chi1=None, 
                          chi2=None, 
                          phiRef=None, 
                          inclination=None, 
                          truncate_at_ISCO=False, 
                          truncate_at_tmin=False, 
                          geometric_units=True, 
                          plot_polarisations=False, 
                          save_fig_polarisations=False, 
                          plot_ISCO_cut_off=False, 
                          save_fig_ISCO_cut_off=False, 
                          update_results=False):
        """
        Simulate plus and cross polarisations of the eccentric IMRPhenomTE waveform (2,2) mode for a user-defined time array (waveform peak at t=0).
        
        Parameters:
        ----------------
        time_array [s], np.array : Time array in seconds.
        ecc_ref [dimensionless], float: Eccentricity of binary at start f_lower
        mean_anomaly_ref [rad], float : Mean anomaly at reference frequency f_ref
        total_mass [M_sun], int : Total mass of the binary in solar masses. =None for geometric units
        luminosity_distance [Mpc], float : Luminosity distance of the binary in megaparsec. =None for geometric units
        mass_ratio [dimensionless], float [1, inf] : Mass ratio of the binary, q >= 1
        f_lower [Hz], float: Start frequency of the waveform
        f_ref [Hz], float: Reference frequency at which the waveform parameters (eccentricity, spin, ...) are defined.
        chi1 [dimensionless], float, ndarray : Spin of primary. If float, interpreted as z component
        chi2 [dimensionless], float, ndarray : Spin of secondary. If float, interpreted as z component
        PhiRef = [rad], float : Reference phase of the waveform. 
        inclination [rad], float : Inclination angle of the binary system. Defaults to 0 (face-on).
        truncate_at_ISCO, True OR False, bool : If the waveform should be truncated at the ISCO frequency, set to True. 
        truncate_at_tmin, True OR False, bool : If the waveform should be truncated at the physical start of the time-domain (tmin), set to True.

        geometric_units, True OR False, bool : If the waveform should be generated in geometric units (c=G=M=1), set to True. For physical units, set to False and provide total_mass and luminosity_distance.
        plot_polarisations, True OR False, bool : For a plot of the plus and cross polarisations, set to True.
        save_polarisations, True Or False, bool : If plot of the polarisations should be saved to a automatically created folder \Images, set to True.
        plot_ISCO_cut_off, True OR False, bool : If the cut-off at ISCO frequency should be plotted in the instantaneous phase frequency evolution, set to True.
        save_fig_ISCO_cut_off, True OR False, bool : If the plot of the ISCO cut-off should be saved to a automatically created folder \Images\ISCO, set to True.
        
        update_results, True OR False, bool : If the class attributes should be updated with the new simulation results (time, hp, hc), set to True.
        
        Returns:
        ----------------
        hp [dimensionless], np.array: Time-domain plus polarisation 
        hc [dimensionless], np.array: Time-domain cross polarisation 
        """

        # ---------------------------- Parameter resolution and consistency checks --------------------------

        # Resolve parameters (if not specified, use instance objects or default values)
        total_mass, luminosity_distance = self._resolve_mass_distance(total_mass, luminosity_distance, geometric_units)
        time_array = self.resolve_property(prop=time_array, default=self.time)
        mass_ratio = self.resolve_property(prop=mass_ratio, default=self.mass_ratio)
        chi1 = self.resolve_property(prop=chi1, default=self.chi1)
        chi2 = self.resolve_property(prop=chi2, default=self.chi2)
        ecc_ref = self.resolve_property(prop=ecc_ref, default=self.ecc_ref)
        mean_ano_ref = self.resolve_property(prop=mean_ano_ref, default=self.mean_ano_ref)
        phiRef = self.resolve_property(prop=phiRef, default=self.phiRef)
        inclination = self.resolve_property(prop=inclination, default=self.inclination)
        f_ref = self.resolve_property(prop=f_ref, default=self.f_ref)
        f_lower = self.resolve_property(prop=f_lower, default=self.f_lower)
        truncate_at_ISCO = self.resolve_property(prop=truncate_at_ISCO, default=self.truncate_at_ISCO)
        truncate_at_tmin = self.resolve_property(prop=truncate_at_tmin, default=self.truncate_at_tmin)

        # If total mass not specified, fall back to PhenomTE reference_mass
        reference_total_mass = 60
        mass_for_conversion = self.resolve_property(prop=total_mass, default=reference_total_mass)

        # Calculate f_geom for conversion between total_mass vs Phenom reference_mass for geometric waveforms
        if geometric_units:
            f_ref, f_lower = self._convert_frequencies(
                total_mass=mass_for_conversion, reference_total_mass=60
            )

        # Calculate symmetric mass ratio eta from mass ratio q = m1/m2 >= 1
        eta = self._eta_from_mass_ratio(q=mass_ratio)

        # --------------------------- Generate waveform with PhenomTE --------------------------

        phen = phenomt.PhenomTE(
            mode=[2, 2],
            times=time_array, # time array in geometric units
            total_mass=mass_for_conversion, # total mass for geometric frequency conversion. If total_mass is not specified, fall back to reference mass of 60 M_sun
            eta=eta, # symmetric mass-ratio eta = [0, 0.25]
            s1=chi1, # spin of primary, dimensionless. If float, interpreted as z component
            s2=chi2, # spin of secondary, dimensionless. If float, interpreted as z component
            eccentricity=ecc_ref,
            f_ref=f_ref,
            f_lower=f_lower,
            phiRef=phiRef,
            inclination=inclination,
            mean_anomaly=mean_ano_ref,
        )

        # Geometric units vs SI units
        # compute_polarizations() only accepts inclination, phiRef, times, distance and total_mass variables. It silently ignores all other parameters.
        if geometric_units:
            phen.compute_polarizations(times=time_array)
        else:
            # for SI units, the distance and total mass are needed for the amplitude scaling, so they must be included in the compute_polarizations call. 
            phen.compute_polarizations(
                times=time_array,
                total_mass=phen.pWF.total_mass,
                distance=luminosity_distance,
            )
        
        # -------------------------- waveform truncation to physical lengths (tmin and ISCO) --------------------------

        # Mask for physical parts of the waveform (tmin - ISCO), initially set to all True (no truncation)
        valid_mask = np.ones_like(time_array, dtype=bool)

        # Check if the waveform is physical for the given time_array. Otherwise cut to physical lengths
        if truncate_at_tmin:
            if phen.pWF.tmin > time_array[0]:
                warnings.warn(self.colored_text(
                    "t_min is larger than parts of the specified time-domain, resulting in unphysical waveforms. "
                    "Either use the truncate_at_tmin=True setting to automatically truncate to physical start of the time-domain "
                    "or adjust the time-array manually to start at higher values."
                , 'red'))
                # mask to only include the physical range of the time-domain
                # Set to False for everything before tmin, effectively truncating the waveform to the physical start of the time-domain.
                valid_mask &= (time_array >= phen.pWF.tmin)

            print(self.colored_text(
                f'NEW TIME-DOMAIN after truncate at tmin (in geometric units): '
                f'[{int(time_array[valid_mask][0])}, {int(time_array[valid_mask][-1])}] M',
                'green'
            ))

        # True because it's smallest truncated waveform AND true because the surrogate is called with the ISCO cut-off.
        if truncate_at_ISCO:
            # Truncate the waveform at ISCO frequency
            idx_cut = self._compute_idx_ISCO(phen.hp[valid_mask], phen.hc[valid_mask], time_array[valid_mask], plot_ISCO_cut_off=plot_ISCO_cut_off, save_fig_ISCO_cut_off=save_fig_ISCO_cut_off)
            
            # Map masked indices back to full-array indices
            masked_indices = np.where(valid_mask)[0]

            # Turn off everything from idx_cut onward in the currently valid region
            valid_mask[masked_indices[idx_cut:]] = False

            print(self.colored_text(
                f'NEW TIME-DOMAIN after truncate at ISCO (in geometric units): '
                f'[{int(time_array[valid_mask][0])}, {int(time_array[valid_mask][-1])}] M', 
                'green'
            ))
        
            
        # Update the waveform and time array to only include the valid (physical) parts of the waveform. This is done after both truncation steps, so the final waveform is truncated to the shortest physical waveform.
        hp = phen.hp[valid_mask]
        hc = phen.hc[valid_mask]
        time_array = time_array[valid_mask]

        # ---------------------------------- Save waveform run results in WaveformResult object  --------------------------
        if geometric_units is False:
            time_array = MasstoSecond(time_array, total_mass)

        # print(f'time : SimInspiral_M_independent ecc = {round(ecc_ref, 3)}, len = {len(phen.hp)}, M = {self.total_mass}, lum_dist={self.luminosity_distance}, t=[{int(time_array[0])}, {int(time_array[-1])}, num={len(time_array)}], f_lower={self.f_lower}, f_ref={self.f_ref} | computation time = {(timer()-start)} seconds')
        
        self.last_result = WaveformResult(
            hp=hp,
            hc=hc,
            time=time_array,
            total_mass=total_mass,
            mass_ratio=mass_ratio,
            chi1=chi1,
            chi2=chi2,
            f_lower=f_lower,
            f_ref=f_ref,
            ecc_ref=ecc_ref,
            mean_ano_ref=mean_ano_ref,
            geometric_units=geometric_units,
            luminosity_distance=luminosity_distance,
        )


        # --------------------------- Plot polarizations --------------------------
        if plot_polarisations is True:
            self._plot_polarisations(save_fig=save_fig_polarisations)
        
        # --------------------------- Update instance attributes --------------------------
        if update_results is True:
            self.time = time_array
            print(10, len(self.time))
            self.hp_ecc, self.hc_ecc = hp, hc
        
        return hp, hc, time_array
        

    def _resolve_mass_distance(self, total_mass, luminosity_distance, geometric_units):
        """
        Checks for consistency between parameters (total_mass and luminosity_distance) and the choice of geometric units vs SI units. 
        If asked for SI units and total_mass and luminosity distance are not available, the function checks for existing instance objects.
        """
        if geometric_units:
            if total_mass is not None or luminosity_distance is not None:
                print(self.colored_text(
                    "total_mass and luminosity_distance are ignored when geometric_units=True.", 'red'
                ))
            return None, None

        total_mass = self.resolve_property(total_mass, self.total_mass)
        luminosity_distance = self.resolve_property(luminosity_distance, self.luminosity_distance)

        if total_mass is None or luminosity_distance is None:
            print(self.colored_text(
                "For geometric_units=False, both total_mass and luminosity_distance must be provided.", 'red'
            ))

        return total_mass, luminosity_distance


    def _convert_frequencies(self, total_mass, reference_total_mass=60):
        """
        Conversion for geometric reference frequency based on the total mass of the system. 
        PhenomTE always uses M_ref=60 for f_geom conversion, so given a different total_mass, this can give different waveform results due to incompatibility with the M_ref.
        PhenomTE also uses a reference mass for geometric systems! This is always needed for f_geom conversion.
        """
        # Convert f_ref and f_lower to geometric frequencies based on the reference total mass
        f_ref_geom = HztoMf(self.f_ref, reference_total_mass)
        f_lower_geom = HztoMf(self.f_lower, reference_total_mass)
        # Convert geometric frequencies to physical frequencies based on the actual total mass of the system
        f_ref = MftoHz(f_ref_geom, total_mass)
        f_lower = MftoHz(f_lower_geom, total_mass)

        return f_ref, f_lower


    def _eta_from_mass_ratio(self, q):
        # Calculate symmetric mass ratio eta from mass ratio q = m1/m2 >= 1
        eta = q / (1 + q)**2
        return eta
    

    def _plot_polarisations(self, save_fig=True):
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

        if not hasattr(self, "last_result"):
            raise ValueError("Run simulate_waveform() first.")

        # Load latest waveform run specifics
        r = self.last_result
            
        fig = plt.figure(figsize=(12,5))
      
        plt.plot(r.time[-len(r.hp):], r.hp, label = f'$h_+$', linewidth=0.6)
        plt.plot(r.time[-len(r.hc):], r.hc, label=r'$h_{\times}$', linewidth=0.6)

        plt.legend(loc = 'upper left')
        if r.geometric_units:
            plt.xlabel('t [M]')
            plt.title(f'e_ref={round(r.ecc_ref, 3)}, q={r.mass_ratio}, l_ref={round(r.mean_ano_ref, 3)}, chi1={r.chi1}, chi2={r.chi2}, f_min={round(r.f_lower, 2)} Hz, f_ref={round(r.f_ref, 2)} Hz')
        else:
            plt.xlabel('t [s]')
            plt.title(f'M={r.total_mass}, e_ref={round(r.ecc_ref, 3)}, q={r.mass_ratio}, l_ref={round(r.mean_ano_ref, 3)}, chi1={r.chi1}, chi2={r.chi2}, f_min={round(r.f_lower, 2)} Hz, f_ref={round(r.f_ref, 2)} Hz, D_L={r.luminosity_distance} Mpc')
        plt.ylabel('$h_{22}$')

        
        plt.grid(True)

        plt.tight_layout()
        # plt.show()

        if save_fig is True:
            dir = 'Images/Polarisations/'
            figname = f'Polarisations_M={r.total_mass}_ecc_ref={round(r.ecc_ref, 3)}_mean_ano_ref={round(r.mean_ano_ref, 2)}_q={r.mass_ratio}.png'
            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Images/Polarisations', exist_ok=True)
            fig.savefig(figname, dpi=300, bbox_inches='tight')

            print(self.colored_text(f'Figure is saved in {dir + figname}', 'blue'))

        # plt.close('all')

    
    def _compute_idx_ISCO(self, hp, hc, time_array, plot_ISCO_cut_off=False, save_fig_ISCO_cut_off=False):
        """
        Compute the index at which the waveform reaches the ISCO frequency, which is approximately 0.021 in dimensionless units.
        This is done by finding the point where the instantaneous phase frequency Mf crosses the ISCO frequency.

        Parameters:
        ----------------
        hp [dimensionless], np.array: Plus polarization of the waveform
        hc [dimensionless], np.array: Cross polarization of the waveform
        time_array [dimensionless], np.array: Time array corresponding to the waveform
        plot_ISCO_cut_off, True OR False, bool: Set to True to include a plot of the ISCO cut-off
        save_fig_ISCO_cut_off, True OR False, bool: Saves the figure to the directory Images/ISCO
        
        Returns:
        ----------------
        idx_cut [int]: Index at which the waveform reaches the ISCO frequency
        """
        # Compute instantaneous phase frequency Mf = dϕ/dt / 2π
        phase = np.unwrap(np.arctan2(hc, hp))
        phase -= phase[0] # Normalize phase to start at zero, correcting for the initial phase offset.

        # Calculate phase from plus and cross polarizations
        dphi_dt = np.gradient(phase, time_array)
        Mf = dphi_dt / (2 * np.pi)

        # Calculate ISCO frequency (dimensionless): Mf_ISCO = 1 / (6^(3/2) * π) ≈ 0.021
        f_isco = 1 / (6**1.5 * np.pi)  # dimensionless Mf_ISCO

        above_isco = np.where(Mf >= f_isco)[0]

        if len(above_isco) == 0:
            idx_cut = len(time_array) # in case there is no ISCO wihtin the specified time-array range
        else:
            idx_cut = above_isco[0]

        if idx_cut == len(time_array):
            plot_ISCO_cut_off = False

        if plot_ISCO_cut_off is True:
            ISCO_vs_Mf_after = plt.figure(figsize=(12,5))

            plt.plot(time_array, Mf, label=f'Mf before ISCO cut')
            plt.axhline(f_isco, color='red', linestyle='--', label='Mf ISCO $e$=0', linewidth=0.6)
            plt.scatter(time_array[idx_cut], Mf[idx_cut], color='r', s=6, label='ISCO cut')
            plt.plot(time_array[:idx_cut], Mf[:idx_cut], label=f'Mf after ISCO cut')
            plt.plot(time_array[:idx_cut], np.linspace(min(Mf), max(Mf), num=len(time_array[:idx_cut])), linestyle='--', label=f't_ISCO = {time_array[idx_cut]}', color='black', linewidth=0.6)
            plt.legend()
            plt.show()

            if save_fig_ISCO_cut_off is True:
                os.makedirs('Images/ISCO', exist_ok=True)  # Ensure the directory exists

                figname = 'Images/ISCO/truncate_at_ISCO_vs_Mf_M={}_f_ref={}_f_lower={}.svg'.format(self.total_mass, round(self.f_ref, 2), self.f_lower)
                ISCO_vs_Mf_after.savefig(figname, dpi=300, bbox_inches='tight')
                # plt.close('all') 
                print(self.colored_text(f'Figure is saved in {figname}', 'blue'))
            
        # Clean memory
        del phase, dphi_dt, Mf, f_isco, above_isco

        return idx_cut

    



class Waveform_Properties(Simulate_Waveform):
    """
    Calculates and plots residuals (residual = eccentric - non-eccentric) of waveform properties: amplitude, phase and frequency.
    """

    def __init__(self, 
                 time_array, 
                 f_lower=10, 
                 f_ref=20, 
                 ecc_ref=None, 
                 mean_anomaly_ref=0, 
                 total_mass=None, 
                 luminosity_distance=None, 
                 mass_ratio=1, 
                 chi1=0, 
                 chi2=0, 
                 phiRef=0.,
                 inclination=0., 
                 truncate_at_ISCO=True, 
                 truncate_at_tmin=True,
                 geometric_units=True
                 ):
        """
        Parameters:
        ----------------
        time_array [s], np.array : Time array in seconds.
        ecc_ref [dimensionless], float: Eccentricity of binary at start f_lower
        mean_anomaly_ref [rad], float : Mean anomaly at reference frequency f_ref
        total_mass [M_sun], int : Total mass of the binary in solar masses. =None for geometric units
        luminosity_distance [Mpc], float : Luminosity distance of the binary in megaparsec. =None for geometric units
        mass_ratio [dimensionless], float [1, inf] : Mass ratio of the binary, q >= 1
        f_lower [Hz], float: Start frequency of the waveform
        f_ref [Hz], float: Reference frequency at which the waveform parameters (eccentricity, spin, ...) are defined.
        chi1 [dimensionless], float, ndarray : Spin of primary. If float, interpreted as z component
        chi2 [dimensionless], float, ndarray : Spin of secondary. If float, interpreted as z component
        PhiRef = [rad], float : Reference phase of the waveform. 
        inclination [rad], float : Inclination angle of the binary system. Defaults to 0 (face-on).
        truncate_at_ISCO, True OR False, bool : If the waveform should be truncated at the ISCO frequency, set to True. 
        truncate_at_tmin, True OR False, bool : If the waveform should be truncated at the physical start of the time-domain (tmin), set to True.
        geometric_units, True OR False, bool : If the waveform should be generated in geometric units (c=G=M=1), set to True. For physical units, set to False and provide total_mass and luminosity_distance.
        """

        # Inherit parameters from Simulate_Inspiral class
        super().__init__(time_array=time_array, 
                         ecc_ref=ecc_ref, 
                         total_mass=total_mass, 
                         mass_ratio=mass_ratio,
                         luminosity_distance=luminosity_distance, 
                         f_lower=f_lower, 
                         f_ref=f_ref, 
                         mean_anomaly_ref=mean_anomaly_ref, 
                         chi1=chi1, 
                         chi2=chi2, 
                         phiRef=phiRef, 
                         inclination=inclination, 
                         truncate_at_ISCO=truncate_at_ISCO, 
                         truncate_at_tmin=truncate_at_tmin,
                         geometric_units=geometric_units
                         )

    def phase(self, hplus, hcross):
        """
        Calculate the phase from the plus and cross polarizations. Unitless.
        """
        phase = np.unwrap(np.arctan2(hcross, hplus))
        phase -= phase[0] # Normalize phase to start at zero, correcting for the initial phase offset.
        
        self.phase_ecc = phase
        return phase
    

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
            self.simulate_waveform(truncate_at_tmin=True, truncate_at_ISCO=True, update_results=True)

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
                figname = 'Images/Orbital_Parameters/Orbital_Parameters_vs_time_e_ref={}_l_ref={}f_lower={}_f_ref={}.png'.format(round(self.ecc_ref, 2), round(self.mean_ano_ref,2), self.f_lower, self.f_ref)
                fig_orbital_parameters.savefig(figname, dpi=300, bbox_inches='tight')
                
                print(self.colored_text(f'Figure is saved in {figname}', 'blue'))

        return gwecc_object
    

    def circulair_wf(self, time_array=None):
        """
        Simulate plus and cross polarisations of NON-ECCENTRIC waveform Inspiral for t in units [M]. 
        Also saves the phase and amplitude accordingly.
       
        Returns:
        ----------------
        hp_circ [dimensionless], np.array: Time-domain plus polarisation of NON-ECCENTRIC waveform
        hc_circ [dimensionless], np.array: Time-domain cross polarisation of NON-ECCENTRIC waveform

        """
        time_array = self.resolve_property(time_array, self.time)

        if (self.phase_circ is None) or (self.amp_circ is None):
            self.hp_circ, self.hc_circ, _ = self.simulate_waveform(ecc_ref=0, time_array=time_array)
            
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
    
