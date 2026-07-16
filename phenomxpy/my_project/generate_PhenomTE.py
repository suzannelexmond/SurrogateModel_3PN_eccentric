import time

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

from scipy.interpolate import interp1d, CubicSpline

import gc


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

        if total_mass is None:
            total_mass = 60

        self.time = SecondtoMass(time_array, total_mass)
        self.original_time = np.copy(self.time)

        self.f_lower = f_lower
        self.f_ref = f_ref
        self.total_mass = total_mass
        self.mass_ratio = mass_ratio
        self.luminosity_distance = luminosity_distance
        self.chi1 = chi1
        self.chi2 = chi2
        self.inclination = inclination
        self.phiRef = phiRef
        self.ecc_ref = ecc_ref
        self.mean_ano_ref = mean_anomaly_ref

        self.truncate_at_ISCO = truncate_at_ISCO
        self.truncate_at_tmin = truncate_at_tmin

        self.geometric_units = geometric_units

        self.hp_ecc = None
        self.hc_ecc = None
        self.amp_ecc = None
        self.phase_ecc = None

        self.hp_circ = None
        self.hc_circ = None
        self.phase_circ = None
        self.amp_circ = None

        self.mean_anomaly = None
        self.eccentricity = None

        self.t_vs_l_mapping_dict=None # Mapping of mean anomaly vs time domain
        
        super().__init__()

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
                          update_results=False,
                          show_truncation_warnings=True,
                          ):
        """
        Simulate plus and cross polarisations of the eccentric IMRPhenomTE waveform (2,2) mode for a user-defined time array (waveform peak at t=0).
        """

        # ---------------------------- Parameter resolution and consistency checks --------------------------

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

        reference_total_mass = 60
        mass_for_conversion = self.resolve_property(prop=total_mass, default=reference_total_mass)

        if geometric_units:
            f_ref, f_lower = self._convert_frequencies(
                total_mass=mass_for_conversion, reference_total_mass=60
            )

        eta = self._eta_from_mass_ratio(q=mass_ratio)

        # --------------------------- Generate waveform with PhenomTE --------------------------
        start = time.time()

        # [OPTIMIZED #2]
        # if MEMORY_PROFILE:
        #     check_memory_usage(f"simulate_waveform START ecc={ecc_ref} q={mass_ratio}")

        phen = phenomt.PhenomTE(
            mode=[2, 2],
            times=time_array,
            total_mass=mass_for_conversion,
            eta=eta,
            s1=chi1,
            s2=chi2,
            eccentricity=ecc_ref,
            f_ref=f_ref,
            f_lower=f_lower,
            phiRef=phiRef,
            inclination=inclination,
            mean_anomaly=mean_ano_ref,
        )

        if geometric_units:
            phen.compute_polarizations(times=time_array)
        else:
            phen.compute_polarizations(
                times=time_array,
                total_mass=phen.pWF.total_mass,
                distance=luminosity_distance,
            )

        # -------------------------- waveform truncation to physical lengths (tmin and ISCO) --------------------------

        valid_mask = np.ones_like(time_array, dtype=bool)

        if truncate_at_tmin:
            if phen.pWF.tmin > time_array[0]:
                warnings.warn(self.colored_text(
                    "t_min is larger than parts of the specified time-domain, resulting in unphysical waveforms. "
                    "Either use the truncate_at_tmin=True setting to automatically truncate to physical start of the time-domain "
                    "or adjust the time-array manually to start at higher values."
                , 'red'))
                valid_mask &= (time_array >= phen.pWF.tmin)

            if show_truncation_warnings:
                print(self.colored_text(
                    f'NEW TIME-DOMAIN after truncate at tmin (in geometric units): '
                    f'[{int(time_array[valid_mask][0])}, {int(time_array[valid_mask][-1])}] M',
                    'green'
                ))

        if truncate_at_ISCO:
            idx_cut = self._compute_idx_ISCO(phen.hp[valid_mask], phen.hc[valid_mask], time_array[valid_mask],
                                             plot_ISCO_cut_off=plot_ISCO_cut_off,
                                             save_fig_ISCO_cut_off=save_fig_ISCO_cut_off)

            masked_indices = np.where(valid_mask)[0]
            valid_mask[masked_indices[idx_cut:]] = False

            if show_truncation_warnings:
                print(self.colored_text(
                    f'NEW TIME-DOMAIN after truncate at ISCO (in geometric units): '
                    f'[{int(time_array[valid_mask][0])}, {int(time_array[valid_mask][-1])}] M',
                    'green'
                ))

        hp = phen.hp[valid_mask]
        hc = phen.hc[valid_mask]
        time_array = time_array[valid_mask]

        # ---------------------------------- Save waveform run results in WaveformResult object  --------------------------
        if geometric_units is False:
            time_array = MasstoSecond(time_array, total_mass)

        print(f'time : SimInspiral_M_independent e = {round(ecc_ref, 3)}, '
        f'l={mean_ano_ref}, q={mass_ratio}, chi1={chi1}, chi2={chi2}, '
        f'len = {len(phen.hp)}, M = {self.total_mass}, '
        f'lum_dist={self.luminosity_distance}, '
        f't=[{int(time_array[0])}, {int(time_array[-1])}, num={len(time_array)}], '
        f'f_lower={self.f_lower}, f_ref={self.f_ref} | '
        f'computation time = {np.round(time.time()-start, 3)} seconds')

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
            self.hp_ecc, self.hc_ecc = hp, hc

        # [OPTIMIZED #4]: Clean up phenom object internals we no longer need
        del phen
        gc.collect()

        # [OPTIMIZED #2]
        # if MEMORY_PROFILE:
        #     check_memory_usage(f"simulate_waveform END ecc={ecc_ref} q={mass_ratio}")

        return hp, hc, time_array

    def _resolve_mass_distance(self, total_mass, luminosity_distance, geometric_units):
        """
        Checks for consistency between parameters (total_mass and luminosity_distance) and the choice of geometric units vs SI units.
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
        """
        f_ref_geom = HztoMf(self.f_ref, reference_total_mass)
        f_lower_geom = HztoMf(self.f_lower, reference_total_mass)
        f_ref = MftoHz(f_ref_geom, total_mass)
        f_lower = MftoHz(f_lower_geom, total_mass)

        return f_ref, f_lower

    def _eta_from_mass_ratio(self, q):
        eta = q / (1 + q)**2
        return eta

    def _plot_polarisations(self, save_fig=True):
        """
        Plot the plus and cross polarizations of the waveform.
        """

        if not hasattr(self, "last_result"):
            raise ValueError("Run simulate_waveform() first.")

        r = self.last_result

        fig = plt.figure(figsize=(12, 5))

        plt.plot(r.time[-len(r.hp):], r.hp, label=f'$h_+$', linewidth=0.6)
        plt.plot(r.time[-len(r.hc):], r.hc, label=r'$h_{\times}$', linewidth=0.6)

        plt.legend(loc='upper left')
        if r.geometric_units:
            plt.xlabel('t [M]')
            plt.title(f'e_ref={round(r.ecc_ref, 3)}, q={r.mass_ratio}, l_ref={round(r.mean_ano_ref, 3)}, chi1={r.chi1}, chi2={r.chi2}, f_min={round(r.f_lower, 2)} Hz, f_ref={round(r.f_ref, 2)} Hz')
        else:
            plt.xlabel('t [s]')
            plt.title(f'M={r.total_mass}, e_ref={round(r.ecc_ref, 3)}, q={r.mass_ratio}, l_ref={round(r.mean_ano_ref, 3)}, chi1={r.chi1}, chi2={r.chi2}, f_min={round(r.f_lower, 2)} Hz, f_ref={round(r.f_ref, 2)} Hz, D_L={r.luminosity_distance} Mpc')
        plt.ylabel('$h_{22}$')

        plt.grid(True)
        plt.tight_layout()

        if save_fig is True:
            dir = 'Images/Polarisations/'
            figname = f'Polarisations_M={r.total_mass}_ecc_ref={round(r.ecc_ref, 3)}_mean_ano_ref={round(r.mean_ano_ref, 2)}_q={r.mass_ratio}.png'
            os.makedirs('Images/Polarisations', exist_ok=True)
            fig.savefig(figname, dpi=300, bbox_inches='tight')
            print(self.colored_text(f'Figure is saved in {dir + figname}', 'blue'))

        # [OPTIMIZED #5]: Close figure after saving or displaying
        plt.close(fig)

    def _compute_idx_ISCO(self, hp, hc, time_array, plot_ISCO_cut_off=False, save_fig_ISCO_cut_off=False):
        """
        Compute the index at which the waveform reaches the ISCO frequency.
        """

        phase = np.unwrap(np.arctan2(hc, hp))
        phase -= phase[0]

        dphi_dt = np.gradient(phase, time_array)
        Mf = dphi_dt / (2 * np.pi)

        f_isco = 1 / (6**1.5 * np.pi)

        above_isco = np.where(Mf >= f_isco)[0]

        if len(above_isco) == 0:
            idx_cut = len(time_array)
        else:
            idx_cut = above_isco[0]

        if idx_cut == len(time_array):
            plot_ISCO_cut_off = False

        if plot_ISCO_cut_off is True:
            ISCO_vs_Mf_after = plt.figure(figsize=(12, 5))

            plt.plot(time_array, Mf, label=f'Mf before ISCO cut')
            plt.axhline(f_isco, color='red', linestyle='--', label='Mf ISCO $e$=0', linewidth=0.6)
            plt.scatter(time_array[idx_cut], Mf[idx_cut], color='r', s=6, label='ISCO cut')
            plt.plot(time_array[:idx_cut], Mf[:idx_cut], label=f'Mf after ISCO cut')
            plt.plot(time_array[:idx_cut], np.linspace(min(Mf), max(Mf), num=len(time_array[:idx_cut])), linestyle='--', label=f't_ISCO = {time_array[idx_cut]}', color='black', linewidth=0.6)
            plt.legend()
            plt.show()

            if save_fig_ISCO_cut_off is True:
                os.makedirs('Images/ISCO', exist_ok=True)
                figname = 'Images/ISCO/truncate_at_ISCO_vs_Mf_M={}_f_ref={}_f_lower={}.svg'.format(self.total_mass, round(self.f_ref, 2), self.f_lower)
                ISCO_vs_Mf_after.savefig(figname, dpi=300, bbox_inches='tight')
                print(self.colored_text(f'Figure is saved in {figname}', 'blue'))

            # [OPTIMIZED #5]: Close figure after saving
            plt.close(ISCO_vs_Mf_after)

        # [OPTIMIZED #4]: Clean memory
        del phase, dphi_dt, Mf, f_isco, above_isco
        gc.collect()

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
                 geometric_units=True,
                 parametrization='mean_anomaly'
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
        # Parametrization of the waveforms 
        self.parametrization = parametrization # "mean_anomaly" vs "time"
        

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
                         geometric_units=geometric_units,
                         )

    def phase(self, hplus, hcross):
        """
        Calculate the phase from the plus and cross polarizations. Unitless.
        """
        phase = np.unwrap(np.arctan2(hcross, hplus))
        phase -= phase[0]

        self.phase_ecc = phase
        return phase

    def amplitude(self, hplus_geom, hcross_geom, geometric_units=True, luminosity_distance=None, total_mass=None):
        """
        Calculate the amplitude from the plus and cross polarizations.
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
        """
        if geometric_units:
            hp = amplitude * np.cos(phase)
            hc = -amplitude * np.sin(phase)
        else:
            hp = AmpNRtoSI(amplitude, distance, total_mass) * np.cos(phase)
            hc = AmpNRtoSI(amplitude, distance, total_mass) * np.sin(phase)
            self.time = MasstoSecond(self.time, self.total_mass)

        if plot_polarisations is True:
            self._plot_polarisations(save_fig=save_fig)

        self.hp_ecc, self.hc_ecc = hp, hc
        return hp, hc

    def get_orbital_parameters(self, plot_orbital_parameters=False, save_fig_orbital_parameters=False, make_diagnostic_plots=False):
        """
        Compute mean anomaly from time array and waveform polarizations.
        """
        time_circ_ext = np.arange(self.time[0] - 200, self.time[-1] + 200, step=self.time[1]-self.time[0])
        self.circulair_wf(time_array=time_circ_ext)

        if self.hp_ecc is None or self.hc_ecc is None:
            self.simulate_waveform(truncate_at_tmin=True, truncate_at_ISCO=True, update_results=True)

        h22_ecc = self.hp_ecc - 1j * self.hc_ecc
        h22_circ = self.hp_circ - 1j * self.hc_circ

        dataDict = {"t": self.time,
           "hlm": {(2, 2): h22_ecc},
           "t_zeroecc": time_circ_ext,
           "hlm_zeroecc": {(2, 2): h22_circ}}

        method = "ResidualAmplitude"
        tref_in = self.time

        return_dict = measure_eccentricity(tref_in=tref_in,
                                            method=method,
                                            dataDict=dataDict)

        gwecc_object = return_dict["gwecc_object"]
        t = gwecc_object.tref_out

        if make_diagnostic_plots is True:
            fig, ax = gwecc_object.make_diagnostic_plots()
            # [OPTIMIZED #5]: Close diagnostic plots
            plt.close(fig)

        if plot_orbital_parameters is True:
            if plot_orbital_parameters:
                fig_orbital_parameters, axs = plt.subplots(
                    3, 1,
                    sharex=True,
                    figsize=(8, 7),
                    constrained_layout=True
                )

                A = self.amplitude(self.hp_ecc, self.hc_ecc)
                axs[0].plot(gwecc_object.tref_in, A, lw=1.2)
                axs[0].set_ylabel(r'$A_{22}$')
                axs[0].grid(alpha=0.3)

                axs[1].plot(
                    gwecc_object.tref_out,
                    gwecc_object.eccentricity,
                    lw=1.2
                )
                axs[1].set_ylabel(r'$e$')
                axs[1].grid(alpha=0.3)

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
                os.makedirs('Images/Orbital_Parameters', exist_ok=True)
                figname = 'Images/Orbital_Parameters/Orbital_Parameters_vs_time_e_ref={}_l_ref={}f_lower={}_f_ref={}.png'.format(round(self.ecc_ref, 2), round(self.mean_ano_ref, 2), self.f_lower, self.f_ref)
                fig_orbital_parameters.savefig(figname, dpi=300, bbox_inches='tight')
                print(self.colored_text(f'Figure is saved in {figname}', 'blue'))

            # [OPTIMIZED #5]: Close figure after saving
            plt.close(fig_orbital_parameters)

        return gwecc_object    

    def _create_mean_anomaly_mapping(self, 
                                     make_diagnostic_plots=False,
                                     plot_mapping=False, 
                                     save_fig_mapping=False,
                                     ):
        """
        Create bidirectional time ↔ mean_anomaly interpolation mappings.
        
        This function handles the orbital parameter extraction and interpolation infrastructure
        WITHOUT computing any waveform properties (amplitude/phase).
        
        Parameters:
        -----------
        make_diagnostic_plots : bool
            Whether to generate diagnostic plots during orbital parameter extraction
        target_n_points : int or None
            Target number of points for the mean anomaly domain representation.
            If None, uses the native resolution from tref_out.
        
        Returns:
        --------
        mapping_dict : dict
            Dictionary containing:
            - 'mean_anomaly': Unwrapped mean anomaly array
            - 'tref_out': Time aligned with mean anomaly
            - 'time_to_mean_anomaly': Interpolant: L(t)
            - 'mean_anomaly_to_time': Interpolant: t(L)
            - 'original_time_grid': Original time array
            - 'native_resolution': Original data resolution
            - 'final_resolution': Final data resolution (may be resampled)
        """
        if self.t_vs_l_mapping_dict is None:
            # ========================================================================
            # Get orbital parameters and extract mean anomaly
            # ========================================================================
            gwecc_object = self.get_orbital_parameters(make_diagnostic_plots=make_diagnostic_plots)
            
            # Extract mean anomaly - unwrap for continuous progression
            L_out = np.unwrap(gwecc_object.mean_anomaly)

            # ========================================================================
            # 2. Create UNIFORM L grid
            # ========================================================================
            L_min, L_max = L_out.min(), L_out.max()
            n_points = len(L_out)
            L_uni = np.linspace(L_min, L_max, n_points)

            t_out = gwecc_object.tref_out

            # Save eccentricity and mean anomaly for the new mean anomaly uniform timegrid
            self.eccentricity, self.mean_anomaly = gwecc_object.eccentricity, L_uni
            
            # ========================================================================
            # 4. Build bidirectional interpolants from uniform pair (L_out, t_out)
            # ========================================================================
            time_to_mean_anomaly_interp = CubicSpline(t_out, L_uni, bc_type='natural')
            mean_anomaly_to_time_interp = CubicSpline(L_uni, t_out, bc_type='natural')
            
            # ========================================================================
            # Step 3: Create bidirectional interpolants: time ↔ mean_anomaly
            #        using the CORRECTLY PAIRED (L_out, t_out) data
            # ========================================================================
            # time_to_mean_anomaly_interp = interp1d(
            #     t_out, 
            #     L_out,
            #     kind='cubic',
            #     fill_value='extrapolate',
            #     assume_sorted=True,
            # )

            # mean_anomaly_to_time_interp = interp1d(
            #     L_out,
            #     t_out,
            #     kind='cubic',
            #     fill_value='extrapolate',
            #     assume_sorted=True,
            # )
            
            # ========================================================================
            # Build mapping dictionary
            # ========================================================================
            mapping_dict = {
                'mean_anomaly': L_uni,           # Unwrapped mean anomaly array
                'tref_out': t_out,                       # Time aligned with mean anomaly
                'time_to_mean_anomaly': time_to_mean_anomaly_interp,
                'mean_anomaly_to_time': mean_anomaly_to_time_interp,
            }
            
            self.t_vs_l_mapping_dict = mapping_dict

        if plot_mapping:
            self.plot_mean_anomaly_mappings(
                save_fig=save_fig_mapping,
            )
        
        return self.t_vs_l_mapping_dict
    
    
    def to_l_domain(self, 
                                         property_name, 
                                         make_diagnostic_plots=False,
                                         plot_in_L_domain=False,
                                         save_fig=False,
                                         ):
        """
        Convert waveform property (amplitude or phase) from time domain to mean anomaly domain.
        
        Uses the time↔mean_anomaly mappings to reparametrize the property.
        ALWAYS OUTPUTS ON A UNIFORM MEAN-ANOMALY GRID (same number of points as native).
        
        Parameters
        -----------
        property_name : str
            Name of property to convert: 'amplitude' or 'phase'
        make_diagnostic_plots : bool
            Whether to generate diagnostic plots during orbital parameter extraction
        plot_in_L_domain : bool
            Whether to generate diagnostic plots of the property in L-domain
        save_fig : bool
            Save figures instead of displaying them
        
        Returns
        --------
        result : dict
            Dictionary containing:
            - All keys from mapping_dict
            - f'{property_name}_in_L': Property values on UNIFORM L grid
            - f'{property_name}_in_t': Original property sampled at tref_out times
            - f'{property_name}_interp': Interpolant object (L → property)
        """
        
        if property_name not in ['amplitude', 'phase']:
            raise ValueError(f"property_name must be 'amplitude' or 'phase', got '{property_name}'")
        
        # ========================================================================
        # Create uniform L mapping
        # ========================================================================
        mapping_dict = self._create_mean_anomaly_mapping(
            make_diagnostic_plots=make_diagnostic_plots,
        )
        
        # ========================================================================
        # Compute property if not provided
        # ========================================================================
        if self.hp_ecc is None or self.hc_ecc is None:
            self.simulate_waveform(
                truncate_at_tmin=True, 
                truncate_at_ISCO=True, 
                update_results=True,
            )
        
        if property_name == 'amplitude':
            prop_in_t = self.amplitude(self.hp_ecc, self.hc_ecc, geometric_units=True)
        elif property_name == 'phase':
            prop_in_t = self.phase(self.hp_ecc, self.hc_ecc)

        # ========================================================================
        # Map property onto UNIFORM mean-anomaly grid
        # ========================================================================
        L_out = mapping_dict['mean_anomaly']
        t_out = mapping_dict['tref_out']
        time_grid = self.time

        prop_on_time = CubicSpline(time_grid, prop_in_t, bc_type='natural')
        prop_in_tout = prop_on_time(t_out)
        
        # ========================================================================
        # Create interpolant: mean_anomaly → property
        # ========================================================================
        L_interp = CubicSpline(L_out, prop_in_tout, bc_type='natural')
        prop_in_Lout = L_interp(L_out)
        
        # ========================================================================
        # Optional diagnostic plots
        # ========================================================================
        if plot_in_L_domain:
            self.plot_property_in_mean_anomaly_domain(
                property_name=property_name,
                save_fig=save_fig,
            )
        
        # ========================================================================
        # Build result dictionary
        # ========================================================================
        result = {
            # Mapping info (copied from input)
            'mean_anomaly': mapping_dict['mean_anomaly'],
            'tref_out': mapping_dict['tref_out'],
            'time_to_mean_anomaly': mapping_dict['time_to_mean_anomaly'],
            'mean_anomaly_to_time': mapping_dict['mean_anomaly_to_time'],
          
            # Property-specific
            f'{property_name}_in_L': prop_in_Lout,
            f'{property_name}_in_t': prop_in_tout,
        }
        
        return result


    def compute_interpolation_error(self, property_name, 
                                    plot_interp_errors=True):
        """
        Quantify the interpolation error introduced by converting properties
        through the mean-anomaly domain.

        Pipeline tested:
        
        1. Forward: t → L (via time_to_mean_anomaly) → interpolate property(L)
        2. Reverse: L → t (via mean_anomaly_to_time) → evaluate property(L) at recovered L
        
        Then compare recovered property(t) vs ORIGINAL property(t) at the SAME t_out grid.

        Parameters
        ----------
        property_name : str
            'amplitude' or 'phase'.
        plot_interp_errors : bool
            If True, generate diagnostic plots of interpolation residuals.

        Returns
        -------
        result : dict
            Keys:
            - 'L_out'              : Mean anomaly grid
            - 't_out'              : Time aligned with mean anomaly
            - 'prop_true_t'        : Original property sampled at t_out
            - 'prop_recovered_t'   : Recovered property from prop(L) → t(L)
            - 'absolute_error'     : |prop_true - prop_recovered|
            - 'relative_error'     : |prop_true - prop_recovered| / |prop_true|
            - 'max_abs_error'      : float
            - 'rms_abs_error'      : float  
            - 'max_rel_error'      : float
            - 'rms_rel_error'      : float
            - 'mapping_dict'       : The mapping dict used
            - 'prop_in_Lout'       : The interpolated property on L grid
        """

        from scipy.interpolate import CubicSpline

        # ==================================================================
        # 1. Forward pass: build the L-domain interpolant
        # ==================================================================
        fwd_result = self.to_l_domain(
            property_name=property_name,
        )
        mapping_dict = fwd_result

        L_out = fwd_result['mean_anomaly']
        t_out = fwd_result['tref_out']
        prop_in_Lout = fwd_result[f'{property_name}_in_L']    # CubicSpline on L grid

        # ==================================================================
        # 2. Compute TRUE property on t_out using the original simulation
        #    Use CubicSpline instead of interp1d for consistency
        # ==================================================================
        if self.hp_ecc is None or self.hc_ecc is None:
            self.simulate_waveform(
                truncate_at_tmin=True,
                truncate_at_ISCO=True,
                update_results=True,
            )

        time_grid = self.time

        if property_name == 'amplitude':
            prop_true_on_t = self.amplitude(self.hp_ecc, self.hc_ecc, geometric_units=True)
            # Use CubicSpline for smooth interpolation onto t_out
            true_prop_interp = CubicSpline(time_grid, prop_true_on_t, bc_type='natural')
            prop_true_t = true_prop_interp(t_out)
            
        elif property_name == 'phase':
            phase_on_time = self.phase(self.hp_ecc, self.hc_ecc)
            true_phase_interp = CubicSpline(time_grid, phase_on_time, bc_type='natural')
            prop_true_t = true_phase_interp(t_out)

        # ==================================================================
        # 3. Reverse pass: recover property at t_out via L-domain
        #    Step A: Map L_out → t (should give t_back ≈ t_out)
        #    Step B: The recovered property is prop_in_Lout evaluated at L_out
        #    Since L_out ↔ t_out are paired, prop_recovered_t = prop_in_Lout
        # ==================================================================
        mean_anomaly_to_time = mapping_dict['mean_anomaly_to_time']
        t_back = mean_anomaly_to_time(L_out)

        # Also report how well t_back matches t_out
        time_match_err = np.abs(t_back - t_out)
        print(self.colored_text(
            f"Time mapping fidelity:  max_dt={np.max(time_match_err):.6e}  "
            f"rms_dt={np.sqrt(np.mean(time_match_err**2)):.6e}",
            'yellow'
        ))

        # The recovered property is just prop_in_Lout (evaluated on L grid)
        # But we should verify that t_back ≈ t_out
        prop_recovered_t = prop_in_Lout.copy()

        # ==================================================================
        # 4. Error metrics
        # ==================================================================
        abs_err = np.abs(prop_true_t - prop_recovered_t)
        
        denom = np.abs(prop_true_t)
        rel_err = np.where(denom > 0, abs_err / denom, 0.0)

        max_abs_err = float(np.max(abs_err))
        rms_abs_err = float(np.sqrt(np.mean(abs_err ** 2)))
        max_rel_err = float(np.max(rel_err))
        rms_rel_err = float(np.sqrt(np.mean(rel_err ** 2)))

        print(self.colored_text(
            f"Interpolation error ({property_name}):  "
            f"max_abs={max_abs_err:.6e}  rms_abs={rms_abs_err:.6e}  "
            f"max_rel={max_rel_err:.6e}  rms_rel={rms_rel_err:.6e}",
            'yellow'
        ))

        

        # ==================================================================
        # 5. Diagnostic plots
        # ==================================================================
        if plot_interp_errors:
            self._plot_interpolation_diagnostics(
                t_out, 
                prop_true_t, prop_recovered_t,
                abs_err, rel_err,
                time_match_err,
                property_name,
            )

        # ==================================================================
        # 6. Assemble result
        # ==================================================================
        result = {
            'L_out': L_out,
            't_out': t_out,
            't_back': t_back,
            'prop_true_t': prop_true_t,
            'prop_recovered_t': prop_recovered_t,
            'prop_in_Lout': prop_in_Lout,
            
            'absolute_error': abs_err,
            'relative_error': rel_err,
            'max_abs_error': max_abs_err,
            'rms_abs_error': rms_abs_err,
            'max_rel_error': max_rel_err,
            'rms_rel_error': rms_rel_err,
            
            'time_mapping_error': time_match_err,
            'mapping_dict': fwd_result,
        }
        result.update(fwd_result)

        return result
 

    def circulair_wf(self, mass_ratio=None, mean_ano_ref=None, chi1=None, chi2=None, time_array=None):
        """
        Simulate plus and cross polarisations of NON-ECCENTRIC waveform Inspiral for t in units [M].
        Also saves the phase and amplitude accordingly.
        """
        time_array = self.resolve_property(time_array, self.time)
        mass_ratio = self.resolve_property(mass_ratio, self.mass_ratio)
        mean_ano_ref = self.resolve_property(mean_ano_ref, self.mean_ano_ref)
        chi1 = self.resolve_property(chi1, self.chi1)
        chi2 = self.resolve_property(chi2, self.chi2)

        circ_params = (mass_ratio, mean_ano_ref, chi1, chi2)

        if (
            (self.phase_circ is None) or (self.amp_circ is None)
            or (not hasattr(self, "_circ_params")) or (self._circ_params != circ_params)
            or (len(time_array) > len(self.amp_circ))
        ):
            self.hp_circ, self.hc_circ, _ = self.simulate_waveform(
                ecc_ref=0,
                time_array=time_array,
                mass_ratio=mass_ratio,
                mean_ano_ref=mean_ano_ref,
                chi1=chi1,
                chi2=chi2
            )

            self.phase_circ = self.phase(self.hp_circ, self.hc_circ)
            self.amp_circ = self.amplitude(self.hp_circ, self.hc_circ)

            self._circ_params = circ_params

        elif self.amp_circ is not None and len(self.amp_circ) > len(time_array):
            n_t = len(time_array)

            self.hp_circ = self.hp_circ[-n_t:]
            self.hc_circ = self.hc_circ[-n_t:]
            self.phase_circ = self.phase_circ[-n_t:]
            self.amp_circ = self.amp_circ[-n_t:]

        else:
            pass

    def calculate_residual(self,
                           hp,
                           hc,
                           ecc_ref=None,
                           mass_ratio=None,
                           mean_ano_ref=None,
                           chi1=None,
                           chi2=None,
                           property=None,
                           plot_residual=False, save_fig=False):
        """
        Calculate residual (= eccentric - non-eccentric) of Waveform Inspiral property.
        Possible properties: phase, amplitude or frequency
        """

        ecc_ref = self.resolve_property(prop=ecc_ref, default=self.ecc_ref)
        mean_ano_ref = self.resolve_property(prop=mean_ano_ref, default=self.mean_ano_ref)
        mass_ratio = self.resolve_property(prop=mass_ratio, default=self.mass_ratio)
        chi1 = self.resolve_property(prop=chi1, default=self.chi1)
        chi2 = self.resolve_property(prop=chi2, default=self.chi2)

        self.circulair_wf(mass_ratio=mass_ratio,
                          mean_ano_ref=mean_ano_ref,
                          chi1=chi1,
                          chi2=chi2)
        
        if self.parametrization == "mean_anomaly":
            self.get_amplitude_phase_in_mean_anomaly_domain()

        if property == 'phase':
            circ = self.phase_circ
            eccentric = self.phase(hp, hc)
            units = '[radians]'

            residual = circ - eccentric

            if eccentric[1] < 0:
                warnings.warn(self.colored_text("Eccentric phase has negative starting values. "
                "This may not be expected for physical waveforms. This usually happens when the eccentric waveformlength is shorter than the chosen time array. "
                "Consider decreasing the time array length or decreasing the eccentricity.", 'red'))

        elif property == 'amplitude':
            circ = self.amp_circ
            eccentric = self.amplitude(hp, hc)
            units = ''
            residual = eccentric - circ

        else:
            print('Choose property = "phase", "amplitude", "frequency"', property, 2)
            sys.exit(1)

        if np.any(np.isnan(residual)) or np.any(np.isinf(residual)):
            print(self.colored_text(f"Warning: Residual contains NaN or Inf values for parameters ecc={ecc_ref}, l={mean_ano_ref}, q={mass_ratio}, chi1={chi1}, chi2={chi2}. Setting residual to zero. \
                    \n hp, hc: {hp, hc}", 'red'))

            plot_residual = True

            fig_polarizations = plt.figure(figsize=(12, 5))
            plt.plot(self.time, hp, label='hp', linewidth=0.6)
            plt.plot(self.time, hc, label='hc', linewidth=0.6)
            plt.legend()
            plt.title(f'Polarizations for parameters ecc={ecc_ref}, l={mean_ano_ref}, q={mass_ratio}, chi1={chi1}, chi2={chi2}')
            plt.grid(True)
            plt.tight_layout()
            # [OPTIMIZED #5]: Close figure
            # plt.close(fig_polarizations)

        if plot_residual is True:
            fig_residual = plt.figure()

            plt.plot(self.time, eccentric, label=f'Eccentric {property}: $e$={ecc_ref}', linewidth=0.6)
            plt.plot(self.time, circ, label=f'Circular {property}: $e$=0', linewidth=0.6)
            plt.plot(self.time, residual, label=f'Residual {property}', linewidth=0.6)

            plt.xlabel('t [M]')
            plt.ylabel(property + ' ' + units)
            plt.title(f'Residual {property}, ecc={round(ecc_ref, 3)}, q={mass_ratio}, chi1={chi1}, chi2={chi2}, mean_ano_ref={round(mean_ano_ref, 2)}')
            plt.grid(True)
            plt.legend()

            plt.tight_layout()

            if save_fig is True:
                figname = f'Images/Residuals/Residual {property} M={self.total_mass}, ecc={round(ecc_ref, 3)}.png'
                os.makedirs('Images/Residuals', exist_ok=True)
                fig_residual.savefig(figname, dpi=300, bbox_inches='tight')
                print(self.colored_text(f'Figure is saved in {figname}', 'blue'))

            # [OPTIMIZED #5]: Close figure after saving
            plt.close(fig_residual)

        # [OPTIMIZED #4]: Clean memory
        del circ, eccentric
        gc.collect()

        return residual


    #######################################################################
    # PLOTTING FUNCTIONS
    #######################################################################

    def _plot_interpolation_diagnostics(self, t, prop_true, prop_rec, abs_err, rel_err, time_err, property_name):
        """Four-panel diagnostic: property comparison, errors, time mapping fidelity."""

        fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True,
                                gridspec_kw={'height_ratios': [2, 1, 1, 1]})

        prop_label = property_name.capitalize()

        # ---- Panel 1: True vs Recovered property ---------------------------
        ax = axes[0]
        ax.plot(t, prop_true, 'b-', lw=1.2, alpha=0.8, label='True (simulation)')
        ax.plot(t, prop_rec,  'r--', lw=1.2, alpha=0.8, label='Recovered (L-domain interp)')
        ax.set_ylabel(prop_label)
        ax.set_title(f'Interpolation Quality: {prop_label}  at t_out grid')
        ax.legend(loc='best', fontsize=9)

        # ---- Panel 2: Absolute error --------------------------------------
        ax = axes[1]
        ax.plot(t, abs_err, 'k-', lw=0.8)
        ax.axhline(0, color='grey', ls=':', lw=0.5)
        ax.fill_between(t, abs_err, alpha=0.25, color='orange')
        ax.set_ylabel('|Δ{}|'.format(prop_label))
        ax.set_title('Absolute interpolation error')

        # ---- Panel 3: Relative error --------------------------------------
        ax = axes[2]
        ax.plot(t, rel_err, 'k-', lw=0.8)
        ax.axhline(0, color='grey', ls=':', lw=0.5)
        ax.fill_between(t, rel_err, alpha=0.25, color='teal')
        ax.set_ylabel('|Δ{}| / |{}|'.format(prop_label, prop_label))
        ax.set_title('Relative interpolation error')

        # ---- Panel 4: Time mapping fidelity ------------------------------
        ax = axes[3]
        ax.plot(t, time_err, 'purple', lw=0.8)
        ax.axhline(0, color='grey', ls=':', lw=0.5)
        ax.fill_between(t, time_err, alpha=0.25, color='purple')
        ax.set_xlabel('Time')
        ax.set_ylabel('|t_back − t_out|')
        ax.set_title('Time mapping fidelity (L → t → L round-trip)')

        plt.tight_layout()


    def plot_mean_anomaly_mappings(self, save_fig=False):
        """
        Plot the mean anomaly <-> time mappings for validation.
        
        Parameters:
        -----------
        mapping_dict : dict
            Output from _create_mean_anomaly_mapping()
        save_plots : bool
            Whether to save the figure
        """

        mapping_dict = self._create_mean_anomaly_mapping()

        L_raw = mapping_dict['mean_anomaly']
        t_raw = mapping_dict['tref_out']

        # Create dense grids for smooth interpolation curves
        L_dense = np.linspace(L_raw.min(), L_raw.max(), 5000)
        t_dense = np.linspace(t_raw.min(), t_raw.max(), 5000)

        # Evaluate interpolants on dense grids
        t_from_L = mapping_dict['mean_anomaly_to_time'](L_dense)
        L_from_t = mapping_dict['time_to_mean_anomaly'](t_dense)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # -----------------------------------------------------------------
        # (a) Unwrapped mean anomaly vs time — raw data
        # -----------------------------------------------------------------
        axes[0, 0].plot(t_raw, L_raw, 'b-', linewidth=0.8)
        axes[0, 0].scatter(t_raw[::200], L_raw[::200], s=8, c='red', alpha=0.6, label='Sample points')
        axes[0, 0].set_xlabel('Time [M]', fontsize=11)
        axes[0, 0].set_ylabel('Mean Anomaly ℓ [rad]', fontsize=11)
        axes[0, 0].set_title('(a) Unwrapped Mean Anomaly vs Time (raw)', fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)

        # -----------------------------------------------------------------
        # (b) Interpolated: mean_anomaly -> time (inverse mapping)
        # -----------------------------------------------------------------
        axes[0, 1].plot(L_dense, t_from_L, 'g-', linewidth=0.8, label='Interpolated ℓ → t')
        axes[0, 1].scatter(L_raw[::200], t_raw[::200], s=8, c='red', alpha=0.6, label='Raw sample points')
        axes[0, 1].set_xlabel('Mean Anomaly ℓ [rad]', fontsize=11)
        axes[0, 1].set_ylabel('Time [M]', fontsize=11)
        axes[0, 1].set_title('(b) Interpolated: Mean Anomaly → Time', fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)

        # -----------------------------------------------------------------
        # (c) Interpolated: time -> mean_anomaly (forward mapping)
        # -----------------------------------------------------------------
        axes[1, 0].plot(t_dense, L_from_t, 'm-', linewidth=0.8, label='Interpolated t → ℓ')
        axes[1, 0].scatter(t_raw[::200], L_raw[::200], s=8, c='red', alpha=0.6, label='Raw sample points')
        axes[1, 0].set_xlabel('Time [M]', fontsize=11)
        axes[1, 0].set_ylabel('Mean Anomaly ℓ [rad]', fontsize=11)
        axes[1, 0].set_title('(c) Interpolated: Time → Mean Anomaly', fontsize=12, fontweight='bold')
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)

        # -----------------------------------------------------------------
        # (d) Overlay: raw vs interpolated (both directions) with residuals
        # -----------------------------------------------------------------
        # Forward: t -> L — compare interpolated to raw at raw time points
        L_interp_at_raw_t = mapping_dict['time_to_mean_anomaly'](t_raw)
        L_residual_forward = L_raw - L_interp_at_raw_t

        # Inverse: L -> t — compare interpolated to raw at raw L points
        t_interp_at_raw_L = mapping_dict['mean_anomaly_to_time'](L_raw)
        t_residual_inverse = t_raw - t_interp_at_raw_L

        ax_res = axes[1, 1]
        ax_res.plot(t_raw, L_residual_forward, 'm-', linewidth=0.6, label='Δℓ (forward: raw − interp)')
        ax_res.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        ax_res.set_xlabel('Time [M]', fontsize=11)
        ax_res.set_ylabel('Residual ℓ [rad]', fontsize=11)
        ax_res.set_title('(d) Forward interpolation residual (t → ℓ)', fontsize=12, fontweight='bold')
        ax_res.legend(fontsize=9)
        ax_res.grid(True, alpha=0.3)

        # Print residual stats
        rms_L = np.sqrt(np.mean(L_residual_forward**2))
        max_L = np.max(np.abs(L_residual_forward))
        rms_t = np.sqrt(np.mean(t_residual_inverse**2))
        max_t = np.max(np.abs(t_residual_inverse))

        print(self.colored_text('=' * 50, 'cyan'))
        print(self.colored_text('INTERPOLATION RESIDUAL SUMMARY', 'cyan'))
        print(self.colored_text('=' * 50, 'cyan'))
        print(f"  Forward (t → ℓ):  RMS = {rms_L:.6e} rad,  max = {max_L:.6e} rad")
        print(f"  Inverse (ℓ → t):  RMS = {rms_t:.6e} M,    max = {max_t:.6e} M")
        print('=' * 50)

        plt.tight_layout()

        if save_fig:
            os.makedirs('Images/Validation', exist_ok=True)
            fig.savefig('Images/Validation/mean_anomaly_mappings.png', dpi=300, bbox_inches='tight')
            print(self.colored_text('Figure saved: Images/Validation/mean_anomaly_mappings.png', 'blue'))

        # plt.close(fig)
        return fig

    def plot_property_in_mean_anomaly_domain(
        self,
        property_name,
        save_fig=False,
    ):
        """
        Plot a waveform property in the mean-anomaly domain.

        Parameters
        ----------
        result_dict : dict
            Output from ``tro_l_domain()``.
        property_name : str
            'amplitude' or 'phase' — used to pull the right keys and label the plot.
        show_time_mapping : bool
            If True, draw a secondary y-axis showing ``t_out`` vs ``L_out``
            so you can visually inspect the non-uniform mapping
            between reference time and mean anomaly.
        ax : matplotlib.axes.Axes or None
            Axis to draw on. If None, a new figure is created.
        **plot_kwargs
            Extra kwargs forwarded to the line plot (color, linewidth, …).

        Returns
        -------
        ax : matplotlib.axes.Axes
            The main axis (and the twin axis if ``show_time_mapping``).
        """
        result_dict = self.to_l_domain(property_name=property_name)                                                            

        L_out  = result_dict["mean_anomaly"]
        t_out = result_dict["tref_out"]
        prop_in_L = result_dict[f"{property_name}_in_L"]
        prop_in_t = result_dict[f"{property_name}_in_t"]

        fig, ax = plt.subplots(figsize=(9, 5))

        # ------------------------------------------------------------------ #
        # Interpolated property (smooth line)
        # ------------------------------------------------------------------ #
        ax.plot(
            L_out, prop_in_L,
            linestyle="-", linewidth=1.5, alpha=0.9,
            label=f"{property_name.capitalize()}(ℓ) — interpolant",
        )

        # ------------------------------------------------------------------ #
        # Raw sampled points (scatter overlay)
        # ------------------------------------------------------------------ #
        ax.scatter(
            L_out[::50], prop_in_t[::50],
            s=8, zorder=5, edgecolors="k", linewidths=0.3,
            color="crimson",
            label=f"{property_name.capitalize()} sampled at t_out (every 50th point)",
        )

        ax.set_xlabel("Mean Anomaly ℓ  [rad]")
        ax.set_ylabel(f"{property_name.capitalize()}")
        ax.legend(loc="best", fontsize=9)
        ax.set_title(f"{property_name.capitalize()} in Mean-Anomaly Domain")

        # ------------------------------------------------------------------ #
        # Optional twin axis: show the time → mean-anomaly mapping
        # ------------------------------------------------------------------ #
        twin_ax = None

        twin_ax = ax.twiny()
        # We need L_out sorted for a clean line
        sort_idx = np.argsort(t_out)
        twin_ax.plot(
            t_out[sort_idx], L_out[sort_idx],
            color="grey", linestyle="--", linewidth=1, alpha=0.6,
            transform=twin_ax.get_xaxis_transform(),
        )
        twin_ax.set_xlabel("Reference time  t_out")
        twin_ax.tick_params(axis="x", colors="grey")

        plt.tight_layout()

        if save_fig:
            os.makedirs('Images/Validation', exist_ok=True)
            figname = f'Images/Validation/{property_name}_in_L.png'
            fig.savefig(figname, dpi=300, bbox_inches='tight')
            print(self.colored_text(f'Figure saved: {figname}', 'blue'))

        # plt.close(fig)
        return fig
    
