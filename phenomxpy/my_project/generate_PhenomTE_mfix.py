import time

from helper_functions_mfix import *

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

# [OPTIMIZED #2]
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

        print(f'time : SimInspiral_M_independent e = {round(ecc_ref, 3)}, l={mean_ano_ref}, q={mass_ratio}, chi1={chi1}, chi2={chi2}, len = {len(phen.hp)}, M = {self.total_mass}, lum_dist={self.luminosity_distance}, t=[{int(time_array[0])}, {int(time_array[-1])}, num={len(time_array)}], f_lower={self.f_lower}, f_ref={self.f_ref} | computation time = {(time.time()-start)} seconds')

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
        time_circ = np.arange(self.time[0] - 200, self.time[-1] + 200, step=self.time[1]-self.time[0])
        self.circulair_wf(custom_time_array=time_circ)

        if self.hp_ecc is None or self.hc_ecc is None:
            self.simulate_waveform(truncate_at_tmin=True, truncate_at_ISCO=True, update_results=True)

        h22_ecc = self.hp_ecc - 1j * self.hc_ecc
        h22_circ = self.hp_circ - 1j * self.hc_circ

        dataDict = {"t": self.time,
           "hlm": {(2, 2): h22_ecc},
           "t_zeroecc": time_circ,
           "hlm_zeroecc": {(2, 2): h22_circ}}

        method = "ResidualAmplitude"
        tref_in = self.time

        return_dict = measure_eccentricity(tref_in=tref_in,
                                            method=method,
                                            dataDict=dataDict)

        gwecc_object = return_dict["gwecc_object"]

        if make_diagnostic_plots is True:
            fig, ax = gwecc_object.make_diagnostic_plots()
            # [OPTIMIZED #5]: Close diagnostic plots
            plt.close(fig)

        self.mean_anomaly, self.eccentricity = gwecc_object.mean_anomaly, gwecc_object.eccentricity

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

    def circulair_wf(self, mass_ratio, mean_ano_ref, chi1, chi2, time_array=None):
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
            plt.close(fig_polarizations)

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