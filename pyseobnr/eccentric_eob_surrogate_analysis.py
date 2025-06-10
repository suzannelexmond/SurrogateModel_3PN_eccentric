from generate_eob_surrogate import *

from pycbc.filter import match, optimized_match
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.types import timeseries
from pycbc.types import TimeSeries, FrequencySeries
from pycbc import waveform
import time
import traceback
from scipy.fft import rfft, rfftfreq
from pycbc.waveform.utils import taper_timeseries
import seaborn as sns
from matplotlib.lines import Line2D

class Surrogate_analysis(Generate_Surrogate):

    def __init__(self, parameter_space, amount_input_wfs, amount_output_wfs,  N_greedy_vecs_amp=None, N_greedy_vecs_phase=None, min_greedy_error_amp=None, min_greedy_error_phase=None, waveform_size=None, mass_ratio=1, freqmin=650):
        
        self.surrogate_h = None
        self.parameter_space = parameter_space
        self.amount_input_wfs = amount_input_wfs
        self.amount_output_wfs = amount_output_wfs

        Generate_Surrogate.__init__(self, parameter_space, amount_input_wfs, amount_output_wfs,  N_greedy_vecs_amp, N_greedy_vecs_phase, min_greedy_error_amp, min_greedy_error_phase, waveform_size=waveform_size, mass_ratio=mass_ratio, freqmin=freqmin)

    def mismatch_NR_TEHM(self, hpTEHMin, hcTEHMin, hpSXSin, hcSXSin, delta_t, ecc, plots=False, plots_extra=False):
        # Make copies of the input waveforms (hpTEHMin, hcTEHMin for TEHM model and hpSXSin, hcSXSin for SXS model)
        hpTEHM = np.copy(hpTEHMin)
        hcTEHM = np.copy(hcTEHMin)
        hpSXS = np.copy(hpSXSin)
        hcSXS = np.copy(hcSXSin)

        # Convert the time-domain waveforms to PyCBC TimeSeries objects for processing
        hp_td_TEHM = TimeSeries(hpTEHM, delta_t=delta_t)
        hc_td_TEHM = TimeSeries(hcTEHM, delta_t=delta_t)
        
        if plots_extra is True:
            fig1 = plt.figure()
            plt.plot(hp_td_TEHM.get_sample_times(), hp_td_TEHM, label='surr hp')
            plt.plot(hc_td_TEHM.get_sample_times(), hc_td_TEHM, label='surr hc')
            plt.title(ecc)
            plt.legend()

        # Apply a taper to smooth the start and end of the waveforms to avoid edge effects
        hp_td_TEHM = taper_timeseries(hp_td_TEHM, tapermethod="startend")
        hc_td_TEHM = taper_timeseries(hc_td_TEHM, tapermethod="startend")

        if plots_extra is True:
            fig2 = plt.figure()
            plt.plot(hp_td_TEHM.get_sample_times(), hp_td_TEHM, label='surr hp tapered')
            plt.plot(hc_td_TEHM.get_sample_times(), hc_td_TEHM, label='surr hc tapered')
            plt.legend()

        # If the SXS waveforms are provided as numpy arrays, convert them to TimeSeries and taper
        if type(hpSXS) is np.ndarray:
            hp = TimeSeries(hpSXS, delta_t=delta_t)
            hc = TimeSeries(hcSXS, delta_t=delta_t)
            hp = taper_timeseries(hp, tapermethod="startend")
            hc = taper_timeseries(hc, tapermethod="startend")
        else:
            hp = hpSXS
            hc = hcSXS

        # if plots_extra is True:
        fig3 = plt.figure()
        plt.plot(hp_td_TEHM.get_sample_times(), hp_td_TEHM, label='surr hp')
        # plt.plot(hc_td_TEHM.get_sample_times(), hc_td_TEHM, label='surr hc')
        plt.plot(hp.get_sample_times(), hp, label='real hp')
        # plt.plot(hc.get_sample_times(), hc, label='real hc')
        plt.legend()

        # Pad the time-domain waveforms to the next power of 2 for efficient FFT operations
        N = max(len(hp), len(hp_td_TEHM))
        pad = int(2 ** (np.floor(np.log2(N)) + 2))
        hp_td_TEHM.resize(pad)
        hc_td_TEHM.resize(pad)
        hp.resize(pad)
        hc.resize(pad)

        if plots_extra is True:
            fig4 = plt.figure()
            plt.plot(hp_td_TEHM.get_sample_times(), hp_td_TEHM, label='pad surr hp')
            # plt.plot(hc_td_TEHM.get_sample_times(), hc_td_TEHM, label='pad  hc')
            plt.plot(hp.get_sample_times(), hp, label='pad real hp')
            # plt.plot(hc.get_sample_times(), hc, label='pad real hc')
            plt.legend()

        # Calculate the combined complex strain for SXS, then find the peak for time alignment
        s_in = hp.data - 1j * hc.data
        s_inabs = np.abs(s_in)
        peak_pos_s = s_inabs.argmax()  # Get the peak position index
        times_s_full = hp.get_sample_times() - hp.get_sample_times()[peak_pos_s]  # Align peak to time zero

        # Convert the waveforms to the frequency domain
        sp_td = hp.to_frequencyseries()
        sc_td = hc.to_frequencyseries()
        sp_td_TEHM = hp_td_TEHM.to_frequencyseries()
        sc_td_TEHM = hc_td_TEHM.to_frequencyseries()

        if plots_extra is True:
            fig6 = plt.figure()
            plt.plot(sp_td_TEHM.get_sample_frequencies(), sp_td_TEHM, lw=0.6, label='freq surr hp')
            # plt.plot(sc_td_TEHM.get_sample_frequencies(), lw=0.6, sc_td_TEHM, label='freq surr hc')
            plt.plot(sp_td.get_sample_frequencies(), sp_td, lw=0.6, label='freq real hp')
            # plt.plot(sc_td.get_sample_frequencies(), sc_td, lw=0.6, label='freq real hc')
            plt.legend()
            plt.xlim(0, 100)

        # Identify the peak frequencies for SXS and TEHM for frequency cutoff determination
        amp_s = np.abs(sp_td.data - 1j * sc_td.data)
        idx_max = np.argmax(amp_s)
        f_peak = sp_td.get_sample_frequencies()[idx_max]  # Peak frequency for SXS model

        amp_s_TEHM = np.abs(sp_td_TEHM.data - 1j * sc_td_TEHM.data)
        idx_max_TEHM = np.argmax(amp_s_TEHM)
        f_peak_TEHM = sp_td_TEHM.get_sample_frequencies()[idx_max_TEHM]  # Peak frequency for TEHM model

        # Set the minimum frequency cutoff as 1.35 times the highest peak frequency
        fpeak = max([f_peak, f_peak_TEHM])
        # f_minc = 1.35 * fpeak
        f_minc=18

        # Set the physical frequency range for the PSD, with a lower bound of 10 Hz if f_minc is too low
        f_low_phys = max(18., f_minc)
        f_high_phys = 2048.  # Upper frequency cutoff

        # Generate the PSD using the aLIGO Zero Detuned High Power configuration
        psd = aLIGOZeroDetHighPower(len(sp_td), sp_td.delta_f, f_low_phys)

        if plots_extra is True:
            fig7 = plt.figure()
            plt.plot(sp_td_TEHM.get_sample_frequencies(), sp_td_TEHM, lw=0.6, label='freq surr hp')
            plt.scatter(f_low_phys, 0, label='low freq cutoff')
            plt.scatter(f_high_phys, 0, label='high freq cutoff')
            plt.plot(sp_td.get_sample_frequencies(), sp_td, lw=0.6, label='freq real hp')
            plt.plot(psd.get_sample_frequencies(), psd, lw=0.6, label='psd')
            plt.legend()
            plt.xlim(0, 100)
            # plt.show()

        # # Calculate the match for the h_plus (hp) component using an optimized match function
        # mm_hp, index_hp, phi_hp = optimized_match(
        #     sp_td,
        #     sp_td_TEHM,
        #     psd=None,
        #     low_frequency_cutoff=f_low_phys,
        #     high_frequency_cutoff=f_high_phys,
        #     return_phase=True
        # )

        # Calculate the match for the h_plus (hp) component using an optimized match function
        mm_hp, index_hp, phi_hp = optimized_match(
            hp,
            hp_td_TEHM,
            psd=None,
            low_frequency_cutoff=f_low_phys,
            high_frequency_cutoff=f_high_phys,
            return_phase=True
        )

        # Calculate the time delay for h_plus based on the match result
        dt = index_hp * delta_t
        if index_hp > len(hp) / 2:
            dt = dt - hp.duration

        # print("hp", mm_hp, index_hp, dt, phi_hp)

        # # Repeat the match calculation for the h_cross (hc) component
        # mm_hc, index_hc, phi_hc = optimized_match(
        #     sc_td,
        #     sc_td_TEHM,
        #     psd=None,
        #     low_frequency_cutoff=f_low_phys,
        #     high_frequency_cutoff=f_high_phys,
        #     return_phase=True
        # )

        # Calculate the match for the h_plus (hp) component using an optimized match function
        mm_hc, index_hc, phi_hc = optimized_match(
            hc,
            hc_td_TEHM,
            psd=None,
            low_frequency_cutoff=f_low_phys,
            high_frequency_cutoff=f_high_phys,
            return_phase=True
        )

        # Calculate the time delay for h_cross based on the match result
        dt2 = index_hc * delta_t
        if index_hc > len(hc) / 2:
            dt2 = dt2 - hc.duration

        # print("hc", mm_hc, index_hc, dt2, phi_hc)

        # Apply a cutoff for matches below threshold values to avoid small numerical mismatches
        if (max(sp_td.data) and max(sp_td_TEHM.data)) < 1e-30:
            mm_hp = 1 - 1e-10
        if (max(sc_td.data) and max(sc_td_TEHM.data)) < 1e-30:
            mm_hc = 1 - 1e-10

        # Calculate the mean mismatch by averaging the h_plus and h_cross mismatches
        mm_mean = 1. - np.mean([mm_hp, mm_hc])
        # print("mm", mm_mean)

        # Calculate the real part of h_in2 using phase adjustments from match results
        h_in2 = (hp_td_TEHM.data * np.exp(-1j * phi_hp) - 1j * hc_td_TEHM.data * np.exp(-1j * phi_hc))
        h_in2abs = np.abs(h_in2)
        peak_pos_h2 = h_in2abs.argmax()  # Find peak position for h_in2
        times_hTE_full2 = hp_td_TEHM.get_sample_times() - hp_td_TEHM.get_sample_times()[peak_pos_h2] + (dt2 + dt) / 2.

        # Plot waveforms if requested
        if plots:
            plt.figure()
            plt.plot(times_s_full, np.real(s_in), c="violet", label="SXS")
            plt.plot(times_hTE_full2, np.real(h_in2), "--", c="forestgreen", label="TEHM B")
            plt.xlabel(r"$t[s]$")
            plt.ylabel(r"$h_+$")
            plt.xlim(-0.2, 0.02)
            plt.legend()
            plt.show()

            # Optional plotting for h_cross (uncomment to enable)
            # plt.figure()
            # plt.plot(times_s_full, np.imag(s_in), c="violet", label="SXS")
            # plt.plot(times_hTE_full2, np.imag(h_in2), "--", c="forestgreen", label="TEHM B")
            # plt.xlabel(r"$t[s]$")
            # plt.ylabel(r"$h_{\times}$")
            # plt.xlim(-0.2, 0.02)
            # plt.legend()
            # plt.show()

        # Return the mean mismatch, time delays, and phase shifts for h_plus and h_cross
        print(1 - mm_hp, 1 - mm_hc)
        return 1 - mm_hp, 1 - mm_hc, dt, phi_hp, dt2, phi_hc

    def calculate_mismatch(self, h_surrogate, h_real):

        # tlen = len(h_surrogate)
        # h_surr_TS = types.timeseries.TimeSeries(h_surrogate, delta_t=self.DeltaT)
        # TS = -h_surr_TS.sample_times[::-1]

        # TS_fig = plt.figure()
        # plt.plot(TS, h_surrogate, label='surr')
        # plt.plot(TS, h_real, label='real')

        # # Fourier transform of both waveforms
        # freq_surr = rfft(h_surrogate)
        # freq_real = rfft(h_real)

        # # Normalise frequency-domain waveforms
        # freq_surr= freq_surr / np.linalg.norm(freq_surr)
        # freq_real = freq_real / np.linalg.norm(freq_real)
        

        h_surr_TS = types.timeseries.TimeSeries(h_surrogate, delta_t=self.DeltaT)
        h_real_TS = types.timeseries.TimeSeries(h_real, delta_t=self.DeltaT)

        # Frequency array properties
        # seg_len = len(h_surrogate) // 2 + 1
        # delta_f = 1.0 / h_surr_TS.duration  # Frequency resolution

        # Create a flat PSD with ones across the frequency range
        # flat_psd = FrequencySeries(np.ones(seg_len), delta_f=delta_f)
        # fig___ = plt.figure()
        # plt.plot(h_real_TS.get_sample_times(), h_real_TS, label='real')
        # plt.plot(h_surr_TS.get_sample_times(), h_surr_TS, label='surr')
        # plt.legend()

        match_value, idx, phase = match(h_real_TS, h_surr_TS, psd=None, low_frequency_cutoff=self.freqmin, return_phase=True)
        optimatch_value, idx, optiphase = optimized_match(h_real_TS, h_surr_TS, psd=None, low_frequency_cutoff=self.freqmin, return_phase=True)
        print(match_value, phase)
        print(optimatch_value, optiphase)
        # # Compute the inner products
        # def inner_product(a, b, psd, delta_f):
        #     # print(a, b)
        #     return 4 * np.sum((a * np.conj(b) / psd).real) * delta_f
        
        # # Frequency resolution
        # delta_f = freqs[1] - freqs[0]
        
        # # Inner products for match calculation
        # print(freq_surr)
        # norm_surr = np.sqrt(inner_product(freq_surr, freq_surr, psd, delta_f))
        # norm_real = np.sqrt(inner_product(freq_real, freq_real, psd, delta_f))
        # match = inner_product(freq_real, freq_surr, psd, delta_f) / (norm_surr * norm_real)

        # freqs_fig = plt.figure()
        # plt.plot(freqs, freq_surr, label='surr')
        # plt.plot(freqs, freq_real, label='real')
        # plt.xlim(20, 100)
        # plt.show()

        
        # # Calculate the mismatch
        # mismatch = max(0, 1 - match)  # Clamp to zero if small negative due to numerical errors
        # print(norm_surr, norm_real, mismatch)
        return 1 - match_value, phase


    def get_surrogate_mismatches(self, plot_mismatches=False, save_mismatch_fig=False, plot_worst_err=False, save_mismatches_to_file=True):

        h_surrogate, _ , _ , generation_time= self.generate_surrogate_model()

        mismatches_hp = np.zeros(len(self.parameter_space_output))
        mismatches_hc = np.zeros(len(self.parameter_space_output))
        mismatches_hp_phase = np.zeros(len(self.parameter_space_output))
        mismatches_hc_phase = np.zeros(len(self.parameter_space_output))
        true_h_PS = np.zeros((len(self.parameter_space_output), self.waveform_size), dtype=complex)

        try:
            load_mismatches = np.load(f'Straindata/Mismatches/mismatches_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_Ni={len(self.parameter_space_input)}_No={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}.npz_')
            mismatches_hp = load_mismatches['mismatches_hp']
            mismatches_hc = load_mismatches['mismatches_hc']
            self.parameter_space_output = load_mismatches['PS']
            generation_time = load_mismatches['generation_time']
            print('Mismatch loaded')
        except:
            print('Calculating mismatch...')
            try:
                load_true_h = np.load(f'true_h_[{min(self.parameter_space_output)}_{max(self.parameter_space_output)}]_N={len(self.parameter_space_output)}.npz')
                true_h_PS = load_true_h['true_h']
                PS = load_true_h['PS']
                
                for i in range(len(PS)):
                    # mismatches_hp[i], mismatches_hc[i], dt_hp, mismatches_hp_phase[i], dt_hc, mismatches_hc_phase[i] = self.mismatch_NR_TEHM(np.real(h_surrogate[:, i]), np.imag(h_surrogate[:, i]), np.real(true_h_PS[i]), np.imag(true_h_PS[i]), self.DeltaT, PS[i], plots=False, plots_extra=False)
                    mismatches_hp[i], mismatches_hp_phase[i] = self.calculate_mismatch(np.real(h_surrogate[:, i]), np.real(true_h_PS[i]))
                    mismatches_hc[i], mismatches_hc_phase[i] = self.calculate_mismatch(np.imag(h_surrogate[:, i]), np.imag(true_h_PS[i]))

                # TS = np.arange(self.waveform_size)
                # for i in range(0, len(mismatches_hp), 50):

                #     fig_mm = plt.figure()
                #     plt.plot(TS, np.real(h_surrogate[:, i]), label='surr')
                #     plt.plot(TS, np.real(true_h_PS[i]), label='real')
                #     plt.title(f'ecc = {self.parameter_space_output[i]}, mismatch = {mismatches_hp[i]}')
                print('True h set loaded')
            except Exception as e:
                print("An error occurred:", e)
                traceback.print_exc()  # Prints the full traceback of the error

                for i, eccentricity in enumerate(self.parameter_space_output):
                    true_hp, true_hc, TS_M = self.simulate_inspiral_mass_independent(eccentricity)
                    
                    phase = np.array(waveform.utils.phase_from_polarizations(true_hp, true_hc))
                    amp = np.array(waveform.utils.amplitude_from_polarizations(true_hp, true_hc))
                    true_h = amp * np.exp(1j * phase)
                    length_diff = len(true_h) - self.waveform_size
                    true_h_PS[i] = true_h[length_diff:]

                    mismatches_hp[i], mismatches_hc[i], dt_hp, mismatches_hp_phase[i], dt_hc, mismatches_hc_phase[i] = self.mismatch_NR_TEHM(np.real(h_surrogate[:, i]), np.imag(h_surrogate[:, i]), np.real(true_h_PS[i]), np.imag(true_h_PS[i]), self.DeltaT)
                    # mismatches_hp[i] = self.calculate_mismatch(np.real(h_surrogate[:, i]), np.real(true_h[length_diff:]))
                    # mismatches_hc[i] = self.calculate_mismatch(np.imag(h_surrogate[:, i]), np.imag(true_h[length_diff:]))

                np.savez(f'true_h_[{min(self.parameter_space_output)}_{max(self.parameter_space_output)}]_N={len(self.parameter_space_output)}.npz', true_h=true_h_PS, PS=self.parameter_space_output)
            
        if plot_mismatches is True:

            fig_mismatches, axs = plt.subplots(2, sharex=True)

            axs[0].plot(self.parameter_space_output, mismatches_hp)
            axs[1].plot(self.parameter_space_output, mismatches_hc)
            axs[1].set_xlabel('eccentricity')
            axs[0].set_ylabel('mismatch $h_+$')
            axs[1].set_ylabel('mismatch $h_x$')
            axs[0].grid(True)
            axs[1].grid(True)
            axs[0].set_title(f'Mismatches: # input wfs = {len(self.parameter_space_input)}, greedy err $\phi$ = {self.min_greedy_error_phase}, greedy err A = {self.min_greedy_error_amp}')

            fig_mismatches_phase, axs = plt.subplots(2, sharex=True)

            axs[0].plot(self.parameter_space_output, mismatches_hp_phase)
            axs[1].plot(self.parameter_space_output, mismatches_hc_phase)
            axs[1].set_xlabel('eccentricity')
            axs[0].set_ylabel('mismatch $\phi_+$')
            axs[1].set_ylabel('mismatch $\phi_x$')
            axs[0].grid(True)
            axs[1].grid(True)
            axs[0].set_title(f'Mismatches $\phi$: # input wfs = {len(self.parameter_space_input)}, greedy err $\phi$ = {self.min_greedy_error_phase}, greedy err A = {self.min_greedy_error_amp}')

       
            if save_mismatch_fig is True:
                figname = 'Mismatches _q={}_ecc=[{}_{}]_iN={}_oN={}_gp={}_ga={}.png'.format(  self.mass_ratio, min(self.parameter_space_input), max(self.parameter_space_input), len(self.parameter_space_input), len(self.parameter_space_output), self.min_greedy_error_phase, self.min_greedy_error_amp)
                        
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Mismatches', exist_ok=True)
                fig_mismatches.savefig('Images/Mismatches/' + figname)

                print('Figure is saved in Images/Mismatches')

        if plot_worst_err is True:

            worst_err_hp_idx, worst_err_hc_idx = np.argmax(mismatches_hp), np.argmax(mismatches_hc)
            worst_err_hp, worst_err_hc = self.parameter_space_output[worst_err_hp_idx], self.parameter_space_output[worst_err_hc_idx]
            self.generate_surrogate_model(plot_surr_datapiece_at_ecc=worst_err_hp, plot_surr_at_ecc=worst_err_hp, plot_GPRfit=True, save_fig_datapiece=True, save_fig_surr=True)
            self.generate_surrogate_model(plot_surr_datapiece_at_ecc=worst_err_hc, plot_surr_at_ecc=worst_err_hc, save_fig_datapiece=True, save_fig_surr=True)

        if save_mismatches_to_file is not None and not os.path.isfile(f'Straindata/Mismatches/mismatches_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_Ni={len(self.parameter_space_input)}_No={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}.npz'):
            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Straindata/Mismatches', exist_ok=True)
            np.savez(f'Straindata/Mismatches/mismatches_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_Ni={len(self.parameter_space_input)}_No={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}.npz', mismatches_hp=mismatches_hp, mismatches_hc=mismatches_hc, PS=self.parameter_space_output, generation_time=generation_time)
            print('Mismatches saved to Straindata/Mismatches')

        return mismatches_hp, mismatches_hc, generation_time

    def mismatch_analysis(self, parameter_space, amount_output_wfs, greedy_error_check=None, input_wfs_check=None, save_mismatches_inputs_fig=False):

        amount_of_input_wfs = [30, 40, 50, 60, 70, 80, 100]
        worst_errs_hp, best_errs_hp, mean_errs_hp = [], [], []
        worst_errs_hc, best_errs_hc, mean_errs_hc = [], [], []
        generation_times = []

        for input_wfs in amount_of_input_wfs:

            surrogate = Surrogate_analysis(parameter_space=parameter_space, amount_input_wfs=input_wfs, amount_output_wfs=amount_output_wfs, min_greedy_error_amp=1e-3, min_greedy_error_phase=1e-3, waveform_size=self.waveform_size, freqmin=self.freqmin, mass_ratio=self.mass_ratio)
            mismatch_hp, mismatch_hc, generation_time = surrogate.get_surrogate_mismatches()

            generation_times.append(generation_time)
            worst_err_hp, best_err_hp, mean_err_hp = max(mismatch_hp), min(mismatch_hp), np.mean(mismatch_hp)
            worst_err_hc, best_err_hc, mean_err_hc = max(mismatch_hc), min(mismatch_hc), np.mean(mismatch_hc)
            
            worst_errs_hp.append(worst_err_hp), best_errs_hp.append(best_err_hp), mean_errs_hp.append(mean_err_hp)
            worst_errs_hc.append(worst_err_hc), best_errs_hc.append(best_err_hc), mean_errs_hc.append(mean_err_hc)

        fig_input_wfs = plt.figure()
        for i in range(len(amount_of_input_wfs)):
            plt.scatter(amount_of_input_wfs[i], worst_errs_hp[i], marker='+', color='red')
            plt.scatter(amount_of_input_wfs[i], best_errs_hp[i], marker='+', color='green')
            plt.scatter(amount_of_input_wfs[i], mean_errs_hp[i], marker='+', color='blue')
            plt.scatter(amount_of_input_wfs[i], worst_errs_hc[i], marker='x', color='red')
            plt.scatter(amount_of_input_wfs[i], best_errs_hc[i], marker='x', color='green')
            plt.scatter(amount_of_input_wfs[i], mean_errs_hc[i], marker='x', color='blue')
    
        plt.xlabel('# input waveforms')
        plt.ylabel('mismatch')
        plt.grid()

        fig_generation_time = plt.figure()
        for i in range(len(amount_of_input_wfs)):
            plt.scatter(worst_errs_hp[i], generation_times[i], marker='+')
            plt.plot(worst_errs_hp[i], generation_times[i], linewidth=0.6, label=f'N={amount_of_input_wfs[i]}')
            plt.scatter(worst_errs_hc[i], generation_times[i], marker='x')
            plt.plot(worst_errs_hc[i], generation_times[i], linewidth=0.6)
        plt.ylabel('surrogate generation time')
        plt.xlabel('worst mismatch h')
        plt.grid()


        if save_mismatches_inputs_fig is True:
                figname = 'Mismatches_inputs _q={}_ecc=[{}_{}]_o={}_gp={}_ga={}.png'.format(  self.mass_ratio, min(parameter_space), max(parameter_space), len(self.parameter_space_output), self.min_greedy_error_phase, self.min_greedy_error_amp)
                        
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Mismatches', exist_ok=True)
                fig_input_wfs.savefig('Images/Mismatches/' + figname)

                print('Figure is saved in Images/Mismatches')

    def surrogate_pointwise_error(self, plot_surr=True, save_pointwise_error_fig=True, save_pointwise_errors_to_file=True):
        
        try:
            data = np.load(f'Straindata/Pointwise_error/Pointwise_error_{self.parameter_space[0]}_{self.parameter_space[1]}_q={self.mass_ratio}_fmin={self.freqmin}_Ni={self.amount_input_wfs}_No={self.amount_output_wfs}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}_size={self.waveform_size}_2.npz')
            
            # data = np.load(f'Straindata/Pointwise_error/Pointwise_error_{self.parameter_space[0]}_{self.parameter_space[1]}__q={self.mass_ratio}_fmin={self.freqmin}_Ni={self.amount_input_wfs}_No={self.amount_output_wfs}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}_size={self.waveform_size}.npz')
            worst_err_amp = data['worst_err_amp']
            worst_err_phase = data['worst_err_phase']
            best_err_amp = data['best_err_amp']
            best_err_phase = data['best_err_phase']
            mean_err_amp = data['mean_err_amp']
            mean_err_phase = data['mean_err_phase']
            generation_time_amp = data['generation_t_amp']
            generation_time_phase = data['generation_t_phase']
            generation_time_full_wfs = data['generation_t_full_wfs']
            print('Pointwise errors loaded')

        except Exception as e:
            print(e)

            def delete_existing_file(file_path):
                # Check if the file exists before attempting to delete
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"File '{file_path}' has been deleted.")
                else:
                    print(f"File '{file_path}' does not exist.")
                    
            # Delete saved polarisations if exists
            # file_path_polarisations = f'Straindata/Polarisations/polarisations_e=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}]_N={len(self.parameter_space_input)}_fmin={self.freqmin}__q={self.mass_ratio}.npz'
            # file_path_res_amps = f'Straindata/Residuals/residuals_amplitude__q={self.mass_ratio}_fmin={self.freqmin}_e=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}]_N={len(self.parameter_space_input)}.npz'
            # file_path_res_phase = f'Straindata/Residuals/residuals_phase__q={self.mass_ratio}_fmin={self.freqmin}_e=[{min(self.parameter_space_input)}_{max(self.parameter_space_input)}]_N={len(self.parameter_space_input)}.npz'
            # file_path_GPR_amp = f'Straindata/GPRfits/amplitude__q={self.mass_ratio}_fmin={self.freqmin}_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_Ni={len(self.parameter_space_input)}_No={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}_size={self.waveform_size}.npz'
            # file_path_GPR_phase = f'Straindata/GPRfits/phase__q={self.mass_ratio}_fmin={self.freqmin}_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_Ni={len(self.parameter_space_input)}_No={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}_size={self.waveform_size}.npz'
            # file_path_pieces = f'Straindata/Surrogate_datapieces/Surrogate_datapieces__q={self.mass_ratio}_fmin={self.freqmin}_{min(self.parameter_space_input)}_{max(self.parameter_space_input)}_Ni={len(self.parameter_space_input)}_No={len(self.parameter_space_output)}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}_size={self.waveform_size}.npz'
            # delete_existing_file(file_path_polarisations)
            # delete_existing_file(file_path_res_amps)
            # delete_existing_file(file_path_res_phase)
            # delete_existing_file(file_path_GPR_amp)
            # delete_existing_file(file_path_GPR_phase)
            # delete_existing_file(file_path_pieces)

            surrogate_h, surrogate_amp, surrogate_phase, generation_time_amp, generation_time_phase = self.generate_surrogate_model(save_surr_to_file=True)

            try:
                load_polarisations = np.load(f'Straindata/Polarisations/polarisations_e=[{min(self.parameter_space_output)}_{max(self.parameter_space_output)}]_N={len(self.parameter_space_output)}_fmin={self.freqmin}__q={self.mass_ratio}_2.npz', allow_pickle=True)
                true_hp = load_polarisations['hp']
                true_hc = load_polarisations['hc']
                TS = load_polarisations['TS'][-self.waveform_size:]
                generation_time_full_wfs = load_polarisations['computational_t']

            except Exception as e:
                print(e)

                # Simulate the real waveform datapiece
                true_hp = np.empty(len(self.parameter_space_output), dtype=object)
                true_hc = np.empty(len(self.parameter_space_output), dtype=object)

                start_timer = time.time()
                for i, eccentricity in enumerate(self.parameter_space_output):
                    real_hp, real_hc, TS = self.simulate_inspiral_mass_independent(eccentricity)
                    true_hp[i], true_hc[i] = real_hp, real_hc
                    TS = TS[-self.waveform_size:]
                
                end_timer = time.time()
                generation_time_full_wfs = end_timer - start_timer


                np.savez(f'Straindata/Polarisations/polarisations_e=[{min(self.parameter_space_output)}_{max(self.parameter_space_output)}]_N={len(self.parameter_space_output)}_fmin={self.freqmin}__q={self.mass_ratio}_2.npz', hp=true_hp, hc=true_hc, TS=TS, computational_t=generation_time_full_wfs)
                print('True polarisations saved')

            worst_err_amp = np.zeros(len(self.parameter_space_output))
            best_err_amp = np.zeros(len(self.parameter_space_output))
            mean_err_amp = np.zeros(len(self.parameter_space_output))

            worst_err_phase = np.zeros(len(self.parameter_space_output))
            best_err_phase = np.zeros(len(self.parameter_space_output))
            mean_err_phase = np.zeros(len(self.parameter_space_output))

            for i in range(len(self.parameter_space_output)):
                true_amp = np.array(waveform.utils.amplitude_from_polarizations(true_hp[i], true_hc[i]))[-len(surrogate_amp):]
                true_phase = np.array(waveform.utils.phase_from_polarizations(true_hp[i], true_hc[i]))[-len(surrogate_phase):]

                # pointwise_error_phase = abs(surrogate_phase.T[i] - true_phase) / abs(true_phase)
                # pointwise_error_amp = abs(surrogate_amp.T[i] - true_amp) / abs(true_amp)

                # At devision by zero encounter set error equal to zero
                pointwise_error_amp = np.where(true_amp == 0, 0, abs(surrogate_amp.T[i] - true_amp) / abs(true_amp))
                pointwise_error_phase = np.where(true_phase == 0, 0, abs(surrogate_phase.T[i] - true_phase) / abs(true_phase))


                best_err_amp[i], worst_err_amp[i], mean_err_amp[i] = min(pointwise_error_amp), max(pointwise_error_amp), np.mean(pointwise_error_amp)
                best_err_phase[i], worst_err_phase[i], mean_err_phase[i] = min(pointwise_error_phase), max(pointwise_error_phase), np.mean(pointwise_error_phase)

            if save_pointwise_errors_to_file is True and not os.path.isfile(f'Straindata/Pointwise_error/Pointwise_error_{self.parameter_space[0]}_{self.parameter_space[1]}__q={self.mass_ratio}_fmin={self.freqmin}_Ni={self.amount_input_wfs}_No={self.amount_output_wfs}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}_size={self.waveform_size}_2.npz'):
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Straindata/Pointwise_error', exist_ok=True)
                np.savez(f'Straindata/Pointwise_error/Pointwise_error_{self.parameter_space[0]}_{self.parameter_space[1]}__q={self.mass_ratio}_fmin={self.freqmin}_Ni={self.amount_input_wfs}_No={self.amount_output_wfs}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}_Ngp={self.N_greedy_vecs_phase}_Nga={self.N_greedy_vecs_amp}_size={self.waveform_size}_2.npz', best_err_amp=best_err_amp, mean_err_amp=mean_err_amp, worst_err_amp=worst_err_amp, best_err_phase=best_err_phase, mean_err_phase=mean_err_phase, worst_err_phase=worst_err_phase, generation_t_amp=generation_time_amp, generation_t_phase=generation_time_phase, generation_t_full_wfs=generation_time_full_wfs)
                print('Pointwise errors saved to Straindata/Pointwise_error')

        fig_pointwise_err, axs = plt.subplots(2, sharex=True, figsize=(10, 5))
        
        axs[0].plot(self.parameter_space_output, best_err_phase, linewidth=0.6, label=f'best error', color='green')
        axs[0].plot(self.parameter_space_output, worst_err_phase, linewidth=0.6, label=f'worst error', color='red')
        axs[0].plot(self.parameter_space_output, mean_err_phase, linewidth=0.6, label=f'mean error', color='blue')

        axs[0].set_ylabel('|($\phi_S$ - $\phi$) / $\phi$|')
        axs[0].grid(True)
        axs[0].legend()

        axs[1].plot(self.parameter_space_output, best_err_amp, linewidth=0.6, label=f'best error', color='green')
        axs[1].plot(self.parameter_space_output, worst_err_amp, linewidth=0.6, label=f'worst error', color='red')
        axs[1].plot(self.parameter_space_output, mean_err_amp, linewidth=0.6, label=f'mean error', color='blue')
        
        axs[1].set_xlabel('eccentricity')
        axs[1].set_ylabel('|($A_S$ - A) / A|')
        axs[1].grid(True)
        axs[1].legend()
        # if self.N_greedy_vecs_amp is None:
            # axs[0].set_title(f'greedy error = {self.min_greedy_error_amp}')
        # else:
            # axs[0].set_title(f'# greedy basis vectors = {self.N_greedy_vecs_amp}')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        if save_pointwise_error_fig is True:
            figname_pointwise = 'Pointwise_errors_q={}_ecc=[{}_{}]_fmin={}_iN={}_oN={}_gp={}_ga={}_Ngp={}_Nga={}_size={}.png'.format(  self.mass_ratio, min(self.parameter_space), max(self.parameter_space), self.freqmin, self.amount_input_wfs, self.amount_output_wfs, self.min_greedy_error_phase, self.min_greedy_error_amp, self.N_greedy_vecs_phase, self.N_greedy_vecs_amp, self.waveform_size)
            
            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Images/Pointwise_error', exist_ok=True)
            fig_pointwise_err.savefig('Images/Pointwise_error/' + figname_pointwise)

            print('Figures are saved in Images/Pointwise_error')

        if plot_surr is True:
            sorted_phase = np.argsort(mean_err_phase)
            sorted_amp = np.argsort(mean_err_amp)
            worst_phase_no_peak = None

            diff_phase_values = np.diff(sorted_phase)
            peak_indices = np.where(diff_phase_values > 1e-3)[0]

            cleaned_mean_err_phase = np.delete(mean_err_phase, sorted_phase[peak_indices + 1])
            ecc_worst_phase_no_peak = np.argmax(cleaned_mean_err_phase)
            print(np.sort(cleaned_mean_err_phase)[-5:] )
            self.generate_surrogate_model(plot_surr_at_ecc=ecc_worst_phase_no_peak, plot_surr_datapiece_at_ecc=ecc_worst_phase_no_peak, plot_GPRfit=True, save_fig_datapiece=True, save_fig_surr=True, save_fig_fits=True)

            ecc_worst_phase = self.parameter_space_output[sorted_phase[-1]]
            ecc_worst_amp = self.parameter_space_output[sorted_amp[-1]]

            # for i in range(len(worst_err_phase)):
            #     if worst_err_phase[i] > np.mean(worst_err_phase)*10:
            #         worst_err_phase[i] = 0
            # for i in range(len(worst_err_phase) - 1):
            #     if np.diff(worst_err_phase)[i] > 2:
                    # print(worst_err_phase[i], np.diff(worst_err_phase)[i])

            # ecc_second_worst_phase = self.parameter_space_output[np.argmax(worst_err_phase)]
            # ecc_best_phase = self.parameter_space_output[np.argmin(worst_err_phase)]

            # print('worst err phase', worst_err_phase)
            # print(self.parameter_space_output)

            self.generate_surrogate_model(plot_surr_at_ecc=0.2, plot_surr_datapiece_at_ecc=0.2, plot_GPRfit=True, save_fig_datapiece=True, save_fig_surr=True, save_fig_fits=True)
            
            # self.generate_surrogate_model(plot_surr_at_ecc=0.2752, plot_surr_datapiece_at_ecc=0.2752, plot_GPRfit=True, save_fig_datapiece=True, save_fig_surr=True, save_fig_fits=True)
            # self.generate_surrogate_model(plot_surr_at_ecc=0.3, plot_surr_datapiece_at_ecc=0.3, plot_GPRfit=True, save_fig_datapiece=True, save_fig_surr=True, save_fig_fits=True)
            self.generate_surrogate_model(plot_surr_at_ecc=ecc_worst_amp, plot_surr_datapiece_at_ecc=ecc_worst_amp, plot_GPRfit=True, save_fig_datapiece=True, save_fig_surr=True, save_fig_fits=True)

        return best_err_amp, mean_err_amp, worst_err_amp, best_err_phase, mean_err_phase, worst_err_phase, generation_time_amp, generation_time_phase, generation_time_full_wfs
    

    def pointwise_error_analysis(self, parameter_space, amount_input_wfs, amount_output_wfs, greedy_errors=None, N_greedy_vecs_phase=None, N_greedy_vecs_amp=None, save_pointwise_errors_fig=True, save_pointwise_errors_to_file=True):
        
        if (greedy_errors is not None) and (N_greedy_vecs_phase is not None):
            print('Choose either "greedy_errors" or "N_greedy_vecs"!')
            sys.exit(1)

        if (greedy_errors is None) and (N_greedy_vecs_phase is not None):
            greedy_errors = N_greedy_vecs_phase

        worst_of_mean_errs_phase, best_of_mean_errs_phase, mean_of_mean_errs_phase = np.zeros(len(greedy_errors)), np.zeros(len(greedy_errors)), np.zeros(len(greedy_errors))
        worst_of_mean_errs_amp, best_of_mean_errs_amp, mean_of_mean_errs_amp = np.zeros(len(greedy_errors)), np.zeros(len(greedy_errors)), np.zeros(len(greedy_errors))
        generation_times_surr_amp, generation_times_surr_phase = np.zeros(len(greedy_errors)), np.zeros(len(greedy_errors))

        if (N_greedy_vecs_phase is None):
            for i, greedy_err in enumerate(greedy_errors):
                print(f'Calculate pointwise error for: greedy_error = {greedy_err}')

                surrogate = Surrogate_analysis(parameter_space=parameter_space, amount_input_wfs=amount_input_wfs, amount_output_wfs=amount_output_wfs, min_greedy_error_amp=greedy_err, min_greedy_error_phase=greedy_err, waveform_size=self.waveform_size, freqmin=self.freqmin, mass_ratio=self.mass_ratio)
                load_phase_shifts = np.load(f'Straindata/Phaseshift/estimated_phase_shift_q={surrogate.mass_ratio}_fmin={surrogate.freqmin}_e=[0.01_0.4].npz')
                loaded_phase_shift = load_phase_shifts['total_phase_shift']
                loaded_parameter_space = load_phase_shifts['parameter_space']
                print(loaded_phase_shift)
                best_errs_amp, mean_errs_amp, worst_errs_amp, best_errs_phase, mean_errs_phase, worst_errs_phase, generation_times_surr_amp[i], generation_times_surr_phase[i], generation_t_full_wfs = surrogate.surrogate_pointwise_error(save_pointwise_error_fig=True)

                    
                # Get best, mean and worst value of the WORST approximated points in eccentric waveforms.
                # for j in range(len(worst_errs_phase)):
                #     if worst_errs_phase[j] > np.mean(worst_errs_phase)*10:
                #         print(worst_errs_phase[j])

                worst_of_mean_errs_phase[i], best_of_mean_errs_phase[i], mean_of_mean_errs_phase[i] = max(mean_errs_phase), min(mean_errs_phase), np.mean(mean_errs_phase)
                worst_of_mean_errs_amp[i], best_of_mean_errs_amp[i], mean_of_mean_errs_amp[i] = max(mean_errs_amp), min(mean_errs_amp), np.mean(mean_errs_amp)
        
        elif (N_greedy_vecs_phase is not None):
            
            for i, (N_vecs_amp, N_vecs_phase) in enumerate(zip(N_greedy_vecs_amp, N_greedy_vecs_phase)):
                print(f'Calculate pointwise error for: N greedy basis vecs phase = {N_vecs_phase}, N vecs amp = {N_vecs_amp}')

                surrogate = Surrogate_analysis(parameter_space=parameter_space, amount_input_wfs=amount_input_wfs, amount_output_wfs=amount_output_wfs, N_greedy_vecs_amp=N_vecs_amp, N_greedy_vecs_phase=N_vecs_phase, waveform_size=self.waveform_size, freqmin=self.freqmin, mass_ratio=self.mass_ratio)
                best_errs_amp, mean_errs_amp, worst_errs_amp, best_errs_phase, mean_errs_phase, worst_errs_phase, generation_times_surr_amp[i], generation_times_surr_phase[i], generation_t_full_wfs = surrogate.surrogate_pointwise_error(save_pointwise_error_fig=True)

                worst_of_mean_errs_phase[i], best_of_mean_errs_phase[i], mean_of_mean_errs_phase[i] = max(mean_errs_phase), min(mean_errs_phase), np.mean(mean_errs_phase)
                worst_of_mean_errs_amp[i], best_of_mean_errs_amp[i], mean_of_mean_errs_amp[i] = max(mean_errs_amp), min(mean_errs_amp), np.mean(mean_errs_amp)
        else:
            print('Choose "greedy_errors" or "N_greedy_vecs"!')
            sys.exit(1)


        # Create subplots
        fig_pointwise_errs, axs = plt.subplots(1, 2, figsize=(7, 5), sharex=True, gridspec_kw={'wspace': 0.05})

        # Loop through subplots
        axs[0].scatter(greedy_errors, mean_of_mean_errs_phase, marker='s', label='phase')
        axs[1].scatter(greedy_errors, mean_of_mean_errs_amp, marker='^', label='amplitude')
        axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
        axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
        axs[1].yaxis.set_label_position('right')
        axs[1].yaxis.tick_right()
     
        axs[0].grid()
        axs[1].grid()
        axs[0].legend()
        axs[1].legend()

        fig_pointwise_errs.subplots_adjust(bottom=0.1, left=0.1)
        if N_greedy_vecs_amp is not None:
            fig_pointwise_errs.supxlabel('N greedy basis vectors')
        else:
            fig_pointwise_errs.supxlabel('greedy error cut-off')
        fig_pointwise_errs.supylabel('mean of time-domain mean errors')
        




        colors = sns.color_palette("tab20")
        fig_generation_time, axs = plt.subplots(1, 2, figsize=(7, 5), gridspec_kw={'wspace': 0.05})  # Increased width for clarity

        # Plotting data for phase and amplitude with labels
        if N_greedy_vecs_amp is None:
            for i in range(len(greedy_errors)):
                axs[0].scatter(
                    mean_of_mean_errs_phase[i], generation_times_surr_phase[i],
                    marker='s', color=colors[i], label=f'greedy error = {greedy_errors[i]}'
                )
                axs[1].scatter(
                    mean_of_mean_errs_amp[i], generation_times_surr_amp[i],
                    marker='^', color=colors[i], label=f'greedy error = {greedy_errors[i]}'
                )
        else:
            for i in range(len(greedy_errors)):
                axs[0].scatter(
                    mean_of_mean_errs_phase[i], generation_times_surr_phase[i],
                    marker='s', color=colors[i], label=f'N basis vectors = {N_greedy_vecs_phase[i]}'
                )
                axs[1].scatter(
                    mean_of_mean_errs_amp[i], generation_times_surr_amp[i],
                    marker='^', color=colors[i], label=f'N basis vectors = {N_greedy_vecs_amp[i]}'
                )

        # Adding dashed line for complete dataset computation

        # axs[0].plot(
        #     np.linspace(min(mean_of_mean_errs_phase) * 0.8, max(mean_of_mean_errs_phase) * 1.2, num=1000),
        #     np.full(1000, generation_t_full_wfs),
        #     linestyle='dashed',
        #     color='black',
        #     lw=1.1,
        #     label='original polarisations'
        # )
        # axs[1].plot(
        #         np.linspace(min(mean_of_mean_errs_phase) * 0.8, max(mean_of_mean_errs_amp) * 1.2, num=1000),
        #         np.full(1000, generation_t_full_wfs),
        #         linestyle='dashed',
        #         color='black',
        #         lw=1.1,
        #         label='original polarisations'
        #     )

        # Custom legend elements for markers
        custom_legend_element_phase = [Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=7, label='phase')]
        custom_legend_element_amp = [Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=8, label='amplitude')]

        # Combine legends for each subplot

        handles, labels = axs[0].get_legend_handles_labels()  # Get auto-generated legend entries from each axis
        combined_handles = custom_legend_element_phase + handles  # Combine custom and auto-generated elements
        combined_labels = [h.get_label() for h in combined_handles]  # Extract labels from handles
        # axs[0].legend(combined_handles, combined_labels, loc='upper left', bbox_to_anchor=(0.005, 0.93))
        axs[0].legend(combined_handles, combined_labels, fontsize=10)
        
        handles, labels = axs[1].get_legend_handles_labels()  # Get auto-generated legend entries from each axis
        combined_handles = custom_legend_element_amp + handles  # Combine custom and auto-generated elements
        combined_labels = [h.get_label() for h in combined_handles]  # Extract labels from handles
        # axs[0].legend(combined_handles, combined_labels, loc='upper left', bbox_to_anchor=(0.005, 0.93))
        axs[1].legend(combined_handles, combined_labels, fontsize=10)

        # Configure scientific notation and grid for both axes
        for ax in axs:
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
            ax.grid()

        # Adjust right axis ticks and labels
        axs[1].yaxis.set_label_position('right')
        axs[1].yaxis.tick_right()

        # Shared labels for the entire figure
        fig_generation_time.supylabel('surrogate computational time [s]')
        fig_generation_time.subplots_adjust(bottom=0.15)
        fig_generation_time.supxlabel('mean of time-domain mean errors')

        plt.tight_layout()

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        colors = sns.color_palette("tab20")
        fig_generation_time, axs = plt.subplots(1, 2, figsize=(7, 5), gridspec_kw={'wspace': 0.05})  # Adjust spacing for clarity

        # Define break points independently for each subplot
        lower_break_point_phase = max(generation_times_surr_phase) * 1.2  # Lower break point for the phase subplot
        upper_break_point_phase = generation_t_full_wfs * 0.995  # Upper limit for the phase subplot

        lower_break_point_amp = max(generation_times_surr_amp) * 1.2  # Lower break point for the amplitude subplot
        upper_break_point_amp = generation_t_full_wfs * 0.995  # Upper limit for the amplitude subplot

        # Plotting data for phase and amplitude with labels
        if N_greedy_vecs_amp is None:
            for i in range(len(greedy_errors)):
                axs[0].scatter(
                    mean_of_mean_errs_phase[i], generation_times_surr_phase[i],
                    marker='s', color=colors[i], label=f'greedy error = {greedy_errors[i]}'
                )
                axs[1].scatter(
                    mean_of_mean_errs_amp[i], generation_times_surr_amp[i],
                    marker='^', color=colors[i], label=f'greedy error = {greedy_errors[i]}'
                )
        else:
            for i in range(len(greedy_errors)):
                axs[0].scatter(
                    mean_of_mean_errs_phase[i], generation_times_surr_phase[i],
                    marker='s', color=colors[i], label=f'N basis vectors = {N_greedy_vecs_phase[i]}'
                )
                axs[1].scatter(
                    mean_of_mean_errs_amp[i], generation_times_surr_amp[i],
                    marker='^', color=colors[i], label=f'N basis vectors = {N_greedy_vecs_amp[i]}'
                )

        # # Adding dashed line for complete dataset computation
        # axs[0].plot(
        #     np.linspace(min(mean_of_mean_errs_phase) * 0.8, max(mean_of_mean_errs_phase) * 1.2, num=1000),
        #     np.full(1000, generation_t_full_wfs),
        #     linestyle='dashed',
        #     color='black',
        #     lw=1.1,
        #     label='original polarisations'
        # )
        # axs[1].plot(
        #     np.linspace(min(mean_of_mean_errs_amp) * 0.8, max(mean_of_mean_errs_amp) * 1.2, num=1000),
        #     np.full(1000, generation_t_full_wfs),
        #     linestyle='dashed',
        #     color='black',
        #     lw=1.1,
        #     label='original polarisations'
        # )

        # Apply broken y-axis independently for both subplots
        for i, (ax, mean_errs, generation_times, lower_break_point, upper_break_point, marker) in enumerate(
                zip(axs,
                    [mean_of_mean_errs_phase, mean_of_mean_errs_amp],
                    [generation_times_surr_phase, generation_times_surr_amp],
                    [lower_break_point_phase, lower_break_point_amp],
                    [upper_break_point_phase, upper_break_point_amp],
                    ['s','^'])):

            # Lower part of the plot
            ax.set_ylim(0, lower_break_point)
            ax.spines['top'].set_visible(False)

            # Create an upper plot for the broken y-axis
            divider = make_axes_locatable(ax)
            ax_upper = divider.new_vertical(size="40%", pad=0.1)
            fig_generation_time.add_axes(ax_upper)

            # Upper part of the plot
            ax_upper.set_ylim(upper_break_point, generation_t_full_wfs * 1.005)
            ax_upper.spines['bottom'].set_visible(False)
            ax_upper.tick_params(bottom=False, labelbottom=False)

            # Adding dashed line for complete dataset computation
            ax_upper.plot(
                np.linspace(min(mean_of_mean_errs_phase) * 0.8, max(mean_of_mean_errs_phase) * 1.2, num=1000),
                np.full(1000, generation_t_full_wfs),
                linestyle='dashed',
                color='black',
                lw=1.1,
                label='original polarisations'
            )

            # Add diagonal lines for the axis break
            d = 0.015  # Normalized diagonal line size (adjust as needed for consistency)

            # Calculate the relative aspect ratio between lower and upper axes
            lower_height = ax.get_position().height
            upper_height = ax_upper.get_position().height
            aspect_ratio_correction = lower_height / upper_height

            # Adjust the diagonal line slope
            kwargs = dict(color='k', clip_on=False)

            # Diagonal lines for lower axis
            ax.plot((-d, +d), (1 - d, 1 + d * aspect_ratio_correction), transform=ax.transAxes, **kwargs)
            ax.plot((1 - d, 1 + d), (1 - d, 1 + d * aspect_ratio_correction), transform=ax.transAxes, **kwargs)

            # Diagonal lines for upper axis
            ax_upper.plot((-d, +d), (-d * aspect_ratio_correction, +d * aspect_ratio_correction), transform=ax_upper.transAxes, **kwargs)
            ax_upper.plot((1 - d, 1 + d), (-d * aspect_ratio_correction, +d * aspect_ratio_correction), transform=ax_upper.transAxes, **kwargs)

            ax_upper.grid()

            # Enable grid for lower axis only
            ax.grid()

            # Adjust ticks
            if i == 0:
                # For the first subplot, only left ticks
                ax.tick_params(left=True, right=False, labelleft=True, labelright=False)
                ax_upper.tick_params(left=True, right=False, labelleft=True, labelright=False)
            else:
                # For the second subplot, only right ticks
                ax.tick_params(left=False, right=True, labelleft=False, labelright=True)
                ax_upper.tick_params(left=False, right=True, labelleft=False, labelright=True)

        # Custom legend elements for markers
        custom_legend_element_phase = [Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=7, label='phase')]
        custom_legend_element_amp = [Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=8, label='amplitude')]

        # Combine legends for each subplot
        handles, labels = axs[0].get_legend_handles_labels()
        combined_handles = custom_legend_element_phase + handles
        combined_labels = [h.get_label() for h in combined_handles]
        axs[0].legend(combined_handles, combined_labels, fontsize=10)

        handles, labels = axs[1].get_legend_handles_labels()
        combined_handles = custom_legend_element_amp + handles
        combined_labels = [h.get_label() for h in combined_handles]
        axs[1].legend(combined_handles, combined_labels, fontsize=10)

        # Adjust right axis ticks and labels
        axs[1].yaxis.set_label_position('right')

        # Shared labels for the entire figure
        fig_generation_time.supylabel('surrogate computational time [s]')
        fig_generation_time.subplots_adjust(bottom=0.15)
        fig_generation_time.supxlabel('mean of time-domain mean errors')

        plt.tight_layout()








        if save_pointwise_errors_fig is True:
                figname_pointwise = 'Pointwise_errors_analysis_q={}_ecc=[{}_{}]_iN={}_oN={}_fmin={}_size={}.png'.format(self.mass_ratio, min(parameter_space), max(parameter_space), amount_input_wfs, amount_output_wfs, self.freqmin, self.waveform_size)
                figname_compu_cost = 'Computational_cost_q={}_ecc=[{}_{}]_iN={}_oN={}_fmin={}_size={}.png'.format(self.mass_ratio, min(parameter_space), max(parameter_space), amount_input_wfs, amount_output_wfs, self.freqmin, self.waveform_size)
                        
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Pointwise_error', exist_ok=True)
                os.makedirs('Images/Computational_efficiency', exist_ok=True)
                fig_pointwise_errs.savefig('Images/Pointwise_error/' + figname_pointwise)
                fig_generation_time.savefig('Images/Computational_efficiency/' + figname_compu_cost)

                print('Figures are saved in Images/Pointwise_error and Images/Computational_efficiency')
    
    # def mass_system_analysis(self, Ninput, N_greedy_vecs_phase, N_greedy_vecs_amp, sizes, save_figs=True):
    #     generation_times_surr_amp, generation_times_surr_phase = np.zeros(len(Ninput)), np.zeros(len(Ninput))
    #     generation_t_full_wfs, mean_errors_amp, mean_errors_phase = np.zeros(len(Ninput)), np.zeros(len(Ninput)), np.zeros(len(Ninput))

    #     for N_vecs_amp, N_vecs_phase, Ni, size, i  in zip(N_greedy_vecs_amp, N_greedy_vecs_phase, Ninput, sizes, [0, 1, 2, 3, 4]):
    #         surrogate = Surrogate_analysis(parameter_space=self.parameter_space, amount_input_wfs=Ni, amount_output_wfs=self.amount_output_wfs, N_greedy_vecs_amp=N_vecs_amp, N_greedy_vecs_phase=N_vecs_phase, waveform_size=size, freqmin=self.freqmin)
    #         best_err_amp, mean_error_amp, worst_err_amp, best_err_phase, mean_error_phase, worst_err_phase, generation_times_surr_amp[i], generation_times_surr_phase[i], generation_t_full_wfs[i] = surrogate.surrogate_pointwise_error(save_pointwise_error_fig=True)
    #         mean_errors_amp[i], mean_errors_phase[i] = np.mean(mean_error_amp), np.mean(mean_error_phase)

    #     fig_time_efficiency_amp = plt.subplots(len(Ninput), len(N_greedy_vecs_amp))


    #     fig_time_efficiency_phase = plt.subplots(len())
        
        
        
        
    #     fig_mass_analysis = plt.figure(figsize=(7, 5))
    #     colors = sns.color_palette("tab20")

    #     scatter_handles = []  # To collect iterated labels
    #     # Plot the data
    #     for i in range(len(Ninput)):
    #         # First scatter (original polarisations) with no legend
    #         plt.scatter(masses[i], generation_t_full_wfs[i], marker='s', color=colors[i])
    #         # Second scatter (surrogate polarisations) with a label for each iteration
    #         handle = plt.scatter(
    #             masses[i],
    #             generation_times_surr_amp[i] + generation_times_surr_phase[i],
    #             marker='^',
    #             color=colors[i],
    #             label='$A_{err}$ = ' + f'{mean_errors_amp[i]:.2e}' + ' $\phi_{err}$ = ' + f'{mean_errors_phase[i]:.2e}'
    #         )

    #         scatter_handles.append(handle)

    #     # Custom legend for marker shapes
    #     custom_legend_elements = [
    #         Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=8.5, label='original polarisations'),
    #         Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=9.5, label='surrogate polarisations'),
    #     ]

    #     # Add the custom legend
    #     plt.legend(handles=custom_legend_elements + scatter_handles, fontsize=9.5)

    #     # Add labels and grid
    #     plt.xlabel('total binary mass [$M_{\\odot}$]', fontsize=14)
    #     plt.ylabel('computational time [s]', fontsize=14)
    #     # fig_mass_analysis.supxlabel('total binary mass [$M_{\\odot}$]')
    #     # fig_mass_analysis.supylabel('computational time [s]')
    #     plt.yscale('log')
    #     plt.grid()

    #     # plt.tight_layout()

    #     # # Save the figure
    #     # figname = 'Total_surrogate_efficiency_q={}_ecc=[{}_{}]_f_min={}.png'.format(1, 0.01, 0.3, 20)  # Example parameters
    #     # os.makedirs('Images/Surrogate_efficiency', exist_ok=True)
    #     # fig_mass_analysis.savefig('Images/Surrogate_efficiency/' + figname)

    #     # print('Figure is saved in Images/Surrogate_Efficiency')





    #     # Create subplots
    #     fig_pointwise_errs_masses, axs = plt.subplots(1, 2, figsize=(7, 5), sharex=True, gridspec_kw={'wspace': 0.05})
    #     scatter_handles = []
    #     # Loop through subplots
    #     for i in range(len(Ninput)):
    #         handle = axs[0].scatter(N_greedy_vecs_phase[i], mean_errors_phase[i], marker='s', label=f'$M$ = {masses[i]}' + ' $M_{\odot}$', color=colors[i])
    #         axs[1].scatter(N_greedy_vecs_amp[i], mean_errors_amp[i], marker='^', color=colors[i])

    #         scatter_handles.append(handle)

    #     axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    #     axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    #     axs[1].yaxis.set_label_position('right')
    #     axs[1].yaxis.tick_right()
     
    #     axs[0].grid()
    #     axs[1].grid()

    #             # Custom legend for marker shapes
    #     custom_legend_element_phase = [
    #         Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=8.5, label='phase')]
    #     custom_legend_element_amp = [
    #         Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=9.5, label='amplitude')]

    #     axs[0].legend(handles=custom_legend_element_phase + scatter_handles, fontsize=9.5)
    #     axs[1].legend(handles=custom_legend_element_amp + scatter_handles, fontsize=9.5)

    #     # plt.xlabel('N greedy basis vectors')
    #     # plt.ylabel('mean of time-domain mean errors')
    #     fig_pointwise_errs_masses.subplots_adjust(bottom=0.1, left=0.1)
    #     fig_pointwise_errs_masses.supxlabel('N greedy basis vectors', fontsize=14)
    #     fig_pointwise_errs_masses.supylabel('mean of time-domain mean errors', fontsize=14)

    #     # plt.tight_layout()


    #     if save_figs is True:
    #             # Save the figure
    #             figname_masses_times = 'Total_surrogate_efficiency_times_q={}_ecc=[{}_{}]_f_min={}.png'.format(1, 0.01, 0.3, 20)  # Example parameters
    #             figname_masses_errors = 'Total_surrogate_efficient_errors_q={}_ecc=[{}_{}]_f_min={}.png'.format(1, 0.01, 0.3, 20)  # Example parameters
                
    #             os.makedirs('Images/Surrogate_efficiency', exist_ok=True)

    #             fig_pointwise_errs_masses.savefig('Images/Surrogate_efficiency/' + figname_masses_errors)
    #             fig_mass_analysis.savefig('Images/Surrogate_efficiency/' + figname_masses_times)

    #             print('Figures are saved in Images/Pointwise_error and Images/Surrogate_efficiency')

        # fig_mass_analysis = plt.figure()

        # colors = sns.color_palette("tab20")

        # for i in range(len(Ninput)):
        #     plt.scatter(masses[i], generation_t_full_wfs[i], marker='s', color=colors[i])
        #     plt.scatter(masses[i], generation_times_surr_amp[i] + generation_times_surr_phase[i], marker='^', color=colors[i], label=f'N basis vecs A = {N_greedy_vecs_amp[i]}, N basis vecs $\phi$ = {N_greedy_vecs_phase[i]} ')

        # # Custom legend elements for markers
        # custom_legend_elements = [Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=7, label='original polarisations'),
        #                           Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=8, label='surrogate polarisations')]

        # handles, labels = plt.get_legend_handles_labels()  # Get auto-generated legend entries from each axis
        # combined_handles = custom_legend_elements + handles  # Combine custom and auto-generated elements
        # combined_labels = [h.get_label() for h in combined_handles]  # Extract labels from handles
       
        # plt.legend(combined_handles, combined_labels, fontsize=10)
        # plt.xlabel('total binary mass [$M_{\odot}$]')
        # plt.ylabel('computational time [s]')
        # plt.grid()

        # figname = 'Total_surrogate_efficiency_q={}_ecc=[{}_{}]_f_min={}.png'.format(self.mass_ratio, min(self.parameter_space_input), max(self.parameter_space_input), self.freqmin)
                    
        # # Ensure the directory exists, creating it if necessary and save
        # os.makedirs('Images/Surrogate_efficiency', exist_ok=True)
        # fig_mass_analysis.savefig('Images/Surrogate_efficiency/' + figname)

        # print('Figure is saved in Images/Surrogate_Efficiency')
"""
GPR CHANGED FOR 0.4!
"""
# analysis = Surrogate_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=30, amount_output_wfs=1000, min_greedy_error_amp=5e-4, min_greedy_error_phase=5e-4, waveform_size=3500, freqmin=18)
# TS11 = analysis1.get_training_set(property='phase', min_greedy_error=1e-3, plot_greedy_error=True, plot_training_set=True)
# TS12 = analysis1.get_training_set(property='amplitude', N_greedy_vecs=20, plot_greedy_error=True, plot_training_set=True)
# print(TS1.shape, TS2.shape)
# analysis.get_surrogate_mismatches(plot_mismatches=True, save_mismatch_fig=True, plot_worst_err=True)

# analysis = Surrogate_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=35, amount_output_wfs=1500, min_greedy_error_amp=1e-3, min_greedy_error_phase=1e-3, waveform_size=3500, freqmin=20)
# analysis = Surrogate_analysis(parameter_space=[0.01, 0.2], amount_input_wfs=35, amount_output_wfs=1000, min_greedy_error_amp=1e-3, min_greedy_error_phase=1e-3, waveform_size=3500, freqmin=20,  =40)
# print(analysis.parameter_space_output[-500:])
# analysis.generate_property_dataset(np.linspace(0.01, 0.2, num=9), 'phase', plot_residuals=True)
# analysis.surrogate_pointwise_error(plot_surr=True, save_pointwise_error_fig=True)
# analysis.get_training_set(property='phase', N_greedy_vecs=500, plot_greedy_error=True, save_greedy_fig=True)
# analysis.get_training_set(property='amplitude', N_greedy_vecs=500, plot_greedy_error=True, save_greedy_fig=True)
# analysis = Surrogate_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=60, amount_output_wfs=1000, min_greedy_error_amp=1e-3, min_greedy_error_phase=1e-3, waveform_size=2200, freqmin=18)
# # print(analysis.parameter_space_output)
# # analysis.generate_property_dataset(np.linspace(0.01, 0.2, num=9), 'phase', plot_residuals=True)
# analysis.surrogate_pointwise_error(plot_surr=True, save_pointwise_error_fig=True)
# analysis.fit_to_training_set('phase', 1e-2, plot_fits=True)
# analysis.fit_to_training_set('amplitude', 1e-2, plot_fits=True)

# analysis = Surrogate_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=60, amount_output_wfs=1500, min_greedy_error_amp=1e-3, min_greedy_error_phase=1e-3, waveform_size=100000, freqmin=20, mass_ratio=1)
# analysis = Surrogate_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=60, amount_output_wfs=1500, min_greedy_error_amp=1e-3, min_greedy_error_phase=1e-3, waveform_size=4000, freqmin=20, mass_ratio=2)
# analysis.generate_property_dataset(np.linspace(0.01, 0.3, 10), 'amplitude', plot_residuals_t=True)
# analysis.generate_property_dataset(np.linspace(0.01, 0.3, 60), 'phase', save_dataset_to_file=True)
# analysis.pointwise_error_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=130, amount_output_wfs=1500, N_greedy_vecs_amp=[120], N_greedy_vecs_phase=[120], save_pointwise_errors_fig=True, save_pointwise_errors_to_file=True)

analysis = Surrogate_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=50, amount_output_wfs=300, min_greedy_error_amp=1e-3, min_greedy_error_phase=1e-3)
# analysis.pointwise_error_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=60, amount_output_wfs=1500, N_greedy_vecs_amp=[60, 55, 50, 45], N_greedy_vecs_phase=[60, 55, 50, 45], save_pointwise_errors_fig=True, save_pointwise_errors_to_file=True)
# analysis.pointwise_error_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=130, amount_output_wfs=1500, N_greedy_vecs_amp=[110, 100], N_greedy_vecs_phase=[100, 105], save_pointwise_errors_fig=True, save_pointwise_errors_to_file=True)
# print(analysis.parameter_space_output)
analysis.pointwise_error_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=50, amount_output_wfs=200, N_greedy_vecs_amp=[35, 40], N_greedy_vecs_phase=[35, 40], save_pointwise_errors_fig=True, save_pointwise_errors_to_file=True)
# -------------------------------------
# analysis.mass_system_analysis(masses=[50, 40, 30, 20, 10], Ninput=[60, 70, 90, 130, 150], N_greedy_vecs_amp=[45, 55, 80, 100, 130], N_greedy_vecs_phase=[45, 55, 70, 105, 125], sizes=[2547, 3569, 5591, 4000, 2000])



# analysis = Surrogate_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=60, amount_output_wfs=1500, min_greedy_error_amp=1e-3, min_greedy_error_phase=1e-3, waveform_size=3569, freqmin=20, mass_ratio=1)
# # analysis.generate_property_dataset(np.linspace(0.01, 0.3, 8), 'phase', plot_residuals_t=True)
# analysis.pointwise_error_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=110, amount_output_wfs=1500, N_greedy_vecs_amp=[90, 90], N_greedy_vecs_phase=[80, 70], save_pointwise_errors_fig=True, save_pointwise_errors_to_file=True)
# analysis.pointwise_error_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=120, amount_output_wfs=1500, N_greedy_vecs_amp=[120, 110], N_greedy_vecs_phase=[110, 100], save_pointwise_errors_fig=True, save_pointwise_errors_to_file=True)
# analysis = Surrogate_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=150, amount_output_wfs=1000, min_greedy_error_amp=1e-3, min_greedy_error_phase=1e-3, waveform_size=10000, freqmin=20,  mass_ratio=2)
# analysis.pointwise_error_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=150, amount_output_wfs=1500, N_greedy_vecs_amp=[150, 130], N_greedy_vecs_phase=[150, 130], save_pointwise_errors_fig=True, save_pointwise_errors_to_file=True)


# analysis = Surrogate_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=150, amount_output_wfs=1000, min_greedy_error_amp=1e-3, min_greedy_error_phase=1e-3, waveform_size=10000, freqmin=20, total_mass=50, mass_ratio=2)
# analysis.pointwise_error_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=60, amount_output_wfs=1500, N_greedy_vecs_amp=[60], N_greedy_vecs_phase=[60], save_pointwise_errors_fig=True, save_pointwise_errors_to_file=True)


# print(analysis.waveform_size)

# analysis.generate_property_dataset(np.linspace(0.01, 0.2, num=9), 'phase', plot_residuals_t=True)
# # analysis.surrogate_pointwise_error(plot_surr=True, save_pointwise_error_fig=True)
# analysis.fit_to_training_set(property='phase', N_greedy_vecs=45, plot_fits=True, save_fits_to_file=False)
# analysis.fit_to_training_set(property='amplitude', N_greedy_vecs=45, plot_fits=True, save_fits_to_file=False)

# analysis.pointwise_error_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=150, amount_output_wfs=1500, N_greedy_vecs_amp=[150, 130], N_greedy_vecs_phase=[150, 130], save_pointwise_errors_fig=True, save_pointwise_errors_to_file=True)
# analysis.pointwise_error_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=150, amount_output_wfs=1500, N_greedy_vecs_amp=[130], N_greedy_vecs_phase=[130], save_pointwise_errors_fig=True, save_pointwise_errors_to_file=True)

# analysis = Surrogate_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=60, amount_output_wfs=1000, min_greedy_error_amp=1e-3, min_greedy_error_phase=1e-3, waveform_size=5000, freqmin=20, total_mass=60, mass_ratio=1)
# analysis.pointwise_error_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=60, amount_output_wfs=1500, N_greedy_vecs_amp=[60, 55, 50, 45, 40, 35], N_greedy_vecs_phase=[60, 50, 40, 30], save_pointwise_errors_fig=True, save_pointwise_errors_to_file=True)
# 60, 55, 50, 45, 40, 35
# for mass in [80]:
#     analysis = Surrogate_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=20, amount_output_wfs=1000, min_greedy_error_amp=1e-3, min_greedy_error_phase=1e-3, waveform_size=5000, freqmin=20, total_mass=mass, mass_ratio=1)
#     # analysis.generate_property_dataset(np.linspace(0.01, 0.3, num=9), 'phase', plot_residuals_t=True)
#     analysis.fit_to_training_set('phase', plot_fits=True)
#     analysis.fit_to_training_set('amplitude', plot_fits=True)
#     plt.show()
    # analysis.pointwise_error_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=60, amount_output_wfs=1500, N_greedy_vecs_amp=[60], N_greedy_vecs_phase=[60], save_pointwise_errors_fig=True, save_pointwise_errors_to_file=True)
# analysis = Surrogate_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=60, amount_output_wfs=1000, min_greedy_error_amp=1e-3, min_greedy_error_phase=1e-3, waveform_size=5000, freqmin=20, total_mass=50, mass_ratio=1)
# analysis.pointwise_error_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=60, amount_output_wfs=1500, N_greedy_vecs_amp=[60, 55, 50, 45, 40, 35], N_greedy_vecs_phase=[60, 55, 50, 45, 40, 35], save_pointwise_errors_fig=True, save_pointwise_errors_to_file=True)

# analysis.pointwise_error_analysis(parameter_space=[0.01, 0.4], amount_input_wfs=150, amount_output_wfs=1500, N_greedy_vecs_amp=[150], N_greedy_vecs_phase=[150], save_pointwise_errors_fig=True, save_pointwise_errors_to_file=True)
# 5e-4, 7e-4, 8e-4, 1e-3, 3e-3, 5e-3, 1e-2, 
# 5e-4, 7e-4, 8e-4, 1e-3, 3e-3, 5e-3, 1e-2, 
# 5e-4, 7e-4, 8e-4, 1e-3, 3e-3, 5e-3, 1e-2, 
# 55, 50, 45, 40, 35
# print(analysis.parameter_space_output)# , 130, 110, 90, 70; , 130, 110, 90, 70
# analysis.pointwise_error_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=60, amount_output_wfs=1500, greedy_errors=[[1e-3]], save_pointwise_errors_fig=True, save_pointwise_errors_to_file=True)
# analysis = Surrogate_analysis(parameter_space=[0.01, 0.2], amount_input_wfs=35, amount_output_wfs=1000, min_greedy_error_amp=1e-3, min_greedy_error_phase=1e-3, waveform_size=3500, freqmin=20, total_mass=40)
# analysis.pointwise_error_analysis(parameter_space=[0.01, 0.2], amount_input_wfs=35, amount_output_wfs=1000, N_greedy_vecs=[30], save_pointwise_errors_fig=True, save_pointwise_errors_to_file=True)
# masses = [50, 40, 30, 20, 10]
# inputs = [60, 60, 60, 80, 80]
# ga = [45, 50, 55, 80, 80]
# gp = [40, 50, 60, 80, 80]
# ws = [3500, 3500, 3500, 2200, 2200]
# for i, (mass, Ni, gamp, gphase, size) in enumerate(zip(masses, inputs, ga, gp, ws)):
#     analysis = Surrogate_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=100, amount_output_wfs=1000, min_greedy_error_amp=1e-3, min_greedy_error_phase=1e-3, waveform_size=size, freqmin=20, total_mass=mass)
#     analysis.pointwise_error_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=Ni, amount_output_wfs=1000, N_greedy_vecs_amp=[gamp], N_greedy_vecs_phase=[gphase], save_pointwise_errors_fig=True, save_pointwise_errors_to_file=True)
# for mass in [20, 10]:
#     analysis = Surrogate_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=100, amount_output_wfs=1000, min_greedy_error_amp=1e-3, min_greedy_error_phase=1e-3, waveform_size=2200, freqmin=20, total_mass=mass)
#     analysis.pointwise_error_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=80, amount_output_wfs=1000, N_greedy_vecs_amp=[80, 80], N_greedy_vecs_phase=[80, 80], save_pointwise_errors_fig=True, save_pointwise_errors_to_file=True)

# 80, 75, 70, 65, 60
# wp1 = Waveform_Properties(eccmin=0.2, freqmin=20)
# wp2 = Waveform_Properties(eccmin=0.2, freqmin=10)
# hp1, hc1, TS1 = wp1.simulate_inspiral_mass_independent()
# hp2, hc2, TS2 = wp2.simulate_inspiral_mass_independent()
# wp1.calculate_residual(hp1, hc1, 'phase', plot_residual=True)
# wp2.calculate_residual(hp2, hc2, 'phase', plot_residual=True)
plt.show()
# TS21 = analysis.get_training_set(property='phase', min_greedy_error=1e-3, plot_greedy_error=True, plot_training_set=True)
# TS22 = analysis.get_training_set(property='amplitude', min_greedy_error=1e-3, plot_greedy_error=True, plot_training_set=True)
# analysis.get_surrogate_mismatches(plot_mismatches=True, save_mismatch_fig=True)
# analysis.fit_to_training_set('phase', 1e-3, plot_fits=True)
# analysis.fit_to_training_set('amplitude', 1e-3, plot_fits=True)
# analysis.generate_surrogate_model(plot_surr_at_ecc=0.1501)
# analysis.generate_surrogate_model(plot_surr_at_ecc=0.026)
# analysis.generate_surrogate_model(plot_surr_at_ecc=0.061)
# analysis.generate_surrogate_model(plot_surr_at_ecc=0.1147)
# analysis.generate_surrogate_model(plot_surr_at_ecc=0.1208)
# analysis.generate_surrogate_model(plot_surr_at_ecc=0.1269)
# analysis.mismatch_analysis([0.01, 0.3], 500, save_mismatches_inputs_fig=True)


