from generate_eob_surrogate import *

from pycbc.filter import match, optimized_match
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.types import TimeSeries
from pycbc import waveform
import time
import traceback
from scipy.fft import rfft, rfftfreq
from pycbc.waveform.utils import taper_timeseries
import seaborn as sns

class Surrogate_analysis(Generate_Surrogate):

    def __init__(self, parameter_space, amount_input_wfs, amount_output_wfs,  N_greedy_vecs_amp=None, N_greedy_vecs_phase=None, min_greedy_error_amp=None, min_greedy_error_phase=None, waveform_size=None, mass_ratio=1, freqmin=650):
        
        self.surrogate_h = None
        self.parameter_space = parameter_space
        self.amount_input_wfs = amount_input_wfs
        self.amount_output_wfs = amount_output_wfs

        Generate_Surrogate.__init__(self, parameter_space, amount_input_wfs, amount_output_wfs,  N_greedy_vecs_amp, N_greedy_vecs_phase, min_greedy_error_amp, min_greedy_error_phase, waveform_size=waveform_size, mass_ratio=mass_ratio, freqmin=freqmin)

    

    def calculate_mismatch(self, h_surrogate, h_real):


        h_surr_TS = types.timeseries.TimeSeries(h_surrogate, delta_t=self.DeltaT)
        h_real_TS = types.timeseries.TimeSeries(h_real, delta_t=self.DeltaT)


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
                print(f'line {getframeinfo(f).lineno}: {e}')
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
                figname = 'Mismatches_q={}_ecc=[{}_{}]_iN={}_oN={}_gp={}_ga={}.png'.format(self.mass_ratio, min(self.parameter_space_input), max(self.parameter_space_input), len(self.parameter_space_input), len(self.parameter_space_output), self.min_greedy_error_phase, self.min_greedy_error_amp)
                        
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

            surrogate = Surrogate_analysis(parameter_space=parameter_space, amount_input_wfs=input_wfs, amount_output_wfs=amount_output_wfs, min_greedy_error_amp=1e-3, min_greedy_error_phase=1e-3, waveform_size=self.waveform_size, freqmin=self.freqmin)
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
                figname = 'Mismatches_inputs_q={}_ecc=[{}_{}]_o={}_gp={}_ga={}.png'.format(self.mass_ratio, min(parameter_space), max(parameter_space), len(self.parameter_space_output), self.min_greedy_error_phase, self.min_greedy_error_amp)
                        
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Mismatches', exist_ok=True)
                fig_input_wfs.savefig('Images/Mismatches/' + figname)

                print('Figure is saved in Images/Mismatches')

    def surrogate_pointwise_error(self, plot_surr=False, save_pointwise_error_fig=True, save_pointwise_errors_to_file=True):
        
        try:
            # Load the .npz file with errors
            data = np.load(f'Straindata/Pointwise_error/Pointwise_error_{self.parameter_space[0]}_{self.parameter_space[1]}_Ni={self.amount_input_wfs}_No={self.amount_output_wfs}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}.npz')
            worst_err_amp = data['worst_err_amp']
            worst_err_phase = data['worst_err_phase']
            best_err_amp = data['best_err_amp']
            best_err_phase = data['best_err_phase']
            mean_err_amp = data['mean_err_amp']
            mean_err_phase = data['mean_err_phase']
            generation_time_surr = data['generation_t_surr']
            generation_time_full_wfs = data['generation_t_full_wfs']
            print('Pointwise errors loaded')
            
        except Exception as e:
            print(f'line {getframeinfo(f).lineno}: {e}')

            surrogate_h, surrogate_amp, surrogate_phase, generation_time_surr = self.generate_surrogate_model(save_surr_to_file=True)

            try:
                load_polarisations = np.load(f'true_polarisations_[{min(self.parameter_space_output)}_{max(self.parameter_space_output)}]_N={len(self.parameter_space_output)}.npz', allow_pickle=True)
                true_hp = load_polarisations['hp']
                true_hc = load_polarisations['hc']
                TS = load_polarisations['TS'][-self.waveform_size:]
                generation_time_full_wfs = load_polarisations['generation_t']

            except Exception as e:
                print(f'line {getframeinfo(f).lineno}: {e}')
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


                np.savez(f'true_polarisations_[{min(self.parameter_space_output)}_{max(self.parameter_space_output)}]_N={len(self.parameter_space_output)}.npz', hp=true_hp, hc=true_hc, TS=TS, generation_t=generation_time_full_wfs)
                print('True polarisations saved')

            worst_err_amp = np.zeros(len(self.parameter_space_output))
            best_err_amp = np.zeros(len(self.parameter_space_output))
            mean_err_amp = np.zeros(len(self.parameter_space_output))

            worst_err_phase = np.zeros(len(self.parameter_space_output))
            best_err_phase = np.zeros(len(self.parameter_space_output))
            mean_err_phase = np.zeros(len(self.parameter_space_output))

            for i in range(len(self.parameter_space_output)):
                true_amp = np.array(waveform.utils.amplitude_from_polarizations(true_hp[i], true_hc[i]))[-self.waveform_size:]
                true_phase = np.array(waveform.utils.phase_from_polarizations(true_hp[i], true_hc[i]))[-self.waveform_size:]

                pointwise_error_phase = abs(surrogate_phase.T[i] - true_phase) / abs(true_phase)
                pointwise_error_amp = abs(surrogate_amp.T[i] - true_amp) / abs(true_amp)

                best_err_amp[i], worst_err_amp[i], mean_err_amp[i] = min(pointwise_error_amp), max(pointwise_error_amp), np.mean(pointwise_error_amp)
                best_err_phase[i], worst_err_phase[i], mean_err_phase[i] = min(pointwise_error_phase), max(pointwise_error_phase), np.mean(pointwise_error_phase)

            if save_pointwise_errors_to_file is True and not os.path.isfile(f'Straindata/Pointwise_error/Pointwise_error_{self.parameter_space[0]}_{self.parameter_space[1]}_Ni={self.amount_input_wfs}_No={self.amount_output_wfs}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}.npz'):
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Straindata/Pointwise_error', exist_ok=True)
                np.savez(f'Straindata/Pointwise_error/Pointwise_error_{self.parameter_space[0]}_{self.parameter_space[1]}_Ni={self.amount_input_wfs}_No={self.amount_output_wfs}_gp={self.min_greedy_error_phase}_ga={self.min_greedy_error_amp}.npz', best_err_amp=best_err_amp, mean_err_amp=mean_err_amp, worst_err_amp=worst_err_amp, best_err_phase=best_err_phase, mean_err_phase=mean_err_phase, worst_err_phase=worst_err_phase, generation_t_surr=generation_time_surr, generation_t_full_wfs=generation_time_full_wfs)
                print('Pointwise errors saved to Straindata/Pointwise_error')

        fig_pointwise_err, axs = plt.subplots(2, sharex=True)
        
        axs[0].plot(self.parameter_space_output, best_err_phase, linewidth=0.6, label=f'best error', color='green')
        axs[0].plot(self.parameter_space_output, worst_err_phase, linewidth=0.6, label=f'worst error', color='red')
        axs[0].plot(self.parameter_space_output, mean_err_phase, linewidth=0.6, label=f'mean error', color='blue')

        axs[0].set_ylabel('|($\phi_S$ - $\phi$) / $\phi|')
        axs[0].grid(True)
        axs[0].legend(fontsize='small')

        axs[1].plot(self.parameter_space_output, best_err_amp, linewidth=0.6, label=f'best error', color='green')
        axs[1].plot(self.parameter_space_output, worst_err_amp, linewidth=0.6, label=f'worst error', color='red')
        axs[1].plot(self.parameter_space_output, mean_err_amp, linewidth=0.6, label=f'mean error', color='blue')
        
        axs[1].set_xlabel('eccentricity')
        axs[1].set_ylabel('|($A_S$ - A) / A|')
        axs[1].grid(True)
        axs[1].legend(fontsize='small')
        axs[1].set_title(f'greedy error = {self.min_greedy_error_amp}')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        if save_pointwise_error_fig is True:
            figname_pointwise = 'Pointwise_errors_q={}_ecc=[{}_{}]_fmin={}_iN={}_oN={}_gp={}_ga={}.png'.format(self.mass_ratio, min(self.parameter_space), max(self.parameter_space), self.freqmin, self.amount_input_wfs, self.amount_output_wfs, self.min_greedy_error_phase, self.min_greedy_error_amp)
            
            # Ensure the directory exists, creating it if necessary and save
            os.makedirs('Images/Pointwise_error', exist_ok=True)
            fig_pointwise_err.savefig('Images/Pointwise_error/' + figname_pointwise)

            print('Figures are saved in Images/Pointwise_error')
        
        if plot_surr is True:
            ecc_worst_phase = self.parameter_space_output[np.argmax(worst_err_phase)]
            for i in range(len(worst_err_phase)):
                if worst_err_phase[i] > np.mean(worst_err_phase)*10:
                    worst_err_phase[i] = 0
            for i in range(len(worst_err_phase) - 1):
                if np.diff(worst_err_phase)[i] > 2:
                    print(worst_err_phase[i], np.diff(worst_err_phase)[i])

            ecc_second_worst_phase = self.parameter_space_output[np.argmax(worst_err_phase)]
            # ecc_best_phase = self.parameter_space_output[np.argmin(worst_err_phase)]

            print('worst err phase', worst_err_phase)
            # print(self.parameter_space_output)
            # self.generate_surrogate_model(plot_surr_at_ecc=worst_err_phase, plot_surr_datapiece_at_ecc=worst_err_phase, plot_GPRfit=True, save_fig_datapiece=True, save_fig_surr=True, save_fig_fits=True)
            # self.generate_surrogate_model(plot_surr_at_ecc=0.2754, plot_surr_datapiece_at_ecc=0.2752, plot_GPRfit=True, save_fig_datapiece=True, save_fig_surr=True, save_fig_fits=True)
            # self.generate_surrogate_model(plot_surr_at_ecc=0.3, plot_surr_datapiece_at_ecc=0.3, plot_GPRfit=True, save_fig_datapiece=True, save_fig_surr=True, save_fig_fits=True)
            # self.generate_surrogate_model(plot_surr_at_ecc=ecc_best_phase, plot_surr_datapiece_at_ecc=ecc_best_phase, plot_GPRfit=True, save_fig_datapiece=True, save_fig_surr=True)

        return best_err_amp, mean_err_amp, worst_err_amp, best_err_phase, mean_err_phase, worst_err_phase, generation_time_surr, generation_time_full_wfs
    
    def pointwise_error_analysis(self, parameter_space, amount_input_wfs, amount_output_wfs, greedy_errors, save_pointwise_errors_fig=True, save_pointwise_errors_to_file=True):

        worst_of_worst_errs_phase, best_of_worst_errs_phase, mean_of_worst_errs_phase = np.zeros(len(greedy_errors)), np.zeros(len(greedy_errors)), np.zeros(len(greedy_errors))
        worst_of_worst_errs_amp, best_of_worst_errs_amp, mean_of_worst_errs_amp = np.zeros(len(greedy_errors)), np.zeros(len(greedy_errors)), np.zeros(len(greedy_errors))
        generation_times_surr = np.zeros(len(greedy_errors))

        def delete_existing_file(file_path):
            # Check if the file exists before attempting to delete
            if os.path.exists(file_path) and os.path.isfile(file_path):
                os.remove(file_path)
                print(f"File '{file_path}' has been deleted.")
            else:
                print(f"File '{file_path}' does not exist.")

        # delete_existing_file(f'true_polarisations_[{parameter_space[0]}_{parameter_space[1]}]_N={amount_output_wfs}.npz')

        for i, greedy_err in enumerate(greedy_errors):
            print(f'Calculate pointwise error for: greedy_error = {greedy_err}')

            # Delete saved polarisations if exists
            file_path_polarisations = f'Straindata/Polarisations/polarisations_e=[{parameter_space[0]}_{parameter_space[1]}]_N={amount_input_wfs}.npz'
            delete_existing_file(file_path_polarisations)

            surrogate = Surrogate_analysis(parameter_space=parameter_space, amount_input_wfs=amount_input_wfs, amount_output_wfs=amount_output_wfs, min_greedy_error_amp=greedy_err, min_greedy_error_phase=greedy_err, waveform_size=self.waveform_size, freqmin=self.freqmin)
            best_errs_amp, mean_errs_amp, worst_errs_amp, best_errs_phase, mean_errs_phase, worst_errs_phase, generation_times_surr[i], generation_t_full_wfs = surrogate.surrogate_pointwise_error(plot_surr=True, save_pointwise_error_fig=True)

                
            # Get best, mean and worst value of the WORST approximated points in eccentric waveforms.
            for j in range(len(worst_errs_phase)):
                if worst_errs_phase[j] > np.mean(worst_errs_phase)*10:
                    print(worst_errs_phase[j])

            worst_of_worst_errs_phase[i], best_of_worst_errs_phase[i], mean_of_worst_errs_phase[i] = max(worst_errs_phase), min(worst_errs_phase), np.mean(worst_errs_phase)
            worst_of_worst_errs_amp[i], best_of_worst_errs_amp[i], mean_of_worst_errs_amp[i] = max(worst_errs_amp), min(worst_errs_amp), np.mean(worst_errs_amp)
            
          
        fig_pointwise_errs = plt.figure()
        for i in range(len(greedy_errors)):
            plt.scatter(greedy_errors[i], worst_of_worst_errs_phase[i], marker='s', color='red')
            plt.scatter(greedy_errors[i], best_of_worst_errs_phase[i], marker='s', color='green')
            plt.scatter(greedy_errors[i], mean_of_worst_errs_phase[i], marker='s', color='blue')
            plt.scatter(greedy_errors[i], worst_of_worst_errs_amp[i], marker='^', color='red')
            plt.scatter(greedy_errors[i], best_of_worst_errs_amp[i], marker='^', color='green')
            plt.scatter(greedy_errors[i], mean_of_worst_errs_amp[i], marker='^', color='blue')

        plt.xlabel('minimum greedy error')
        plt.ylabel('worst pointwise error')
        plt.grid()
        plt.legend()

        colors = sns.color_palette("tab20")

        fig_generation_time = plt.figure()
        for i in range(len(greedy_errors)):
            plt.scatter(mean_of_worst_errs_phase[i], generation_times_surr[i], marker='s', color=colors[i], label=f'greedy error = {greedy_errors[i]}')
            plt.scatter(mean_of_worst_errs_amp[i], generation_times_surr[i], marker='^', color=colors[i])
        plt.plot(np.linspace(0, max(mean_of_worst_errs_amp + mean_of_worst_errs_phase)*1.2, num=1000), np.full(1000, generation_t_full_wfs), linestyle='dashed', color='black', lw=0.6, label='full waveform set')
        plt.ylabel('surrogate computational time')
        plt.xlabel('mean of worst pointwise error')
        plt.grid()
        plt.legend(loc='upper right')


        if save_pointwise_errors_fig is True:
                figname_pointwise = 'Pointwise_errors_analysis_q={}_ecc=[{}_{}]_iN={}_oN={}_fmin={}.png'.format(self.mass_ratio, min(parameter_space), max(parameter_space), amount_input_wfs, amount_output_wfs, self.freqmin)
                figname_compu_cost = 'Computational_cost_q={}_ecc=[{}_{}]_iN={}_oN={}_fmin={}.png'.format(self.mass_ratio, min(parameter_space), max(parameter_space), amount_input_wfs, amount_output_wfs, self.freqmin)
                        
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Pointwise_error', exist_ok=True)
                os.makedirs('Images/Computational_efficiency', exist_ok=True)
                fig_pointwise_errs.savefig('Images/Pointwise_error/' + figname_pointwise)
                fig_generation_time.savefig('Images/Computational_efficiency/' + figname_compu_cost)

                print('Figures are saved in Images/Pointwise_error and Images/Computational_efficiency')


# analysis = Surrogate_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=30, amount_output_wfs=1000, min_greedy_error_amp=5e-4, min_greedy_error_phase=5e-4, waveform_size=3500, freqmin=18)
# TS11 = analysis1.get_training_set(property='phase', min_greedy_error=1e-3, plot_greedy_error=True, plot_training_set=True)
# TS12 = analysis1.get_training_set(property='amplitude', N_greedy_vecs=20, plot_greedy_error=True, plot_training_set=True)
# print(TS1.shape, TS2.shape)
# analysis.get_surrogate_mismatches(plot_mismatches=True, save_mismatch_fig=True, plot_worst_err=True)

# analysis = Surrogate_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=35, amount_output_wfs=1500, min_greedy_error_amp=1e-3, min_greedy_error_phase=1e-3, waveform_size=3500, freqmin=20)
analysis = Surrogate_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=60, amount_output_wfs=300, N_greedy_vecs_amp=50, N_greedy_vecs_phase=40, freqmin=650)
# print(analysis.parameter_space_output[-500:])
# analysis.generate_property_dataset(np.linspace(0.01, 0.2, num=9), 'phase', plot_residuals=True)
# analysis.surrogate_pointwise_error(plot_surr=True, save_pointwise_error_fig=True)
# analysis.get_training_set(property='phase', N_greedy_vecs=500, plot_greedy_error=True, save_greedy_fig=True)
# analysis.get_training_set(property='amplitude', N_greedy_vecs=500, plot_greedy_error=True, save_greedy_fig=True)
# analysis = Surrogate_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=60, amount_output_wfs=1000, min_greedy_error_amp=1e-3, min_greedy_error_phase=1e-3, waveform_size=2200, freqmin=18)
# # print(analysis.parameter_space_output)
# # analysis.generate_property_dataset(np.linspace(0.01, 0.2, num=9), 'phase', plot_residuals=True)
analysis.surrogate_pointwise_error(plot_surr=True, save_pointwise_error_fig=True)
# analysis.fit_to_training_set('phase', 1e-2, plot_fits=True)
# analysis.fit_to_training_set('amplitude', 1e-2, plot_fits=True)

# analysis = Surrogate_analysis(parameter_space=[0.01, 0.2], amount_input_wfs=35, amount_output_wfs=1000, min_greedy_error_amp=1e-3, min_greedy_error_phase=1e-3, waveform_size=2200, freqmin=18)
# # print(analysis.parameter_space_output)
# # analysis.generate_property_dataset(np.linspace(0.01, 0.2, num=9), 'phase', plot_residuals=True)
# # analysis.surrogate_pointwise_error(plot_surr=True, save_pointwise_error_fig=True)
# analysis.fit_to_training_set('phase', 5e-5, plot_fits=True, save_fits_to_file=False)
# analysis.fit_to_training_set('amplitude', 5e-5, plot_fits=True, save_fits_to_file=False)

# analysis.pointwise_error_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=60, amount_output_wfs=1500, greedy_errors=[8e-4, 1e-3, 3e-3], save_pointwise_errors_fig=True, save_pointwise_errors_to_file=True)
# 5e-4, 7e-4, 8e-4, 1e-3, 3e-3, 5e-3, 1e-2
# print(analysis.parameter_space_output)
# analysis.pointwise_error_analysis(parameter_space=[0.01, 0.3], amount_input_wfs=60, amount_output_wfs=300, greedy_errors=[[1e-3]], save_pointwise_errors_fig=True, save_pointwise_errors_to_file=True)

# analysis.pointwise_error_analysis(parameter_space=[0.2, 0.4], amount_input_wfs=60, amount_output_wfs=1000, greedy_errors=[5e-4, 8e-4, 1e-3, 3e-3], save_pointwise_errors_fig=True, save_pointwise_errors_to_file=True)
# wp1 = Waveform_Properties(eccmin=0.2, freqmin=20)
# wp2 = Waveform_Properties(eccmin=0.2, freqmin=10)
# hp1, hc1, TS1 = wp1.simulate_inspiral_mass_independent()
# hp2, hc2, TS2 = wp2.simulate_inspiral_mass_independent()
# wp1.calculate_residual(hp1, hc1, 'phase', plot_residual=True)
# wp2.calculate_residual(hp2, hc2, 'phase', plot_residual=True)
# plt.show()
# TS21 = analysis.get_training_set(property='phase', min_greedy_error=1e-3, plot_greedy_error=True, plot_training_set=True)
# TS22 = analysis.get_training_set(property='amplitude', min_greedy_error=1e-3, plot_greedy_error=True, plot_training_set=True)
analysis.get_surrogate_mismatches(plot_mismatches=True, save_mismatch_fig=True)
# analysis.fit_to_training_set('phase', 1e-3, plot_fits=True)
# analysis.fit_to_training_set('amplitude', 1e-3, plot_fits=True)
# analysis.generate_surrogate_model(plot_surr_at_ecc=0.1501)
# analysis.generate_surrogate_model(plot_surr_at_ecc=0.026)
# analysis.generate_surrogate_model(plot_surr_at_ecc=0.061)
# analysis.generate_surrogate_model(plot_surr_at_ecc=0.1147)
# analysis.generate_surrogate_model(plot_surr_at_ecc=0.1208)
# analysis.generate_surrogate_model(plot_surr_at_ecc=0.1269)
# analysis.mismatch_analysis([0.01, 0.3], 500, save_mismatches_inputs_fig=True)
plt.show()

