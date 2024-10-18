import lalsimulation as lalsim
import lal
import matplotlib.pyplot as plt
from numba import jit, cuda
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from scipy.integrate import simps
from pycbc import types
from pycbc.types import timeseries
from pycbc import waveform
import sys

plt.switch_backend('WebAgg')

class Simulate_Inspiral:
    """ Simulates Inspiral phase of a binary blackhole merger. 
    Optional: Simulate either mass dependent or mass independent. ; Simulate the frequency and phase of the waveform """
    
    def __init__(self, eccmin, total_mass=50, mass_ratio=1, freqmin=18, waveform_size=None):
        
        self.total_mass = total_mass # Total mass of the BBH system [Solar Mass]
        self.mass_ratio = mass_ratio # Mass ratio 0 < q < 1, so M_1 > M_2
        self.eccmin = eccmin # eccentricity of binary at start frequency
        self.freqmin = freqmin # Start frequency [Hz]
        self.DeltaT = 1./2048. # 
        self.lalDict = lal.CreateDict() # 

        self.waveform_size = waveform_size

        self.hp_TS = None
        self.hc_TS = None
        TS = None

        self.hp_TS_M = None
        self.hc_TS_M = None
        self.TS_M = None


    def sim_inspiral(self, eccmin=None):
        if eccmin is None:
            eccmin = self.eccmin

        start = timer()

        mass1 = self.total_mass / (1 + self.mass_ratio)
        mass2 = self.total_mass - mass1

        hp, hc = lalsim.SimInspiralTD(
            m1=lal.MSUN_SI*mass1, m2=lal.MSUN_SI*mass2,
            S1x=0., S1y=0., S1z=0., S2x=0., S2y=0., S2z=0.,
            distance=400.*1e6*lal.PC_SI, inclination=0.,
            phiRef=0., longAscNodes=0, eccentricity=eccmin, meanPerAno=0.,
            deltaT=self.DeltaT, f_min=self.freqmin, f_ref=self.freqmin,
            LALparams=self.lalDict, approximant=lalsim.EccentricTD
        )

        hp_TS = types.timeseries.TimeSeries(hp.data.data, delta_t=hp.deltaT)  # plus polarisation timeseries
        hc_TS = types.timeseries.TimeSeries(hc.data.data, delta_t=hc.deltaT)  # cross polarisation timeseries
        TS = -hp_TS.sample_times[::-1] # Timeseries 
        
        print('time : SimInspiral_M_independent ecc = {}, M = {}, q={}, freqmin={}'.format(eccmin, self.total_mass, self.mass_ratio, self.freqmin), (timer()-start)/60, ' minutes')

        return hp_TS, hc_TS, TS
        
    # @jit(target_backend='cuda')
    def sim_inspiral_mass_indp(self, eccmin=None):

        if eccmin is None:
            eccmin = self.eccmin

        hp_TS, hc_TS, TS = self.sim_inspiral(eccmin)
        # length_diff = len(hp_TS) - self.waveform_size
        # hp_TS_M = hp_TS / self.total_mass 
        # hc_TS_M = hc_TS / self.total_mass

        hp_TS_M = hp_TS 
        hc_TS_M = hc_TS 

        TS_M = TS / (lal.MTSUN_SI * self.total_mass) 

        # amp = np.array(waveform.utils.amplitude_from_polarizations(hp_TS_M, hc_TS_M))
        # phase = np.array(waveform.utils.phase_from_polarizations(hp_TS_M, hc_TS_M))

        # test_fig, axs = plt.subplots(2, figsize=(10, 6))
        # axs[0].plot(TS_M, amp, label=str(eccmin) + 'amp')
        # axs[1].plot(TS_M, phase, label=str(eccmin) + 'phase')

        # print('first plot')

        # axs[0].legend()
        # axs[1].legend()
        # plt.title('amp and phase')
        # plt.show()
        phase = waveform.utils.phase_from_polarizations(hp_TS, hc_TS)
        amp = waveform.utils.amplitude_from_polarizations(hp_TS, hc_TS)
        # np.savez('Frequency.npz', ecc=self.eccmin, frequency=freq, t=TS_M)
        # np.savez('Amplitude.npz', ecc=self.eccmin, amplitude=amp, t=TS_M)

        return hp_TS, hc_TS, TS_M
        
    def plot_sim_inspiral_mass_indp(self, polarisation = 'plus', eccmin=None):

        if eccmin is None:
            eccmin = self.eccmin

        hp_TS_M, hc_TS_M, TS_M = self.sim_inspiral_mass_indp(eccmin)

        if self.waveform_size is None:
            self.waveform_size = len(TS_M)
        
        length_diff = len(TS_M) - self.waveform_size

        if polarisation == 'plus':
            h = hp_TS_M[length_diff:]
        elif polarisation == 'cross':
            h = hc_TS_M[length_diff:]
        else:
            print('Choose polarisation = "plus" or cross"')
            sys.exit(1)

        # print(hp_TS_M.shape, hc_TS_M.shape, type(hp_TS_M))
        # h = np.array(hp_TS_M)[length_diff:] + 1j*np.array(hc_TS_M)[length_diff:]
        # print(h.shape, type(h))
        fig_sim_inspiral = plt.figure()

        plt.plot(TS_M[length_diff:], h, label = 'Real: M = {} $(M_\odot)$, q = {}, e = {}'.format(self.total_mass, self.mass_ratio, eccmin), linewidth=0.6)
        # plt.plot(TS_M[length_diff:], np.imag(h), label = 'Imag: M = {} $(M_\odot)$, q = {}, e = {}'.format(self.total_mass, self.mass_ratio, self.eccmin), linewidth=0.6)
        # plt.xlim([-50e3, 5e2])
        # plt.ylim(-2e-23, 2e-23)
        plt.legend(loc = 'upper left')
        plt.xlabel('t/M')
        plt.ylabel('h$_{+}$')
        # plt.title('Waveform in units of mass')
        plt.grid()

        print('M_total = {}; q = {}; eccmin = {}; Strain is calculated'.format(self.total_mass, self.mass_ratio, self.eccmin))
        
        # plt.show()

        figname = 'total mass = {}, mass ratio = {}, ecc = {}.png'.format(self.total_mass, self.mass_ratio, self.eccmin)
        # fig_sim_inspiral.savefig('Images/' + figname)
        # print('fig is saved')

        return h

    def plot_sim_inspiral_mass_indp_multiple(self, M_total, mass_ratio, eccmin, waveform_size):
        """ Input: M_total: A list of total masses in solar mass, 
        mass_ratio: A list of mass ratio's for 0 <= q <= 1, 
        eccmin: A list of eccentricities for 0 <= e <= 1
        """

        fig_plot_multiple = plt.figure(figsize=(8,4))


        for total_mass in M_total:
            for ratio in mass_ratio:
                for eccentricity in eccmin:

                    start = timer()

                    self.eccmin = eccentricity
                    self.mass_ratio = ratio
                    self.total_mass = total_mass

                    hp_TS_M, hc_TS_M, TS_M = self.sim_inspiral_mass_indp()

                    if waveform_size is None:
                        waveform_size = len(TS_M)
                                            
                    length_diff = len(TS_M) - waveform_size

                    # plt.plot(TS_M[length_diff:], hp_TS_M[length_diff:], label = 'Real: M = {} $(M_\odot)$, q = {}, e = {}'.format(total_mass, ratio, eccentricity), linewidth=0.6)
                    plt.plot(TS_M[length_diff:], hc_TS_M[length_diff:], label = 'Imag: M = {} $(M_\odot)$, q = {}, e = {}'.format(total_mass, ratio, eccentricity), linewidth=0.6)
                    plt.xlim([-10e3, 5e2])
                    # plt.ylim(-2e-23, 2e-23)
                    plt.legend(loc = 'upper left')
                    plt.xlabel('t/M')
                    plt.ylabel('h$_{+}$')
                    # plt.title('Waveform in units of mass')
                    plt.grid()


                    print("time GPU:", (timer()-start)/60, ' minutes')
                    print('M_total = {}; q = {}; eccmin = {}; Strain is calculated'.format(total_mass, ratio, eccentricity))
                
        figname = 'total mass = {}, mass ratio = {}, ecc = {}.png'.format(M_total, mass_ratio, eccmin)
        # fig_plot_multiple.savefig('Images/' + figname)
        # print('fig is saved')


    # def phase_amplitude_diff(self, total_mass, mass_ratio, eccmin, freqmin):
    #     hp_TS_over_M, hc_TS_over_M = self.sim_inspiral_mass_independent(total_mass, mass_ratio, eccmin, freqmin)

    #     phase_amp_diff = np.diff(hp_TS_over_M.phase)  # Example calculation for phase amplitude difference

    #     fig_change, axs = plt.subplots(2, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [5, 1]})
    #     plt.subplots_adjust(hspace=0.4)

    #     axs[0][0].plot(hp_TS_over_M.sample_times, phase_amp_diff, linewidth=0.6)
    #     axs[0][0].set_ylabel('$\Delta\phi_{22}$ [radians]')
    #     axs[0][0].set_xlabel('t [M]')

    #     axs[0][1].plot(hp_TS_over_M.sample_times, phase_amp_diff, linewidth=0.6)
    #     axs[0][1].set_xlim(-10000, -2000)
    #     axs[0][1].set_ylim(-300, -280)
    #     axs[0][1].set_xlabel('t [M]')

    #     axs[1][0].plot(hp_TS_over_M.sample_times, phase_amp_diff, linewidth=0.6)
    #     axs[1][0].set_ylabel('$\Delta A_{22}$')
    #     axs[1][0].set_xlabel('t [M]')

    #     axs[1][1].plot(hp_TS_over_M.sample_times, phase_amp_diff, linewidth=0.6)
    #     axs[1][1].set_xlim(-10000, -2000)
    #     axs[1][1].set_xlabel('t [M]')

    #     plt.show()

# simulator = Simulate_Inspiral(0.5)
# simulator.plot_sim_inspiral_mass_indp(waveform_size=5000)

# # Test the Sim_inspiral method
# hp_TS, hc_TS = simulator.sim_inspiral()
# print("hp_TS:", hp_TS)
# print("hc_TS:", hc_TS)

# # Test the sim_inspiral_mass_independent method
# hp_TS_M, hc_TS_M, TS_M = simulator.sim_inspiral_mass_independent()
# print("hp_TS_M:", hp_TS_M)
# print("hc_TS_M:", hc_TS_M)
# print("TS_M:", TS_M)

# # Test the plot_sim_inspiral_mass_independent method

# simulator.plot_sim_inspiral_mass_indp_multiple([20, 30], [1], [0.4])
# simulator.plot_sim_inspiral_mass_indp_multiple([10], [1, 2], [0.4])
# simulator.plot_sim_inspiral_mass_indp_multiple([10], [1], [0.1, 0.4])
# plt.show()

class Waveform_properties(Simulate_Inspiral):

    def __init__(self, eccmin, total_mass=50, mass_ratio=1, freqmin=5, waveform_size=None):
        self.freq = None
        self.amp = None
        self.phase = None
 
        self.TS_M_circ = None
        self.hp_TS_circ = None
        self.hc_TS_circ = None
        self.phase_shift = None

        Simulate_Inspiral.__init__(self, eccmin, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin, waveform_size=waveform_size)
        
    def circulair_wf(self):
        self.hp_TS_circ, self.hc_TS_circ, self.TS_M_circ = self.sim_inspiral_mass_indp(eccmin=1e-5)

    def residual_amp(self, eccmin=None):

        if self.hp_TS_circ is None:
            self.circulair_wf()

        if self.hp_TS_M is None or eccmin != None:
            self.hp_TS_M, self.hc_TS_M, self.TS_M = self.sim_inspiral_mass_indp(eccmin)

        amp_circ = np.array(waveform.utils.amplitude_from_polarizations(self.hp_TS_circ, self.hc_TS_circ))
        amp = np.array(waveform.utils.amplitude_from_polarizations(self.hp_TS_M, self.hc_TS_M))
        
        res_amp = amp - amp_circ[len(amp_circ) - len(amp):]
        return amp, amp_circ, res_amp, self.TS_M, self.TS_M_circ

    def residual_freq(self, eccmin=None):

        if self.hp_TS_circ is None:
            self.circulair_wf()

        if self.hp_TS_M is None:
            self.hp_TS_M, self.hc_TS_M, self.TS_M = self.sim_inspiral_mass_indp(eccmin)
        
        freq_circ = waveform.utils.frequency_from_polarizations(self.hp_TS_circ, self.hc_TS_circ)
        freq = waveform.utils.frequency_from_polarizations(self.hp_TS_M, self.hc_TS_M)
        
        # Adjust TS and TS_circ because frequency has one datapoint less
        TS_M = -freq.sample_times[::-1] / (lal.MTSUN_SI * self.total_mass )
        TS_M_circ = -freq_circ.sample_times[::-1] / (lal.MTSUN_SI * self.total_mass )
        
        freq_circ, freq = np.array(freq_circ), np.array(freq)
        res_freq = freq - freq_circ[len(freq_circ) - len(freq):]

        return freq, freq_circ, res_freq, TS_M, TS_M_circ
    
    def residual_phase(self, eccmin=None):
        if self.hp_TS_circ is None:
            self.circulair_wf()

        if self.hp_TS_M is None or eccmin != None:
            self.hp_TS_M, self.hc_TS_M, self.TS_M = self.sim_inspiral_mass_indp(eccmin)
        
        phase_circ = np.array(waveform.utils.phase_from_polarizations(self.hp_TS_circ, self.hc_TS_circ))
        phase = np.array(waveform.utils.phase_from_polarizations(self.hp_TS_M, self.hc_TS_M))

        # Set phase_circ to 0 at start of residual
        # self.phase_shift = phase_circ[len(phase_circ) - len(phase)]
        # phase_circ = phase_circ - self.phase_shift
        # Subtract the circular phase
        res_phase = phase_circ[len(phase_circ) - len(phase):] - phase

        return phase, phase_circ, res_phase, self.TS_M, self.TS_M_circ

    def plot_residuals(self, property='Frequency'):
        
        # if property == 'Phase':
        #     params = [0.0111, 0.0797, 0.098, 0.0119, 0.1844, 0.1707, 0.0595, 0.1943, 0.2, 0.1634, 0.1421, 0.1273, 0.1779, 0.1101, 0.1505, 0.157, 0.1897, 0.0408, 0.1345, 0.1181, 0.0896, 0.0702, 0.1973, 0.1741]
        # elif property == 'Amplitude':
        #     params = [0.0268, 0.1657, 0.192, 0.2, 0.1802, 0.0652, 0.173, 0.1558, 0.1478, 0.1863, 0.1063, 0.1322, 0.1966, 0.1166, 0.1402, 0.1608, 0.1242, 0.0115, 0.0949, 0.1768, 0.0839, 0.1893, 0.1516, 0.0507, 0.1692]
        
        # plt.figure(figsize=(8, 5))


        if property == 'Frequency':
            prop, prop_circ, res_prop, TS_M, TS_M_circ = self.residual_freq(self.eccmin)
            units = '[Hz]'
        elif property == 'Amplitude':
            prop, prop_circ, res_prop, TS_M, TS_M_circ = self.residual_amp(self.eccmin)
            units = ''
        elif property == 'Phase':
            prop, prop_circ, res_prop, TS_M, TS_M_circ = self.residual_phase(self.eccmin)
            units = '[radians]'
        else:
            print('Choose property = "Frequency", "Amplitude" or "Phase"')
            sys.exit(1)

        # plt.plot(TS_M, prop, label= property, linewidth=0.6)
        # plt.plot(TS_M_circ, prop_circ, label='Adjusted circular ' + property, linewidth=0.6)

        plt.plot(TS_M, res_prop, label='Residual ' + property, linewidth=0.6)
        # plt.xlim(-12000, 0)
        # plt.ylim(-30, 30)
        plt.xlabel('t [M]')
        plt.ylabel(property + ' ' + units)
        plt.legend()
        plt.title('residuals')
        plt.grid(True)
        
        # plt.show()


    def plot_constructed_waveform(self, waveform_size):

        hp, hc, TS_M, phase, amp = self.sim_inspiral_mass_indp(self.eccmin)

        waveform = amp * np.exp(-1j * phase)
        # print(np.imag(waveform).shape)

        # plt.figure(figsize=(8, 5))
        # plt.title('constructed waveform')
        # plt.plot(TS_M, waveform, linewidth = 0.6, label='constructed imag')
        # plt.plot(TS_M, hp, linewidth = 0.6, label='constructed real')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        






class Dataset(Waveform_properties, Simulate_Inspiral):

    # def __init__(self, eccmin_list, waveform_size = None, total_mass=10, mass_ratio=1, freqmin=5):
        
    #     self.eccmin_list = eccmin_list
    #     self.waveform_size = waveform_size

    #     Simulate_Inspiral.__init__(self, eccmin=None, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin)
    #     Waveform_properties.__init__(self, eccmin=None, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin)
    #     print('DS: ', waveform_size)
    #     print(f"DS: Initialized Dataset with waveform_size={self.waveform_size}")

    def __init__(self, eccmin_list, waveform_size=None, total_mass=50, mass_ratio=1, freqmin=18):
        self.eccmin_list = eccmin_list
        self.waveform_size = waveform_size

        Simulate_Inspiral.__init__(self, eccmin=None, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin, waveform_size=waveform_size)
        Waveform_properties.__init__(self, eccmin=None, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin, waveform_size=waveform_size)
    
    
    def generate_dataset_polarisations(self, save_dataset=False, eccmin_list=None, training_set=False):
        # For original waveform size set waveform_size = None
        if eccmin_list is None:
            eccmin_list = self.eccmin_list


        hp_dataset = []
        hc_dataset = []
        hp_DS_cut = np.zeros((len(self.eccmin_list), self.waveform_size))
        hc_DS_cut = np.zeros((len(self.eccmin_list), self.waveform_size))

        for i, eccentricity in enumerate(eccmin_list):
            hp_TS, hc_TS, self.TS_M = self.sim_inspiral_mass_indp(eccentricity)
            

            if self.waveform_size is None:
                waveform_size = len(self.TS_M)
            else:
                waveform_size = self.waveform_size

            length_diff = len(self.TS_M) - waveform_size

            hp_dataset.append(hp_TS)
            hc_dataset.append(hc_TS)

            hp_DS_cut[i] = hp_TS[length_diff:]
            hc_DS_cut[i] = hc_TS[length_diff:]

            # self.TS_M = self.TS_M[length_diff:]
        hp_dataset = np.array(hp_dataset, dtype=object)
        hc_dataset = np.array(hc_dataset, dtype=object)

        if save_dataset is True:

            header = str(self.eccmin_list)

            if training_set is True:
                np.savez(f'Straindata/Training_Hp_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz', hp_dataset=hp_dataset, eccentricities=eccmin_list)
                np.savez(f'Straindata/Training_Hc_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz', hc_dataset=hc_dataset, eccentricities=eccmin_list)
                np.savez(f'Straindata/Training_TS_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz', time=self.TS_M, eccentricities=eccmin_list)

            else:
                np.savez(f'Straindata/Hp_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz', hp_dataset=hp_dataset, eccentricities=eccmin_list)
                np.savez(f'Straindata/Hc_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz', hc_dataset=hc_dataset, eccentricities=eccmin_list)
                np.savez(f'Straindata/TS_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz', time=self.TS_M, eccentricities=eccmin_list)

            print('Polarisations saved')

        return hp_DS_cut, hc_DS_cut, self.TS_M[-self.waveform_size:]   

    def generate_dataset_property(self, property = 'Frequency', save_dataset=False, eccmin_list=None, training_set=False, val_vecs=False):

        if eccmin_list is None:
            eccmin_list = self.eccmin_list
        else:
            eccmin_list = eccmin_list

           
        Residual_dataset = np.zeros((len(eccmin_list), self.waveform_size))
        
        try:
            if training_set is True:
                hp_DS = np.load(f'Straindata/Training_Hp_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz', allow_pickle=True)['hp_dataset']
                hc_DS = np.load(f'Straindata/Training_Hc_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz', allow_pickle=True)['hc_dataset']
                self.TS_M = np.load(f'Straindata/Training_TS_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')['time']
                print('Hp and hc validation vectors imported.')
            
            elif val_vecs is True:
                hp_DS, hc_DS, self.TS_M = hp, hc, TS_M = self.generate_dataset_polarisations(save_dataset, eccmin_list, save_dataset=False)
            
            else:
                hp_DS = np.load(f'Straindata/Hp_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz', allow_pickle=True)['hp_dataset']
                hc_DS = np.load(f'Straindata/Hc_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz', allow_pickle=True)['hc_dataset']
                self.TS_M = np.load(f'Straindata/TS_{min(self.eccmin_list)}_{max(self.eccmin_list)}.npz')['time']
                print('Hp and hc imported.')

            for i, eccentricity in enumerate(eccmin_list):
                if property == 'Amplitude':
                    self.hp_TS_M = types.TimeSeries(hp_DS[i], delta_t=self.DeltaT)
                    self.hc_TS_M = types.TimeSeries(hc_DS[i], delta_t=self.DeltaT)
                    self.eccmin = eccentricity

                    residual = self.residual_amp()[2][-self.waveform_size:]

                elif property == 'Phase':
                    self.hp_TS_M = types.TimeSeries(hp_DS[i], delta_t=self.DeltaT)
                    self.hc_TS_M = types.TimeSeries(hc_DS[i], delta_t=self.DeltaT)
                    self.eccmin = eccentricity

                    residual = self.residual_phase()[2][-self.waveform_size:]

                else:
                    print('Choose property = "Amplitude" or "Phase"')
                    sys.exit(1)

                if self.waveform_size is None:
                    waveform_size = len(residual)
                else:
                    waveform_size = self.waveform_size

                Residual_dataset[i] = residual

            self.TS_M = self.TS_M[len(self.TS_M) - self.waveform_size:]
            self.hp_T, self.hc_TS = None, None

            print('Residual {} calculated.'.format(property))
            
            if save_dataset is True:
                header = str(eccmin_list)

                if training_set is True:
                    np.savez(f'Straindata/Training_Res_{property}_{min(eccmin_list)}_{max(eccmin_list)}.npz', Residual_dataset=Residual_dataset, eccentricities=eccmin_list)
                    np.savez(f'Straindata/Training_TS_{min(eccmin_list)}_{max(eccmin_list)}.npz', time=self.TS_M, eccentricities=eccmin_list)
                    print('Dataset training vectors saved')
                
                np.savez(f'Straindata/Res_{property}_{min(eccmin_list)}_{max(eccmin_list)}.npz', Residual_dataset=Residual_dataset, eccentricities=eccmin_list)
                np.savez(f'Straindata/TS_{min(eccmin_list)}_{max(eccmin_list)}.npz', time=self.TS_M, eccentricities=eccmin_list)
                print('Dataset saved')

            return Residual_dataset
        

        except:
            print('Dataset hp/hc is not available. Generating new dataset...')

            if property == 'Amplitude':
                
                self.generate_dataset_polarisations(save_dataset=True, eccmin_list=eccmin_list, training_set=training_set)
                return self.generate_dataset_property('Amplitude', save_dataset, eccmin_list, training_set)
            
            elif property == 'Phase':

                self.generate_dataset_polarisations(save_dataset=True, eccmin_list=eccmin_list, training_set=training_set)
                return self.generate_dataset_property('Phase', save_dataset, eccmin_list, training_set)

            else:
                print('Choose property = "Amplitude" or "Phase"')
                sys.exit(1)


    

    # def plot_dataset_properties(self, property='Frequency', save_dataset=False):
    #     if property == 'Frequency':
    #         units = ' [Hz]'
    #     elif property == 'Amplitude':
    #         units = ''
    #     elif property == 'Phase':
    #         units = ' [radians]'
            
    #     residual_dataset = self.generate_dataset_property(property, save_dataset)

    #     fig_dataset_property = plt.figure(figsize=(12, 8))

    #     for i in range(len(self.eccmin_list)):
    #         plt.plot(self.TS_M, residual_dataset[i], label = 'e = {}'.format(self.eccmin_list[i]), linewidth=0.6)
    #         plt.xlabel('t [M]')
    #         plt.ylabel(property + units)
    #         plt.legend()
    #         plt.grid(True)

    #     plt.show()

        # figname = 'Residual_{}_e={}_{}'.format(property, self.eccmin_list.min, self.eccmin_list.max)
        # fig_dataset_property.savefig('Images/' + figname)

# plt.show()

# ecc_list = np.linspace(0.08, 0.12, num=11).round(4)
# wf = Waveform_properties(eccmin=0.1, freqmin=18, waveform_size=3500)
# # # wf.sim_inspiral_mass_indp()
# for ecc in ecc_list:
#     wf.plot_residuals(ecc, property='Phase')
# # wf.plot_constructed_waveform(waveform_size=3500)
# plt.show()
# # ds = Dataset()
