import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pycbc import waveform, types
import lal
import lalsimulation as lalsim
from pyseobnr.generate_waveform import GenerateWaveform, generate_modes_opt
from timeit import default_timer as timer

import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

plt.switch_backend('WebAgg')

class Simulate_Inspiral:
    """ Simulates time-domain EOB waveform of a binary blackhole merger with peak at t=0 and time in geometric units. """
    
    def __init__(self, eccmin, mass_ratio=1, freqmin=650, waveform_size=None):
        
        self.mass_ratio = mass_ratio # Mass ratio 0 < q < 1, so M_1 > M_2
        self.eccmin = eccmin # eccentricity of binary at start frequency
        self.freqmin = freqmin # Start frequency [Hz]
        # self.t_backwards_geom = t_backwards_geom # backward time in geometric units, c=G=M=1
        self.DeltaT = 1./2048. # 
        self.lalDict = lal.CreateDict() # 

        self.waveform_size = waveform_size # Waveform size used for Surrogate model. Amount of indices before merger.

        # if (t_backwards_geom == None and freqmin == None) or (t_backwards_geom is not None and freqmin is not None):
        #     print('Choose either start frequency (freqmin) OR negative starting time (t_backwards_geom)')
        #     sys.exit(1)

    def simulate_inspiral_mass_independent(self, eccmin=None, plot_polarisations=False, save_fig=False):
        """
        Simulate plus and cross polarisations of Waveform Inspiral for t in units [seconds].
        
        Parameters:
        ----------------
        eccmin [dimensionless], {0,...,1}, float : For other eccentricity than Class specified eccmin, set new value

        Returns:
        ----------------
        hp_TS [dimensionless], TimeSeries: Time-domain plus polarisation 
        hc_TS [dimensionless], TimeSeries: Time-domain cross polarisation 
        TS [M], TimeSeries: Time-domain in mass independent geometric units c=G=M=1
        """

        if eccmin is None:
            eccmin = self.eccmin
        else:
            eccmin = eccmin

        start = timer()

        def hz_to_omega(f_gw_hz, mass_msun=1):
            """
            Convert GW frequency in Hz to orbital angular frequency in geometric units.

            Parameters:
            - f_gw_hz: gravitational wave frequency in Hz
            - mass_msun: total mass in solar masses

            Returns:
            - orbital angular frequency (dimensionless, geometric units)
            """
            M_sec = mass_msun * 4.9255e-6
            f_gw_geom = f_gw_hz * M_sec
            omega_geom = f_gw_geom * 3.14159
            return omega_geom

        # settings = dict(
        #     t_backwards=self.t_backwards_geom,  # Geometric units
        #     warning_bwd_int=False,  # Setting this to False will avoid the warning message
        # )
        

        t, modes = generate_modes_opt(
            q=self.mass_ratio,
            chi1=0,
            chi2=0,
            omega_start=hz_to_omega(self.freqmin), # orbital frequency in geometric units, M=1, c=G=1
            eccentricity=eccmin,
            rel_anomaly=0,
            approximant="SEOBNRv5EHM",
        )

        hp = modes["2,2"].real # mass independent hp
        hc = modes["2,2"].imag # mass independent hc

        print(f'time : SimInspiral_M_independent ecc = {round(eccmin, 3)}, q = {self.mass_ratio}, freqmin = {self.freqmin}', (timer()-start), ' seconds')


        if plot_polarisations is True:
            
            if self.waveform_size is None:
                self.waveform_size = len(t)

            fig_simulate_inspiral = plt.figure(figsize=(12,5))

            length_diff = len(t) - self.waveform_size

            plt.plot(t[length_diff:], hp[length_diff:], label = f'$h_+$', linewidth=0.6)

            plt.legend(loc = 'upper left')
            plt.xlabel('t [s]')
            plt.ylabel('$h_{22}]$')
            plt.title(f'q={self.mass_ratio}, e={eccmin}, f_min={self.freqmin} Hz')
            plt.grid(True)

            plt.tight_layout()

            if save_fig is True:
                figname = 'Polarisations q={}, ecc={}.png'.format(self.mass_ratio, eccmin)
                
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Polarisations', exist_ok=True)
                fig_simulate_inspiral.savefig('Images/Polarisations/' + figname)

                print('Figure is saved in Images/Polarisations')

        return hp, hc, t





class Waveform_Properties(Simulate_Inspiral):

    def __init__(self, eccmin, mass_ratio=1, freqmin=650, waveform_size=None):
 
        self.t_circ = None
        self.hp_circ = None
        self.hc_circ = None

        Simulate_Inspiral.__init__(self, eccmin, mass_ratio=mass_ratio, freqmin=freqmin, waveform_size=waveform_size)
        
    def circulair_wf(self):
        """
        Simulate plus and cross polarisations of NON-ECCENTRIC waveform Inspiral for t in units [M].
       
        Returns:
        ----------------
        hp_TS_circ [dimensionless]: Time-domain plus polarisation of NON-ECCENTRIC waveform
        hc_TS_circ [dimensionless]: Time-domain cross polarisation of NON-ECCENTRIC waveform
        TS_circ [M]: Time-domain of NON-ECCENTRIC waveform
        """
        self.hp_circ, self.hc_circ, self.t_circ = self.simulate_inspiral_mass_independent(eccmin=0)

    def calculate_residual(self, hp, hc, t, property=None, plot_residual=False, save_fig=False):
        """
        Calculate residual (= eccentric - non-eccentric) of Waveform Inspiral property.
        Possible properties: phase, amplitude or frequency
        
        Parameters: 
        ----------------
       
        property : Choose residual for ['phase', 'amplitude', 'frequency']
        plot_residual: Set to True to include a plot of the residual including eccentric and non-eccentric case
        save_fig : Saves the figure to the directory Images/Residuals
        
        Returns:
        ----------------
        residual : residual = eccentric - non-eccentric for chosen property
        """       

        if self.hp_circ is None:
            self.circulair_wf()

        # Convert to TimeSeries for eccentric case
        hp_TS = types.timeseries.TimeSeries(hp, delta_t=self.DeltaT)  # plus polarisation TimeSeries
        hc_TS = types.timeseries.TimeSeries(hc, delta_t=self.DeltaT)  # cross polarisation TimeSeries
        # Convert to TimeSeries for circular case
        hp_TS_circ = types.timeseries.TimeSeries(self.hp_circ, delta_t=self.DeltaT)  # plus polarisation TimeSeries
        hc_TS_circ = types.timeseries.TimeSeries(self.hc_circ, delta_t=self.DeltaT)  # cross polarisation TimeSeries

        if property == 'phase':
            circ = np.array(waveform.utils.phase_from_polarizations(hp_TS_circ, hc_TS_circ))
            eccentric = np.array(waveform.utils.phase_from_polarizations(hp_TS, hc_TS))
            units = '[radians]'

            # Residual = circular - eccentric to prevent negative residual values
            waveform_size = min(len(circ), len(eccentric))
            residual = circ[-waveform_size:] - eccentric[-waveform_size:]

        elif property == 'amplitude':
            circ = np.array(waveform.utils.amplitude_from_polarizations(hp_TS_circ, hc_TS_circ))
            eccentric = np.array(waveform.utils.amplitude_from_polarizations(hp_TS, hc_TS))
            units = ''

            waveform_size = min(len(circ), len(eccentric))
            residual = eccentric[-waveform_size:] - circ[-waveform_size:] 

        elif property == 'frequency':
            circ = waveform.utils.frequency_from_polarizations(hp_TS_circ, hc_TS_circ)
            eccentric = waveform.utils.frequency_from_polarizations(hp_TS, hc_TS)
            units = '[Hz]'

            # Adjust TS and TS_circ because frequency has one datapoint less
            t = t[-len(eccentric):]
            self.t_circ = t[-len(circ):]
            waveform_size = min(len(circ), len(eccentric))
            circ, eccentric = np.array(circ)[-waveform_size:], np.array(eccentric)[-waveform_size:]

            waveform_size = min(len(circ), len(eccentric))
            residual = eccentric[-waveform_size:] - circ[-waveform_size:] 
        else:
            print('Choose property = "phase", "amplitude", "frequency"', property, 2)
            sys.exit(1)


        if plot_residual is True:
            fig_residual = plt.figure()
            
            plt.plot(t, eccentric, label= property, linewidth=0.6)
            plt.plot(self.t_circ, circ, label='Circular ' + property, linewidth=0.6)
            plt.plot(t[-waveform_size:], residual, label='Residual ' + property, linewidth=0.6)
            
            plt.scatter(t[-waveform_size], eccentric[-waveform_size], label= property, linewidth=0.6)
            plt.scatter(self.t_circ[-waveform_size], circ[-waveform_size], label='Circular ' + property, linewidth=0.6)
            plt.scatter(t[-waveform_size:][0], residual[0], label='Residual ' + property, linewidth=0.6)
            
            plt.xlabel('t [M]')
            plt.ylabel(property + ' ' + units)
            plt.title('Residual')
            plt.grid(True)
            plt.legend()

            plt.tight_layout()

            if save_fig is True:
                figname = f'Residual {property} q={self.mass_ratio}, ecc={self.eccmin}.png'
                
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Residuals', exist_ok=True)
                fig_residual.savefig('Images/Residuals/' + figname)

                print('Figure is saved in Images/Residuals')
        return residual
    
# si = Simulate_Inspiral(eccmin=0, mass_ratio=1, freqmin=650)
# si.simulate_inspiral_mass_independent(plot_polarisations=True)

# wp = Waveform_Properties(eccmin=0.2, mass_ratio=1, freqmin=650)
# hp_TS, hc_TS, TS = wp.simulate_inspiral_mass_independent(plot_polarisations=True)
# wp.calculate_residual(hp_TS, hc_TS, TS, property='phase', plot_residual=True)
# wp.calculate_residual(hp_TS, hc_TS, TS, property='amplitude', plot_residual=True)
# wp.calculate_residual(hp_TS, hc_TS, TS, property='frequency', plot_residual=True)
# plt.show()



