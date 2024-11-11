import lalsimulation as lalsim
import lal
import matplotlib.pyplot as plt
from numba import jit, cuda
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from pycbc import types
from pycbc.types import timeseries
from pycbc import waveform
import sys
import os

plt.switch_backend('WebAgg')

class Simulate_Inspiral:
    """ Simulates time-domain post-Newtonian Inspiral of a binary blackhole merger up to 3PN-order. 
    Optional: Simulate either mass dependent or mass independent. ; Simulate the frequency and phase of the waveform """
    
    def __init__(self, eccmin, total_mass=50, mass_ratio=1, freqmin=18, waveform_size=None):
        
        self.total_mass = total_mass # Total mass of the BBH system [Solar Mass]
        self.mass_ratio = mass_ratio # Mass ratio 0 < q < 1, so M_1 > M_2
        self.eccmin = eccmin # eccentricity of binary at start frequency
        self.freqmin = freqmin # Start frequency [Hz]
        self.DeltaT = 1./2048. # 
        self.lalDict = lal.CreateDict() # 

        self.waveform_size = waveform_size # Waveform size used for Surrogate model. Amount of indices before merger.

        self.hp_TS = None
        self.hc_TS = None
        TS = None

        self.hp_TS_M = None
        self.hc_TS_M = None
        self.TS_M = None


    def simulate_inspiral(self, eccmin=None):
        """
        Simulate plus and cross polarisations of Waveform Inspiral for t in units [seconds].
        
        Parameters:
        ----------------
        eccmin [dimensionless], {0,...,1} : For other eccentricity than Class specified eccmin, set new value

        Returns:
        ----------------
        hp_TS [dimensionless]: Time-domain plus polarisation
        hc_TS [dimensionless]: Time-domain cross polarisation
        TS [seconds]: Time-domain
        """

        if eccmin == 0:
            print("eccentricity doesn't accept zero. Eccentricity is by default set to e=1e-5")
            eccmin = 1e-5
        elif eccmin is None:
            eccmin = self.eccmin
        else:
            eccmin = eccmin

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
      
        print(f'time : SimInspiral_M_independent ecc = {eccmin}, M_total = {self.total_mass} M_sol, q = {self.mass_ratio}, freqmin = {self.freqmin}', (timer()-start)/60, ' minutes')

        return hp_TS, hc_TS, TS
    

    def simulate_inspiral_mass_independent(self, eccmin=None, plot_polarisations=False, save_fig=False):
        """
        Simulate plus and cross polarisations of Waveform Inspiral for t in units [M].
        
        Parameters: 
        ----------------
        eccmin [dimensionless], {0,...,1} : For other eccentricity than Class specified eccmin, set new value
        plot_polarisations: Set to True to include a plot of the polarisations
        save_fig : Saves the figure to the directory Images/Polarisations
        
        Returns:
        ----------------
        hp_TS [dimensionless]: Time-domain plus polarisation
        hc_TS [dimensionless]: Time-domain cross polarisation
        TS [M]: Time-domain
        """

        if eccmin is None:
            eccmin = self.eccmin

        hp_TS, hc_TS, TS = self.simulate_inspiral(eccmin)
        TS_M = TS / (lal.MTSUN_SI * self.total_mass) 

        phase = waveform.utils.phase_from_polarizations(hp_TS, hc_TS)
        amp = waveform.utils.amplitude_from_polarizations(hp_TS, hc_TS)

        h = amp * np.exp(-1j * phase)
        
        if plot_polarisations is True:
            
            if self.waveform_size is None:
                self.waveform_size = len(TS_M)

            fig_simulate_inspiral = plt.figure()
            
            length_diff = len(TS) - self.waveform_size
            plt.plot(TS[length_diff:], hp_TS[length_diff:], label = f'$h_+$', linewidth=0.2)
            # plt.plot(TS, np.real(h))
            # plt.plot(TS_M[length_diff:], hc_TS[length_diff:], label = f'$h_x$', linestyle='dashed', linewidth=0.6)
            plt.legend(loc = 'upper left')
            plt.xlabel('t [M]')
            plt.ylabel('h$_{22}$')
            plt.title(f'M={self.total_mass}$M_\odot$, q={self.mass_ratio}, e={eccmin}, f_min={self.freqmin} Hz')
            plt.grid(True)

            plt.tight_layout()

            if save_fig is True:
                figname = 'Polarisations M={}, q={}, ecc={}.png'.format(self.total_mass, self.mass_ratio, eccmin)
                
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Polarisations', exist_ok=True)
                fig_simulate_inspiral.savefig('Images/Polarisations/' + figname)

                print('Figure is saved in Images/Polarisations')

        return hp_TS, hc_TS, TS_M
 



class Waveform_Properties(Simulate_Inspiral):

    def __init__(self, eccmin, total_mass=50, mass_ratio=1, freqmin=18, waveform_size=None):
 
        self.TS_M_circ = None
        self.hp_TS_circ = None
        self.hc_TS_circ = None

        Simulate_Inspiral.__init__(self, eccmin, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin, waveform_size=waveform_size)
        
    def circulair_wf(self):
        """
        Simulate plus and cross polarisations of NON-ECCENTRIC waveform Inspiral for t in units [M].
       
        Returns:
        ----------------
        hp_TS_circ [dimensionless]: Time-domain plus polarisation of NON-ECCENTRIC waveform
        hc_TS_circ [dimensionless]: Time-domain cross polarisation of NON-ECCENTRIC waveform
        TS_circ [M]: Time-domain of NON-ECCENTRIC waveform
        """
        self.hp_TS_circ, self.hc_TS_circ, self.TS_M_circ = self.simulate_inspiral_mass_independent(eccmin=1e-5)

    def calculate_residual(self, hp, hc, property=None, plot_residual=False, save_fig=False):
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

        if self.hp_TS_circ is None:
            self.circulair_wf()

        if property == 'phase':
            circ = np.array(waveform.utils.phase_from_polarizations(self.hp_TS_circ, self.hc_TS_circ))
            eccentric = np.array(waveform.utils.phase_from_polarizations(hp, hc))
            units = '[radians]'

            # Apply universal phaseshift to set all phases to zero at t_ref
            # eccentric -= eccentric[-self.waveform_size]
            # circ -= circ[-self.waveform_size]

            residual = circ[len(circ) - len(eccentric):] - eccentric

        elif property == 'amplitude':
            circ = np.array(waveform.utils.amplitude_from_polarizations(self.hp_TS_circ, self.hc_TS_circ))
            eccentric = np.array(waveform.utils.amplitude_from_polarizations(hp, hc))
            units = ''

            residual = eccentric - circ[len(circ) - len(eccentric):]

        elif property == 'frequency':
            circ = waveform.utils.frequency_from_polarizations(self.hp_TS_circ, self.hc_TS_circ)
            eccentric = waveform.utils.frequency_from_polarizations(hp, hc)
            units = '[Hz]'

            # Adjust TS and TS_circ because frequency has one datapoint less
            self.TS_M = -eccentric.sample_times[::-1] / (lal.MTSUN_SI * self.total_mass )
            self.TS_M_circ = -circ.sample_times[::-1] / (lal.MTSUN_SI * self.total_mass )
            circ, eccentric = np.array(circ), np.array(eccentric)

            residual = eccentric - circ[len(circ) - len(eccentric):]
        else:
            print('Choose property = "phase", "amplitude", "frequency"', property, 2)
            sys.exit(1)

        
            



        if plot_residual is True:
            fig_residual = plt.figure()

            length_diff = len(circ) - len(eccentric)
            plt.plot(self.TS_M_circ[length_diff + 20:], eccentric[20:], label= property, linewidth=0.6)
            plt.scatter(self.TS_M_circ[length_diff:][length_diff], eccentric[length_diff])
            plt.scatter(self.TS_M_circ[length_diff:][length_diff], circ[length_diff])
            plt.scatter(self.TS_M_circ[length_diff:][length_diff], residual[length_diff])
            plt.plot(self.TS_M_circ, circ, label='Circular ' + property, linewidth=0.6)
            plt.plot(self.TS_M_circ[length_diff:], residual, label='Residual ' + property, linewidth=0.6)
            plt.xlabel('t [M]')
            plt.ylabel(property + ' ' + units)
            plt.title('Residual')
            plt.grid(True)
            plt.legend()

            plt.tight_layout()

            if save_fig is True:
                figname = f'Residual {property} M={self.total_mass}, q={self.mass_ratio}, ecc={self.eccmin}.png'
                
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Residuals', exist_ok=True)
                fig_residual.savefig('Images/Residuals/' + figname)

                print('Figure is saved in Images/Residuals')
        return residual
    
# si = Simulate_Inspiral(0.2)

# # si.simulate_inspiral_mass_independent(plot_polarisations=True)
# wp = Waveform_Properties(0.2, waveform_size=3500)
# hp, hc, wp.TS_M = wp.simulate_inspiral_mass_independent()
# wp.calculate_residual(hp, hc, 'phase', plot_residual=True)
# wp.calculate_residual(hp, hc, 'amplitude', plot_residual=True)
# plt.show()


# import os
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# from timeit import default_timer as timer
# from pycbc import types, waveform
# from pycbc.types import timeseries
# import lal
# import lalsimulation as lalsim

# plt.switch_backend('WebAgg')


# class Simulate_Inspiral:
#     """ Simulates time-domain post-Newtonian inspiral of a binary black hole (BBH) merger up to 3PN order, using the lalsimulation library.
#     """
    
#     def __init__(self, eccmin, total_mass=50, mass_ratio=1, freqmin=18, waveform_size=None):
#         self.total_mass = total_mass            # Total mass of the BBH system [Solar Mass]
#         self.mass_ratio = mass_ratio            # Mass ratio (0 < q < 1), where M1 > M2
#         self.eccmin = eccmin                    # Eccentricity of binary at reference frequency
#         self.freqmin = freqmin                  # Reference frequency [Hz]
#         self.DeltaT = 1.0 / 2048.               # Sampling rate
#         self.lalDict = lal.CreateDict()         # LAL dictionary for simulation settings

#         self.waveform_size = waveform_size      # Cut-off length of the waveforms, (N time-samples before merger)
        
#         self.hp_TS = None                       # Time-domain plus polarization
#         self.hc_TS = None                       # Time-domain cross polarization
#         self.TS = None                          # Time-domain for the signal
        

#     def simulate_inspiral(self, eccmin=None):
#         """
#         Simulates the plus and cross polarizations of the waveform inspiral in time domain.

#         Parameters:
#         eccmin (float): Eccentricity. Can be set manually, otherwise take class default value.

#         Returns:
#         tuple: (hp_TS, hc_TS, TS) representing the time-domain plus and cross polarizations and the time-domain array.
#         """
#         if eccmin is None:
#             eccmin = self.eccmin

#         start = timer()
        
#         # Mass assignments based on mass ratio
#         mass1 = self.total_mass / (1 + self.mass_ratio)
#         mass2 = self.total_mass - mass1

#         # Generate waveform using LALSimulation
#         hp, hc = lalsim.SimInspiralTD(
#             m1=lal.MSUN_SI * mass1, m2=lal.MSUN_SI * mass2,
#             S1x=0., S1y=0., S1z=0., S2x=0., S2y=0., S2z=0.,
#             distance=400. * 1e6 * lal.PC_SI, inclination=0.,
#             phiRef=0., longAscNodes=0, eccentricity=eccmin, meanPerAno=0.,
#             deltaT=self.DeltaT, f_min=self.freqmin, f_ref=self.freqmin,
#             LALparams=self.lalDict, approximant=lalsim.EccentricTD
#         )

#         # Convert to time series
#         hp_TS = types.timeseries.TimeSeries(hp.data.data, delta_t=hp.deltaT)
#         hc_TS = types.timeseries.TimeSeries(hc.data.data, delta_t=hc.deltaT)
#         TS = -hp_TS.sample_times[::-1]  # Time array

#         print(f'Inspiral calculated: Eccentricity = {eccmin}, Total Mass = {self.total_mass} M_sun, Mass Ratio = {self.mass_ratio}, Reference Frequency = {self.freqmin} Hz | Time = {timer() - start:.2f} seconds ')
        
#         return hp_TS, hc_TS, TS
    

#     def simulate_inspiral_mass_independent(self, eccmin=None, plot_polarisations=False, save_fig=False):
#         """
#         Simulate waveform polarizations with time-domain independent of mass, with extra options for plotting and saving.

#         Parameters: 
#         eccmin (float): Eccentricity. Can be set manually, otherwise take class default value.
#         plot_polarisations (bool): Set to True for a plot of the polarisations.
#         save_fig (bool): If True, saves the plot to Images/Polarisations

#         Returns:
#         tuple: (hp_TS, hc_TS, TS_M) with the plus, cross polarizations, and time array in units [M]
#         """
#         if eccmin is None:
#             eccmin = self.eccmin

#         hp_TS, hc_TS, TS = self.simulate_inspiral(eccmin)
#         TS_M = TS / (lal.MTSUN_SI * self.total_mass)  # Time in units of M

#         if plot_polarisations:
#             self._plot_polarisations(hp_TS, hc_TS, TS_M, eccmin, save_fig)

#         return hp_TS, hc_TS, TS_M

#     def _plot_polarisations(self, hp_TS, hc_TS, TS_M, eccmin, save_fig):
#         """ Plots the polarizations with extra option for saving. """
#         fig, ax = plt.subplots()
#         waveform_size = self.waveform_size or len(TS_M)
#         length_diff = len(TS_M) - waveform_size

#         ax.plot(TS_M[length_diff:], hp_TS[length_diff:], label='$h_+$', linewidth=0.2)
#         ax.set(xlabel='t [M]', ylabel='$h_{22}$', title=f'M={self.total_mass} $M_\\odot$, q={self.mass_ratio}, e={eccmin}, f_min={self.freqmin} Hz')
#         ax.legend(loc='upper left')
#         ax.grid(True)
        
#         if save_fig:
#             os.makedirs('Images/Polarisations', exist_ok=True)
#             fig.savefig(f'Images/Polarisations/Polarisations_M={self.total_mass}_q={self.mass_ratio}_e={eccmin}.png')
#             print('Figure saved in Images/Polarisations')


# class Waveform_Properties(Simulate_Inspiral):
#     """ Extends SimulateInspiral to include calculations of the waveform frequency, phase and amplitude including it's residuals (eccentric - non-eccentric). """

#     def __init__(self, eccmin, total_mass=50, mass_ratio=1, freqmin=18, waveform_size=None):
#         self.hp_TS_circ = None
#         self.hc_TS_circ = None
#         self.TS_M_circ = None
        
#         super().__init__(eccmin, total_mass, mass_ratio, freqmin, waveform_size)
        
#     def calculate_residual(self, hp, hc, property=None, plot_residual=False, save_fig=False):
#         """
#         Calculates the residual between eccentric and non-eccentric waveforms for phase, amplitude, or frequency.

#         Parameters:
#         property (str): 'phase', 'amplitude', or 'frequency'.
#         plot_residual (bool): Set to True for a plot of the property residuals.
#         save_fig (bool): If True, saves the plot in Images/Residuals.

#         Returns:
#         np.ndarray: Residual values
#         """
#         if self.hp_TS_circ is None:
#             self._generate_circular_waveform()

#         if property not in {'phase', 'amplitude', 'frequency'}:
#             print("Error: Choose 'phase', 'amplitude', or 'frequency'")
#             sys.exit(1)

#         # Calculate residual of phase/amplitude/frequency
#         circ_values, ecc_values, units = self._get_waveform_property_values(hp, hc, property)
#         residual = ecc_values - circ_values[-len(ecc_values):]

#         # Plot residual for plot_residual is True
#         if plot_residual:
#             self._plot_residual(property, ecc_values, circ_values, residual, units, save_fig)

#         return residual

#     def _generate_circular_waveform(self):
#         """ Generates non-eccentric waveform for reference case. """
#         # Eccmin can not be set to zero, due to limitations of the siminspiral function of lalsim. Value is set for close to zero.
#         self.hp_TS_circ, self.hc_TS_circ, self.TS_M_circ = self.simulate_inspiral_mass_independent(eccmin=1e-5)

#     def _get_waveform_property_values(self, hp, hc, property):
#         """ Retrieves waveform property values for eccentric and non-eccentric waveforms. 
        
#         Parameters:
#         hp (np.ndarray) : plus polarisation of the waveform.
#         hc (np.ndarray) : cross polarisation of the waveform.
#         """
#         if property == 'phase':
#             circ_values = waveform.utils.phase_from_polarizations(self.hp_TS_circ, self.hc_TS_circ)
#             ecc_values = waveform.utils.phase_from_polarizations(hp, hc)
#             units = '[radians]'
#         elif property == 'amplitude':
#             circ_values = waveform.utils.amplitude_from_polarizations(self.hp_TS_circ, self.hc_TS_circ)
#             ecc_values = waveform.utils.amplitude_from_polarizations(hp, hc)
#             units = ''
#         else:
#             circ_values = waveform.utils.frequency_from_polarizations(self.hp_TS_circ, self.hc_TS_circ)
#             ecc_values = waveform.utils.frequency_from_polarizations(hp, hc)
#             units = '[Hz]'

#         return np.array(circ_values), np.array(ecc_values), units

#     def _plot_residual(self, property, ecc_values, circ_values, residual, units, save_fig):
#         """ Plots the residual for phase, frequency or amplitude of the waveform. """
#         fig, ax = plt.subplots()
#         length_diff = len(circ_values) - len(ecc_values)
        
#         ax.plot(self.TS_M_circ[length_diff:], ecc_values, label=property, linewidth=0.6)
#         ax.plot(self.TS_M_circ, circ_values, label='Circular ' + property, linewidth=0.6)
#         ax.plot(self.TS_M_circ[length_diff:], residual, label='Residual ' + property, linewidth=0.6)

#         ax.set(xlabel='t [M]', ylabel=f'{property} {units}', title='Residual')

