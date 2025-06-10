import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pycbc import waveform, types
from pyseobnr.generate_waveform import generate_modes_opt
from timeit import default_timer as timer

import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

plt.switch_backend('Agg')

class Simulate_Inspiral:
    """ Simulates time-domain (2,2) mode EOB waveform of a binary blackhole merger. Generates time-domain from starting frequency (freqmin) till peak at t=0 for time in geometric units. """
    
    def __init__(self, eccmin, mass_ratio=1, freqmin=650, chi1=0, chi2=0, rel_anomaly=0, waveform_size=None):
        """
        Parameters:
        ----------------
        eccmin [dimensionless], float: Eccentricity of binary at start freqmin
        mass_ratio [dimensionless], float : Mass ratio of the binary, mass ratio >= 1.
        freqmin [Hz], float: Start frequency of the waveform
        waveform_size [dimensionless], int: Waveform size used for Surrogate model. Amount of indices before merger. By default set to None, for which it generates the full waveform from fmin to t=0. 
        chi1 [dimensionless], float, ndarray : Spin of primary. If float, interpreted as z component
        chi2 [dimensionless], float, ndarray : Spin of secondary. If float, interpreted as z component
        rel_anomaly [rad], float : Relativistic anomaly. Radial phase which parametrizes the orbit within the Keplerian (relativistic) parametrization. Defaults to 0 (periastron).
        DeltaT [s], float : Sampling interval
        
        """

        # Initial parameters
        self.mass_ratio = mass_ratio # Mass ratio >= 1
        self.eccmin = eccmin # eccentricity of binary at start frequency
        self.freqmin = freqmin # Start frequency [Hz]
        self.chi1 = chi1 # Dimensionless spin of primary [float,ndarray]. If float, interpreted as z component
        self.chi2 = chi2 # Dimensionless spin of secondary [float,ndarray]. If float, interpreted as z component
        self.rel_anomaly= rel_anomaly # Relativistic anomaly. Radial phase which parametrizes the orbit within the Keplerian (relativistic) parametrization. Defaults to 0 (periastron)
        # self.t_backwards_geom = t_backwards_geom # backward time in geometric units, c=G=M=1
        self.DeltaT = 1./2048. # sampling interval [s]
        self.waveform_size = waveform_size # Waveform size used for Surrogate model. Amount of indices before merger.

        # if (t_backwards_geom == None and freqmin == None) or (t_backwards_geom is not None and freqmin is not None):
        #     print('Choose either start frequency (freqmin) OR negative starting time (t_backwards_geom)')
        #     sys.exit(1)

    def simulate_inspiral_mass_independent(self, eccmin=None, plot_polarisations=False, save_fig=False):
        """
        Simulate mass-independent plus and cross polarisations of the eccentric eob waveform (pyseobnr) (2,2) mode from f_start till t0 (waveform peak at t=0).
        
        Parameters:
        ----------------
        eccmin [dimensionless], float : For other eccentricity than Class specified eccmin, set new value.
        plot_polarisations, True OR False, bool : For a plot of the plus and cross polarisations, set to True.
        save_fig, True Or False, bool : If plot of the polarisations should be saved to a automatically created folder \Images, set to True.
        
        Returns:
        ----------------
        hp_TS [dimensionless], TimeSeries: Time-domain plus polarisation 
        hc_TS [dimensionless], TimeSeries: Time-domain cross polarisation 
        t_TS [M], TimeSeries: Time-domain in mass independent geometric units c=G=M=1
        """

        # Either set eccmin specifically or use the class defined value
        if eccmin is None:
            eccmin = self.eccmin
        else:
            eccmin = eccmin

        # To compute the runtime for 1 simulated waveform
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
            omega_geom = f_gw_geom * 3.14159 # geometric units 
            return omega_geom
        
        # eccentric waveform generator, using the psyeobnr model with eccentric EOB approximant
        t, modes = generate_modes_opt( 
            q=self.mass_ratio, # Mass ratio >= 1
            chi1=self.chi1, # Dimensionless spin of primary [float,np.ndarray]. If float, interpreted as z component
            chi2=self.chi2, # Dimensionless spin of secondary [float,np.ndarray]. If float, interpreted as z component
            omega_start=hz_to_omega(self.freqmin), # orbital start frequency in geometric units, M=1, c=G=1
            eccentricity=eccmin, # eccentricity of binary at start frequency
            rel_anomaly=self.rel_anomaly, # Relativistic anomaly. Radial phase which parametrizes the orbit within the Keplerian (relativistic) parametrization. Defaults to 0 (periastron)
            approximant="SEOBNRv5EHM", # eccentric EOB approximant
            )

        hp = modes["2,2"].real # mass independent hp (2,2) mode
        hc = modes["2,2"].imag # mass independent hc (2,2) mode

        # Convert to TimeSeries for eccentric case
        hp_TS = types.timeseries.TimeSeries(hp, delta_t=self.DeltaT, dtype=np.float32)  # plus polarisation in TimeSeries
        hc_TS = types.timeseries.TimeSeries(hc, delta_t=self.DeltaT, dtype=np.float32)  # cross polarisation in TimeSeries
        t_TS = -hp_TS.sample_times[::-1] # Timeseries time-domain
        
        del modes, hp, hc # Clear large variables after usage
        
        t0_idx = np.argmax(t > 0)  # First index where value > 0 because we only care about waveform up until peak at t=0
        hp_TS, hc_TS, t_TS = hp_TS[:t0_idx], hc_TS[:t0_idx], t_TS[:t0_idx]

        print(f'time : SimInspiral_M_independent ecc = {round(eccmin, 3)}, q = {self.mass_ratio}, freqmin = {self.freqmin}', (timer()-start), ' seconds')

        if plot_polarisations is True:
            
            # Set a specific shorter waveform length, or by default set to the length of the shortest waveform in the dataset to ensure all waveforms are equal in length.
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
                figname = 'Polarisations_q={}_ecc={}.png'.format(self.mass_ratio, eccmin)
                
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Polarisations', exist_ok=True)
                fig_simulate_inspiral.savefig('Images/Polarisations/' + figname)

                print('Figure is saved in Images/Polarisations')

            plt.close('all')  # Clean up plots

        return hp_TS, hc_TS, t_TS




class Waveform_Properties(Simulate_Inspiral):
    """
    Calculates and plots residuals (residual = eccentric - non-eccentric) of waveform properties: amplitude, phase and frequency.
    """

    def __init__(self, eccmin, mass_ratio=1, freqmin=650, waveform_size=None):
        """
        Parameters:
        ----------------
        eccmin [dimensionless], float: Eccentricity of binary at start freqmin
        mass_ratio [dimensionless], float : Mass ratio of the binary, mass ratio >= 1.
        freqmin [Hz], float: Start frequency of the waveform
        waveform_size [dimensionless], int: Waveform size used for Surrogate model. Amount of indices before merger.
        t_TS_circ [M], TimeSeries : Time array for non-eccentric inspiral
        hp_TS_circ [dimensionless], TimeSeries : plus polarisation of non-eccentric inspiral
        hc_TS_circ [dimensionless], TimesSeries : cross polarisation of non-eccentric inspiral
        """

        self.t_TS_circ = None # TimeSeries object of time array for non-eccentric inspiral
        self.hp_TS_circ = None # TimeSeries object of plus polarisation for non-eccentric inspiral 
        self.hc_TS_circ = None # TimeSeries object of cross polarisation for non-eccentric inspiral 

        # Inherit parameters from Simulate_Inspiral class
        Simulate_Inspiral.__init__(self, eccmin, mass_ratio=mass_ratio, freqmin=freqmin, waveform_size=waveform_size)
        
    def circulair_wf(self):
        """
        Simulate plus and cross polarisations of NON-ECCENTRIC waveform Inspiral for t in units [M].
       
        Returns:
        ----------------
        hp_TS_circ [dimensionless]: Time-domain plus polarisation of NON-ECCENTRIC waveform
        hc_TS_circ [dimensionless]: Time-domain cross polarisation of NON-ECCENTRIC waveform
        TS_circ [M]: Time-domain of NON-ECCENTRIC waveform in geometric units c=G=1
        """
        self.hp_TS_circ, self.hc_TS_circ, self.t_TS_circ = self.simulate_inspiral_mass_independent(eccmin=0)

    def calculate_residual(self, hp_TS, hc_TS, t_TS, property=None, plot_residual=False, save_fig=False):
        """
        Calculate residual (= eccentric - non-eccentric) of Waveform Inspiral property.
        Possible properties: phase, amplitude or frequency
        
        Parameters: 
        ----------------
        hp_TS [dimensionless], TimeSeries : mass independent plus polarisation
        hc_TS [dimensionless], TimeSeries : mass independent cross polarisation
        t_TS [M], TimeSeries : time-domain in for c=G=M = 1
        property, str: Choose residual for ['phase', 'amplitude', 'frequency']
        plot_residual, True OR False, bool: Set to True to include a plot of the residual including eccentric and non-eccentric case
        save_fig, True OR False, bool: Saves the figure to the directory Images/Residuals
        
        Returns:
        ----------------
        residual : residual = eccentric - non-eccentric for chosen property
        """       

        # Generate non-eccentric waveform only once 
        if self.hp_TS_circ is None:
            self.circulair_wf()

        # Calculate phase from plus and cross polarizations
        if property == 'phase':
            circ = np.array(waveform.utils.phase_from_polarizations(self.hp_TS_circ, self.hc_TS_circ)) # non-eccentric case
            eccentric = np.array(waveform.utils.phase_from_polarizations(hp_TS, hc_TS)) # eccentric case
            units = '[radians]'

            # Choose smallest waveform to calculate residual
            waveform_size = min(len(circ), len(eccentric))
            # Residual = circular - eccentric to prevent negative residual values
            residual = circ[-waveform_size:] - eccentric[-waveform_size:]

        # Calculate amplitude from plus and cross polarisations
        elif property == 'amplitude':
            circ = np.array(waveform.utils.amplitude_from_polarizations(self.hp_TS_circ, self.hc_TS_circ)) # non-eccentric case
            eccentric = np.array(waveform.utils.amplitude_from_polarizations(hp_TS, hc_TS)) # eccentric case
            units = '' # for plotting 

            # Choose smallest waveform to calculate residual
            waveform_size = min(len(circ), len(eccentric))
            residual = eccentric[-waveform_size:] - circ[-waveform_size:] 

        # Calculate frequency from plus and cross polarisations
        elif property == 'frequency':
            circ = waveform.utils.frequency_from_polarizations(self.hp_TS_circ, self.hc_TS_circ) # non-eccentric case
            eccentric = waveform.utils.frequency_from_polarizations(hp_TS, hc_TS) # eccentric case
            units = '[Hz]'

            # Adjust TS and TS_circ because frequency has one datapoint less
            t_TS = t_TS[-len(eccentric):]
            self.t_TS_circ = t_TS[-len(circ):]

            # Choose smallest waveform to calculate residual
            waveform_size = min(len(circ), len(eccentric))
            circ, eccentric = np.array(circ)[-waveform_size:], np.array(eccentric)[-waveform_size:]

            residual = eccentric - circ
        else:
            print('Choose property = "phase", "amplitude", "frequency"', property, 2)
            sys.exit(1)


        if plot_residual is True:
            fig_residual = plt.figure()
            
            plt.plot(t_TS, eccentric, label= property, linewidth=0.6) # eccentric property
            plt.plot(self.t_TS_circ, circ, label='Circular ' + property, linewidth=0.6) # non-eccentric property
            plt.plot(t_TS[-waveform_size:], residual, label='Residual ' + property, linewidth=0.6) # residual property
            
            plt.scatter(t_TS[-waveform_size], eccentric[-waveform_size], label= property, linewidth=0.6) # eccentric property at waveform cut
            plt.scatter(self.t_TS_circ[-waveform_size], circ[-waveform_size], label='Circular ' + property, linewidth=0.6) # non-eccentric property at waveform cut
            plt.scatter(t_TS[-waveform_size:][0], residual[0], label='Residual ' + property, linewidth=0.6) # residual property at waveform cut
            
            plt.xlabel('t [M]')
            plt.ylabel(property + ' ' + units)
            plt.title('Residual')
            plt.grid(True)
            plt.legend()

            plt.tight_layout()

            plt.close('all')  # Clean up plots

            if save_fig is True:
                figname = f'Residual {property} q={self.mass_ratio}, ecc={self.eccmin}.png'
                
                # Ensure the directory exists, creating it if necessary and save
                os.makedirs('Images/Residuals', exist_ok=True)
                fig_residual.savefig('Images/Residuals/' + figname)

                print('Figure is saved in Images/Residuals')
        return residual
    

wp = Waveform_Properties(eccmin=0.2, mass_ratio=1, freqmin=650)
hp_TS, hc_TS, TS = wp.simulate_inspiral_mass_independent(plot_polarisations=True, save_fig=True)
wp.calculate_residual(hp_TS, hc_TS, TS, property='phase', plot_residual=True, save_fig=True)



