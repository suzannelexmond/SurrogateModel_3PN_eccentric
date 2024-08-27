from Generate_eccentric import *

from scipy.signal import find_peaks
from scipy.optimize import curve_fit

plt.switch_backend('WebAgg')

class Get_Eccentricity(Simulate_Inspiral):
    """ Simulates Inspiral phase of a binary blackhole merger. 
    Optional: Simulate either mass dependent or mass independent. ; Simulate the frequency and phase of the waveform """
    
    def __init__(self, eccmin, total_mass=50, mass_ratio=1, freqmin=18, waveform_size=None):
        
        self.TS = None
        self.prop = None

        Simulate_Inspiral.__init__(self, eccmin=eccmin, total_mass=total_mass, mass_ratio=mass_ratio, freqmin=freqmin, waveform_size=waveform_size)

    
    def get_peaks(self, t_ref, property=None, waveform_prop=None, plot=False):
        """
        Input:
        t_ref in [M]
        property = 'Frequency' [Hz] OR 'Amplitude' [-]
        """
        
        hp_TS, hc_TS, self.TS = self.sim_inspiral_mass_indp(self.eccmin)           

        if property == 'Amplitude':
            prop = waveform.utils.amplitude_from_polarizations(hp_TS, hc_TS)

            units = ''
        elif property == 'Frequency':
            prop = waveform.utils.frequency_from_polarizations(hp_TS, hc_TS)
            prop = prop * 2*np.pi * self.total_mass * lal.MTSUN_SI # Convert Hz to rad/M_total
            
            units = ' [rad/M]'
        else:
            print('Choose property = "Amplitude" OR "Frequency"')

        
        TS_prop = -prop.sample_times[::-1] / (lal.MTSUN_SI * self.total_mass) 
        self.prop = prop[TS_prop > t_ref]
        self.TS = TS_prop[TS_prop > t_ref]

        valuesmin = -prop
        pericenters_idx, _per = find_peaks(prop, height=0)
        apocenters_idx, _apo = find_peaks(valuesmin, height=-100)



        if plot == True:

            fig_apos_peris, axs = plt.subplots(2, figsize=(10, 10))
            plt.subplots_adjust(hspace=0.5)

            axs[0].plot(self.TS, hp_TS[self.TS > t_ref], label='Real', linewidth=0.6, linestyle='-', color='blue')
            axs[0].plot(self.TS, hc_TS[self.TS > t_ref], label='Imag', linewidth=0.6, linestyle='--', color='blue')
            axs[0].set_ylabel('h$_{22}$')
            axs[0].set_xlabel('t [M]')
            axs[0].set_title('Waveform; total mass={} $M_\odot$, mass ratio={}, eccmin={}, freqmin={} Hz'.format(self.total_mass, self.mass_ratio, self.eccmin, self.freqmin))
            axs[0].grid()
            axs[0].legend()

            axs[1].plot(self.TS, self.prop, color='orange', linewidth=0.6)
            axs[1].scatter(self.TS[pericenters_idx], self.prop[pericenters_idx], color='blue', s=5 , label='pericenters')
            axs[1].scatter(self.TS[apocenters_idx], self.prop[apocenters_idx], color='magenta', s=5 , label='apocenters')
            axs[1].set_ylabel(property + units)
            axs[1].set_xlabel('t [M]')
            axs[1].legend(loc='upper left')
            axs[1].set_title('Apo- and pericenters')
            axs[1].grid()


            plt.show()
            figname = f'Apos_Peris_{property}_e={self.eccmin}.png'
            fig_apos_peris.savefig('Images/Apo_and_pericenters/' + figname)

            np.savez(f'{property}.npz', t=self.TS, property=self.prop)

        # def fit(t_start, t_end):

        #     self.get_peaks()

            # Example data generation (as before, but simplified)
            



        return pericenters_idx, apocenters_idx
    

    def fitting_model(self, t, A, n, t_merg):
        epsilon = 1e-10
        return A * np.sign(t_merg - t)* (abs(t_merg - t) + epsilon)**n


    def fit_peaks(self, peak_choice='Pericenters', convergence_value=2e-2, plot=False):

        # Initial parameters
        tL = self.TS[0]
        N_orbits = 10  # Initial number of orbits to consider
        converged = False

        # Store fitted parameters and points for each interval
        fitted_params = []
        fitted_intervals = []

        # Perform the initial fit for the first guess. Fit to self.prop

        if property == 'Frequency':
            p0=[0.1, -0.25, 0]
        elif property == 'Amplitude':
            p0=[1e-21, -0.3, 330]

        params, _ = curve_fit(fitting_model, self.TS, self.prop, p0=p0, maxfev=2000)
        A_fit, n_fit, t_merg_fit = params
        fitted_params.append(params)

        
        if peak_choice == 'Pericenters':
            peaks, _ = find_peaks(self.prop)
        elif peak_choice == 'Apocenters':
            peaks, _ = find_peaks(-self.prop)

        # Step 2: Iterate through the waveform, adjusting the interval dynamically
        count_iteration = 0

        while not converged:
            for i in range(len(peaks) - N_orbits):
                # Define the interval [tL, tR] dynamically
                print('length: ', len(peaks) - N_orbits, peaks[i + N_orbits - 1])
                tL = self.TS[peaks[i]]
                tR = self.TS[peaks[i + N_orbits - 1]]
                t_hat = self.TS[peaks[i + N_orbits//2]]

                # Store the points within the interval
                interval_indices = (self.TS >= tL) & (self.TS <= tR)
                fitted_intervals.append((self.TS[interval_indices], self.prop[interval_indices]))
                
                # Check if it reached the last interval 
                print(t[peaks[-1]])
                if tL >= self.TS[peaks][len(peaks) - N_orbits]:
                    print("Interval has reached the last peak, stopping iteration.")
                    converged = True
                    break

                # print('interval: ', self.TS[interval_indices], tL, tR)

                # fig_interval = plt.figure()
                # plt.plot(t, self.prop)
                # plt.plot(t[interval_indices], self.prop[interval_indices])
                # plt.scatter(t[peaks], self.prop[peaks])
                # plt.scatter(t[peaks][len(peaks) - N_orbits], self.prop[peaks][len(peaks) - N_orbits])
                # plt.show()

                # How many cycles to convergence 
                count_iteration += 1

                # Calculate the current fit
                prop_fit = self.fitting_model(self.TS, A_fit, n_fit, t_merg_fit)

                # Calculate local maxima of the difference between the data and the fit
                if peak_choice == 'Pericenters':
                    residual = self.prop - prop_fit
                elif peak_choice == 'Apocenters':
                    residual = -(self.prop - prop_fit)

                local_peaks, _ = find_peaks(residual[interval_indices])

                # Adjust the interval based on the new peaks
                num_extrema = len(local_peaks)
                if num_extrema > N_orbits:
                    tR = self.TS[local_peaks[N_orbits-1]]
                elif num_extrema < N_orbits:
                    tL = self.TS[local_peaks[0]]
                else:
                    converged = True

                # Refit with the new interval
                params, _ = curve_fit(self.fitting_model, t[interval_indices][local_peaks], self.prop[interval_indices][local_peaks], p0=params, maxfev=2000)
                fitted_params.append(params)

                error = abs(self.fitting_model(self.TS, *params)[peaks] - self.prop[peaks])

                if all (err < convergence_value for err in error): 
                    print("Convergence criterion met")
                    converged = True
                    break

                A_fit, n_fit, t_merg_fit = params

                # Fit for new interval
                prop_fit = self.fitting_model(self.TS, A_fit, n_fit, t_merg_fit)

        print(count_iteration, ' Cycles till convergence', '\n fit:', A_fit, n_fit, t_merg_fit)

        if plot is True:
            # Final plot of the fits with points
            fig_fits = plt.figure(figsize=(10, 6))

            plt.plot(self.TS, self.prop, label='Original Data', color='orange', alpha=0.5)
            # plt.plot(t, self.fitting_model(t, A_fit, n_fit, t_merg_fit))
            # Plot each fitted interval and the corresponding fit
            for i, (params, (t_interval, omega_interval)) in enumerate(zip(fitted_params, fitted_intervals)):
                plt.scatter(t[peaks], self.prop[peaks], color='red', s=10, label=f'Interval {i+1} Data' if i == 0 else "")
                plt.plot(t_interval, self.fitting_model(t_interval, *params), linestyle='--', color='blue', label=f'Interval {i+1} Fit' if i == 0 else "")

            plt.xlabel('Time [M]')
            plt.ylabel(r'$\omega_{22} [rad/M]$')
            plt.legend()
            plt.title('Fitting Intervals with Corresponding Fits')
            # plt.show()

            # Output the last set of fitted parameters
            print(f"Last Fitted Parameters: A = {A_fit}, n = {n_fit}, t_merg = {t_merg_fit}")

            # Final plot of the fits with points
            fig_final_fit = plt.figure(figsize=(10, 6))
            plt.plot(self.TS, self.prop, label='Original Data', color='orange', alpha=0.5)
            # plt.plot(t, self.fitting_model(t, A_fit, n_fit, t_merg_fit))
            # Plot each fitted interval and the corresponding fit
            plt.scatter(self.TS[peaks], self.prop[peaks], color='red', s=10, label=f'Interval {i+1} Data' if i == 0 else "")
            plt.plot(self.TS, self.fitting_model(self.TS, *fitted_params[-1]), linestyle='--', color='blue', label=f'Interval {i+1} Fit' if i == 0 else "")

            plt.xlabel('Time [M]')
            plt.ylabel(r'$\omega_{22} [rad/M]$')
            plt.legend()
            plt.title('Fitting Intervals with Corresponding Fits')
            # plt.show()

        return fitted_params[-1]


    def calc_eccentricity(self, w_fit_peris, w_fit_apos, plot_fit=False, plot_ecc=True):

        fit_peris = self.fitting_model(self.TS, *self.fit_peaks(peak_choice='Pericenters', plot=plot_fit))
        fit_apos = self.fitting_model(self.TS, *self.fit_peaks(peak_choice='Apocenters', plot=plot_fit))

        ecc = (np.sqrt(fit_peris) - np.sqrt(fit_apos)) / (np.sqrt(w_fit_peris) + np.sqrt(fit_apos))
        

        if plot_ecc is True:
            fig_ecc = plt.figure()
            plt.plot(self.TS, ecc(fit_peris, fit_apos))
            plt.plot()
            plt.show()

        return ecc



apos_a_pers = Get_Eccentricity(eccmin=0.4, freqmin=10, mass_ratio=4)
apos_a_pers.get_peaks(t_ref=-8000, property='Amplitude', plot=True)
# apos_a_pers.fit(-8000, )

##################################################
"""
Implement get peaks in fit function
"""