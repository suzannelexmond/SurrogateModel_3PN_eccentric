import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

plt.switch_backend('WebAgg')

# Load the data
data = np.load('Frequency.npz')
t = data['t']  # Time array
omega_22 = data['property']  # Frequency array

# Define the fitting model with an epsilon to avoid divide-by-zero issues
epsilon = 1e-10
# Add minus sign for invalid encounter in power
def fitting_model(t, A, n, t_merg):
    return A * np.sign(t_merg - t)* (abs(t_merg - t) + epsilon)**n

def fit(peak_choice='Pericenters', convergence_value=1e-2):
    # Step 1: Identify all pericenters by finding peaks in the frequency data

    # Initial parameters
    tL = t[0]
    N_orbits = 10  # Initial number of orbits to consider
    converged = False

    # Store fitted parameters and points for each interval
    fitted_params = []
    fitted_intervals = []

    # Perform the initial fit for the first guess. Fit to omega_22
    params, _ = curve_fit(fitting_model, t, omega_22, p0=[0.1, -0.25, 0], maxfev=2000)
    A_fit, n_fit, t_merg_fit = params
    fitted_params.append(params)


    if peak_choice == 'Pericenters':
        peaks, _ = find_peaks(omega_22)
    elif peak_choice == 'Apocenters':
        peaks, _ = find_peaks(-omega_22)

    # Step 2: Iterate through the waveform, adjusting the interval dynamically
    count_iteration = 0

    print(len(peaks) - N_orbits)
    print('last t: ', t[-1])
    
    while not converged:
        for i in range(len(peaks) - N_orbits):
            # Define the interval [tL, tR] dynamically
            print('length: ', len(peaks) - N_orbits, peaks[i + N_orbits - 1])
            tL = t[peaks[i]]
            tR = t[peaks[i + N_orbits - 1]]
            t_hat = t[peaks[i + N_orbits//2]]

            # Store the points within the interval
            interval_indices = (t >= tL) & (t <= tR)
            fitted_intervals.append((t[interval_indices], omega_22[interval_indices]))
            
            # Check if it reached the last interval 
            print(t[peaks[-1]])
            if tL >= t[peaks][len(peaks) - N_orbits]:
                print("Interval has reached the last peak, stopping iteration.")
                converged = True
                break

            print('interval: ', t[interval_indices], tL, tR)

            fig_interval = plt.figure()
            plt.plot(t, omega_22)
            plt.plot(t[interval_indices], omega_22[interval_indices])
            plt.scatter(t[peaks], omega_22[peaks])
            plt.scatter(t[peaks][len(peaks) - N_orbits], omega_22[peaks][len(peaks) - N_orbits])
            plt.show()

            # How many cycles to convergence 
            count_iteration += 1

            # Calculate the current fit
            omega_fit = fitting_model(t, A_fit, n_fit, t_merg_fit)

            # Calculate local maxima of the difference between the data and the fit
            if peak_choice == 'Pericenters':
                residual = omega_22 - omega_fit
            elif peak_choice == 'Apocenters':
                residual = -(omega_22 - omega_fit)

            local_peaks, _ = find_peaks(residual[interval_indices])

            # Adjust the interval based on the new peaks
            num_extrema = len(local_peaks)
            if num_extrema > N_orbits:
                tR = t[local_peaks[N_orbits-1]]
            elif num_extrema < N_orbits:
                tL = t[local_peaks[0]]
            else:
                converged = True

            # Refit with the new interval
            params, _ = curve_fit(fitting_model, t[interval_indices][local_peaks], omega_22[interval_indices][local_peaks], p0=params, maxfev=2000)
            fitted_params.append(params)

            error = abs(fitting_model(t, *params)[peaks] - omega_22[peaks])

            if all (err < convergence_value for err in error): 
                print("Convergence criterion met")
                converged = True
                break

            A_fit, n_fit, t_merg_fit = params

            # Fit for new interval
            omega_fit = fitting_model(t, A_fit, n_fit, t_merg_fit)


    print(count_iteration, ' Cycles till convergence', '\n fit:', A_fit, n_fit, t_merg_fit)

    # Final plot of the fits with points
    fig_fits = plt.figure(figsize=(10, 6))
    plt.plot(t, omega_22, label='Original Data', color='orange', alpha=0.5)
    # plt.plot(t, fitting_model(t, A_fit, n_fit, t_merg_fit))
    # Plot each fitted interval and the corresponding fit
    for i, (params, (t_interval, omega_interval)) in enumerate(zip(fitted_params, fitted_intervals)):
        plt.scatter(t[peaks], omega_22[peaks], color='red', s=10, label=f'Interval {i+1} Data' if i == 0 else "")
        plt.plot(t_interval, fitting_model(t_interval, *params), linestyle='--', color='blue', label=f'Interval {i+1} Fit' if i == 0 else "")

    plt.xlabel('Time [M]')
    plt.ylabel(r'$\omega_{22} [rad/M]$')
    plt.legend()
    plt.title('Fitting Intervals with Corresponding Fits')
    # plt.show()

    # Output the last set of fitted parameters
    print(f"Last Fitted Parameters: A = {A_fit}, n = {n_fit}, t_merg = {t_merg_fit}")

    # Final plot of the fits with points
    fig_final_fit = plt.figure(figsize=(10, 6))
    plt.plot(t, omega_22, label='Original Data', color='orange', alpha=0.5)
    # plt.plot(t, fitting_model(t, A_fit, n_fit, t_merg_fit))
    # Plot each fitted interval and the corresponding fit
    plt.scatter(t[peaks], omega_22[peaks], color='red', s=10, label=f'Interval {i+1} Data' if i == 0 else "")
    plt.plot(t, fitting_model(t, *fitted_params[-1]), linestyle='--', color='blue', label=f'Interval {i+1} Fit' if i == 0 else "")

    plt.xlabel('Time [M]')
    plt.ylabel(r'$\omega_{22} [rad/M]$')
    plt.legend()
    plt.title('Fitting Intervals with Corresponding Fits')
    # plt.show()

    return fitted_params[-1]
print('Pericenters')
w_fit_peris = fitting_model(t, *fit('Pericenters'))
print('Apocenters')
w_fit_apos = fitting_model(t, *fit('Apocenters'))

def calc_eccentricity(w_fit_peris, w_fit_apos):
    ecc_w = (np.sqrt(w_fit_peris) - np.sqrt(w_fit_apos)) / (np.sqrt(w_fit_peris) + np.sqrt(w_fit_apos))
    return ecc_w

fig_ecc = plt.figure()
plt.plot(t, calc_eccentricity(w_fit_peris, w_fit_apos))
plt.plot()
plt.show()