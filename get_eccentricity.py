import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

plt.switch_backend('WebAgg')

# Load the data
data = np.load('Frequency.npz')
print(list(data.keys()))
t = data['t'][:-1]  # Time array
omega_22 = data['frequency']  # Frequency array
N_orbits = 3 # Minimum orbits (3 free parameters {A, n, t_merg})

# Define the fitting model with an epsilon to avoid divide-by-zero issues
epsilon = 1e-10
# Add minus sign for invalid encounter in power
def fitting_model(t, A, n, t_merg):
    return A * np.sign(t_merg - t)* (abs(t_merg - t) + epsilon)**n

def get_initial_parameters():
    """ Initial fit over the entire frequency-domain, to determine starting guess for fit parameters. Also sets inital tL and tR"""
    peaks, _ = find_peaks(omega_22)
    # print(len(peaks))

    tL = t[peaks[2]]
    t_mid = t[peaks[2 + N_orbits]]
    tR = t[peaks[2 + 2*N_orbits]]

    # inital fit of first 10 orbits
    fit_params, _ = curve_fit(fitting_model, t, omega_22, p0=[0.1, -0.25, 0], maxfev=2000)

    return tL, t_mid, tR, fit_params




def find_peaks_of_interval(tL, tR, fit_params, peak_choice):
    interval_indices = (t >= tL) & (t <= tR)

    # Calculate the current fit
    omega_fit = fitting_model(t, *fit_params)

    # Calculate local maxima of the difference between the data and the fit
    if peak_choice == 'Pericenters':
        residual = omega_22 - omega_fit
    elif peak_choice == 'Apocenters':
        residual = -(omega_22 - omega_fit)

    local_peaks, _ = find_peaks(residual[interval_indices])

    return local_peaks


def fit(peak_choice='Pericenters', convergence_value=2e-2):

    # Initial parameters
    tL, t_mid, tR, fit_params = get_initial_parameters()

    interval_indices = (t >= tL) & (t <= tR)
    # # Initial parameters
    # tL = t[0]
    # N_orbits = 7  # Initial number of orbits to consider
    converged = False

    # Store fitted parameters and points for each interval
    fitted_params = [fit_params]
    fitted_intervals = [(t[interval_indices], omega_22[interval_indices])]

################################ loop for every fit of new interval
    local_peaks = find_peaks_of_interval(tL, tR, fit_params, peak_choice)

    peaks_found_fig = plt.figure()

    peaks_found = False
    shift_count = 0

    plt.plot(t[interval_indices], omega_22[interval_indices], label='old interval')
    plt.scatter(t[interval_indices][local_peaks], omega_22[interval_indices][local_peaks], label='old peaks')

    while peaks_found is not True:
        shift_count += 1

        

        # Move interval with half an orbit to search for next peak
        tR = tR + (t[local_peaks[1]] - t[local_peaks[0]])/2
        interval_indices = (t >= tL) & (t <= tR)

        # Determine new peaks of interval
        local_peaks = find_peaks_of_interval(tL, tR, fit_params, peak_choice)

        # Check if interval has reached next peak
        if len(local_peaks) > 2*N_orbits:
            peaks_found = True
            tL = interval_indices[local_peaks[0]]
            t_mid = interval_indices[local_peaks[0]]

            interval_indices = (t >= tL) & (t <= tR)
            fitted_intervals.append(interval_indices)

    plt.plot(t[interval_indices], omega_22[interval_indices], label='new interval')
    plt.scatter(t[interval_indices][local_peaks], omega_22[interval_indices][local_peaks], label='new peaks')
    plt.legend()
    plt.show()

    """ Local peaks does not match the shift of tL. At the same time shift tL and tR and check for same points on t[local_peaks] for setting True"""
#     while len(local_peaks) != (2*N_orbits + 1):

#         print(t[interval_indices], omega_22[interval_indices], label='interval')
        
        
#         if len(local_peaks) < (2*N_orbits + 1):
#             tL = ( t[local_peaks[0]] + t[local_peaks[1]] ) / 2
#             t_mid = ( t[local_peaks[N_orbits]] + t[local_peaks[2 * N_orbits]] ) / 2
#             tR = t[local_peaks[2 * N_orbits]] + 1.5 * ( t[local_peaks[2 * N_orbits]] - t[local_peaks[0]])


#         local_peaks = find_peaks_of_interval(tL, tR, fit_params)
#         interval_indices = (t >= tL) & (t <= tR)
#         shift_count += 1

#         plt.legend()
#         plt.xlabel('time [M]')
#         plt.ylabel('Frequency [Hz]')
#         plt.show()

#         print(f'local peaks found after {shift_count} shifts')
#         print(f'new interval: [{tL}, {tR}]')

#         tL = tL + local_peaks[1]
#         tR = tR + local_peaks[1]
#         t_mid = t_mid + local_peaks[1]

#         # Refit with the new interval
#         fit_params, _ = curve_fit(fitting_model, t[interval_indices][local_peaks], omega_22[interval_indices][local_peaks], p0=fit_params, maxfev=2000)
#         fitted_params.append(fit_params)

#         # error = abs(fitting_model(t, *params)[peaks] - omega_22[peaks])

#         # if all (err < convergence_value for err in error): 
#         #     print("Convergence criterion met")
#         #     converged = True
#         #     break

#         A_fit, n_fit, t_merg_fit = fit_params

#         # Fit for new interval
#         omega_fit = fitting_model(t, A_fit, n_fit, t_merg_fit)
#         # # Find pericenters and apocenters
#         # if peak_choice == 'Pericenters':
#         #     peaks, _ = find_peaks(omega_22)
#         # elif peak_choice == 'Apocenters':
#         #     peaks, _ = find_peaks(-omega_22)

#         # Perform the initial fit for the first guess, fitted to omega_22
#         # params, _ = curve_fit(fitting_model, t, omega_22, p0=[0.1, -0.25, 0], maxfev=2000)
#         # A_fit, n_fit, t_merg_fit = params
        
        
fit()

# #         # Step 2: Iterate through the waveform, adjusting the interval dynamically
#         count_iteration = 0

#         print(len(peaks) - N_orbits)
#         print('last t: ', t[-1])
    
#     while not converged:
#         for i in range(2, len(peaks) - N_orbits - 1): # Skip first 2 and last peak for better fit
#             # Define the interval [tL, tR] dynamically
#             print('length: ', len(peaks) - N_orbits, peaks[i + N_orbits - 1])
#             tL = t[peaks[i]]
#             tR = t[peaks[i + N_orbits - 1]]
#             t_hat = t[peaks[i + N_orbits//2]]

#             # Store the points within the interval
#             interval_indices = (t >= tL) & (t <= tR)
#             fitted_intervals.append((t[interval_indices], omega_22[interval_indices]))
            
#             # Check if it reached the last interval 
#             print(t[peaks[-1]])
#             if tL >= t[peaks][len(peaks) - N_orbits - 1]:
#                 print("Interval has reached the last peak, stopping iteration.")
#                 converged = True
#                 break

#             # print('interval: ', t[interval_indices], tL, tR)

#             fig_interval = plt.figure()
#             plt.plot(t, omega_22)
#             plt.plot(t[interval_indices], omega_22[interval_indices])
#             plt.scatter(t[peaks], omega_22[peaks])
#             plt.scatter(tL, omega_22[peaks[i]], label='L')
#             plt.scatter(tR, omega_22[peaks[i + N_orbits - 1]], label='R')
#             plt.scatter(t[peaks][len(peaks) - N_orbits - 1], omega_22[peaks][len(peaks) - N_orbits - 1], label='final')
#             plt.show()

#             # How many cycles to convergence 
#             count_iteration += 1

#             # Calculate the current fit
#             omega_fit = fitting_model(t, A_fit, n_fit, t_merg_fit)

#             # Calculate local maxima of the difference between the data and the fit
#             if peak_choice == 'Pericenters':
#                 residual = omega_22 - omega_fit
#             elif peak_choice == 'Apocenters':
#                 residual = -(omega_22 - omega_fit)

#             local_peaks, _ = find_peaks(residual[interval_indices])

#             # Adjust the interval based on the new peaks
#             num_extrema = len(local_peaks)
#             if num_extrema > N_orbits:
#                 tR = t[local_peaks[N_orbits-1]]
#             elif num_extrema < N_orbits:
#                 tL = t[local_peaks[0]]
#             else:
#                 converged = True

#             # Refit with the new interval
#             params, _ = curve_fit(fitting_model, t[interval_indices][local_peaks], omega_22[interval_indices][local_peaks], p0=params, maxfev=2000)
#             fitted_params.append(params)

#             error = abs(fitting_model(t, *params)[peaks] - omega_22[peaks])

#             if all (err < convergence_value for err in error): 
#                 print("Convergence criterion met")
#                 converged = True
#                 break

#             A_fit, n_fit, t_merg_fit = params

#             # Fit for new interval
#             omega_fit = fitting_model(t, A_fit, n_fit, t_merg_fit)


#     print(count_iteration, ' Cycles till convergence', '\n fit:', A_fit, n_fit, t_merg_fit)

#     # Final plot of the fits with points
#     fig_fits = plt.figure(figsize=(10, 6))
#     plt.plot(t, omega_22, label='Original Data', color='orange', alpha=0.5)
#     # plt.plot(t, fitting_model(t, A_fit, n_fit, t_merg_fit))
#     # Plot each fitted interval and the corresponding fit
#     for i, (params, (t_interval, omega_interval)) in enumerate(zip(fitted_params, fitted_intervals)):
#         plt.scatter(t[peaks], omega_22[peaks], color='red', s=10, label=f'Interval {i+1} Data' if i == 0 else "")
#         plt.plot(t_interval, fitting_model(t_interval, *params), linestyle='--', color='blue', label=f'Interval {i+1} Fit' if i == 0 else "")

#     plt.xlabel('Time [M]')
#     plt.ylabel(r'$\omega_{22} [rad/M]$')
#     plt.legend()
#     plt.title('Fitting Intervals with Corresponding Fits')
#     # plt.show()

#     # Output the last set of fitted parameters
#     print(f"Last Fitted Parameters: A = {A_fit}, n = {n_fit}, t_merg = {t_merg_fit}")

#     # Final plot of the fits with points
#     fig_final_fit = plt.figure(figsize=(10, 6))
#     plt.plot(t, omega_22, label='Original Data', color='orange', alpha=0.5)
#     # plt.plot(t, fitting_model(t, A_fit, n_fit, t_merg_fit))
#     # Plot each fitted interval and the corresponding fit
#     plt.scatter(t[peaks], omega_22[peaks], color='red', s=10, label=f'Interval {i+1} Data' if i == 0 else "")
#     plt.plot(t, fitting_model(t, *fitted_params[-1]), linestyle='--', color='blue', label=f'Interval {i+1} Fit' if i == 0 else "")

#     plt.xlabel('Time [M]')
#     plt.ylabel(r'$\omega_{22} [rad/M]$')
#     plt.legend()
#     plt.title('Fitting Intervals with Corresponding Fits')
#     # plt.show()

#     return fitted_params[-1]
# print('Pericenters')
# w_fit_peris = fitting_model(t, *fit('Pericenters'))
# print('Apocenters')
# w_fit_apos = fitting_model(t, *fit('Apocenters'))

# def calc_eccentricity(w_fit_peris, w_fit_apos):
#     # eccentricity with wrong post-newtonian limits
#     ecc_w = (np.sqrt(w_fit_peris) - np.sqrt(w_fit_apos)) / (np.sqrt(w_fit_peris) + np.sqrt(w_fit_apos))
    
#     # correct post-newtonian limit
#     psi = np.arctan((1 - ecc_w**2)/(2*ecc_w))
#     ecc_gw = np.cos(psi/3) - np.sqrt(3)*np.sin(psi/3)
    
#     return ecc_gw

# fig_ecc = plt.figure()
# plt.plot(t, calc_eccentricity(w_fit_peris, w_fit_apos))
# plt.plot()
# plt.show()
