import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orth

plt.switch_backend('WebAgg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orth

# Function to compute the waveform from plus and cross polarizations
def compute_waveform(h_plus, h_cross):
    return h_plus + 1j * h_cross

# Function to normalize a waveform
def normalize_waveform(waveform):
    norm = np.linalg.norm(waveform)
    return waveform / norm if norm != 0 else waveform

# Function to compute the inner product between two waveforms
def inner_product(h, e):
    return np.sum(h * np.conj(e))

# Function to compute the projection coefficient
def projection_coefficient(h, e):
    return inner_product(h, e) / inner_product(e, e)

# Function to compute the projection of a normalized waveform onto a normalized reduced basis
def compute_projection(waveform, reduced_basis):
    projection = np.zeros_like(waveform, dtype=complex)
    for e in reduced_basis:
        c_i = projection_coefficient(waveform, e)
        projection += c_i * e
    return projection

# Function to compute the projection error
def compute_projection_error(waveform, reduced_basis):
    projection = compute_projection(waveform, reduced_basis)
    error = np.linalg.norm(waveform - projection) / np.linalg.norm(waveform)
    return error

# Function to orthogonalize and normalize the reduced basis
def orthogonalize_normalize(basis):
    orthogonal_basis = orth(np.array(basis).T).T
    normalized_basis = [normalize_waveform(e) for e in orthogonal_basis]
    return normalized_basis

# Empirical Interpolation Method (EIM)
def empirical_interpolation_method(reduced_basis):
    empirical_nodes = []
    interpolation_basis = []

    for i, e_i in enumerate(reduced_basis):
        # Initialize residual
        residual = e_i

        # Orthogonalize with respect to previous interpolation_basis
        for j, b_j in enumerate(interpolation_basis):
            residual -= np.dot(b_j, e_i) * b_j

        # Select empirical node as the point of maximum residual
        T_i = np.argmax(np.abs(residual))
        empirical_nodes.append(T_i)
        interpolation_basis.append(normalize_waveform(residual))

    return empirical_nodes, interpolation_basis

# Function to fit amplitude and phase at empirical nodes
def fit_amplitude_phase(waveforms, empirical_nodes, parameter_space):
    amplitudes = []
    phases = []

    for node in empirical_nodes:
        amp_node = []
        phase_node = []

        for waveform in waveforms:
            amp = np.abs(waveform[node])
            phase = np.angle(waveform[node])
            amp_node.append(amp)
            phase_node.append(phase)
        
        amplitudes.append(amp_node)
        phases.append(phase_node)
    
    # Fit the amplitudes and phases with polynomials
    amp_fits = [np.polyfit(parameter_space, amp, deg=5) for amp in amplitudes]
    phase_fits = [np.polyfit(parameter_space, phase, deg=5) for phase in phases]

    return amp_fits, phase_fits

# Function to evaluate the surrogate model
def evaluate_surrogate(parameter, empirical_nodes, amp_fits, phase_fits, interpolation_basis, t):
    amp_evals = [np.polyval(fit, parameter) for fit in amp_fits]
    phase_evals = [np.polyval(fit, parameter) for fit in phase_fits]

    surrogate_waveform = np.zeros_like(t, dtype=complex)
    for i, basis in enumerate(interpolation_basis):
        surrogate_waveform += amp_evals[i] * np.exp(1j * phase_evals[i]) * basis
    
    return surrogate_waveform

# Function to plot waveforms with empirical nodes
def plot_waveforms_with_empirical_nodes(waveforms, empirical_nodes, t):
    plt.figure(figsize=(14, 8))
    
    for i, waveform in enumerate(waveforms):
        plt.plot(t, np.real(waveform), label=f'Waveform {i+1}')
    
    # Highlight empirical nodes
    for node in empirical_nodes:
        plt.axvline(x=t[node], color='k', linestyle='--', label=f'Empirical Node {node}')

    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Waveforms with Empirical Nodes')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to calculate greedy error
def calculate_greedy_error(waveforms, reduced_basis):
    errors = []
    for wf in waveforms:
        projection = compute_projection(wf, reduced_basis)
        error = np.linalg.norm(wf - projection) / np.linalg.norm(wf)
        errors.append(error)
    return np.mean(errors)

# Function to calculate empirical interpolation error
def calculate_eim_error(waveforms, empirical_nodes, interpolation_basis):
    errors = []
    for wf in waveforms:
        empirical_interp = sum([amp * np.exp(1j * phase) * basis for amp, phase, basis in zip(
            [np.abs(wf[node]) for node in empirical_nodes],
            [np.angle(wf[node]) for node in empirical_nodes],
            interpolation_basis)])
        error = np.linalg.norm(wf - empirical_interp) / np.linalg.norm(wf)
        errors.append(error)
    return np.mean(errors)

# Function to calculate surrogate error
def calculate_surrogate_error(waveforms, empirical_nodes, amp_fits, phase_fits, interpolation_basis, parameter_space, t):
    errors = []
    for wf, param in zip(waveforms, parameter_space):
        surrogate_wf = evaluate_surrogate(param, empirical_nodes, amp_fits, phase_fits, interpolation_basis, t)
        error = np.linalg.norm(wf - surrogate_wf) / np.linalg.norm(wf)
        errors.append(error)
    return np.mean(errors)

# Function to compute and plot errors
def compute_and_plot_errors(waveforms, reduced_basis, empirical_nodes, interpolation_basis, amp_fits, phase_fits, parameter_space, t):
    greedy_errors = []
    empirical_errors = []
    surrogate_errors = []

    for i in range(1, len(reduced_basis) + 1):
        greedy_errors.append(calculate_greedy_error(waveforms, reduced_basis[:i]))
        empirical_errors.append(calculate_eim_error(waveforms, empirical_nodes[:i], interpolation_basis[:i]))
        surrogate_errors.append(calculate_surrogate_error(waveforms, empirical_nodes[:i], amp_fits[:i], phase_fits[:i], interpolation_basis[:i], parameter_space, t))

    # Plot errors in subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))

    axs[0].plot(range(1, len(greedy_errors) + 1), greedy_errors, label='Greedy Error', marker='o')
    axs[0].set_xlabel('Number of Basis Functions')
    axs[0].set_ylabel('Error')
    axs[0].set_title('Greedy Error')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(range(1, len(empirical_errors) + 1), empirical_errors, label='EIM Error', marker='o')
    axs[1].set_xlabel('Number of Basis Functions')
    axs[1].set_ylabel('Error')
    axs[1].set_title('Empirical Interpolation Error')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(range(1, len(surrogate_errors) + 1), surrogate_errors, label='Surrogate Error', marker='o')
    axs[2].set_xlabel('Number of Basis Functions')
    axs[2].set_ylabel('Error')
    axs[2].set_title('Surrogate Error')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    return greedy_errors, empirical_errors, surrogate_errors

# Parameters and corresponding waveforms
parameter_space = np.linspace(0, 1, 10)  # Example parameter space
t = np.linspace(0, 1, 100)  # Time array
h_plus_waveforms = [np.sin(2 * np.pi * t + param) for param in parameter_space]  # Example h_plus waveforms
h_cross_waveforms = [np.cos(2 * np.pi * t + param) for param in parameter_space]  # Example h_cross waveforms

# Combine h_plus and h_cross to form complex waveforms
waveforms = [compute_waveform(h_plus, h_cross) for h_plus, h_cross in zip(h_plus_waveforms, h_cross_waveforms)]

# Normalize the waveforms
normalized_waveforms = [normalize_waveform(waveform) for waveform in waveforms]

# Initialize the reduced basis
reduced_basis = []

# Greedy algorithm to select reduced basis waveforms
threshold = 1e-6  # Set the threshold low to ensure many empirical nodes
for waveform in normalized_waveforms:
    projection_error = compute_projection_error(normalize_waveform(waveform), reduced_basis)
    if projection_error > threshold:
        reduced_basis.append(waveform)
        # Orthogonalize and normalize the reduced basis
        reduced_basis = orthogonalize_normalize(reduced_basis)

# Ensure multiple empirical nodes are selected
if len(reduced_basis) < len(parameter_space):
    reduced_basis = normalized_waveforms  # Force the use of all waveforms to select more nodes

# Apply the empirical interpolation method to the reduced basis
empirical_nodes, interpolation_basis = empirical_interpolation_method(reduced_basis)

# Fit amplitude and phase at empirical nodes
amp_fits, phase_fits = fit_amplitude_phase(normalized_waveforms, empirical_nodes, parameter_space)

# Plot the waveforms and empirical nodes
plot_waveforms_with_empirical_nodes(normalized_waveforms, empirical_nodes, t)

# Example parameter to evaluate the surrogate model
parameter = 0.5
surrogate_waveform = evaluate_surrogate(parameter, empirical_nodes, amp_fits, phase_fits, interpolation_basis, t)

# Plot the surrogate waveform and the actual waveform for comparison
actual_waveform = normalized_waveforms[5]  # For parameter 0.5, index 5
plt.figure(figsize=(14, 8))
plt.plot(t, np.real(actual_waveform), label='Actual Waveform')
plt.plot(t, np.real(surrogate_waveform), label='Surrogate Waveform', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Actual vs Surrogate Waveform')
plt.legend()
plt.grid(True)
plt.show()

# Compute and plot errors
greedy_errors, empirical_errors, surrogate_errors = compute_and_plot_errors(
    normalized_waveforms, reduced_basis, empirical_nodes, interpolation_basis, amp_fits, phase_fits, parameter_space, t)
