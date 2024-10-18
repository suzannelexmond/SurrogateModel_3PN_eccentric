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
import sys
import os
import math
from pycbc.types import TimeSeries, FrequencySeries, Array, float32, float64, complex_same_precision_as, real_same_precision_as

def phase_from_polarizations(h_plus, h_cross, TS, ecc, remove_start_phase=True):
    """Return gravitational wave phase

    Return the gravitation-wave phase from the h_plus and h_cross
    polarizations of the waveform. The returned phase is always
    positive and increasing with an initial phase of 0.

    Parameters
    ----------
    h_plus : TimeSeries
        An PyCBC TmeSeries vector that contains the plus polarization of the
        gravitational waveform.
    h_cross : TimeSeries
        A PyCBC TmeSeries vector that contains the cross polarization of the
        gravitational waveform.

    Returns
    -------
    GWPhase : TimeSeries
        A TimeSeries containing the gravitational wave phase.

    Examples
    --------s
    >>> from pycbc.waveform import get_td_waveform, phase_from_polarizations
    >>> hp, hc = get_td_waveform(approximant="TaylorT4", mass1=10, mass2=10,
                         f_lower=30, delta_t=1.0/4096)
    >>> phase = phase_from_polarizations(hp, hc)

    """

    p = np.unwrap(np.arctan2(h_cross.data, h_plus.data)).astype(
        real_same_precision_as(h_plus))
    

    
    axs[0].plot(TS[:4], h_cross.data[:4], label=f'{ecc} x', linewidth=0.6)
    axs[0].plot(TS[:4], h_plus.data[:4], label='+', linewidth=0.6)
    axs[5].plot(TS[:4], np.arctan2(h_cross.data, h_plus.data)[:4]/np.pi, label= 'arctan2', linewidth=0.6)


    if remove_start_phase:
        # print(p[0])
        p += -p[0]

    return TimeSeries(p, delta_t=h_plus.delta_t, epoch=h_plus.start_time,
        copy=False)

def frequency_from_polarizations(h_plus, h_cross, TS, ecc):
    """Return gravitational wave frequency

    Return the gravitation-wave frequency as a function of time
    from the h_plus and h_cross polarizations of the waveform.
    It is 1 bin shorter than the input vectors and the sample times
    are advanced half a bin.

    Parameters
    ----------
    h_plus : TimeSeries
        A PyCBC TimeSeries vector that contains the plus polarization of the
        gravitational waveform.
    h_cross : TimeSeries
        A PyCBC TimeSeries vector that contains the cross polarization of the
        gravitational waveform.

    Returns
    -------
    GWFrequency : TimeSeries
        A TimeSeries containing the gravitational wave frequency as a function
        of time.

    Examples
    --------
    >>> from pycbc.waveform import get_td_waveform, phase_from_polarizations
    >>> hp, hc = get_td_waveform(approximant="TaylorT4", mass1=10, mass2=10,
                         f_lower=30, delta_t=1.0/4096)
    >>> freq = frequency_from_polarizations(hp, hc)

    """
    phase = phase_from_polarizations(h_plus, h_cross, TS, ecc)
    freq = np.diff(phase) / ( 2 * lal.PI * phase.delta_t )
    start_time = phase.start_time + phase.delta_t / 2

    return TimeSeries(freq.astype(real_same_precision_as(h_plus)),
        delta_t=phase.delta_t, epoch=start_time)


plt.switch_backend('WebAgg')



ecc_list = np.linspace(0.13, 0.135, num=15).round(4)

total_mass = 50
mass_ratio = 1
freqmin = 18
DeltaT = 1./2048. # 
lalDict = lal.CreateDict() # 


start = timer()

mass1 = total_mass / (1 + mass_ratio)
mass2 = total_mass - mass1

phases=[]

# fig, axs = plt.subplots(4, sharex=True)
fig__, axs = plt.subplots(6, figsize=(8, 10))
for eccmin in ecc_list[3:6]:
    hp, hc = lalsim.SimInspiralTD(
        m1=lal.MSUN_SI*mass1, m2=lal.MSUN_SI*mass2,
        S1x=0., S1y=0., S1z=0., S2x=0., S2y=0., S2z=0.,
        distance=400.*1e6*lal.PC_SI, inclination=0.,
        phiRef=0., longAscNodes=0, eccentricity=eccmin, meanPerAno=0.,
        deltaT=DeltaT, f_min=freqmin, f_ref=freqmin,
        LALparams=lalDict, approximant=lalsim.EccentricTD
    )

    hp_TS = types.timeseries.TimeSeries(hp.data.data, delta_t=hp.deltaT)  # plus polarisation timeseries
    hc_TS = types.timeseries.TimeSeries(hc.data.data, delta_t=hc.deltaT)  # cross polarisation timeseries
    TS = -hp_TS.sample_times[::-1] # Timeseries 






    # phase = phase_from_polarizations(hp_TS, hc_TS, TS, remove_start_phase=True)
    print(eccmin)
    phase_remove = phase_from_polarizations(hp_TS, hc_TS, TS, eccmin, remove_start_phase=False)
    print('before', phase_remove[:3], hp_TS[:3], hc_TS.data[:3])
    phases.append(phase_remove)
    if math.copysign(1, hp_TS[0]) != math.copysign(1, hc_TS[0]):
        print(eccmin, 'check')
        phase_remove -= 2*np.pi
    print('after', phase_remove[:3], hp_TS[:3], hc_TS.data[:3])
    freq = frequency_from_polarizations(hp_TS, hc_TS, TS, eccmin)
    freq_time = -freq.sample_times[::-1]


    # axs[0].plot(TS, phase, label=f'e = {eccmin}', linewidth=0.6)

    axs[0].legend()
    axs[0].set_ylabel('$\phi$')
    axs[1].plot(freq_time, freq, linewidth=0.6)
    axs[1].set_ylabel('freq')

    axs[2].plot(TS, hp_TS, linewidth=0.6)
    axs[2].set_ylabel('$h_+$')
    axs[3].plot(TS, phase_remove, linewidth=0.6)
    axs[3].set_ylabel('remove $\phi$')
    # axs[2].set_xlim(-1.40, -1.38)

# print((phases[0][:len(phases[-1])] - phases[-1])/np.pi)
plt.show()