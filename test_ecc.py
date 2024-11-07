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

def phase_from_polarizations(h_plus, h_cross, remove_start_phase=True):
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
    # print(math.copysign(1, h_plus.data[0]), math.copysign(1, h_cross.data[0]))
    # if math.copysign(1, h_cross.data[0]) == -1:
    #     # Calculate difference between consecutive points
    #     print(h_cross.data[:5])
    #     old_sign = math.copysign(1, h_cross.data[0])
    #     for i in range(1, len(h_cross.data)):
    #         new_sign = math.copysign(1, h_cross.data[i])
    #         if old_sign != new_sign:
    #             h_cross.data[:i] = h_cross.data[:i] * -1
    #             print('adjust dat cross', i)
    #             break
    #         print('i', i)


    # if math.copysign(1, h_plus.data[0]) == -1:
    #     print(h_plus.data[:5])
    #     old_sign = math.copysign(1, h_plus.data[0])
    #     for i in range(1, len(h_plus.data)):
    #         # print(i)
    #         new_sign = math.copysign(1, h_plus.data[i])
    #         if old_sign != new_sign:
    #             h_plus.data[:i] = h_plus.data[:i] * -1
    #             print('adjust dat plus', i)
    #             break
    #         print('i', i)

        

    p = np.unwrap(np.arctan2(h_cross.data, h_plus.data)).astype(
        real_same_precision_as(h_plus))

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

ecc_list_out = np.linspace(0.01, 0.2, num=300).round(4)
ecc_list_in = np.linspace(0.01, 0.2, num=150).round(4)

try:
    phases_in = np.load('phases_in.npz')['phase']
    phases_out = np.load('phases_out.npz')['phase']
    hp_TS_out = np.load('phases_out.npz')['hp']
    hc_TS_out = np.load('phases_out.npz')['hc']
    hp_TS_in = np.load('phases_in.npz')['hp']
    hc_TS_in = np.load('phases_in.npz')['hc']
    TS = np.load('phases_in.npz')['TS']
    print('phases loaded')
except:


    total_mass = 50
    mass_ratio = 1
    freqmin = 18
    DeltaT = 1./2048. # 
    lalDict = lal.CreateDict() # 


    start = timer()

    mass1 = total_mass / (1 + mass_ratio)
    mass2 = total_mass - mass1

    phases_in=np.zeros((len(ecc_list_in), 3000))
    hp_TS_in = np.zeros((len(ecc_list_in), 3000))
    hc_TS_in= np.zeros((len(ecc_list_in), 3000))

    hp0, hc0 = lalsim.SimInspiralTD(
                m1=lal.MSUN_SI*mass1, m2=lal.MSUN_SI*mass2,
                S1x=0., S1y=0., S1z=0., S2x=0., S2y=0., S2z=0.,
                distance=400.*1e6*lal.PC_SI, inclination=0.,
                phiRef=0., longAscNodes=0, eccentricity=1e-10, meanPerAno=0.,
                deltaT=DeltaT, f_min=freqmin, f_ref=freqmin,
                LALparams=lalDict, approximant=lalsim.EccentricTD
            )
    
    hp0 = types.timeseries.TimeSeries(hp0.data.data, delta_t=hp0.deltaT)[:3000]
    hc0 = types.timeseries.TimeSeries(hc0.data.data, delta_t=hc0.deltaT)[:3000]
    phase0 = phase_from_polarizations(hp0, hc0)[:3000]
    
        

    for i, eccmin in enumerate(ecc_list_in):
        hp, hc = lalsim.SimInspiralTD(
            m1=lal.MSUN_SI*mass1, m2=lal.MSUN_SI*mass2,
            S1x=0., S1y=0., S1z=0., S2x=0., S2y=0., S2z=0.,
            distance=400.*1e6*lal.PC_SI, inclination=0.,
            phiRef=0., longAscNodes=0, eccentricity=eccmin, meanPerAno=0.,
            deltaT=DeltaT, f_min=freqmin, f_ref=freqmin,
            LALparams=lalDict, approximant=lalsim.EccentricTD
        )
        hp = types.timeseries.TimeSeries(hp.data.data, delta_t=hp.deltaT)
        hc = types.timeseries.TimeSeries(hc.data.data, delta_t=hc.deltaT)
        hp_TS_in[i] = hp[:3000]  # plus polarisation timeseries
        hc_TS_in[i] = hc[:3000] # cross polarisation timeseries
        TS = -hp.sample_times[::-1][:3000] # Timeseries 
        phase = phase_from_polarizations(hp, hc)
        phases_in[i] = phase[:3000] - phase0
        print(eccmin)

    np.savez('phases_in.npz', phase=phases_in, hp=hp_TS_in, hc=hc_TS_in, TS=TS)

    phases_out=np.zeros((len(ecc_list_out), 3000))
    hp_TS_out = np.zeros((len(ecc_list_out), 3000))
    hc_TS_out= np.zeros((len(ecc_list_out), 3000))

    for i, eccmin in enumerate(ecc_list_out):
        hp, hc = lalsim.SimInspiralTD(
            m1=lal.MSUN_SI*mass1, m2=lal.MSUN_SI*mass2,
            S1x=0., S1y=0., S1z=0., S2x=0., S2y=0., S2z=0.,
            distance=400.*1e6*lal.PC_SI, inclination=0.,
            phiRef=0., longAscNodes=0, eccentricity=eccmin, meanPerAno=0.,
            deltaT=DeltaT, f_min=freqmin, f_ref=freqmin,
            LALparams=lalDict, approximant=lalsim.EccentricTD
        )
        hp = types.timeseries.TimeSeries(hp.data.data, delta_t=hp.deltaT)
        hc = types.timeseries.TimeSeries(hc.data.data, delta_t=hc.deltaT)
        hp_TS_out[i] = hp[:3000]  # plus polarisation timeseries
        hc_TS_out[i] = hc[:3000] # cross polarisation timeseries
        TS = -hp.sample_times[::-1][:3000] # Timeseries 
        phase = phase_from_polarizations(hp, hc)
        phases_out[i] = phase[:3000] - phase0
        print(eccmin)

    np.savez('phases_out.npz', phase=phases_out, hp=hp_TS_out, hc=hc_TS_out, TS=TS)
    # fig, axs = plt.subplots(4, sharex=True)

# List of values you want to find the indices for
values_to_find = [0.02, 0.1, 0.13]
print(ecc_list_in)

# Use np.isin to find the indices where arr has any of the values in values_to_find
indices_in = np.where(np.isin(ecc_list_in, values_to_find))[0]
indices_out = np.where(np.isin(ecc_list_out, values_to_find))[0]

# print(phases_in[:, 0].shape, phases_out[:, 0].shape)

fig__ = plt.figure()
# print(phases_in[:3], phases_out[:3])
# print(phases_in[:, 1], phases_out[:, 1])

for i in range(len(phases_in[:5])):
    plt.plot(TS, phases_in[i], label=i)
# plt.plot(ecc_list_out, phases_out[:, 2000], label='out')
plt.legend()
plt.show()