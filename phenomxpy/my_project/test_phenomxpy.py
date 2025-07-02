import phenomxpy.phenomt as phenomt
import lal
import matplotlib.pyplot as plt
import lalsimulation as lalsim
from pycbc import types, waveform
from phenomxpy.internals import pAmp, pPhase, pWFHM, Cache
import numpy as np

plt.switch_backend('WebAgg')

eccmin=0.2 

total_mass = 50
mass_ratio = 1
DeltaT = 1./2048.
freqmin = 20
distance=400.*1e6*lal.PC_SI
inclination=0.
phiRef=0.

mass1 = total_mass / (1 + mass_ratio)
mass2 = total_mass - mass1

sim_hp, sim_hc = lalsim.SimInspiralTD(
            m1=lal.MSUN_SI*mass1, m2=lal.MSUN_SI*mass2,
            S1x=0., S1y=0., S1z=0., S2x=0., S2y=0., S2z=0.,
            distance=distance, inclination=inclination,
            phiRef=phiRef, longAscNodes=0, eccentricity=eccmin, meanPerAno=0.,
            deltaT=DeltaT, f_min=freqmin, f_ref=freqmin,
            LALparams=lal.CreateDict() , approximant=lalsim.EccentricTD
        )

sim_hp_TS = types.timeseries.TimeSeries(sim_hp.data.data, delta_t=sim_hp.deltaT)  # plus polarisation timeseries
sim_hc_TS = types.timeseries.TimeSeries(sim_hc.data.data, delta_t=sim_hc.deltaT)  # cross polarisation timeseries
sim_TS = -sim_hp_TS.sample_times[::-1] / (lal.MTSUN_SI * total_mass) 
 # Timeseries 
times= sim_TS.numpy()

phen0 = phenomt.PhenomT(mode=[2,2], freqmin=freqmin, inclination=inclination, phiRef=phiRef, times=times, distance=400., total_mass=total_mass)
phen0.compute_polarizations(inclination=inclination, phiRef=phiRef, times=times, distance=400., total_mass=total_mass, freqmin=freqmin)
# phen0_amp = phen0.pAmp.inspiral_ansatz(times=times)
# phen0_phase = phen0.pPhase.inspiral_ansatz(times=times)

phen = phenomt.PhenomTE(mode=[2,2], eccentricity=eccmin, inclination=inclination, phiRef=phiRef, times=times, distance=400., total_mass=total_mass, f_ref=freqmin, f_lower=freqmin)
phen.compute_polarizations(inclination=inclination, phiRef=phiRef, times=times, distance=400., total_mass=total_mass, f_ref=freqmin, f_lower=freqmin)
phen_amp = phen.pAmp.inspiral_ansatz(times=times)
phen_phase = phen.pPhase.inspiral_ansatz(times=times)
# phen_insp = phen.inspiral_ansatz(times=None)
# phen_phase_res = phen_phase - phen0_phase
# phen_amp_res = phen_amp - phen0_amp

print(phen.pAmp.inspiral_ansatz(times=times), phen.pPhase.inspiral_ansatz(times=times))
# print(f'hp: {phen.hp} \n hc: {phen.hc} \n times: {phen.times}')

wf = plt.figure()
# plt.plot(phen.times, phen_insp)
# plt.plot(phen.times, phen.pPhase, label=' phase')
# plt.plot(times, phen_amp, label=' amp')
# plt.plot(times, phen_phase_res, label='phase')
# plt.plot(times, phen0_phase, label=' phase 0')
plt.plot(times, phen.hp, label='hp phenomxpy')
plt.plot(sim_TS, sim_hp_TS, label='hp sim')
# plt.plot(phen.times, phen.hc, label='hc')
plt.xlabel('times (?)')
plt.ylabel('strain')
plt.legend()
plt.show()

wf.savefig('test_generation.png')