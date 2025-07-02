import numpy as np
import matplotlib.pyplot as plt
import lalsimulation as ls
import lal
from pycbc import waveform, types
from phenomxpy.phenomt import PhenomTE, PhenomT
from scipy.constants import G, c, parsec
from phenomxpy.common import Waveform
from phenomxpy.utils import SecondtoMass, AmpSItoNR, AmpNRtoSI, m1ofq, m2ofq, HztoMf
import os
from scipy.signal import hilbert

plt.switch_backend('WebAgg')

def test_phenom(total_mass=50, mass_ratio=1, distance=500, f_min=20, f_ref=20, geometric_units=False):
    # Parameters
    m1 = m1ofq(mass_ratio, total_mass)
    m2 = m2ofq(mass_ratio, total_mass)
    S1z = 0.0
    S2z = 0.0
    # distance = 500.0  # Mpc
    incl = 0.0
    phi_ref = 0.0
    long_asc_nodes = 0.0
    ecc = 0.2
    deltaT = 1.0 / 5000.  # sec


    

    if geometric_units is True:
        # only works for phenomT
        # in geometric units, mass is in solar masses, distance in Mpc


        M_phys_ref = total_mass # for Hz to Mf conversion (otherwise error for f_min and f_ref)
        f_min_phys = f_min
        f_ref_phys = f_ref
        f_min = HztoMf(f_min_phys, M_phys_ref) # convert f_min to mass units
        f_ref = HztoMf(f_ref_phys, M_phys_ref) # convert f_ref to mass units
        deltaT = SecondtoMass(deltaT, M_phys_ref)  # convert deltaT to mass units

        total_mass = 1.0  # Mpc
        distance = 1.0  # Mpc
        m1 = m1ofq(mass_ratio, total_mass)
        m2 = m2ofq(mass_ratio, total_mass) 
        print(f_min_phys, f_ref_phys, f_min, f_ref, deltaT)


    ## common.Waveform
    # print(m1, m2, distance)
    parameters = {'mass1': m1, 
                'mass2':m2,             
                'distance':distance, # Mpc
                'longAscNodes':long_asc_nodes,
                'spin1x':0., 
                'spin1y':0., 
                'spin1z':S1z,
                'spin2x':0., 
                'spin2y':0., 
                'spin2z':S2z, 
                'delta_t_sec':deltaT,
                'inclination':incl, 
                'phiRef':phi_ref,  # LAL uses phiRef - pi/2
                'eccentricity':ecc,
                'mean_anomaly':0.,
                'mode_array':[[2,2]]
                }
    
    waveform_arguments={}
    waveform_arguments["f_lower"] = f_min # Hz
    waveform_arguments["f_ref"] = f_ref # Hz
    print(' f' , f_min, f_ref)

    # print(Waveform(parameters, rhs_eqs="pn_phenomT", return_py_object=True, approximant="IMRPhenomTHM", domain="TD", free_memory=True, prints=True,**waveform_arguments))
    hplus_phen, hcross_phen, times_phen, py = Waveform(parameters, rhs_eqs="pn_phenomT", return_py_object=True, approximant="IMRPhenomTHM", free_memory=True, prints=True,**waveform_arguments)
    # times_phen = times_phen / (lal.MTSUN_SI * total_mass) 
    return 0

    # M_phys = 30
    # distance = 500
    # hp_phys = AmpNRtoSI(hplus_phen, distance, M_phys)
    # hc_phys = AmpNRtoSI(hcross_phen, distance, M_phys)
    # Scaling factor
    # distance_m = distance * 1e6 * lal.PC_SI  # convert Mpc to meters
    # geom_length = ( G * total_mass * lal.MSUN_SI ) / c**2

    # scaling_factor = geom_length / distance_m  # dimensionless strain to SI units

    # h(t) in physical units
    # hplus_phen_geom = py.hp / scaling_factor
    # hcross_phen_geom = py.hc / scaling_factor

    # hp_phen_TS = types.timeseries.TimeSeries(hplus_phen, delta_t=deltaT)  # plus polarisation timeseries
    # hc_phen_TS = types.timeseries.TimeSeries(hcross_phen, delta_t=deltaT)  # cross polarisation timeseries

    # phen_phase = np.array(waveform.utils.phase_from_polarizations(py.hp, py.hc))
    # phen_amp = np.array(waveform.utils.amplitude_from_polarizations(hp_phen_TS, hc_phen_TS))
    # phen_amp = np.sqrt(py.hp**2 + py.hc**2)
    # phen_amp_geom = AmpSItoNR(phen_amp, distance, total_mass)
    # phen_phase = np.unwrap(np.arctan2(py.hc, py.hp))

    phen_amp = np.sqrt(hp_phys**2 + hc_phys**2)
    # phen_amp_geom = AmpSItoNR(phen_amp, distance, total_mass)
    phen_phase = np.unwrap(np.arctan2(hc_phys, hp_phys))

    # print(hplus_phen, type(hcross_phen), type(times_phen))


    # times = times + 150
    # print(times, len(times), len(phen.hp))

    fig_compare = plt.figure(figsize=(6, 3))
    # plt.plot(py.times, phen_phase_arc - np.pi, label='PhenomTE phase arc', alpha=0.7)
    # plt.plot(py.times, phen_phase, label='phenom phase')
    # plt.plot(t_LAL_ecc, phase_ecc, label='LAL ecc phase')
    plt.plot(py.times, phen_amp, label=' phenom amp')
    # plt.plot(t_LAL_ecc, amp_ecc, label='LAL ecc amp')
   
    # plt.plot(py.times, py.hp, label='PhenomTE $h_+$', alpha=0.7)
    # plt.plot(t_LAL_ecc, hp_LAL_ecc, label='LAL EccentricTD $h_+$', alpha=0.7)
    # plt.plot(py.times, py.hc, label='PhenomTE $h_x$', alpha=0.7)
    # plt.plot(t_LAL_ecc, hc_LAL_ecc, label='LAL EccentricTD $h_x$', alpha=0.7)

    # plt.plot(t_LAL_T4, hp_LAL_T4, label='LAL TaylorT3 $h_+$', alpha=0.7)
    # plt.plot(times, phen0.hp, label='PhenomT $h_+$', alpha=0.7)
    plt.xlabel("Time [M]")
    plt.ylabel("Strain")
    if geometric_units is True:
        plt.title(f"Geometric units: f_ref={f_ref}, ecc={ecc}")
    else:
        plt.title(f"SI units: q={mass_ratio}, M={total_mass}, r={distance}, f_ref={f_ref}, ecc={ecc}")
    plt.title(f"q={mass_ratio}, M={total_mass}, r={distance}, f_ref={f_ref}, ecc={ecc}")
    plt.legend()
    plt.grid(True)
    # plt.xlim(-4000, 0)
    # plt.ylim(-1.5e-21, 1.5e-21)
    plt.tight_layout()

    figname = f'hp_Phenom_q={mass_ratio}_M={total_mass}_r={distance}_fmin={f_min}_ecc={ecc}.png'
    path = '/home/suzanne/Python_scripts/SurrogateModel_3PN_eccentric/phenomxpy/my_project/'
    os.makedirs(path + ' Images/', exist_ok=True)
    fig_compare.savefig(path + 'Images/' + figname)


# test_phenom(m1=20, m2=50, distance=500)
test_phenom(mass_ratio=1, total_mass=30, distance=500, f_min=20, f_ref=20, geometric_units=True)
# test_phenom(mass_ratio=1, total_mass=30, distance=500, f_min=10, f_ref=10, geometric_units=False)
plt.show()