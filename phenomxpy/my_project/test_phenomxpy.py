import phenomxpy.phenomt as phenomt
import lal
import matplotlib.pyplot as plt

phen = phenomt.PhenomT(mode=[2,2])
phen.compute_polarizations(inclination=0, phiRef=0, times=None, distance=400.*1e6*lal.PC_SI, total_mass=50)
print(f'hp: {phen.hp} \n hc: {phen.hc} \n times: {phen.times}')

wf = plt.figure()

plt.plot(phen.times, phen.hp, label='hp')
plt.plot(phen.times, phen.hc, label='hc')
plt.xlabel('times (?)')
plt.ylabel('strain')
plt.legend()

wf.savefig('test_generation.png')