from tcintegral import molecular_orbital
from yutility import orbitals
import matplotlib.pyplot as plt
import numpy as np
import os


j = os.path.join


overlaps = []
ds = []
for d in os.listdir(r"../test/fixtures/reactants/MeMe_ADF_distance"):
    print(j(r"../test/fixtures/reactants/MeMe_ADF_distance", d))

    orbs = orbitals.Orbitals(j(r"../test/fixtures/reactants/MeMe_ADF_distance", d, 'sp.results', 'adf.rkf'))

    ds.append(float(d))
    overlaps.append(orbs.sfos['Me1(SUMO)'] @ orbs.sfos['Me2(SOMO)'] * 100)
plt.plot(ds, overlaps, label='ADF')


ds = np.linspace(0, 6, 34)
overlaps = []
for d in ds:
    mo1 = molecular_orbital.get(r"../test/fixtures/reactants/methyl/sp.results/adf.rkf", 'SOMO', bs_file=r"../src/tcintegral/basis_sets/sto-6g.1.cp2k")
    mo1.translate([0, 0, d])
    mo2 = molecular_orbital.get(r"../test/fixtures/reactants/methyl/sp.results/adf.rkf", 'SOMO', bs_file=r"../src/tcintegral/basis_sets/sto-6g.1.cp2k")
    overlaps.append(mo1.overlap(mo2, method='numeric')*100)
plt.plot(ds, overlaps, label='Numeric (STO-6G)')

ds = np.linspace(0, 6, 34)
overlaps = []
for d in ds:
    print(d)
    mo1 = molecular_orbital.get(r"../test/fixtures/reactants/methyl/sp.results/adf.rkf", 'SOMO', bs_file=r"../src/tcintegral/basis_sets/sto-6g.1.cp2k")
    mo1.translate([0, 0, d])
    mo2 = molecular_orbital.get(r"../test/fixtures/reactants/methyl/sp.results/adf.rkf", 'SOMO', bs_file=r"../src/tcintegral/basis_sets/sto-6g.1.cp2k")
    overlaps.append(mo1.overlap(mo2)*100)
plt.plot(ds, overlaps, label='Exact (STO-6G)')

ds = np.linspace(0, 6, 34)
overlaps = []
for d in ds:
    print(d)
    mo1 = molecular_orbital.get(r"../test/fixtures/reactants/methyl/sp.results/adf.rkf", 'SOMO', bs_file=r"../src/tcintegral/basis_sets/cc-pvdz.1.cp2k")
    mo1.translate([0, 0, d])
    mo2 = molecular_orbital.get(r"../test/fixtures/reactants/methyl/sp.results/adf.rkf", 'SOMO', bs_file=r"../src/tcintegral/basis_sets/cc-pvdz.1.cp2k")
    overlaps.append(mo1.overlap(mo2)*100)
plt.plot(ds, overlaps, label='Exact (cc-PVDZ)')


plt.xlabel('Me - Me Distance (Angstrom)')
plt.ylabel(r'$\langle$ SOMO | SUMO $\rangle$ (%)')
plt.legend()

plt.savefig('Me_Me_distance.png')
