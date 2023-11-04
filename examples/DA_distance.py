from tcintegral import molecular_orbital
from yutility import orbitals
import matplotlib.pyplot as plt
import numpy as np
import os


j = os.path.join

ds = np.linspace(0, 6, 34)
overlaps = []
# mols = []
for d in ds:
    print(d)
    mo1 = molecular_orbital.get(r"../test/fixtures/reactants/ethene/sp.results/adf.rkf", 'HOMO', bs_file=r"../src/tcintegral/basis_sets/cc-pvdz.1.cp2k")
    mo1.rotate(y=np.pi/2)
    mo1.translate([d, 0, 1])
    mo2 = molecular_orbital.get(r"../test/fixtures/reactants/butadiene/go.results/adf.rkf", 'LUMO', bs_file=r"../src/tcintegral/basis_sets/cc-pvdz.1.cp2k")
    overlaps.append(mo1.overlap(mo2)*100)
plt.plot(ds, overlaps, label='Exact (cc-PVDZ)')
ds = np.linspace(0, 6, 34)

overlaps = []
for d in ds:
    print(d)
    mo1 = molecular_orbital.get(r"../test/fixtures/reactants/ethene/sp.results/adf.rkf", 'HOMO', bs_file=r"../src/tcintegral/basis_sets/sto-6g.1.cp2k")
    mo1.rotate(y=np.pi/2)
    mo1.translate([d, 0, 1])
    mo2 = molecular_orbital.get(r"../test/fixtures/reactants/butadiene/go.results/adf.rkf", 'LUMO', bs_file=r"../src/tcintegral/basis_sets/sto-6g.1.cp2k")
    overlaps.append(mo1.overlap(mo2)*100)
plt.plot(ds, overlaps, label='Exact (STO-6G)')

ds = np.linspace(0, 6, 34)
overlaps = []
for d in ds:
    print(d)
    mo1 = molecular_orbital.get(r"../test/fixtures/reactants/ethene/sp.results/adf.rkf", 'HOMO', bs_file=r"../src/tcintegral/basis_sets/sto-6g.1.cp2k")
    mo1.rotate(y=np.pi/2)
    mo1.translate([d, 0, 1])
    mo2 = molecular_orbital.get(r"../test/fixtures/reactants/butadiene/go.results/adf.rkf", 'LUMO', bs_file=r"../src/tcintegral/basis_sets/sto-6g.1.cp2k")
    overlaps.append(mo1.overlap(mo2, method='numeric')*100)
plt.plot(ds, overlaps, label='Numeric (STO-6G)')

plt.xlabel('Ethene - Butadiene Distance (Angstrom)')
plt.ylabel(r'$\langle$ HOMO | LUMO $\rangle$ (%)')
plt.legend()

plt.savefig('DA_distance.png')
