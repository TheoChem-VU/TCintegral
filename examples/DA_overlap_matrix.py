from tcintegral import Reactant, overlap_matrix
import numpy as np
from yutility import orbitals
from yviewer import viewer

rct1 = Reactant(r"../test/fixtures/reactants/ethene", bs_file=r"../src/tcintegral/basis_sets/sto-3g.1.cp2k", moleculename='Dienophile')
rct1.rotate(y=np.pi/2)
rct1.translate([3, 0, 0])
rct1.load_mos('HOMO-4', 'LUMO+4')

rct2 = Reactant(r"../test/fixtures/reactants/butadiene", bs_file=r"../src/tcintegral/basis_sets/sto-3g.1.cp2k", moleculename='Diene')
rct2.load_mos('HOMO-4', 'LUMO+4')

viewer.show(rct1.mol + rct2.mol)

shc = orbitals.plot_property(rct2.mos, rct1.mos, overlap_matrix(rct2, rct1, method='numeric'), figsize=(5, 5), title='S$_{pred}$')
shc.savefig('DA_overlaps.png')
