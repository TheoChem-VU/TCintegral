from tcintegral import Reactant, overlap_matrix
import numpy as np
from yutility import orbitals
from yviewer import viewer

# load dienophile
rct1 = Reactant(r"../test/fixtures/reactants/ethene", bs_file=r"../src/tcintegral/basis_sets/sto-3g.1.cp2k", moleculename='Dienophile')
# put the dienophile in its place
rct1.rotate(y=np.pi/2)
rct1.translate([3, 0, 0])
# load import MOs
rct1.load_mos('HOMO-4', 'LUMO+4')

# load diene
rct2 = Reactant(r"../test/fixtures/reactants/butadiene", bs_file=r"../src/tcintegral/basis_sets/sto-3g.1.cp2k", moleculename='Diene')
rct2.load_mos('HOMO-4', 'LUMO+4')

S = overlap_matrix(rct2, rct1, method='numeric')
shc = orbitals.plot_property(rct2.mos, rct1.mos, S, figsize=(5, 5), title='S$_{pred}$')
shc.savefig('DA_overlaps.png').close()

viewer.show(rct1.mol + rct2.mol)
