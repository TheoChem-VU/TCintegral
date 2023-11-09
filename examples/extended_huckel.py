from tcintegral import basis_set, MolecularOrbital
from TCutility import results
import numpy as np

# get a basis set from basis_sets
bs = basis_set.STO2G
    
# read a molecule to use
res = results.read(r"../test/fixtures/reactants/benzene")
mol = res.molecule.output

# get atomic orbitals for this molecule
# orb_to_get = ['1S', '2S', '3S', '1P:x', '1P:y', '1P:z', '2P:x', '2P:y', '2P:z']  # list of possible AO
orb_to_get = ['1P:z']  # list of possible AOs
aos = []
for atom in mol:
    for orb_name in orb_to_get:
        try:  # try to get the AO for this atom, it will fail if it does not exist
            aos.append(bs.get(atom.symbol, orb_name, atom.coords))
        except IndexError:
            pass

# build the Hamiltonian
ionization_potentials = {  # in Hartrees, obtained at the OLYP/QZ4P level in ADF
    'H(1S)': -0.238455,
    'C(1S)': -10.063764,
    'C(2S)': -0.506704,
    'C(1P:x)': -0.188651,
    'C(1P:y)': -0.188651,
    'C(1P:z)': -0.188651,
}

K = 1.75
H = np.zeros((len(aos), len(aos)))
# build diagonal elements
for i, ao in enumerate(aos):
    H[i, i] = ionization_potentials.get(ao.name)

# build off-diagonal elements
for i, ao1 in enumerate(aos):
    for j, ao2 in enumerate(aos[i+1:], start=i+1):
        H[i, j] = H[j, i] = K * ao1.overlap(ao2) * (H[i, i] + H[j, j]) / 2

# get MO energies and coeffs
energies, coefficients = np.linalg.eigh(H)
print([e*27.211324570273 for e in energies])
for mo_index in range(len(energies)):
    mo = MolecularOrbital(aos, coefficients[:, mo_index], mol)
    mo.screenshot(f'MO_{mo_index}.png')
