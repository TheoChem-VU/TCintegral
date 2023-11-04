from yutility import orbitals, make_gif, timer
from TCutility import results
import basis_set
from scm import plams
import numpy as np
from yviewer import viewer
from math import sqrt, cos, sin, pi
import matplotlib.pyplot as plt


def get_rotmat(x=0, y=0, z=0):
    Rx = np.array([[1, 0, 0],
                   [0, cos(x), -sin(x)],
                   [0, sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, sin(y)],
                   [0, 1, 0],
                   [-sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), -sin(z), 0],
                   [sin(z), cos(z), 0],
                   [0, 0, 1]])
    return Rx @ Ry @ Rz



def get(rkf_file, orb_name, bs_file=r"basis_sets/cc-pvdz.1.cp2k"):
    bs = basis_set.BasisSet(bs_file)
    orbs = orbitals.Orbitals(rkf_file)
    xyz = np.array(orbs.reader.read('Geometry', 'xyz')).reshape(-1, 3) * 0.529177
    nats = xyz.shape[0]
    atom_type_index = orbs.reader.read('Geometry', 'fragment and atomtype index')[nats:]
    ats = [orbs.reader.read('Geometry', 'atomtype').split()[i-1] for i in atom_type_index]
    atom_unique_names = []
    atom_unique_pos = {}
    for i, (at, x) in enumerate(zip(ats, xyz)):
        is_unique = len([at_ for at_ in ats if at_ == at]) == 1
        if is_unique:
            name = at
        else:
            index = i+1
            name = f'{at}:{index}'

        atom_unique_names.append(name)
        atom_unique_pos[name] = x

    mol = plams.Molecule()
    for at, x in zip(ats, xyz):
        mol.add_atom(plams.Atom(symbol=at, coords=x))
    mol.guess_bonds()

    # viewer.show(mol)
    orb = orbs.mos[orb_name]
    aos = []
    ao_coeffs = {}
    for sfo, coeff in zip(orbs.sfos.sfos, orb.coeffs):
        if (sfo.fragment, sfo.name) in bs:
            ao = bs.get(sfo.fragment, sfo.name, atom_unique_pos[sfo.fragment_unique_name])
            ao.fragment_unique_name = sfo.fragment_unique_name
            ao_coeffs[ao] = coeff

    mo = MolecularOrbital(ao_coeffs.keys(), ao_coeffs.values(), mol)
    mo.energy = orb.energy
    mo.name = repr(orb)
    mo.spin = orb.spin
    mo.kfpath = orb.kfpath
    mo.occupation = orb.occupation
    mo.occupied = mo.occupation > 0
    return mo


class MolecularOrbital:
    def __init__(self, basis_functions, coefficients, molecule):
        self.basis_functions = list(basis_functions)
        self.coefficients = list(coefficients)
        self.molecule = molecule
        self._norm = None

    def __call__(self, r):
        r = np.atleast_2d(r)
        wf = np.zeros(r.shape[0])
        for f, coeff in zip(self.basis_functions, self.coefficients):
            wf += f(r.T) * coeff
        return wf / sqrt(sum(wf**2))

    def get_cub(self, p=None, cutoff=[.4, .45]):
        if p is None:
            x = np.linspace(-6, 6, 80).reshape(-1, 1)
            y = np.linspace(-6, 6, 80).reshape(-1, 1)
            z = np.linspace(-6, 6, 80).reshape(-1, 1)

            p = np.meshgrid(x, y, z)
            p = [r_.flatten() for r_ in p]
            p = np.vstack(p).T
        wf = self(p)
        wf_abs = abs(wf)/np.max(abs(wf))
        idx = np.where(np.logical_and(wf_abs > cutoff[0], wf_abs < cutoff[1]))[0]
        COL1 = np.array((255, 0, 0)) if self.occupied else np.array((255, 165, 0))
        COL2 = np.array((0, 0, 255)) if self.occupied else np.array((0, 255, 255))
        return [p[idx], np.where(wf[idx]>0, 0, 1).reshape(-1, 1) * COL1 + np.where(wf[idx]<0, 0, 1).reshape(-1, 1) * COL2]

    def show(self, p=None):
        viewer.show(self.molecule, molinfo=[{'cub': self.get_cub(p)}])

    def translate(self, trans):
        for f in self.basis_functions:
            f.translate(trans)
        self.molecule.translate(trans)

    def rotate(self, R=None, x=0, y=0, z=0):
        if R is None:
            R = get_rotmat(x=x, y=y, z=z)

        unq_atoms = set([f.fragment_unique_name for f in self.basis_functions])
        unq_ls = set([f.l for f in self.basis_functions])
        f_by_atom = {atom: [f for f in self.basis_functions if f.fragment_unique_name == atom] for atom in unq_atoms}
        f_by_atom_and_l = {atom: {l: [f for f in fs if f.l == l] for l in unq_ls} for atom, fs in f_by_atom.items()}

        new_coeffs = []
        for f, coeff in zip(self.basis_functions, self.coefficients):
            f.rotate(R)
            like_fs = f_by_atom_and_l[f.fragment_unique_name][f.l]
            like_fs = [f_ for f_ in like_fs if f_.n == f.n]
            if f.l == 0:  # we dont have to rotate s-orbitals
                new_coeffs.append(coeff)
                continue

            # for atom in unq_atoms:
            coeff_vector = np.sum([f_.index * self.coefficients[self.basis_functions.index(f_)] for f_ in like_fs], axis=0)
            coeff_vector_rot = coeff_vector @ R.T
            new_coeffs.append(coeff_vector_rot.flatten() @ f.index)

        self.molecule.rotate(R)
        self.coefficients = new_coeffs

    @property
    @timer.Time
    def norm(self):
        '''The overlap integral of this contracted basis function with itself should be 1
        '''
        if self._norm is None:
            S = 0
            for coeff1, f1 in zip(self.coefficients, self.basis_functions):
                for coeff2, f2 in zip(self.coefficients, self.basis_functions):
                    S += coeff1 * coeff2 * f1.overlap(f2)
            self._norm = 1/sqrt(S)
        return self._norm

    @timer.Time
    def overlap(self, other: 'MolecularOrbital', method='exact'):
        if method == 'exact':
            S = 0
            for coeff1, f1 in zip(self.coefficients, self.basis_functions):
                for coeff2, f2 in zip(other.coefficients, other.basis_functions):
                    S += coeff1 * coeff2 * f1.overlap(f2)
            return S * self.norm * other.norm
        elif method == 'numeric':
            x = np.linspace(-5, 5, 15).reshape(-1, 1)
            y = np.linspace(-5, 5, 15).reshape(-1, 1)
            z = np.linspace(-10, 10, 60).reshape(-1, 1)

            p = np.meshgrid(x, y, z)
            p = [r_.flatten() for r_ in p]
            p = np.vstack(p).T
            wf1 = self(p)
            wf2 = other(p)
            return (wf1 * wf2).sum()
        else:
            raise KeyError(f'Unknown method {method}, must be "exact" or "numeric"')



if __name__ == '__main__':
    # mo = get(r"/Users/yumanhordijk/PhD/fast_EDA/calculations/benzene/go.results/adf.rkf", 'LUMO')

    # mols = []
    # cubs = []
    # for rot in np.linspace(0, pi*2, 240):
    #     mo = get(r"/Users/yumanhordijk/PhD/fast_EDA/calculations/benzene/go.results/adf.rkf", 'LUMO')
    #     mo.rotate(x=rot, y=rot, z=rot)
    #     mo.translate([0, 0, -2])
    #     mols.append(mo.molecule)
    #     print('Ik ben Yuman')
    #     # cubs.append(mo.basis_functions[3].get_cub())
    #     cubs.append(mo.get_cub())
    #     print()
    # files = [f'figs/{i}.png' for i in range(len(mols))]
    # viewer.screen_shot_mols(mols, files, [{'cub': cub} for cub in cubs], simple=True)
    # make_gif('rotate_PVDZ_LUMO.mp4', files, fps=60)


    ds = np.linspace(0, 3, 34)

    overlaps = []
    for d in ds:
        print(d)
        mo1 = get(r"/Users/yumanhordijk/PhD/fast_EDA/calculations/methyl/sp.results/adf.rkf", 'SOMO')
        mo1.translate([0, 0, d])
        mo2 = get(r"/Users/yumanhordijk/PhD/fast_EDA/calculations/methyl/sp.results/adf.rkf", 'SOMO')
        overlaps.append(-mo1.overlap(mo2)*100)
    plt.plot(ds, overlaps, label='Exact')

    ds = np.linspace(0, 3, 34)

    overlaps = []
    for d in ds:
        print(d)
        mo1 = get(r"/Users/yumanhordijk/PhD/fast_EDA/calculations/methyl/sp.results/adf.rkf", 'SOMO')
        mo1.translate([0, 0, d])
        mo2 = get(r"/Users/yumanhordijk/PhD/fast_EDA/calculations/methyl/sp.results/adf.rkf", 'SOMO')
        overlaps.append(-mo1.overlap(mo2, method='numeric')*100)
    plt.plot(ds, overlaps, label='Numeric')

    plt.xlabel('Me - Me Distance (Angstrom)')
    plt.ylabel('Overlap (%)')
    plt.legend()
    timer.print_timings2()

    plt.show()
