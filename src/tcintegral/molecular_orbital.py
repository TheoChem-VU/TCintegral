# from yutility import orbitals, timer
import pyfmo
from tcintegral import basis_set, grid
from scm import plams
import numpy as np
# from yviewer import viewer
from math import sqrt, cos, sin
import matplotlib.pyplot as plt
import os

j = os.path.join


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


def get(rkf_file, orb, bs_file=r"basis_sets/sto-2g.1.cp2k"):
    bs = basis_set.BasisSet(bs_file)
    orbs = pyfmo.orbitals.Orbitals(rkf_file)
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
    # orb = orbs.mos[orb_name]
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

    def __repr__(self):
        r = ''
        if hasattr(self, 'moleculename'):
            r += self.moleculename + '('

        r += self.name

        if hasattr(self, 'moleculename'):
            r += ')'
            
        return r

    def __call__(self, r):
        r = np.atleast_2d(r)
        wf = np.zeros(r.shape[0])
        for f, coeff in zip(self.basis_functions, self.coefficients):
            wf += f(r.T) * coeff

        N = np.sqrt(sum(wf * wf))
        print(N, ",", 1/self.norm)

        # return wf / np.sqrt(N)
        return wf / self.norm

    def get_cub(self, p=None, cutoff=[.003, 1]):
        if p is None:
            # x = np.linspace(-6, 6, 80).reshape(-1, 1)
            # y = np.linspace(-6, 6, 80).reshape(-1, 1)
            # z = np.linspace(-6, 6, 80).reshape(-1, 1)

            # p = np.meshgrid(x, y, z)
            # p = [r_.flatten() for r_ in p]
            # p = np.vstack(p).T
            p = grid.from_molecule(self.molecule, atom_scale=5, spacing=.1).points
        wf = self(p)
        wf_abs = abs(wf)
        idx = np.where(np.logical_and(wf_abs > cutoff[0], wf_abs < cutoff[1]))[0]
        # idx = np.arange(len(wf_abs))
        try:
            COL1 = np.array((255, 0, 0)) if self.occupied else np.array((255, 165, 0))
            COL2 = np.array((0, 0, 255)) if self.occupied else np.array((0, 255, 255))
        except AttributeError:
            COL1 = np.array((255, 0, 0))
            COL2 = np.array((0, 0, 255))
        return [p[idx], np.where(wf[idx] > 0, 0, 1).reshape(-1, 1) * COL1 + np.where(wf[idx] < 0, 0, 1).reshape(-1, 1) * COL2]

    # def show(self, p=None):
    #     viewer.show(self.molecule, molinfo=[{'cub': self.get_cub(p)}])

    # def screenshot(self, file, p=None):
    #     viewer.screen_shot_mols(self.molecule, [file], molinfo=[{'cub': self.get_cub(p)}])

    def translate(self, trans):
        for f in self.basis_functions:
            f.translate(trans)
        self.molecule.translate(trans)

    def rotate(self, R=None, x=0, y=0, z=0):
        if R is None:
            R = get_rotmat(x=x, y=y, z=z)

        unq_atoms = set([f.fragment_unique_name for f in self.basis_functions])
        unq_angulars = set([f.angular for f in self.basis_functions])
        f_by_atom = {atom: [f for f in self.basis_functions if f.fragment_unique_name == atom] for atom in unq_atoms}
        f_by_atom_and_angular = {atom: {angular: [f for f in fs if f.angular == angular] for angular in unq_angulars} for atom, fs in f_by_atom.items()}

        new_coeffs = []
        for f, coeff in zip(self.basis_functions, self.coefficients):
            f.rotate(R)
            like_fs = f_by_atom_and_angular[f.fragment_unique_name][f.angular]
            like_fs = [f_ for f_ in like_fs if f_.principal == f.principal]
            if f.angular == 0:  # we dont have to rotate s-orbitals
                new_coeffs.append(coeff)
                continue

            # for atom in unq_atoms:
            coeff_vector = np.sum([f_.index * self.coefficients[self.basis_functions.index(f_)] for f_ in like_fs], axis=0)
            coeff_vector_rot = coeff_vector @ R.T
            new_coeffs.append(coeff_vector_rot.flatten() @ f.index)

        self.molecule.rotate(R)
        self.coefficients = new_coeffs

    @property
    # @timer.Time
    def norm(self):
        '''The overlap integral of this contracted basis function with itself should be 1
        '''
        if self._norm is None:
            S = 0
            for coeff1, f1 in zip(self.coefficients, self.basis_functions):
                for coeff2, f2 in zip(self.coefficients, self.basis_functions):
                    S += coeff1 * coeff2 * f1.overlap(f2) * f1.norm * f2.norm
            self._norm = 1/sqrt(S)
        return self._norm

    # @timer.Time
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
            # g = grid.from_molecule(self.molecule, atom_scale=5)
            # g += grid.from_molecule(other.molecule, atom_scale=5)
            wf1 = self(p)
            wf2 = other(p)
            return (wf1 * wf2).sum()
        else:
            raise KeyError(f'Unknown method {method}, must be "exact" or "numeric"')


if __name__ == '__main__':
    from tcviewer import Screen
    # overlaps = []
    # ds = []
    # for d in os.listdir(r"../../test/fixtures/reactants/MeMe_ADF_distance"):
    #     print(j(r"../../test/fixtures/reactants/MeMe_ADF_distance", d))

    #     orbs = orbitals.Orbitals(j(r"../../test/fixtures/reactants/MeMe_ADF_distance", d, 'sp.results', 'adf.rkf'))

    #     ds.append(float(d))
    #     overlaps.append(orbs.sfos['Me1(SUMO)'] @ orbs.sfos['Me2(SOMO)'] * 100)

    # plt.plot(ds, overlaps, label='ADF')

    ds = np.linspace(0, 6, 34)
    overlaps = []
    # mols = []
    for d in ds:
        print(d)
        mo1 = get(r"../../test/fixtures/reactants/ethene/sp.results/adf.rkf", 'HOMO', bs_file=r"basis_sets/cc-pvdz.1.cp2k")
        mo1.rotate(y=np.pi/2)
        mo1.translate([d, 0, 1])
        mo2 = get(r"../../test/fixtures/reactants/butadiene/go.results/adf.rkf", 'LUMO', bs_file=r"basis_sets/cc-pvdz.1.cp2k")

        with Screen() as scr:
            gridd1 = grid.molecule_bounding_box(mo1.molecule + mo2.molecule)
            gridd2 = grid.molecule_bounding_box(mo1.molecule + mo2.molecule)
            gridd1.values = mo1(gridd1.points)
            gridd2.values = mo2(gridd2.points)
            scr.draw_isosurface(gridd1)
            scr.draw_isosurface(gridd2)

        exit()

        overlaps.append(mo1.overlap(mo2)*100)
    plt.plot(ds, overlaps, label='Exact (cc-PVDZ)')
    ds = np.linspace(0, 6, 34)

    overlaps = []
    for d in ds:
        print(d)
        mo1 = get(r"../../test/fixtures/reactants/ethene/sp.results/adf.rkf", 'HOMO', bs_file=r"basis_sets/sto-6g.1.cp2k")
        mo1.rotate(y=np.pi/2)
        mo1.translate([d, 0, 1])
        mo2 = get(r"../../test/fixtures/reactants/butadiene/go.results/adf.rkf", 'LUMO', bs_file=r"basis_sets/sto-6g.1.cp2k")
        overlaps.append(mo1.overlap(mo2)*100)
    plt.plot(ds, overlaps, label='Exact (STO-6G)')

    ds = np.linspace(0, 6, 34)
    overlaps = []
    for d in ds:
        print(d)
        mo1 = get(r"../../test/fixtures/reactants/ethene/sp.results/adf.rkf", 'HOMO', bs_file=r"basis_sets/sto-6g.1.cp2k")
        mo1.rotate(y=np.pi/2)
        mo1.translate([d, 0, 1])
        mo2 = get(r"../../test/fixtures/reactants/butadiene/go.results/adf.rkf", 'LUMO', bs_file=r"basis_sets/sto-6g.1.cp2k")
        overlaps.append(mo1.overlap(mo2, method='numeric')*100)
    plt.plot(ds, overlaps, label='Numeric (STO-6G)')

    plt.xlabel('Ethene - Butadiene Distance (Angstrom)')
    plt.ylabel(r'$\langle$ HOMO | LUMO $\rangle$ (%)')
    plt.legend()
    timer.print_timings2()

    plt.show()
