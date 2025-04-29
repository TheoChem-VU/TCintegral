from tcutility import results, geometry, ensure_list
import pyfmo
# from yutility import geometry, orbitals, ensure_list, timer
import numpy as np
# from yviewer import viewer
import os
from tcintegral import molecular_orbital
from math import cos, sin
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


class Reactant:
    def __init__(self, rct_calc_dir, moleculename=None, bs_file=r"basis_sets/sto-6g.1.cp2k"):
        self.rct_calc_dir = rct_calc_dir
        self.moleculename = moleculename
        if self.moleculename is None:
            self.moleculename = os.path.split(rct_calc_dir)[-1]

        self.rkf_res = results.read(rct_calc_dir)
        self.mol = self.rkf_res.molecule.input
        self.loaded_wfs = []
        self.transform = geometry.Transform()
        self._orbitals = pyfmo.orbitals.Orbitals(self.rkf_res.files['adf.rkf'])
        self.mos = []
        self.bs_file = bs_file
        # self.mos = [molecular_orbital.get(self.rkf_res.files['adf.rkf'], mo.name, bs_file) for mo in self._orbitals.mos]

    def translate(self, trans):
        self.mol.translate(trans)
        [mo.translate(trans) for mo in self.mos]

    def rotate(self, *args, **kwargs):
        # self.transform.rotate(*args, **kwargs)
        R = get_rotmat(**kwargs)
        self.mol.rotate(R)
        [mo.rotate(R) for mo in self.mos]

    def center(self):
        self.translate(np.mean(self.coords, axis=0))

    @property
    def coords(self):
        return self.transform.apply(np.array([atom.coords for atom in self.mol]))

    def load_mos(self, start, end=None):
        if end:
            mos = self._orbitals.mos[start:end]
        else:
            mos = self._orbitals.mos[start]

        for mo in ensure_list(mos):
            orb = molecular_orbital.get(self.rkf_res.files['adf.rkf'], mo.name, self.bs_file)
            orb.moleculename = self.moleculename
            self.mos.append(orb)

    # def show(self, p=None):
    #     if p is None:
    #         x = np.linspace(-6, 6, 50).reshape(-1, 1)
    #         y = np.linspace(-6, 6, 50).reshape(-1, 1)
    #         z = np.linspace(-6, 6, 50).reshape(-1, 1)

    #         p = np.meshgrid(x, y, z)
    #         p = [r_.flatten() for r_ in p]
    #         p = np.vstack(p).T

    #     nmos = len(self.mos)
    #     viewer.show([self.mol] * nmos, molinfo=[{'cub': mo.get_cub(p)} for mo in self.mos])

    # @timer.time
    def overlap(self, other: 'Reactant'):
        nmo1 = len(self.mos)
        nmo2 = len(other.mos)
        S = np.zeros((nmo1, nmo2))
        for i, mo1 in enumerate(self.mos):
            for j, mo2 in enumerate(other.mos):
                S[i, j] = mo1.overlap(mo2)
        return S


if __name__ == '__main__':
    # with timer.Timer('Load reactants'):
    rct1 = Reactant(r"/Users/yumanhordijk/PhD/fast_EDA/calculations/butadiene")
    rct2 = Reactant(r"/Users/yumanhordijk/PhD/fast_EDA/calculations/ethene")
    rct1.load_mos('HOMO-4', 'LUMO+4')
    rct2.load_mos('HOMO-4', 'LUMO+4')

    # rct1.load_mos('LUMO+1')
    # rct2.load_mos('HOMO')
    # with timer.Timer('Place reactants'):
    rct2.rotate(y=np.pi/2)
    rct2.translate([3, 0, 0])

    # print(rct1.overlap(rct2))

    # # rct.show()
    # mo1 = rct1.mos[0]
    # mo2 = rct2.mos[0]
    # cub1 = mo1.get_cub()
    # cub2 = mo2.get_cub()

    # viewer.show(rct1.mol + rct2.mol, molinfo=[{'cub': [np.vstack([cub1[0], cub2[0]]), np.vstack([cub1[1], cub2[1]])]}])
    S = rct1.overlap(rct2)
    plt.imshow(abs(S), origin='lower')
    plt.xticks(range(len(rct2.mos)), [mo.name for mo in rct2.mos])
    plt.yticks(range(len(rct1.mos)), [mo.name for mo in rct1.mos])

    timer.print_timings()

    plt.show()
