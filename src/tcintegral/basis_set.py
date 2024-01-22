from tcutility.results.result import Result
import numpy as np
from tcintegral import contracted
import os
import matplotlib.pyplot as plt


j = os.path.join

l_to_name = list('SPDF')
axis_order = 'xyz'


def cp2k_reader(file):
    with open(file) as inp:
        lines = inp.readlines()
        lines = [line.strip() for line in lines if line.strip() != '' and not line.startswith('#')]

    read = True
    atom_start = 0
    atom_data = Result()
    while read:
        atom = lines[atom_start].split()[0]
        ncontracted = int(lines[atom_start+1])
        contracted_start = atom_start + 2
        atom_data[atom] = []
        for n in range(ncontracted):
            _, lmin, lmax, Nexp, *nl  = lines[contracted_start].split()
            atom_data[atom].append(Result())
            atom_data[atom][n].lmin = int(lmin)
            atom_data[atom][n].lmax = int(lmax)
            atom_data[atom][n].Nexp = int(Nexp)

            data_lines = lines[contracted_start+1:contracted_start+int(Nexp)+1]
            data_lines = [[float(x) for x in line.split()] for line in data_lines]
            atom_data[atom][n].exps = np.array([line[0] for line in data_lines]).flatten()
            atom_data[atom][n].coeffs = np.atleast_2d([line[1:] for line in data_lines])
            contracted_start += int(Nexp) + 1

        atom_start = contracted_start

        if atom_start >= len(lines):
            read = False

    return atom_data


class BasisSet:
    def __init__(self, file):
        self.bs_data = cp2k_reader(file)

    def get(self, atom, func, center):
        '''
        Access basis functions as bs['Xx(nl)'] or bs['Xx', 'nl']
        '''
        # decode the func
        principal = int(func[0])
        angular = l_to_name.index(func[1])
        magnetic = func[1:]
        if magnetic == 'S':
            index = [0, 0, 0]
        elif magnetic == 'P:x':
            index = [1, 0, 0]
        elif magnetic == 'P:y':
            index = [0, 1, 0]
        elif magnetic == 'P:z':
            index = [0, 0, 1]
        elif magnetic == 'D:x2':
            index = [2, 0, 0]
        elif magnetic == 'D:y2':
            index = [0, 2, 0]
        elif magnetic == 'D:z2':
            index = [0, 0, 2]
        elif magnetic == 'D:xy':
            index = [1, 1, 0]
        elif magnetic == 'D:xz':
            index = [1, 0, 1]
        elif magnetic == 'D:yz':
            index = [0, 1, 1]

        atom_data = self.bs_data[atom]
        atom_data = [data for data in atom_data if data.lmin <= angular <= data.lmax][principal-1]
        angular_index = angular - atom_data.lmin
        coeffs = atom_data.coeffs[:, angular_index]
        cont = contracted.Contracted(atom_data.exps, np.array(center), index, coeffs)
        cont.name = f'{atom}({func})'
        cont.atom = atom
        cont.principal = principal
        cont.angular = angular
        cont.magnetic = magnetic
        return cont

    def __contains__(self, key):
        try:
            self.get(key[0], key[1], 0)
            return True
        except (IndexError, UnboundLocalError):
            return False


STO6G = BasisSet(j(os.path.split(__file__)[0], 'basis_sets', 'sto-6g.1.cp2k'))
STO3G = BasisSet(j(os.path.split(__file__)[0], 'basis_sets', 'sto-3g.1.cp2k'))
STO2G = BasisSet(j(os.path.split(__file__)[0], 'basis_sets', 'sto-2g.1.cp2k'))
ccPVDZ = BasisSet(j(os.path.split(__file__)[0], 'basis_sets', 'cc-pvdz.1.cp2k'))


if __name__ == '__main__':
    bs = BasisSet('basis_sets/cc-pvdz.1.cp2k')
    Li1S = bs.get('C', '1S', 0)
    Li2S = bs.get('C', '2S', 0)

    # print(Li2S.coefficients, Li2S.exponents)
    # print(Li1S.norm)
    x = np.linspace(-10, 10, 10000)
    print(np.atleast_2d(x).shape)
    dx = x[1] - x[0]
    print((Li1S(x)**2).sum()*dx)
    print((Li2S(x)**2).sum()*dx)
    # print(Li1S(x))
    plt.plot(x, Li1S(x)**2, label='Li(1S)')
    plt.plot(x, Li2S(x)**2, label='Li(2S)')
    plt.legend()
    plt.show()
