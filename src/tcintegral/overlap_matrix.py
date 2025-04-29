# from tcintegral import reactant
import numpy as np


def overlap_matrix(rct1, rct2, method='exact'):
    S = np.zeros((len(rct1.mos + rct2.mos), len(rct2.mos)))
    for i, mo1 in enumerate(rct1.mos + rct2.mos):
        for j, mo2 in enumerate(rct2.mos):
            S[i, j] = abs(mo1.overlap(mo2, method=method))
    return S
