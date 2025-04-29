import numpy as np
from math import sqrt
from tcintegral import primitive
# from yutility import timer
from math import cos, sin


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


class Contracted:
    def __init__(self, exponents: list[float], center: list[float], index: list[int], coefficients: list[float], name=None):
        self.center = np.atleast_1d(center)  # position of the primitive center
        self.exponents = np.atleast_1d(exponents)  # exponent for the gaussian part
        self.index = np.atleast_1d(index)  # indices of the angular part
        self.primitives = [primitive.Primitive(exponent, self.center, self.index) for exponent in self.exponents]
        self.coefficients = np.atleast_1d(coefficients)
        self._norm = None
        self.name = name

    def __repr__(self):
        if self.name is not None:
            return self.name
        return super().__repr__()

    @property
    def ndims(self):
        return self.center.size

    @property
    # @timer.Time
    def norm(self):
        '''The overlap integral of this contracted basis function with itself should be 1
        '''
        if self._norm is None:
            S = 0
            for coeff1, prim1 in zip(self.coefficients, self.primitives):
                for coeff2, prim2 in zip(self.coefficients, self.primitives):
                    S += coeff1 * coeff2 * prim1.overlap(prim2)
            self._norm = 1/sqrt(S)
        return self._norm

    def __call__(self, r):
        r = np.atleast_2d(r).T
        wf = np.zeros(r.shape[0])
        for coeff, prim in zip(self.coefficients, self.primitives):
            wf += coeff * prim(r)
        return wf * self.norm

    def get_cub(self, p=None, cutoff=[.25, .3]):
        if p is None:
            x = np.linspace(-6, 6, 50).reshape(-1, 1)
            y = np.linspace(-6, 6, 50).reshape(-1, 1)
            z = np.linspace(-6, 6, 50).reshape(-1, 1)

            p = np.meshgrid(x, y, z)
            p = [r_.flatten() for r_ in p]
            p = np.vstack(p).T
        wf = self(p)
        wf_abs = abs(wf)/np.max(abs(wf))
        idx = np.where(np.logical_and(wf_abs > cutoff[0], wf_abs < cutoff[1]))[0]
        COL1 = np.array((255, 0, 0)) 
        COL2 = np.array((0, 0, 255)) 
        return [p[idx], np.where(wf[idx] > 0, 0, 1).reshape(-1, 1) * COL1 + np.where(wf[idx] < 0, 0, 1).reshape(-1, 1) * COL2]

    # @timer.Time
    def overlap(self, other: 'Contracted'):
        S = 0
        for coeff1, prim1 in zip(self.coefficients, self.primitives):
            for coeff2, prim2 in zip(other.coefficients, other.primitives):
                S += coeff1 * coeff2 * prim1.overlap(prim2)
        return S * self.norm * other.norm

    def translate(self, trans):
        self.center += trans
        for p in self.primitives:
            p.center += trans

    def rotate(self, R=None, x=0, y=0, z=0):
        if R is None:
            R = get_rotmat(x=x, y=y, z=z)

        for p in self.primitives:
            p.center = p.center @ R.T
