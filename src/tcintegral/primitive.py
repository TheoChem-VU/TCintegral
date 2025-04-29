import numpy as np
from math import pi, sqrt
# from yutility import timer


def double_factorial(n):
    prod = 1
    for i in range(n % 2, n+1, 2):
        if i == 0:
            continue
        prod *= i
    return prod


class Primitive:
    def __init__(self, exponent: float, center: list[float], index: list[int]):
        self.exponent = exponent  # exponent for the gaussian part
        self.center = np.atleast_1d(center) * 0.52918  # position of the primitive center converted from Angstrom to Bohr
        self.index = np.atleast_1d(index)  # indices of the angular part
        self._norm = None

    @property
    def ndims(self):
        return self.center.size

    @property
    # @timer.Time
    def norm(self):
        r'''Calculate the normalization constant N for this primitive.
        Recall that we have to calculate the square integral of this primitive:

        $$N^2\prod_{r\in{\{x, y, z, ...\}}}\int_{-\infty}^\infty r^{2n_i} e^{-2ar^2}dr=1$$

        For a closed form of half this integral see: https://en.wikipedia.org/wiki/List_of_integrals_of_exponential_functions
        '''
        if self._norm is None:
            S = 1
            a = self.exponent * 2
            for i in range(self.ndims):
                n = self.index[i]
                S *= double_factorial(2*n - 1)/(2 * a)**n * sqrt(pi/a)
            self._norm = 1/sqrt(S)
        return self._norm

    def __call__(self, r):
        r = np.atleast_2d(r)
        wf = np.ones(r.shape[0])
        for i in range(self.ndims):
            rc = r[:, i] - self.center[i]
            ang = rc ** self.index[i]
            rad = np.exp(-self.exponent * rc**2)
            wf *= ang * rad
        return wf * self.norm

    def overlap(self, other: 'Primitive'):
        '''
        Calculate the overlap between this and another primitive using the Obara-Saika recurrence relations for overlap integrals
        '''
        assert self.ndims == other.ndims

        # with timer.Timer('Primitive.overlap.prepare'):
            # define some usefull constants
        zeta = self.exponent + other.exponent
        # xi = self.exponent * other.exponent / zeta
        c1 = self.center
        c2 = other.center
        P = (self.exponent * c1 + other.exponent * c2)/zeta
        dPA = P - c1
        dPB = P - c2

        S00 = sqrt(pi/zeta) * np.exp(-self.exponent * other.exponent / zeta * (c1 - c2)**2)

        def obara_saika(N1, N2, idx):
            # first calculate the s-s overlap, this will be the first entry in our recurrence relations
            # with timer.Timer('_overlap.obara_saika.S00'):
            # S00s.setdefault()
            # S00 = sqrt(pi/zeta) * exp(-self.exponent * other.exponent / zeta * R**2)
            if N1 == 0 and N2 == 0:
                return S00[idx]

            # if we are not in the s-s overlap we will create our triangle here
            # we have N1+2 x N2+2 because we need to access the N1, N2 position, which is only calculated if the triangle is large enough
            S = np.zeros((N1+2, N2+2)) - 1
            S[0, 0] = S00[idx]
            for n1 in range(N1+1):
                for n2 in range(N2+1):
                    # use the recurrence relations, this is recalculating some values, but the operations are very cheap so it does not matter
                    common = (n1 * S[n1-1, n2] + n2 * S[n1, n2-1])/(2*zeta)
                    S[n1+1, n2] = dPA[idx] * S[n1, n2] + common
                    S[n1, n2+1] = dPB[idx] * S[n1, n2] + common
            return S[N1, N2]

        # total overlap is the product of all one-dimensional overlaps
        overlap = self.norm * other.norm  # normalization constants include all dimensions, so we include them one time
        for i in range(self.ndims):
            # with timer.Timer('Primitive.overlap.obara_saika'):
            overlap *= obara_saika(self.index[i], other.index[i], i)
        return overlap


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # build the grid of points to evaluate
    x = np.linspace(-10, 10, 100).reshape(-1, 1)
    y = np.linspace(-10, 10, 100).reshape(-1, 1)
    z = np.linspace(-10, 10, 100).reshape(-1, 1)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    r2 = np.meshgrid(x, y)
    r2 = [r_.flatten() for r_ in r2]
    r2 = np.vstack(r2).T

    r3 = np.meshgrid(x, y, z)
    r3 = [r_.flatten() for r_ in r3]
    r3 = np.vstack(r3).T

    # test 1D:
    p = Primitive(1, 0, 0)  # s-orbital
    wf = p(x)**2
    print(wf.sum() * dx, p.norm)

    # test 2D:
    p = Primitive(1, (0, 0), (0, 0))  # s-orbital
    wf = p(r2).reshape(100, 100)**2
    print(wf.sum(), p.norm)
    # plt.imshow(wf)
    # plt.show()

    # test 3D:
    p = Primitive(1, (0, 0, 0), (0, 0, 0))  # s-orbital
    wf = p(r3).reshape(100, 100, 100)**2
    print(wf.sum() * dx * dy * dz, p.norm)

    # test 1D:
    p = Primitive(1, 0, 1)  # p-orbital
    wf = p(x)**2
    print(wf.sum() * dx, p.norm)

    # test 2D:
    p = Primitive(1, (0, 0), (0, 1))  # p-orbital
    wf = p(r2).reshape(100, 100)**2
    print(wf.sum() * dx * dy, p.norm)

    # test 3D:
    p = Primitive(1, (0, 0, 0), (0, 0, 1))  # p-orbital
    wf = p(r3).reshape(100, 100, 100)**2
    print(wf.sum() * dx * dy * dz, p.norm)

    # test 1D:
    p = Primitive(1, 0, 2)  # d-orbital
    wf = p(x)**2
    print(wf.sum() * dx, p.norm)

    # test 2D:
    p = Primitive(1, (0, 0), (0, 2))  # dy2-orbital
    wf = p(r2).reshape(100, 100)**2
    print(wf.sum() * dx * dy, p.norm)

    # test 2D:
    p = Primitive(1, (0, 0), (1, 1))  # dxy-orbital
    wf = p(r2).reshape(100, 100)**2
    print(wf.sum() * dx * dy, p.norm)

    # test 2D:
    p = Primitive(1, (0, 0), (2, 0))  # dx2-orbital
    wf = p(r2).reshape(100, 100)**2
    print(wf.sum() * dx * dy, p.norm)

    # test 3D:
    p = Primitive(1, (0, 0, 0), (0, 0, 2))  # d-orbital
    wf = p(r3).reshape(100, 100, 100)**2
    print(wf.sum() * dx * dy * dz, p.norm)


    print()
