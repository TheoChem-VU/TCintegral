import numpy as np
from listfunc import ensure_list
from collections.abc import Iterable
from math import ceil, floor
import matplotlib.pyplot as plt
from tcutility import data
from scm import plams
import copy


class Grid:
    '''
    Class that defines positions and values on a grid.
    '''
    def __init__(self, spacing: Iterable[float] or float = None):
        '''
        Defines a grid with given spacing.
        
        Args:
            spacing: the distance between gridpoints. If a single value, the distance will be the same in all directions. A list/tuple value will define the grid along each direction.
        '''
        self.spacing = ensure_list(spacing)
        self.origin = None
        self.indices = None
        self.cutoff_indices = None
        self.values = None
        self.colors = None
        self._points = None

        self.sub_grids = []

    @property
    def points(self):
        if self._points is not None:
            return self._points
            
        if self.indices is None:
            self.set_points()

        spacing = self.spacing
        if len(spacing) == 1 and len(spacing) != self.ndims:
            spacing = spacing * self.ndims
        return self.indices * spacing

    @property
    def extent(self):
        spacing = self.spacing
        if len(spacing) == 1 and len(spacing) != self.ndims:
            spacing = spacing * self.ndims

        extents = [[(x, x) for x in self.origin]] + [sub_grid[1].extent for sub_grid in self.sub_grids]
        extent = []
        for i in range(self.ndims):
            # print(i, extents)
            mi = min(extent_[i][0] for extent_ in extents)
            ma = max(extent_[i][1] for extent_ in extents)
            # we are interested in the index locations
            extent.append((mi/spacing[i], ma/spacing[i]))
        # return [(x, x) for x in self.origin]
        return extent

    def set_points(self):
        # make sure spacing is the correct dimension
        spacing = self.spacing
        if len(spacing) == 1 and len(spacing) != self.ndims:
            spacing = spacing * self.ndims

        # generate axes, indices and point coordinates
        axes = [np.arange(floor(ex[0]), ceil(ex[1]+1)) for ex in self.extent]
        meshed_axes = np.meshgrid(*axes, indexing='ij')
        meshed_axes = [axis.flatten() for axis in meshed_axes]
        indices = np.vstack(meshed_axes).T
        points = indices * spacing
        to_keep = np.full(len(indices), False)  # start with an empty grid
        for sign, grid in self.sub_grids:
            if sign == '+':
                to_keep = np.logical_or(to_keep, grid.__contains__(points))
            if sign == '-':
                to_keep = np.logical_and(to_keep, 1-grid.__contains__(points))

        self.indices = indices[to_keep]
        if self.values is None:
            self.values = np.zeros(len(self.indices))

    def set_colors(self, func):
        self.colors = func(self.values)

    @property
    def ndims(self):
        return len(self.origin)

    @property
    def shape(self):
        if hasattr(self, '_shape'):
            return self._shape
            
        self.set_points()
        return np.max(self.indices, axis=0) - np.min(self.indices, axis=0) + 1

    def __sub__(self, other):
        if isinstance(other, Grid):
            self.sub_grids.append(('-', other))
        return self

    def __add__(self, other):
        if self.origin is None:
            self.origin = other.origin

        if isinstance(other, Grid):
            self.sub_grids.append(('+', other))
        return self

    def __contains__(self, p):
        return True

    def set_cutoff(self, val, use_abs=True):
        values = self.values
        if use_abs:
            values = abs(values)
        self.cutoff_indices = np.where(values > val)

    def copy(self):
        return copy.deepcopy(self)

    def interpolate(self, p):
        p = np.atleast_3d(p)
        closest = self.points.T - p
        # print(closest.shape)
        closest[closest > 0] = -np.inf
        closest_idx = np.argmax(np.sum(closest, axis=1), axis=1)

        index_position = (p - self.origin) / self.spacing
        indices = index_position.astype(int)
        xd = index_position - indices

        C = np.zeros((2, 2, 2, p.shape[1]))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    displacement = np.array([i, j, k])
                    target = self.indices[closest_idx] + displacement
                    print((np.atleast_3d(target) - self.points.T).shape)
                    print(np.where((np.atleast_3d(target) - self.points.T) == 0))
                    # print(np.equal(self.indices, target))
                    print((self.indices == target))
                    # index = np.where(self.indices == target)[0]
                    index = np.where((np.atleast_3d(target) - self.points.T) == 0)[1]
                    print(index.shape)
                    C[i, j, k, :] = self.values[index]

        C_bi = np.array([[
             C[0, 0, 0, :] * (1 - xd[0]) + C[1, 0, 0, :] * xd[0],
             C[0, 0, 1, :] * (1 - xd[0]) + C[1, 0, 1, :] * xd[0],],
            [C[0, 1, 0, :] * (1 - xd[0]) + C[1, 1, 0, :] * xd[0],
             C[0, 1, 1, :] * (1 - xd[0]) + C[1, 1, 1, :] * xd[0],
            ]])

        C_tri = [
            C_bi[0, 0, :] * (1 - xd[1]) + C_bi[1, 0, :] * xd[1],
            C_bi[0, 1, :] * (1 - xd[1]) + C_bi[1, 1, :] * xd[1],
        ]

        c = C_tri[0, :] * (1 - xd[2]) + C_tri[1, :] * xd[2]

        return c


class Cube(Grid):
    def __init__(self, origin: Iterable[float] or float = None, 
                 size: Iterable[float] or float = None,
                 *args, **kwargs):
        '''
        Build a grid of points in a cube.
        
        Args:
            origin: The origin of the cube to be added.
            size: The distance the cube goes from the origin. For example, for a 2D box, the size would be (width, height).
        '''
        super().__init__(*args, **kwargs)
        self.origin = ensure_list(origin)
        self.size = ensure_list(size)

    def __contains__(self, p):
        '''
        Check if points p are inside this Grid
        '''
        p = np.array(p).squeeze()
        disp = p - self.origin
        if p.ndim == 2:
            return np.logical_and(np.all(0 <= disp, axis=1), np.all(disp <= self.size, axis=1))
        return np.logical_and(np.all(0 <= disp), np.all(disp <= self.size))

    @property
    def extent(self):
        top_corner = np.array(self.origin) + np.array(self.size)
        return list(zip(self.origin, top_corner))


class Sphere(Grid):
    def __init__(self, origin: Iterable[float] or float = None, 
                 radius: float = None,
                 *args, **kwargs):
        '''
        Build a grid of points in a cube.
        
        Args:
            origin: The origin of the cube to be added.
            radius: The distance from the origin of the sphere to its edge. Can be tuple or single value.
        '''
        super().__init__(*args, **kwargs)
        self.origin = ensure_list(origin)
        self.radius = radius

    def __contains__(self, p):
        p = np.array(p).squeeze()
        if p.ndim == 2:
            dists = np.linalg.norm(p - self.origin, axis=1)
            return dists <= self.radius
        return np.linalg.norm(p - self.origin) <= self.radius

    @property
    def extent(self):
        lower_corner = np.array(self.origin) - np.array(self.radius)
        top_corner = np.array(self.origin) + np.array(self.radius)
        return list(zip(lower_corner, top_corner))


def from_molecule(mol, spacing=.5, atom_scale=1):
    grid = Grid(spacing)
    for atom in mol:
        rad = data.atom.radius(atom.symbol)
        grid += Sphere(atom.coords, rad*atom_scale)
    return grid


def molecule_bounding_box(mol, margin=4, spacing=.5):
    lower_corner = np.array([min(atom.coords[dim] for atom in mol) for dim in range(3)])
    top_corner = np.array([max(atom.coords[dim] for atom in mol) for dim in range(3)])
    g = Grid(spacing)
    g += Cube((lower_corner - margin).tolist(), (top_corner - lower_corner + margin * 2).tolist())
    return g


def from_cub_file(file):
    with open(file) as cub:
        lines = [line.strip() for line in cub.readlines()]

    natoms, origin = abs(int(lines[2].split()[0])), np.array(lines[2].split()[1:]).astype(float) * 0.52918
    xvec, yvec, zvec = np.array([line.split()[1:] for line in lines[3:6]]).astype(float) * 0.52918
    xn, yn, zn = np.array([line.split()[0] for line in lines[3:6]]).astype(int)
    
    values = []
    for line in lines[6+natoms:]:
        values.extend(line.split())
    values = np.array(values).astype(float)

    gridd = Grid(sum([xvec, yvec, zvec]).tolist())
    gridd += Cube(origin.tolist(), sum([xvec*xn, yvec*yn, zvec*zn]).tolist())
    gridd.values = values
    gridd._shape = (xn, yn, zn)

    atomcoords = []
    atnums = []
    for line in lines[6:6+natoms]:
        atomcoords.append(np.array(line.split()[2:]).astype(float) * 0.52918)
        atnums.append(int(line.split()[0]))
    mol = plams.Molecule()
    for atnum, atcoord in zip(atnums, atomcoords):
        mol.add_atom(plams.Atom(atnum=atnum, coords=atcoord))
    mol.guess_bonds()

    gridd.molecule = mol

    return gridd



if __name__ == '__main__':
    from tcviewer import Screen, materials
    gridd = from_cub_file('/Users/yumanhordijk/PhD/yutility/test/orbitals/rkf/5%SFO_A%5.t41')
    # print(((gridd.values * gridd.spacing[0])**2).sum())
    # print(gridd.interpolate([0.1, -0.1, 0]))
    # print(gridd.extent)
    # with Screen() as scr:
    #     scr.draw_molecule(gridd.molecule)
    #     scr.draw_isosurface(gridd, .03, color=([0, 0, 1], [1, 0, 0]), material=materials.orbital_shiny)
    # print(gridd.shape)
    # for p in gridd.points:
    #     print(p)
    import matplotlib.pyplot as plt


    # z = np.zeros((50, 50))
    X, Y, Z = np.linspace(-9, 7, 50), np.linspace(-9, 7, 50), [0]
    p = np.vstack(np.meshgrid(X, Y, Z))
    meshed_axes = np.meshgrid(X, Y, Z, indexing='ij')
    meshed_axes = [axis.flatten() for axis in meshed_axes]
    p = np.vstack(meshed_axes).T
    z = gridd.interpolate(p)
    # for i, x in enumerate(np.linspace(-9, 7, 50)):
    #     print(i)
    #     for j, y in enumerate(np.linspace(-9, 7, 50)):
    #         try:
    #             z[i, j] = gridd.interpolate([x, y, 0])
    #         except:
    #             raise
    #             z[i, j] = np.nan

    plt.imshow(z)
    plt.show()
