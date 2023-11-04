import numpy as np
from listfunc import ensure_list
from collections.abc import Iterable
from math import ceil, floor


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
        self.values = None
        self.colors = None

        self.sub_grids = []

    @property
    def points(self):
        if self.indices is None:
            self.set_points()

        spacing = self.spacing
        if len(spacing) == 1 and len(spacing) != self.ndims:
            spacing = spacing * self.ndims
        return self.indices * spacing + self.origin

    @property
    def extent(self):
        return [(x, x) for x in self.origin]

    def set_points(self):
        # make sure spacing is the correct dimension
        spacing = self.spacing
        if len(spacing) == 1 and len(spacing) != self.ndims:
            spacing = spacing * self.ndims

        # gather extents, this will be the maximum gridpoint sampling area
        extents = [self.extent] + [sub_grid[1].extent for sub_grid in self.sub_grids]
        extent = []
        for i in range(self.ndims):
            mi = min(extent_[i][0] for extent_ in extents)
            ma = max(extent_[i][1] for extent_ in extents)
            # we are interested in the index locations
            extent.append((floor(mi/spacing[i]), ceil(ma/spacing[i])))
        
        # generate axes, indices and point coordinates
        axes = [np.arange(ex[0], ex[1]+1) for ex in extent]
        meshed_axes = np.meshgrid(*axes)
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
        self.values = np.zeros(len(self.indices))

    def set_colors(self, func):
        self.colors = func(self.values)

    @property
    def ndims(self):
        return len(self.origin)

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
