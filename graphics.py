import math
import numpy as np
import png
from typing import Any, Generator
from util import normalized, rotated
from structure import TreeBranchSegment, TreeStructure

class Image:
    width: int 
    '''
    The width of the image.
    '''

    height: int
    '''
    The height of the image.
    '''

    pixels: list[int]
    '''
    The pixels of the image.
    '''

    z_index: list[float]
    '''
    The z-indices of each pixel in the image.
    '''

    def __init__(self, width: int, height: int):
        '''
        Constructs an empty image.
        '''
        self.width = width
        self.height = height 
        self.pixels = [0 for _ in range(4 * width * height)]
        self.z_index = [-math.inf for _ in range(width * height)]

    def __getitem__(self, key: tuple[int, int]) -> tuple[int, int, int, int, float]:
        x, y = key
        return tuple(self.pixels[(4 * ((self.width * y) + x)):(4 * ((self.width * y) + x + 1))] + self.z_index[((self.width * y) + x):((self.width * y) + x + 1)])
    
    def __setitem__(self, key: tuple[int, int], value: tuple[int, int, int, int, float]):
        x, y = key
        r, g, b, a, z = value
        self.pixels[(4 * ((self.width * y) + x)):(4 * ((self.width * y) + x + 1))] = (r, g, b, a)
        self.z_index[((self.width * y) + x)] = z

    def save(self, filename: str):
        '''
        Saves the image to a file.

        Args:
            filename (str): The file path.
        '''
        # Construct rows
        rows = [self.pixels[(4 * self.width * i):(4 * self.width * (i + 1))] for i in range(self.height)]
        # Save to file
        data = png.from_array(rows, 'RGBA')
        data.save(filename)


class Ray:
    start: np.typing.NDArray[np.floating[Any]]
    '''
    The start of the ray.
    '''

    orientation: np.typing.NDArray[np.floating[Any]]
    '''
    The orientation of the ray.
    '''

    def __init__(self, start: np.typing.NDArray[np.floating[Any]], orientation: np.typing.NDArray[np.floating[Any]]):
        self.start = start
        self.orientation = orientation

    class NoIntersectionException(Exception):
        def __init__(self):
            super().__init__("No intersection with ray.")

    def find_all_intersections(self, segment: TreeBranchSegment) -> Generator[tuple[float, np.typing.NDArray[np.floating[Any]], np.typing.NDArray[np.floating[Any]]], None, None]:
        '''
        Finds all potentially forward-facing intersections with the cylinder.
        '''
        # Forward-facing intersection with cylinder wall
        b: np.typing.NDArray[np.floating[Any]] = segment.start - self.start 
        a: np.typing.NDArray[np.floating[Any]] = normalized(segment.vec)
        n_cross_a: np.typing.NDArray[np.floating[Any]] = np.cross(self.orientation, a)
        d_denom: float = np.dot(n_cross_a, n_cross_a)
        d_det: float = (d_denom * (segment.radius ** 2)) - (np.dot(b, n_cross_a) ** 2)
        if not np.isclose(d_denom, 0.0) and (d_det > 0.0 or np.isclose(d_det, 0.0)):
            d_numer: float = np.dot(n_cross_a, np.cross(b, a)) + (0.0 if np.isclose(d_det, 0.0) else math.sqrt(d_det))
            d: float = d_numer / d_denom
            pos: np.typing.NDArray[np.floating[Any]] = self.start + (d * self.orientation)
            t: float = np.dot(a, pos - segment.start)
            if t > 0.0 and t < 1.0:
                norm: np.typing.NDArray[np.floating[Any]] = normalized(pos - (segment.start + (segment.vec * t)))
                yield (d, pos, norm)

        n_dot_a: float = np.dot(a, self.orientation)
        if not np.isclose(n_dot_a, 0.0):
            # End cap 1
            d1: float = np.dot(a, b) / n_dot_a 
            pos1: np.typing.NDArray[np.floating[Any]] = self.start + (d1 * self.orientation)
            r1: float = np.linalg.norm(pos1 - segment.start)
            if r1 < segment.radius:
                yield (d1, pos1, -a)

            # End cap 2
            d2: float = np.dot(a, b + segment.vec)
            pos2: np.typing.NDArray[np.floating[Any]] = self.start + (d2 * self.orientation)
            r2: float = np.linalg.norm(pos2 - (segment.start + segment.vec))
            if r2 < segment.radius:
                yield (d2, pos2, a)

    def find_intersection(self, segment: TreeBranchSegment) -> tuple[np.typing.NDArray[np.floating[Any]], np.typing.NDArray[np.floating[Any]]]:
        '''
        Finds the most forward-facing intersection with the cylinder.
        '''
        d_best: float = math.inf
        pos_norm_best: tuple[np.typing.NDArray[np.floating[Any]], np.typing.NDArray[np.floating[Any]]] | None = None
        for d, pos, norm in self.find_all_intersections(segment):
            if d < d_best:
                d_best = d 
                pos_norm_best = (pos, norm)
        if pos_norm_best == None:
            raise self.NoIntersectionException()
        return pos_norm_best 
    

class TreeRenderer:
    zoom: float 

    def __init__(self, zoom: float = 1.0):
        self.zoom = zoom

    def fragment(self, 
                 img: Image, 
                 frag: tuple[int, int], 
                 pos: np.typing.NDArray[np.floating[Any]], 
                 normal: np.typing.NDArray[np.floating[Any]]
                 ) -> tuple[int, int, int, int]:
        pass

    def render_branch_segment(self, img: Image, segment: TreeBranchSegment):
        orth: np.typing.NDArray[np.floating[Any]] = rotated(
            input=segment.radius * normalized(segment.vec),
            axis=np.array([0.0, 1.0, 0.0]),
            angle=math.pi / 2
        )
        corner00: np.typing.NDArray[np.floating[Any]] = (segment.start + orth) * self.zoom
        corner01: np.typing.NDArray[np.floating[Any]] = (segment.start - orth) * self.zoom
        corner10: np.typing.NDArray[np.floating[Any]] = (segment.start + segment.vec + orth) * self.zoom 
        corner11: np.typing.NDArray[np.floating[Any]] = (segment.start + segment.vec - orth) * self.zoom 
        x_min: int = min(int(math.floor(corner00[0])), int(math.floor(corner01[0])), int(math.floor(corner10[0])), int(math.floor(corner11[0])))
        x_max: int = max(int(math.ceil(corner00[0])), int(math.ceil(corner01[0])), int(math.ceil(corner10[0])), int(math.ceil(corner11[0])))
        y_min: int = min(int(math.floor(corner00[2])), int(math.floor(corner01[2])), int(math.floor(corner10[2])), int(math.floor(corner11[2])))
        y_max: int = max(int(math.ceil(corner00[2])), int(math.ceil(corner01[2])), int(math.ceil(corner10[2])), int(math.ceil(corner11[2])))
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                offset: float = 100.0
                orientation: np.typing.NDArray[np.floating[Any]] = normalized(np.array([0.0, -0.0, -1.0]))
                ray: Ray = Ray(
                    start=np.array([x / self.zoom, 0.0, (y / self.zoom)]) - (offset * orientation),
                    orientation=orientation
                )
                frag_x: int = x + (img.width // 2)
                frag_y: int = img.height - y
                if frag_x >= 0 and frag_x < img.width and frag_y >= 0 and frag_y < img.height:
                    try:
                        pos, normal = ray.find_intersection(segment)
                        _, _, _, _, prior_depth_index = img[(x, y)]
                        if prior_depth_index < pos[1]:
                            r, g, b, a = self.fragment(
                                img,
                                frag=(frag_x, frag_y),
                                pos=pos,
                                normal=normal
                            )
                            img[(frag_x, frag_y)] = (r, g, b, a, pos[1])
                    except ray.NoIntersectionException:
                        pass

    def render(self, structure: TreeStructure, filename: str):
        '''
        Renders the tree to a PNG file.

        Args:
            structure (TreeStructure): The structure of the tree to render.
            filename (str): The filename to render to.
        '''
        img: Image = Image(300, 400)
        for k in range(len(structure.branch_segments)):
            self.render_branch_segment(img, structure.branch_segments[k])
        img.save(filename)

class TreeSimpleRenderer(TreeRenderer):
    '''
    A simple renderer that renders a tree in grayscale.
    '''

    def __init__(self, zoom: float = 1.0):
        super().__init__(zoom)

    def fragment(self, img: Image, frag: tuple[int, int], pos: np.ndarray[tuple[Any, ...], np.dtype[np.floating[Any]]], normal: np.ndarray[tuple[Any, ...], np.dtype[np.floating[Any]]]) -> tuple[int, int, int, int]:
        luma: int = 255
        return (luma, luma, luma, 255)

class TreeNormalmapRenderer(TreeRenderer):
    '''
    A normalmap renderer for a tree.
    '''

    def __init__(self, zoom: float = 1.0):
        super().__init__(zoom)

    def fragment(self, img: Image, frag: tuple[int, int], pos: np.ndarray[tuple[Any, ...], np.dtype[np.floating[Any]]], normal: np.ndarray[tuple[Any, ...], np.dtype[np.floating[Any]]]) -> tuple[int, int, int, int]:
        n: np.typing.NDArray[np.floating[Any]] = np.floor(256 * (normal + np.array([1.0, 1.0, 1.0])))
        return (
            255 if n[0] > 255 else int(n[0]),
            255 if n[1] > 255 else int(n[1]),
            255 if n[2] > 255 else int(n[2]),
            255
        )