import math
import numpy as np
import png
import typing
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

    def __init__(self, width: int, height: int):
        '''
        Constructs an empty image.
        '''
        self.width = width
        self.height = height 
        self.pixels = [0 for _ in range(4 * width * height)]

    def __getitem__(self, key: tuple[int, int]) -> tuple[int, int, int, int]:
        x, y = key
        return tuple(self.pixels[(4 * ((self.width * y) + x)):(4 * ((self.width * y) + x + 1))])
    
    def __setitem__(self, key: tuple[int, int], value: tuple[int, int, int, int]):
        x, y = key
        self.pixels[(4 * ((self.width * y) + x)):(4 * ((self.width * y) + x + 1))] = value

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

class TreeRenderer:
    def render(self, structure: TreeStructure, filename: str):
        '''
        Renders the tree to a PNG file.

        Args:
            structure (TreeStructure): The structure of the tree to render.
            filename (str): The filename to render to.
        '''
        pass

class TreeSimpleRenderer(TreeRenderer):
    '''
    A simple renderer that renders a tree in grayscale.
    '''

    width_parameter: float 
    '''
    Controls the width of branches, relative to their descendants.
    '''

    zoom: float 
    '''
    How much to zoom in by.
    '''

    def __init__(self, width_parameter: float = 2.0, zoom: float = 1.0):
        self.width_parameter = width_parameter
        self.zoom = zoom 

    def render_branch_segment(self, img: Image, seg: TreeBranchSegment):
        '''
        Renders the skeleton of a single branch segment in a tree.
        '''

        # Render the branch
        length: float = np.linalg.norm(seg.vec)
        unit: np.typing.NDArray[typing.Any] = (0.5 / self.zoom) * seg.vec / length
        num_units: int = math.ceil(length * self.zoom / 0.5)
        for n in range(num_units):
            pos_base: np.typing.NDArray[typing.Any] = seg.start + (n * unit)
            vec1: np.typing.NDArray[typing.Any] = np.array([unit[1], -unit[0], 0.0]) / math.sqrt(pow(unit[0], 2.0) + pow(unit[1], 2.0)) if abs(unit[0]) > 0.001 or abs(unit[1]) > 0.001 else np.array([1.0, 0.0, 0.0])
            vec2: np.typing.NDArray[typing.Any] = np.cross(unit, vec1) / np.linalg.norm(unit)
            angle: float = 0.0
            while angle < math.tau:
                pos: np.typing.NDArray[typing.Any] = pos_base + ((seg.radius * 0.25 / self.zoom) * ((math.cos(angle) * vec1) + (math.sin(angle) * vec2)))
                x: int = (img.width // 2) + int(round(self.zoom * pos[0]))
                y: int = img.height - int(round(self.zoom * pos[2]))
                luma: int = min(255, max(0, int(round(255 * (0.5 + (0.5 * self.zoom * pos[1] / img.width))))))
                if x >= 0 and x < img.width and y >= 0 and y < img.height:
                    _, existing_luma, _, _ = img[x, y]
                    luma = max(existing_luma, luma)
                    img[x, y] = (luma, luma, luma, 255)

                angle += math.pi / (seg.radius * self.zoom)

    def render(self, structure: TreeStructure, filename: str):
        '''
        Renders the skeleton of the tree to a PNG file.

        Args:
            structure (TreeStructure): The structure of the tree to render.
            filename (str): The path of the file to save to.
        '''
        img: Image = Image(300, 400)
        for k in range(len(structure.branch_segments)):
            self.render_branch_segment(img, structure.branch_segments[k])
        img.save(filename)