import math
import numpy as np
import png
from structure import TreeNode, TreeStructure

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

    def __init__(self, width, height):
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

class TreeSkeletonImager:
    structure: TreeStructure
    '''
    The structure of a tree.
    '''

    def __init__(self, structure: TreeStructure):
        self.structure = structure

    def render_node(self, img: Image, parent_global_position: np.array, node: TreeNode):
        '''
        Renders the skeleton of a single node in a tree.
        '''
        length: float = np.linalg.norm(node.local_vector)
        unit: np.array = 0.5 * node.local_vector / length
        num_units: int = math.ceil(length / 0.5)
        for n in range(num_units):
            pos: np.array = parent_global_position + (n * unit)
            x: int = (img.width // 2) + int(round(pos[0]))
            y: int = img.height - int(round(pos[2]))
            luma: int = int(round(255 * (0.5 + (0.5 * pos[1] / img.width))))
            if x >= 0 and x < img.width and y >= 0 and y < img.height:
                img[x, y] = (luma, luma, luma, 255)

    def render(self, filename):
        '''
        Renders the skeleton of the tree to a PNG file.

        Args:
            filename (str): The path of the file to save to.
        '''
        img: Image = Image(150, 400)
        self.render_node(img, np.array((0.0, 0.0, 0.0)), self.structure.root)
        img.save(filename)