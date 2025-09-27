import math
import numpy as np
import png
import typing
from structure import TreeNode, TreeBud, TreeBranch, TreeStructure

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

class Renderer:
    def render(self, structure: TreeStructure, filename: str):
        '''
        Renders the tree to a PNG file.

        Args:
            structure (TreeStructure): The structure of the tree to render.
            filename (str): The filename to render to.
        '''
        pass

class TreeSkeletonRenderer(Renderer):
    '''
    A renderer that renders only the skeleton of a tree.
    '''

    zoom: float 
    '''
    How much to zoom in by.
    '''

    def __init__(self, zoom: float = 1.0):
        self.zoom = zoom 

    def render_node(self, img: Image, parent_global_position: np.typing.NDArray[typing.Any], node: TreeNode):
        '''
        Renders the skeleton of a single node in a tree.
        '''
        length: float = np.linalg.norm(node.local_vector)
        unit: np.typing.NDArray[typing.Any] = (0.5 / self.zoom) * node.local_vector / length
        num_units: int = math.ceil(length * self.zoom / 0.5)
        for n in range(num_units):
            pos: np.typing.NDArray[typing.Any] = parent_global_position + (n * unit)
            x: int = (img.width // 2) + int(round(self.zoom * pos[0]))
            y: int = img.height - int(round(self.zoom * pos[2]))
            luma: int = min(255, max(0, int(round(255 * (0.5 + (0.5 * self.zoom * pos[1] / img.width))))))
            if x >= 0 and x < img.width and y >= 0 and y < img.height:
                _, existing_luma, _, _ = img[x, y]
                luma = max(existing_luma, luma)
                img[x, y] = (0 if isinstance(node, TreeBud) else luma, luma, 0 if isinstance(node, TreeBud) else  luma, 255)
        node_global_position: np.typing.NDArray[typing.Any] = parent_global_position + node.local_vector
        if isinstance(node, TreeBranch):
            for k in range(len(node.children)):
                #print(f"  ({node} is parent of {node.children[k]}): {node.children[k].local_vector}")
                self.render_node(img, node_global_position, node.children[k])

    def render(self, structure: TreeStructure, filename: str):
        '''
        Renders the skeleton of the tree to a PNG file.

        Args:
            structure (TreeStructure): The structure of the tree to render.
            filename (str): The path of the file to save to.
        '''
        img: Image = Image(300, 400)
        self.render_node(img, np.array((0.0, 0.0, 0.0)), structure.root)
        img.save(filename)

class TreeRenderer(Renderer):
    '''
    A renderer that renders a tree.
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

    def render_node(self, img: Image, parent_global_position: np.typing.NDArray[typing.Any], node: TreeNode) -> float:
        '''
        Renders the skeleton of a single node in a tree.
        '''

        # Render children
        node_global_position: np.typing.NDArray[typing.Any] = parent_global_position + node.local_vector
        width: float = 0.0
        if isinstance(node, TreeBranch) and len(node.children) > 0:
            for k in range(len(node.children)):
                if isinstance(node.children[k], TreeBranch):
                    width += pow(self.render_node(img, node_global_position, node.children[k]), self.width_parameter)
                else:
                    width += 1.0
        else:
            width = 1.0

        # Calculate the width of the branch
        width = pow(width, 1.0 / self.width_parameter)
        #print(f"    Descendant node count: {node.get_descendant_nodes()}")
        #print(f"    Branch width: {width}")

        # Render the branch
        length: float = np.linalg.norm(node.local_vector)
        unit: np.typing.NDArray[typing.Any] = (0.5 / self.zoom) * node.local_vector / length
        num_units: int = math.ceil(length * self.zoom / 0.5)
        for n in range(num_units):
            pos_base: np.typing.NDArray[typing.Any] = parent_global_position + (n * unit)
            vec1: np.typing.NDArray[typing.Any] = np.array([unit[1], -unit[0], 0.0]) / math.sqrt(pow(unit[0], 2.0) + pow(unit[1], 2.0)) if abs(unit[0]) > 0.001 or abs(unit[1]) > 0.001 else np.array([1.0, 0.0, 0.0])
            vec2: np.typing.NDArray[typing.Any] = np.cross(unit, vec1) / np.linalg.norm(unit)
            angle: float = 0.0
            while angle < math.tau:
                pos: np.typing.NDArray[typing.Any] = pos_base + ((width * 0.25 / self.zoom) * ((math.cos(angle) * vec1) + (math.sin(angle) * vec2)))
                x: int = (img.width // 2) + int(round(self.zoom * pos[0]))
                y: int = img.height - int(round(self.zoom * pos[2]))
                luma: int = min(255, max(0, int(round(255 * (0.5 + (0.5 * self.zoom * pos[1] / img.width))))))
                if x >= 0 and x < img.width and y >= 0 and y < img.height:
                    _, existing_luma, _, _ = img[x, y]
                    luma = max(existing_luma, luma)
                    img[x, y] = (0 if isinstance(node, TreeBud) else luma, luma, 0 if isinstance(node, TreeBud) else  luma, 255)

                angle += math.pi / (width * self.zoom)

        if isinstance(node, TreeBranch):
            for k in range(len(node.children)):
                child_node: TreeNode = node.children[k]
                if isinstance(child_node, TreeBud):
                    self.render_node(img, node_global_position + (child_node.local_vector * width), child_node)

        return width

    def render(self, structure: TreeStructure, filename: str):
        '''
        Renders the skeleton of the tree to a PNG file.

        Args:
            structure (TreeStructure): The structure of the tree to render.
            filename (str): The path of the file to save to.
        '''
        img: Image = Image(300, 400)
        self.render_node(img, np.array((0.0, 0.0, 0.0)), structure.root)
        img.save(filename)