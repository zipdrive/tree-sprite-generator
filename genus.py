import random
import math
import datetime
import os
import numpy as np
from structure import TreeLevelHyperparameters, TreeBranchHyperparameters, TreeLeafHyperparameters, TreeStructure
from graphics import TreeLeafColormapRenderer, TreeLeafNormalmapRenderer, TreeColormapRenderer, TreeNormalmapRenderer, TreeSampleRenderer, TreeRenderer, Image

class Genus:
    name: str 
    '''
    The name of the genus.
    '''

    levels: list[TreeLevelHyperparameters]
    '''
    The keyframe phases of growth.
    '''

    render_with_leaves: list[TreeRenderer] 
    '''
    Renders the tree with leaves.
    '''

    render_bare: list[TreeRenderer]
    '''
    Renders only the branches.
    '''

    render_sample: list[TreeRenderer] 
    '''
    Renders a sample of what the tree will look like with a color palette and lighting applied.
    '''

    def __init__(self, 
                    name: str, 
                    levels: list[TreeLevelHyperparameters],
                    bark_color_texture: str,
                    bark_normalmap_texture: str,
                    bark_heightmap_texture: str,
                    leaf_color_texture: str,
                    sample_bark_primary_color: tuple[int, int, int],
                    sample_bark_secondary_color: tuple[int, int, int],
                    sample_leaf_color: tuple[int, int, int]
                    ):
        '''
        Args:
            name (str): The name of the genus.
            levels (list[TreeLevelHyperparameters]): The hyperparameters for each branch level.
            color_texture (str): The filename for the bark color texture.
            normalmap_texture (str): The filename for the bark normalmap texture.
            heightmap_texture (str): The filename for the bark heightmap texture.
            leaf_color_texture (str): The filename for the leaf color texture.
        '''
        self.name = name
        self.levels = levels

        # Create the renderers
        self.render_with_leaves = [
                TreeLeafColormapRenderer(bark_color_texture, leaf_color_texture, width=1, height=1),
                TreeLeafNormalmapRenderer(bark_normalmap=bark_normalmap_texture, bark_heightmap=bark_heightmap_texture, leafmap=leaf_color_texture, x=1, width=1, height=1),
            ]
        self.render_bare = [
                TreeColormapRenderer(bark_color_texture, width=1, height=1),
                TreeNormalmapRenderer(normalmap=bark_normalmap_texture, heightmap=bark_heightmap_texture, x=1, width=1, height=1),
            ]
        self.render_sample = [
                TreeSampleRenderer(
                    bark_colormap=bark_color_texture,
                    bark_normalmap=bark_normalmap_texture,
                    bark_heightmap=bark_heightmap_texture,
                    leaf_colormap=leaf_color_texture,
                    primary_bark_color=sample_bark_primary_color,
                    secondary_bark_color=sample_bark_secondary_color,
                    leaf_color=sample_leaf_color
                )
            ]


    def generate_tree_structure(self) -> TreeStructure:
        # Generate the structure
        structure: TreeStructure = TreeStructure(self.levels)
        print(f"  Completed building structure with seed {structure.seed}.")

        # Create directory to store the tree files in
        directory: str = f"file/{self.name}"
        os.makedirs(directory, exist_ok=True)

        # Use the maximum abs(x) and y coordinates of any branch to determine what the width and height of the generated image should be
        max_x: float = 0.0
        max_y: float = 0.0
        for k in range(len(structure.branch_segments)):
            branch_segment = structure.branch_segments[k]
            max_x = max(abs(branch_segment.end.horizontal), max_x)
            max_y = max(branch_segment.end.vertical - branch_segment.end.depth, max_y)
        width: int = int(math.ceil(2.0 * max_x)) + 20
        height: int = int(math.ceil(max_y)) + 40

        # Render tree with leaves
        self.render_with_leaves[0].width = width 
        self.render_with_leaves[0].height = height 
        self.render_with_leaves[1].width = width 
        self.render_with_leaves[1].height = height
        self.render_with_leaves[1].x = width
        render_with_leaves: Image = Image(renderers=self.render_with_leaves)
        render_with_leaves.render(structure)
        render_with_leaves.save(f"{directory}/{structure.seed}_leaves.png")

        # Render sample with lighting
        self.render_sample[0].width = width 
        self.render_sample[0].height = height 
        render_sample: Image = Image(renderers=self.render_sample)
        render_sample.render(structure)
        render_sample.save(f"{directory}/{structure.seed}_sample.png")

        return structure


###########################
# Birch trees
###########################

BIRCH_LARGE: Genus = Genus( 
    name='Betula', 
    levels=[
        TreeLevelHyperparameters(
            branch=TreeBranchHyperparameters(
                length=600.47,
                radius=25.0,
                gnarliness=0.5,
                num_segments=12,
                num_children=15,
                child_branching_angle=48*math.pi/180,
                child_min_branching_point=0.23,
                tropism_factor=0.3
            )
        ),
        TreeLevelHyperparameters(
            branch=TreeBranchHyperparameters(
                length=300.14,
                radius=0.35,
                gnarliness=0.5,
                num_segments=8,
                num_children=6,
                child_branching_angle=75*math.pi/180,
                child_min_branching_point=0.33
            ),
            leaves=TreeLeafHyperparameters(
                density=0.75,
                minimum_size=15.0,
                maximum_size=20.0
            )
        ),
        TreeLevelHyperparameters(
            branch=TreeBranchHyperparameters(
                length=40.51,
                radius=0.4,
                gnarliness=0.20,
                num_segments=6,
                num_children=3,
                child_branching_angle=60*math.pi/180,
                child_min_branching_point=0.0
            ),
            leaves=TreeLeafHyperparameters(
                density=0.975,
                minimum_size=8.0,
                maximum_size=15.0
            )
        ),
        TreeLevelHyperparameters(
            branch=TreeBranchHyperparameters(
                length=8.6,
                radius=0.70,
                gnarliness=0.09,
                num_segments=4
            ),
            leaves=TreeLeafHyperparameters(
                density=0.975,
                minimum_size=3.0,
                maximum_size=5.0
            )
        )
    ],
    bark_color_texture='assets/birch/color.png',
    bark_normalmap_texture='assets/birch/normal.png',
    bark_heightmap_texture='assets/birch/height.png',
    leaf_color_texture='assets/birch/leaf.png',
    sample_bark_primary_color=(192, 190, 195),
    sample_bark_secondary_color=(69, 64, 61),
    sample_leaf_color=(77, 96, 65)
)