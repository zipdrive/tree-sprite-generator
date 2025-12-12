import random
import math
import datetime
import os
import numpy as np
from structure import TreeLevelHyperparameters, TreeBranchHyperparameters, TreeLeafHyperparameters, TreeStructure
from graphics import TreeLeafColormapRenderer, TreeLeafNormalmapRenderer, TreeColormapRenderer, TreeNormalmapRenderer, TreeRenderer, Image

class Genus:
    name: str 
    '''
    The name of the genus.
    '''

    levels: list[TreeLevelHyperparameters]
    '''
    The keyframe phases of growth.
    '''

    rendered_image: Image
    '''
    The final outputted image.
    '''

    def __init__(self, 
                    name: str, 
                    levels: list[TreeLevelHyperparameters],
                    bark_color_texture: str,
                    bark_normalmap_texture: str,
                    bark_heightmap_texture: str,
                    leaf_color_texture: str
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

        # Create the renderer
        self.rendered_image = Image(
            renderers=[
                TreeLeafColormapRenderer(bark_color_texture, leaf_color_texture, width=600, height=1200),
                TreeLeafNormalmapRenderer(bark_normalmap=bark_normalmap_texture, bark_heightmap=bark_heightmap_texture, leafmap=leaf_color_texture, x=600, width=600, height=1200),
                TreeColormapRenderer(bark_color_texture, zoom=1.0, y=1200, width=600, height=1200),
                TreeNormalmapRenderer(normalmap=bark_normalmap_texture, heightmap=bark_heightmap_texture, x=600, y=1200, width=600, height=1200)
            ],
            width=1200,
            height=2400
        )


    def generate_tree_structure(self, render_full: bool = False) -> TreeStructure:
        directory: str = f"file/{self.name}/{datetime.datetime.now().timestamp()}"
        os.makedirs(directory, exist_ok=True)

        # Generate the structure
        structure: TreeStructure = TreeStructure(self.levels)

        # Render the structure
        self.rendered_image.render(structure)
        self.rendered_image.save(f"{directory}/render.png")

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
    leaf_color_texture='assets/birch/leaf.png'
)