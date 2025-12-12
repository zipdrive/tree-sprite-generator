import random
import math
import datetime
import os
import numpy as np
from structure import TreeBranchHyperparameters, TreeStructure
from graphics import TreeBarkRenderer, TreeNormalmapRenderer, TreeRenderer, Image

class Genus:
    name: str 
    '''
    The name of the genus.
    '''

    branches: list[TreeBranchHyperparameters]
    '''
    The keyframe phases of growth.
    '''

    rendered_image: Image
    '''
    The final outputted image.
    '''

    def __init__(self, 
                    name: str, 
                    branches: list[TreeBranchHyperparameters],
                    color_texture: str,
                    normalmap_texture: str,
                    heightmap_texture: str
                    ):
        '''
        Args:
            name (str): The name of the genus.
            branches (list[TreeBranchHyperparameters]): The hyperparameters for each branch.
        '''
        self.name = name
        self.branches = branches

        # Create the renderer
        self.rendered_image = Image(
            renderers=[
                TreeBarkRenderer(color_texture, zoom=1.0, width=600, height=1200),
                TreeNormalmapRenderer(normalmap=normalmap_texture, heightmap=heightmap_texture, x=600, width=600, height=1200)
            ],
            width=1200,
            height=1200
        )


    def generate_tree_structure(self, render_full: bool = False) -> TreeStructure:
        directory: str = f"file/{self.name}/{datetime.datetime.now().timestamp()}"
        os.makedirs(directory, exist_ok=True)

        # Generate the structure
        structure: TreeStructure = TreeStructure(self.branches)

        # Render the structure
        self.rendered_image.render(structure)
        self.rendered_image.save(f"{directory}/render.png")

        return structure


###########################
# Birch trees
###########################

BIRCH_LARGE: Genus = Genus( 
    name='Betula', 
    branches=[
        TreeBranchHyperparameters(
            length=600.47,
            radius=25.0,
            gnarliness=0.5,
            num_segments=12,
            num_children=15,
            child_branching_angle=48*math.pi/180,
            child_min_branching_point=0.23,
            tropism_factor=0.3
        ),
        TreeBranchHyperparameters(
            length=300.14,
            radius=0.35,
            gnarliness=0.5,
            num_segments=8,
            num_children=6,
            child_branching_angle=75*math.pi/180,
            child_min_branching_point=0.33
        ),
        TreeBranchHyperparameters(
            length=40.51,
            radius=0.4,
            gnarliness=0.20,
            num_segments=6,
            num_children=3,
            child_branching_angle=60*math.pi/180,
            child_min_branching_point=0.0
        ),
        TreeBranchHyperparameters(
            length=8.6,
            radius=0.70,
            gnarliness=0.09,
            num_segments=4
        )
    ],
    color_texture='assets/birch/color.png',
    normalmap_texture='assets/birch/normal.png',
    heightmap_texture='assets/birch/height.png'
)