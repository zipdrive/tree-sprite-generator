import random
import math
import datetime
import os
import numpy as np
from structure import TreeBranchHyperparameters, TreeStructure
from graphics import TreeSimpleRenderer, TreeRenderer

class Genus:
    name: str 
    '''
    The name of the genus.
    '''

    branches: list[TreeBranchHyperparameters]
    '''
    The keyframe phases of growth.
    '''

    full_renderer: TreeRenderer 
    '''
    Renders a complete tree.
    '''

    def __init__(self, 
                    name: str, 
                    branches: list[TreeBranchHyperparameters]
                    ):
        '''
        Args:
            name (str): The name of the genus.
            branches (list[TreeBranchHyperparameters]): The hyperparameters for each branch.
        '''
        self.name = name
        self.branches = branches

        # Create the renderer
        self.full_renderer = TreeSimpleRenderer(zoom=1.0)


    def generate_tree_structure(self, render_full: bool = False) -> TreeStructure:
        directory: str = f"file/{self.name}/{datetime.datetime.now().timestamp()}"
        os.makedirs(directory, exist_ok=True)

        # Generate the structure
        structure: TreeStructure = TreeStructure(self.branches)

        # Render the structure
        self.full_renderer.render(structure, f"{directory}/render.png")

        return structure


###########################
# Test
###########################

TEST_GENUS_01: Genus = Genus( 
    name='test01', 
    branches=[
        TreeBranchHyperparameters(
            length=20,
            radius=2.0,
            gnarliness=0.1,
            num_children=6
        ),
        TreeBranchHyperparameters(
            length=20,
            
        )
    ]
)