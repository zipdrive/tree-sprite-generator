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
            length=43.47,
            radius=20.0,
            gnarliness=0.03,
            num_segments=12,
            num_children=7,
            child_branching_angle=48*math.pi/180,
            child_min_branching_point=0.23
        ),
        TreeBranchHyperparameters(
            length=27.14,
            radius=0.63,
            gnarliness=0.25,
            num_segments=8,
            num_children=4,
            child_branching_angle=75*math.pi/180,
            child_min_branching_point=0.33
        ),
        TreeBranchHyperparameters(
            length=9.51,
            radius=0.76,
            gnarliness=0.20,
            num_segments=6,
            num_children=3,
            child_branching_angle=60*math.pi/180,
            child_min_branching_point=0.0
        ),
        TreeBranchHyperparameters(
            length=4.6,
            radius=0.70,
            gnarliness=0.09,
            num_segments=4
        )
    ]
)
'''
TreeBranchHyperparameters(
    length=43.47,
    radius=8.0,
    gnarliness=0.03,
    num_segments=12,
    num_children=7,
    child_branching_angle=48*math.pi/180,
    child_min_branching_point=0.23
),
TreeBranchHyperparameters(
    length=27.14,
    radius=0.63,
    gnarliness=0.25,
    num_segments=8,
    num_children=4,
    child_branching_angle=75*math.pi/180,
    child_min_branching_point=0.33
),
TreeBranchHyperparameters(
    length=9.51,
    radius=0.76,
    gnarliness=0.20,
    num_segments=6,
    num_children=3,
    child_branching_angle=60*math.pi/180,
    child_min_branching_point=0.0
),
TreeBranchHyperparameters(
    length=4.6,
    radius=0.70,
    gnarliness=0.09,
    num_segments=4
)
'''