import random
import math
import datetime
import os
import json
import numpy as np
from util import Vector
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


    def generate_tree_structure(self, render_leaves: bool = True, render_bare: bool = True, render_sample: bool = True) -> TreeStructure:
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
            max_y = max(branch_segment.end.vertical + 0.5 * (branch_segment.end.depth + branch_segment.radius_end * abs(Vector.construct(depth=1.0).cross(branch_segment.vec.normalize()).cross(branch_segment.vec.normalize()).depth)), max_y)
        width: int = int(math.ceil(2.0 * max_x)) + 20
        height: int = int(math.ceil(max_y + 0.5 * structure.branch_segments[0].radius_base)) + 10

        # Render tree with leaves
        if render_leaves:
            self.render_with_leaves[0].width = width 
            self.render_with_leaves[0].height = height 
            self.render_with_leaves[1].width = width 
            self.render_with_leaves[1].height = height
            self.render_with_leaves[1].x = width
            img_with_leaves: Image = Image(renderers=self.render_with_leaves)
            img_with_leaves.render(structure)
            img_with_leaves.save(f"{directory}/{structure.seed}_leaves.png")

        # Render tree without leaves
        if render_bare:
            self.render_bare[0].width = width 
            self.render_bare[0].height = height 
            self.render_bare[1].width = width 
            self.render_bare[1].height = height
            self.render_bare[1].x = width
            img_bare: Image = Image(renderers=self.render_bare)
            img_bare.render(structure)
            img_bare.save(f"{directory}/{structure.seed}_bare.png")

        # Save metadata for the tree with/without leaves
        metadata: dict = {}
        metadata['palette'] = {}
        metadata['palette']['x'] = 0
        metadata['palette']['y'] = 0
        metadata['palette']['width'] = width 
        metadata['palette']['height'] = height
        metadata['normalmap'] = {}
        metadata['normalmap']['x'] = width
        metadata['normalmap']['y'] = 0
        metadata['normalmap']['width'] = width 
        metadata['normalmap']['height'] = height
        metadata['center'] = {}
        metadata['center']['x'] = 0.5 * width
        metadata['center']['y'] = height - 0.5 * structure.branch_segments[0].radius_base
        if render_leaves:
            with open(f"{directory}/{structure.seed}_leaves.json", 'w') as metadata_with_leaves_file:
                metadata_with_leaves_file.write(json.dumps(metadata))
        if render_bare:
            with open(f"{directory}/{structure.seed}_bare.json", 'w') as metadata_bare_file:
                metadata_bare_file.write(json.dumps(metadata))

        # Render sample with lighting
        if render_sample:
            self.render_sample[0].width = width 
            self.render_sample[0].height = height 
            img_sample: Image = Image(renderers=self.render_sample)
            img_sample.render(structure)
            img_sample.save(f"{directory}/{structure.seed}_sample.png")

        return structure


#region Birch trees

BIRCH_YOUNG: Genus = Genus( 
    name='Betula', 
    levels=[
        TreeLevelHyperparameters(
            branch=TreeBranchHyperparameters(
                length=150.47,
                radius=10.0,
                gnarliness=0.1,
                num_segments=12,
                num_children=12,
                child_branching_angle=48*math.pi/180,
                child_min_branching_point=0.35,
                tropism_factor=0.35
            )
        ),
        TreeLevelHyperparameters(
            branch=TreeBranchHyperparameters(
                length=75.14,
                radius=0.25,
                gnarliness=0.1,
                num_segments=8,
                num_children=6,
                child_branching_angle=48*math.pi/180,
                child_min_branching_point=0.33,
                tropism_factor=0.05
            ),
            leaves=TreeLeafHyperparameters(
                density=0.25,
                minimum_size=8.0,
                maximum_size=12.0
            )
        ),
        TreeLevelHyperparameters(
            branch=TreeBranchHyperparameters(
                length=25.51,
                radius=0.6,
                gnarliness=0.50,
                num_segments=6,
                num_children=3,
                child_branching_angle=60*math.pi/180,
                child_min_branching_point=0.0
            ),
            leaves=TreeLeafHyperparameters(
                density=0.4,
                minimum_size=7.0,
                maximum_size=11.0
            )
        ),
        TreeLevelHyperparameters(
            branch=TreeBranchHyperparameters(
                length=11.6,
                radius=0.70,
                gnarliness=0.09,
                num_segments=4
            ),
            leaves=TreeLeafHyperparameters(
                density=0.975,
                minimum_size=5.0,
                maximum_size=9.0
            )
        )
    ],
    bark_color_texture='assets/birch/color.png',
    bark_normalmap_texture='assets/birch/normal.png',
    bark_heightmap_texture='assets/birch/height.png',
    leaf_color_texture='assets/birch/leaf.png',
    sample_bark_primary_color=(192, 190, 195),
    sample_bark_secondary_color=(69, 64, 61),
    sample_leaf_color=(223, 133, 0)#(77, 96, 65)
)

BIRCH_OLD: Genus = Genus( 
    name='Betula', 
    levels=[
        TreeLevelHyperparameters(
            branch=TreeBranchHyperparameters(
                length=300.47,
                radius=15.0,
                gnarliness=0.1,
                num_segments=12,
                num_children=20,
                child_branching_angle=48*math.pi/180,
                child_min_branching_point=0.35,
                tropism_factor=0.35
            )
        ),
        TreeLevelHyperparameters(
            branch=TreeBranchHyperparameters(
                length=125.14,
                radius=0.25,
                gnarliness=0.1,
                num_segments=8,
                num_children=8,
                child_branching_angle=48*math.pi/180,
                child_min_branching_point=0.33,
                tropism_factor=0.05
            )
        ),
        TreeLevelHyperparameters(
            branch=TreeBranchHyperparameters(
                length=50.51,
                radius=0.6,
                gnarliness=0.50,
                num_segments=6,
                num_children=3,
                child_branching_angle=60*math.pi/180,
                child_min_branching_point=0.0
            ),
            leaves=TreeLeafHyperparameters(
                density=0.5,
                minimum_size=8.0,
                maximum_size=12.0
            )
        ),
        TreeLevelHyperparameters(
            branch=TreeBranchHyperparameters(
                length=11.6,
                radius=0.70,
                gnarliness=0.09,
                num_segments=4
            ),
            leaves=TreeLeafHyperparameters(
                density=0.975,
                minimum_size=5.0,
                maximum_size=9.0
            )
        )
    ],
    bark_color_texture='assets/birch/color.png',
    bark_normalmap_texture='assets/birch/normal.png',
    bark_heightmap_texture='assets/birch/height.png',
    leaf_color_texture='assets/birch/leaf.png',
    sample_bark_primary_color=(192, 190, 195),
    sample_bark_secondary_color=(69, 64, 61),
    sample_leaf_color=(223, 133, 0)#(77, 96, 65)
)

#endregion

#region Ash trees

ASH_OLD: Genus = Genus( 
    name='Fraxinus', 
    levels=[
        TreeLevelHyperparameters(
            branch=TreeBranchHyperparameters(
                length=100.47,
                radius=30.0,
                radius_tapering=0.1,
                gnarliness=0.05,
                num_segments=4,
                num_children=3,
                child_branching_angle=80*math.pi/180,
                child_min_branching_point=0.8,
                tropism_factor=0.3
            )
        ),
        TreeLevelHyperparameters(
            branch=TreeBranchHyperparameters(
                length=100.14,
                radius=0.75,
                radius_tapering=0.2,
                gnarliness=0.2,
                num_segments=8,
                num_children=6,
                child_branching_angle=75*math.pi/180,
                child_min_branching_point=0.33,
                tropism_factor=0.1
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
    bark_color_texture='assets/basic/color.png',
    bark_normalmap_texture='assets/basic/normal.png',
    bark_heightmap_texture='assets/basic/height.png',
    leaf_color_texture='assets/ash/leaf.png',
    sample_bark_primary_color=(82, 76, 64),
    sample_bark_secondary_color=(161, 154, 138),
    sample_leaf_color=(130, 154, 6)
)

#endregion