import random
import math
import datetime
import os
import numpy as np
from structure import TreeNode, TreeStructureHyperparameters, TreeStructure
from graphics import TreeSkeletonRenderer, TreeRenderer

skeleton_render: TreeSkeletonRenderer = TreeSkeletonRenderer(zoom=10.0)

class Keyframe:
    cycles_min: int 
    '''
    The minimum number of growth cycles to perform.
    '''

    cycles_max: int 
    '''
    The maximum number of growth cycles to perform.
    '''

    hyperparameters: TreeStructureHyperparameters
    '''
    The growth hyperparameters at the keyframe.
    '''

    def __init__(self, cycles_min: int, cycles_max: int, hyperparameters: TreeStructureHyperparameters):
        '''
        

        Args:
            cycles_min (int): The minimum number of growth cycles to perform.
            cycles_max (int): The maximum number of growth cycles to perform.
            hyperparameters (TreeStructureHyperparameters): The growth hyperparameters at the keyframe.
        '''
        self.cycles_min = cycles_min
        self.cycles_max = cycles_max
        self.hyperparameters = hyperparameters

    def get_num_cycles(self) -> int:
        '''
        Generates a random number of cycles.
        '''
        return random.randrange(self.cycles_min, self.cycles_max + 1)

class Genus:
    name: str 
    '''
    The name of the genus.
    '''

    prolepsis: int 
    '''
    The resting period of a bud before it can sprout a new bud.
    '''

    keyframes: list[Keyframe]
    '''
    The keyframe phases of growth.
    '''

    full_renderer: TreeRenderer 
    '''
    Renders a complete tree.
    '''

    def __init__(self, \
                    name: str, \
                    prolepsis: int, \
                    keyframes: list[Keyframe]):
        '''
        
        '''
        self.name = name
        self.prolepsis = prolepsis
        self.keyframes = keyframes

        # Create the renderer
        self.full_renderer = TreeRenderer(zoom=1.0)


    def generate_tree_structure(self, render_skeleton: bool = False, render_full: bool = False) -> TreeStructure:
        directory: str = f"file/{self.name}/{datetime.datetime.now().timestamp()}"
        os.makedirs(directory, exist_ok=True)

        intermediary_directory: str = directory + "/intermediary"
        if render_skeleton or render_full:
            os.makedirs(intermediary_directory, exist_ok=True)

        structure: TreeStructure = TreeStructure( \
            prolepsis=self.prolepsis, \
            hyperparameters=self.keyframes[0].hyperparameters \
        )

        # Record the total number of iterations
        total_iterations: int = 1

        # Perform growth, interpolating hyperparameters between successive keyframes        
        iteration: int
        num_iterations: int
        for k in range(len(self.keyframes) - 1):
            keyframe: Keyframe = self.keyframes[k]
            next_keyframe: Keyframe = self.keyframes[k + 1]
            iteration = 1
            num_iterations = (self.prolepsis + 1) * keyframe.get_num_cycles()
            while iteration <= num_iterations:
                structure.grow()
                structure.hyperparameters = TreeStructureHyperparameters.interpolate(keyframe.hyperparameters, next_keyframe.hyperparameters, iteration / num_iterations)
                iteration += 1
                total_iterations += 1

                if self.prolepsis == 0 or (self.prolepsis > 0 and iteration % self.prolepsis == 0):
                    # Record progress
                    print(f"Iteration {total_iterations} ({int(100*iteration/num_iterations):03d}% of current keyframe): {structure.root.get_descendant_nodes()} nodes")

                    if render_skeleton:
                        skeleton_render.render(structure, f"{intermediary_directory}/skeleton_{iteration:03d}.png")

                    if render_full:
                        self.full_renderer.render(structure, f"{intermediary_directory}/full_{iteration:03d}.png")

        # Perform growth for the last keyframe, not interpolating with anything else
        iteration = 1
        num_iterations = (self.prolepsis + 1) * self.keyframes[-1].get_num_cycles()
        structure.hyperparameters = self.keyframes[-1].hyperparameters
        while iteration <= num_iterations:
            structure.grow()
            iteration += 1
            total_iterations += 1

            if (self.prolepsis == 0 or (self.prolepsis > 0 and iteration % self.prolepsis == 0)) and iteration <= num_iterations:
                # Record progress
                print(f"Iteration {total_iterations} ({int(100*iteration/num_iterations):03d}% of current keyframe): {structure.root.get_descendant_nodes()} nodes")

                if render_skeleton:
                    skeleton_render.render(structure, f"{intermediary_directory}/skeleton_{iteration:03d}.png")

                if render_full:
                    self.full_renderer.render(structure, f"{intermediary_directory}/full_{iteration:03d}.png")

        # Do final render
        self.full_renderer.render(structure, f"{directory}/render.png")

        return structure


###########################
# Test
###########################

test_keyframe_01: Keyframe = Keyframe(
    cycles_min=10,
    cycles_max=10,
    hyperparameters=TreeStructureHyperparameters(
        vigor=2.5, \
        main_stem_apical_lambda=0.9, \
        lateral_stem_apical_lambda=0.55, \
        growth_zeta=0.025, \
        tropism=np.array((0.0, 0.0, 0.025)), \
        shadow_a=1.0, \
        shadow_b=1.45, \
        shadow_voxel_size=1.0, \
        cull_threshold=0.01, \
    )
)
test_keyframe_02: Keyframe = Keyframe(
    cycles_min=10,
    cycles_max=10,
    hyperparameters=TreeStructureHyperparameters.alter()
)

TEST_GENUS_01: Genus = Genus( \
    name = 'test01', \
    prolepsis=4, \
    cycles_min=20, \
    cycles_max=20, \
    vigor=2.5, \
    main_stem_apical_lambda_initial=0.9, \
    main_stem_apical_lambda_decay=0.25, \
    lateral_stem_apical_lambda_final=0.55, \
    lateral_stem_apical_lambda_undecay=0.999, \
    growth_zeta=0.025, \
    upwards_growth_eta=0.025, \
    downwards_growth_eta=0.005, \
    shadow_a=1.0, \
    shadow_b=1.45, \
    shadow_voxel_size=1.0, \
    cull_threshold=0.01, \
)