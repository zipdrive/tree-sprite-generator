from __future__ import annotations
import math
import random
import numpy as np
from typing import Callable, Self, Any
from util import Vector

class TreeBranchHyperparameters:
    parent_length: float
    '''
    The length of the parent branch.
    '''

    parent_radius: float 
    '''
    The radius of the parent branch.
    '''

    parent_radius_taper: float 
    '''
    The overall tapering of the radius of the parent branch.
    '''

    parent_gnarliness: float 
    '''
    Perturbations in the parent branch.
    '''

    parent_segments: int 
    '''
    The number of segments in the parent branch.
    '''

    child_num: int 
    '''
    How many branches should sprout from the parent branch.
    '''

    child_angle: float 
    '''
    The angle at which lateral child branches should sprout from the parent branch.
    '''

    min_branching_point: float 
    '''
    The minimum ratio of the parent branch where a child branch can sprout.
    '''

    tropism_vec: Vector
    '''
    The tropism vector that influences the weight.
    '''

    tropism_factor: float 
    '''
    The weight of the tropism.
    '''

    def __init__(self,
        length: float,
        radius: float,
        radius_tapering: float = 0.7,
        gnarliness: float = 0.05,
        num_segments: int = 7,
        num_children: int = 0,
        child_branching_angle: float = math.pi / 3,
        child_min_branching_point: float = 0.3,
        tropism_vector: Vector = Vector.construct(vertical=1.0),
        tropism_factor: float = 0.025
        ):
        '''
        Args:
            length (float): The length of the branch.
            radius (float): The radius of the branch.
            radius_tapering (float): How much the branch tapers down by the end.
            gnarliness (float): How twisted and gnarled the branch gets. 0 is ungnarled, 1 is very gnarled.
            num_segments (int): How many segments there are in the branch.
            num_children (int): How many branches branch from this one.
            child_branching_angle (float): The angle at which branches branch from this one.
            child_min_branching_point (float): Child branches will not sprout between the start of the branch and this point (relative to the start and end of the branch). 0 allows child branches to sprout anywhere on the branch, 0.5 only allows child branches to sprout starting halfway down the branch, and so on.
            tropism_vector (np.typing.NDArray[np.floating[Any]]): A direction that the branch will grow towards.
            tropism_factor (float): The strength of the tropism vector.
        '''
        self.parent_length = length / num_segments
        self.parent_radius = radius
        self.parent_radius_taper = radius_tapering
        self.parent_gnarliness = gnarliness
        self.parent_segments = num_segments
        self.child_num = num_children
        self.child_angle = child_branching_angle
        self.min_branching_point = child_min_branching_point
        self.tropism_vec = tropism_vector
        self.tropism_factor = tropism_factor



class TreeBranchSegment:
    start: Vector
    '''
    The root position of the branch.
    '''

    vec: Vector
    '''
    The direction and magnitude of the branch.
    '''

    radius_base: float 
    '''
    The radius of the base end.
    '''

    radius_end: float 
    '''
    The radius of the terminal end.
    '''

    is_end_cap: bool = False 
    '''
    True if the segment is on the end of the branch.
    '''

    def __init__(self, start: Vector, vec: Vector, radius: float):
        self.start = start
        self.vec = vec 
        self.radius_base = radius
        self.radius_end = radius

class TreeStructure:
    branch_segments: list[TreeBranchSegment]
    '''
    The branch segments of the tree.
    '''

    def __init__(self, structure_hyperparameters: list[TreeBranchHyperparameters]):
        '''
        Generates the tree.
        '''
        class TreeBranch:
            level: int 
            '''
            The level of the branch.
            '''

            segments: list[TreeBranchSegment]
            '''
            The segments of the branch.
            '''

            def __init__(self, level: int):
                self.level = level
                self.segments = []

        self.branch_segments = []
        branch_queue: list[TreeBranch] = []
        
        def generate_branch(parent: TreeBranch | None = None) -> TreeBranch | None:
            '''
            Generates a tree branch.
            '''
            child: TreeBranch = TreeBranch(parent.level + 1 if parent != None else 0)
            if child.level >= len(structure_hyperparameters):
                return None
            
            # Figure out the starting position and orientation of the branch
            child_hyperparameters: TreeBranchHyperparameters = structure_hyperparameters[child.level]
            next_segment_start: Vector
            next_segment_orientation: Vector
            base_radius: float = child_hyperparameters.parent_radius
            if parent != None:
                # Randomly generate a branching point
                parent_hyperparameters: TreeBranchHyperparameters = structure_hyperparameters[parent.level]
                min_branching_point: float = len(parent.segments) * parent_hyperparameters.min_branching_point
                max_branching_point: float = len(parent.segments)
                ratio: float = min_branching_point + ((max_branching_point - min_branching_point) * random.random())
                q: int = int(math.floor(ratio))
                r: float = ratio - q 
                next_segment_start = parent.segments[q].start + (r * parent.segments[q].vec)
                base_radius *= parent.segments[q].radius_base

                # Calculate a branching orientation
                parent_orientation: Vector = parent.segments[q].vec.normalize()
                norm1: np.typing.NDArray[np.floating[Any]] = parent_orientation.cross(Vector.construct(horizontal=1.0))
                next_segment_orientation = parent_orientation.rotate(
                        angle=parent_hyperparameters.child_angle, 
                        axis=norm1
                    ).rotate(
                        axis=parent_orientation,
                        angle=random.random() * math.tau
                    )
            else:
                # No parent, so generate a basic trunk
                next_segment_start = Vector.construct()
                next_segment_orientation = Vector.construct(vertical=1.0)
            next_segment_radius: float = base_radius

            # Create each segment of the branch
            for _ in range(child_hyperparameters.parent_segments):
                next_segment: TreeBranchSegment = TreeBranchSegment(
                    start=next_segment_start,
                    vec=next_segment_orientation * child_hyperparameters.parent_length,
                    radius=next_segment_radius
                )
                child.segments.append(next_segment)

                # Determine the start of the next segment
                next_segment_start = next_segment.start + next_segment.vec

                # Taper the radius of the next segment
                next_segment_radius -= base_radius * child_hyperparameters.parent_radius_taper / child_hyperparameters.parent_segments
                next_segment.radius_end = next_segment_radius

                # Perturbate the orientation of the next segment
                next_segment_orientation = next_segment_orientation.rotate(
                        axis=next_segment_orientation.cross(Vector.construct(horizontal=1.0)),
                        angle=random.random() * child_hyperparameters.parent_gnarliness * math.pi / 2
                    ).rotate(
                        axis=next_segment_orientation,
                        angle=random.random() * math.tau
                    )
                next_segment_orientation = (child_hyperparameters.tropism_factor * child_hyperparameters.tropism_vec) + ((1.0 - child_hyperparameters.tropism_factor) * next_segment_orientation)
            return child 
        
        trunk: TreeBranch | None = generate_branch()
        if trunk != None:
            branch_queue.append(trunk)
        
        while len(branch_queue) > 0:
            parent: TreeBranch = branch_queue.pop(0)
            self.branch_segments += parent.segments
            
            if len(parent.segments) > 0:
                parent.segments[-1].is_end_cap = True

            # Generate new branches
            parent_hyperparameters: TreeBranchHyperparameters = structure_hyperparameters[parent.level]
            for _ in range(parent_hyperparameters.child_num):
                child: TreeBranch | None = generate_branch(parent)
                if child != None:
                    branch_queue.append(child)