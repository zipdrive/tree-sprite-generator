from __future__ import annotations
import math
import random
import numpy as np
from typing import Callable, Self, Any
from util import Vector

class TreeBranchHyperparameters:
    parent_segment_length: float
    '''
    The length of each segment of the parent branch.
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
    The tropism vector that influences the orientation of the branch segments.
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
        self.parent_segment_length = length / num_segments
        self.parent_radius = radius
        self.parent_radius_taper = radius_tapering
        self.parent_gnarliness = gnarliness
        self.parent_segments = num_segments
        self.child_num = num_children
        self.child_angle = child_branching_angle
        self.min_branching_point = child_min_branching_point
        self.tropism_vec = tropism_vector
        self.tropism_factor = tropism_factor

class TreeLeafHyperparameters:
    minimum_size: float 
    '''
    The minimum size of the leaves.
    '''

    maximum_size: float 
    '''
    The maximum size of the leaves.
    '''

    density: float 
    '''
    The average density of leaves on the branch.
    '''

    tropism_plane: Vector 
    '''
    The vector perpendicular to the tropism plane influencing the leaf orientation.
    '''

    tropism_factor: float 
    '''
    The weight of the tropism.
    '''

    def __init__(self, density: float, minimum_size: float, maximum_size: float | None = None, tropism_plane: Vector = Vector.construct(horizontal=1.0), tropism_factor: float = 0.025):
        self.minimum_size = minimum_size
        self.maximum_size = maximum_size if maximum_size != None else minimum_size
        self.density = density
        self.tropism_plane = tropism_plane
        self.tropism_factor = tropism_factor

    @property
    def average_size(self) -> float:
        '''
        The average size of the leaves.
        '''
        return 0.5 * (self.minimum_size + self.maximum_size)

class TreeLevelHyperparameters:
    branch: TreeBranchHyperparameters
    '''
    The hyperparameters for branch generation.
    '''

    leaves: TreeLeafHyperparameters
    '''
    The hyperparameters for leaf generation.
    '''

    def __init__(self, branch: TreeBranchHyperparameters, leaves: TreeLeafHyperparameters | None = None):
        self.branch = branch
        self.leaves = leaves if leaves != None else TreeLeafHyperparameters(0.0, 0.0)



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

    next_segment: Self | None = None 
    '''
    The next branch segment.
    '''


    def __init__(self, start: Vector, vec: Vector, radius: float):
        self.start = start
        self.vec = vec 
        self.radius_base = radius
        self.radius_end = radius

    @property
    def is_end_cap(self) -> bool:
        '''
        True if the segment is on the end of the branch.
        '''
        return self.next_segment == None
    
    @property 
    def end(self) -> Vector:
        '''
        The terminal end of the branch.
        '''
        return self.start + self.vec 

class TreeLeaf:
    anchor: Vector 
    '''
    The point where the leaf is anchored.
    '''

    dir: Vector 
    '''
    The direction that the leaf is pointing towards.
    '''

    size: float 
    '''
    The size of the leaf.
    '''

    def __init__(self, anchor: Vector, dir: Vector, size: float):
        self.anchor = anchor
        self.dir = dir 
        self.size = size 

class TreeStructure:
    branch_segments: list[TreeBranchSegment]
    '''
    The branch segments of the tree.
    '''

    leaves: list[TreeLeaf]
    '''
    The leaves of the tree.
    '''

    def __init__(self, structure_hyperparameters: list[TreeLevelHyperparameters]):
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
        self.leaves = []
        branch_queue: list[TreeBranch] = []
        
        def generate_branch(parent: TreeBranch | None = None, resume_from_end: bool = False) -> TreeBranch | None:
            '''
            Generates a tree branch.
            '''
            child: TreeBranch = TreeBranch(parent.level + 1 if parent != None else 0)
            if child.level >= len(structure_hyperparameters):
                return None
            child_branch_hyperparameters: TreeBranchHyperparameters = structure_hyperparameters[child.level].branch
            child_leaf_hyperparameters: TreeLeafHyperparameters = structure_hyperparameters[child.level].leaves
            
            # Figure out the starting position and orientation of the branch
            next_segment_start: Vector
            next_segment_orientation: Vector
            base_radius: float = child_branch_hyperparameters.parent_radius
            if parent != None:
                parent_orientation: Vector 
                if resume_from_end and len(parent.segments) > 0:
                    next_segment_start = parent.segments[-1].start + parent.segments[-1].vec
                    parent_orientation = parent.segments[-1].vec.normalize()
                    next_segment_orientation = parent_orientation.rotate(
                            axis=parent_orientation.cross(Vector.construct(horizontal=1.0)),
                            angle=random.random() * child_branch_hyperparameters.parent_gnarliness * math.pi / 2
                        ).rotate(
                            axis=parent_orientation,
                            angle=random.random() * math.tau
                        )
                    next_segment_radius: float = parent.segments[-1].radius_end
                    base_radius = parent.segments[-1].radius_end
                else:
                    # Randomly generate a branching point
                    parent_branch_hyperparameters: TreeBranchHyperparameters = structure_hyperparameters[parent.level].branch
                    min_branching_point: float = len(parent.segments) * parent_branch_hyperparameters.min_branching_point
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
                            angle=parent_branch_hyperparameters.child_angle, 
                            axis=norm1
                        ).rotate(
                            axis=parent_orientation,
                            angle=random.random() * math.tau
                        )
                    next_segment_radius: float = base_radius
            else:
                # No parent, so generate a basic trunk
                next_segment_start = Vector.construct()
                next_segment_orientation = Vector.construct(vertical=1.0)
                next_segment_radius: float = base_radius

            # Create each segment of the branch
            for k in range(child_branch_hyperparameters.parent_segments):
                next_segment: TreeBranchSegment = TreeBranchSegment(
                    start=next_segment_start,
                    vec=next_segment_orientation * child_branch_hyperparameters.parent_segment_length,
                    radius=next_segment_radius
                )
                if len(child.segments) > 0:
                    child.segments[-1].next_segment = next_segment
                elif resume_from_end and len(parent.segments) > 0:
                    parent.segments[-1].next_segment = next_segment
                child.segments.append(next_segment)

                # Generate leaves attached to that branch segment
                if child_leaf_hyperparameters.density > 0.0 and not np.isclose(child_leaf_hyperparameters.density, 0.0) and not np.isclose(child_leaf_hyperparameters.average_size, 0.0):
                    max_leaf_count: int = int(round(child_branch_hyperparameters.parent_segment_length / (0.125 * child_leaf_hyperparameters.average_size)))
                    for _ in range(max_leaf_count):
                        if random.random() < child_leaf_hyperparameters.density:
                            # Create leaf
                            outward_dir: Vector = next_segment_orientation.cross(Vector.construct(horizontal=1.0)).rotate(axis=next_segment_orientation, angle=math.tau * random.random())
                            tangent_dir: Vector = outward_dir.cross(next_segment_orientation)
                            axial_dist: float = random.random()
                            leaf_anchor: Vector = next_segment.start + (axial_dist * next_segment.vec) + \
                                ((next_segment.radius_base + (axial_dist * (next_segment.radius_end - next_segment.radius_base))) * outward_dir)
                            leaf_dir_random: Vector = outward_dir.rotate(axis=tangent_dir, angle=(random.random() * math.tau / 3.0) - (math.tau / 6.0))
                            leaf_dir_trop: Vector = (outward_dir - outward_dir.dot(child_leaf_hyperparameters.tropism_plane) * child_leaf_hyperparameters.tropism_plane).normalize()
                            leaf: TreeLeaf = TreeLeaf(
                                anchor=leaf_anchor,
                                dir=(leaf_dir_random * (1.0 - child_leaf_hyperparameters.tropism_factor)) + (leaf_dir_trop * child_leaf_hyperparameters.tropism_factor),
                                size=child_leaf_hyperparameters.minimum_size + (random.random() * (child_leaf_hyperparameters.maximum_size - child_leaf_hyperparameters.minimum_size))
                            )
                            self.leaves.append(leaf)

                # Determine the start of the next segment
                next_segment_start = next_segment.end

                # Taper the radius of the next segment
                next_segment_radius = base_radius * (1.0 - child_branch_hyperparameters.parent_radius_taper * k / child_branch_hyperparameters.parent_segments)
                next_segment.radius_end = next_segment_radius

                # Perturbate the orientation of the next segment
                next_segment_orientation = next_segment_orientation.rotate(
                        axis=next_segment_orientation.cross(Vector.construct(horizontal=1.0)),
                        angle=random.random() * child_branch_hyperparameters.parent_gnarliness * math.pi / 2
                    ).rotate(
                        axis=next_segment_orientation,
                        angle=random.random() * math.tau
                    )
                next_segment_orientation = (child_branch_hyperparameters.tropism_factor * child_branch_hyperparameters.tropism_vec) + ((1.0 - child_branch_hyperparameters.tropism_factor) * next_segment_orientation)
            

            return child 
        
        trunk: TreeBranch | None = generate_branch()
        if trunk != None:
            branch_queue.append(trunk)
        
        while len(branch_queue) > 0:
            parent: TreeBranch = branch_queue.pop(0)
            self.branch_segments += parent.segments

            # Generate new branches
            parent_branch_hyperparameters: TreeBranchHyperparameters = structure_hyperparameters[parent.level].branch
            for _ in range(parent_branch_hyperparameters.child_num):
                child: TreeBranch | None = generate_branch(parent, len(parent.segments) > 0 and parent.segments[-1].is_end_cap)
                if child != None:
                    branch_queue.append(child)