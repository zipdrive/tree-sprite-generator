from __future__ import annotations
import math
import random
import numpy as np
from typing import Callable, Self, Any

def normalized(vec: np.typing.NDArray[Any]) -> np.typing.NDArray[Any]:
    '''
    Normalizes a vector.
    '''
    return vec / np.linalg.norm(vec)

class TreeNode:
    '''
    Data structure representing a node of the tree's structure.
    '''

    main_stem: bool 
    '''
    True if this node is on the "main" stem. False otherwise.
    '''

    parent: TreeBranch | None
    '''
    The parent node.
    '''

    light: float 
    '''
    How much light the node receives, either from its leaves or from its children.
    '''

    vigor: float 
    '''
    How much vigor the node receives from its parent.
    '''

    def __init__(self, parent: TreeBranch | None, local_vector: np.typing.NDArray[Any], prolepsis: int, main_stem: bool = False):
        self.parent = parent 
        self.local_vector = local_vector
        self.main_stem = main_stem

    def get_descendant_buds(self) -> int:
        '''
        Gets the number of descendant buds.
        '''
        return 0 

    def get_descendant_nodes(self) -> int:
        '''
        Gets the number of descendant nodes.
        '''
        return 0

    def get_global_vector(self) -> np.typing.NDArray[Any]:
        '''
        The global position of the end of the node.
        '''
        return (self.parent.get_global_vector() + self.local_vector) if self.parent != None else self.local_vector

    def reset_light(self):
        '''
        Resets the light received by the node and its children to 0.
        Call this function before calculating light.
        '''
        pass

    def propagate_light(self):
        '''
        Propagates light received from the buds up to the root, and recalculates the weight of each child.
        '''
        pass

    def construct_raw_queue(self) -> list[tuple[TreeNode, float]]:
        '''
        Constructs an unsorted priority queue of each child node on the main stem of this node, excluding the children that compose the main stem branch.
        '''
        return [(self, self.light)]

    def distribute_vigor(self, vigor_received: float, weight_min: float, weight_max: float = 1.0, weight_kappa: float = 0.5):
        '''
        Distributes vigor to the children of the node.

        Args:
            vigor_received (float): The vigor received from the parent.
            weight_min (float): The minimum weight of a child.
            weight_max (float): The maximum weight of a child.
            weight_kappa (float): Parameter in the range (0, 1]. Lower values make the branches receiving the most light be more aggressively supported.
        '''
        self.vigor = vigor_received

    def cull(self, culling_ratio: float) -> list[TreeBud]:
        '''
        Culls any bud children of this node if the light received from that bud falls below a certain threshold.

        Args:
            culling_ratio (float): The threshold at which to cull.

        Returns:
            A list of all descendant buds.
        '''
        return []


class TreeBud(TreeNode):
    default_orientation: np.typing.NDArray[Any]
    '''
    The default orientation of a branch sprouting from this bud.
    '''

    remaining_resting_period: int 
    '''
    How long this bud has to wait for before it can start producing new buds.
    '''

    def __init__(self, parent: TreeBranch, local_vector: np.typing.NDArray[Any], prolepsis: int, main_stem: bool = False):
        super().__init__(parent, local_vector, main_stem)
        self.remaining_resting_period = prolepsis

    def reset_light(self):
        '''
        Resets the light received by the node and its children to 0.
        Call this function before calculating light.
        '''
        self.light = 0
        self.vigor = 0

    def cull(self, culling_ratio: float) -> list[TreeBud]:
        return [self]


class TreeBranch(TreeNode):
    local_vector: np.typing.NDArray[Any]
    '''
    The direction and magnitude of the branch, relative to the end of the previous node.
    '''

    children: list[TreeNode]
    '''
    The child nodes.
    '''

    children_weights: list[float]
    '''
    The amount that each child node is favored.
    '''

    num_descendant_buds: int 
    '''
    The total number of buds which are descendants of this node.
    '''

    num_descendant_nodes: int 
    '''
    The total number of nodes which are descendants of this node.
    '''

    apical_lambda: float 
    '''
    Parameter for apical preference. Controls whether the main child or lateral children are favored more.
    '''

    def __init__(self, parent: TreeBranch | None, local_vector: np.typing.NDArray[Any], main_stem: bool = False):
        super().__init__(parent, local_vector, main_stem)
        self.children = []
        self.children_weights = []
        self.num_descendant_buds = 0
        self.num_descendant_nodes = 0

    def get_descendant_buds(self) -> int:
        return self.num_descendant_buds
    
    def get_descendant_nodes(self) -> int:
        return self.num_descendant_nodes

    def reset_light(self):
        self.light = 0
        self.vigor = 0
        for k in range(len(self.children)):
            self.children[k].reset_light()
            
    def propagate_light(self):
        # Determine the average amount of light being received by the buds from each child
        self.num_descendant_buds = 0
        self.num_descendant_nodes = 0
        for k in range(len(self.children)):
            child: TreeNode = self.children[k]
            child.propagate_light()
            child_num_buds: int = child.get_descendant_buds() + (1 if isinstance(child, TreeBud) else 0)
            self.num_descendant_buds += child_num_buds
            self.num_descendant_nodes += child.get_descendant_nodes() + 1
            self.light += child.light

    def construct_raw_queue(self) -> list[tuple[TreeNode, float]]:
        if len(self.children) > 0:
            main_raw_queue: list[tuple[TreeNode, float]] = self.children[0].construct_raw_queue()
            for k in range(1, len(self.children)):
                main_raw_queue.append((self.children[k], self.children[k].light / (1 + self.children[k].get_descendant_nodes())))
            return main_raw_queue
        return []

    def distribute_vigor(self, vigor_received: float, weight_min: float, weight_max: float = 1.0, weight_kappa: float = 0.5):
        self.vigor = vigor_received

        raw_queue: list[tuple[TreeNode, float]] = self.construct_raw_queue()
        priority_queue: list[tuple[TreeNode, float]] = [(raw_queue[k][0], (self.apical_lambda if k == 0 else (1.0 - self.apical_lambda)) * raw_queue[k][1]) for k in range(len(raw_queue))]
        priority_queue.sort(key=lambda tup: -tup[1])

        weights: list[float] = []
        for k in range(len(priority_queue)):
            k_pct: float
            try:
                k_pct = min(1.0, k / (weight_kappa * (len(priority_queue) - 1)))
            except ZeroDivisionError:
                k_pct = 0.0
            weights.append(weight_max + ((weight_min - weight_max) * k_pct))
        total_weight: float = sum(weights)
        for k in range(len(priority_queue)):
            node, _ = priority_queue[k]
            try:
                node.distribute_vigor(vigor_received * weights[k] / total_weight, weight_min, weight_max, weight_kappa)
            except ZeroDivisionError:
                # I don't think this should ever happen, tbh
                node.distribute_vigor(vigor_received, weight_min, weight_max, weight_kappa)

    def cull(self, culling_ratio: float) -> list[TreeBud]:
        if hasattr(self, 'light') and self.num_descendant_nodes > 0 and self.light / self.num_descendant_nodes < culling_ratio:
            #print(f"  {self.num_descendant_nodes} nodes ({self.num_descendant_buds} buds) culled. Total light collected from buds was {self.light}.")
            self.children = []
            return []
        else:
            buds: list[TreeBud] = []
            for k in range(len(self.children)):
                buds += self.children[k].cull(culling_ratio)
            return buds




class TreeStructureHyperparameters:
    vigor: float 
    '''
    Parameter that controls how many branches are sprouted.
    '''

    priority_min: float 
    '''
    The weight of low-priority branches in terms of receiving growth resources.
    '''

    priority_kappa: float
    '''
    Parameter in the range (0, 1]. Lower values increase the strictness of which branches are considered high-priority.
    '''

    apical_theta: float 
    '''
    The angle at which new branches are branched off.
    In nature, this angle is usually one of the following: [PI*1/3, PI*2/5, PI*3/8, PI*5/13]
    '''
    
    main_stem_apical_lambda: float 
    '''
    Controls how much the primary branch is favored in the main stem.
    '''

    lateral_stem_apical_lambda: float
    '''
    Controls how much the primary branch is favored in the lateral stems.
    '''

    growth_zeta: float 
    '''
    The growth direction parameter zeta.
    '''

    tropism: np.typing.NDArray[Any] 
    '''
    The tropism vector.
    '''

    shadow_a: float 
    '''
    The shadow parameter a.
    '''

    shadow_b: float 
    '''
    The shadow parameter b.
    '''

    shadow_voxel_size: float 
    '''
    The size of shadow voxels.
    '''

    cull_threshold: float
    '''
    The threshold to cull branches at.
    '''

    def __init__(self, \
                    vigor: float = 1.0, \
                    priority_min: float = 0.9, \
                    priority_kappa: float = 0.75, \
                    apical_theta: float = math.pi * 2 / 5, \
                    main_stem_apical_lambda: float = 0.9, \
                    lateral_stem_apical_lambda: float = 0.9, \
                    growth_zeta: float = 0.2, \
                    tropism: np.typing.NDArray[Any] = np.array((0.0, 0.0, -0.05)), \
                    shadow_a: float = 1.0, \
                    shadow_b: float = 16.0, \
                    shadow_voxel_size: float = 1.0, \
                    cull_threshold: float = 0.01):
        '''
        Args:
            prolepsis (int): The number of cycles that a new bud rests for before it can start producing new buds. Set to 0 for sylleptic branching, or set to >0 for proleptic branching.
            shadow_a (float): Parameter for determining how buds overshadow other buds. Controls the magnitude of shadows.
            shadow_b (float): Parameter for determining how buds overshadow other buds. Controls the dropoff of shadows over increasing vertical distance.
            shadow_voxel_size (float): The resolution of shadow voxels.
            apical_theta (float): The angle at which new branches will branch off.
            growth_zeta (float): Parameter for determining which direction buds grow in. Controls how strong the tendency to grow towards light is.
            tropism (np.array): A three-dimensional tropism vector that represents a preferred direction and magnitude (e.g. downwards, to simulate gravity).
            cull_threshold (float): The threshold of average light gathered by a branch for that branch to avoid being shed.
        '''
        self.vigor = vigor
        self.priority_min = priority_min
        self.priority_kappa = priority_kappa
        self.apical_theta = apical_theta
        self.main_stem_apical_lambda = main_stem_apical_lambda
        self.lateral_stem_apical_lambda = lateral_stem_apical_lambda
        self.growth_zeta = growth_zeta
        self.tropism = tropism
        self.shadow_a = shadow_a
        self.shadow_b = shadow_b
        self.shadow_voxel_size = shadow_voxel_size
        self.cull_threshold = cull_threshold

    @classmethod
    def from_other(cls, original: TreeStructureHyperparameters, \
                    vigor: float | None = None, \
                    priority_min: float | None = None, \
                    priority_kappa: float | None = None, \
                    apical_theta: float | None = None, \
                    main_stem_apical_lambda: float | None = None, \
                    lateral_stem_apical_lambda: float | None = None, \
                    growth_zeta: float | None = None, \
                    tropism: np.typing.NDArray[Any] | None = None, \
                    shadow_a: float | None = None, \
                    shadow_b: float | None = None, \
                    shadow_voxel_size: float | None = None, \
                    cull_threshold: float | None = None) -> TreeStructureHyperparameters:
        return cls(
            vigor=vigor if vigor != None else original.vigor,
            priority_min=priority_min if priority_min != None else original.priority_min,
            priority_kappa=priority_kappa if priority_kappa != None else original.priority_kappa,
            apical_theta=apical_theta if apical_theta != None else original.apical_theta,
            main_stem_apical_lambda=main_stem_apical_lambda if main_stem_apical_lambda != None else original.main_stem_apical_lambda,
            lateral_stem_apical_lambda=lateral_stem_apical_lambda if lateral_stem_apical_lambda != None else original.lateral_stem_apical_lambda,
            growth_zeta=growth_zeta if growth_zeta != None else original.growth_zeta,
            tropism=tropism if type(tropism) == np.ndarray else original.tropism,
            shadow_a=shadow_a if shadow_a != None else original.shadow_a,
            shadow_b=shadow_b if shadow_b != None else original.shadow_b,
            shadow_voxel_size=shadow_voxel_size if shadow_voxel_size != None else original.shadow_voxel_size,
            cull_threshold=cull_threshold if cull_threshold != None else original.cull_threshold
        )

    @staticmethod 
    def interpolate(lhs: TreeStructureHyperparameters, rhs: TreeStructureHyperparameters, a: float) -> TreeStructureHyperparameters:
        return TreeStructureHyperparameters(
            vigor=lhs.vigor + (a * (rhs.vigor - lhs.vigor)),
            priority_min=lhs.priority_min + (a * (rhs.priority_min - lhs.priority_min)),
            priority_kappa=lhs.priority_kappa + (a * (rhs.priority_kappa - lhs.priority_kappa)),
            apical_theta=lhs.apical_theta + (a * (rhs.apical_theta - lhs.apical_theta)),
            main_stem_apical_lambda=lhs.main_stem_apical_lambda + (a * (rhs.main_stem_apical_lambda - lhs.main_stem_apical_lambda)),
            lateral_stem_apical_lambda=lhs.lateral_stem_apical_lambda + (a * (rhs.lateral_stem_apical_lambda - lhs.lateral_stem_apical_lambda)),
            growth_zeta=lhs.growth_zeta + (a * (rhs.growth_zeta - lhs.growth_zeta)),
            tropism=lhs.tropism + (a * (rhs.tropism - lhs.tropism)),
            shadow_a=lhs.shadow_a + (a * (rhs.shadow_a - lhs.shadow_a)),
            shadow_b=lhs.shadow_b + (a * (rhs.shadow_b - lhs.shadow_b)),
            shadow_voxel_size=lhs.shadow_voxel_size + (a * (rhs.shadow_voxel_size - lhs.shadow_voxel_size)),
            cull_threshold=lhs.cull_threshold + (a * (rhs.cull_threshold - lhs.cull_threshold))
        )

    def get_apical_lambda(self, node: TreeNode) -> float:
        '''
        Gets the apical lambda for a node.

        Args:
            node (TreeNode): The node to get the apical lambda for.
        '''
        return self.main_stem_apical_lambda if node.main_stem else self.lateral_stem_apical_lambda


class TreeNodeShadowVoxel:
    '''
    Data structure representing a single voxel of shadows.
    '''

    nodes: list[TreeNode]
    '''
    The buds within the voxel.
    '''

    shadow: float 
    '''
    How much the voxel is shaded.
    '''

    optimal_growth_direction: np.typing.NDArray[Any] 
    '''
    The optimal growth direction for buds inside this voxel.
    '''

    def __init__(self, startingNode: TreeNode | None = None):
        self.nodes = [startingNode] if startingNode != None else []
        self.shadow = 0.0

class TreeNodeShadowVoxels:
    '''
    Data structure for storing all voxels of shadows.
    '''

    shadow_a: float 
    '''
    The shadow parameter a.
    '''

    shadow_b: float 
    '''
    The shadow parameter b.
    '''

    shadow_voxel_size: float 
    '''
    The size of shadow voxels.
    '''

    voxels: dict[tuple[int, int, int], TreeNodeShadowVoxel]
    '''
    The voxels of shadows.    
    '''

    def __init__(self, hyperparameters: TreeStructureHyperparameters):
        self.shadow_a = hyperparameters.shadow_a
        self.shadow_b = hyperparameters.shadow_b
        self.shadow_voxel_size = hyperparameters.shadow_voxel_size
        self.voxels = {}

    def add(self, bud: TreeNode):
        '''
        Adds a bud to the corresponding shadow voxel.

        Args:
            bud (TreeNode): The bud to add.
        '''
        bud_global_vector: np.typing.NDArray[Any] = bud.get_global_vector()
        x: int = int(math.floor((float(bud_global_vector[0]) / self.shadow_voxel_size) + 0.5))
        y: int = int(math.floor((float(bud_global_vector[1]) / self.shadow_voxel_size) + 0.5))
        z: int = int(math.floor((float(bud_global_vector[2]) / self.shadow_voxel_size) + 0.5))
        if (x, y, z) in self.voxels:
            self.voxels[(x, y, z)].nodes.append(bud)
        else:
            self.voxels[(x, y, z)] = TreeNodeShadowVoxel(bud)

    def update_light(self):
        '''
        Updates the light received by each bud, and by the parents of each bud.
        '''
        keys: list[tuple[int, int, int]] = [key for key in self.voxels]
        keys.sort(key=lambda key: key[2])
        for key_idx in range(len(keys)):

            # Calculate shadows propagated from the voxels above
            key: tuple[int, int, int] = keys[key_idx]
            voxel: TreeNodeShadowVoxel = self.voxels[key]
            for key_above_idx in range(key_idx + 1, len(keys)):
                key_above: tuple[int, int, int] = keys[key_above_idx]
                if abs(key_above[0] - key[0]) < key_above[2] - key[2] and abs(key_above[1] - key[1]) < key_above[2] - key[2]:
                    voxel.shadow += self.shadow_a * math.pow(self.shadow_b, key[2] - key_above[2])

            # Apply shadows to each bud inside the voxel
            for k in range(len(voxel.nodes)):
                voxel.nodes[k].light = max(1.0 - voxel.shadow, 0.0)

        # Calculate the optimal growth direction for each voxel with buds in it
        for key_idx in range(len(keys)):
            key: tuple[int, int, int] = keys[key_idx]
            least_shadow: float = math.inf
            least_shadow_dir: np.typing.NDArray[Any] = np.array((0.0, 0.0, 1.0))

            x, y, z = key
            for dx in [0, -1, 1]:
                for dy in [0, -1, 1]:
                    for dz in [1, 0, -1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        if z + dz < 0:
                            continue
                        key_adj: tuple[int, int, int] = (x + dx, y + dy, z + dz)
                        if key_adj not in self.voxels:
                            # Construct voxel and calculate shadow
                            adj_voxel: TreeNodeShadowVoxel = TreeNodeShadowVoxel()
                            self.voxels[key_adj] = adj_voxel
                            for key_above_idx in range(len(keys)):
                                key_above_x, key_above_y, key_above_z = keys[key_above_idx]
                                if key_above_z > key_adj[2] and \
                                    abs(key_above_x - key_adj[0]) < key_above_z - key_adj[2] and \
                                    abs(key_above_y - key_adj[1]) < key_above_z - key_adj[2]:
                                    adj_voxel.shadow += self.shadow_a * math.pow(self.shadow_b, key_adj[2] - key_above_z)

                        # Check if adjacent voxel has lowest shadow found so far
                        if self.voxels[key_adj].shadow < least_shadow:
                            least_shadow = self.voxels[key_adj].shadow
                            least_shadow_dir = np.array((dx, dy, dz)) 

            # Record the direction of the voxel with the least amount of shadow
            self.voxels[key].optimal_growth_direction = least_shadow_dir

    def get_optimal_growth_direction(self, bud: TreeNode) -> np.typing.NDArray[Any]:
        '''
        Finds the optimal growth direction for the bud.

        Args:
            bud (TreeNode): The bud to get the optimal growth direction for.
        '''
        bud_global_vector: np.typing.NDArray[Any] = bud.get_global_vector()
        x: int = int(math.floor((float(bud_global_vector[0]) / self.shadow_voxel_size) + 0.5))
        y: int = int(math.floor((float(bud_global_vector[1]) / self.shadow_voxel_size) + 0.5))
        z: int = int(math.floor((float(bud_global_vector[2]) / self.shadow_voxel_size) + 0.5))
        if (x, y, z) in self.voxels:
            return self.voxels[(x, y, z)].optimal_growth_direction
        else:
            return np.array([0.0, 0.0, 1.0])

            



class TreeStructure:
    prolepsis: int 
    '''
    The resting period after a new bud is formed before it itself can form a new bud.
    '''

    hyperparameters: TreeStructureHyperparameters
    '''
    The hyperparameters of structure growth.
    '''


    root: TreeNode 
    '''
    The root of the tree.
    '''

    buds: list[TreeBud]
    '''
    The terminal ends of the tree.
    '''

    def __init__(self, \
                    prolepsis: int, \
                    hyperparameters: TreeStructureHyperparameters):
        '''
        Initializes a tree.

        Args:
            prolepsis (int): The number of cycles that a new bud rests for before it can start producing new buds. Set to 0 for sylleptic branching, or set to >0 for proleptic branching.
            hyperparameters (TreeStructureHyperparameters): The hyperparameters of growth.
        '''
        self.prolepsis = prolepsis
        self.hyperparameters = hyperparameters

        # Create the root node
        self.root = TreeBranch(None, np.array([0.0, 0.0, 1.0]), main_stem=True)
        first_bud: TreeBud = TreeBud(self.root, self.root.local_vector, 0, main_stem=True)
        self.root.apical_lambda = self.hyperparameters.get_apical_lambda(self.root)
        self.root.light = 1.0
        self.root.children = [first_bud]
        self.root.children_weights = [1.0]
        self.buds = [first_bud]
    
    def grow(self):
        '''
        Performs the next step of growth.

        Args:
            vigor (float): The vigor that is injected into the tree.
        '''

        # Update light received by buds
        self.root.reset_light()
        voxels: TreeNodeShadowVoxels = TreeNodeShadowVoxels(self.hyperparameters)
        for k in range(len(self.buds)):
            bud: TreeBud = self.buds[k]
            voxels.add(bud)
        voxels.update_light() # Updates the light received by each bud
        self.root.propagate_light() # Updates the light received by parent nodes, grandparent nodes, etc.

        # Update vigor of each bud
        self.root.distribute_vigor(self.hyperparameters.vigor * self.root.light, self.hyperparameters.priority_min, 1.0, self.hyperparameters.priority_kappa)

        # Sprout new branches from buds
        growth_eta: float = np.linalg.norm(self.hyperparameters.tropism)
        for k in range(len(self.buds)):
            bud: TreeBud = self.buds[k]
            if bud.remaining_resting_period == 0 and bud.vigor >= 1.0: # Sprout if bud is not resting and has enough resources to grow

                # Determine how many buds to sprout from the end of the new branch
                num_sprouted_buds: int = math.floor(bud.vigor)
                branch_length: float = bud.vigor / num_sprouted_buds

                # Determine the directions of each bud
                bud_global_pos: np.typing.NDArray[Any] = bud.get_global_vector()
                straight_direction: np.typing.NDArray[Any] = normalized(bud.local_vector)
                straight_normal_direction1: np.typing.NDArray[Any] = normalized(np.array([0.0, -straight_direction[2], straight_direction[1]]))
                straight_normal_direction2: np.typing.NDArray[Any] = np.cross(straight_direction, straight_normal_direction1)
                optimal_growth_direction: np.typing.NDArray[Any] = voxels.get_optimal_growth_direction(bud)
                twist: float = random.random() * math.tau 

                # Create the new branch
                branch: TreeBranch = TreeBranch(bud.parent, straight_direction * branch_length, bud.main_stem)
                branch.apical_lambda = self.hyperparameters.get_apical_lambda(branch)

                # Create the buds sprouting from the end of the new branch
                for n in range(num_sprouted_buds):
                    expected_direction: np.typing.NDArray[Any]
                    if n == 0:
                        expected_direction = straight_direction
                    else:
                        twist += random.random() * math.tau / (num_sprouted_buds - 1)
                        expected_direction = (math.cos(self.hyperparameters.apical_theta) * straight_direction) + (math.sin(self.hyperparameters.apical_theta) * ((math.cos(twist) * straight_normal_direction1) + (math.sin(twist) * straight_normal_direction2)))
                    new_bud: TreeBud = TreeBud( \
                        parent=branch, \
                        local_vector=\
                            (expected_direction * (1.0 - self.hyperparameters.growth_zeta - growth_eta)) + \
                            (optimal_growth_direction * self.hyperparameters.growth_zeta) + \
                            self.hyperparameters.tropism \
                        , \
                        prolepsis=0 if n == 0 else self.prolepsis, \
                        main_stem=(bud.main_stem and n == 0)
                    )
                    if bud_global_pos[2] + new_bud.local_vector[2] < 0:
                        new_bud.local_vector = new_bud.local_vector * bud_global_pos[2] / abs(new_bud.local_vector[2])
                    branch.children.append(new_bud)
                    branch.children_weights.append(1.0)

                # Replace the old bud with the new branch
                bud_index: int = bud.parent.children.index(bud)
                bud.parent.children[bud_index] = branch
            elif bud.remaining_resting_period > 0:
                bud.remaining_resting_period -= 1
        
        # Cull branches
        self.buds = self.root.cull(self.hyperparameters.cull_threshold)