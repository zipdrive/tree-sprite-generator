import math
import random
import numpy as np
from typing import Self

def normalized(vec: np.array) -> np.array:
    '''
    Normalizes a vector.
    '''
    return vec / np.linalg.norm(vec)


class TreeNode:
    '''
    Data structure representing a node of the tree's structure.
    '''

    parent: Self 
    '''
    The parent node.
    '''

    children: list[Self]
    '''
    The child nodes.
    '''

    children_weights: np.array 
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

    local_vector: np.array
    '''
    The direction and magnitude of the end of the node, relative to the end of the previous node.
    '''

    apical_lambda: float 
    '''
    Parameter for apical preference. Controls whether the main branch or lateral branches are favored more.
    '''

    light: float 
    '''
    How much light the node receives, either from its leaves or from its children.
    '''

    vigor: float 
    '''
    How much vigor the node receives from its parent.
    '''

    remaining_resting_period: int 
    '''
    How long this bud has to wait for before it can start producing new buds.
    '''

    def __init__(self, parent: Self, local_vector: np.array, prolepsis: int, apical_lambda: float):
        self.parent = parent 
        self.children = []
        self.children_weights = []
        self.num_descendant_buds = 0
        self.num_descendant_nodes = 0
        self.local_vector = local_vector
        self.remaining_resting_period = prolepsis
        self.apical_lambda = apical_lambda

    def get_global_vector(self) -> np.array:
        '''
        The global position of the end of the node.
        '''
        return (self.parent.get_global_vector() + self.local_vector) if self.parent != None else self.local_vector

    def reset_light(self):
        '''
        Resets the light received by the node and its children to 0.
        Call this function before calculating light.
        '''
        self.light = 0
        for k in range(len(self.children)):
            self.children[k].reset_light()

    def propagate_light(self, weight_min: float, weight_max: float, kappa: float):
        '''
        Propagates light received from the buds up to the root, and recalculates the weight of each child.

        Args:
            weight_min (float): The minimum weight of a child.
            weight_max (float): The maximum weight of a child.
            kappa (float): Parameter in the range (0, 1]. Lower values make the branches receiving the most light be more aggressively supported.
        '''
        if len(self.children) == 0:
            return
        
        # Determine the average amount of light being received by the buds from each child
        self.num_descendant_buds = 0
        self.num_descendant_nodes = 0
        child_avg_light_amts: list[tuple[int, float]] = []
        for k in range(len(self.children)):
            child: Self = self.children[k]
            child.propagate_light(weight_min, weight_max, kappa)
            child_num_buds: int = child.num_descendant_buds + (1 if len(child.children) == 0 else 0)
            self.num_descendant_buds += child_num_buds
            self.num_descendant_nodes += child.num_descendant_nodes + 1
            self.light += child.light
            child_avg_light_amts.append((k, (self.apical_lambda if k == 0 else (1.0 - self.apical_lambda)) * child.light / child_num_buds))

        # Sort in order of average light received
        child_avg_light_amts.sort(key=lambda tup: tup[1])

        # Calculate weights as function of rank of average light received
        for idx in range(len(child_avg_light_amts)):
            rank_pct = (idx / (len(child_avg_light_amts) - 1)) + kappa - 1.0
            k, _ = child_avg_light_amts[idx]
            weight: float = weight_min
            if rank_pct >= 0.0:
                weight += (weight_max - weight_min) * rank_pct / kappa
            self.children_weights[k] = weight

    def distribute_vigor(self, vigor_received: float):
        '''
        Distributes vigor to the children of the node.

        Args:
            vigor_received (float): The vigor received from the parent.
        '''
        self.vigor = vigor_received
        child_ratios: list[float] = [self.children[k].light * self.children_weights[k] for k in range(len(self.children))]
        total_child_ratios: float = sum(child_ratios)
        for k in range(len(self.children)):
            try:
                self.children[k].distribute_vigor(self.vigor * child_ratios[k] / total_child_ratios)
            except ZeroDivisionError:
                self.children[k].distribute_vigor(0.0)

    def cull(self, culling_ratio: float):
        '''
        Culls the descendants of this node if the ratio of the total light received from the buds relative to the total number of descendants of this node falls below a certain threshold.

        Args:
            culling_ratio (float): The threshold at which to cull.
        '''
        if self.num_descendant_nodes > 0:
            if self.light / self.num_descendant_nodes < culling_ratio:
                self.children = []
            else:
                for k in range(len(self.children)):
                    self.children[k].cull(culling_ratio)


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

    optimal_growth_direction: np.array 
    '''
    The optimal growth direction for buds inside this voxel.
    '''

    def __init__(self, startingNode: TreeNode = None):
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

    def __init__(self, shadow_a: float, shadow_b: float, shadow_voxel_size: float = 1.0):
        self.shadow_a = shadow_a
        self.shadow_b = shadow_b
        self.shadow_voxel_size = shadow_voxel_size
        self.voxels = {}

    def add(self, bud: TreeNode):
        '''
        Adds a bud to the corresponding shadow voxel.

        Args:
            bud (TreeNode): The bud to add.
        '''
        bud_global_vector: np.array = bud.get_global_vector()
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
                    voxel.shadow += self.shadow_a * math.pow(self.shadow_b, key_above[2] - key[2])

            # Apply shadows to each bud inside the voxel
            for k in range(len(voxel.nodes)):
                voxel.nodes[k].light = max(1.0 - voxel.shadow, 0.0)

        # Calculate the optimal growth direction for each voxel with buds in it
        for key_idx in range(len(keys)):
            key: tuple[int, int, int] = keys[key_idx]
            least_shadow: float = math.inf
            least_shadow_dir: np.array = np.array((0.0, 0.0, 1.0))

            x, y, z = key
            for dx in [0, -1, 1]:
                for dy in [0, -1, 1]:
                    for dz in [1, 0, -1]:
                        if dx == 0 and dy == 0 and dz == 0:
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
                                    adj_voxel.shadow += self.shadow_a * math.pow(self.shadow_b, key_above_z - key_adj[2])

                        # Check if adjacent voxel has lowest shadow found so far
                        if self.voxels[key_adj].shadow < least_shadow:
                            least_shadow = self.voxels[key_adj].shadow
                            least_shadow_dir = np.array((dx, dy, dz)) 

            # Record the direction of the voxel with the least amount of shadow
            self.voxels[key].optimal_growth_direction = least_shadow_dir

    def get_optimal_growth_direction(self, bud: TreeNode) -> np.array:
        '''
        Finds the optimal growth direction for the bud.

        Args:
            bud (TreeNode): The bud to get the optimal growth direction for.
        '''
        bud_global_vector: np.array = bud.get_global_vector()
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

    proportionality: float 
    '''
    The parameter alpha determining how proportional vigor received is to light intake.
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

    priority_min: float 
    priority_kappa: float

    apical_theta: float 
    '''
    The angle at which new branches are branched off.
    In nature, this angle is usually one of the following: [1/3, 2/5, 3/8, 5/13]
    '''

    apical_lambda: float 

    growth_zeta: float 
    '''
    The growth direction parameter zeta.
    '''

    growth_eta: float 
    '''
    The growth direction parameter eta.
    '''

    tropism: np.array 
    '''
    The tropism vector.
    '''


    root: TreeNode 
    '''
    The root of the tree.
    '''

    buds: list[TreeNode]
    '''
    The terminal ends of the tree.
    '''

    def __init__(self, \
                    prolepsis: int, \
                    proportionality: float, \
                    shadow_a: float, \
                    shadow_b: float, \
                    shadow_voxel_size: float, \
                    priority_min: float, \
                    priority_kappa: float, \
                    apical_theta: float, \
                    apical_lambda: float, \
                    growth_zeta: float, \
                    growth_eta: float, \
                    initial_tropism: np.array, \
                    cull_threshold: float):
        '''
        Initializes a tree.

        Args:
            prolepsis (int): The number of cycles that a new bud rests for before it can start producing new buds. Set to 0 for sylleptic branching, or set to >0 for proleptic branching.
            proportionality (float): How long branches will be, approximately proportional to the light they receive.
            shadow_a (float): Parameter for determining how buds overshadow other buds. Controls the magnitude of shadows.
            shadow_b (float): Parameter for determining how buds overshadow other buds. Controls the dropoff of shadows over increasing vertical distance.
            shadow_voxel_size (float): The resolution of shadow voxels.
            apical_theta (float): The angle at which new branches will branch off.
            growth_zeta (float): Parameter for determining which direction buds grow in. Controls how strong the tendency to grow towards light is.
            growth_eta (float): Parameter for determining which direction buds grow in. Controls how strong the tendency towards the tropism vector is.
            initial_tropism (np.array): A three-dimensional tropism vector that represents a preferred direction (e.g. downwards, to simulate gravity).
            cull_threshold (float): The threshold of average light gathered by a branch for that 
        '''
        self.prolepsis = prolepsis
        self.proportionality = proportionality
        self.shadow_a = shadow_a
        self.shadow_b = shadow_b
        self.shadow_voxel_size = shadow_voxel_size
        self.priority_min = priority_min
        self.priority_kappa = priority_kappa
        self.apical_theta = apical_theta
        self.apical_lambda = apical_lambda
        self.growth_zeta = growth_zeta
        self.growth_eta = growth_eta
        self.tropism = initial_tropism
        self.cull_threshold = cull_threshold

        # Create the root node
        self.root = TreeNode(None, np.array([0.0, 0.0, self.proportionality]), self.prolepsis, self.apical_lambda)
        self.root.light = 1.0
        self.buds = [self.root]
    
    def iterate(self):
        # Update light received by buds
        self.root.reset_light()
        voxels: TreeNodeShadowVoxels = TreeNodeShadowVoxels(self.shadow_a, self.shadow_b, self.shadow_voxel_size)
        for k in range(len(self.buds)):
            bud: TreeNode = self.buds[k]
            voxels.add(bud)
        voxels.update_light() # Updates the light received by each bud
        self.root.propagate_light(self.priority_min, 1.0, self.priority_kappa) # Updates the light received by parent nodes, grandparent nodes, etc.

        # Update vigor of each bud
        self.root.distribute_vigor(self.proportionality * self.root.light)

        # Sprout new branches from buds
        new_buds: list[TreeNode] = []
        for k in range(len(self.buds)):
            bud: TreeNode = self.buds[k]
            if bud.remaining_resting_period == 0 and bud.vigor >= 1.0:
                num_sprouts: int = math.floor(bud.vigor)
                sprout_length: float = bud.vigor / num_sprouts
                straight_direction: np.array = normalized(bud.local_vector)
                straight_normal_direction1: np.array = normalized(np.array([0.0, -straight_direction[2], straight_direction[1]]))
                straight_normal_direction2: np.array = np.cross(straight_direction, straight_normal_direction1)
                optimal_growth_direction: np.array = voxels.get_optimal_growth_direction(bud)
                twist: float = random.random() * math.tau 
                for n in range(num_sprouts):
                    expected_direction: np.array
                    if n == 0:
                        expected_direction = straight_direction
                    else:
                        twist += math.tau / (num_sprouts - 1)
                        expected_direction = (math.cos(self.apical_theta) * straight_direction) + (math.sin(self.apical_theta) * ((math.cos(twist) * straight_normal_direction1) + (math.sin(twist) * straight_normal_direction2)))
                    bud.children.append( \
                        TreeNode( \
                            parent=bud, \
                            local_vector=sprout_length * ( \
                                (expected_direction * (1.0 - self.growth_zeta - self.growth_eta)) + \
                                (optimal_growth_direction * self.growth_zeta) + \
                                (self.tropism * self.growth_eta) \
                            ), \
                            prolepsis=self.prolepsis, \
                            apical_lambda=self.apical_lambda \
                        ) \
                    )
                    bud.children_weights.append(1.0)
                new_buds += bud.children
            else:
                bud.remaining_resting_period -= 1
                new_buds.append(bud)
        
        # Cull branches
        self.root.cull(self.cull_threshold)