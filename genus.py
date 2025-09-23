import random
import math
import datetime
import os
from structure import TreeNode, TreeStructure
from graphics import TreeSkeletonRenderer

skeleton_render: TreeSkeletonRenderer = TreeSkeletonRenderer(zoom=10.0)

class Genus:
    name: str 
    '''
    The name of the genus.
    '''

    prolepsis: int 
    '''
    The resting period of a bud before it can sprout a new bud.
    '''

    cycles_min: int 
    '''
    The minimum number of growth cycles to perform.
    '''

    cycles_max: int 
    '''
    The maximum number of growth cycles to perform.
    '''

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

    growth_zeta: float 
    '''
    The growth direction parameter zeta.
    '''

    growth_eta: float 
    '''
    The growth direction parameter eta.
    '''

    main_stem_apical_lambda_initial: float 
    main_stem_apical_lambda_decay: float 
    lateral_stem_apical_lambda_initial: float 
    lateral_stem_apical_lambda_decay: float 

    def __init__(self, \
                    name: str, \
                    prolepsis: int = 4, \
                    cycles_min: int = 10, \
                    cycles_max: int = 15, \
                    vigor: float = 1.0, \
                    priority_min: float = 0.9, \
                    priority_kappa: float = 0.75, \
                    apical_theta: float = math.pi * 2 / 5, \
                    main_stem_apical_lambda_initial: float = 0.9, \
                    lateral_stem_apical_lambda_initial: float = 0.75, \
                    main_stem_apical_lambda_decay: float = 0.9, \
                    lateral_stem_apical_lambda_decay: float = 0.9, \
                    growth_zeta: float = 0.2, \
                    growth_eta: float = 0.1):
        '''
        
        '''
        self.name = name
        self.prolepsis = prolepsis
        self.cycles_min = cycles_min
        self.cycles_max = cycles_max
        self.vigor = vigor
        self.priority_min = priority_min
        self.priority_kappa = priority_kappa
        self.apical_theta = apical_theta
        self.main_stem_apical_lambda_initial = main_stem_apical_lambda_initial
        self.lateral_stem_apical_lambda_initial = lateral_stem_apical_lambda_initial
        self.main_stem_apical_lambda_decay = main_stem_apical_lambda_decay
        self.lateral_stem_apical_lambda_decay = lateral_stem_apical_lambda_decay
        self.growth_zeta = growth_zeta
        self.growth_eta = growth_eta


    def generate_tree_structure(self, render_skeleton: bool = False) -> TreeStructure:
        directory: str = f"file/{self.name}/{datetime.datetime.now().timestamp()}"
        os.makedirs(directory, exist_ok=True)

        iteration: int = 1
        num_iterations: int = (self.prolepsis + 1) * random.randrange(self.cycles_min, self.cycles_max)
        main_stem_apical_lambda: float = self.main_stem_apical_lambda_initial
        lateral_stem_apical_lambda: float = self.lateral_stem_apical_lambda_initial
        def apical_lambda_fn(node: TreeNode) -> float:
            if node.main_stem:
                return main_stem_apical_lambda
            else:
                return lateral_stem_apical_lambda

        structure: TreeStructure = TreeStructure( \
            prolepsis=self.prolepsis, \
            priority_min=self.priority_min, \
            priority_kappa=self.priority_kappa, \
            apical_theta=self.apical_theta, \
            apical_lambda_fn=apical_lambda_fn, \
            growth_zeta=self.growth_zeta, \
            growth_eta=self.growth_eta \
        )
        while iteration <= num_iterations:
            structure.grow(self.vigor * pow(iteration / (self.prolepsis + 1), 1.5))
            print(f"Iteration {iteration}: {structure.root.get_descendant_nodes()} nodes")

            if render_skeleton:
                skeleton_render.render(structure, f"{directory}/{iteration:03d}.png")

            iteration += 1
            main_stem_apical_lambda = 0.5 + ((main_stem_apical_lambda - 0.5) * pow(self.main_stem_apical_lambda_decay, 1.0 / num_iterations))
            lateral_stem_apical_lambda = 0.5 + ((lateral_stem_apical_lambda - 0.5) * pow(self.lateral_stem_apical_lambda_decay, 1.0 / num_iterations))
        return structure
    
TEST_GENUS_01: Genus = Genus( \
    name = 'test01', \
    prolepsis=0, \
    cycles_min=10, \
    cycles_max=15, \
    vigor=5.0, \
    main_stem_apical_lambda_initial=0.9, \
    main_stem_apical_lambda_decay=0.25, \
    lateral_stem_apical_lambda_initial=0.6, \
    lateral_stem_apical_lambda_decay=0.999, \
    growth_eta=0.15, \
)