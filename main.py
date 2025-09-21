import math
import numpy as np
from structure import TreeStructure
from graphics import TreeSkeletonImager

def skeleton_test():
    structure: TreeStructure = TreeStructure( \
        prolepsis=0, \
        proportionality=10.0, \
        shadow_a=1.0, \
        shadow_b=0.5, \
        shadow_voxel_size=1.0, \
        priority_min=0.5, \
        priority_kappa=1.0, \
        apical_theta=math.tau * 1/3, \
        apical_lambda=0.5, \
        growth_zeta=0.4, \
        growth_eta=0.2, \
        initial_tropism=np.array([0.0, 0.0, -1.0]), \
        cull_threshold=0.2 \
    )
    skeleton_renderer: TreeSkeletonImager = TreeSkeletonImager(structure)
    for k in range(20):
        structure.iterate()
        skeleton_renderer.render(f"file/skeleton_test_{k:02d}.png")

skeleton_test()