import math
import numpy as np
import datetime
import os
from structure import TreeStructure
from graphics import TreeSkeletonImager

def skeleton_test():
    structure: TreeStructure = TreeStructure( \
        prolepsis=0, \
        shadow_a=1.0, \
        shadow_b=1.0001, \
        shadow_voxel_size=2.0, \
        priority_min=0.9, \
        priority_kappa=0.75, \
        apical_theta=math.pi * 1/3, \
        apical_lambda_fn=(lambda node: 0.52 if node.main_stem else 0.5), \
        growth_zeta=0.2, \
        growth_eta=0.1, \
        initial_tropism=np.array([0.0, 0.0, -1.0]), \
        cull_threshold=0.01 \
    )
    skeleton_renderer: TreeSkeletonImager = TreeSkeletonImager(structure, zoom=10.0)

    directory: str = f"file/skeleton_test/{datetime.datetime.now().timestamp()}"
    os.makedirs(directory, exist_ok=True)
    for k in range(30):
        structure.grow(1.5 * (k + 1))
        print(f"Iteration {(k + 1):02d}: {(structure.root.num_descendant_nodes + 1)} nodes")
        skeleton_renderer.render(f"{directory}/{(k + 1):03d}.png")

skeleton_test()