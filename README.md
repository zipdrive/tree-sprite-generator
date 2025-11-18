# Tree Sprite Generator

This is a small program which generates pixel art sprites for trees, along with normalmaps. The algorithm to generate the structure of the trees is based on *[Self-organizing tree models for image synthesis](https://algorithmicbotany.org/papers/selforg.sig2009.pdf)*.

## TreeStructureHyperparameters

`TreeStructureHyperparameters` is defined in `structure.py`. The hyperparameters, corresponding to those laid out in the paper above, define how the tree grows.

## Genus

`Genus` is defined in `genus.py`. The structure of a genus of trees is defined by prolepsis (how many iterations it takes before a bud can sprout into a branch), and one or more `Keyframe` objects. The hyperparameters of the tree are interpolated over the keyframes.

### Keyframe

`Keyframe` is also defined in `genus.py`. A keyframe consists of a range of possible cycles that the keyframe lasts for, as well as a `TreeStructureHyperparameters` objects that defines how the tree grows at that keyframe.

