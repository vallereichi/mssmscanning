"""Calculate the Coverage of the parameter space"""

import itertools
import numpy as np
import vallog as vl
from de import Diver
import objectives

msg = vl.Logger("Debug")


# calculate volume of parameter space
def calc_volume(parameter_space: list[list[int]]) -> int:
    """calculate volume of parameter space"""
    dim_lens: list[int] = [dim[-1] - dim[0] for dim in parameter_space]
    volume: int = np.prod(dim_lens)
    return volume


# floor n-dimensional vectors
diver = Diver(objective_function=objectives.four_valleys, population_size=1000, conv_thresh=1e-5)
diver.run()
point_list = list(itertools.chain(*diver.population_list))
point_list_floored = np.floor(point_list)
unique_points_floored = np.unique(point_list_floored, axis=0)


# calculate coverage
vol = calc_volume(diver.parameter_space)
coverage = len(unique_points_floored) / vol

msg.log(f"Parameter Space volume: {vol}", vl.info)
msg.log(f"unique unit cubes discovered: {len(unique_points_floored)}", vl.info)
msg.log(f"Parameter Space coverage: {coverage*100}%", vl.info)
