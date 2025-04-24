"""Calculate the Coverage of the parameter space"""

import numpy as np


# define parameter space
d: int = 2
parameter_space: list[list[int]] = [[0, 3]] * d

# calculate volume of parameter space
dim_lens = [dim[-1] - dim[0] for dim in parameter_space]
volume = np.prod(dim_lens)
print(volume)
