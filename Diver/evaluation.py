import numpy as np
import matplotlib.pyplot as plt
from de import diver, parabolic


'''test parameters'''
NP = 50
F = 0.7
Cr = 0.7
ranges = [(-50,50)]*2

conv_thresh = 1e-5
MAX_ITER = 10000

obj_func = parabolic

'''create data'''
populations, improvements, update_times = diver(NP, ranges, F, Cr, obj_func)