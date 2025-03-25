import uproot
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

scan_dir: str = "runs/scan/"
output_file: str = "ntuple.0.0.root:susy"
path_to_plots: str = scan_dir + "plots/"
if not os.path.isdir(path_to_plots): os.makedirs(path_to_plots)


prefix: list[str] = ["SS", "SP", "MO", "SI", "GM2"]

plt_colors: list[str] = ['mediumorchid', 'lightseagreen', 'crimson'] 


parameter: str = "m_h"                                                              # try to find this in data



data = uproot.open(scan_dir + output_file)

# extract data keys to txt file
with open("parameter_keys.txt", 'w') as file:
    for key in data.keys():
        file.write(key + '\n')

# collect all parameter occurences in one dataframe
parameter_df: pd.DataFrame = data.arrays(filter_name=f'*{parameter}*', library='pd')


# plot the parameter
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)

bins = np.histogram_bin_edges(parameter_df[parameter_df.keys()[0]], bins=50)

for id, par in enumerate(parameter_df.keys()):
    ax.hist(parameter_df[par], bins=bins, color=plt_colors[id], alpha=0.5,
            label=par.split('_')[0])
    
ax.set_xlabel(parameter)
ax.set_ylabel("counts")
ax.legend()
fig.tight_layout()
fig.savefig(path_to_plots + parameter, dpi=500)


