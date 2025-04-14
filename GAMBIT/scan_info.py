"""
read scan output files and collect some information
"""

import h5py
import numpy as np
import vallog as vl

"""initialize logger"""
msg = vl.Logger("Debug")

scan_output_file = "runs/MSSM19atQ/samples/MSSM19atQ.hdf5"
group = "MSSM"

"""read the file"""
scan = h5py.File(scan_output_file, "r")

for key in scan[group].keys():
    print(key)


logL_label = group + "/pointID"
isvalid_label = group + "/pointID_isvalid"
isvalid_mask = np.array(scan[isvalid_label], dtype=np.bool)

n_points = len(scan[logL_label])
n_valid_points = np.sum(isvalid_mask)

msg.sep()
msg.heading("Scan Evaluation")
msg.log(f"Final dataset size: {n_points}", vl.info)
msg.log(f"Number of valid points: {n_valid_points}", vl.info)
msg.sep()

