import h5py
import numpy as np

scan_output_file = "runs/MSSM19atQ/samples/MSSM19atQ.hdf5"
group= "MSSM"

'''read the file'''
scan = h5py.File(scan_output_file, 'r')

for key in scan[group].keys():
    print(key)


logL_label = group + "/pointID"
isvalid_label = group + "/pointID_isvalid"
isvalid_mask  = np.array(scan[isvalid_label], dtype=np.bool)

n_points = len(scan[logL_label])
n_valid_points = np.sum(isvalid_mask)

print("number of points in scan:        ", n_points)
print("number of valid points in scan:  ", n_valid_points)
