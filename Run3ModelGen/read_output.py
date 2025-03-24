import uproot
import re

output_file = "runs/default_scan/ntuple.0.0.root:susy"


data = uproot.open(output_file)
data.show(filter_name='*')



