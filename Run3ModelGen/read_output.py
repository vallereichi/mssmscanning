import os
import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


prefix: list[str] = ["SS", "SP", "MO", "SI", "GM2"]



def fetch_parameter(scan_dir: str, parameter_name: str, prefix: str, scan_file: str = "ntuple.0.0.root:susy") -> pd.DataFrame:
    '''create a pandas dataframe from a scan output file'''
    scan_data = uproot.open(scan_dir + scan_file)
    parameter_df: pd.DataFrame = scan_data.arrays(filter_name=f'{prefix}*{parameter_name}*', library='pd')
    return parameter_df


# plot the parameter
# if not os.path.isdir(path_to_plots): os.makedirs(path_to_plots)
# fig = plt.figure(figsize=(7,5))
# ax = fig.add_subplot(111)

# bins = np.histogram_bin_edges(parameter_df[parameter_df.keys()[0]], bins=50)

# for id, par in enumerate(parameter_df.keys()):
#     ax.hist(parameter_df[par], bins=bins, color=plt_colors[id], alpha=0.5,
#             label=par.split('_')[0])
    
# ax.set_xlabel(parameter)
# ax.set_ylabel("counts")
# ax.legend()
# fig.tight_layout()
# fig.savefig(path_to_plots + parameter, dpi=500)
def plot_parameter(parameter_name: str, data: pd.DataFrame, output_path: str = "plots/") -> None:
    '''create a histogramm from a pandas dataframe'''
    plt_colors: list[str] = ['mediumorchid', 'lightseagreen', 'crimson'] 

    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111)

    bins = np.histogram_bin_edges(data[data.keys()[0]], bins=50)

    for id, par in enumerate(data.keys()):
        ax.hist(data[par], bins=bins, color=plt_colors[id], alpha=0.5, label=par.split('#')[0])

    ax.set_xlabel(parameter_name)
    ax.set_ylabel('counts')
    ax.legend()
    fig.tight_layout()

    if not os.path.isdir(output_path): os.makedirs(output_path)
    fig.savefig(output_path + parameter_name, dpi=500)


def main():
    mssm19atq_scan = "runs/MSSM19atQ/"
    mssm19atq_ma_scan = "runs/MSSM19atQ_mA/"
    scan_list: list[str] = [mssm19atq_scan, mssm19atq_ma_scan]
    prefix = "SP"
    parameter_name = "m_h"

    df_list: list[pd.DataFrame] = []
    for scan in scan_list:
        df = fetch_parameter(scan, parameter_name, prefix)
        column_names = [scan.split('/')[-2] + '#' + df.columns[i] for i in range(len(df.columns))]
        df = df.set_axis(column_names, axis=1)
        df_list.append(df)

    # concatenate scan dataframes
    parameter_df = pd.concat(df_list, axis=1)
    print(parameter_df)

    # plot the scan data
    plot_parameter(parameter_name, parameter_df)




if __name__ =="__main__":
    main()