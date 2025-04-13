"""
read scan data and create plots
"""

import os
import h5py
import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fetch_parameter_r3mg(
    scan_dir: str, parameter_name: str, prefix: str, scan_file: str = "ntuple.0.0.root:susy"
) -> pd.DataFrame:
    """create a pandas dataframe from a root output file"""
    scan_data = uproot.open(scan_dir + scan_file)
    parameter_df: pd.DataFrame = scan_data.arrays(filter_name=f"{prefix}*{parameter_name}*", library="pd")
    return parameter_df


def fetch_parameter_gambit(scan_path: str, parameter_name: str, prefix: str = None) -> pd.DataFrame:
    """create a pandas dataframe from a hdf5 output file"""
    hdf5 = h5py.File(scan_path, "r")
    scan = hdf5["MSSM"]
    search_key = parameter_name if prefix is None else f"{prefix}.*{parameter_name}"
    parameter_key = ""
    for key in scan.keys():
        if key.match(search_key):
            parameter_key = key
            break
    parameter_df = pd.DataFrame(data=scan[parameter_key])
    print(parameter_df)
    return parameter_df


def plot_parameter(parameter_name: str, data: pd.DataFrame, output_path: str = "plots/") -> None:
    """create a histogramm from a pandas dataframe"""
    plt_colors: list[str] = ["lightseagreen", "crimson", "mediumorchid"]

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    bins = np.histogram_bin_edges(data[data.keys()[0]], bins=50)

    for i, par in enumerate(data.keys()):
        ax.hist(data[par], bins=bins, color=plt_colors[i], alpha=0.5, label=par.split("#")[0])

    ax.set_xlabel(parameter_name)
    ax.set_ylabel("counts")
    ax.legend()
    fig.tight_layout()

    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    fig.savefig(output_path + parameter_name, dpi=500)


def main():
    """entry point"""
    mssm19atq_scan = "runs/MSSM19atQ/"
    mssm19atq_ma_scan = "runs/MSSM19atQ_mA/"
    mssm19atq_gambit_scan = "../GAMBIT/runs/MSSM19atQ_random_bare/samples/MSSM19atQ.hdf5"
    scan_list: list[str] = [mssm19atq_scan, mssm19atq_ma_scan, mssm19atq_gambit_scan]
    labels: list[str] = ["MSSM19atQ_MG", "MSSM19atQ_mA_MG", "MSSM19atQ_GAMBIT"]
    prefix = "SP"
    parameter_name = "m_h"

    df_list: list[pd.DataFrame] = []
    for i, scan in enumerate(scan_list):
        try:
            df = fetch_parameter_r3mg(scan, parameter_name, prefix)
        except FileNotFoundError:
            continue
        try:
            df = fetch_parameter_gambit(scan, parameter_name)
        except FileNotFoundError:
            continue
        column_names = [labels[i]]
        df = df.set_axis(column_names, axis=1)
        df_list.append(df)

    # concatenate scan dataframes
    parameter_df = pd.concat(df_list, axis=1)
    print(parameter_df)

    # plot the scan data
    plot_parameter(parameter_name, parameter_df)


if __name__ == "__main__":
    main()
