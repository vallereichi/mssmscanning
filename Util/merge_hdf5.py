"""Module to merge output files to a single hdf5 file"""

import os
import sys
import glob
import argparse
import h5py
import uproot
import numpy as np
import vallog as vl

msg = vl.Logger()


def scan_info(scan: dict[str, np.ndarray], file_path: str, file_type: str, logfile) -> None:
    """
    recieve some inforamation about a scan and keep track of it in the logs
    """
    logfile.write(f"found {len(scan)} keys in scan {file_path}\n")
    # create filters for valid models
    if file_type == "hdf5":
        try:
            mask = scan["LogLike_isvalid"] == 1
            n_valid_models = sum(mask)
        except KeyError:
            logfile.write("WARNING: 'LogLike_isvalid' key not found, no filter for valid models can be applied\n")
    elif file_type == "root":
        try:
            mask = scan["GM2_gmuon"] != -1
            n_valid_models = sum(mask)
        except KeyError:
            logfile.write("WARNING: 'GM2_gmuon' key not found, no filter for valid models can be applied\n")
    else:
        raise ValueError(f"FileType '{file_type}' is not supported")

    # check if all keys have the same length
    f = len(scan[next(iter(scan))])
    if all(len(x) == f for x in scan.values()):
        if "n_valid_models" in locals():
            logfile.write(f"\tVALID MODELS: {n_valid_models}/{f}\n")
    else:
        logfile.write("WARNING: keys have different lengths\n")
        if "n_valid_models" in locals():
            logfile.write(f"\tVALID MODELS: {n_valid_models}/{f}\n")
        for key, value in scan.items():
            logfile.write(f"\tKEY: {key} SHAPE: {value.shape}\n")


def search_hdf5_files(directory: str, logfile) -> list[str]:
    """
    recursivly search the directory for hdf5 files and return the found paths in a list
    """
    msg.heading(f"searching for hdf5 files in '{directory}' ...")
    logfile.write(f"searching for hdf5 files in '{directory}' ...\n")
    hdf5_files = glob.glob(os.path.join(directory, "**", "*.hdf5"), recursive=True)
    if not hdf5_files:
        msg.log(f"No hdf5 files found in '{directory}'", vl.error)
        logfile.write(f"No hdf5 files found in '{directory}'\n")
        exit(1)
    for file in hdf5_files:
        msg.log(f"found: {file}", vl.info)
        logfile.write(f"found: {file}\n")
    logfile.write(f"found {len(hdf5_files)} hdf5 files\n\n")
    return hdf5_files


def search_root_files(directory: str, logfile) -> list[str]:
    """
    recursivly search the directory for root files and return the found paths in a list
    """
    msg.heading(f"searching for root files in '{directory}' ...")
    logfile.write(f"searching for root files in '{directory}' ...\n")
    root_files = glob.glob(os.path.join(directory, "**", "*.root"), recursive=True)
    if not root_files:
        msg.log(f"No root files found in '{directory}'", vl.error)
        logfile.write(f"No root files found in '{directory}'\n")
        exit(1)
    for file in root_files:
        msg.log(f"found: {file}", vl.info)
        logfile.write(f"found: {file}\n")
    logfile.write(f"found {len(root_files)} hdf5 files\n\n")
    return root_files


def read_hdf5_file(file_path: str, logfile) -> dict:
    """
    extract the data from the specified hdf5 file and return it as a dictionary
    """
    msg.heading(f"reading: {file_path}")
    with h5py.File(file_path, "r") as file:
        dataset = file["MSSM"]
        if isinstance(dataset, h5py.Group):
            data_dict = {key: np.array(dset) for key, dset in dataset.items()}
        else:
            raise TypeError(f"expected type 'h5py.Group' but got {type(dataset)}")
    msg.log(f"read {len(data_dict)} keys from '{file_path}'", vl.info)
    scan_info(data_dict, file_path, "hdf5", logfile)
    return data_dict


def read_root_file(file_path: str, logfile) -> dict:
    """
    extract the data from the specified root file and return it as a dictionary
    """
    msg.heading(f"reading: {file_path}")
    dataset = uproot.open(file_path + ":susy")
    data_dict = {key: np.array(dataset[key]) for key in dataset.keys()}
    msg.log(f"read {len(data_dict)} keys from '{file_path}'", vl.info)
    scan_info(data_dict, file_path, "root", logfile)
    return data_dict


def merge_datasets(datasets: list[dict], logfile) -> dict:
    """
    merge multiple datasets into one dictionary
    """
    msg.heading("merging datasets ...")
    merged_data = {}
    for dataset in datasets:
        msg.log("extracting data...", vl.info)
        for key, value in dataset.items():
            if key not in merged_data:
                merged_data[key] = []
            merged_data[key].append(value)

    msg.log("concatonate data...", vl.info)
    for key in merged_data:
        merged_data[key] = np.concatenate(merged_data[key], axis=0)
    msg.log(f"merged {len(datasets)} datasets", vl.info)
    scan_info(merged_data, "merged_dataset", "hdf5", logfile)
    return merged_data


def save_dataset_to_hdf5(dataset: dict, output_path: str) -> None:
    """
    save the merged dataset to a hdf5 file
    """
    msg.heading(f"saving merged dataset to '{output_path}'")
    with h5py.File(output_path, "w") as file:
        group = file.create_group("MSSM")
        for key, value in dataset.items():
            group.create_dataset(key, data=value)
    return


def main(directory: str, output_filename: str, file_type: str):
    """Entry point"""
    logfile_path = directory + "/merge.log"
    logfile = open(logfile_path, 'w')
    if file_type == "hdf5":
        hdf5_files = search_hdf5_files(directory, logfile)
        datasets = [read_hdf5_file(file, logfile) for file in hdf5_files]
    elif file_type == "root":
        root_files = search_root_files(directory, logfile)
        datasets = [read_root_file(file, logfile) for file in root_files]
    else:
        raise ValueError(f"FileType '{file_type}' is not supported")

    merged_data = merge_datasets(datasets, logfile)
    save_dataset_to_hdf5(merged_data, directory + "/" + output_filename)
    logfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, default=".", help="directory to search for files")
    parser.add_argument(
        "-o", "--output", type=str, default="merged_dataset.hdf5", help="output path for the merged dataset"
    )
    parser.add_argument(
        "-f",
        "--file_type",
        type=str,
        help="select the file type of the files you want to merge. Supported file types are ['.hdf5', '.root']",
    )
    args = parser.parse_args()

    main(args.directory, args.output, args.file_type)
