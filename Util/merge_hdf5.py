import os
import glob
import h5py
import argparse
import numpy as np
import vallog as vl

msg = vl.Logger()

def search_hdf5_files(directory: str) -> list[str]:
    """
    recursivly search the directory for hdf5 files and return the found paths in a list
    """
    msg.heading(f"searching for hdf5 files in '{directory}' ...")
    hdf5_files = glob.glob(os.path.join(directory, "**", "*.hdf5"), recursive=True)
    if not hdf5_files:
        msg.log(f"No hdf5 files found in '{directory}'", vl.error)
        exit(1)
    for file in hdf5_files:
        msg.log(f"found: {file}", vl.info)
    return hdf5_files

def read_hdf5_file(file_path: str) -> dict:
    """
    extract the data from the specified hdf5 file and return it as a dictionary
    """
    msg.heading(f"reading: {file_path}")
    with h5py.File(file_path, 'r') as file:
        dataset = file['MSSM']
        if isinstance(dataset, h5py.Group):
            data_dict = {key: np.array(dataset[key]) for key in dataset.keys()}
        else:
            raise TypeError(f"expected type 'h5py.Group' but got {type(dataset)}")
    msg.log(f"read {len(data_dict)} keys from '{file_path}'", vl.info)
    return data_dict

def merge_datasets(datasets: list[dict]) -> dict:
    """
    merge multiple datasets into one dictionary
    """
    msg.heading(f"merging datasets ...")
    merged_data = {}
    for dataset in datasets:
        msg.log(f"extracting data...", vl.info)
        for key, value in dataset.items():
            if key not in merged_data:
                merged_data[key] = []
            merged_data[key].append(value)

    msg.log("concatonate data...", vl.info)
    for key in merged_data:
        merged_data[key] = np.concatenate(merged_data[key], axis=0)
    msg.log(f"merged {len(datasets)} datasets", vl.info)
    return merged_data

def save_dataset_to_hdf5(dataset: dict, output_path:str) -> None:
    """
    save the merged dataset to a hdf5 file
    """
    msg.heading(f"saving merged dataset to '{output_path}'")
    with h5py.File(output_path, 'w') as file:
        group = file.create_group('MSSM')
        for key, value in dataset.items():
            group.create_dataset(key, data=value)
    return



def main(directory: str, output_path: str):
    hdf5_files = search_hdf5_files(directory)
    datasets = [read_hdf5_file(file) for file in hdf5_files]
    merged_data = merge_datasets(datasets)
    save_dataset_to_hdf5(merged_data, output_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, default='.', help="directory to search for hdf5 files")
    parser.add_argument('-o', '--output', type=str, default='merged_dataset.hdf5', help="output path for the merged dataset")
    args = parser.parse_args()
    main(args.directory, args.output)
