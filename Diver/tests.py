"""Test function to run and store test data"""

import os
import tqdm
import numpy as np
import pandas as pd
import vallog as vl

from de import Diver
import objectives


def run_test(test_configuration: dict[str, list], output_path: str) -> None:
    """start a test run based on the diver configuration"""
    config = {
        "parameter_space": None,
        "population_size": None,
        "conv_thresh": None,
        "max_iter": None,
        "objective_func": None,
        "select_target": None,
        "mutation_scheme": None,
        "mutation_scale_factor": None,
        "crossover_rate": None,
    }

    if os.path.isfile(output_path):
        data_frame = pd.read_csv(output_path)
    else:
        de = Diver()
        de.run()
        data_frame = pd.DataFrame(columns=de.__dict__.keys())

    # setup the config
    for key, test_range in test_configuration.items():
        for par in test_range:
            if key in config:
                config[key] = par

            # run diver
            diver = Diver(
                parameter_space=config["parameter_space"],
                objective_function=config["objective_func"],
                population_size=config["population_size"],
                conv_thresh=config["conv_thresh"],
                max_iter=config["max_iter"],
            )
            diver.run(
                crossover_rate=config["crossover_rate"],
                select_target=config["select_target"],
                mutation_scheme=config["mutation_scheme"],
                mutation_scale_factor=config["mutation_scale_factor"],
            )

    print(data_frame)


if __name__ == "__main__":
    out_path = "tests/diver_test.csv"

    Cr = np.linspace(0.1, 1, 10).tolist()
    de_test_config: dict = {"crossover_rate": Cr}

    run_test(de_test_config, out_path)
