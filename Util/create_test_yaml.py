"""
module to create yaml files for specifix test cases based on existing yaml files.
This module provides functionality to easily adapt existing yaml configurations for further testing
"""

import os
import argparse
from typing import Any
import yaml
import vallog as vl

msg = vl.Logger("Debug")


def clear_section(yaml_config: dict, section: str) -> None:
    """
    Clear the specified section in the yaml file
    :param yaml: yaml file to modify
    :param section: section to clear
    """
    msg.log(f"Clearing section '{section}'", vl.debug)
    yaml_config[section] = {}


def append_section(yaml_config: dict, section: str, key: str, value: Any) -> None:
    """
    Append a key-value pair to the specified section in the yaml file
    :param yaml: yaml file to modify
    :param section: section to append to
    :param key: key to append
    :param value: value to append
    """
    msg.log(f"Appending key/value pair '{key}: {value}' to section '{section}'", vl.debug)
    yaml_config[section][key] = value


def create_yaml(base_yaml: str, section: str, keys: list[str] | str, values: list[Any], clear: bool = False) -> None:
    """
    Entry point
    """

    if not os.path.exists(base_yaml):
        msg.log(f"Base yaml file {base_yaml} does not exist", vl.error)
        raise FileNotFoundError()
    else:
        msg.log(f"Using base yaml file: {base_yaml}", vl.debug)

    with open(base_yaml, "r", encoding="utf-8") as file:
        yaml_config = yaml.safe_load(file)

    if section not in yaml_config:
        msg.log(f"Section {section} not found in base yaml file", vl.warning)
        if input("Do you want to create the section? (y/n)") == "y":
            yaml_config[section] = {}
            msg.log(f"Section {section} created", vl.info)
        else:
            msg.log("Exit without modification", vl.error)
            raise KeyError()
    else:
        msg.log(f"Modifying section '{section}' in base yaml file", vl.debug)

    if clear:
        clear_section(yaml_config, section)

    if type(keys) is str:
        append_section(yaml_config, section, keys, values)
    else:
        for key, idx in enumerate(keys):
            append_section(yaml_config, section, key, values[idx])

    output = yaml.dump(yaml_config)
    if msg.mode == "Debug":
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility to easily adapt existing yaml configurations")
    parser.add_argument("-b", "--base_yaml", default="raw/FS_MSSM7atQ.yaml", type=str, help="Base yaml file to adapt")
    parser.add_argument("-s", "--section", default="Parameters", type=str, help="Specifiy the section to adapt")
    parser.add_argument("-c", "--clear_section", action="store_true", help="Clear the section before modifiying it")
    parser.add_argument(
        "-v",
        "--value",
        nargs="*",
        type=str,
        help="Key/value pairs to append to the section, e.g. 'key1 value1 key2 value2'",
    )

    args = parser.parse_args()

    base_yaml = args.base_yaml
    section = args.section
    keys = args.value[::2] if args.value else []
    values = args.value[1::2] if args.value else []

    create_yaml(base_yaml, section, keys, values, args.clear_section)
