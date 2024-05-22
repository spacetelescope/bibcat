# !/usr/bin/env python
# -*- coding: utf-8 -*-
#

import os
import pathlib
from typing import Any, Dict

import yaml  # type: ignore
from deepmerge import Merger  # type: ignore

# For instance, fraction_TVT:list of floats concatenates the values twice.
merger = Merger([(list, ["override"]), (dict, ["merge"])], ["override"], ["override"])


class ddict(Dict[str, Any]):
    """Create a dottable dictionary

    This allows for dictionary keys to be accessed
    like class attributes.  For example, in x = {'a': 1, 'b': 2},
    one can access x['a'] or x.a

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """override init"""
        # initialize normal dict
        super().__init__(*args, **kwargs)

        # Convert any nested dictionaries into ddict instances
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ddict(value)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

    def __getattr__(self, name: str) -> Any:
        """override getattr"""
        # Allow attribute access to existing dictionary keys
        if name in self:
            return self[name]
        raise AttributeError(f"'ddict' object has no attribute '{name}'")


def read_yaml(filename: pathlib.Path) -> dict:
    """Read a yaml configuration file

    Expands any environment variables in the yaml file

    Parameters
    ----------
    filename : str
        the filepath to the configuration

    Returns
    -------
    dict
        the yaml configuration
    """
    with open(filename, "r") as f:
        data: Dict[str, Any] = yaml.load(os.path.expandvars(f.read()), Loader=yaml.SafeLoader)
        return data


def get_custom_config() -> dict | None:
    """Look up and read in any custom configuration

    Looks for a user custom configuration for bibcat and reads
    it in if found.  Looks for a "bibcat_config.yaml" file in
    a user environment variable directory, $BIBCAT_CONFIG_DIR,
    or the user's home directory.

    Returns
    -------
    dict | None
        the custom yaml configuration
    """
    # build custom config path
    # look for config in BIBCAT_CONFIG_DIR envvar or in user home directory
    bc_dir = os.getenv("BIBCAT_CONFIG_DIR")
    user_dir = os.path.expanduser("~")
    root = bc_dir or user_dir
    path = pathlib.Path(root) / "bibcat_config.yaml"

    # if file doesn't exist, return
    if not path.exists():
        return None

    # read the config
    return read_yaml(path)


def get_config() -> ddict:
    """Read in the bibcat configuration

    Reads in the bibcat configuration from a yaml file.
    The default config is in etc/bibcat_config.yaml.  It
    looks for an optional user configuration for bibcat,
    and merges it with the default. All user configs take
    precedence and override any default configs.

    Returns
    -------
    dict
        the bibcat config object
    """
    # get the default configuration
    root = pathlib.Path(__file__).resolve().parent.parent
    config_path = root / "etc/bibcat_config.yaml"
    config = read_yaml(config_path)

    # get any custom configuration
    custom_config = get_custom_config()

    # merge the two
    if custom_config:
        config = merger.merge(config, custom_config)

    return ddict(config)
