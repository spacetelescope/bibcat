import os

from bibcat.core.config import get_config

from ._version import version as __version__

__all__ = ["__version__"]


def check_env(name: str):
    """Check for a user environment variable existence

    Parameters
    ----------
    name : str
        the name of the environment variable
    """
    name = name.upper()
    # check for user environment variables
    if name not in os.environ:
        print(f"User environment variable {name} not found. Setting to user home directory.")
        os.environ[name] = os.path.expanduser("~")


# check for envvars
check_env("BIBCAT_DATA_DIR")
check_env("BIBCAT_OUTPUT_DIR")
check_env("BIBCAT_OPSDATA_DIR")

# create the bibcat configuration object
config = get_config()


# setup paths
def setup_paths(config: dict) -> dict:
    """Setup fixed global paths"""
    config.paths.root = os.path.dirname(__file__)
    config.paths.parent = os.path.dirname(config.paths.root)
    config.paths.config = os.path.join(config.paths.root, "config")
    config.paths.docs = os.path.join(config.paths.parent, "docs")
    # set up output
    config.paths.models = os.path.join(config.output.root_path, "models")
    config.paths.output = os.path.join(config.output.root_path, "output")
    config.paths.partitioned = os.path.join(config.output.root_path, "partitioned_datasets")
    config.paths.modiferrors = os.path.join(config.paths.partitioned, config.output.name_model, "dict_modiferrors.npy")
    config.paths.TVTinfo = os.path.join(config.paths.partitioned, config.output.name_model, "dict_TVTinfo.npy")

    os.makedirs(config.paths.models, exist_ok=True)
    os.makedirs(config.paths.output, exist_ok=True)
    os.makedirs(config.paths.partitioned, exist_ok=True)

    return config


config = setup_paths(config)
