import os
from importlib.metadata import version as _version, PackageNotFoundError
try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass


# check for BIBCAT_DATA_DIR
if "BIBCAT_DATA_DIR" not in os.environ:
    print('User environment variable BIBCAT_DATA_DIR not found. Setting to user home directory.')
    os.environ['BIBCAT_DATA_DIR'] = os.path.expanduser('~')


# create the bibcat configuration object
from bibcat.core.config import get_config
config = get_config()


# setup paths
def setup_paths(config: dict) -> dict:
    """ Setup fixed global paths """
    config.paths.root = os.path.dirname(__file__)
    config.paths.parent = os.path.dirname(config.paths.root)
    config.paths.config = os.path.join(config.paths.root, "config")
    config.paths.docs = os.path.join(config.paths.parent, "docs")
    config.paths.models = os.path.join(config.paths.root, "models")
    config.paths.output = os.path.join(config.paths.parent, "output")
    config.paths.partitioned = os.path.join(config.paths.root, "data", "partitioned_datasets")
    config.paths.modiferrors = os.path.join(config.paths.partitioned, config.output.name_model, "dict_modiferrors.npy")
    config.paths.TVTinfo = os.path.join(config.paths.partitioned, config.output.name_model, "dict_TVTinfo.npy")

    os.makedirs(config.paths.models, exist_ok=True)
    os.makedirs(config.paths.output, exist_ok=True)
    os.makedirs(config.paths.partitioned, exist_ok=True)

    return config


config = setup_paths(config)

