import json
from functools import lru_cache

from bibcat import config
from bibcat.utils.logger_config import setup_logger

# set up logger
logger = setup_logger(__name__)
logger.setLevel(config.logging.level)

# map model_settings back to settings
settings = config.dataprep


@lru_cache
def load_source_dataset(do_verbose: bool):
    """
    Load the original source dataset that is a combined set of papertrack classification and ADS full text. Return a dictionary of the JSON content.
    """
    with open(config.inputs.path_source_data, "r") as openfile:
        logger.info(f"Loading source dataset: {config.inputs.path_source_data}")
        source_dataset = json.load(openfile)
        if do_verbose:
            logger.debug(f"{len(source_dataset)} papers have been loaded")

    return source_dataset
