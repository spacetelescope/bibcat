import json
from functools import lru_cache
from typing import Dict

from bibcat import config
from bibcat.core import parameters as params
from bibcat.core.operator import Operator
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


def streamline_dataset(source_dataset: Dict, operator_ML: Operator, do_verbose_text_summary=False):
    """
    Organize a new version of the data with: key:text,class,id,mission structure
    """
    logger.info("Start streamlining dataset:")
    # keep track of used bibcodes avoiding duplicate dataset entries
    list_bibcodes = []

    # Track number of papers kept from original dataset
    i_track = 0
    dict_texts = {}
    for ii, curr_data in enumerate(source_dataset):
        # Extract mission classifications for current text
        # curr_data = source_dataset[ii]

        # Skip if no valid text at all for this text
        if "body" not in curr_data:
            continue

        # Skip if no valid missions at all for this text
        # This logic has to change to include all bib text for future work. see ASB-25819.
        if "class_missions" not in curr_data:
            continue

        # Otherwise, extract the bibcodes and missions
        curr_bibcode = curr_data["bibcode"]
        curr_missions = curr_data["class_missions"]
        logger.info(curr_bibcode)
        logger.info(curr_missions)

        # Skip if bibcode already encountered (and so duplicate entry)
        if curr_bibcode in list_bibcodes:
            logger.warning(f"Duplicate bibcode encountered: {curr_bibcode}. Skipping.")
            continue

        # Iterate through missions for this text
        i_mission = 0
        for curr_key in curr_missions:
            # If this is not an allowed mission, skip
            if curr_missions[curr_key]["papertype"] not in config.pretrained.papertypes:
                continue

            # Otherwise, check if this mission is a target mission
            fetched_kobj = operator_ML._fetch_keyword_object(lookup=curr_key, do_raise_emptyerror=False)
            # Skip if not a target
            if fetched_kobj is None:
                continue

            # Otherwise, store classification info for this entry
            curr_class = curr_missions[curr_key]["papertype"]
            new_dict = {
                "text": curr_data["body"],
                "bibcode": curr_data["bibcode"],
                "class": curr_class,  # Classification for this mission
                "mission": curr_key,
                "id": ("paper{0}_mission{1}_{2}_{3}".format(ii, i_mission, curr_key, curr_class)),
            }
            dict_texts[str(i_track)] = new_dict

            # Increment counters
            i_mission += 1  # Count of kept missions for this paper
            i_track += 1  # Count of kept classifications overall
        #

        # Record this bibcode as stored
        list_bibcodes.append(curr_bibcode)

        # Terminate early if requested number of papers reached
        if (settings.num_papers is not None) and (i_track >= settings.num_papers):
            break

    # Throw error if not enough text entries collected

    if (settings.num_papers is not None) and (len(dict_texts) < settings.num_papers):
        raise ValueError(
            "Err: Something went wrong during initial processing. "
            + "Insufficient number of texts extracted."
            + "\nRequested number of texts:"
            + " {0}\nActual number of texts: {1}\nChange the setting of num_papers to 'None' or a smaller number than the actual number in the model configuration file if you still want to continue.".format(
                settings.num_papers, len(dict_texts)
            )
        )

    # logger.info a snippet of each of the entries in the dataset.
    logger.info(f"Number of processed texts: {i_track}={len(dict_texts)}\n")
    if do_verbose_text_summary:
        for curr_key in dict_texts:
            logger.debug(f"Text #{curr_key}:")
            logger.debug(f"Classification: {dict_texts[curr_key]['class']}")
            logger.debug(f"Mission: {dict_texts[curr_key]['mission']}")
            logger.debug(f"ID: {dict_texts[curr_key]['id']}")
            logger.debug(f"Bibcode: {dict_texts[curr_key]['bibcode']}")
            logger.debug(f"Text snippet:{dict_texts[curr_key]["text"][0:500]} ---\n\n")

    # Print number of texts that fell under given parameters
    logger.info("Target missions:")
    for curr_kobj in params.all_kobjs:
        logger.info(curr_kobj + "\n")
    logger.info(f"\n{len(dict_texts)} of valid text entries have been streamlined.")

    return dict_texts
