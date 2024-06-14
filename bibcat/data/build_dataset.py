"""
:title: build_dataset.py

This module will produce the input corpus data in JSON format by
combining the MAST papertrack JSON file and the ADS full text JSON file.

Run example: bibcat train

"""

import json
import os

import numpy as np

from bibcat import config
from bibcat.utils.logger_config import setup_logger

logger = setup_logger(__name__)


def build_dataset():
    logger.info("The script is building the dataset for bibcat!")

    # Throw an error if any of these files already exist
    tmp_list = [
        config.inputs.path_source_data,
        config.inputs.path_not_in_papertext,
        config.inputs.path_not_in_papertrack,
    ]
    if any([os.path.isfile(item) for item in tmp_list]):
        logger.error(
            "Err: File in the following list already exists and will not be overwritten."
            + f"\nPlease change the save destination for the notebook file(s) or move the existing files.\n{tmp_list}"
        )
        return

    else:
        # Load paper texts and papertrack classes
        with open(config.inputs.path_papertext) as openfile:
            dataset_papers_orig = json.load(openfile)
        with open(config.inputs.path_papertrack) as openfile:
            dataset_classes_orig = json.load(openfile)

        # Extract information from the paper text dataset
        bibcodes_papers = [item["bibcode"] for item in dataset_papers_orig]
        dates_papers = [item["pubdate"] for item in dataset_papers_orig]
        min_year_papers = min(dates_papers)
        max_year_papers = max(dates_papers)

        # Extract information from the paper classification dataset
        num_classifs = len(dataset_classes_orig)
        adssearch_classes = [None] * num_classifs
        bibcodes_classes = [None] * num_classifs
        all_missions_classes = [None] * num_classifs
        for ii in range(0, num_classifs):
            adssearch_classes[ii] = dataset_classes_orig[ii]["searches"]
            bibcodes_classes[ii] = dataset_classes_orig[ii]["bibcode"]
            all_missions_classes[ii] = dataset_classes_orig[ii]["class_missions"]

        # Throw an error if there are duplicate bibcodes in the paper classification dataset
        if not np.array_equal(np.sort(np.unique(bibcodes_classes)), np.sort(bibcodes_classes)):
            raise ValueError("Err: Duplicate bibcodes in database of paper classifications!")

        logger.debug(f"Min. date of papers within text database: {min_year_papers}.")
        logger.debug(f"Max. date of papers within text database: {max_year_papers}.")

        # Trim papertrack dictionary down to only columns to include
        try:
            storage = [
                {key: value for key, value in thisdict.items() if (key in config.inputs.keys_papertext)}.copy()
                for thisdict in dataset_papers_orig
            ]
        except AttributeError:  # If this error raised, probably earlier Python vers.
            storage = [
                {key: value for key, value in thisdict.iteritems() if (key in config.inputs.keys_papertext)}.copy()
                for thisdict in dataset_papers_orig
            ]

        # Verify that all papers within papertrack are within the papertext database
        bibcodes_notin_papertext = [val for val in np.unique(bibcodes_classes) if (val not in bibcodes_papers)]
        if len(bibcodes_notin_papertext) > 0:
            errstr = (
                "Note! Papers in papertrack not in text database!"
                + f"\n{bibcodes_notin_papertext}\n{len(bibcodes_notin_papertext)} of {len(bibcodes_classes)} papertrack entries in all.\n"
            )
            # raise ValueError(errstr)
            logger.debug(errstr)

        # Iterate through paper dictionary
        num_notin_papertrack = 0
        bibcodes_notin_papertrack = []
        for ii in range(0, len(storage)):
            # Extract information for current paper within text database
            curr_dict = storage[ii]  # Current dictionary
            curr_bibcode = curr_dict["bibcode"]

            # Extract index for current paper within papertrack (paper classification database)
            curr_ind_classes = None
            try:
                curr_ind_classes = bibcodes_classes.index(curr_bibcode)
            except ValueError:
                logger.debug(f"Bibcode ({ii}, {curr_bibcode}) not in papertrack database. Continuing...")
                bibcodes_notin_papertrack.append(curr_bibcode)
                num_notin_papertrack += 1
                continue

            # Copy over data from papertrack into text database
            curr_dict["class_missions"] = {}
            for jj in range(0, len(all_missions_classes[curr_ind_classes])):  # Missions for bibcode
                curr_mission = all_missions_classes[curr_ind_classes][jj]["mission"]
                curr_papertype = all_missions_classes[curr_ind_classes][jj]["paper_type"]
                # Store inner dictionary under mission name
                inner_dict = {}
                curr_dict["class_missions"][curr_mission] = inner_dict

                # Store information in inner dict
                inner_dict["bibcode"] = bibcodes_classes[curr_ind_classes]
                inner_dict["papertype"] = curr_papertype

            # Store search ignore flags
            for jj in range(0, len(adssearch_classes[curr_ind_classes])):
                curr_searchname = adssearch_classes[curr_ind_classes][jj]["search_key"]
                curr_ignored = adssearch_classes[curr_ind_classes][jj]["ignored"]
                curr_dict[f"is_ignored_{curr_searchname}"] = curr_ignored

        logger.debug("Done generating dictionaries of combined papertrack+text data.")
        logger.debug(f"NOTE: {num_notin_papertrack} papers in text data that were not in papertrack.")

        # Save the combined dataset
        with open(config.inputs.path_source_data, "w") as openfile:
            json.dump(storage, openfile, indent=2)
        # Also save the papertrack classifications not found in papertext
        np.savetxt(
            config.inputs.path_not_in_papertext,
            np.asarray(bibcodes_notin_papertext).astype(str),
            delimiter="\n",
            fmt="%s",
        )
        # Also save the paper-texts not found in papertrack
        np.savetxt(
            config.inputs.path_not_in_papertrack,
            np.asarray(bibcodes_notin_papertrack).astype(str),
            delimiter="\n",
            fmt="%s",
        )

        logger.debug("Dataset generation complete.\n")
        logger.debug(f"Combined .json file saved to:\n{config.inputs.path_source_data}\n")
        logger.debug(f"Bibcodes not in papertext saved to:\n{config.inputs.path_not_in_papertext}\n")
        logger.debug(f"Bibcodes not in papertrack saved to:\n{config.inputs.path_not_in_papertrack}\n")
