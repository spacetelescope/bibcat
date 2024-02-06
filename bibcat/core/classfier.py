"""
:title: classify.py

This module is a class purely meant to be inherited by the various Classifier_* classes and inherits _Base() class.  
In short, the _Classifier class is a collection of methods used by different classifier types.

The primary methods and use cases of _Classifier include:
* `classify_text`: Base classification method, overwritten by various classifier types during inheritance.
* `_load_text`: Load text from a given filepath.
* `_process_text`: Use the Grammar class (and internally the Paper class) to process given text into modifs.
* `_write_text`: Write a text file to a given file path.

"""

import collections
import os

import numpy as np

import bibcat.config as config
from bibcat.core.base import _Base
from bibcat.core.grammar import Grammar


class _Classifier(_Base):
    """
    WARNING! This class is *not* meant to be used directly by users.
    -
    Class: _Classifier
    Purpose:
     - Container for common underlying methods used in Classifier_* classes.
     - Purely meant to be inherited by Classifier_* classes.
    -
    """

    # Initialize this class instance
    def __init__(self):
        """
        Method: __init__
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Initializes instance of _Classifier class.
        """
        # Nothing to see here - inheritance base
        pass

    # Base classification; overwritten by inheritance as needed
    def classify_text(self, text):
        """
        Method: classify_text
        WARNING! This method is a placeholder for inheritance by other classes.
        """
        # Nothing to see here - inheritance base
        pass

    # Split a given dictionary of classified texts into directories containing training,
    # validation, and testing datasets
    def generate_directory_TVT(
        self, dir_model, fraction_TVT, dict_texts, mode_TVT="uniform", do_shuffle=True, seed=10, do_verbose=None
    ):
        """
        Method: generate_directory_TVT
        Purpose: !!!
        """

        # Load global variables
        dataset = dict_texts
        filepath_dictinfo = config.path_TVTinfo
        name_folderTVT = [config.folders_TVT["train"], config.folders_TVT["validate"], config.folders_TVT["test"]]

        num_TVT = len(name_folderTVT)
        if num_TVT != len(fraction_TVT):
            raise ValueError("Err: fraction_TVT ({0}) needs {1} fractions.".format(fraction_TVT, num_TVT))

        if do_verbose is None:
            do_verbose = self._get_info("do_verbose")

        # Set random seed, if requested
        if do_shuffle:
            np.random.seed(seed)

        # Print some notes
        if do_verbose:
            print("\n> Running generate_directory_TVT().")
            if do_shuffle:
                print("Random seed set to: {0}".format(seed))

        # Load all unique paper identifiers (the bibcodes) from across texts
        unique_bibcodes = np.unique([dataset[key]["bibcode"] for key in dataset]).tolist()
        # Print some notes
        if do_verbose:
            print(
                "\nDataset contains {0} papers with {1} unique bibcodes.\n".format(len(dataset), len(unique_bibcodes))
            )

        # Load all unique classifs across all texts
        count_classes = collections.Counter([dataset[key]["class"] for key in dataset])
        unique_classes = count_classes.keys()
        # Print some notes
        if do_verbose:
            print("\nClass breakdown of given dataset:\n{0}\n".format(count_classes))

        # Invert dataset so that classifs are stored under bibcode keys
        dict_bibcode_classifs = {key: [] for key in unique_bibcodes}
        dict_bibcode_textids = {key: [] for key in unique_bibcodes}
        for curr_key in dataset:
            curr_id = curr_key
            curr_classif = dataset[curr_key]["class"]
            curr_bibcode = dataset[curr_key]["bibcode"]
            # Store id for this bibcode
            dict_bibcode_textids[curr_bibcode].append(curr_key)
            # Store this classif for this bibcode, if not already stored
            if curr_classif not in dict_bibcode_classifs[curr_bibcode]:
                dict_bibcode_classifs[curr_bibcode].append(curr_classif)  # Store

        # Record the number of processed text ids for later
        num_textids = sum([len(dict_bibcode_textids[key]) for key in unique_bibcodes])

        # Print some notes
        if do_verbose:
            print("\nGiven dataset inverted. Bibcode count: {0}".format(len(dict_bibcode_classifs)))

        # Label bibcodes with representative classif
        # Separate bibcodes based on number of unique associated classifs
        all_bibcodes_oneclassif = {
            key1: [key2 for key2 in unique_bibcodes if (dict_bibcode_classifs[key2] == [key1])]
            for key1 in unique_classes
        }
        all_bibcodes_multiclassif = [key for key in unique_bibcodes if (len(dict_bibcode_classifs[key]) > 1)]

        # Print some notes
        if do_verbose:
            print(
                (
                    "\nNumber of processed text ids: {2}"
                    + "\nNumber of bibcodes with single unique classif: {0}"
                    + "\nNumber of bibcodes with multiple classifs: {1}"
                    + "\nNumber of bibcodes with multiple text ids: {3}"
                ).format(
                    sum([len(all_bibcodes_oneclassif[key]) for key in unique_classes]),
                    len(all_bibcodes_multiclassif),
                    num_textids,
                    len([key for key in dict_bibcode_textids if (len(dict_bibcode_textids[key]) > 1)]),
                )
            )

        # Throw an error if separated bibcode count does not equal original count
        tmp_check = sum([len(all_bibcodes_oneclassif[key]) for key in unique_classes]) + len(all_bibcodes_multiclassif)
        if tmp_check != len(unique_bibcodes):
            raise ValueError(
                "Err: Invalid separation of bibcodes occurred.\n"
                + "{0} + {1} != {2}".format(
                    len([all_bibcodes_oneclassif[key] for key in unique_classes]),
                    len(all_bibcodes_multiclassif),
                    len(unique_bibcodes),
                )
            )

        # Shuffle the list of multi-classif bibcodes, if requested
        if do_shuffle:
            np.random.shuffle(all_bibcodes_multiclassif)

        # Partition multi-classif bibcodes into representative classif lists
        all_bibcodes_partitioned_lists = all_bibcodes_oneclassif.copy()
        all_bibcodes_partitioned_counts = {
            key: len(all_bibcodes_partitioned_lists[key]) for key in unique_classes
        }  # Starting count of bibcodes per repr.clf.

        # Iterate through multi-classif bibcodes
        for curr_bibcode in all_bibcodes_multiclassif:
            curr_classifs = dict_bibcode_classifs[curr_bibcode]
            # Determine which classif has least number of bibcodes so far
            curr_counts = [all_bibcodes_partitioned_counts[key] for key in curr_classifs]  # Current bibcode counts
            # Store this bibcode under representative classif with min. count
            ind_min = np.argmin(curr_counts)  # Index of classif with min. count
            min_classif = curr_classifs[ind_min]
            all_bibcodes_partitioned_lists[min_classif].append(curr_bibcode)  # Append bibc.
            # Update count of this representative classif
            all_bibcodes_partitioned_counts[min_classif] += 1

        # Throw an error if any bibcodes are missing/duplicated in partition
        tmp_check = sum(all_bibcodes_partitioned_counts.values())

        if tmp_check != len(unique_bibcodes):
            raise ValueError(
                "Err: Bibcode count does not match original"
                + " bibcodes.\n{0}\nvs. {1}".format(tmp_check, len(unique_bibcodes))
            )
        tmp_check = [
            item for key in all_bibcodes_partitioned_lists for item in all_bibcodes_partitioned_lists[key]
        ]  # Part. bibc.

        if sorted(tmp_check) != sorted(unique_bibcodes):
            raise ValueError(
                "Err: Bibcode partitioning does not match original"
                + " bibcodes.\n{0}\nvs. {1}".format(tmp_check, unique_bibcodes)
            )

        # Print some notes
        if do_verbose:
            print("\nBibcode partitioning of representative classifs.:\n{0}\n".format(all_bibcodes_partitioned_counts))

        # Split bibcode counts into TVT sets: training, validation, testing =TVT
        valid_modes = ["uniform", "available"]
        fraction_TVT = np.asarray(fraction_TVT) / sum(fraction_TVT)  # Normalize
        # For mode where training sets should be uniform in size
        if mode_TVT.lower() == "uniform":
            min_count = min(all_bibcodes_partitioned_counts.values())
            dict_split = {
                key: (np.round((fraction_TVT * min_count))).astype(int) for key in unique_classes
            }  # Partition per classif per TVT
            # Update split to send remaining bibcodes into testing datasets
            for curr_key in unique_classes:
                curr_max = all_bibcodes_partitioned_counts[curr_key]
                curr_used = dict_split[curr_key][0] + dict_split[curr_key][1]
                dict_split[curr_key][2] = curr_max - curr_used

        # For mode where training sets should use fraction of data available
        elif mode_TVT.lower() == "available":
            max_count = max(all_bibcodes_partitioned_counts.values())
            dict_split = {
                key: (np.round((fraction_TVT * all_bibcodes_partitioned_counts[key]))).astype(int)
                for key in unique_classes
            }  # Partition per class per TVT

        # Otherwise, throw error if mode not recognized
        else:
            raise ValueError(
                (
                    "Err: The given mode for generating the TVT" + " directory {0} is invalid. Valid modes are: {1}"
                ).format(mode_TVT, valid_modes)
            )

        # Print some notes
        if do_verbose:
            print("Fractions given for TVT split: {0}\nMode requested: {1}".format(fraction_TVT, mode_TVT))
            print("Target TVT partition for bibcodes:")
            for curr_key in unique_classes:
                print("{0}: {1}".format(curr_key, dict_split[curr_key]))

        # Verify splits add up to original file count
        for curr_key in unique_classes:
            if all_bibcodes_partitioned_counts[curr_key] != sum(dict_split[curr_key]):
                raise ValueError("Err: Split did not use all data available!")

        # Prepare indices for extracting TVT sets per class
        dict_bibcodes_perTVT = {key: [None for ii in range(0, num_TVT)] for key in unique_classes}
        for curr_key in unique_classes:
            # Fetch bibcodes represented by this class
            curr_list = np.asarray(all_bibcodes_partitioned_lists[curr_key])

            # Fetch available indices to select these bibcodes
            curr_inds = np.arange(0, all_bibcodes_partitioned_counts[curr_key], 1)
            # Shuffle, if requested
            if do_shuffle:
                np.random.shuffle(curr_inds)

            # Split out the bibcodes
            i_start = 0  # Accumulated place within overarching index array
            for ii in range(0, num_TVT):  # Iterate through TVT
                i_end = i_start + dict_split[curr_key][ii]  # Ending point
                dict_bibcodes_perTVT[curr_key][ii] = curr_list[curr_inds[i_start:i_end]]
                i_start = i_end  # Update latest starting place in array

        # Throw an error if any bibcodes not accounted for
        tmp_check = [
            item2 for key in dict_bibcodes_perTVT for item1 in dict_bibcodes_perTVT[key] for item2 in item1
        ]  # All bibcodes used
        if not np.array_equal(np.sort(tmp_check), np.sort(unique_bibcodes)):
            raise ValueError(
                "Err: Split bibcodes do not match up with"
                + " original bibcodes:\n\n{0}\nvs.\n{1}".format(np.sort(tmp_check), np.sort(unique_bibcodes))
            )

        # Print some notes
        if do_verbose:
            print("\nIndices split per bibcode, per TVT. Shuffling={0}.".format(do_shuffle))
            print("Number of indices per class, per TVT:")
            for curr_key in unique_classes:
                print("{0}: {1}".format(curr_key, [len(item) for item in dict_bibcodes_perTVT[curr_key]]))

        # Build new directories to hold TVT (or throw error if exists)
        # Build model directory, if does not already exist
        if not os.path.exists(dir_model):
            os.mkdir(dir_model)

        # Verify TVT directories do not already exist
        if any([os.path.exists(os.path.join(dir_model, item)) for item in name_folderTVT]):
            raise ValueError(
                "Err: TVT directories exist in model directory"
                + " and will not be overwritten. Please remove or"
                + " change the given model directory (dir_model)."
                + "\nCurrent dir_model: {0}".format(dir_model)
            )

        # Otherwise, make the directories
        for curr_folder in name_folderTVT:
            os.mkdir(os.path.join(dir_model, curr_folder))
            # Iterate through classes and create subfolder per class
            for curr_key in unique_classes:
                os.mkdir(os.path.join(dir_model, curr_folder, curr_key))

        # Print some notes
        if do_verbose:
            print("Created new directories for TVT files.\nStored in: {0}".format(dir_model))

        # Save texts to .txt files within class directories
        dict_info = {}
        saved_filenames = [None] * num_textids
        used_bibcodes = [None] * len(unique_bibcodes)
        used_textids = [None] * num_textids
        all_texts_partitioned_counts = {
            key1: {key2: 0 for key2 in unique_classes} for key1 in name_folderTVT
        }  # Count of texts per TVT, per classif

        # Iterate through classes
        i_track = 0
        i_bibcode = 0
        curr_key = None
        for repr_key in unique_classes:
            # Save each text to assigned TVT
            for ind_TVT in range(0, num_TVT):  # Iterate through TVT
                # Iterate through bibcodes assigned to this TVT
                for curr_bibcode in dict_bibcodes_perTVT[repr_key][ind_TVT]:
                    # Throw error if used bibcode
                    if curr_bibcode in used_bibcodes:
                        raise ValueError("Err: Used bibcode {0} in {1}:TVT={2}".format(curr_bibcode, repr_key, ind_TVT))

                    # Record assigned TVT and representative classif for bibcode
                    dict_info[curr_bibcode] = {
                        "repr_class": repr_key,
                        "folder_TVT": name_folderTVT[ind_TVT],
                        "storage": {},
                    }

                    # Iterate through texts associated with this bibcode
                    for curr_textid in dict_bibcode_textids[curr_bibcode]:
                        curr_data = dataset[curr_textid]  # Data for this text
                        act_key = curr_data["class"]  # Actual class of text id
                        curr_filename = "{0}_{1}_{2}".format("text", act_key, curr_textid)

                        if curr_data["id"] is not None:  # Add id
                            curr_filename += "_{0}".format(curr_data["id"])

                        # Throw error if used text id
                        if curr_textid in used_textids:
                            raise ValueError("Err: Used text id: {0}".format(curr_textid))

                        # Throw error if not unique filename
                        if curr_filename in saved_filenames:
                            raise ValueError("Err: Non-unique filename: {0}".format(curr_filename))

                        # Write this text to new file
                        curr_filebase = os.path.join(
                            dir_model, name_folderTVT[ind_TVT], act_key
                        )  # TVT path; use actual class!
                        self._write_text(
                            text=curr_data["text"], filepath=os.path.join(curr_filebase, (curr_filename + ".txt"))
                        )

                        # Store record of this storage in info dictionary
                        dict_info[curr_bibcode]["storage"][curr_textid] = {
                            "filename": curr_filename,
                            "class": act_key,
                            "mission": dataset[curr_textid]["mission"],
                        }

                        # Increment count of texts in this classif. and TVT dir.
                        all_texts_partitioned_counts[name_folderTVT[ind_TVT]][act_key] += 1
                        used_textids[i_track] = curr_textid  # Check off text id
                        saved_filenames[i_track] = curr_filename  # Check off file
                        i_track += 1  # Increment place in list of filenames

                # Store and increment count of used bibcodes
                used_bibcodes[i_bibcode] = curr_bibcode
                i_bibcode += 1

        # Throw an error if any bibcodes not accounted for in partitioning
        tmp_check = len(dict_info)
        if tmp_check != len(unique_bibcodes):
            raise ValueError(
                "Err: Count of bibcodes does not match original"
                + " bibcode count.\n{0} vs. {1}".format(tmp_check, len(unique_bibcodes))
            )

        # Throw an error if any text ids not accounted for in partitioning
        tmp_check = sum([sum(all_texts_partitioned_counts[key].values()) for key in all_texts_partitioned_counts])
        if tmp_check != num_textids:
            raise ValueError(
                "Err: Count of text ids does not match original"
                + " text id count.\n{0}\nvs. {1}\n{2}".format(tmp_check, num_textids, all_texts_partitioned_counts)
            )

        # Throw an error if not enough filenames saved
        if None in saved_filenames:
            tmp_check = len([item for item in saved_filenames if (item is not None)])
            raise ValueError("Err: Only subset of filenames saved: {0} vs {1}.".format(tmp_check, num_textids))

        # Print some notes
        if do_verbose:
            print("\nFiles saved to new TVT directories.")
            print("Final partition of texts across classes and TVT dirs.:\n{0}".format(all_texts_partitioned_counts))

        # Save the dictionary of TVT bibcode partitioning to its own file
        # tmp_filesave = filepath_dictinfo
        np.save(filepath_dictinfo, dict_info)
        # Print some notes
        if do_verbose:
            print("Dictionary of TVT bibcode partitioning info saved at: {0}.".format(filepath_dictinfo))

        # Verify that count of saved .txt files adds up to original data count
        for curr_key in unique_classes:
            # Count items in this class across TVT directories
            curr_count = sum(
                [
                    len(
                        [
                            item2
                            for item2 in os.listdir(os.path.join(dir_model, item1, curr_key))
                            if (item2.endswith(".txt"))
                        ]
                    )
                    for item1 in name_folderTVT
                ]
            )

            # Verify count
            if curr_count != count_classes[curr_key]:
                raise ValueError(
                    "Err: Unequal class count in {0}!\n{1} vs {2}".format(curr_key, curr_count, count_classes[curr_key])
                )
            #
        #

        # Exit the method
        if do_verbose:
            print("\nRun of generate_directory_TVT() complete.\n---\n")

        return

    # Load text and process into modifs using Grammar class
    def _process_text(self, text, keyword_obj, which_mode, do_check_truematch, buffer=0, do_verbose=False):
        """
        Method: _process_text
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Process text into modifs using Grammar class.
        """

        # Generate and store instance of Grammar class for this text
        use_these_modes = list(set([which_mode, "none"]))
        grammar = Grammar(
            text, keyword_obj=keyword_obj, do_check_truematch=do_check_truematch, do_verbose=do_verbose, buffer=buffer
        )
        grammar.run_modifications(which_modes=use_these_modes)
        self._store_info(grammar, "grammar")

        # Fetch modifs and grammar information
        set_info = grammar.get_modifs(which_modes=[which_mode], do_include_forest=True)
        modif = set_info["modifs"][which_mode]
        forest = set_info["_forest"]

        # Return all processed statements
        return {"modif": modif, "forest": forest}
