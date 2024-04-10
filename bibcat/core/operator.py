"""
:title: operator.py

The primary purpose of this class is to direct and run the entire workflow of 'bibcat',
from reading in a given block of text to ultimately classifying that block of text.
Using its `classify` method, the Operator class internally handles all calls to
the other classes (Paper, Grammar, and the given classifier).

The primary methods and use cases of Operator are:
* `_fetch_keyword_object`: A hidden method for fetching the stored Keyword instance
   that matches a given term.
* `classify`: A method designed for users that prepares and runs the entire 'bibcat' workflow,
   from input raw text to classified output.
* `process`: A method designed for users that processes given text into modifs,
   from input raw text to output modifs. It does not include classification (for that, run `classify`);
   it is useful for preprocessing raw text.
* `train_model_ML`: A method designed for users that trains a machine learning (ML) model
   on input raw text. Under the hood, this is called `process` to preprocess the raw text.
"""

import os

import numpy as np

import bibcat.config as config
from bibcat.core.base import Base
from bibcat.core.grammar import Grammar
from bibcat.data.partition_dataset import generate_directory_TVT


class Operator(Base):
    """
    Class: Operator
    Purpose:
        - Run full workflow of text classification, from input text to internal text processing to output classification.
    Initialization Arguments:
        - classifier [*Classifier instance]:
          - Classifier to use for classification.
        - keyword_objs [list of Keyword instances]:
          - Target missions; terms will be used to search the text.
        - mode [str]:
          - Mode of modification for generating modifs using the Grammar class.
        - name [str (default="operator")]:
          - A unique name for this Operator.
        - load_check_truematch [bool (default=True)]:
          - Whether nor not to load external data for verifying ambiguous terms as true vs false matches to mission.
        - do_verbose [bool (default=False)]:
          - Whether or not to print surface-level log information and tests.
    """

    # Initialize this class instance
    def __init__(
        self,
        classifier,
        mode,
        keyword_objs,
        do_verbose,
        name="operator",
        load_check_truematch=True,
        do_verbose_deep=False,
    ):
        """
        Method: __init__
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Initializes instance of Operator class.
        """
        # Initialize storage
        self._storage = {}
        self._store_info(classifier, "classifier")
        self._store_info(mode, "mode")
        self._store_info(name, "name")
        self._store_info(keyword_objs, "keyword_objs")
        self._store_info(do_verbose, "do_verbose")
        self._store_info(load_check_truematch, "load_check_truematch")
        self._store_info(do_verbose_deep, "do_verbose_deep")

        # Load and process ambiguous (ambig.) data, if so requested
        if load_check_truematch:
            # Run method to load and process external ambig. database
            dict_ambigs = self._process_database_ambig(keyword_objs=keyword_objs)
            lookup_ambigs = dict_ambigs["lookup_ambigs"]

            # Print some notes
            if do_verbose_deep:
                print("Loaded+Assembled data for ambiguous phrases.")

        # Otherwise, set empty placeholders
        else:
            dict_ambigs = None
            lookup_ambigs = None

        # Store the processed data in this object instance
        self._store_info(dict_ambigs, "dict_ambigs")
        self._store_info(lookup_ambigs, "lookup_ambigs")

        # Exit the method
        if do_verbose:
            print("Instance of Operator successfully initialized!")
            print("Keyword objects:")
            for ii in range(0, len(keyword_objs)):
                print("{0}: {1}".format(ii, keyword_objs[ii]))

        return

    # Fetch a keyword object that matches the given lookup
    def _fetch_keyword_object(self, lookup: str, do_verbose: None | bool = None, do_raise_emptyerror: bool = True):
        """
        Method: _fetch_keyword_object
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Finds stored Keyword instance that matches to given lookup term.
        """
        # Load Global variables
        if do_verbose is None:
            do_verbose = self._get_info("do_verbose")
        keyword_objs = self._get_info("keyword_objs")
        num_keyobjs = len(keyword_objs)
        # Print some notes
        if do_verbose:
            print("> Running _fetch_keyword_object() for lookup term {0}.".format(lookup))

        # Find keyword object that matches to given lookup term
        match = None
        for ii in range(0, num_keyobjs):
            # If current keyword object matches, record and stop loop
            if keyword_objs[ii].is_keyword(lookup):
                match = keyword_objs[ii]
                break

        # Throw error if no matching keyword object found
        if match is None:
            errstr = "No matching keyword object for {0}.\n".format(lookup)
            errstr += "Available keyword objects are:\n"
            for ii in range(0, num_keyobjs):
                errstr += "{0}\n".format(keyword_objs[ii])

            # Raise error if so requested
            if do_raise_emptyerror:
                raise ValueError(errstr)
            # Otherwise, return None
            else:
                return None

        # Return the matching keyword object
        return match

    # Inspect text and either reject as false target or give classifications
    def classify(
        self,
        text,
        lookup,
        threshold,
        do_check_truematch,
        do_raise_innererror,
        modif=None,
        forest=None,
        buffer=0,
        do_verbose=None,
        do_verbose_deep=None,
    ):
        """
        Method: classify
        Purpose:
          - Accept text and process it into modifs (using Grammar, Paper classes).
          - Classify that text (using stored classifier).
        Arguments:
          - buffer [int (default=0)]:
            - Number of +/- sentences around a sentence containing a target mission to include in the paragraph.
          - do_check_truematch [bool]:
            - Whether or not to check that mission phrases found in text are known true vs. false matches. (E.g., 'Edwin Hubble' as false match for the Hubble Space Telescope).
          - do_raise_innererror [bool]:
            - Whether or not to throw an explicit error if an internal error is encountered. If False, will print notes and continue running code.
          - lookup [str]:
            - A term for looking up the target Keyword instance (e.g. 'HST').
          - text [str]:
            - The text to classify.
          - threshold [number]:
            - The minimum uncertainty allowed for returning classification (instead of a flag).
          - do_verbose [bool (default=False)]:
            - Whether or not to print surface-level log information and tests.
          - do_verbose_deep [bool (default=False)]:
            - Whether or not to print inner log information and tests.
        Returns:
          - dict:
            - 'modif': the modif.
            - 'verdict': the classification.
            - 'scores_comb': the final score.
            - 'scores_indiv': the individual scores.
            - 'uncertainty': the uncertainty of the classification.
        """
        # Fetch global variables
        if do_verbose is None:
            do_verbose = self._get_info("do_verbose")
        if do_verbose_deep is None:
            do_verbose_deep = self._get_info("do_verbose_deep")

        classifier = self._get_info("classifier")

        # Fetch keyword object matching to the given keyword
        keyobj = self._fetch_keyword_object(lookup=lookup, do_verbose=do_verbose)
        if do_verbose:
            print("Best matching keyword object (keyobj) for keyword {0}:\n{1}".format(lookup, keyobj))

        # Process text into modifs using Grammar class, if modif not given
        if modif is None:
            if do_verbose:
                print("\nPreprocessing and extracting modifs from the text...")

            if do_raise_innererror:  # Allow stop from any exceptions encountered
                output = self.process(
                    text=text,
                    do_check_truematch=do_check_truematch,
                    buffer=buffer,
                    lookup=lookup,
                    keyword_obj=keyobj,
                    do_verbose=do_verbose,
                    do_verbose_deep=do_verbose_deep,
                )
            else:  # Otherwise, print any exceptions and keep moving forward
                try:
                    output = self.process(
                        text=text,
                        do_check_truematch=do_check_truematch,
                        buffer=buffer,
                        lookup=lookup,
                        keyword_obj=keyobj,
                        do_verbose=do_verbose,
                        do_verbose_deep=do_verbose_deep,
                    )

                # Catch any exceptions and force-print some notes
                except Exception as err:
                    dict_verdicts = config.dictverdict_error.copy()
                    print("-\nThe following err. was encountered in operate:")
                    print(repr(err))
                    print("Error was noted. Returning error as verdict.\n-")
                    dict_verdicts["modif"] = "<PROCESSING ERROR:\n{0}>".format(repr(err))
                    dict_verdicts["modif_none"] = None
                    return dict_verdicts

            # Fetch the generated output
            modif = output["modif"]
            modif_none = output["modif_none"]
            forest = output["forest"]

            # Print some notes
            if do_verbose:
                print("Text has been processed into modif.")

        # Otherwise, use given modif
        else:
            # Print some notes
            if do_verbose:
                print("Modif given. No text processing will be done.")
            #
            modif_none = None
            pass

        # Set rejected verdict if empty text
        if modif.strip() == "":
            if do_verbose:
                print("No text found matching keyword object.")
                print("Returning rejection verdict.")

            dict_verdicts = config.dictverdict_rejection.copy()

        # Classify the text using stored classifier with raised error
        elif do_raise_innererror:  # If True, allow raising of inner errors
            dict_verdicts = classifier.classify_text(
                text=modif,
                do_check_truematch=do_check_truematch,
                forest=forest,
                keyword_obj=keyobj,
                do_verbose=do_verbose,
                threshold=threshold,
            )

        # Otherwise, run classification while ignoring inner errors
        else:
            # Try running classification
            try:
                dict_verdicts = classifier.classify_text(
                    text=modif,
                    do_check_truematch=do_check_truematch,
                    forest=forest,
                    keyword_obj=keyobj,
                    do_verbose=do_verbose,
                    threshold=threshold,
                )

            # Catch certain exceptions and force-print some notes
            except Exception as err:
                dict_verdicts = config.dictverdict_error.copy()
                print("-\nThe following err. was encountered in operate:")
                print(repr(err))
                print("Error was noted. Continuing.\n-")

        # Return the verdict with modif included
        dict_verdicts["modif"] = modif
        dict_verdicts["modif_none"] = modif_none
        return dict_verdicts

    # Classify set of texts as false target or give classifications
    def classify_set(
        self,
        texts,
        threshold,
        do_check_truematch,
        do_raise_innererror,
        modifs=None,
        forests=None,
        buffer=0,
        print_freq=25,
        do_verbose=None,
        do_verbose_deep=None,
    ):
        """
        Method: classify_set
        Purpose:
          - !
          - Accept text and process it into modifs (using Grammar, Paper classes).
          - Classify that text (using stored classifier).
        Arguments:
          - !
          - do_verbose [bool (default=False)]:
            - Whether or not to print surface-level log information and tests.
          - do_verbose_deep [bool (default=False)]:
            - Whether or not to print inner log information and tests.
        Returns:
          - List of dicts, each containing:
            - 'modif': the modif.
            - 'verdict': the classification.
            - 'scores_comb': the final score.
            - 'scores_indiv': the individual scores.
            - 'uncertainty': the uncertainty of the classification.
        """
        # Fetch global variables
        if do_verbose is None:
            do_verbose = self._get_info("do_verbose")
        if do_verbose_deep is None:
            do_verbose_deep = self._get_info("do_verbose_deep")

        all_kobjs = self._get_info("keyword_objs")
        num_kobjs = len(all_kobjs)

        # Throw error if both texts and modifs given
        if (texts is not None) and (modifs is not None):
            raise ValueError("Err: texts OR modifs should be given, not both.")
        elif texts is not None:
            num_texts = len(texts)
        elif modifs is not None:
            num_texts = len(modifs)

        # Print some notes
        if do_verbose:
            print("\n> Running classify_set()!")

        # Classify every text against every mission
        list_results = [None] * num_texts
        curr_text = None
        curr_modif = None
        curr_forest = None
        # Iterate through texts
        for ii in range(0, num_texts):
            curr_dict = {}  # Dictionary to hold set of results for this text
            list_results[ii] = curr_dict  # Store this dictionary

            # Extract current text if given in raw (not processed) form
            if texts is not None:
                curr_text = texts[ii]  # Current text to classify

            # Extract current modifs and forests if already processed text
            if modifs is not None:
                curr_modif = modifs[ii]

            if forests is not None:
                curr_forest = forests[ii]

            # Iterate through keyword objects
            for jj in range(0, num_kobjs):
                curr_kobj = all_kobjs[jj]
                curr_name = curr_kobj._get_info("name")
                # Classify current text for current mission
                curr_result = self.classify(
                    text=curr_text,
                    lookup=curr_name,
                    modif=curr_modif,
                    forest=curr_forest,
                    threshold=threshold,
                    buffer=buffer,
                    do_check_truematch=do_check_truematch,
                    do_raise_innererror=do_raise_innererror,
                    do_verbose=do_verbose_deep,
                )

                # Store current result
                curr_dict[curr_name] = curr_result

            # Print some notes at given frequency, if requested
            if do_verbose and ((ii % print_freq) == 0):
                print("Classification for text #{0} of {1} complete...".format((ii + 1), num_texts))

        # Return the classification results
        if do_verbose:
            print("\nRun of classify_set() complete!\n")

        return list_results

    # Process text into modifs
    def process(
        self, text, do_check_truematch, buffer=0, lookup=None, keyword_obj=None, do_verbose=None, do_verbose_deep=None
    ):
        """
        Method: process
        Purpose:
          - Accept text and process it into modifs (using Grammar, Paper classes).
          - Classify that text (using stored classifier).
        Arguments:
          - buffer [int (default=0)]:
            - Number of +/- sentences around a sentence containing a target mission to include in the paragraph.
          - do_check_truematch [bool]:
            - Whether or not to check that mission phrases found in text are known true vs. false matches. (E.g., 'Edwin Hubble' as false match for the Hubble Space Telescope).
          - keyword_obj [<Keyword object> or None]:
            - Target Keyword instance. If None, the input variable lookup will be used to look up the Keyword instance.
          - lookup [str or None]:
            - A term for looking up the target Keyword instance (e.g. 'HST'). Required if keyobj is None.
          - text [str]:
            - The text to classify.
          - do_verbose [bool (default=False)]:
            - Whether or not to print surface-level log information and tests.
          - do_verbose_deep [bool (default=False)]:
            - Whether or not to print inner log information and tests.
        Returns:
          - dict:
            - 'modif': the modif.
            - 'forest': the output from internal text processing.
        """

        # Fetch global variables
        if do_verbose is None:
            do_verbose = self._get_info("do_verbose")
        if do_verbose_deep is None:
            do_verbose_deep = self._get_info("do_verbose_deep")

        mode = self._get_info("mode")
        dict_ambigs = self._get_info("dict_ambigs")

        # Fetch keyword object matching to the given keyword
        if keyword_obj is None:
            keyword_obj = self._fetch_keyword_object(lookup=lookup, do_verbose=do_verbose_deep)
            if do_verbose:
                print("Best matching Keyword object for keyword {0}:\n{1}".format(lookup, keyword_obj))

        # Process text into modifs using Grammar class
        if do_verbose:
            print("\nRunning Grammar on the text...")
        use_these_modes = list(set([mode, "none"]))
        grammar = Grammar(
            text=text,
            keyword_obj=keyword_obj,
            do_check_truematch=do_check_truematch,
            dict_ambigs=dict_ambigs,
            do_verbose=do_verbose_deep,
            buffer=buffer,
        )
        grammar.run_modifications(which_modes=use_these_modes)
        output = grammar.get_modifs(do_include_forest=True)
        modif = output["modifs"][mode]
        modif_none = output["modifs"]["none"]  # Include unmodified vers. as well
        forest = output["_forest"]

        # Print some notes
        if do_verbose:
            print("Text has been processed into modifs.")

        # Return the modif and internal processing output
        return {"modif": modif, "modif_none": modif_none, "forest": forest}

    # Process text into modifs and then train ML model on the modifs
    def train_model_ML(
        self,
        dir_model,
        dir_data,
        name_model,
        do_reuse_run,
        dict_texts,
        mapper,
        do_check_truematch,
        seed_TVT=10,
        seed_ML=8,
        buffer=0,
        fraction_TVT=[0.8, 0.1, 0.1],
        mode_TVT="uniform",
        do_shuffle=True,
        print_freq=25,
        do_verbose=None,
        do_verbose_deep=None,
    ):
        """
        Method: train_model_ML
        Purpose:
          - Accept set of texts and process them into modifs (using Grammar, Paper classes).
          - Train stored machine learning (ML) classifier on the modifs.
        Arguments:
          - buffer [int (default=0)]:
            - Number of +/- sentences around a sentence containing a target mission to include in the paragraph.
          - do_reuse_run [bool]:
            - Whether or not to reuse outputs (e.g., existing training, validation directories) from any existing previous run with the same model name.
          - do_check_truematch [bool]:
            - Whether or not to check that mission phrases found in text are known true vs. false matches. (E.g., 'Edwin Hubble' as false match for the Hubble Space Telescope).
          - do_shuffle [bool]:
            - Whether or not to shuffle texts when generating training, validation, and testing directories.
          - do_verbose [bool (default=False)]:
            - Whether or not to print surface-level log information and tests.
          - do_verbose_deep [bool (default=False)]:
            - Whether or not to print inner log information and tests.
        Returns:
          - dict:
            - 'modif': the modif.
            - 'verdict': the classification.
            - 'scores_comb': the final score.
            - 'scores_indiv': the individual scores.
            - 'uncertainty': the uncertainty of the classification.
        """
        # Fetch global variables
        if do_verbose is None:
            do_verbose = self._get_info("do_verbose")
        if do_verbose_deep is None:
            do_verbose_deep = self._get_info("do_verbose_deep")

        dataset = dict_texts
        classifier = self._get_info("classifier")
        folders_TVT = config.folders_TVT
        savename_ML = config.tfoutput_prefix + name_model
        savename_model = name_model + ".npy"
        filepath_dicterrors = config.path_modiferrors

        # Print some notes
        if do_verbose:
            print("\n> Running train_model_ML()!")

        # Throw error if invalid classifier given
        allowed_types = ["MachineLearningClassifier"]
        if type(classifier).__name__ not in allowed_types:
            raise ValueError("Err: Classifier ({0}) not allowed type ({1})".format(type(classifier), allowed_types))

        # Preprocess texts into modifs and store in TVT directories
        # NOTE: TVT = training, validation, testing datasets
        is_exist = os.path.exists(os.path.join(dir_data, folders_TVT["train"])) or os.path.exists(
            os.path.join(dir_data, folders_TVT["validate"])
        )
        # If TVT directories already exist, either print note or raise error
        if is_exist:
            str_err = None  # Placeholder
            # Print some notes
            if do_verbose:
                print("Previous training/validation directories already exist.")

            # Skip ahead if previous data should be reused
            if do_reuse_run:
                print("Reusing the existing training/validation data in {0}.".format(dir_data))
                pass

            # Otherwise, raise error if not to reuse previous run data
            else:
                raise ValueError(
                    (
                        "Err: Training/validation data already exists"
                        + " in {0}. Either delete it, or rerun method"
                        + " with do_reuse_run=True."
                    ).format(dir_data)
                )

        # Otherwise, preprocess the text and store in TVT directories
        else:
            # Print some notes
            if do_verbose:
                print("Processing text data into modifs...")

            # Process each text within the database into a modif
            dict_errors = {}  # Container for modifs with caught processing errors
            dict_modifs = {}  # Container for modifs and text classification info
            i_track = 0
            i_skipped = 0
            str_err = ""
            num_data = len(dataset)
            for curr_key in dataset:
                old_dict = dataset[curr_key]
                masked_class = mapper[old_dict["class"].lower()]

                # Extract modif for current text
                if do_check_truematch:  # Catch and print unknown ambig. phrases
                    try:
                        curr_res = self.process(
                            text=old_dict["text"],
                            do_check_truematch=do_check_truematch,
                            buffer=buffer,
                            lookup=old_dict["mission"],
                            keyword_obj=None,
                            do_verbose=do_verbose_deep,
                            do_verbose_deep=do_verbose_deep,
                        )
                    except NotImplementedError as err:
                        curr_str = (
                            "\n-\n" + "Printing Error:\nID: {0}\nBibcode: {1}\n" + "Mission: {2}\nMasked class: {3}\n"
                        ).format(old_dict["id"], old_dict["bibcode"], old_dict["mission"], masked_class)
                        curr_str += "The following err. was encountered" + " in train_model_ML:\n"
                        curr_str += repr(err)
                        curr_str += "\nError was noted. Skipping this paper.\n-"
                        print(curr_str)  # Print current error
                        #
                        # Store this error-modif
                        err_dict = {
                            "text": curr_str,
                            "class": masked_class,  # Mask class
                            "id": old_dict["id"],
                            "mission": old_dict["mission"],
                            "forest": None,
                            "bibcode": old_dict["bibcode"],
                        }
                        if old_dict["bibcode"] not in dict_errors:
                            dict_errors[old_dict["bibcode"]] = {}
                        dict_errors[old_dict["bibcode"]][curr_key] = err_dict

                        str_err += curr_str  # Tack this error onto full string
                        i_skipped += 1  # Increment count of skipped papers
                        continue

                else:  # Otherwise, run without ambig. phrase check
                    curr_res = self.process(
                        text=old_dict["text"],
                        do_check_truematch=do_check_truematch,
                        buffer=buffer,
                        lookup=old_dict["mission"],
                        keyword_obj=None,
                        do_verbose=do_verbose_deep,
                        do_verbose_deep=do_verbose_deep,
                    )

                # Store the modif and previous classification information
                new_dict = {
                    "text": curr_res["modif"],
                    "class": masked_class,  # Mask class
                    "id": old_dict["id"],
                    "mission": old_dict["mission"],
                    "forest": curr_res["forest"],
                    "bibcode": old_dict["bibcode"],
                }
                dict_modifs[curr_key] = new_dict

                # Increment count of modifs generated
                i_track += 1

                # Print some notes at desired frequency
                if do_verbose and ((i_track % print_freq) == 0):
                    print("{0} of {1} total texts have been processed...".format(i_track, num_data))

            # Print some notes
            if do_verbose:
                print("{0} texts have been processed into modifs.".format(i_track))
                if do_check_truematch:
                    print("{0} texts skipped due to unknown ambig. phrases.".format(i_skipped))
                print("Storing the data in train+validate+test directories...")

            # Store the modifs in new TVT directories
            generate_directory_TVT(
                dir_data=dir_data,
                fraction_TVT=fraction_TVT,
                mode_TVT=mode_TVT,
                dict_texts=dict_modifs,
                do_shuffle=do_shuffle,
                seed=seed_TVT,
                do_verbose=do_verbose,
            )
            # Save the modifs with caught processing errors
            np.save(filepath_dicterrors, dict_errors)
            # Print some notes
            if do_verbose:
                print("Train+validate+test directories created in {0}.".format(dir_model))

        # Train a new machine learning (ML) model
        is_exist = os.path.exists(os.path.join(dir_model, savename_model)) or os.path.exists(
            os.path.join(dir_model, savename_ML)
        )
        # If ML model or output already exists, either print note or raise error
        if is_exist:
            # Print some notes
            if do_verbose:
                print("ML model already exists for {0} in {1}.".format(name_model, dir_model))

            # Skip ahead if previous data should be reused
            if do_reuse_run:
                print("Reusing the existing ML model in {0}.".format(dir_model))
                pass

            # Otherwise, raise error if not to reuse previous run data
            else:
                raise ValueError(
                    (
                        "Err: ML model/output already exists"
                        + " in {0}. Either delete it, or rerun method"
                        + " with do_reuse_run=True."
                    ).format(dir_model)
                )

        # Otherwise, train new ML model on the TVT directories
        else:
            # Print some notes
            if do_verbose:
                print("Training new ML model on training data in {0}...".format(dir_model))

            # Train new ML model
            model = classifier.train_ML(
                dir_model=dir_model,
                dir_data=dir_data,
                name_model=name_model,
                seed=seed_ML,
                do_verbose=do_verbose,
                do_return_model=True,
            )

            # Print some notes
            if do_verbose:
                print("New ML model trained and stored in {0}.".format(dir_model))

        # Exit the method with error string
        # Print some notes
        if do_verbose:
            print("Run of train_model_ML() complete!\nError string returned.")

        return str_err
