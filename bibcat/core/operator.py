"""
:title: operator.py

The primary purpose of this class is to direct and run the entire workflow of 'bibcat',
from reading in a given block of text to ultimately classifying that block of text.
Using its `classify` method, the Operator class internally handles all calls to
the other classes (Paper, Grammar, and the given classifier).

The primary methods and use cases of Operator are:
* `classify`: A method designed for users that prepares and runs the entire 'bibcat' workflow,
   from input raw text to classified output.
* `process`: A method designed for users that processes given text into modifs,
   from input raw text to output modifs. It does not include classification (for that, run `classify`);
   it is useful for preprocessing raw text.
"""

import os
from typing import Any

import numpy as np

from bibcat import config
from bibcat.core.base import Base
from bibcat.core.grammar import Grammar
from bibcat.core.keyword import Keyword
from bibcat.core.paper import Paper
from bibcat.data.partition_dataset import generate_directory_TVT
from bibcat.utils.logger_config import setup_logger

logger = setup_logger(__name__)


# TODO - this class is too complicated and may be unneccesary; break down into smaller components and
# TODO - move to other relevant classes like Paper, Grammar, Keyword, etc.
class Operator(Base):
    """Operator class for running a bibcat workflow

    Class for running the complete workflow of text classification, from input text to internal text processing,
    to output classification.  The ``classsify`` method is the primary method that handles the actual classification.
    ``classify_set`` is a wrapper method to classify a set of texts against all mission keywords.

    Parameters
    ----------
    classifier : object
        the ML classifier instance to use
    mode : str
        the mode of text modification to use when processing text using the Grammar class
    keyword_objs : list
        the target mission keywords
    verbose : bool, optional
        Flag to turn on verbosity, by default False
    name : str, optional
        A unique name for the operator, by default "operator"
    load_check_truematch : bool, optional
        Flag to check that mission phrases found in text are known true vs. false matches, by default True
    deep_verbose : bool, optional
        Flag to turn on deep verbosity, by default False

    Raises
    ------
    ValueError
        when the reserved character '|' is used in the name for the Operator
    """

    # Initialize this class instance
    def __init__(
        self,
        classifier,
        mode: str,
        keyword_objs: list,
        verbose: bool = False,
        name: str = "operator",
        load_check_truematch: bool = True,
        deep_verbose: bool = False,
    ):
        """Initialize the Operator class"""
        # object attributes
        self.name = name
        self.classifier = classifier
        self.mode = mode
        self.verbose = verbose
        self.deep_verbose = deep_verbose
        self.load_check_truematch = load_check_truematch

        # Throw an error if reserved character '|' in name
        if "|" in name:
            raise ValueError("Please do not use the reserved character '|' in the name for your Operator.")

        # keyword object info
        self.keyword_objs = keyword_objs
        self.num_keyobjs = len(keyword_objs)

        # ambiguous keyword data
        self.dict_ambigs = None
        self.lookup_ambigs = None

        # Load and process ambiguous (ambig.) data, if so requested
        if load_check_truematch:
            # Run method to load and process external ambig. database
            paper = Paper(text="", keyword_objs=keyword_objs, do_check_truematch=True)
            self.dict_ambigs = paper._process_database_ambig(keyword_objs=keyword_objs)
            self.lookup_ambigs = self.dict_ambigs["lookup_ambigs"]

            # Print some notes
            if self.deep_verbose:
                logger.info("Loaded+Assembled data for ambiguous phrases.")

        # print the keyword objects
        if self.verbose:
            logger.info("Instance of Operator successfully initialized!")
            logger.info("Keyword objects:")
            for kobj in self.keyword_objs:
                logger.info(f"{kobj}")

    # Inspect text and either reject as false target or give classifications
    def classify(
        self,
        text: str | None,
        keyword: str,
        modif: str | None = None,
        do_check_truematch: bool = False,
        buffer: int = 0,
    ) -> dict[str, Any] | Any:
        """Classify a text

        Classify a text against a target mission keyword as "science", "mention", or "data_influencded".
        First converts the input text into a modif, or uses the input modif if given.  Then runs the
        ml classifier's classify_text method to predict the category probabilities.  The output is a dictionary
        with the original modif, verdict category, scores_comb (the final score), scores_indiv (the individual scores),
        and uncertainty (the prediction probabilities).

        Parameters
        ----------
        text : str | None
            the text to classify
        keyword : str
            the target mission keyword
        modif : str | None, optional
            the modif to classify, by default None
        do_check_truematch : bool, optional
            Flag to check that mission phrases found in text are known true vs. false matches, by default False
        buffer : int, optional
            Number of +/- sentences around a sentence containing a target mission to include in the paragraph, by default 0

        Returns
        -------
        dict[str, Any] | Any
            the classification results
            - modif (str): the modified text
            - modif_none (str): the unmodified text
            - verdict (str): the classification.
            - scores_comb (str|Any): the final score.
            - scores_indiv (str|Any): the individual scores.
            - uncertainty (dict[str, float]|None): the uncertainty of the classifications, e.g., "science", "mention"

        """

        modif_none = None

        # Process text into modifs using Grammar class, if modif not given
        if modif is None:
            if self.verbose:
                logger.info("\nPreprocessing and extracting modifs from the text...")

            try:
                output = self.process(text, keyword_obj=keyword, do_check_truematch=do_check_truematch, buffer=buffer)
            except Exception as err:
                verdicts = config.results.dictverdict_error.copy()
                logger.error("-\nThe following err. was encountered in operate:")
                logger.error(err)
                logger.error("Error was noted. Returning error as verdict.\n-")
                verdicts["modif"] = f"<PROCESSING ERROR:\n{err}>"
                verdicts["modif_none"] = None
                return verdicts

            # Fetch the generated output
            modif = output["modif"]
            modif_none = output["modif_none"]

            # Print some notes
            if self.verbose:
                logger.info("Text has been processed into modif.")

        # Set rejected verdict if empty text
        if modif.strip() == "":
            if self.verbose:
                logger.info("No text found matching keyword object.")
                logger.info("Returning rejection verdict.")

            verdicts = config.results.dictverdict_rejection.copy()
        # Set not-classified verdict if flagged for no classification
        elif keyword._get_info("do_not_classify"):
            verdicts = config.results.dictverdict_donotclassify.copy()
        else:
            try:
                verdicts = self.classifier.classify_text(text=modif)
            except Exception as err:
                verdicts = config.results.dictverdict_error.copy()
                logger.error("-\nThe following err. was encountered in operate:")
                logger.error(err)
                logger.error("Error was noted. Continuing.\n-")

        # Return the verdict with modif included
        verdicts["modif"] = modif
        verdicts["modif_none"] = modif_none
        return verdicts

    # Classify set of texts as false target or give classifications
    def classify_set(
        self,
        texts: list[str] | None,
        modifs: list[str] | None = None,
        do_check_truematch: bool = False,
        buffer: int = 0,
        print_freq: int = 25,
    ) -> list[dict[str, Any] | Any]:
        """Classify a set of texts

        Classify a list of texts against the list of all mission keywords.

        Parameters
        ----------
        texts : list[str] | None
            a list of texts to classify
        modifs : list[str] | None, optional
            a list of modifs to classify, by default None
        do_check_truematch : bool, optional
            Flag to check that mission phrases found in text are known true vs. false matches, by default False
        buffer : int, optional
            Number of +/- sentences around a sentence containing a target mission to include in the paragraph, by default 0
        print_freq : int, optional
            The frequency to print updates, by default 25

        Returns
        -------
        list[dict[str, Any] | Any]
            the output classification results for each text

        Raises
        ------
        ValueError
            when both texts and modifs are given
        """

        # Throw error if both texts and modifs given
        if (texts is not None) and (modifs is not None):
            raise ValueError("Err: texts OR modifs should be given, not both.")

        # get the number of texts
        if texts is not None:
            num_texts = len(texts)
        elif modifs is not None:
            num_texts = len(modifs)

        # Print some notes
        if self.verbose:
            logger.info("\n> Running classify_set()!")

        # Classify every text against every mission
        results = [{}] * num_texts

        # Iterate through texts
        for ii, text in enumerate(texts):
            item = {}  # Dictionary to hold set of results
            results[ii] = item  # Store this dictionary

            # Extract current modifs if already processed text
            modif = modifs[ii] if modifs else None

            # Iterate through keyword objects
            for kobj in self.keyword_objs:
                name = kobj._get_info("name")
                # Classify current text for current mission
                result = self.classify(
                    text=text, keyword=kobj, modif=modif, do_check_truematch=do_check_truematch, buffer=buffer
                )

                # Store current result
                item[name] = result

            # Print some notes at given frequency, if requested
            if self.verbose and (((ii % print_freq) == 0) or (ii == (num_texts - 1))):
                logger.info(f"Classification for text #{(ii + 1)} of {num_texts} complete...")

        # Return the classification results
        if self.verbose:
            logger.info("\nRun of classify_set() complete!\n")

        return results

    # Process text into modifs
    def process(
        self,
        text: str,
        lookup: str = None,
        keyword_obj: Keyword = None,
        do_check_truematch: bool = False,
        buffer: int = 0,
    ) -> dict:
        """Process text into modifs

        Processes the text using the Grammar and Paper classes into modifs and forest.  A "modif" is a modified version of the text
        that has been processed to identify and remove references to the target keyword mission, i.e. ambiguates the text. A "modif_none" is a unmodified version of the text. "forest" contains a dictionary, which stores some determined 'grammar' data about a given text. For example, "forest" stores verbs that are identified in a sentence, the identified clauses of a sentence, the assigned noun chunks of a sentence, and so on. The “forest” dictionary is optionally used for some of the optional text trimming methods of bibcat - for example, if the user wants to remove adjectives from a sentence, which are often not relevant for classification. Ultimately, the goal was to reduce how much "noise" (e.g., irrelevant text) was fed into one of the trained classifiers for classification.

        Parameters
        ----------
        text : str
            the text to classify
        lookup : str, optional
            a term for looking up the target Keyword instance (e.g. HST), by default None
        keyword_obj : Keyword, optional
            a target Keyword instance, by default None
        do_check_truematch : bool, optional
            Flag to check that mission phrases found in text are known true vs. false matches, by default False
        buffer : int, optional
            Number of +/- sentences around a sentence containing a target mission to include in the paragraph, by default 0

        Returns
        -------
        dict: the output modif and forest (internal text processing output)
            A dictionary containing the following keys:
            - modif (str): the modified paragraphs
            - modif_none (str): the unmodifed paragraphs
            - forest (str): a dictionary, which stores some determined ‘grammar’ data about a given text.



        """

        if self.verbose:
            logger.info("\nRunning Grammar on the text...")

        # Fetch keyword object matching to the given keyword
        if keyword_obj is None:
            keyword_obj = Keyword._fetch_keyword_object(self.keyword_objs, lookup=lookup, verbose=self.verbose)
            if self.verbose:
                logger.info(f"Best matching Keyword object for keyword {lookup}:\n{keyword_obj}")

        # Process text into modifs using Grammar class
        use_these_modes = [self.mode, "none"]
        grammar = Grammar(
            text=text,
            keyword_obj=keyword_obj,
            do_check_truematch=do_check_truematch,
            dict_ambigs=self.dict_ambigs,
            do_verbose=self.deep_verbose,
            buffer=buffer,
        )
        grammar.run_modifications(which_modes=use_these_modes)
        output = grammar.get_modifs()

        # update outputs
        modif = output["modifs"][self.mode]
        modif_none = output["modifs"]["none"]  # Include unmodified vers. as well
        forest = output["_forest"]

        # Print some notes
        if self.verbose:
            logger.info("Text has been processed into modifs.")

        # Return the modif and internal processing output
        return {"modif": modif, "modif_none": modif_none, "forest": forest}
