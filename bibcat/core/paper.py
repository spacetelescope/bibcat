"""
:title: paper.py

The primary purpose is to extract all sentences within a larger text block 
that refers to a given mission(s).  This collection of sentences for each mission, 
denoted from here on as a 'paragraph,' is created to focus on the portions of 
the original text related to the target mission.  Using paragraphs for classification 
instead of the full text allows us to remove much of the 'noise' inherent to 
the rest of the text.

"""
import re

import numpy as np
from core.base import _Base

import bibcat.config as config


class Paper(_Base):
    """
    Class: Paper
    Purpose:
        - Load in text.
        - Split text into sentences containing target terms, if any found.
        - Gather sentences into 'paragraph'.
    Initialization Arguments:
        - dict_ambigs [None or dict (default=None)]:
          - If None, will load and process external database of ambiguous mission phrases.
            If given, will use what is given.
        - do_check_truematch [bool]:
          - Whether or not to check that mission phrases found in text are known true
             vs. false matches. (E.g., 'Edwin Hubble' as false match for the Hubble Space Telescope).
        - keyword_objs [list of Keyword instances]:
          - Target missions; terms will be used to search the text.
        - text [str]:
          - The text to search.
        - do_verbose [bool (default=False)]:
          - Whether or not to print surface-level log information and tests.
        - do_verbose_deep [bool (default=False)]:
          - Whether or not to print inner log information and tests.
    """

    def __init__(
        self, text, keyword_objs, do_check_truematch, dict_ambigs=None, do_verbose=False, do_verbose_deep=False
    ):
        """
        Method: __init__
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Initialize instance of Paper class.
        """

        # Initialize global storage variable
        self._storage = {}  # Dictionary to hold all information

        # Store information about this paper
        text_original = text
        self._store_info(text_original, key="text_original")  # Original text
        self._store_info(keyword_objs, key="keyword_objs")  # Keyword groups
        self._store_info(do_verbose, "do_verbose")
        self._store_info(do_verbose_deep, "do_verbose_deep")
        self._store_info(do_check_truematch, key="do_check_truematch")

        # Process ambig. phrase data, if not given
        if do_check_truematch:
            if dict_ambigs is None:
                dict_ambigs = self._process_database_ambig()

            lookup_ambigs = dict_ambigs["lookup_ambigs"]
            self._store_info(dict_ambigs, key="dict_ambigs")
            self._store_info(lookup_ambigs, key="lookup_ambigs")

        # Preprocess the data
        # Cleanse extra whitespace, strange chars, etc.
        text_clean = self._streamline_phrase(text=text_original)
        # Split cleansed text into naive sentences
        text_clean_split = self._split_text(text=text_clean)

        # Store the preprocessed text
        self._store_info(text_clean, key="text_clean")
        self._store_info(text_clean_split, key="text_clean_split")

        return

    # Purpose: Fetch paragraphs for given Keyword instances that were previously stored in this instance
    def get_paragraphs(self, keyword_objs=None):
        """
        Method: get_paragraphs
        Purpose: Fetch and return paragraphs previously assembled for given Keyword instances.
        Arguments:
          - "keyword_objs" [list of Keyword instances, or None (default=None)]:
             List of Keyword instances for which previously constructed paragraphs will be extracted.
        Returns:
          - dict:
            - keys = Representative names of the Keyword instances.
            - values = The paragraphs corresponding to the Keyword instances.
        """

        # Attempt to access previously extracted paragraphs
        try:
            dict_paragraphs = self._get_info("_paragraphs")
        # Throw an error if no paragraphs extracted yet
        except KeyError:
            errstring = (
                "Whoa there! Looks like you don't have any paragraphs "
                + "stored in this class instance yet. Please run the method "
                + "'process_paragraphs(...)' first."
            )
            raise ValueError(errstring)

        # Extract paragraphs associated with keyword objects, if given
        if keyword_objs is not None:
            paragraphs = {key.get_name(): dict_paragraphs[key.get_name()] for key in keyword_objs}

        # Otherwise, return all paragraphs
        else:
            paragraphs = dict_paragraphs

        return paragraphs

    #

    # Process paragraph that contains given keywords/verified acronyms
    def process_paragraphs(self, buffer=0, do_overwrite=False):
        """
        Method: process_paragraphs
        Purpose: Assemble collection of sentences (a 'paragraph') that contain references
                 to target missions (as indicated by stored keyword objects).
        Arguments:
          - "buffer" [int (default=0)]: Number of +/- sentences around a sentence containing
             a target mission to include in the paragraph.
          - "do_overwrite" [bool (default=False)]: Whether or not to overwrite
             any previously extracted and stored paragraphs.
        Returns: None
        """

        # Extract clean, naively split paragraphs
        text_clean_split = self._get_info("text_clean_split")
        keyword_objs = self._get_info("keyword_objs")
        do_check_truematch = self._get_info("do_check_truematch")

        # If overwrite not allowed, check if paragraphs already extracted+saved
        if not do_overwrite:
            is_exist = True
            # Check for previously stored paragraphs
            try:
                self._get_info("_paragraphs")
            # Catch error raised if no paragraphs exist
            except KeyError:
                is_exist = False

            # Raise error if previously stored paragraphs after all
            if is_exist:
                errstring = (
                    "Whoa there. You already have paragraphs "
                    + "stored in this class instance, and we don't want you "
                    + "to accidentally overwrite them!\nIf you DO want to "
                    + "overwrite previously extracted paragraphs, please "
                    + "rerun this method with do_overwrite=True."
                )
                raise ValueError(errstring)

        # Extract paragraphs for each keyword
        dict_paragraphs = {}  # Dictionary to hold paragraphs for keyword objects
        dict_setup = {"keyword_objs": keyword_objs, "buffer": buffer}  # Parameters
        dict_results_ambig = {item.get_name(): None for item in keyword_objs}  # To hold ambig. output
        dict_acronym_meanings = {item.get_name(): None for item in keyword_objs}  # False acr. meanings
        for ii in range(0, len(keyword_objs)):
            # Extract all paragraphs containing keywords/verified acronyms
            tmp_res = self._extract_paragraph(keyword_obj=keyword_objs[ii], buffer=buffer)
            paragraphs = tmp_res["paragraph"]
            dict_results_ambig[keyword_objs[ii].get_name()] = tmp_res["ambig_matches"]
            dict_acronym_meanings[keyword_objs[ii].get_name()] = tmp_res["acronym_meanings"]

            # Store the paragraphs under name of first given keyword
            dict_paragraphs[keyword_objs[ii].get_name()] = paragraphs

        # Store the extracted paragraphs and setup information
        self._store_info(dict_paragraphs, "_paragraphs")
        self._store_info(dict_setup, "_paragraphs_setup")
        self._store_info(dict_results_ambig, "_results_ambig")
        self._store_info(dict_acronym_meanings, "_dict_acronym_meanings")

        return

    # Apply +/- buffer to given list of indices
    def _buffer_indices(self, indices, buffer, max_index):
        """
        Method: _buffer_indices
        WARNING! This method is *not* meant to be used directly by users.
        Purpose:
          - Add a +/- buffer to each index in a set of indices.
          - Merge buffered indices that have overlapping buffers.
        """

        # Build spans for extent of each buffered index
        spans_buffered_init = [[(ind - buffer), (ind + buffer)] for ind in indices]

        # Merge overlapping spans and truncate at min/max bounds
        spans_buffered_merged = []
        for ii in range(0, len(spans_buffered_init)):
            # First span starts at max([0, first_ind])
            if ii == 0:
                spans_buffered_merged.append([max([0, spans_buffered_init[0][0]]), spans_buffered_init[0][1]])
                # Terminate loop early if latest span reaches maximum boundary
                if spans_buffered_merged[-1][1] >= max_index:
                    break

                # Otherwise, skip ahead
                continue

            # For all other spans:
            # If overlap, expand previous span
            if spans_buffered_merged[-1][1] >= spans_buffered_init[ii][0]:
                spans_buffered_merged[-1][1] = spans_buffered_init[ii][1]

            # Otherwise, append new span
            else:
                spans_buffered_merged.append(
                    [spans_buffered_init[ii][0], min([spans_buffered_init[ii][1], max_index])]
                )  # Written out to avoid shallow copy

            # Terminate loop early if latest span reaches maximum boundary
            if spans_buffered_merged[-1][1] >= max_index:
                break

        # Return the buffered index spans
        return spans_buffered_merged

    # Search for paragraphs that contain target mission terms
    def _extract_paragraph(self, keyword_obj, buffer):
        """
        Method: _extract_paragraph
        WARNING! This method is *not* meant to be used directly by users.
        Purpose:
          - Extract sentences from stored text that contain target terms (based on Keyword instance).
          - Keep sentences that do not have false-matches to external ambiguous phrase database.
          - Buffer sentences if non-zero buffer given.
        """
        # Fetch global variables
        do_verbose = self._get_info("do_verbose")
        do_verbose_deep = self._get_info("do_verbose_deep")
        do_check_truematch = self._get_info("do_check_truematch")
        sentences = np.asarray(self._get_info("text_clean_split"))
        num_sentences = len(sentences)

        # Load ambiguous phrases, if necessary
        if do_check_truematch:
            # Print some notes
            if do_verbose:
                print("do_check_truematch=True, so will verify ambig. phrases.")

            # Load previously stored ambig. phrase data
            dict_ambigs = self._get_info("dict_ambigs")
            lookup_ambigs = dict_ambigs["lookup_ambigs"]

        # Print some notes
        if do_verbose:
            print("Fetching inds of target sentences...")

        # Get indices of sentences that contain any target mission terms
        # For keyword terms
        inds_with_keywords_init = [
            ii for ii in range(0, num_sentences) if keyword_obj.is_keyword(sentences[ii], mode="keyword")
        ]

        # For acronym terms
        inds_with_acronyms = [
            ii for ii in range(0, num_sentences) if keyword_obj.is_keyword(sentences[ii], mode="acronym")
        ]

        # Print some notes
        if do_verbose:
            print(
                ("Found:\n# of keyword sentences: {0}\n" + "# of acronym sentences: {1}...").format(
                    len(inds_with_keywords_init), len(inds_with_acronyms)
                )
            )

        # If only acronym terms found, run a check of possible false meanings
        if len(inds_with_acronyms) > 0:
            acronym_meanings = self._verify_acronyms(keyword_obj=keyword_obj)
        # Otherwise, set empty
        else:
            acronym_meanings = None
        #

        # If requested, run a check for ambiguous phrases if any ambig. keywords
        if do_check_truematch and any([keyword_obj.is_keyword(item) for item in lookup_ambigs]):
            # Print some notes
            if do_verbose:
                print("Verifying ambiguous phrases...")

            # Run ambiguous phrase check on all sentences with keyword terms
            output_truematch = [
                {
                    "ind": ind,
                    "result": self._check_truematch(
                        text=sentences[ind], keyword_objs=[keyword_obj], dict_ambigs=dict_ambigs
                    ),
                }
                for ind in inds_with_keywords_init
            ]

            ambig_matches = [item["result"] for item in output_truematch]

            # Keep indices that have true matches
            inds_with_keywords_truematch = [item["ind"] for item in output_truematch if (item["result"]["bool"])]

            # Print some notes
            if do_verbose:
                print("Done verifying ambiguous phrases.")
                print("Match output:\n{0}".format(output_truematch))
                print("Indices with true matches:\n{0}".format(inds_with_keywords_truematch))
                print("Keyword sentences with true matches:\n{0}".format(sentences[inds_with_keywords_truematch]))

        # Otherwise, set empty
        else:
            output_truematch = None
            ambig_matches = None
            inds_with_keywords_truematch = inds_with_keywords_init  # Copy over

        # Take note of if keywords and/or acronyms have matches
        dict_has_matches = {
            "keywords": (len(inds_with_keywords_truematch) > 0),
            "acronyms": (len(inds_with_acronyms) > 0),
        }

        # Pool together unique indices of sentences with keywords, acronyms
        inds_with_terms = list(set().union(inds_with_keywords_truematch, inds_with_acronyms))

        # Determine buffered sentences, if requested
        if buffer > 0:
            # Print some notes
            if do_verbose:
                print("Buffering sentences with buffer={0}...".format(buffer))
            #
            ranges_buffered = self._buffer_indices(
                indices=inds_with_terms, buffer=buffer, max_index=(num_sentences - 1)
            )
            sentences_buffered = [
                " ".join(sentences[item[0] : item[1] + 1]) for item in ranges_buffered
            ]  # Combined sentences
            # Print some notes
            if do_verbose:
                print("Done buffering sentences.\nRanges = {0}.".format(ranges_buffered))

        # Otherwise, just copy over previous indices
        else:
            ranges_buffered = None
            sentences_buffered = sentences[inds_with_terms].tolist()

        # Return outputs
        return {
            "paragraph": sentences_buffered,
            "ambig_matches": ambig_matches,
            "acronym_meanings": acronym_meanings,
            "has_matches": dict_has_matches,
        }

    # Split text into sentences at assumed sentence breaks
    def _split_text(self, text):
        """
        Method: _split_text
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Split given text into sentences based on assumed sentence boundaries.
        """
        # Split by line breaks first
        text_lines = text.split("\n")
        # Split by sentences starting with brackets
        text_flat = [item for phrase in text_lines for item in re.split(config.exp_splitbracketstarts, phrase)]
        # Split by sentences ending with brackets
        text_flat = [item for phrase in text_flat for item in re.split(config.exp_splitbracketends, phrase)]
        # Then split by assumed sentence structure
        text_flat = [item for phrase in text_flat for item in re.split(config.exp_splittext, phrase)]
        # Return the split text
        return text_flat

    #

    # Find possible meanings of acronyms in text
    def _verify_acronyms(self, keyword_obj):
        """
        Method: _verify_acronyms
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Extract all possible matches from the stored text to the acronyms of the Keyword instance.
        """
        # Fetch global variables
        acronyms = [item.upper() for item in keyword_obj._get_info("acronyms")]
        text = self._get_info("text_original")

        # Build regular expression for all acronyms
        list_exp = [
            (r"\b" + (r"[a-z]+\b" + config.exp_acronym_midwords).join(letterset) + r"[a-z]+\b")
            for letterset in acronyms
        ]
        combined_exp = r"(?:" + (r")|(?:".join(list_exp)) + r")"

        # Search full text for possible acronym meanings
        matches = re.findall(combined_exp, text)
        # Throw error if any tuple entries found (e.g. loophole in regex)
        if any([(isinstance(item, tuple)) for item in matches]):
            raise ValueError("Err: Regex must have holes!\n{0}\n{1}".format(combined_exp, matches))

        # Return all determined matches
        return matches
        return matches
        return matches
