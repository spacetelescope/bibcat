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
from dataclasses import dataclass
from typing import Any

import numpy as np
import spacy
from nltk.corpus import wordnet  # type: ignore

from bibcat import config
from bibcat.core.base import Base

nlp = spacy.load(config.grammar.spacy_language_model)


class Paper(Base):
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
        text_clean = self._streamline_phrase(text=text_original, do_streamline_etal=False)
        # Split cleansed text into naive sentences
        text_clean_split = self._split_text(text=text_clean)

        # Store the preprocessed text
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
        keyword_objs = self._get_info("keyword_objs")

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
        for ii in range(0, len(keyword_objs)):
            # Extract all paragraphs containing keywords/verified acronyms
            tmp_res = self._extract_paragraph(keyword_obj=keyword_objs[ii], buffer=buffer)
            paragraphs = tmp_res["paragraph"]

            # Store the paragraphs under name of first given keyword
            dict_paragraphs[keyword_objs[ii].get_name()] = paragraphs

        # Store the extracted paragraphs and setup information
        self._store_info(dict_paragraphs, "_paragraphs")

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

    # Return boolean for whether or not text contains a true vs false match to the given keywords
    def _check_truematch(self, text, keyword_objs, dict_ambigs, do_verbose=None, do_verbose_deep=False):  # noqa: C901
        """
        Method: _check_truematch
        WARNING! This method is *not* meant to be used directly by users.
        Purpose:
          - Determine if given text contains a true vs. false match to keywords.
            - E.g.: 'Edwin Hubble' as a false match to Hubble Space Telescope.
        Arguments:
        - text [str]:
          - The text to search.
        - keyword_objs [list of Keyword instances]:
          - Target missions; terms will be used to search the text.
        - dict_ambigs [None or dict (default=None)]:
          - If None, will load and process external database of ambiguous mission phrases.
            If given, will use what is given.
        - do_verbose [bool (default=False)]:
          - Whether or not to print surface-level log information and tests.
        - do_verbose_deep [bool (default=False)]:
          - Whether or not to print inner log information and tests.
        Returns:
          - A dictionary with the following parameters
            - bool: boolean for whether or not text contains a true vs false match to the given keywords
            - info (if no keywords found or non-ambiguous match): an array with one item
               - bool: same value as bool above
               - text_wordchunk: status of search
               - text_database: None
               - matcher: None
               - set: None if non-ambiguous match, otherwise omitted
            - info (if going through ambiguous phrases): an array with multiple items, one for each phrase
               - bool: result of that particular phrase. If any phrase matches true, the "bool" above is true
               - text_wordchunk: the chunk of text searched
               - text_database: the phrase searched within the chunk of text
               - matcher: the result of regex search
               - ind: index number within the database
               - exp: ambiguous phrases searched

        """

        # Set up initial variables
        setup_data = self._setup_check_truematch_vars(text, dict_ambigs, keyword_objs, do_verbose, do_verbose_deep)

        # Short-circuit checks
        for check in (
            self._early_true_non_ambig_keywords,
            self._early_false_no_keyword_match,
            self._early_true_acronym_match,
            self._early_true_non_ambig_phrases,
        ):
            result = check(setup_data)
            if result is not None:
                return result

        # Assemble makeshift wordchunks (not using NLP ones here)
        # Not sure why happened, but NLP sometimes failed to identify nouns/num.
        list_wordchunks = self._assemble_keyword_wordchunks_wrapper(setup_data)

        # Short-circuit check for exact wordchunks
        result = self._early_true_exact_wordchunk(list_wordchunks, setup_data)
        if result is not None:
            return result

        # Iterate through wordchunks to determine true vs false match status
        list_results = [self._consider_wordchunk(curr_chunk, setup_data) for curr_chunk in list_wordchunks]

        # Combine the results and return overall boolean match
        fin_result = {
            "bool": any([(item["bool"]) for item in list_results]),
            "info": [item["info"][0] for item in list_results],
        }
        return fin_result

    @dataclass
    class TruematchSetup:
        text: Any
        dict_ambigs: Any
        keyword_objs: Any
        do_verbose: Any
        do_verbose_deep: Any
        list_kw_ambigs: Any
        list_exp_exact_ambigs: Any
        list_exp_meaning_ambigs: Any
        list_bool_ambigs: Any
        list_text_ambigs: Any
        lookup_ambigs: Any
        lookup_ambigs_lower: Any
        num_ambigs: Any
        text: Any
        keyword_objs_ambigs: Any
        dict_kobjinfo: Any

        def log_if_verbose(self, str):
            if self.do_verbose:
                print(str)

    def _build_single_info_entry(self, **kwargs):
        return {
            "bool": kwargs.bool,
            "info": [kwargs],
        }

    def _setup_check_truematch_vars(self, text, dict_ambigs, keyword_objs, do_verbose=None, do_verbose_deep=None):
        # Load global variables
        if do_verbose is None:
            do_verbose = self._get_info("do_verbose", do_flag_hidden=True)

        # Process ambig. phrase data, if not given
        if dict_ambigs is None:
            dict_ambigs = self._process_database_ambig(do_verbose=do_verbose_deep, keyword_objs=keyword_objs)

        # Extract info from ambig. database
        list_kw_ambigs = dict_ambigs["all_kw_ambigs"]
        list_exp_exact_ambigs = dict_ambigs["all_exp_exact_ambigs"]
        list_exp_meaning_ambigs = dict_ambigs["all_exp_meaning_ambigs"]
        list_bool_ambigs = dict_ambigs["all_bool_ambigs"]
        list_text_ambigs = dict_ambigs["all_text_ambigs"]
        lookup_ambigs = dict_ambigs["lookup_ambigs"]
        lookup_ambigs_lower = [item.lower() for item in dict_ambigs["lookup_ambigs"]]
        num_ambigs = len(list_kw_ambigs)

        # Replace hyphenated numerics with placeholders
        text_orig = text
        placeholder_number = config.textprocessing.placeholder_number
        text = re.sub(r"-\b[0-9]+\b", ("-" + placeholder_number), text_orig)

        # Print some notes
        if do_verbose:
            print(
                ("\n> Running _check_truematch for text: '{0}'" + "\nOriginal text: {1}\nLookups: {2}").format(
                    text, text_orig, lookup_ambigs
                )
            )

        # Extract keyword objects that are potentially ambiguous
        keyword_objs_ambigs = [
            item1 for item1 in keyword_objs if any([item1.identify_keyword(item2)["bool"] for item2 in lookup_ambigs])
        ]

        # Extract keyword identification information for each kobj
        dict_kobjinfo = {item._get_info("name"): item.identify_keyword(text) for item in keyword_objs}

        return self.TruematchSetup(
            text=text,
            dict_ambigs=dict_ambigs,
            keyword_objs=keyword_objs,
            do_verbose=do_verbose,
            do_verbose_deep=do_verbose_deep,
            list_kw_ambigs=list_kw_ambigs,
            list_exp_exact_ambigs=list_exp_exact_ambigs,
            list_exp_meaning_ambigs=list_exp_meaning_ambigs,
            list_bool_ambigs=list_bool_ambigs,
            list_text_ambigs=list_text_ambigs,
            lookup_ambigs=lookup_ambigs,
            lookup_ambigs_lower=lookup_ambigs_lower,
            num_ambigs=num_ambigs,
            keyword_objs_ambigs=keyword_objs_ambigs,
            dict_kobjinfo=dict_kobjinfo,
        )

    def _early_true_non_ambig_keywords(self, setup_data):
        """
        Return status as true match if non-ambig keywords match to text
        """
        keyword_objs_non_ambigs = [
            item1 for item1 in setup_data.keyword_objs if item1 not in setup_data.keyword_objs_ambigs
        ]
        if any([setup_data.dict_kobjinfo[item1._get_info("name")]["bool"] for item1 in keyword_objs_non_ambigs]):
            # Print some notes
            setup_data.log_if_verbose("Text matches unambiguous keyword. Returning true state.")

            # Return status as true match
            return self._build_single_info_entry(
                matcher=None, set=None, bool=True, text_wordchunk="<Not ambig.>", text_database=None
            )

    def _early_false_no_keyword_match(self, setup_data):
        """
        Return status as false match if no keywords match at all
        """
        if not any(
            [setup_data.dict_kobjinfo[item._get_info("name")]["bool"] for item in setup_data.keyword_objs_ambigs]
        ):
            # Print some notes
            setup_data.log_if_verbose("Text matches no keywords at all. Returning false state.")

            # Return status as true match
            return self._build_single_info_entry(
                matcher=None, text_database=None, bool=False, text_wordchunk="<No matching keywords at all.>"
            )

    def _early_true_acronym_match(self, setup_data):
        """
        Return status as true match if any acronyms match
        """
        if any(
            [
                setup_data.dict_kobjinfo[item._get_info("name")]["bool_acronym_only"]
                for item in setup_data.keyword_objs_ambigs
            ]
        ):
            # Print some notes
            setup_data.log_if_verbose("Text matches acronym. Returning true state.")

            # Return status as true match
            return self._build_single_info_entry(
                matcher=None, set=None, bool=True, text_wordchunk="<Not ambig.>", text_database=None
            )

    def _early_true_non_ambig_phrases(self, setup_data):
        # Return status as true match if any non-ambig. phrases match to text
        for obj in setup_data.keyword_objs_ambigs:
            for kw in obj._get_info("keywords"):
                if kw in obj._get_info("ambig_words"):
                    continue
                if re.search(rf"\b{re.escape(kw)}\b", setup_data.text, flags=re.IGNORECASE):
                    # Print some notes
                    setup_data.log_if_verbose("Text matches unambiguous keyword. Returning true state.")

                    # Return status as true match
                    return self._build_single_info_entry(
                        matcher=None, set=None, bool=True, text_wordchunk="<Not ambig.>", text_database=None
                    )

    def _assemble_keyword_wordchunks_wrapper(self, setup_data):
        """
        A wrapper for _assemble_keyword_wordchunks with verbose logging and error handling
        """
        # Print some notes
        setup_data.log_if_verbose("Building noun chunks around keywords...")

        # Generate the keyword-based wordchunks
        list_wordchunks = self._assemble_keyword_wordchunks(
            text=setup_data.text,
            keyword_objs=setup_data.keyword_objs,
            do_verbose=setup_data.do_verbose,
            do_include_verbs=False,
        )
        # Throw error if no wordchunks identified
        if len(list_wordchunks) == 0:
            errstr = (
                "No final wordchunks!: {0}\nText: '{1}'".format(list_wordchunks, setup_data.text)
                + "\nAll words and p.o.s.:\n"
            )
            tmp_sents = list(nlp(str(setup_data.text)).sents)
            for aa in range(0, len(tmp_sents)):
                for bb in range(0, len(tmp_sents[aa])):
                    tmp_word = tmp_sents[aa][bb]
                    errstr += "{0}: {1}, {2}, {3}\n".format(tmp_word, tmp_word.dep_, tmp_word.pos_, tmp_word.tag_)

            raise ValueError(errstr)

        # Print some notes
        setup_data.log_if_verbose("\n- Wordchunks determined for text: {0}".format(list_wordchunks))

        return list_wordchunks

    def _early_true_exact_wordchunk(self, list_wordchunks, setup_data):
        # Exit method early if any wordchunk is an exact keyword match
        if any([(item.text.lower() in setup_data.lookup_ambigs) for item in list_wordchunks]):
            # Print some notes
            setup_data.log_if_verbose("Exact keyword match found. Returning true status...")

            return self._build_single_info_entry(
                matcher=None, text_database=None, bool=True, text_wordchunk="<Wordchunk has exact term match.>"
            )

    def _consider_wordchunk(self, curr_chunk, setup_data):
        curr_chunk_text = curr_chunk.text
        # Print some notes
        setup_data.log_if_verbose("Considering wordchunk: {0}".format(curr_chunk_text))

        # Short-circuit check
        result = self._early_true_non_ambig_term(curr_chunk_text, setup_data)
        if result is not None:
            return result

        # Setup variables
        curr_meaning, curr_inner_kw = self._setup_consider_wordchunk(curr_chunk, setup_data)

        # Extract all ambig. phrases+substrings that match to this meaning
        set_matches = self._extract_ambig_phrases_substrings(
            setup_data.list_exp_exact_ambigs, "matches", curr_chunk_text, curr_meaning, curr_inner_kw, setup_data
        ) or self._extract_ambig_phrases_substrings(
            setup_data.list_exp_meaning_ambigs, "meanings", curr_chunk_text, curr_meaning, curr_inner_kw, setup_data
        )

        # Throw error if no match found
        if len(set_matches) == 0:
            # Raise a unique for-user error (using NotImplementedError)
            # Allows this exception to be uniquely caught elsewhere in code
            # Use-case isn't exactly what NotImplemented means, but that's ok
            # RuntimeError could also work but seems more for general use
            raise NotImplementedError(
                ("Err: Unrecognized ambig. phrase:\n{0}" + "\nTaken from this text snippet:\n{1}").format(
                    curr_chunk, setup_data.text
                )
            )

        # Determine and extract best match (=match with shortest substring)
        list_results = self._assemble_consider_wordchunk_results(set_matches, curr_chunk, curr_meaning, setup_data)

        return list_results

    def _early_true_non_ambig_term(self, curr_chunk_text, setup_data):
        # Store as non-ambig. phrase and skip ahead if non-ambig. term
        is_exact = any(
            [
                (curr_chunk_text.lower().replace(".", "") == item2.lower())
                for item1 in setup_data.keyword_objs
                for item2 in (
                    item1._get_info("keywords")
                    + item1._get_info("acronyms_casesensitive")
                    + item1._get_info("acronyms_caseinsensitive")
                )
                if (item2.lower() not in setup_data.lookup_ambigs_lower)
            ]
        )  # Check if wordchunk matches to any non-ambig terms
        if is_exact:
            # Print some notes
            setup_data.log_if_verbose("Exact match to non-ambig. phrase. Marking true...")

            # Store info for this true match
            list_results = self._build_single_info_entry(
                matcher=None, bool=True, text_wordchunk=curr_chunk_text, text_database=None
            )
            # Skip ahead
            return list_results

    def _setup_consider_wordchunk(self, curr_chunk, keyword_objs_ambigs, setup_data):
        # Extract representation of core meaning of current wordchunk
        tmp_res = self._extract_core_from_phrase(
            phrase_NLP=curr_chunk,
            keyword_objs=keyword_objs_ambigs,
            do_skip_useless=False,
            do_verbose=setup_data.do_verbose,
        )
        curr_meaning = tmp_res["str_meaning"]  # Str representation of meaning
        curr_inner_kw = tmp_res["keywords"]  # Matched keywords

        return curr_meaning, curr_inner_kw

    def _extract_ambig_phrases_substrings(
        self, exp_list, label, curr_chunk_text, curr_meaning, curr_inner_kw, setup_data
    ):
        set_matches_raw = [
            {
                "ind": jj,
                "text_database": setup_data.list_text_ambigs[jj],
                "text_wordchunk": curr_chunk_text,
                "exp": exp_list[jj],
                "matcher": re.search(exp_list[jj], curr_meaning, flags=re.IGNORECASE),
                "bool": setup_data.list_bool_ambigs[jj],
            }
            for jj in range(0, setup_data.num_ambigs)
            if (setup_data.list_kw_ambigs[jj] in curr_inner_kw)
        ]
        set_matches = [item for item in set_matches_raw if (item["matcher"] is not None)]

        # Print some notes
        if setup_data.do_verbose_deep:
            print(f"Set of {label} assembled from ambig. database:")
            for item1 in set_matches_raw:
                print(item1)

        return set_matches

    def _assemble_consider_wordchunk_results(self, set_matches, curr_chunk, curr_meaning, setup_data):
        best_set = sorted(set_matches, key=(lambda w: len(w["matcher"][0])))[0]

        # Print some notes
        setup_data.log_if_verbose(
            "Current wordchunk: {0}\nMeaning: {2}\nBest set: {1}-".format(curr_chunk, best_set, curr_meaning)
        )

        # Exit method early since match found
        setup_data.log_if_verbose("Match found. Returning status...")

        list_results = self._build_single_info_entry(
            matcher=best_set["matcher"],
            bool=best_set["bool"],
            text_wordchunk=best_set["text_wordchunk"],
            text_database=best_set["text_database"],
        )

        return list_results

    # Extract core meaning (e.g., synsets) from given phrase
    def _extract_core_from_phrase(self, phrase_NLP, do_skip_useless, do_verbose=None, keyword_objs=None):  # noqa: C901
        """
        Method: _extract_core_from_phrase
        WARNING! This method is *not* meant to be used directly by users.
        Purpose:
         - Extract representative "meaning" (i.e., synsets) of given phrase.
        """

        # Set global variables
        num_words = len(phrase_NLP)
        if do_verbose is None:
            do_verbose = self._get_info("do_verbose")
        if keyword_objs is None:
            try:
                keyword_objs = [self._get_info("keyword_obj", do_flag_hidden=True)]
            except KeyError:
                keyword_objs = self._get_info("keyword_objs", do_flag_hidden=True)

        # Print some notes
        if do_verbose:
            print("\n> Running _extract_core_from_phrase for phrase: {0}".format(phrase_NLP))

        # Initialize containers of core information
        core_keywords = []
        core_synsets = []

        # Iterate through words within phrase
        for ii in range(0, num_words):
            curr_word = phrase_NLP[ii]
            # Print some notes
            if do_verbose:
                print("-\nLatest keywords: {0}".format(core_keywords))
                print("Latest synsets: {0}".format(core_synsets))
                print("-Now considering word: {0}".format(curr_word))

            # Skip if this word is punctuation or possessive marker
            if self._is_pos_word(word=curr_word, pos="PUNCTUATION") or self._is_pos_word(
                word=curr_word, pos="POSSESSIVE"
            ):
                # Print some notes
                if do_verbose:
                    print("Word is punctuation or possessive. Skipping.")

                continue

            # Store the keyword itself and skip ahead if this word is a keyword
            matched_kobjs = [item for item in keyword_objs if (item.identify_keyword(curr_word.text)["bool"])]
            if len(matched_kobjs) > 0:  # If word is a keyword
                # If word contains hyphen-esque|punct., keep whole word as synset
                if bool(re.search(r"(?:[^\w\s]|_)", curr_word.text)):
                    name_kobj = matched_kobjs[0].get_name()  # Fetch name for kobj
                    core_keywords.append(name_kobj.lower())
                    core_synsets.append([curr_word.text.lower()])

                    # Print some notes
                    if do_verbose:
                        print("Word itself is keyword. Stored synset: {0}".format(core_synsets))

                    continue
                # Otherwise, store keyword itself
                else:
                    name_kobj = matched_kobjs[0].get_name()  # Fetch name for kobj
                    core_keywords.append(name_kobj.lower())
                    core_synsets.append([name_kobj.lower()])

                    # Print some notes
                    if do_verbose:
                        print("Word itself is keyword. Stored synset: {0}".format(core_synsets))

                    continue

            # Store a representative synset and skip ahead if word is a numeral
            if bool(re.search(("^(ID)?[0-9]+"), curr_word.text, flags=re.IGNORECASE)):
                tmp_rep = config.grammar.string_numeral_ambig
                core_synsets.append([tmp_rep])
                # Print some notes
                if do_verbose:
                    print("Word itself is a numeral. Stored synset: {0}".format(core_synsets))

                continue

            # Ignore this word if not a relevant p.o.s.
            check_useless = self._is_pos_word(word=curr_word, pos="USELESS", keyword_objs=keyword_objs)
            check_adj = self._is_pos_word(word=curr_word, pos="ADJECTIVE")
            if do_skip_useless and (check_useless and (not check_adj)):
                # Print some notes
                if do_verbose:
                    print("Word itself is useless. Skipping.")

                continue

            # Gather and store the noun synsets
            curr_synsets_raw = [
                item.name() for item in wordnet.synsets(curr_word.text) if (".n." in item.name())
            ]  # Noun synsets only

            # If no synsets known, store word itself
            if len(curr_synsets_raw) == 0:
                core_synsets.append([curr_word.text.lower()])
            # Otherwise, store synsets
            else:
                core_synsets += [curr_synsets_raw]

            # Print some notes
            if do_verbose:
                print("-Done considering word: {0}".format(curr_word))
                print("Updated synsets: {0}".format(core_synsets))
                print("Updated keywords: {0}".format(core_keywords))

        # Throw an error if any empty strings passed as synsets
        if any([("" in item) for item in core_synsets]):
            raise ValueError("Err: Empty synset?\n{0}".format(core_synsets))

        # Extract unique roots of sets of synsets
        exp_synset = config.grammar.regex.exp_synset
        core_roots = [
            np.unique(
                [item2.split(".")[0] if bool(re.search(exp_synset, item2)) else item2 for item2 in item1]
            ).tolist()
            for item1 in core_synsets
        ]

        # Convert core meaning into string representation
        str_meaning = " ".join([" ".join(item) for item in core_roots])  # Long spaced string

        # Return the core components
        if do_verbose:
            print(
                (
                    "\n-\nPhrase '{0}':\nKeyword: {1}\nSynsets: {2}" + "\nRoots: {3}\nString representation: {4}\n-\n"
                ).format(phrase_NLP, core_keywords, core_synsets, core_roots, str_meaning)
            )

        return {
            "keywords": core_keywords,
            "synsets": core_synsets,
            "roots": core_roots,
            "text": phrase_NLP.text,
            "str_meaning": str_meaning,
        }

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
        # do_verbose_deep = self._get_info("do_verbose_deep")
        do_check_truematch = self._get_info("do_check_truematch")
        sentences = np.asarray(self._get_info("text_clean_split"))
        do_not_classify = keyword_obj._get_info("do_not_classify")
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
            ii for ii in range(0, num_sentences) if keyword_obj.identify_keyword(sentences[ii], mode="keyword")["bool"]
        ]

        # For acronym terms
        inds_with_acronyms = [
            ii for ii in range(0, num_sentences) if keyword_obj.identify_keyword(sentences[ii], mode="acronym")["bool"]
        ]

        # Print some notes
        if do_verbose:
            print(
                ("Found:\n# of keyword sentences: {0}\n" + "# of acronym sentences: {1}...").format(
                    len(inds_with_keywords_init), len(inds_with_acronyms)
                )
            )

        # If requested, run a check for ambiguous phrases if any ambig. keywords
        if (
            (not do_not_classify)
            and do_check_truematch
            and any([keyword_obj.identify_keyword(item)["bool"] for item in lookup_ambigs])
        ):
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
            inds_with_keywords_truematch = inds_with_keywords_init  # Copy over

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
            sentences_buffered = sentences[inds_with_terms].tolist()

        # Return outputs
        return {
            "paragraph": sentences_buffered,
        }

    # Process database of ambig. phrases into lookups and dictionary
    def _process_database_ambig(self, keyword_objs=None, do_verbose=False):
        """
        Method: _process_database_ambig
        WARNING! This method is *not* meant to be used directly by users.
        Purpose:
         - Process database of ambiguous keyword phrases into dictionary of keywords, regular expressions, boolean verdicts, etc.
        """

        # Load the keywords
        if keyword_objs is None:
            try:
                keyword_objs = [self._get_info("keyword_obj", do_flag_hidden=True)]
            except KeyError:
                keyword_objs = self._get_info("keyword_objs", do_flag_hidden=True)

        # Load the ambig. lookup phrases
        lookup_ambigs = [
            item._get_info("name").lower() for item in keyword_objs if (len(item._get_info("ambig_words")) > 0)
        ]

        # Load the ambig. phrase data
        data_ambigs = np.array(config.textprocessing.phrases_ambig)
        if len(data_ambigs.shape) == 1:  # If single row, reshape to 2D
            data_ambigs = data_ambigs.reshape(1, data_ambigs.shape[0])
        num_ambigs = data_ambigs.shape[0]

        str_anymatch_ambig = config.grammar.string_anymatch_ambig.lower()

        ind_keyword = 0
        ind_phrase = 1
        ind_bool = 2

        # Initialize containers for processed ambig. data
        list_kw_ambigs = []
        list_exp_exact_ambigs = []
        list_exp_meaning_ambigs = []
        list_bool_ambigs = []
        list_text_ambigs = []

        # Convert known ambig. phrase database into dict. of components
        for ii in range(0, num_ambigs):
            # Convert to NLP notation
            curr_text = data_ambigs[ii, ind_phrase]
            curr_NLP = nlp(str(curr_text))

            # Extract current boolean value
            if data_ambigs[ii, ind_bool].lower().strip() in ["true", "yes", "t", "y"]:
                curr_bool = True
            elif data_ambigs[ii, ind_bool].lower().strip() in ["false", "no", "f", "n"]:
                curr_bool = False
            else:
                raise ValueError("Err: {0}:{1} in ambig. database not bool!".format(ii, data_ambigs[ii, ind_bool]))

            # Formulate current regular expression
            curr_extraction = self._extract_core_from_phrase(
                phrase_NLP=curr_NLP, do_verbose=do_verbose, do_skip_useless=False, keyword_objs=keyword_objs
            )
            curr_roots = curr_extraction["roots"]
            curr_exp_exact = r"\b(" + re.escape(curr_text) + r")\b"
            curr_exp_meaning = (
                r"(" + r") (\w+ )*(".join([(r"\b(" + r"|".join(item) + r")\b") for item in curr_roots]) + r")"
            )  # Convert to reg. exp. for substring search later

            # Extract current keywords
            curr_kw_raw = data_ambigs[ii, ind_keyword].lower()
            if str_anymatch_ambig == curr_kw_raw:  # Match any keyword
                curr_kw = [item.lower() for item in lookup_ambigs]
            else:  # Otherwise, store given keyword
                curr_kw = [curr_kw_raw]

            # Store the extracted data for each keyword
            tmp_num = len(curr_kw)
            list_kw_ambigs += curr_kw
            list_exp_exact_ambigs += [
                re.sub(str_anymatch_ambig, curr_kw[jj], curr_exp_exact, flags=re.IGNORECASE) for jj in range(0, tmp_num)
            ]
            list_exp_meaning_ambigs += [
                re.sub(str_anymatch_ambig, curr_kw[jj], curr_exp_meaning, flags=re.IGNORECASE)
                for jj in range(0, tmp_num)
            ]
            list_bool_ambigs += [curr_bool] * tmp_num
            list_text_ambigs += [curr_text] * tmp_num

        # Gather all of the results into a dictionary
        dict_ambigs = {
            "lookup_ambigs": lookup_ambigs,
            "all_kw_ambigs": list_kw_ambigs,
            "all_exp_exact_ambigs": list_exp_exact_ambigs,
            "all_exp_meaning_ambigs": list_exp_meaning_ambigs,
            "all_bool_ambigs": list_bool_ambigs,
            "all_text_ambigs": list_text_ambigs,
        }

        # Return the processed results
        return dict_ambigs

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
        text_flat = [
            item for phrase in text_lines for item in re.split(config.grammar.regex.exp_splitbracketstarts, phrase)
        ]
        # Split by sentences ending with brackets
        text_flat = [
            item for phrase in text_flat for item in re.split(config.grammar.regex.exp_splitbracketends, phrase)
        ]
        # Then split by assumed sentence structure
        text_flat = [item for phrase in text_flat for item in re.split(config.grammar.regex.exp_splittext, phrase)]
        # Return the split text
        return text_flat

    # Cleanse given (short) string of extra whitespace, dashes, etc,
    # with uniform placeholders.
    def _streamline_phrase(self, text, do_streamline_etal):
        """
        Method: _streamline_phrase
        WARNING! This method is *not* meant to be used directly by users.
        Purpose:
         - Run _cleanse_text with citation streamlining.
         - Replace websites with uniform placeholder.
         - Replace some common science abbreviations that confuse external NLP package sentence parsing.
        """

        # Extract global variables
        dict_exp_abbrev = config.grammar.regex.dict_exp_abbrev

        # Remove any initial excessive whitespace
        text = self._cleanse_text(text=text, do_streamline_etal=do_streamline_etal)

        # Replace annoying <> inserts (e.g. html)
        text = re.sub(r"<[A-Z|a-z|/]+>", "", text)

        # Replace annoying abbreviations that confuse NLP sentence parser
        for key1 in dict_exp_abbrev:
            text = re.sub(key1, dict_exp_abbrev[key1], text)

        # Remove any new excessive whitespace and punctuation spaces
        text = self._cleanse_text(text=text, do_streamline_etal=do_streamline_etal)

        # Return streamlined text
        return text
