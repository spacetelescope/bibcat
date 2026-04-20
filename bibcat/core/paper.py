"""
:title: paper.py

The primary purpose is to extract all sentences within a larger text block
that refers to a given mission(s).  This collection of sentences for each mission,
denoted from here on as a 'paragraph,' is created to focus on the portions of
the original text related to the target mission.  Using paragraphs for classification
instead of the full text allows us to remove much of the 'noise' inherent to
the rest of the text.

"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import spacy
import spacy.tokens
from nltk.corpus import wordnet  # type: ignore

from bibcat import config
from bibcat.core.core_utils import check_importance, cleanse_text, is_pos_word
from bibcat.core.keyword import Keyword
from bibcat.utils.logger_config import setup_logger

# set up logger
logger = setup_logger(__name__)
logger.setLevel(config.logging.level)

nlp = spacy.load(config.grammar.spacy_language_model)


@dataclass
class TruematchSetup:
    """
    Container for all variables required during the `_check_truematch` process in paper.py.

    Attributes
    ----------
    text : str
        The input text being analyzed.
    dict_ambigs : dict of str to list of str
        Database of ambiguous mission phrases loaded or provided.
    keyword_objs : list of Keyword
        List of keyword objects used for matching.
    list_kw_ambigs : list of str
        List of keywords associated with ambiguous phrases.
    list_exp_exact_ambigs : list of str
        List of regex patterns for exact ambiguous matches.
    list_exp_meaning_ambigs : list of str
        List of regex patterns for meaning-based ambiguous matches.
    list_bool_ambigs : list of str
        List of boolean flags indicating ambiguity status.
    list_text_ambigs : list of str
        Original ambiguous phrases from the database.
    lookup_ambigs : list of str
        List of lookup terms for ambiguity detection.
    lookup_ambigs_lower : list of str
        Lowercased version of lookup terms.
    num_ambigs : int
        Total number of ambiguous phrases.
    keyword_objs_ambigs : list of Keyword
        Subset of keyword objects that are potentially ambiguous.
    dict_kobjinfo : dict
        Mapping of keyword objects to their match info provided by
        `Keyword.identify_keyword`.
    """

    text: str
    dict_ambigs: Dict[str, List[str]]
    keyword_objs: List[Keyword]
    list_kw_ambigs: List[str]
    list_exp_exact_ambigs: List[str]
    list_exp_meaning_ambigs: List[str]
    list_bool_ambigs: List[str]
    list_text_ambigs: List[str]
    lookup_ambigs: List[str]
    lookup_ambigs_lower: List[str]
    num_ambigs: int
    keyword_objs_ambigs: List[Keyword]
    dict_kobjinfo: Dict[str, Dict[str, bool | List[List[int]]]]


class Paper:
    """
    Load text and extract sentences containing target mission keywords.

    Splits the input text into sentences, identifies those containing
    target terms from the provided keyword objects, and gathers matching
    sentences into a paragraph for downstream processing.

    Parameters
    ----------
    text : str
        The text to search.
    keyword_objs : list of Keyword
        Target mission keyword instances whose terms are used to search
        the text.
    do_check_truematch : bool
        If True, verifies that mission phrases found in the text are known
        true matches rather than false positives (e.g., "Edwin Hubble" as
        a false match for the Hubble Space Telescope).
    dict_ambigs : dict or None, optional
        Dictionary of ambiguous mission phrases. If None, the external
        ambiguity database is loaded and processed automatically.
        Default is None.
    """

    def __init__(
        self,
        text: str,
        keyword_objs: list[Keyword],
        do_check_truematch: bool,
        dict_ambigs: Optional[dict] = None,
    ) -> None:
        """
        Initialize an instance of the Paper class.

        Stores the input text and keyword objects, optionally loads the
        ambiguous phrase database, cleanses the text, and splits it into
        sentences for downstream paragraph extraction.

        Parameters
        ----------
        text : str
            The text to search.
        keyword_objs : list of Keyword
            Target mission keyword instances whose terms are used to search
            the text.
        do_check_truematch : bool
            If True, verifies that mission phrases found in the text are known
            true matches rather than false positives. Triggers automatic loading
            of the ambiguous phrase database when ``dict_ambigs`` is None.
        dict_ambigs : dict or None, optional
            Pre-loaded dictionary of ambiguous mission phrases. If None and
            ``do_check_truematch`` is True, the database is loaded and processed
            automatically. Default is None.
        """

        # Initialize global storage variable
        self._storage = {}  # Dictionary to hold all information

        # Store information about this paper
        text_original = text
        self._text_original = text_original  # Original text
        self._keyword_objs = keyword_objs  # Keyword groups
        self._do_check_truematch = do_check_truematch

        # Process ambig. phrase data, if not given
        if do_check_truematch:
            if dict_ambigs is None:
                dict_ambigs = self._process_database_ambig()

            lookup_ambigs = dict_ambigs["lookup_ambigs"]
            self._dict_ambigs = dict_ambigs
            self._lookup_ambigs = lookup_ambigs

        # Preprocess the data
        # Cleanse extra whitespace, strange chars, etc.
        text_clean = self._streamline_phrase(text=text_original, do_streamline_etal=False)
        # Split cleansed text into naive sentences
        text_clean_split = self._split_text(text=text_clean)

        # Store the preprocessed text
        self._text_clean_split = text_clean_split

        return

    def get_paragraphs(self, keyword_objs: Optional[list[Keyword]] = None) -> dict:
        """
        Fetch paragraphs previously assembled for the given keyword instances.

        Retrieves stored paragraphs from this instance, optionally filtered to
        the provided keyword objects. If no keyword objects are given, all
        stored paragraphs are returned.

        Parameters
        ----------
        keyword_objs : list of Keyword or None, optional
            Keyword instances for which to retrieve paragraphs. If None, all
            previously assembled paragraphs are returned. Default is None.

        Returns
        -------
        dict
            Mapping of keyword name strings to their corresponding paragraph
            strings.

        Raises
        ------
        ValueError
            If ``process_paragraphs`` has not been called yet on this instance.
        """

        # Attempt to access previously extracted paragraphs
        try:
            dict_paragraphs = self._paragraphs
        # Throw an error if no paragraphs extracted yet
        except AttributeError:
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

    def process_paragraphs(self, buffer: int = 0, do_overwrite: bool = False) -> None:
        """
        Assemble sentences containing target mission keywords into paragraphs.

        Iterates over the stored keyword objects, extracts sentences from the
        preprocessed text that contain matching terms, and stores the resulting
        paragraphs in the instance. By default, raises an error if paragraphs
        have already been extracted to prevent accidental overwrites.

        Parameters
        ----------
        buffer : int, optional
            Number of sentences before and after each keyword-containing
            sentence to include in the paragraph. Default is 0.
        do_overwrite : bool, optional
            If True, allows previously stored paragraphs to be overwritten.
            Default is False.

        Returns
        -------
        None
            Results are stored in the instance attribute ``_paragraphs``,
            retrievable via `get_paragraphs`.

        Raises
        ------
        ValueError
            If paragraphs have already been extracted and ``do_overwrite``
            is False.
        """

        # Extract clean, naively split paragraphs
        keyword_objs = self._keyword_objs

        # If overwrite not allowed, check if paragraphs already extracted+saved
        if not do_overwrite:
            is_exist = True
            # Check for previously stored paragraphs
            try:
                self._paragraphs
            # Catch error raised if no paragraphs exist
            except AttributeError:
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
        self._paragraphs = dict_paragraphs

        return

    def _buffer_indices(self, indices: list[int], buffer: int, max_index: int) -> list[list[int]]:
        """
        Expand a set of indices by a buffer and merge any overlapping spans.

        For each index, creates a span of ``[index - buffer, index + buffer]``,
        then merges spans that overlap and clamps the result to the range
        ``[0, max_index]``.

        Parameters
        ----------
        indices : list of int
            The indices to expand.
        buffer : int
            Number of positions to extend each index in both directions.
        max_index : int
            Upper bound for the resulting spans. Spans are truncated at this
            value and iteration stops early if it is reached.

        Returns
        -------
        list of list of int
            Merged and clamped spans, where each element is a two-element
            list ``[start, end]``.
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

    def _check_truematch(
        self,
        text: str,
        keyword_objs: list[Keyword],
        dict_ambigs: Optional[dict],
    ) -> dict:  # noqa: C901
        """Determine if `text` contains a true or false match to the given keywords.

        Determine if given text contains a true vs. false match to the given keywords.
        A match is a true match if it refers to the given mission (e.g. HST as a reference
            to Hubble).
        A match is a false match if it has a different meaning altogether (e.g. Edwin
            Hubble as a reference to the Hubble Space Telescope).

        This function checks the text against the ambiguous phrases in
            config.textprocessing.phrases_ambig; new ambiguities should be added there.

        The decision proceeds in four stages, short-circuiting when a definitive verdict
            is reached:

        1. Early checks that do not require noun chunk analysis:
            - Non-ambiguous keyword present -> TRUE
            - No keyword match at all -> FALSE
            - Acronym-only match -> TRUE
            - Non-ambiguous phrase present -> TRUE

        2. Build noun wordchunks around keyword terms. A wordchunk is a noun and the words
            that describe it (e.g. "our Hubble results").

        3. If any wordchunk exactly equals a lookup term -> TRUE.

        4. For each wordchunk, look up ambiguous phrase/meaning patterns and
            decide true/false -> TRUE if any wordchunk is TRUE.


        Parameters
        ----------
        text : str
            The text to search.
        keyword_objs : list
            Target missions; terms will be used to search the text.
        dict_ambigs : dict or None
            If None, will load and process external database of ambiguous mission phrases.
            If given, will use what is given.

        Returns
        -------
        dict
            A dictionary with the following parameters:
            - bool: overall true vs false decision
            - info (if result is non-ambiguous): an array with one item representing the
                non-ambiguous result
                - bool: true vs false decision for the non-ambiguous result
                - text_wordchunk: The wordchunk in the text that matches a known
                    non-ambiguous phrase, if applicable. One of these placeholder strings
                    may appear instead:
                    - `"<Not ambig.>"`
                        A non-ambiguous phrase, acronym or keyword was found in the text
                        (true decision).
                    - `"<No matching keywords at all.>"`
                        Text matches no keywords in the list at all (false decision).
                    - `"<Wordchunk has exact term match.>"`
                        A wordchunk in the text is an exact term match (true decision).
                - text_database: None
                - matcher: None
            - info (if only ambiguous matches were found): an array with one item for each
                ambiguous phrase considered
                - bool: true vs false decision for that particular phrase
                - text_wordchunk: the chunk of text searched (e.g. "our Hubble results")
                - text_database: the phrase searched within the chunk of text
                - matcher: the result of regex search

        Raises
        ------
        ValueError
            If no wordchunks are identified after short-circuit checks,
            raises an error with detailed part-of-speech diagnostics.
        NotImplementedError
            If a wordchunk was found after short-circuit checks,
            but no matches or meanings are found in the ambiguous database for the
            wordchunk.
        """
        # Set up initial variables
        setup_data = self._setup_check_truematch_vars(text, dict_ambigs, keyword_objs)

        # Short-circuit checks
        # If any of these return true or false, we can return the result right away
        for check in (
            self._early_true_non_ambig_keywords,
            self._early_false_no_keyword_match,
            self._early_true_acronym_match,
            self._early_true_non_ambig_phrases,
        ):
            result = check(setup_data)
            if result is not None:
                return result

        # Assemble makeshift wordchunks
        # A wordchunk is a noun and the words that describe it (e.g. "our Hubble results")
        # We are interested in wordchunks around mission keywords
        # Not using NLP wordchunks here
        # Not sure why it happened, but NLP sometimes failed to identify nouns/num.
        list_wordchunks = self._assemble_keyword_wordchunks_wrapper(setup_data)

        # Short-circuit check for exact wordchunks
        # If this returns true, we can return the result right away
        result = self._early_true_exact_wordchunk(list_wordchunks, setup_data)
        if result is not None:
            return result

        # Iterate through each wordchunk to determine true vs false match status
        # If any wordchunk is a true match, the overall result will be true
        list_results = [self._consider_wordchunk(curr_chunk, setup_data) for curr_chunk in list_wordchunks]

        # Combine the results and return overall boolean match
        fin_result = {
            "bool": any([(item["bool"]) for item in list_results]),
            "info": [item["info"][0] for item in list_results],
        }
        return fin_result

    def _build_single_info_entry(
        self,
        bool: bool,
        text_wordchunk: str,
        text_database: str | None = None,
        matcher: re.Match | None = None,
    ) -> dict:
        """Construct a single info entry in the return format of `_check_truematch`.

        The returned structure mirrors the dictionary format used across
            `_check_truematch` and related helpers:

        - top-level boolean decision under `bool`
        - a single-element list under `info` containing the details for
            the stage or wordchunk that produced the decision

        Parameters
        ----------
        bool : bool
            Decision for this stage or wordchunk.
        text_wordchunk : str
            The text chunk evaluated (e.g., `our Hubble results`) or placeholder string
            (e.g. `<Not ambig.>`).
        text_database : str or None, optional
            The ambiguous phrase from the database that matched, when applicable.
        matcher : re.Match or None, optional
            The regex match object from the ambiguous phrase match, when applicable.

        Returns
        -------
        dict
            A dictionary with the keys:

            - `bool` : bool
            The same boolean passed in `bool`.
            - `info` : list of dict
            A single-element list containing a dictionary with:
                * `bool` : bool
                * `text_wordchunk` : str
                * `text_database` : str or None
                * `matcher` : re.Match or None
        """
        info_item = {
            "matcher": matcher,
            "bool": bool,
            "text_wordchunk": text_wordchunk,
            "text_database": text_database,
        }
        return {"bool": bool, "info": [info_item]}

    def _setup_check_truematch_vars(
        self,
        text: str,
        dict_ambigs: Optional[dict],
        keyword_objs: list[Keyword],
    ) -> TruematchSetup:
        """Sets up variables and data structures required by `_check_truematch`.

        Parameters
        ----------
        text : str
            The text to search.
        keyword_objs : list
            Target missions; terms will be used to search the text.
        dict_ambigs : dict or None
            If None, will load and process external database of ambiguous mission phrases.
            If given, will use what is given.

        Returns
        -------
        TruematchSetup
            An object containing all relevant variables for `_check_truematch`.
        """

        # Process ambiguous phrase data, if not given
        if dict_ambigs is None:
            dict_ambigs = self._process_database_ambig(keyword_objs=keyword_objs)

        # Extract info from ambiguous database
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
        logger.debug(
            "\n> Running `_check_truematch` for text: '%s'\n"
            "Original text: '%s'\n"
            "Available mission lookups for the ambiguous phrase database: %s",
            text,
            text_orig,
            lookup_ambigs,
        )

        # Extract keyword objects that are potentially ambiguous
        keyword_objs_ambigs = [
            item1 for item1 in keyword_objs if any([item1.identify_keyword(item2)["bool"] for item2 in lookup_ambigs])
        ]

        # Extract keyword identification information for each keyword object
        dict_kobjinfo = {item._name: item.identify_keyword(text) for item in keyword_objs}

        return TruematchSetup(
            text=text,
            dict_ambigs=dict_ambigs,
            keyword_objs=keyword_objs,
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

    def _early_true_non_ambig_keywords(self, setup_data: TruematchSetup) -> dict | None:
        """Return status as a true match if non-ambiguous keywords match to text.

        Parameters
        ----------
        setup_data : TruematchSetup
            An object containing all relevant variables for `_check_truematch`.

        Returns
        -------
        dict or None
            A dictionary indicating a true match in the `_check_truematch` return format,
            if any non-ambiguous keywords are found:
            - bool: True
            - info: A single-element list containing a dictionary with:
                * `bool` : True
                * `text_wordchunk` : "<Not ambig.>"
                * `text_database` : None
                * `matcher` : None
            If no matches are found, returns None.

        See Also
        --------
        _build_single_info_entry : Return format used by `_check_truematch` and related
            helper functions.
        """
        keyword_objs_non_ambigs = [
            item1 for item1 in setup_data.keyword_objs if item1 not in setup_data.keyword_objs_ambigs
        ]
        if any([setup_data.dict_kobjinfo[item1._name]["bool"] for item1 in keyword_objs_non_ambigs]):
            # Print some notes
            logger.debug("Text matches unambiguous keyword. Returning true state.")

            # Return status as true match
            return self._build_single_info_entry(
                matcher=None, bool=True, text_wordchunk="<Not ambig.>", text_database=None
            )

    def _early_false_no_keyword_match(self, setup_data: TruematchSetup) -> dict | None:
        """Return status as a false match if no keywords match at all.

        Parameters
        ----------
        setup_data : TruematchSetup
            An object containing all relevant variables for `_check_truematch`.

        Returns
        -------
        dict or None
            A formatted dictionary indicating a false match, if no keywords found:
            - bool: False
            - info: A single-element list containing a dictionary with:
                * `bool` : False
                * `text_wordchunk` : "<No matching keywords at all.>"
                * `text_database` : None
                * `matcher` : None
            If any matches are found, returns None.

        See Also
        --------
        _build_single_info_entry : Return format used by `_check_truematch` and related
            helper functions.
        """
        if not any([setup_data.dict_kobjinfo[item._name]["bool"] for item in setup_data.keyword_objs_ambigs]):
            # Print some notes
            logger.debug("Text matches no keywords at all. Returning false state.")

            # Return status as false match
            return self._build_single_info_entry(
                matcher=None, text_database=None, bool=False, text_wordchunk="<No matching keywords at all.>"
            )

    def _early_true_acronym_match(self, setup_data: TruematchSetup) -> dict | None:
        """Return status as a true match if any acronyms match.

        Parameters
        ----------
        setup_data : TruematchSetup
            An object containing all relevant variables for `_check_truematch`.

        Returns
        -------
        dict or None
            A formatted dictionary indicating a true match, if any acronyms match:
            - bool: True
            - info: A single-element list containing a dictionary with:
                * `bool` : True
                * `text_wordchunk` : "<Not ambig.>"
                * `text_database` : None
                * `matcher` : None
            If any matches are found, returns None.

        See Also
        --------
        _build_single_info_entry : Return format used by `_check_truematch` and related
            helper functions.
        """
        if any([setup_data.dict_kobjinfo[item._name]["bool_acronym_only"] for item in setup_data.keyword_objs_ambigs]):
            # Print some notes
            logger.debug("Text matches acronym. Returning true state.")

            # Return status as true match
            return self._build_single_info_entry(
                matcher=None, bool=True, text_wordchunk="<Not ambig.>", text_database=None
            )

    def _early_true_non_ambig_phrases(self, setup_data: TruematchSetup) -> dict | None:
        """Return status as a true match if non-ambiguous phrases match to text.

        Parameters
        ----------
        setup_data : TruematchSetup
            An object containing all relevant variables for `_check_truematch`.

        Returns
        -------
        dict or None
            A formatted dictionary indicating a true match, if any non-ambiguous phrases
            are found:
            - bool: True
            - info: A single-element list containing a dictionary with:
                * `bool` : True
                * `text_wordchunk` : "<Not ambig.>"
                * `text_database` : None
                * `matcher` : None
            If no matches are found, returns None.

        See Also
        --------
        _build_single_info_entry : Return format used by `_check_truematch` and related
            helper functions.
        """
        for obj in setup_data.keyword_objs_ambigs:
            for kw in obj._keywords:
                if kw in obj._ambig_words:
                    continue
                if re.search(rf"\b{re.escape(kw)}\b", setup_data.text, flags=re.IGNORECASE):
                    # Print some notes
                    logger.debug("Text matches unambiguous phrase. Returning true state.")

                    # Return status as true match
                    return self._build_single_info_entry(
                        matcher=None, bool=True, text_wordchunk="<Not ambig.>", text_database=None
                    )

    def _assemble_keyword_wordchunks_wrapper(self, setup_data: TruematchSetup) -> List[spacy.tokens.Doc]:
        """Wrapper for Base._assemble_keyword_wordchunks() used by _check_truematch()

        Wraps the `_assemble_keyword_wordchunks` method (inherited from `base.py`) with
        debug logging and error handling. The base method assembles noun chunks around
        keyword terms found in the input text.

        Parameters
        ----------
        setup_data : TruematchSetup
            An object containing all relevant variables for `_check_truematch`.

        Returns
        -------
        List of spacy.tokens.Doc
            A list of wordchunks (noun phrases), provided as spacy Doc objects,
            that can be used by `_early_true_exact_wordchunk` and `_consider_wordchunk`.

        Raises
        ------
        ValueError
            If no wordchunks are identified, raises an error with detailed part-of-speech
            diagnostics.
        """
        # Print some notes
        logger.debug("Building noun chunks around keywords...")

        # Generate the keyword-based wordchunks
        list_wordchunks = self._assemble_keyword_wordchunks(
            text=setup_data.text,
            keyword_objs=setup_data.keyword_objs,
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
        logger.debug("\n- Wordchunks determined for text: {0}".format(list_wordchunks))

        return list_wordchunks

    def _early_true_exact_wordchunk(
        self, list_wordchunks: List[spacy.tokens.Doc], setup_data: TruematchSetup
    ) -> dict | None:
        """Return status as a true match if any wordchunk is an exact keyword match.

        Parameters
        ----------
        list_wordchunks : list of spacy.tokens.Doc
            A list of wordchunks returned by `_assemble_keyword_wordchunks_wrapper`.
        setup_data : TruematchSetup
            An object containing all relevant variables for `_check_truematch`.

        Returns
        -------
        dict or None
            A formatted dictionary indicating a true match, if any exact keyword matches
            are found:
            - bool: True
            - info: A single-element list containing a dictionary with:
                * `bool` : True
                * `text_wordchunk` : "<Wordchunk has exact term match.>"
                * `text_database` : None
                * `matcher` : None
            If no matches are found, returns None.

        See Also
        --------
        _build_single_info_entry : Return format used by `_check_truematch` and related
            helper functions.
        """
        if any([(item.text.lower() in setup_data.lookup_ambigs) for item in list_wordchunks]):
            # Print some notes
            logger.debug("Exact keyword match found. Returning true status...")

            return self._build_single_info_entry(
                matcher=None, text_database=None, bool=True, text_wordchunk="<Wordchunk has exact term match.>"
            )

    def _consider_wordchunk(self, curr_chunk: spacy.tokens.Doc, setup_data: TruematchSetup) -> list:
        """Evaluates a given wordchunk for an ambiguous match or meaning.

        Evaluates a given wordchunk (noun phrase) to determine if it matches any known
        ambiguous phrases or meanings. If a match is found, returns a list of formatted
        match results. If no match is found, raises a `NotImplementedError` to signal an
        unrecognized ambiguous phrase.

        Parameters
        ----------
        curr_chunk : spacy.tokens.Doc
            The wordchunk (noun phrase) to consider.
        setup_data : TruematchSetup
            An object containing all relevant variables for `_check_truematch`.

        Returns
        -------
        list
            A list of formatted match results if a valid ambiguous phrase match is found.

        Raises
        ------
        NotImplementedError
            If no matches or meanings are found in the ambiguous database for the
            wordchunk.
        """
        curr_chunk_text = curr_chunk.text
        # Print some notes
        logger.debug("Considering wordchunk: {0}".format(curr_chunk_text))

        # Short-circuit check
        result = self._early_true_non_ambig_term(curr_chunk_text, setup_data)
        if result is not None:
            return result

        # Setup variables
        curr_meaning, curr_inner_kw = self._setup_consider_wordchunk(curr_chunk, setup_data)

        # Extract all ambiguous phrases and substrings that are either in:
        # - the list of exact ambiguous matches (setup_data.list_exp_exact_ambigs)
        # - the list of meaning-based ambiguous matches (setup_data.list_exp_meaning_ambigs)
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

        # Determine and extract best match (match with shortest substring)
        list_results = self._assemble_consider_wordchunk_results(set_matches, curr_chunk, curr_meaning, setup_data)

        return list_results

    def _early_true_non_ambig_term(self, curr_chunk_text: str, setup_data: TruematchSetup) -> dict | None:
        """Evaluates if a wordchunk matches a non-ambiguous phrase.

        Evaluates a given wordchunk (noun phrase) to determine if it matches any known
        non-ambiguous phrases. If a match is found, returns a true match.

        Parameters
        ----------
        curr_chunk_text : str
            The wordchunk (noun phrase) to consider.
        setup_data : TruematchSetup
            An object containing all relevant variables for `_check_truematch`.

        Returns
        -------
        dict or None
            A formatted dictionary indicating a true match, if a non-ambiguous term is
            found:
            - bool: True
            - info: A single-element list containing a dictionary with:
                * `bool` : True
                * `text_wordchunk` : The chunk of text searched.
                    (e.g. "our Hubble results")
                * `text_database` : None
                * `matcher` : None
            If no matches are found, returns None.

        See Also
        --------
        _build_single_info_entry : Return format used by `_check_truematch` and related
            helper functions.
        """
        # Store as non-ambig. phrase and skip ahead if non-ambig. term
        is_exact = any(
            [
                (curr_chunk_text.lower().replace(".", "") == item2.lower())
                for item1 in setup_data.keyword_objs
                for item2 in (item1._keywords + item1._acronyms_casesensitive + item1._acronyms_caseinsensitive)
                if (item2.lower() not in setup_data.lookup_ambigs_lower)
            ]
        )  # Check if wordchunk matches to any non-ambig terms
        if is_exact:
            # Print some notes
            logger.debug("Exact match to non-ambig. phrase. Marking true...")

            # Store info for this true match
            list_results = self._build_single_info_entry(
                matcher=None, bool=True, text_wordchunk=curr_chunk_text, text_database=None
            )
            # Skip ahead
            return list_results

    def _setup_consider_wordchunk(
        self, curr_chunk: spacy.tokens.Doc, setup_data: TruematchSetup
    ) -> tuple[str, list[str]]:
        """Returns setup variables for use by _extract_ambig_phrases_substrings.

        Parameters
        ----------
        curr_chunk : spacy.tokens.Doc
            The wordchunk (noun phrase) to consider.
        setup_data : TruematchSetup
            An object containing all relevant variables for `_check_truematch`.

        Returns
        -------
        tuple
            The string representation of the core meaning of the current wordchunk, and
            any matched keywords.
        """
        # Extract representation of core meaning of current wordchunk
        tmp_res = self._extract_core_from_phrase(
            phrase_NLP=curr_chunk,
            keyword_objs=setup_data.keyword_objs_ambigs,
            do_skip_useless=False,
        )
        curr_meaning = tmp_res["str_meaning"]  # Str representation of meaning
        curr_inner_kw = tmp_res["keywords"]  # Matched keywords

        return curr_meaning, curr_inner_kw

    def _extract_ambig_phrases_substrings(
        self,
        exp_list: list[str],
        label: str,
        curr_chunk_text: str,
        curr_meaning: str,
        curr_inner_kw: list[str],
        setup_data: TruematchSetup,
    ) -> list:
        """Extract all ambiguous phrases and substrings that match to this meaning.

        `_consider_wordchunk` runs this first initially against `list_exp_exact_ambigs`,
        and if no matches are found, retries with `list_exp_meaning_ambigs`.

        Parameters
        ----------
        exp_list : list of str
            The ambiguous dictionary to use, `list_exp_exact_ambigs` or
            `list_exp_meaning_ambigs`.
        label : str
            The name of the ambiguous dictionary used for logging/debugging, "exact" or
            "meaning".
        curr_chunk_text : str
            The wordchunk (noun phrase) to consider.
        curr_meaning : str
            The string representation of the core meaning of the wordchunk.
        curr_inner_kw : list of str
            A list of keywords extracted from the wordchunk.
        setup_data : TruematchSetup
            An object containing all relevant variables for `_check_truematch`.

        Returns
        -------
        list
            A list of match dictionaries, the best of which will be returned by
            `_assemble_consider_wordchunk_results`. Each dictionary contains:
            - "ind": Index of the match in the ambiguous list.
            - "text_database": Ambiguous phrase from the database.
            - "text_wordchunk": The wordchunk being evaluated.
            - "exp": The regular expression in the database.
            - "matcher": The regular expression match object (if any).
            - "bool": The boolean flag associated with the ambiguous phrase.
        """
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
        logger.debug(f"Set of {label} assembled from ambig. database:")
        for item1 in set_matches_raw:
            logger.debug(item1)

        return set_matches

    def _assemble_consider_wordchunk_results(
        self, set_matches: list[dict], curr_chunk: spacy.tokens.Doc, curr_meaning: str, setup_data: TruematchSetup
    ) -> dict:
        """Selects the best match from a list of ambiguous phrase matches.

        The ambiguous phrase with the shortest matched substring will be chosen as the
            best match.

        Parameters
        ----------
        set_matches : list of dict
            A list of matches provided by `_extract_ambig_phrases_substrings`.
        curr_chunk : spacy.tokens.Doc
            The wordchunk (noun phrase) to consider.
        curr_meaning : str
            The string representation of the core meaning of the wordchunk.
        setup_data : TruematchSetup
            An object containing all relevant variables for `_check_truematch`.

        Returns
        -------
        dict
            A formatted dictionary representing the best match result:
            - bool: True or False decision for the match result.
            - info: A single-element list containing a dictionary with:
                * `bool` : True or False decision for the match result.
                * `text_wordchunk` : The chunk of text searched for the match result.
                    (e.g. "our Hubble results")
                * `text_database` : The database which matches with the result.
                * `matcher` : The regular expression used to obtain the match result.

        See Also
        --------
        _build_single_info_entry : Return format used by `_check_truematch` and related
            helper functions.
        """
        best_set = sorted(set_matches, key=(lambda w: len(w["matcher"][0])))[0]

        # Print some notes
        logger.debug("Current wordchunk: {0}\nMeaning: {2}\nBest set: {1}-".format(curr_chunk, best_set, curr_meaning))

        # Exit method early since match found
        logger.debug("Match found. Returning status...")

        list_results = self._build_single_info_entry(
            matcher=best_set["matcher"],
            bool=best_set["bool"],
            text_wordchunk=best_set["text_wordchunk"],
            text_database=best_set["text_database"],
        )

        return list_results

    def _assemble_keyword_wordchunks(  # noqa: C901
        self,
        text: str,
        keyword_objs: list[Keyword],
        do_include_verbs: bool = False,
        do_include_brackets: bool = False,
    ) -> list[spacy.tokens.Doc]:
        """
        Assemble noun chunks around keyword terms found in the given text.

        For each sentence in ``text``, locates tokens that match any keyword
        object, then expands outward in both directions to accumulate adjacent
        nouns, adjectives, numerals, possessives, dashes, and important terms
        into a single word chunk. Expansion stops at the first token that does
        not meet any inclusion criterion.

        Parameters
        ----------
        text : str
            The input text to search for keyword-containing word chunks.
        keyword_objs : list of Keyword
            Keyword instances used to identify matching tokens.
        do_include_verbs : bool, optional
            If True, verbs are also included during rightward expansion.
            Useful for cases such as hyphenated noun-verbs (e.g.,
            "Hubble-imaged"). Default is False.
        do_include_brackets : bool, optional
            If True, bracket tokens are included during both leftward and
            rightward expansion. Default is False.

        Returns
        -------
        list of spacy.tokens.Doc
            One NLP-processed Doc per assembled word chunk, in order of
            appearance across all sentences in ``text``.
        """

        # Find indices of keywords within text
        tmp_sents = list(nlp(str(text)).sents)
        list_wordchunks = []
        for curr_sent in tmp_sents:
            # Check against all keyword objects
            set_inds = [
                ind
                for ind in range(0, len(curr_sent))
                if any([item.identify_keyword(curr_sent[ind].text)["bool"] for item in keyword_objs])
            ]
            # Print some notes
            logger.info("Current sentence: '{0}'".format(curr_sent))
            logger.info("Indices of lookups in sent.: '{0}'".format(set_inds))
            # Build wordchunks from indices of current sentence
            last_ind = -np.inf
            for curr_start in set_inds:
                # Print some notes
                logger.info("\n-Building wordchunk from {0}: {1}:".format(curr_start, curr_sent[curr_start]))
                # Skip if this index has already been surpassed
                if curr_start <= last_ind:
                    continue
                curr_wordtext = [curr_sent[curr_start].text]

                # Build wordchunk from accumulating nouns on the left
                for ii in range(0, curr_start)[::-1]:  # Do not include start here
                    # Store index if noun or numeral or adjective
                    check_noun = is_pos_word(word=curr_sent[ii], pos="NOUN")
                    check_adj = is_pos_word(word=curr_sent[ii], pos="ADJECTIVE")
                    check_num = is_pos_word(word=curr_sent[ii], pos="NUMBER")
                    check_pos = is_pos_word(word=curr_sent[ii], pos="POSSESSIVE")
                    check_dash = curr_sent[ii].text == "-"
                    check_imp = check_importance(
                        curr_sent[ii].text, keyword_objs=keyword_objs, version_NLP=curr_sent[ii]
                    )["bools"]["is_any"]
                    # Include punctuation, if so requested
                    if do_include_brackets:
                        check_brackets = is_pos_word(word=curr_sent[ii], pos="BRACKET")
                    else:
                        check_brackets = False
                    tmp_list = [check_noun, check_adj, check_num, check_dash, check_pos, check_imp, check_brackets]

                    # Keep word if relevant p.o.s.
                    if any(tmp_list):
                        curr_wordtext.insert(0, curr_sent[ii].text)
                    #
                    # Otherwise, break and end this makeshift wordchunk
                    else:
                        break

                # Build wordchunk from accumulating nouns on the right
                for ii in range((curr_start + 1), len(curr_sent)):
                    # Store index if noun or numeral or adjective, etc.
                    check_noun = is_pos_word(word=curr_sent[ii], pos="NOUN")
                    check_adj = is_pos_word(word=curr_sent[ii], pos="ADJECTIVE")
                    check_num = is_pos_word(word=curr_sent[ii], pos="NUMBER")
                    check_pos = is_pos_word(word=curr_sent[ii], pos="POSSESSIVE")
                    check_imp = check_importance(
                        curr_sent[ii].text, keyword_objs=keyword_objs, version_NLP=curr_sent[ii]
                    )["bools"]["is_any"]
                    check_dash = curr_sent[ii].text == "-"
                    # Include brackets, if so requested
                    if do_include_brackets:
                        check_brackets = is_pos_word(word=curr_sent[ii], pos="BRACKET")
                    else:
                        check_brackets = False

                    # Tack on verb check if requested (e.g., to cover noun-verbs)
                    if do_include_verbs:  # E.g., ambig. 'Hubble-imaged data'
                        check_verb = is_pos_word(word=curr_sent[ii], pos="VERB")
                    else:
                        check_verb = False
                    #
                    tmp_list = [
                        check_noun,
                        check_adj,
                        check_num,
                        check_brackets,
                        check_verb,
                        check_pos,
                        check_imp,
                        check_dash,
                    ]

                    # Keep word if relevant p.o.s.
                    if any(tmp_list):
                        curr_wordtext.append(curr_sent[ii].text)
                        last_ind = ii  # Update latest index

                    # Otherwise, break and end this makeshift wordchunk
                    else:
                        break

                # Store the makeshift wordchunk
                curr_str_fin = cleanse_text(" ".join(curr_wordtext), do_streamline_etal=False)
                list_wordchunks.append(nlp(curr_str_fin))

                # Print some notes
                logger.info(
                    "All wordchunks so far: {0}\nNewest wordchunk: {1}".format(list_wordchunks, list_wordchunks[-1])
                )
                logger.info(
                    "pos_ values: {0}\ndep_ values: {1}\ntag_ values: {2}".format(
                        [item.pos_ for item in list_wordchunks[-1]],
                        [item.dep_ for item in list_wordchunks[-1]],
                        [item.tag_ for item in list_wordchunks[-1]],
                    )
                )

        # Return the assembled wordchunks
        logger.info("Assembled keyword wordchunks:\n{0}".format(list_wordchunks))

        return list_wordchunks

    def _extract_core_from_phrase(  # noqa: C901
        self,
        phrase_NLP: spacy.tokens.Doc,
        do_skip_useless: bool,
        keyword_objs: Optional[list[Keyword]] = None,
    ) -> dict:
        """
        Extract the representative meaning of a phrase as synsets and root terms.

        Iterates over each token in ``phrase_NLP``, skipping punctuation,
        possessives, and optionally useless words, then resolves each remaining
        token to one of the following:

        - A keyword name, if the token matches a keyword object.
        - A numeral placeholder, if the token is a number.
        - WordNet noun synsets, or the token text itself if no synsets are found.

        Synset names are reduced to their root form (the portion before the
        first ``.``) and joined into a single string representation.

        Parameters
        ----------
        phrase_NLP : spacy.tokens.Doc
            The NLP-processed phrase to extract meaning from.
        do_skip_useless : bool
            If True, tokens classified as useless (excluding adjectives) are
            skipped during processing.
        keyword_objs : list of Keyword or None, optional
            Keyword instances used to identify matching tokens. If None,
            falls back to the instance's stored keyword object(s).
            Default is None.

        Returns
        -------
        dict
            A dictionary with the following keys:

            - ``"keywords"`` : list of str -- Lowercase keyword names matched
            within the phrase.
            - ``"synsets"`` : list of list of str -- Raw synset names (or
            token text) for each processed token.
            - ``"roots"`` : list of list of str -- Unique root terms extracted
            from ``"synsets"``.
            - ``"text"`` : str -- Original text of ``phrase_NLP``.
            - ``"str_meaning"`` : str -- Space-joined string of all root terms,
            representing the core meaning of the phrase.

        Raises
        ------
        ValueError
            If any empty string is found among the assembled synsets.
        """

        # Set global variables
        num_words = len(phrase_NLP)
        if keyword_objs is None:
            try:
                keyword_objs = [self._keyword_obj]
            except AttributeError:
                keyword_objs = self._keyword_objs

        # Print some notes
        logger.info("\n> Running _extract_core_from_phrase for phrase: {0}".format(phrase_NLP))

        # Initialize containers of core information
        core_keywords = []
        core_synsets = []

        # Iterate through words within phrase
        for ii in range(0, num_words):
            curr_word = phrase_NLP[ii]
            # Print some notes
            logger.info("-\nLatest keywords: {0}".format(core_keywords))
            logger.info("Latest synsets: {0}".format(core_synsets))
            logger.info("-Now considering word: {0}".format(curr_word))

            # Skip if this word is punctuation or possessive marker
            if is_pos_word(word=curr_word, pos="PUNCTUATION") or is_pos_word(word=curr_word, pos="POSSESSIVE"):
                # Print some notes
                logger.info("Word is punctuation or possessive. Skipping.")

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
                    logger.info("Word itself is keyword. Stored synset: {0}".format(core_synsets))

                    continue
                # Otherwise, store keyword itself
                else:
                    name_kobj = matched_kobjs[0].get_name()  # Fetch name for kobj
                    core_keywords.append(name_kobj.lower())
                    core_synsets.append([name_kobj.lower()])

                    # Print some notes
                    logger.info("Word itself is keyword. Stored synset: {0}".format(core_synsets))

                    continue

            # Store a representative synset and skip ahead if word is a numeral
            if bool(re.search(("^(ID)?[0-9]+"), curr_word.text, flags=re.IGNORECASE)):
                tmp_rep = config.grammar.string_numeral_ambig
                core_synsets.append([tmp_rep])
                # Print some notes
                logger.info("Word itself is a numeral. Stored synset: {0}".format(core_synsets))

                continue

            # Ignore this word if not a relevant p.o.s.
            check_useless = is_pos_word(word=curr_word, pos="USELESS", keyword_objs=keyword_objs)
            check_adj = is_pos_word(word=curr_word, pos="ADJECTIVE")
            if do_skip_useless and (check_useless and (not check_adj)):
                # Print some notes
                logger.info("Word itself is useless. Skipping.")

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
            logger.info("-Done considering word: {0}".format(curr_word))
            logger.info("Updated synsets: {0}".format(core_synsets))
            logger.info("Updated keywords: {0}".format(core_keywords))

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
        logger.info(
            ("\n-\nPhrase '{0}':\nKeyword: {1}\nSynsets: {2}" + "\nRoots: {3}\nString representation: {4}\n-\n").format(
                phrase_NLP, core_keywords, core_synsets, core_roots, str_meaning
            )
        )

        return {
            "keywords": core_keywords,
            "synsets": core_synsets,
            "roots": core_roots,
            "text": phrase_NLP.text,
            "str_meaning": str_meaning,
        }

    def _extract_paragraph(self, keyword_obj: Keyword, buffer: int) -> dict:
        """
        Extract sentences containing target mission terms from the stored text.

        Identifies sentences matching keyword or acronym terms from
        ``keyword_obj``, optionally filters out false positives against the
        ambiguous phrase database, then applies a sentence buffer to include
        surrounding context. Matching sentences are returned as a list of
        strings.

        Parameters
        ----------
        keyword_obj : Keyword
            The keyword instance whose terms are used to identify matching
            sentences.
        buffer : int
            Number of sentences before and after each matching sentence to
            include. If 0, only the matching sentences themselves are returned.

        Returns
        -------
        dict
            A dictionary with the following key:

            - ``"paragraph"`` : list of str -- Extracted sentences, with
            buffered sentences joined into single strings where applicable.
        """

        # Fetch global variables
        do_check_truematch = self._do_check_truematch
        sentences = np.asarray(self._text_clean_split)
        do_not_classify = keyword_obj._do_not_classify
        num_sentences = len(sentences)

        # Load ambiguous phrases, if necessary
        if do_check_truematch:
            # Print some notes
            logger.info("do_check_truematch=True, so will verify ambig. phrases.")

            # Load previously stored ambig. phrase data
            dict_ambigs = self._dict_ambigs
            lookup_ambigs = dict_ambigs["lookup_ambigs"]

        # Print some notes
        logger.info("Fetching inds of target sentences...")

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
        logger.info(
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
            logger.info("Verifying ambiguous phrases...")

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
            logger.info("Done verifying ambiguous phrases.")
            logger.info("Match output:\n{0}".format(output_truematch))
            logger.info("Indices with true matches:\n{0}".format(inds_with_keywords_truematch))
            logger.info("Keyword sentences with true matches:\n{0}".format(sentences[inds_with_keywords_truematch]))

        # Otherwise, set empty
        else:
            output_truematch = None
            inds_with_keywords_truematch = inds_with_keywords_init  # Copy over

        # Pool together unique indices of sentences with keywords, acronyms
        inds_with_terms = list(set().union(inds_with_keywords_truematch, inds_with_acronyms))

        # Determine buffered sentences, if requested
        if buffer > 0:
            # Print some notes
            logger.info("Buffering sentences with buffer={0}...".format(buffer))

            ranges_buffered = self._buffer_indices(
                indices=inds_with_terms, buffer=buffer, max_index=(num_sentences - 1)
            )
            sentences_buffered = [
                " ".join(sentences[item[0] : item[1] + 1]) for item in ranges_buffered
            ]  # Combined sentences
            # Print some notes
            logger.info("Done buffering sentences.\nRanges = {0}.".format(ranges_buffered))

        # Otherwise, just copy over previous indices
        else:
            sentences_buffered = sentences[inds_with_terms].tolist()

        # Return outputs
        return {
            "paragraph": sentences_buffered,
        }

    def _process_database_ambig(self, keyword_objs: Optional[list[Keyword]] = None) -> dict:
        """
        Process the ambiguous phrase database into lookup structures.

        Loads the configured database of ambiguous keyword phrases, converts
        each entry into exact and meaning-based regular expressions using
        synset root extraction, resolves boolean true/false match verdicts,
        and returns all components as a dictionary for use in false-positive
        filtering.

        Parameters
        ----------
        keyword_objs : list of Keyword or None, optional
            Keyword instances used to resolve wildcard keyword entries in the
            database. If None, falls back to the instance's stored keyword
            object(s). Default is None.

        Returns
        -------
        dict
            A dictionary with the following keys:

            - ``"lookup_ambigs"`` : list of str -- Lowercase keyword names
            that have associated ambiguous phrases.
            - ``"all_kw_ambigs"`` : list of str -- Keyword name for each
            processed database entry.
            - ``"all_exp_exact_ambigs"`` : list of str -- Exact-match regular
            expressions for each entry.
            - ``"all_exp_meaning_ambigs"`` : list of str -- Meaning-based
            regular expressions derived from synset roots for each entry.
            - ``"all_bool_ambigs"`` : list of bool -- True/false match verdict
            for each entry.
            - ``"all_text_ambigs"`` : list of str -- Original phrase text for
            each entry.

        Raises
        ------
        ValueError
            If a boolean verdict field in the database cannot be parsed as
            True or False.
        """

        # Load the keywords
        if keyword_objs is None:
            try:
                keyword_objs = [self._keyword_obj]
            except AttributeError:
                keyword_objs = self._keyword_objs

        # Load the ambig. lookup phrases
        lookup_ambigs = [item._name.lower() for item in keyword_objs if (len(item._ambig_words) > 0)]

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
                phrase_NLP=curr_NLP, do_skip_useless=False, keyword_objs=keyword_objs
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

    def _split_text(self, text: str) -> list[str]:
        """
        Split text into sentences based on assumed sentence boundaries.

        Applies splitting in three sequential passes: line breaks, bracket-
        delimited boundaries (both opening and closing), and general sentence
        structure patterns defined in the grammar configuration.

        Parameters
        ----------
        text : str
            The input text to split.

        Returns
        -------
        list of str
            The text split into individual sentences.
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

    def _streamline_phrase(self, text: str, do_streamline_etal: bool) -> str:
        """
        Cleanse and streamline a phrase for downstream NLP processing.

        Applies ``cleanse_text`` to normalize whitespace and punctuation,
        removes HTML-style tags, replaces common science abbreviations that
        confuse the NLP sentence parser, then applies ``cleanse_text`` once
        more to catch any newly introduced whitespace issues.

        Parameters
        ----------
        text : str
            The input text to streamline.
        do_streamline_etal : bool
            Passed directly to ``cleanse_text``. If True, author citation
            patterns are replaced with a uniform placeholder.

        Returns
        -------
        str
            The cleansed and streamlined text.
        """

        # Extract global variables
        dict_exp_abbrev = config.grammar.regex.dict_exp_abbrev

        # Remove any initial excessive whitespace
        text = cleanse_text(text=text, do_streamline_etal=do_streamline_etal)

        # Replace annoying <> inserts (e.g. html)
        text = re.sub(r"<[A-Z|a-z|/]+>", "", text)

        # Replace annoying abbreviations that confuse NLP sentence parser
        for key1 in dict_exp_abbrev:
            text = re.sub(key1, dict_exp_abbrev[key1], text)

        # Remove any new excessive whitespace and punctuation spaces
        text = cleanse_text(text=text, do_streamline_etal=do_streamline_etal)

        # Return streamlined text
        return text
