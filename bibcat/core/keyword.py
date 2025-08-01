"""
:title: keyword.py

This class's primary purpose is to extract all sentences from within
a larger text block that refers to a given mission(s).  This collection of sentences
for each mission, denoted from here on as a 'paragraph', is created to focus
on the portions of the original text related to the target mission.
Using paragraphs for classification instead of the full text allows us to remove
much of the 'noise' inherent to the rest of the text.

"""

import re
from typing import Any

from bibcat import config
from bibcat.core.base import Base
from bibcat.utils.logger_config import setup_logger


class Keyword(Base):
    """
    A Keyword instance stores terms, e.g. titles and acronyms, that describe a mission (e.g., HST, JWST, TESS) for a user.  Methods of a Keyword instance can identify and/or replace snippets within texts that match to the mission.
    """

    def __init__(
        self,
        keywords,
        acronyms_caseinsensitive,
        acronyms_casesensitive,
        banned_overlap,
        ambig_words,
        do_not_classify,
        do_verbose=False,
    ):
        """
        Initialize an instance of the Keyword class, which stores terms, e.g. titles and acronyms, that describe a mission (e.g., HST, JWST, TESS) for a user.

        A Keyword instance is a container for a user-defined mission, such as HST or TESS.  Each instance contains phrases and acronyms that refer to the mission (e.g., "Hubble Space Telescope" and "HST" for HST).
        Each instance also contains optional lists of strings that handle potential overlap of the mission phrases with other phrases.  For example, "Hubble" is common between both HST and the Hubble Legacy Archive.  And so "Hubble Legacy Archive" can be flagged within a Keyword instance for HST to prevent "Hubble Legacy Archive" from being treated as a phrase matching to the HST Keyword instance.
        Additionally, each instance includes a boolean that determines whether or not this mission will be classified by the code (as opposed to only extracting and returning subtexts that contain text about this mission).  This boolean is useful for particularly ambiguous mission names.  The K2 mission is a prime example of an ambiguous mission, where "K2" can be used for both mission and non-mission contexts (like "K2" as a telescope and "K2" as a stellar spectral type) and is often hard for a machine to automatically distinguish.

        Parameters
        ----------
        keywords : list[str]
            List of full phrases that name the mission (e.g., "Hubble Space Telescope").  Not case-sensitive.
        acronyms_caseinsensitive : list[str]
            List of acronyms that can describe the mission; capitalization is not preserved (e.g., "HST" and "hst" are treated in the same manner).  Punctuation should be omitted (as it is handled internally within the code).
        acronyms_casesensitive : list[str]
            List of acronyms that can describe the mission; capitalization is preserved (e.g., "STScI").  Punctuation should be omitted (as it is handled internally within the code).
        banned_overlap : list[str]
            Phrases that overlap with the target mission keywords but should not be treated as the same mission.  E.g., "Hubble Legacy Archive" can be a distinct mission from "Hubble"; therefore "Hubble Legacy Archive" is banned overlap for the Hubble mission, to avoid matching "Hubble Legacy Archive" to a Keyword instance for HST.
        ambig_words : list[str]
            Phrases for which the user requests false positive checks to be done against the internal database of false positives.  E.g., "Hubble" can be found in the mission phrase "Hubble Telescope" and also in the false positive (i.e., non-mission) phrase "Hubble constant".  By specifying "Hubble" as a false positive phrase for the Hubble mission, the code knows to internally check phrases in the text associated with Hubble against the internal false positive database and procedure.
        do_not_classify : bool
            If True, text for the mission will be processed, extracted, and presented to the user, but not classified.  This can be useful for missions for which only human classification is desired.  This can also be useful for missions for which false positives are too difficult to automatically screen out (e.g., "K2", which can be a mission and also a stellar spectral type).
        do_verbose : bool = False
            If True, will print statements and internal reports within applicable methods while the code is running.

        Returns
        -------
        None
        """
        # Initialize storage
        self._storage = {}
        self._store_info(do_verbose, "do_verbose")
        self._store_info(banned_overlap, "banned_overlap")
        self._store_info(ambig_words, "ambig_words")
        self._store_info(do_not_classify, "do_not_classify")

        # Cleanse keywords of extra whitespace, punctuation, etc.
        keywords_clean = sorted(
            [self._cleanse_text(text=phrase, do_streamline_etal=True) for phrase in keywords], key=(lambda w: len(w))
        )[::-1]  # Sort by desc. length
        # Store keywords
        self._store_info(keywords_clean, key="keywords")

        # Cleanse banned overlap of extra whitespace, punctuation, etc.
        banned_overlap_lowercase = [
            self._cleanse_text(text=phrase.lower(), do_streamline_etal=True) for phrase in banned_overlap
        ]
        # Store keywords
        self._store_info(banned_overlap_lowercase, key="banned_overlap_lowercase")

        # Also cleanse+store acronyms, if given
        # For case-insensitive acronyms
        if acronyms_caseinsensitive is not None:
            acronyms_mid = [
                re.sub(config.grammar.regex.exp_nopunct, "", item, flags=re.IGNORECASE)
                for item in acronyms_caseinsensitive
            ]
            # Remove all whitespace
            acronyms_mid = [re.sub(" ", "", item) for item in acronyms_mid]
            acronyms_clean = sorted(acronyms_mid, key=(lambda w: len(w)))[::-1]  # Sort by desc. length
            self._store_info(acronyms_clean, key="acronyms_caseinsensitive")
        else:
            self._store_info([], key="acronyms_caseinsensitive")
        # For case-sensitive acronyms
        if acronyms_casesensitive is not None:
            acronyms_mid = [
                re.sub(config.grammar.regex.exp_nopunct, "", item, flags=re.IGNORECASE)
                for item in acronyms_casesensitive
            ]
            # Remove all whitespace
            acronyms_mid = [re.sub(" ", "", item) for item in acronyms_mid]
            acronyms_clean = sorted(acronyms_mid, key=(lambda w: len(w)))[::-1]  # Sort by desc. length
            self._store_info(acronyms_clean, key="acronyms_casesensitive")
        else:
            self._store_info([], key="acronyms_casesensitive")

        # Store representative name for this keyword object
        repr_name = (
            self._get_info("keywords")[::-1]
            + self._get_info("acronyms_casesensitive")
            + self._get_info("acronyms_caseinsensitive")
        )[0]  # Take shortest keyword or longest acronym as repr. name
        self._store_info(repr_name, key="name")

        # Store regular expression for keywords
        exps_k = [(r"\b" + phrase + r"\b") for phrase in self._get_info("keywords")]

        # Store regular expression for case-insensitive acronyms
        if (acronyms_caseinsensitive is not None) and (len(acronyms_caseinsensitive) > 0):
            # Update acronyms to allow optional spaces and puncuations `.` and `-`
            acronyms_upd = [(r"(\.?)( ?)(-?)".join(item)) for item in self._get_info("acronyms_caseinsensitive")]
            # Build regular expression to recognize acronyms
            exp_a_caseinsensitive = (
                r"(^|[^\.])((\b" + r"\b)|(\b".join([phrase for phrase in acronyms_upd]) + r"\b)(\.?))($|[^A-Z|a-z])"
            )  # Matches any acronym
        else:
            exp_a_caseinsensitive = None

        # Store regular expression for case-sensitive acronyms
        if (acronyms_casesensitive is not None) and (len(acronyms_casesensitive) > 0):
            # Update acronyms to allow optional spaces and puncuations `.` and `-`
            acronyms_upd = [(r"(\.?)( ?)(-?)".join(item)) for item in self._get_info("acronyms_casesensitive")]
            # Build regular expression to recognize acronyms
            exp_a_casesensitive = (
                r"(^|[^\.])((\b" + r"\b)|(\b".join([phrase for phrase in acronyms_upd]) + r"\b)(\.?))($|[^A-Z|a-z])"
            )  # Matches any acronym
        else:
            exp_a_casesensitive = None

        self._store_info(exps_k, key="exps_keywords")
        self._store_info(exp_a_caseinsensitive, key="exp_acronyms_caseinsensitive")
        self._store_info(exp_a_casesensitive, key="exp_acronyms_casesensitive")

        return

    def __repr__(self):
        return f"<Keyword (name='{self.get_name()}')"

    # Generate string representation of this class instance
    def __str__(self):
        # Build string of characteristics of this instance
        print_str = (
            "Keyword Object:\n"
            + f"Name: {self.get_name()}\n"
            + f"Keywords: {self._get_info('keywords')}\n"
            + f"Acronyms (Case-Insensitive): {self._get_info('acronyms_caseinsensitive')}\n"
            + f"Acronyms (Case-Sensitive): {self._get_info('acronyms_casesensitive')}\n"
            + f"Banned Overlap: {self._get_info('banned_overlap')}\n"
        )

        # Return the completed string for printing
        return print_str

    # Purpose: Fetch the representative name for this keyword object
    def get_name(self):
        """
        Fetch and return the representative name for the mission described by this Keyword instance.

        Parameters
        ----------
        None

        Returns
        -------
        out : str
            The representative name for this Keyword instance.
        """
        # Fetch and return representative name
        return self._get_info("name")

    # Purpose: Check if text matches to this keyword object; return match inds
    def identify_keyword(self, text: str, mode: str | None = None):
        """
        Return whether or not the given text contains terms (keywords, acronyms) matching this instance.

        Parameters
        ----------
        text : str
            The text to search within for terms.
        mode : {None, "keyword", "acronym"}, optional
            If a mode is specified, then the code will only search for terms of that type (i.e., either specifically for full phrases OR of acronyms matching to this Keyword instance).  The default is `None`, which searches for all term types.

        Returns
        -------
        out : dict
            Output is a key-value dictionary, containing the following keys and values:
                key="bool", value=bool
                    True if any matches exist.
                key="charspans", value=list[list[int, int]]
                    List of character spans of all matches.  Each character span contains the index location within the string that contains the starting character of the matching substring, and the index location within the string that contains the final character of the matching substring.  For example: for the sentence "HST is a Hubble Space Telescope acronym", the value would be [[0,2], [9, 30]].
                key="bool_acronym_only", value=bool
                    True if any acronym matches exist.
        """
        # Fetch global variables
        exps_k = self._get_info("exps_keywords")
        keywords = self._get_info("keywords")
        exp_a_yescase = self._get_info("exp_acronyms_casesensitive")
        exp_a_nocase = self._get_info("exp_acronyms_caseinsensitive")
        acronyms_casesensitive = self._get_info("acronyms_casesensitive")
        acronyms_caseinsensitive = self._get_info("acronyms_caseinsensitive")
        banned_overlap_lowercase = self._get_info("banned_overlap_lowercase")
        do_verbose = self._get_info("do_verbose")
        allowed_modes = [None, "keyword", "acronym"]

        # Throw error is specified mode not recognized
        if (mode is not None) and (mode.lower() not in allowed_modes):
            raise ValueError("Err: Invalid mode '{0}'.\nAllowed modes: {1}".format(mode, allowed_modes))

        # Modify text (locally only) to block banned overlap
        text_mod = text
        for curr_ban in banned_overlap_lowercase:
            # Below replaces banned phrases with mask '#' string of custom length
            text_mod = re.sub(curr_ban, (lambda x: ("#" * len(x.group()))), text_mod, flags=re.IGNORECASE)
        #

        # Check if this text contains keywords
        if (mode is None) or (mode.lower() == "keyword"):
            set_keywords = [list(re.finditer(item1, text_mod, flags=re.IGNORECASE)) for item1 in exps_k]
            charspans_keywords = [
                (item2.start(), (item2.end() - 1)) for item1 in set_keywords for item2 in item1
            ]  # Char. span of matches
            check_keywords = any([(len(item) > 0) for item in set_keywords])  # If any matches
        else:
            charspans_keywords = []
            check_keywords = False

        # Check if this text contains case-sensitive acronyms
        if ((mode is None) or (mode.lower() == "acronym")) and (exp_a_yescase is not None):
            set_acronyms_yescase = list(re.finditer(exp_a_yescase, text_mod))
            charspans_acronyms_yescase = [(item.start(), (item.end() - 1)) for item in set_acronyms_yescase]
            check_acronyms_yescase = len(set_acronyms_yescase) > 0
        else:
            charspans_acronyms_yescase = []
            check_acronyms_yescase = False

        # Check if this text contains case-insensitive acronyms
        if ((mode is None) or (mode.lower() == "acronym")) and (exp_a_nocase is not None):
            set_acronyms_nocase = list(re.finditer(exp_a_nocase, text_mod, flags=re.IGNORECASE))
            charspans_acronyms_nocase = [(item.start(), (item.end() - 1)) for item in set_acronyms_nocase]
            check_acronyms_nocase = len(set_acronyms_nocase) > 0
        else:
            charspans_acronyms_nocase = []
            check_acronyms_nocase = False

        # Combine acronym results
        check_acronyms_all = check_acronyms_nocase or check_acronyms_yescase
        charspans_acronyms_all = charspans_acronyms_nocase + charspans_acronyms_yescase

        # Print some notes
        if do_verbose:
            print("Keywords: {0}\nKeyword regex:\n{1}".format(keywords, exps_k))
            print("Acronyms (Case-Sensitive): {0}\nAcronym regex:\n{1}".format(acronyms_casesensitive, exp_a_yescase))
            print(
                "Acronyms (Case-Insensitive): {0}\nAcronym regex:\n{1}".format(acronyms_caseinsensitive, exp_a_nocase)
            )
            print("Keyword bool: {0}\nAcronym bool: {1}".format(check_keywords, check_acronyms_all))
            print(
                "Keyword char. spans: {0}\nAcronym char. spans: {1}".format(charspans_keywords, charspans_acronyms_all)
            )

        # Return booleans
        return {
            "bool": (check_acronyms_all or check_keywords),
            "charspans": (charspans_keywords + charspans_acronyms_all),
            "bool_acronym_only": check_acronyms_all,
        }

    # Purpose: Replace any text that matches to this keyword object
    def replace_keyword(self, text: str, placeholder: str):
        """
        Replace any substrings within given text that contain terms (keywords, acronyms) matching this instance.

        Parameters
        ----------
        text : str
            The text to search within for terms.
        placeholder : str
            The substring to replace matched terms with.

        Returns
        -------
        out : str
            The updated text; in the updated text, any keywords/acronyms that matched to this Keyword instance will have been replaced with `placeholder`.
        """
        # Fetch global variables
        exps_k = self._get_info("exps_keywords")
        exp_a_yescase = self._get_info("exp_acronyms_casesensitive")
        exp_a_nocase = self._get_info("exp_acronyms_caseinsensitive")
        if exp_a_yescase is not None:
            exps_a_yescase = [exp_a_yescase]
        else:
            exps_a_yescase = []
        if exp_a_nocase is not None:
            exps_a_nocase = [exp_a_nocase]
        else:
            exps_a_nocase = []
        #
        do_verbose = self._get_info("do_verbose")
        text_new = text
        #

        # Replace terms with given placeholder
        # For keywords
        for curr_exp in exps_k:
            text_new = re.sub(curr_exp, placeholder, text_new, flags=re.IGNORECASE)
        # For acronyms: avoids matches to acronyms in larger acronyms
        # For case-sensitive acronyms
        for curr_exp in exps_a_yescase:
            # Group processing below prevents substitution of spaces around acr.
            str_tot = str(re.compile(curr_exp).groups)  # Get # of last group
            text_new = re.sub(curr_exp, (r"\1" + placeholder + ("\\" + str_tot)), text_new)
        # For case-insensitive acronyms
        for curr_exp in exps_a_nocase:
            # Group processing below prevents substitution of spaces around acr.
            str_tot = str(re.compile(curr_exp).groups)  # Get # of last group
            text_new = re.sub(curr_exp, (r"\1" + placeholder + ("\\" + str_tot)), text_new, flags=re.IGNORECASE)

        # Print some notes
        if do_verbose:
            print("Keyword regex:\n{0}".format(exps_k))
            print("Acronyms (Case-Sensitive) regex: {0}".format(exp_a_yescase))
            print("Acronyms (Case-Insensitive) regex: {0}".format(exp_a_nocase))
            print("Updated text: {0}".format(text_new))

        # Return updated text
        return text_new

    # Fetch a keyword object that matches the given lookup
    def _fetch_keyword_object(
        keyword_objs, lookup: str, do_raise_emptyerror: bool = True, verbose: bool = False
    ) -> Any | None:
        """Fetch a keyword object

        Given an input lookup string, tries to match it to a stored Keyword instance.

        Parameters
        ----------
        lookup : str
            the input string
        do_raise_emptyerror : bool, optional
            Flag to raise an error, by default True

        Returns
        -------
        Any | None
            the matching Keyword instance

        Raises
        ------
        ValueError
            when no match is found
        """

        # Print some notes
        if verbose:
            logger = setup_logger(__name__)
            logger.info(f"> Running _fetch_keyword_object() for lookup term {lookup}.")

        # Find keyword object that matches to given lookup term
        match = None
        for kobj in keyword_objs:
            # If current keyword object matches, record and stop loop
            if kobj.identify_keyword(lookup)["bool"]:
                match = kobj
                break

        # Throw error if no matching keyword object found
        if match is None:
            errstr = f"No matching keyword object for {lookup}.\n"
            errstr += "Available keyword objects are:\n"
            # just use the names of the keywords
            names = ", ".join(a._get_info("name") for a in keyword_objs)
            errstr += f"{names}\n"

            # Raise error if so requested
            if do_raise_emptyerror:
                raise ValueError(errstr)

        # Return the matching keyword object
        return match
