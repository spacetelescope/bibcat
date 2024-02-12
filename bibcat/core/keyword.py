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

from bibcat import config
from bibcat.core.base import Base


class Keyword(Base):
    """
    Class: Keyword
    Purpose: Store terms, i.e. titles and acronyms, that refer to a mission.
    Initialization Arguments:
        - acronyms [list of strings or Nones, or None (default=None)]:
          - Acronyms of the mission
        - keywords [list of strings]:
          - Titles of the mission
        - do_verbose [bool (default=False)]:
          - Whether or not to print log information and tests
    """

    def __init__(self, keywords, acronyms=None, banned_overlap=[], do_verbose=False):
        """
        Method: __init__
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Initialize instance of Keyword class.
        """
        # Initialize storage
        self._storage = {}
        self._store_info(do_verbose, "do_verbose")
        self._store_info(banned_overlap, "banned_overlap")

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
        if acronyms is not None:
            acronyms_mid = [re.sub(config.exp_nopunct, "", item, flags=re.IGNORECASE) for item in acronyms]
            # Remove all whitespace
            acronyms_mid = [re.sub(" ", "", item) for item in acronyms_mid]
            acronyms_clean = sorted(acronyms_mid, key=(lambda w: len(w)))[::-1]  # Sort by desc. length
            self._store_info(acronyms_clean, key="acronyms")
        else:
            self._store_info([], key="acronyms")

        # Store representative name for this keyword object
        repr_name = (self._get_info("keywords")[::-1] + self._get_info("acronyms"))[
            0
        ]  # Take shortest keyword or longest acronym as repr. name
        self._store_info(repr_name, key="name")

        # Store regular expressions for keywords and acronyms
        exps_k = [(r"\b" + phrase + r"\b") for phrase in self._get_info("keywords")]
        if (acronyms is not None) and (len(acronyms) > 0):
            # Update acronyms to allow optional spaces
            acronyms_upd = [(r"(\.?)( ?)".join(item)) for item in self._get_info("acronyms")]
            # Build regular expression to recognize acronyms
            exp_a = (
                r"(^|[^\.])((\b" + r"\b)|(\b".join([phrase for phrase in acronyms_upd]) + r"\b)(\.?))($|[^A-Z|a-z])"
            )  # Matches any acronym
        else:
            exp_a = None

        self._store_info(exps_k, key="exps_keywords")
        self._store_info(exp_a, key="exp_acronyms")

        return

    # Generate string representation of this class instance
    def __str__(self):
        """
        Method: __str__
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Print instance of Keyword class.
        """
        # Build string of characteristics of this instance
        print_str = (
            "Keyword Object:\n"
            + "Name: {0}\n".format(self.get_name())
            + "Keywords: {0}\n".format(self._get_info("keywords"))
            + "Acronyms: {0}\n".format(self._get_info("acronyms"))
            + "Banned Overlap: {0}\n".format(self._get_info("banned_overlap"))
        )

        # Return the completed string for printing
        return print_str

    # Purpose: Fetch the representative name for this keyword object
    def get_name(self):
        """
        Method: get_name
        Purpose: Return representative name for this instance.
        Arguments: None
        Returns:
          - [str]
        """
        # Fetch and return representative name
        return self._get_info("name")

    # Purpose: Check if text matches to this keyword object
    def is_keyword(self, text, mode=None):
        """
        Method: is_keyword
        Purpose: Return whether or not the given text contains terms (keywords, acronyms) matching this instance.
        Arguments:
          - "text" [str]: The text to search within for terms
        Returns:
          - [bool]
        """
        # Fetch global variables
        exps_k = self._get_info("exps_keywords")
        keywords = self._get_info("keywords")
        exp_a = self._get_info("exp_acronyms")
        banned_overlap_lowercase = self._get_info("banned_overlap_lowercase")
        do_verbose = self._get_info("do_verbose")
        allowed_modes = [None, "keyword", "acronym"]

        # Throw error is specified mode not recognized
        if (mode is not None) and (mode.lower() not in allowed_modes):
            raise ValueError("Err: Invalid mode '{0}'.\nAllowed modes: {1}".format(mode, allowed_modes))

        # Check if this text contains keywords
        if (mode is None) or (mode.lower() == "keyword"):
            check_keywords = any(
                [
                    bool(re.search(item1, text, flags=re.IGNORECASE))
                    for item1 in exps_k
                    if (not any([(ban1 in text.lower()) for ban1 in banned_overlap_lowercase]))
                ]
            )
        else:
            check_keywords = False

        # Check if this text contains acronyms
        if ((mode is None) or (mode.lower() == "acronym")) and (exp_a is not None):
            check_acronyms = bool(re.search(exp_a, text, flags=re.IGNORECASE))
        else:
            check_acronyms = False

        # Print some notes
        if do_verbose:
            print("Keywords: {0}\nKeyword regex:\n{1}".format(keywords, exps_k))
            print("Acronyms: {0}\nAcronym regex:\n{1}".format(acronyms, exp_a))
            print("Keyword bool: {0}\nAcronym bool: {1}".format(check_keywords, check_acronyms))

        # Return booleans
        return check_acronyms or check_keywords

    # Purpose: Replace any text that matches to this keyword object
    def replace_keyword(self, text, placeholder):
        """
        Method: replace_keyword
        Purpose: Replace any substrings within given text that contain terms (keywords, acronyms) matching this instance.
        Arguments:
          - "text" [str]: The text to search within for terms
          - "placeholder" [str]: The substring to replace terms with
        Returns:
          - [str]
        """
        # Fetch global variables
        exps_k = self._get_info("exps_keywords")
        exp_a = self._get_info("exp_acronyms")
        if exp_a is not None:
            exps_a = [exp_a]
        else:
            exps_a = []
        #
        do_verbose = self._get_info("do_verbose")
        text_new = text
        #

        # Replace terms with given placeholder
        # For keywords
        for curr_exp in exps_k:
            text_new = re.sub(curr_exp, placeholder, text_new, flags=re.IGNORECASE)
        # For acronyms: avoids matches to acronyms in larger acronyms
        for curr_exp in exps_a:
            # Group processing below prevents substitution of spaces around acr.
            str_tot = str(re.compile(curr_exp).groups)  # Get # of last group
            text_new = re.sub(curr_exp, (r"\1" + placeholder + ("\\" + str_tot)), text_new, flags=re.IGNORECASE)

        # Print some notes
        if do_verbose:
            print("Keywords: {0}\nKeyword regex:\n{1}".format(keywords, exps_k))
            print("Acronyms: {0}\nAcronym regex:\n{1}".format(acronyms, exp_a))
            print("Updated text: {0}".format(text_new))

        # Return updated text
        return text_new
        return text_new
