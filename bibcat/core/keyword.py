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

    def __init__(self, keywords, acronyms_caseinsensitive, acronyms_casesensitive, banned_overlap, ambig_words, do_not_classify, do_verbose=False):
        """
        Method: __init__
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Initialize instance of Keyword class.
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
        if (acronyms_caseinsensitive is not None):
            acronyms_mid = [re.sub(config.grammar.regex.exp_nopunct, "", item, flags=re.IGNORECASE) for item in acronyms_caseinsensitive]
            # Remove all whitespace
            acronyms_mid = [re.sub(" ", "", item) for item in acronyms_mid]
            acronyms_clean = sorted(acronyms_mid, key=(lambda w: len(w)))[::-1]  # Sort by desc. length
            self._store_info(acronyms_clean, key="acronyms_caseinsensitive")
        else:
            self._store_info([], key="acronyms_caseinsensitive")
        # For case-sensitive acronyms
        if (acronyms_casesensitive is not None):
            acronyms_mid = [re.sub(config.grammar.regex.exp_nopunct, "", item, flags=re.IGNORECASE) for item in acronyms_casesensitive]
            # Remove all whitespace
            acronyms_mid = [re.sub(" ", "", item) for item in acronyms_mid]
            acronyms_clean = sorted(acronyms_mid, key=(lambda w: len(w)))[::-1]  # Sort by desc. length
            self._store_info(acronyms_clean, key="acronyms_casesensitive")
        else:
            self._store_info([], key="acronyms_casesensitive")

        # Store representative name for this keyword object
        repr_name = (self._get_info("keywords")[::-1] + self._get_info("acronyms_casesensitive") + self._get_info("acronyms_caseinsensitive"))[
            0
        ]  # Take shortest keyword or longest acronym as repr. name
        self._store_info(repr_name, key="name")

        # Store regular expression for keywords
        exps_k = [(r"\b" + phrase + r"\b") for phrase in self._get_info("keywords")]

        # Store regular expression for case-insensitive acronyms
        if (acronyms_caseinsensitive is not None) and (len(acronyms_caseinsensitive) > 0):
            # Update acronyms to allow optional spaces
            acronyms_upd = [(r"(\.?)( ?)".join(item)) for item in self._get_info("acronyms_caseinsensitive")]
            # Build regular expression to recognize acronyms
            exp_a_caseinsensitive = (
                r"(^|[^\.])((\b" + r"\b)|(\b".join([phrase for phrase in acronyms_upd]) + r"\b)(\.?))($|[^A-Z|a-z])"
            )  # Matches any acronym
        else:
            exp_a_caseinsensitive = None

        # Store regular expression for case-sensitive acronyms
        if ((acronyms_casesensitive is not None) and (len(acronyms_casesensitive) > 0)):
            # Update acronyms to allow optional spaces
            acronyms_upd = [(r"(\.?)( ?)".join(item)) for item in self._get_info("acronyms_casesensitive")]
            # Build regular expression to recognize acronyms
            exp_a_casesensitive = (
                r"(^|[^\.])((\b" + r"\b)|(\b".join([phrase for phrase in acronyms_upd]) + r"\b)(\.?))($|[^A-Z|a-z])"
            ) # Matches any acronym
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
        """
        Method: __str__
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Print instance of Keyword class.
        """
        # Build string of characteristics of this instance
        print_str = (
            "Keyword Object:\n"
            + f"Name: {self.get_name()}\n"
            + f'Keywords: {self._get_info("keywords")}\n'
            + f'Acronyms (Case-Insensitive): {self._get_info("acronyms_caseinsensitive")}\n'
            + f'Acronyms (Case-Sensitive): {self._get_info("acronyms_casesensitive")}\n'
            + f'Banned Overlap: {self._get_info("banned_overlap")}\n'
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

    # Purpose: Check if text matches to this keyword object; return match inds
    def identify_keyword(self, text, mode=None):
        """
        Method: identify_keyword
        Purpose: Return whether or not the given text contains terms (keywords, acronyms) matching this instance.
        Arguments:
          - "text" [str]: The text to search within for terms
        Returns:
          - [dict] containing:
            - "bool":[bool] - if any matches
            - "charspans":[lists of 2 ints] - character spans of matches
            - "bool_acronym_only":[bool] - if any acronym matches
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
        if ((mode is not None) and (mode.lower() not in allowed_modes)):
            raise ValueError("Err: Invalid mode '{0}'.\nAllowed modes: {1}".format(mode, allowed_modes))

        # Modify text (locally only) to block banned overlap
        text_mod = text
        for curr_ban in banned_overlap_lowercase:
            # Below replaces banned phrases with mask '#' string of custom length
            text_mod = re.sub(
                curr_ban, (lambda x: ("#" * len(x.group()))), text_mod, flags=re.IGNORECASE
            )
        #

        # Check if this text contains keywords
        if ((mode is None) or (mode.lower() == "keyword")):
            set_keywords = [
                list(re.finditer(item1, text_mod, flags=re.IGNORECASE))
                for item1 in exps_k
            ]
            charspans_keywords = [
                (item2.start(), (item2.end()-1))
                for item1 in set_keywords
                for item2 in item1
            ] # Char. span of matches
            check_keywords = any(
                [
                    (len(item) > 0)
                    for item in set_keywords
                ]
            ) #If any matches
        else:
            charspans_keywords = []
            check_keywords = False

        # Check if this text contains case-sensitive acronyms
        if (((mode is None) or (mode.lower() == "acronym")) and (exp_a_yescase is not None)):
            set_acronyms_yescase = list(re.finditer(exp_a_yescase, text_mod))
            charspans_acronyms_yescase = [(item.start(), (item.end()-1))
                                        for item in set_acronyms_yescase]
            check_acronyms_yescase = (len(set_acronyms_yescase) > 0)
        else:
            charspans_acronyms_yescase = []
            check_acronyms_yescase = False

        # Check if this text contains case-insensitive acronyms
        if (((mode is None) or (mode.lower() == "acronym")) and (exp_a_nocase is not None)):
            set_acronyms_nocase = list(
                re.finditer(exp_a_nocase, text_mod, flags=re.IGNORECASE)
            )
            charspans_acronyms_nocase = [
                (item.start(), (item.end()-1))
                for item in set_acronyms_nocase
            ]
            check_acronyms_nocase = (len(set_acronyms_nocase) > 0)
        else:
            charspans_acronyms_nocase = []
            check_acronyms_nocase = False

        # Combine acronym results
        check_acronyms_all = (check_acronyms_nocase or check_acronyms_yescase)
        charspans_acronyms_all = (charspans_acronyms_nocase + charspans_acronyms_yescase)

        # Print some notes
        if do_verbose:
            print("Keywords: {0}\nKeyword regex:\n{1}".format(keywords, exps_k))
            print("Acronyms (Case-Sensitive): {0}\nAcronym regex:\n{1}".format(acronyms_casesensitive, exp_a_yescase))
            print("Acronyms (Case-Insensitive): {0}\nAcronym regex:\n{1}".format(acronyms_caseinsensitive, exp_a_nocase))
            print("Keyword bool: {0}\nAcronym bool: {1}".format(check_keywords, check_acronyms_all))
            print("Keyword char. spans: {0}\nAcronym char. spans: {1}".format(charspans_keywords, charspans_acronyms_all))

        # Return booleans
        return {"bool":(check_acronyms_all or check_keywords),
                "charspans":(charspans_keywords + charspans_acronyms_all),
                "bool_acronym_only":check_acronyms_all}

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
            str_tot = str(re.compile(curr_exp).groups) #Get # of last group
            text_new = re.sub(curr_exp, (r"\1" + placeholder + ("\\" + str_tot)), text_new, flags=re.IGNORECASE)

        # Print some notes
        if do_verbose:
            print("Keyword regex:\n{0}".format(exps_k))
            print("Acronyms (Case-Sensitive) regex: {0}".format(exp_a_yescase))
            print("Acronyms (Case-Insensitive) regex: {0}".format(exp_a_nocase))
            print("Updated text: {0}".format(text_new))

        # Return updated text
        return text_new
