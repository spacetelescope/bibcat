"""
:title: base.py

The class module is a collection of methods that other classes often use.

The primary methods and use cases of _Base include:
* `_get_info`, `_store_info`: Store and retrieve information (values, booleans, etc.)
   for a given class instance.
* `_assemble_keyword_wordchunks`: Build noun chunks containing target mission keywords
   from the text.
* `_check_importance`: Check if some given text contains any important terms
   (where important terms include mission keywords, 1st-person and 3rd-person pronouns,
   a paper citation, etc.).
* `_cleanse_text`: Cleanse some given text, e.g., excessive whitespace and punctuation.
   Can also, e.g., replace citations with an 'Authoretal' placeholder of sorts.
* `_is_pos_word`: Check if some given word (of the NLP type) has a particular part of speech.
* `_search_text`: Search some given text for mission keywords/acronyms
   (e.g., search for "HST").
"""

import re

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import spacy
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore
from nltk.corpus import wordnet  # type: ignore

from bibcat import config

nlp = spacy.load(config.grammar.spacy_language_model)


# TODO - remove this class ; want to separate out workflow classes from object classes ; workflow classes maybe should be functions?
class Base:
    """
    WARNING! This class is *not* meant to be used directly by users.
    -
    Class: _Base
    Purpose:
     - Container for common underlying methods used in other classes.
     - Purely meant to be inherited by other classes.
    -
    """

    def __init__(self):
        """
        Method: __init__
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Initializes instance of _Base class.
        """
        # Initialize storage
        self._storage = {}

        return

    # Retrieve specified data via given key
    # TODO - do we really need this? no
    def _get_info(self, key: str, do_flag_hidden=False):
        """
        Method: _get_info
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Fetches values, etc., for this class instance from storage.
        """
        # Attempt to retrieve data stored under the given key
        try:
            return self._storage[key]
        # Throw helpful error if retrieval attempt failed
        except KeyError:
            # Return a specialized testing error, if likely hidden method called
            if do_flag_hidden:
                errstr = (
                    "Whoa there. This error likely happened because"
                    + " you are testing or exploring a hidden ('_') method."
                    + " If so, you likely need to pass in this parameter -"
                    + " '{0}' - as an input to the method."
                ).format(key)
            # Otherwise, return generic error for available stored data
            else:
                errstr = (
                    "Whoa there. Looks like you requested data from a"
                    + " key ({1}) that does not exist. Available keys"
                    + " are:\n{0}"
                ).format(sorted(self._storage.keys()), key)

            # Raise the custom error
            raise KeyError(errstr)
        return

    # Store given data into class instance in a unified way
    # TODO - do we really need this? no
    def _store_info(self, data, key):
        """
        Method: _store_info
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Stores values, etc., for this class instance in storage.
        """
        # Store the data into underlying dictionary
        self._storage[key] = data
        return

    # Assemble wordchunks containing keywords from given text
    # TODO - remove this entirely - this is for rules
    def _assemble_keyword_wordchunks(
        self, text, keyword_objs, do_include_verbs=False, do_include_brackets=False, do_verbose=False
    ):  # noqa: C901
        """
        Method: _assemble_keyword_wordchunks
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Assembles noun chunks around any keyword terms within given text.
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
            if do_verbose:
                print("Current sentence: '{0}'".format(curr_sent))
                print("Indices of lookups in sent.: '{0}'".format(set_inds))
            # Build wordchunks from indices of current sentence
            last_ind = -np.inf
            for curr_start in set_inds:
                # Print some notes
                if do_verbose:
                    print("\n-Building wordchunk from {0}: {1}:".format(curr_start, curr_sent[curr_start]))
                # Skip if this index has already been surpassed
                if curr_start <= last_ind:
                    continue
                curr_wordtext = [curr_sent[curr_start].text]

                # Build wordchunk from accumulating nouns on the left
                for ii in range(0, curr_start)[::-1]:  # Do not include start here
                    # Store index if noun or numeral or adjective
                    check_noun = self._is_pos_word(word=curr_sent[ii], pos="NOUN")
                    check_adj = self._is_pos_word(word=curr_sent[ii], pos="ADJECTIVE")
                    check_num = self._is_pos_word(word=curr_sent[ii], pos="NUMBER")
                    check_pos = self._is_pos_word(word=curr_sent[ii], pos="POSSESSIVE")
                    check_dash = curr_sent[ii].text == "-"
                    check_imp = self._check_importance(
                        curr_sent[ii].text, keyword_objs=keyword_objs, version_NLP=curr_sent[ii]
                    )["bools"]["is_any"]
                    # Include punctuation, if so requested
                    if do_include_brackets:
                        check_brackets = self._is_pos_word(word=curr_sent[ii], pos="BRACKET")
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
                    check_noun = self._is_pos_word(word=curr_sent[ii], pos="NOUN")
                    check_adj = self._is_pos_word(word=curr_sent[ii], pos="ADJECTIVE")
                    check_num = self._is_pos_word(word=curr_sent[ii], pos="NUMBER")
                    check_pos = self._is_pos_word(word=curr_sent[ii], pos="POSSESSIVE")
                    check_imp = self._check_importance(
                        curr_sent[ii].text, keyword_objs=keyword_objs, version_NLP=curr_sent[ii]
                    )["bools"]["is_any"]
                    check_dash = curr_sent[ii].text == "-"
                    # Include brackets, if so requested
                    if do_include_brackets:
                        check_brackets = self._is_pos_word(word=curr_sent[ii], pos="BRACKET")
                    else:
                        check_brackets = False

                    # Tack on verb check if requested (e.g., to cover noun-verbs)
                    if do_include_verbs:  # E.g., ambig. 'Hubble-imaged data'
                        check_verb = self._is_pos_word(word=curr_sent[ii], pos="VERB")
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
                curr_str_fin = self._cleanse_text(" ".join(curr_wordtext), do_streamline_etal=False)
                list_wordchunks.append(nlp(curr_str_fin))

                # Print some notes
                if do_verbose:
                    print(
                        "All wordchunks so far: {0}\nNewest wordchunk: {1}".format(list_wordchunks, list_wordchunks[-1])
                    )
                    print(
                        "pos_ values: {0}\ndep_ values: {1}\ntag_ values: {2}".format(
                            [item.pos_ for item in list_wordchunks[-1]],
                            [item.dep_ for item in list_wordchunks[-1]],
                            [item.tag_ for item in list_wordchunks[-1]],
                        )
                    )

        # Return the assembled wordchunks
        if do_verbose:
            print("Assembled keyword wordchunks:\n{0}".format(list_wordchunks))

        return list_wordchunks

    # Plot rectangular confusion matrix for given data and labels
    # TODO - move this under the Performance Class
    def _ax_confusion_matrix(
        self,
        matr,
        ax,
        x_labels,
        y_labels,
        x_title,
        y_title,
        cbar_title,
        ax_title,
        is_norm,
        minmax_inds=None,
        cmap=plt.cm.BuPu,
        fontsize=16,
        ticksize=16,
        valsize=14,
        y_rotation=30,
        x_rotation=30,
    ):
        """
        Method: _ax_confusion_matrix
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Plots confusion matrix within given axis.
        """

        # Set global variables
        if is_norm:
            vmin = 0
            vmax = 1
        elif minmax_inds is not None:
            vmin = 0
            # Ignore non-target verdicts to avoid color spikes scaling if present
            tmpmatr = matr.copy()
            # Remove max scaling for non-target classifs along y-axis
            for yind in minmax_inds["y"]:
                # Remove non-target classifications from max consideration
                tmpmatr[yind, :] = -1
            # Remove max scaling for non-target classifs along x-axis
            for xind in minmax_inds["x"]:
                # Remove non-target classifications from max consideration
                tmpmatr[:, xind] = -1
            #
            vmax = tmpmatr.max()
        else:
            vmin = 0
            vmax = matr.max()

        # Plot the confusion matrix and colorbar
        image = ax.imshow(matr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)

        # Fetch the matrix dimensions
        ydim = matr.shape[0]
        xdim = matr.shape[1]

        # Write in the values
        for yy in range(0, ydim):
            for xx in range(0, xdim):
                # Set current text color based on background grayscale value
                if is_norm:
                    curr_gray = np.mean(cmap(matr[yy, xx])[0:3])
                else:
                    curr_gray = np.mean(cmap(matr[yy, xx] / vmax)[0:3])

                if curr_gray <= 0.6:
                    curr_color = "white"
                else:
                    curr_color = "black"

                # Write current text
                if is_norm:
                    plt.text(
                        xx,
                        yy,
                        "{0:.3f}".format(matr[yy, xx]),
                        color=curr_color,
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=valsize,
                    )
                else:
                    plt.text(
                        xx,
                        yy,
                        "{0:.0f}".format(matr[yy, xx]),
                        color=curr_color,
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=valsize,
                    )

        # Generate the colorbar
        cbarax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(image, cax=cbarax, extend="max")
        cbar.ax.tick_params(labelsize=valsize)

        # Set the tick and axis labels
        ax.tick_params(axis="both", which="both", labelsize=ticksize)
        ax.set_xticks(np.arange(0, xdim, 1))
        ax.set_xticklabels([item.title() for item in x_labels], rotation=x_rotation)
        ax.set_yticks(np.arange(0, ydim, 1))
        ax.set_yticklabels([item.title() for item in y_labels], rotation=y_rotation)
        ax.set_xlabel(x_title, fontsize=fontsize)
        ax.set_ylabel(y_title, fontsize=fontsize)

        # Set the subplot title
        ax.set_title("{0}\n{1}".format(ax_title, cbar_title), fontsize=fontsize)

        # Exit the method
        return

    # Determine if given text is important (e.g., is a keyword)
    # TODO - move this to grammar class
    def _check_importance(
        self, text, include_Ipronouns=True, include_terms=True, include_etal=True, keyword_objs=None, version_NLP=None
    ):
        """
        Method: _check_importance
        WARNING! This method is *not* meant to be used directly by users.
        Purpose:
         - Checks if given text contains any important terms.
           - Important terms includes keywords, 1st,3rd person pronouns, etc.
         - Returns dictionary of bools for presence/absence of important terms.
        """

        # Extract the NLP version of this text, if not given
        if version_NLP is None:
            version_NLP = nlp(text)

        # Ensure NLP version is iterable
        if not hasattr(version_NLP, "__iter__"):
            version_NLP = [version_NLP]

        # Extract keyword objects from storage, if not given
        if keyword_objs is None:
            try:
                keyword_objs = [self._get_info("keyword_obj", do_flag_hidden=True)]
            except KeyError:
                keyword_objs = self._get_info("keyword_objs", do_flag_hidden=True)

        # Cleanse and streamline the given text
        text = self._cleanse_text(text, do_streamline_etal=True)

        # Initialize container for booleans
        dict_results = {}

        ##Check if text contains keywords, acronyms, important terms, etc
        # For target keywords and acronyms
        tmp_res = self._search_text(text=text, keyword_objs=keyword_objs)
        dict_results["is_keyword"] = tmp_res["bool"]
        charspans_keyword = tmp_res["charspans"]

        # Check for first-person pronouns, if requested
        if include_Ipronouns:
            list_pos_pronoun = config.grammar.speech.pos_pronoun
            nlp_lookup_person = config.grammar.nlp_lookup_person
            check_pronounI = any(
                [
                    (
                        (item.pos_ in list_pos_pronoun)  # Pronoun
                        and ("1" in item.morph.get(nlp_lookup_person))
                    )
                    for item in version_NLP
                ]
            )  # Check if 1st-person
            dict_results["is_pron_1st"] = check_pronounI
        else:  # Otherwise, remove pronoun contribution
            dict_results["is_pron_1st"] = False

        # Check for special terms, if requested
        if include_terms:
            list_pos_pronoun = config.grammar.speech.pos_pronoun
            nlp_lookup_person = config.grammar.nlp_lookup_person
            special_synsets_fig = config.grammar.special_synsets_fig

            # For 'they' pronouns
            check_terms_they = any(
                [
                    (
                        (item.pos_ in list_pos_pronoun)  # Pronoun
                        and ("3" in item.morph.get(nlp_lookup_person))
                    )
                    for item in version_NLP
                ]
            )  # Check if 3rd-person
            # For 'figure', etc, terms
            check_terms_fig = any(
                [
                    (item2.name() in special_synsets_fig)
                    for item1 in version_NLP
                    for item2 in wordnet.synsets(item1.text)
                ]
            )  # Check if any words have figure, etc, synsets

            # Store the booleans
            dict_results["is_pron_3rd"] = check_terms_they
            dict_results["is_term_fig"] = check_terms_fig
        else:  # Otherwise, remove term contribution
            dict_results["is_pron_3rd"] = False
            dict_results["is_term_fig"] = False

        # Check for etal terms, if requested
        if include_etal:
            exp = config.grammar.regex.exp_etal_cleansed  # Reg.ex. to find cleansed et al
            check_etal = bool(re.search(exp, text, flags=re.IGNORECASE))
            dict_results["is_etal"] = check_etal
        else:  # Otherwise, remove term contribution
            dict_results["is_etal"] = False

        # Store overall status of if any booleans set to True
        dict_results["is_any"] = any([dict_results[key] for key in dict_results])

        # Return the booleans
        return {"bools": dict_results, "charspans_keyword": charspans_keyword}

    # Cleanse given (any length) string of extra whitespace, dashes, etc.
    # TODO - move this to Keyword class
    def _cleanse_text(self, text, do_streamline_etal):
        """
        Method: _cleanse_text
        WARNING! This method is *not* meant to be used directly by users.
        Purpose:
         - Cleanse text of extra whitespace, punctuation, etc.
         - Replace paper citations (e.g. 'et al.') with uniform placeholder.
        """

        # Extract global punctuation expressions
        set_apostrophe = config.grammar.regex.set_apostrophe
        set_punctuation = config.grammar.regex.set_punctuation
        exp_punctuation = config.grammar.regex.exp_punctuation
        set_openbrackets = config.grammar.regex.set_openbrackets
        set_closebrackets = config.grammar.regex.set_closebrackets

        # Remove any starting punctuation
        text = re.sub((r"^(" + "|".join(exp_punctuation) + r")"), "", text)  # Remove starting punct.

        # Remove extra whitespace in general
        text = re.sub("  +", " ", text)  # Removes spaces > length=1

        # Remove excessive whitespace around punctuation
        # For opening brackets
        tmp_exp_inner = "\\" + "|\\".join(set_openbrackets)
        text = re.sub(("(" + tmp_exp_inner + ") ?"), r"\1", text)
        # For closing brackets and punctuation
        tmp_exp_inner = "\\" + "|\\".join((set_closebrackets + set_punctuation))
        text = re.sub((" ?(" + tmp_exp_inner + ")"), r"\1", text)
        # For apostrophes
        tmp_exp_inner = "\\" + "|\\".join(set_apostrophe)
        text = re.sub((" ?(" + tmp_exp_inner + ") ?"), r"\1", text)

        # Remove empty brackets and doubled-up punctuation
        # Extract ids of empty brackets and doubled-up punctuation
        ids_rem = [
            ii
            for ii in range(0, (len(text) - 1))
            if (
                ((text[ii] in set_openbrackets) and (text[ii + 1] in set_closebrackets))  # Empty brackets
                or ((text[ii] in set_punctuation) and (text[ii + 1] in set_punctuation))
            )
        ]  # Double punct.
        # Remove the characters (in reverse order!) at the identified ids
        for ii in sorted(ids_rem)[::-1]:  # Reverse-sorted
            text = text[0:ii] + text[(ii + 1) : len(text)]

        # Remove 'et al.' period phrasing (can mess up sentence splitter later)
        text = re.sub(r"\bet al\b\.", "et al", text)
        #

        # Replace pesky "Author & Author (date)", et al., etc., wordage
        if do_streamline_etal:
            # Adapted from:
            # https://regex101.com/r/xssPEs/1
            # https://stackoverflow.com/questions/63632861/
            #                           python-regex-to-get-citations-in-a-paper
            # bit_author = r"(?:[A-Z][A-Za-z'`-]+)"
            bit_author = r"(?:(\b[A-Z]\. )*[A-Z][A-Za-z'`-]+)"
            bit_etal = r"(?:et al\.?)"
            # bit_additional = f"(?:,? (?:(?:and |& )?{bit_author}|{bit_etal}))"
            bit_additional = f"(?: (?:(?:and |& ){bit_author}|{bit_etal}))"
            # Regular expressions for years (with or without brackets)
            exp_year_yesbrackets = (
                r"( (\(|\[|\{)" + r"([0-9]{4,4}|[0-9]{2,2})" + r"((,|;) ?([0-9]{4,4}|[0-9]{2,2}))*" + r"(\)|\]|\}))"
            )
            exp_year_nobrackets = r" " + r"([0-9]{4,4}|[0-9]{2,2})" + r"((,|;) ?([0-9]{4,4}|[0-9]{2,2}))*"
            # Regular expressions for citations (with or without brackets)
            exp_cites_yesbrackets = (
                r"(\(|\[|\{)"
                + rf"{bit_author}{bit_additional}*{exp_year_nobrackets}"
                + (r"((,|;) " + rf"{bit_author}{bit_additional}*{exp_year_nobrackets}" + r")*")
                + r"(\)|\]|\})"
            )
            exp_cites_nobrackets = rf"{bit_author}{bit_additional}*{exp_year_yesbrackets}"

            # Replace not-bracketed citations or remove bracketed citations
            text = re.sub(exp_cites_yesbrackets, "", text)
            text = re.sub(exp_cites_nobrackets, config.textprocessing.placeholder_author, text)

            # Replace singular et al. (e.g. SingleAuthor et al.) wordage as well
            # text = re.sub(r" et al\b\.?", "etal", text)
            text = re.sub(
                r"\b([A-Z]\. )*(\b[A-Z][A-Z|a-z]+) et al\b\.?", config.textprocessing.placeholder_author, text
            )

            # Collapse adjacent author terms
            text = re.sub(
                r"{0}((,|;|(,? and))( )+{0})+".format(config.textprocessing.placeholder_author),
                config.textprocessing.placeholder_author,
                text,
            )

        # Remove starting+ending whitespace
        text = text.lstrip().rstrip()

        # Return cleansed text
        return text

    # Return boolean for if given word (NLP type word) is of conjoined given part of speech
    # TODO - move this to grammar class
    def _is_pos_conjoined(self, word, pos):
        """
        Method: _is_pos_conjoined
        WARNING! This method is *not* meant to be used directly by users.
        Purpose:
         - Determine if original part-of-speech (p.o.s.) for given word (if conjoined) matches given p.o.s.
        """

        # Return False if pos is aux (which may not be conjoined)
        if pos in config.grammar.speech.pos_aux:
            return False

        # Check if this word is conjoined
        is_conjoined = self._is_pos_word(word=word, pos="CONJOINED")
        if not is_conjoined:  # Terminate early if not conjoined
            return False

        # Check if this word has any previous nodes
        word_ancestors = list(word.ancestors)  # All previous nodes leading to word
        if len(word_ancestors) == 0:  # Terminate early if no previous nodes
            return False

        # Follow chain upward to find if original p.o.s. matches given p.o.s.
        for pre_node in word_ancestors:
            # Continue if previous word also conjoined
            if self._is_pos_word(word=pre_node, pos="CONJOINED"):
                continue

            # Otherwise, check if original p.o.s. matches given p.o.s.
            return self._is_pos_word(word=pre_node, pos=pos)

        # If no original p.o.s. found, throw an error
        raise ValueError("Err: No original p.o.s. for conjoined word {0}!\n{1}".format(word, word_ancestors))

    # Return boolean for if given word (NLP type word) is of given part of speech
    # TODO - move this to Grammar class
    def _is_pos_word(self, word, pos, keyword_objs=None, do_verbose=False):  # noqa: C901
        """
        Method: _is_pos_word
        WARNING! This method is *not* meant to be used directly by users.
        Purpose:
         - Return if the given word (of NLP type) is of the given part-of-speech.
        """

        ##Load global variables
        # word_i = word.i  # Index
        word_dep = word.dep_  # dep label
        word_pos = word.pos_  # p.o.s. label
        word_tag = word.tag_  # tag label
        word_text = word.text  # Text version of word
        word_ancestors = list(word.ancestors)  # All previous nodes leading to word

        # Print some notes
        if do_verbose:
            print("Running _is_pos_word for: {0}".format(word))
            print("dep_: {0}\npos_: {1}\ntag_: {2}".format(word_dep, word_pos, word_tag))
            print("Node head: {0}\nSentence: {1}".format(word.head, word.sent))
            print("Node lefts: {0}\nNode rights: {1}".format(list(word.lefts), list(word.rights)))

        # Check if given word is of given part-of-speech
        # Identify roots
        if pos in ["ROOT"]:
            check_all = word_dep in config.grammar.speech.dep_root

        # Identify verbs
        elif pos in ["VERB"]:
            check_posaux = word_pos in config.grammar.speech.pos_aux
            # check_isrightword = (len(list(word.rights)) > 0)
            # NOTE: 'isrightword' check, since aux-verb would have right word(s)
            # (E.g., 'The star is observable')

            check_root = self._is_pos_word(word=word, pos="ROOT")
            check_conj = self._is_pos_conjoined(word=word, pos=pos)
            check_ccomp = word_dep in config.grammar.speech.dep_ccomp
            check_tag = word_tag in config.grammar.speech.tag_verb_any
            check_pos = word_pos in config.grammar.speech.pos_verb
            check_dep = word_dep in config.grammar.speech.dep_verb
            tag_approved = (
                config.grammar.speech.tag_verb_present
                + config.grammar.speech.tag_verb_past
                + config.grammar.speech.tag_verb_future
            )
            check_approved = word_tag in tag_approved

            # For ambiguous adjectival modifier sentences
            # (E.g. "Hubble calibrated data")
            check_nounroot = (
                (len(word_ancestors) > 0)
                and self._is_pos_word(word=word.head, pos="ROOT")
                and self._is_pos_word(word=word.head, pos="NOUN")
            )
            check_amod = word_dep in config.grammar.speech.dep_adjective
            check_islefts = len(list(word.lefts)) > 0
            check_valid_amod = check_nounroot and check_amod and check_islefts

            check_all = (
                (
                    (
                        (check_dep or check_root or check_conj or check_ccomp)
                        # and check_pos and check_tag)
                        and check_tag
                    )
                    or (check_root and check_posaux)
                )
                # or (check_isrightword and check_posaux))
                or (check_valid_amod)
            ) and check_approved

        # Identify useless words
        elif pos in ["USELESS"]:
            # Fetch keyword objects
            if keyword_objs is None:
                try:
                    keyword_objs = [self._get_info("keyword_obj")]
                except KeyError:
                    keyword_objs = self._get_info("keyword_objs", do_flag_hidden=True)

            # Check p.o.s. components
            check_tag = word_tag in config.grammar.speech.tag_useless
            check_dep = word_dep in config.grammar.speech.dep_useless
            check_pos = word_pos in config.grammar.speech.pos_useless
            check_use = self._check_importance(word_text, version_NLP=word, keyword_objs=keyword_objs)["bools"][
                "is_any"
            ]  # Useful
            check_root = self._is_pos_word(word=word, pos="ROOT")
            check_neg = self._is_pos_word(word=word, pos="NEGATIVE")
            check_subj = self._is_pos_word(word=word, pos="SUBJECT")
            check_all = (check_tag and check_dep and check_pos) and (
                not (check_use or check_neg or check_subj or check_root)
            )

        # Identify subjects
        elif pos in ["SUBJECT"]:
            check_noun = self._is_pos_word(word=word, pos="NOUN")
            check_adj = self._is_pos_word(word=word, pos="ADJECTIVE")
            check_obj = self._is_pos_word(word=word, pos="BASE_OBJECT")

            # Determine if to left of verb or root, if applicable
            is_leftofverb = False
            if len(word_ancestors) > 0:
                tmp_verb = self._is_pos_word(word=word_ancestors[0], pos="VERB")
                # tmp_root = self._is_pos_word(word=word_ancestors[0], pos="ROOT")
                # if (tmp_verb or tmp_root):
                if tmp_verb:
                    is_leftofverb = word in word_ancestors[0].lefts

            # Determine if conjoined to subject, if applicable
            is_conjsubj = self._is_pos_conjoined(word, pos=pos)
            is_root = self._is_pos_word(word=word, pos="ROOT")
            check_dep = word_dep in config.grammar.speech.dep_subject
            check_all = (
                (check_dep and is_leftofverb)
                or (is_conjsubj)
                or (check_noun and is_root)
                or ((check_noun or check_adj) and is_leftofverb)
            ) and (not check_obj)

        # Identify prepositions
        elif pos in ["PREPOSITION"]:
            check_dep = word_dep in config.grammar.speech.dep_preposition
            check_pos = word_pos in config.grammar.speech.pos_preposition
            check_tag = word_tag in config.grammar.speech.tag_preposition
            check_prepaux = (
                (word_dep in config.grammar.speech.dep_aux)
                and (word_pos in config.grammar.speech.pos_aux)
                and (check_tag)
            )  # For e.g. mishandled 'to'
            check_all = (check_dep and check_pos and check_tag) or (check_prepaux)

        # Identify base objects (so either direct or prep. objects)
        elif pos in ["BASE_OBJECT"]:
            check_dep = word_dep in config.grammar.speech.dep_object
            check_noun = self._is_pos_word(word=word, pos="NOUN")
            check_all = check_noun and check_dep

        # Identify direct objects
        elif pos in ["DIRECT_OBJECT"]:
            check_baseobj = self._is_pos_word(word=word, pos="BASE_OBJECT")
            is_conjdirobj = self._is_pos_conjoined(word, pos=pos)
            # Check preceding term is a verb
            check_afterprep = False
            check_afterverb = False
            for pre_node in word_ancestors:
                # If preceding preposition found first
                if self._is_pos_word(word=pre_node, pos="PREPOSITION"):
                    check_afterprep = True
                    break
                # If preceding verb found first
                elif self._is_pos_word(word=pre_node, pos="VERB"):
                    check_afterverb = True
                    break

            check_all = ((not check_afterprep) and (check_afterverb) and (check_baseobj)) or is_conjdirobj

        # Identify prepositional objects
        elif pos in ["PREPOSITION_OBJECT"]:
            check_baseobj = self._is_pos_word(word=word, pos="BASE_OBJECT")
            is_conjprepobj = self._is_pos_conjoined(word, pos=pos)
            # Check if this word follows preposition
            check_objprep = False
            for pre_node in word_ancestors:
                # If preceding preposition found first
                if self._is_pos_word(word=pre_node, pos="PREPOSITION"):
                    pre_pre_node = list(pre_node.ancestors)[0]
                    # Ensure prepositional object instead of prep. subject
                    check_objprep = not self._is_pos_word(word=pre_pre_node, pos="SUBJECT")
                    break
                # If preceding verb found first
                elif self._is_pos_word(word=pre_node, pos="VERB"):
                    check_objprep = False
                    break

            check_all = is_conjprepobj or (check_baseobj and check_objprep)

        # Identify prepositional subjects
        elif pos in ["PREPOSITION_SUBJECT"]:
            check_obj = self._is_pos_word(word=word, pos="BASE_OBJECT")
            is_conjprepsubj = self._is_pos_conjoined(word, pos=pos)
            # Check if this word follows preposition
            check_subjprep = False
            for pre_node in word_ancestors:
                # If preceding preposition found first
                if self._is_pos_word(word=pre_node, pos="PREPOSITION"):
                    pre_pre_node = list(pre_node.ancestors)[0]
                    # Ensure prepositional subject instead of prep. object
                    check_subjprep = self._is_pos_word(word=pre_pre_node, pos="SUBJECT")
                    break
                # If preceding verb found first
                elif self._is_pos_word(word=pre_node, pos="VERB"):
                    check_subjprep = False
                    break

            check_all = is_conjprepsubj or (check_obj and check_subjprep)

        # Identify markers
        elif pos in ["MARKER"]:
            check_dep = word_dep in config.grammar.speech.dep_marker
            check_tag = word_tag in config.grammar.speech.tag_marker
            check_marker = check_dep or check_tag
            # Check if subject marker after non-root verb
            is_notroot = len(word_ancestors) > 0
            is_afterroot = is_notroot and self._is_pos_word(word=word_ancestors[0], pos="ROOT")
            check_subjmark = False
            if (is_notroot) and (not is_afterroot):
                check_subj = self._is_pos_word(word=word, pos="SUBJECT")
                check_det = self._is_pos_word(word=word, pos="DETERMINANT")
                check_subjmark = check_det and check_subj

            check_all = (check_marker or check_subjmark) and (not is_afterroot)

        # Identify improper X-words (for improper sentences)
        elif pos in ["X"]:
            check_dep = word_dep in config.grammar.speech.dep_xpos
            check_pos = word_pos in config.grammar.speech.pos_xpos
            check_all = check_dep or check_pos

        # Identify conjoined words
        elif pos in ["CONJOINED"]:
            check_conj = word_dep in config.grammar.speech.dep_conjoined
            check_appos = word_dep in config.grammar.speech.dep_appos
            check_det = word_tag in config.grammar.speech.tag_determinant
            check_all = (check_conj or check_appos) and (not check_det)

        # Identify determinants
        elif pos in ["DETERMINANT"]:
            check_pos = word_pos in config.grammar.speech.pos_determinant
            check_tag = word_tag in config.grammar.speech.tag_determinant
            check_all = check_pos and check_tag

        # Identify aux
        elif pos in ["AUX"]:
            check_dep = word_dep in config.grammar.speech.dep_aux
            check_pos = word_pos in config.grammar.speech.pos_aux
            check_prep = word_tag in config.grammar.speech.tag_preposition
            check_num = word_tag in config.grammar.speech.tag_number

            tags_approved = (
                config.grammar.speech.tag_verb_past
                + config.grammar.speech.tag_verb_present
                + config.grammar.speech.tag_verb_future
                + config.grammar.speech.tag_verb_purpose
            )
            check_approved = word_tag in tags_approved

            check_all = (check_dep and check_pos and check_approved) and (not (check_prep or check_num))

        # Identify nouns
        elif pos in ["NOUN"]:
            check_pos = word_pos in config.grammar.speech.pos_noun
            check_det = word_tag in config.grammar.speech.tag_determinant
            check_all = check_pos and (not check_det)

        # Identify pronouns
        elif pos in ["PRONOUN"]:
            check_tag = word_tag in config.grammar.speech.tag_pronoun
            check_pos = word_pos in config.grammar.speech.pos_pronoun
            check_all = check_tag or check_pos

        # Identify adjectives
        elif pos in ["ADJECTIVE"]:
            check_adjverb = (
                (word_dep in config.grammar.speech.dep_adjective)
                and (word_pos in config.grammar.speech.pos_verb)
                and (word_tag in config.grammar.speech.tag_verb_any)
            )
            check_pos = word_pos in config.grammar.speech.pos_adjective
            check_tag = word_tag in config.grammar.speech.tag_adjective
            check_all = check_tag or check_pos or check_adjverb

        # Identify  conjunctions
        elif pos in ["CONJUNCTION"]:
            check_pos = word_pos in config.grammar.speech.pos_conjunction
            check_tag = word_tag in config.grammar.speech.tag_conjunction
            check_all = check_pos and check_tag

        # Identify passive verbs and aux
        elif pos in ["PASSIVE"]:
            check_dep = word_dep in config.grammar.speech.dep_verb_passive
            check_all = check_dep

        # Identify negative words
        elif pos in ["NEGATIVE"]:
            check_dep = word_dep in config.grammar.speech.dep_negative
            check_all = check_dep

        # Identify punctuation
        elif pos in ["PUNCTUATION"]:
            check_punct = word_dep in config.grammar.speech.dep_punctuation
            check_letter = bool(re.search(".*[a-z|0-9].*", word_text, flags=re.IGNORECASE))
            check_all = check_punct and (not check_letter)

        # Identify punctuation
        elif pos in ["BRACKET"]:
            check_brackets = word_tag in (config.grammar.speech.tag_brackets)
            check_all = check_brackets

        # Identify possessive markers
        elif pos in ["POSSESSIVE"]:
            check_possessive = word_tag in config.grammar.speech.tag_possessive
            check_all = check_possessive

        # Identify numbers
        elif pos in ["NUMBER"]:
            check_number = word_pos in config.grammar.speech.pos_number
            check_all = check_number

        # Otherwise, raise error if given pos is not recognized
        else:
            raise ValueError("Err: {0} is not a recognized part of speech.".format(pos))

        # Print some notes
        if do_verbose:
            print("Is pos={0}? {1}\n-".format(pos, check_all))
        # Return the final verdict
        return check_all

    # Search text for given keywords and acronyms and return metric
    # TODO - move this to grammar class
    def _search_text(self, text, keyword_objs, do_verbose=False):
        """
        Method: _search_text
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Return boolean for whether or not given text contains keywords/acronyms from given keyword objects.
        """
        # Check if keywords and/or acronyms present in given text
        tmp_res = [item.identify_keyword(text) for item in keyword_objs]
        check_keywords = any([item["bool"] for item in tmp_res])
        charspans_keywords = []
        for ii in range(0, len(tmp_res)):
            charspans_keywords += tmp_res[ii]["charspans"]

        # Print some notes
        if do_verbose:
            # Extract global variables
            keywords = [item2 for item1 in keyword_objs for item2 in item1._get_info("keywords")]
            acronyms = [
                item2
                for item1 in keyword_objs
                for item2 in item1._get_info("acronyms_casesensitive") + item1._get_info("acronyms_caseinsensitive")
            ]

            print("Completed _search_text().")
            print("Keywords={0}\nAcronyms={1}".format(keywords, acronyms))
            print("Boolean: {0}".format(check_keywords))
            print("Char. Spans: {0}".format(charspans_keywords))

        # Return boolean result
        return {"bool": check_keywords, "charspans": charspans_keywords}

    # Write given text to given filepath
    # TODO - move this to performance class
    @staticmethod
    def _write_text(text: str, filepath: str) -> None:
        """
        Method: _write_text
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Write given text to given filepath.
        """

        # Write text to file
        with open(filepath, "w") as openfile:
            openfile.write(text)
        # Exit the method
        return
