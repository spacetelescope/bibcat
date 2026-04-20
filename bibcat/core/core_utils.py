"""
:title: core_utils.py

The class module is a collection of methods that other classes often use.

The primary methods and use cases of core_utils include:
* `check_importance`: Check if some given text contains any important terms
   (where important terms include mission keywords, 1st-person and 3rd-person pronouns,
   a paper citation, etc.).
* `cleanse_text`: Cleanse some given text, e.g., excessive whitespace and punctuation.
   Can also, e.g., replace citations with an 'Authoretal' placeholder of sorts.
* `is_pos_conjoined`: Check if a conjoined word's original part of speech matches
   a given POS tag by traversing the dependency tree to the root of the conjunction chain.
* `is_pos_word`: Check if some given word (of the NLP type) has a particular part of speech.
* `search_text`: Search some given text for mission keywords/acronyms
   (e.g., search for "HST").
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional, Union

import spacy
import spacy.tokens
from nltk.corpus import wordnet  # type: ignore

from bibcat import config
from bibcat.utils.logger_config import setup_logger

if TYPE_CHECKING:
    from bibcat.core.keyword import Keyword

nlp = spacy.load(config.grammar.spacy_language_model)

logger = setup_logger(__name__)
logger.setLevel(config.logging.level)


def check_importance(
    text: str,
    keyword_objs: list[Keyword],
    include_Ipronouns: bool = True,
    include_terms: bool = True,
    include_etal: bool = True,
    version_NLP: Optional[Union[spacy.tokens.Doc, list]] = None,
) -> dict:
    """
    Check if given text contains any important terms.

    Evaluates the presence of keywords, acronyms, first- and third-person
    pronouns, figure-related terms, and "et al." expressions. Returns a
    dictionary of boolean flags for each category.

    Parameters
    ----------
    text : str
        The input text to evaluate.
    keyword_objs : list
        Collection of keyword objects used to search for keywords and acronyms.
    include_Ipronouns : bool, optional
        Whether to check for first-person pronouns. Default is True.
    include_terms : bool, optional
        Whether to check for third-person pronouns and figure-related terms.
        Default is True.
    include_etal : bool, optional
        Whether to check for "et al." expressions. Default is True.
    version_NLP : spacy.tokens.Doc or iterable, optional
        Pre-computed NLP representation of ``text``. If None, it is computed
        internally. Default is None.

    Returns
    -------
    dict
        A dictionary with the following keys:

        - ``"bools"`` : dict of bool
            Boolean flags for each importance category:

            - ``"is_keyword"`` -- text matches a keyword or acronym.
            - ``"is_pron_1st"`` -- text contains a first-person pronoun.
            - ``"is_pron_3rd"`` -- text contains a third-person pronoun.
            - ``"is_term_fig"`` -- text contains a figure-related term.
            - ``"is_etal"`` -- text contains an "et al." expression.
            - ``"is_any"`` -- any of the above flags is True.

        - ``"charspans_keyword"`` : list
            Character spans of matched keywords within the text.
    """

    # Extract the NLP version of this text, if not given
    if version_NLP is None:
        version_NLP = nlp(text)

    # Ensure NLP version is iterable
    if not hasattr(version_NLP, "__iter__"):
        version_NLP = [version_NLP]

    # Cleanse and streamline the given text
    text = cleanse_text(text, do_streamline_etal=True)

    # Initialize container for booleans
    dict_results = {}

    ##Check if text contains keywords, acronyms, important terms, etc
    # For target keywords and acronyms
    tmp_res = search_text(text=text, keyword_objs=keyword_objs)
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
            [(item2.name() in special_synsets_fig) for item1 in version_NLP for item2 in wordnet.synsets(item1.text)]
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


def cleanse_text(text: str, do_streamline_etal: bool) -> str:
    """
    Cleanse a string of extra whitespace, punctuation, and citation expressions.

    Removes leading punctuation, normalizes whitespace, strips empty brackets
    and doubled punctuation, and optionally replaces author citation patterns
    (e.g., "Smith et al. (2020)") with a uniform placeholder string.

    Parameters
    ----------
    text : str
        The input text to cleanse.
    do_streamline_etal : bool
        If True, detect and replace author citation patterns such as
        "Author (year)", "Author & Author (year)", and "Author et al."
        with a configured placeholder. Bracketed citations are removed
        entirely; unbracketed citations are replaced with the placeholder.

    Returns
    -------
    str
        The cleansed text with normalized whitespace, punctuation, and
        (optionally) citation expressions replaced or removed.
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
        text = re.sub(r"\b([A-Z]\. )*(\b[A-Z][A-Z|a-z]+) et al\b\.?", config.textprocessing.placeholder_author, text)

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


def is_pos_conjoined(word: spacy.tokens.Token, pos: str) -> bool:
    """
    Determine if a conjoined word's original part of speech matches a given POS tag.

    Traverses the dependency tree upward from a conjoined word to find the
    root of the conjunction chain, then checks whether that root's part of
    speech matches ``pos``. Returns False immediately if the word is not
    conjoined, has no ancestors, or if ``pos`` is an auxiliary POS tag.

    Parameters
    ----------
    word : spacy.tokens.Token
        The NLP token to evaluate.
    pos : str
        The part-of-speech tag to match against (e.g., ``"NOUN"``,
        ``"VERB"``). Auxiliary POS tags (as defined in
        ``config.grammar.speech.pos_aux``) always return False.

    Returns
    -------
    bool
        True if the root of the conjunction chain has a POS tag matching
        ``pos``, False otherwise.

    Raises
    ------
    ValueError
        If ``word`` is conjoined but no non-conjoined ancestor can be found
        in the dependency tree.
    """

    # Return False if pos is aux (which may not be conjoined)
    if pos in config.grammar.speech.pos_aux:
        return False

    # Check if this word is conjoined
    is_conjoined = is_pos_word(word=word, pos="CONJOINED")
    if not is_conjoined:  # Terminate early if not conjoined
        return False

    # Check if this word has any previous nodes
    word_ancestors = list(word.ancestors)  # All previous nodes leading to word
    if len(word_ancestors) == 0:  # Terminate early if no previous nodes
        return False

    # Follow chain upward to find if original p.o.s. matches given p.o.s.
    for pre_node in word_ancestors:
        # Continue if previous word also conjoined
        if is_pos_word(word=pre_node, pos="CONJOINED"):
            continue

        # Otherwise, check if original p.o.s. matches given p.o.s.
        return is_pos_word(word=pre_node, pos=pos)

    # If no original p.o.s. found, throw an error
    raise ValueError("Err: No original p.o.s. for conjoined word {0}!\n{1}".format(word, word_ancestors))


def is_pos_word(word: spacy.tokens.Token, pos: str, keyword_objs: Optional[list[Keyword]] = None) -> bool:  # noqa: C901, E501
    """
    Determine if a spaCy token belongs to a given part-of-speech category.

    Evaluates a token against a named POS category using a combination of
    spaCy attributes (``dep_``, ``pos_``, ``tag_``), dependency tree
    traversal, and grammar configuration rules. Supports a broad set of
    custom POS labels beyond standard spaCy tags, including structural roles
    such as subjects, objects, and conjunctions.

    Parameters
    ----------
    word : spacy.tokens.Token
        The NLP token to evaluate.
    pos : str
        The part-of-speech category to check. Must be one of:

        - ``"ROOT"`` -- syntactic root of the sentence.
        - ``"VERB"`` -- main verb, including adjectival modifier verbs.
        - ``"USELESS"`` -- non-informative word (not a keyword, subject, or negation).
        - ``"SUBJECT"`` -- grammatical subject.
        - ``"PREPOSITION"`` -- preposition or mishandled auxiliary "to".
        - ``"BASE_OBJECT"`` -- direct or prepositional object (noun).
        - ``"DIRECT_OBJECT"`` -- object directly following a verb.
        - ``"PREPOSITION_OBJECT"`` -- object following a preposition.
        - ``"PREPOSITION_SUBJECT"`` -- subject following a preposition.
        - ``"MARKER"`` -- subordinating conjunction or subject marker.
        - ``"X"`` -- improper or foreign word.
        - ``"CONJOINED"`` -- word joined via conjunction or apposition.
        - ``"DETERMINANT"`` -- determiner (e.g., "the", "a").
        - ``"AUX"`` -- auxiliary verb.
        - ``"NOUN"`` -- noun (excluding determiners).
        - ``"PRONOUN"`` -- pronoun.
        - ``"ADJECTIVE"`` -- adjective or adjectival verb.
        - ``"CONJUNCTION"`` -- coordinating conjunction.
        - ``"PASSIVE"`` -- passive verb or auxiliary.
        - ``"NEGATIVE"`` -- negation word.
        - ``"PUNCTUATION"`` -- punctuation mark (non-alphanumeric).
        - ``"BRACKET"`` -- bracket character.
        - ``"POSSESSIVE"`` -- possessive marker.
        - ``"NUMBER"`` -- numeric token.

    keyword_objs : list, optional
        Collection of keyword objects required when ``pos="USELESS"``.
        Ignored for all other POS categories. Default is None.

    Returns
    -------
    bool
        True if ``word`` belongs to the specified POS category, False otherwise.

    Raises
    ------
    ValueError
        If ``pos="USELESS"`` and ``keyword_objs`` is None.
    ValueError
        If ``pos`` is not one of the recognized category strings listed above.
    """

    # Load global variables
    # word_i = word.i  # Index
    word_dep = word.dep_  # dep label
    word_pos = word.pos_  # p.o.s. label
    word_tag = word.tag_  # tag label
    word_text = word.text  # Text version of word
    word_ancestors = list(word.ancestors)  # All previous nodes leading to word

    # Print some notes
    logger.info("Running is_pos_word for: {0}".format(word))
    logger.info("dep_: {0}\npos_: {1}\ntag_: {2}".format(word_dep, word_pos, word_tag))
    logger.info("Node head: {0}\nSentence: {1}".format(word.head, word.sent))
    logger.info("Node lefts: {0}\nNode rights: {1}".format(list(word.lefts), list(word.rights)))

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

        check_root = is_pos_word(word=word, pos="ROOT")
        check_conj = is_pos_conjoined(word=word, pos=pos)
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
            and is_pos_word(word=word.head, pos="ROOT")
            and is_pos_word(word=word.head, pos="NOUN")
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
            raise ValueError("Err: keyword_objs is required when pos='USELESS'.")

        # Check p.o.s. components
        check_tag = word_tag in config.grammar.speech.tag_useless
        check_dep = word_dep in config.grammar.speech.dep_useless
        check_pos = word_pos in config.grammar.speech.pos_useless
        check_use = check_importance(word_text, version_NLP=word, keyword_objs=keyword_objs)["bools"][
            "is_any"
        ]  # Useful
        check_root = is_pos_word(word=word, pos="ROOT")
        check_neg = is_pos_word(word=word, pos="NEGATIVE")
        check_subj = is_pos_word(word=word, pos="SUBJECT")
        check_all = (check_tag and check_dep and check_pos) and (
            not (check_use or check_neg or check_subj or check_root)
        )

    # Identify subjects
    elif pos in ["SUBJECT"]:
        check_noun = is_pos_word(word=word, pos="NOUN")
        check_adj = is_pos_word(word=word, pos="ADJECTIVE")
        check_obj = is_pos_word(word=word, pos="BASE_OBJECT")

        # Determine if to left of verb or root, if applicable
        is_leftofverb = False
        if len(word_ancestors) > 0:
            tmp_verb = is_pos_word(word=word_ancestors[0], pos="VERB")
            # tmp_root = is_pos_word(word=word_ancestors[0], pos="ROOT")
            # if (tmp_verb or tmp_root):
            if tmp_verb:
                is_leftofverb = word in word_ancestors[0].lefts

        # Determine if conjoined to subject, if applicable
        is_conjsubj = is_pos_conjoined(word, pos=pos)
        is_root = is_pos_word(word=word, pos="ROOT")
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
            (word_dep in config.grammar.speech.dep_aux) and (word_pos in config.grammar.speech.pos_aux) and (check_tag)
        )  # For e.g. mishandled 'to'
        check_all = (check_dep and check_pos and check_tag) or (check_prepaux)

    # Identify base objects (so either direct or prep. objects)
    elif pos in ["BASE_OBJECT"]:
        check_dep = word_dep in config.grammar.speech.dep_object
        check_noun = is_pos_word(word=word, pos="NOUN")
        check_all = check_noun and check_dep

    # Identify direct objects
    elif pos in ["DIRECT_OBJECT"]:
        check_baseobj = is_pos_word(word=word, pos="BASE_OBJECT")
        is_conjdirobj = is_pos_conjoined(word, pos=pos)
        # Check preceding term is a verb
        check_afterprep = False
        check_afterverb = False
        for pre_node in word_ancestors:
            # If preceding preposition found first
            if is_pos_word(word=pre_node, pos="PREPOSITION"):
                check_afterprep = True
                break
            # If preceding verb found first
            elif is_pos_word(word=pre_node, pos="VERB"):
                check_afterverb = True
                break

        check_all = ((not check_afterprep) and (check_afterverb) and (check_baseobj)) or is_conjdirobj

    # Identify prepositional objects
    elif pos in ["PREPOSITION_OBJECT"]:
        check_baseobj = is_pos_word(word=word, pos="BASE_OBJECT")
        is_conjprepobj = is_pos_conjoined(word, pos=pos)
        # Check if this word follows preposition
        check_objprep = False
        for pre_node in word_ancestors:
            # If preceding preposition found first
            if is_pos_word(word=pre_node, pos="PREPOSITION"):
                pre_pre_node = list(pre_node.ancestors)[0]
                # Ensure prepositional object instead of prep. subject
                check_objprep = not is_pos_word(word=pre_pre_node, pos="SUBJECT")
                break
            # If preceding verb found first
            elif is_pos_word(word=pre_node, pos="VERB"):
                check_objprep = False
                break

        check_all = is_conjprepobj or (check_baseobj and check_objprep)

    # Identify prepositional subjects
    elif pos in ["PREPOSITION_SUBJECT"]:
        check_obj = is_pos_word(word=word, pos="BASE_OBJECT")
        is_conjprepsubj = is_pos_conjoined(word, pos=pos)
        # Check if this word follows preposition
        check_subjprep = False
        for pre_node in word_ancestors:
            # If preceding preposition found first
            if is_pos_word(word=pre_node, pos="PREPOSITION"):
                pre_pre_node = list(pre_node.ancestors)[0]
                # Ensure prepositional subject instead of prep. object
                check_subjprep = is_pos_word(word=pre_pre_node, pos="SUBJECT")
                break
            # If preceding verb found first
            elif is_pos_word(word=pre_node, pos="VERB"):
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
        is_afterroot = is_notroot and is_pos_word(word=word_ancestors[0], pos="ROOT")
        check_subjmark = False
        if (is_notroot) and (not is_afterroot):
            check_subj = is_pos_word(word=word, pos="SUBJECT")
            check_det = is_pos_word(word=word, pos="DETERMINANT")
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
    logger.info("Is pos={0}? {1}\n-".format(pos, check_all))
    # Return the final verdict
    return check_all


def search_text(text: str, keyword_objs: list[Keyword]) -> dict:
    """
    Search text for keywords and acronyms from a collection of keyword objects.

    Iterates over each keyword object, calling its ``identify_keyword`` method,
    and aggregates the results into a single boolean match flag and a combined
    list of character spans for all matches found.

    Parameters
    ----------
    text : str
        The input text to search.
    keyword_objs : list
        Collection of keyword objects, each exposing an ``identify_keyword``
        method that returns a dict with keys ``"bool"`` and ``"charspans"``.

    Returns
    -------
    dict
        A dictionary with the following keys:

        - ``"bool"`` : bool -- True if any keyword or acronym was found in ``text``.
        - ``"charspans"`` : list of tuple -- Character spans ``(start, end)``
          for every match across all keyword objects.
    """

    # Check if keywords and/or acronyms present in given text
    tmp_res = [item.identify_keyword(text) for item in keyword_objs]
    check_keywords = any([item["bool"] for item in tmp_res])
    charspans_keywords = []
    for ii in range(0, len(tmp_res)):
        charspans_keywords += tmp_res[ii]["charspans"]

    # Print some notes
    # Extract global variables
    keywords = [item2 for item1 in keyword_objs for item2 in item1._keywords]
    acronyms = [
        item2 for item1 in keyword_objs for item2 in item1._acronyms_casesensitive + item1._acronyms_caseinsensitive
    ]

    logger.info("Completed search_text().")
    logger.info("Keywords={0}\nAcronyms={1}".format(keywords, acronyms))
    logger.info("Boolean: {0}".format(check_keywords))
    logger.info("Char. Spans: {0}".format(charspans_keywords))

    # Return boolean result
    return {"bool": check_keywords, "charspans": charspans_keywords}
