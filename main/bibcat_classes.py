###FILE: bibcat_classes.py
###PURPOSE: Container for all classes used for the BibCat package.
###DATE CREATED: 2022-01-24
###DEVELOPERS: (Jamila Pegues, Others)




##Below Section: Runs all imports and presets/constants
#External packages
import re
import os
import numpy as np
import itertools as iterer
import collections
import matplotlib.pyplot as plt
#
#Internal presets/constants
import bibcat_constants as preset
import bibcat_config as config
#
import spacy
nlp = spacy.load(preset.spacy_language_model)
import nltk
from nltk.corpus import wordnet
#
import tensorflow as tf
from official.nlp import optimization as tf_opt
import tensorflow_hub as tfhub
import tensorflow_text as tftext
#


##Class: _Base
class _Base():
    """
    WARNING! This class is *not* meant to be used directly by users.
    -
    Class: _Base
    Purpose:
     - Container for common underlying methods used in other classes.
     - Purely meant to be inherited by other classes.
    -
    """
    ##Method: __init__
    ##Purpose: Initialize this class instance
    def __init__(self):
        """
        Method: __init__
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Initializes instance of _Base class.
        """
        #Initialize storage
        self._storage = {}

        #Close method
        return
    #

    ##Method: _get_info()
    ##Purpose: Retrieve specified data via given key
    def _get_info(self, key, do_flag_hidden=False):
        """
        Method: _get_info
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Fetches values, etc., for this class instance from storage.
        """
        #Attempt to retrieve data stored under the given key
        try:
            return self._storage[key]
        #
        #Throw helpful error if retrieval attempt failed
        except KeyError:
            #Return a specialized testing error, if likely hidden method called
            if do_flag_hidden:
                errstr = (("Whoa there. This error likely happened because"
                        +" you are testing or exploring a hidden ('_') method."
                        +" If so, you likely need to pass in this parameter -"
                        +" '{0}' - as an input to the method.").format(key))
            #Otherwise, return generic error for available stored data
            else:
                errstr = (("Whoa there. Looks like you requested data from a"
                        +" key ({1}) that does not exist. Available keys"
                        +" are:\n{0}")
                        .format(sorted(self._storage.keys()), key))
            #

            #Raise the custom error
            raise KeyError(errstr)
        #

        #Close method
        return
    #

    ##Method: _store_info()
    ##Purpose: Store given data into class instance in a unified way
    def _store_info(self, data, key):
        """
        Method: _store_info
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Stores values, etc., for this class instance in storage.
        """
        #Store the data into underlying dictionary
        self._storage[key] = data

        #Close method
        return
    #

    ##Method: _assemble_keyword_wordchunks()
    ##Purpose: Assemble wordchunks containing keywords from given text
    def _assemble_keyword_wordchunks(self, text, keyword_objs, do_include_verbs=False, do_include_brackets=False, lookup_terms=None, do_verbose=False):
        """
        Method: _assemble_keyword_wordchunks
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Assembles noun chunks around any keyword terms within given text.
        """
        #Find indices of keywords within text
        tmp_sents = list(nlp(str(text)).sents)
        list_wordchunks = []
        for curr_sent in tmp_sents:
            #Extract keyword indices of current sentence
            if (lookup_terms is not None): #If specific keyword terms requested
                set_inds = [ind for ind in range(0, len(curr_sent))
                            if any([bool(re.search((r"\b"+item+r"\b"),
                                                curr_sent[ind].text,
                                                flags=re.IGNORECASE))
                                    for item in lookup_ambigs])]
            #Otherwise, check against all keyword objects
            else:
                set_inds = [ind for ind in range(0, len(curr_sent))
                            if any([item.is_keyword(curr_sent[ind].text)
                                    for item in keyword_objs])]
            #

            #Print some notes
            if do_verbose:
                print("Current sentence: '{0}'".format(curr_sent))
                print("Indices of lookups in sent.: '{0}'".format(set_inds))
            #

            #Build wordchunks from indices of current sentence
            first_ind = np.inf
            last_ind = -np.inf
            for curr_start in set_inds:
                #Print some notes
                if do_verbose:
                    print("\n-Building wordchunk from {0}: {1}:"
                            .format(curr_start, curr_sent[curr_start]))
                #
                #Skip if this index has already been surpassed
                if (curr_start <= last_ind):
                    continue
                #
                curr_wordtext = [curr_sent[curr_start].text]

                #Build wordchunk from accumulating nouns on the left
                for ii in range(0, curr_start)[::-1]:#Do not include start here
                    #Store index if noun or numeral or adjective
                    check_noun = self._is_pos_word(word=curr_sent[ii],
                                                    pos="NOUN")
                    check_adj = self._is_pos_word(word=curr_sent[ii],
                                                    pos="ADJECTIVE")
                    check_num = self._is_pos_word(word=curr_sent[ii],
                                                    pos="NUMBER")
                    check_pos = self._is_pos_word(word=curr_sent[ii],
                                                    pos="POSSESSIVE")
                    check_dash = (curr_sent[ii].text == "-")
                    check_imp = self._check_importance(curr_sent[ii].text,
                                        keyword_objs=keyword_objs,
                                        version_NLP=curr_sent[ii])["is_any"]
                    #Include punctuation, if so requested
                    if do_include_brackets:
                        check_brackets = self._is_pos_word(word=curr_sent[ii],
                                                        pos="BRACKET")
                    else:
                        check_brackets = False
                    #
                    tmp_list = [check_noun, check_adj, check_num, check_dash,
                                check_pos, check_imp, check_brackets]
                    #

                    #Keep word if relevant p.o.s.
                    if any(tmp_list):
                        curr_wordtext.insert(0, curr_sent[ii].text)
                        first_ind = ii #Update latest index
                    #
                    #Otherwise, break and end this makeshift wordchunk
                    else:
                        break
                    #
                #

                #Build wordchunk from accumulating nouns on the right
                for ii in range((curr_start+1), len(curr_sent)):
                    #Store index if noun or numeral or adjective, etc.
                    check_noun = self._is_pos_word(word=curr_sent[ii],
                                                    pos="NOUN")
                    check_adj = self._is_pos_word(word=curr_sent[ii],
                                                    pos="ADJECTIVE")
                    check_num = self._is_pos_word(word=curr_sent[ii],
                                                    pos="NUMBER")
                    check_pos = self._is_pos_word(word=curr_sent[ii],
                                                    pos="POSSESSIVE")
                    check_imp = self._check_importance(curr_sent[ii].text,
                                        keyword_objs=keyword_objs,
                                        version_NLP=curr_sent[ii])["is_any"]
                    check_dash = (curr_sent[ii].text == "-")
                    #Include brackets, if so requested
                    if do_include_brackets:
                        check_brackets = self._is_pos_word(word=curr_sent[ii],
                                                        pos="BRACKET")
                    else:
                        check_brackets = False
                    #
                    #Tack on verb check if requested (e.g., to cover noun-verbs)
                    if do_include_verbs: #E.g., ambig. 'Hubble-imaged data'
                        check_verb = self._is_pos_word(word=curr_sent[ii],
                                                        pos="VERB")
                    else:
                        check_verb = False
                    #
                    tmp_list =[check_noun, check_adj, check_num, check_brackets,
                                check_verb, check_pos, check_imp, check_dash]
                    #

                    #Keep word if relevant p.o.s.
                    if any(tmp_list):
                        curr_wordtext.append(curr_sent[ii].text)
                        last_ind = ii #Update latest index
                    #
                    #Otherwise, break and end this makeshift wordchunk
                    else:
                        break
                    #
                #

                #Store the makeshift wordchunk
                curr_str_fin = self._cleanse_text(" ".join(curr_wordtext),
                                                do_streamline_etal=False)
                list_wordchunks.append(nlp(curr_str_fin))

                #Print some notes
                if do_verbose:
                    print("Latest makeshift wordchunks: {0}"
                            .format(list_wordchunks))
                #
            #
        #

        #Return the assembled wordchunks
        if do_verbose:
            print("Assembled keyword wordchunks:\n{0}".format(list_wordchunks))
        #
        return list_wordchunks
    #

    ##Method: _check_importance()
    ##Purpose: Determine if given text is important (e.g., is a keyword)
    def _check_importance(self, text, include_Ipronouns=True, include_terms=True, include_etal=True, keyword_objs=None, version_NLP=None):
        """
        Method: _check_importance
        WARNING! This method is *not* meant to be used directly by users.
        Purpose:
         - Checks if given text contains any important terms.
           - Important terms includes keywords, 1st,3rd person pronouns, etc.
         - Returns dictionary of bools for presence/absence of important terms.
        """
        #Extract the NLP version of this text, if not given
        if (version_NLP is None):
            version_NLP = nlp(text)
        #
        #Ensure NLP version is iterable
        if (not hasattr(version_NLP, "__iter__")):
            version_NLP = [version_NLP]
        #

        #Extract keyword objects from storage, if not given
        if (keyword_objs is None):
            try:
                keyword_objs = [self._get_info("keyword_obj",
                                            do_flag_hidden=True)]
            except KeyError:
                keyword_objs = self._get_info("keyword_objs",
                                            do_flag_hidden=True)
        #

        #Cleanse and streamline the given text
        text = self._cleanse_text(text, do_streamline_etal=True)
        #

        #Initialize container for booleans
        dict_results = {}

        ##Check if text contains keywords, acronyms, important terms, etc
        #For target keywords and acronyms
        dict_results["is_keyword"] = self._search_text(text=text,
                                                    keyword_objs=keyword_objs)
        #

        #Check for first-person pronouns, if requested
        if include_Ipronouns:
            list_pos_pronoun = preset.pos_pronoun
            nlp_lookup_person = preset.nlp_lookup_person
            check_pronounI = any([((item.pos_ in list_pos_pronoun) #Pronoun
                                and ("1" in item.morph.get(nlp_lookup_person)))
                                for item in version_NLP]) #Check if 1st-person
            dict_results["is_pron_1st"] = check_pronounI
        else: #Otherwise, remove pronoun contribution
            dict_results["is_pron_1st"] = False
        #

        #Check for special terms, if requested
        if include_terms:
            list_pos_pronoun = preset.pos_pronoun
            nlp_lookup_person = preset.nlp_lookup_person
            special_synsets_fig = preset.special_synsets_fig
            #
            #For 'they' pronouns
            check_terms_they = any([((item.pos_ in list_pos_pronoun) #Pronoun
                                and ("3" in item.morph.get(nlp_lookup_person)))
                                for item in version_NLP]) #Check if 3rd-person
            #For 'figure', etc, terms
            check_terms_fig = any([(item2.name() in special_synsets_fig)
                                for item1 in version_NLP
                                for item2 in wordnet.synsets(item1.text)]
                                ) #Check if any words have figure, etc, synsets
            #
            #Store the booleans
            dict_results["is_pron_3rd"] = check_terms_they
            dict_results["is_term_fig"] = check_terms_fig
        else: #Otherwise, remove term contribution
            dict_results["is_pron_3rd"] = False
            dict_results["is_term_fig"] = False
        #

        #Check for etal terms, if requested
        if include_etal:
            exp = preset.exp_etal_cleansed #Reg.ex. to find cleansed et al
            check_etal = bool(re.search(exp, text, flags=re.IGNORECASE))
            dict_results["is_etal"] = check_etal
        else: #Otherwise, remove term contribution
            dict_results["is_etal"] = False
        #

        #Store overall status of if any booleans set to True
        dict_results["is_any"] =any([dict_results[key] for key in dict_results])
        #

        #Return the booleans
        return dict_results
    #

    ##Method: _check_truematch()
    ##Purpose: Return boolean for whether or not text contains a true vs false match to the given keywords
    def _check_truematch(self, text, keyword_objs, dict_ambigs, do_verbose=None, do_verbose_deep=False):
        """
        Method: _check_truematch
        WARNING! This method is *not* meant to be used directly by users.
        Purpose:
         - Determine if given text contains a true vs. false match to keywords.
           - E.g.: 'Edwin Hubble' as a false match to Hubble Space Telescope.
        """
        #Load global variables
        if (do_verbose is None):
            do_verbose = self._get_info("do_verbose", do_flag_hidden=True)
        #
        #Process ambig. phrase data, if not given
        if (dict_ambigs is None):
            dict_ambigs = self._process_database_ambig(
                                                    do_verbose=do_verbose_deep,
                                                    keyword_objs=keyword_objs)
        #

        #Extract info from ambig. database
        list_kw_ambigs = dict_ambigs["all_kw_ambigs"]
        list_exp_ambigs = dict_ambigs["all_exp_ambigs"]
        list_bool_ambigs = dict_ambigs["all_bool_ambigs"]
        list_text_ambigs = dict_ambigs["all_text_ambigs"]
        lookup_ambigs = dict_ambigs["lookup_ambigs"]
        lookup_ambigs_lower = [item.lower()
                                for item in dict_ambigs["lookup_ambigs"]]
        num_ambigs = len(list_kw_ambigs)
        #

        #Replace numerics and citation numerics with placeholders
        text_orig = text
        placeholder_number = preset.placeholder_number
        text = re.sub(r"\(?\b[0-9]+\b\)?", placeholder_number, text_orig)

        #Print some notes
        if do_verbose:
            print(("\n> Running _check_truematch for text: '{0}'"
                    +"\nOriginal text: {1}\nLookups: {2}")
                    .format(text, text_orig, lookup_ambigs))
        #

        #Extract keyword objects that are potentially ambiguous
        keyword_objs_ambigs = [item1 for item1 in keyword_objs
                                if any([item1.is_keyword(item2)
                                        for item2 in lookup_ambigs])]
        #

        #Return status as true match if non-ambig keywords match to text
        if any([item1.is_keyword(text) for item1 in keyword_objs
                if (item1 not in keyword_objs_ambigs)]):
            #Print some notes
            if do_verbose:
                print("Text matches unambiguous keyword. Returning true state.")
            #
            #Return status as true match
            return {"bool":True, "info":[{"matcher":None, "set":None,
                        "bool":True,
                        "text_wordchunk":"<Not ambig.>", "text_database":None}]}
        #
        #Return status as false match if no keywords match at all
        elif (not any([item.is_keyword(text) for item in keyword_objs_ambigs])):
            #Print some notes
            if do_verbose:
                print("Text matches no keywords at all. Returning false state.")
            #
            #Return status as true match
            return {"bool":False, "info":[{"matcher":None, "text_database":None,
                            "bool":False,
                            "text_wordchunk":"<No matching keywords at all.>"}]}
        #

        #Assemble makeshift wordchunks (not using NLP ones here)
        #Not sure why happened, but NLP sometimes failed to identify nouns/num.
        #Print some notes
        if do_verbose:
            print("Building noun chunks around keywords...")
        #
        #Generate the keyword-based wordchunks
        list_wordchunks = self._assemble_keyword_wordchunks(text=text,
                                                    keyword_objs=keyword_objs,
                                                    do_verbose=do_verbose,
                                                    do_include_verbs=False)
        #Throw error if no wordchunks identified
        if (len(list_wordchunks) == 0):
            errstr = ("No final wordchunks!: {0}\n{1}\nText: '{2}'"
                    .format(list_wordchunks, list_wordchunks_raw, text)
                    +"\nAll words and p.o.s.:\n")
            for aa in range(0, len(tmp_sents)):
                for bb in range(0, len(tmp_sents[aa])):
                    tmp_word = tmp_sents[aa][bb]
                    errstr += ("{0}: {1}, {2}, {3}\n"
                                .format(tmp_word, tmp_word.dep_,
                                        tmp_word.pos_, tmp_word.tag_))
            #
            raise ValueError(errstr)
        #

        #Print some notes
        if do_verbose:
            print("\n- Wordchunks determined for text: {0}"
                    .format(list_wordchunks))
        #

        #Exit method early if any wordchunk is an exact keyword match
        if any([(item.text.lower() in lookup_ambigs)
                        for item in list_wordchunks]):
            #Print some notes
            if do_verbose:
                print("Exact keyword match found. Returning true status...")
            #
            return {"bool":True,
                    "info":[{"matcher":None, "text_database":None, "bool":True,
                        "text_wordchunk":"<Wordchunk has exact term match.>"}]}
        #

        #Iterate through wordchunks to determine true vs false match status
        num_wordchunks = len(list_wordchunks)
        list_status = [None]*num_wordchunks
        list_results = [None]*num_wordchunks
        for ii in range(0, num_wordchunks):
            curr_chunk = list_wordchunks[ii] #Current wordchunk
            curr_chunk_text = curr_chunk.text
            #Print some notes
            if do_verbose:
                print("Considering wordchunk: {0}".format(curr_chunk_text))
            #

            #Store as non-ambig. phrase and skip ahead if non-ambig. term
            is_exact = any([(curr_chunk_text.lower().replace(".","")
                                == item2.lower())
                            for item1 in keyword_objs
                            for item2 in (item1._get_info("keywords")
                                            +item1._get_info("acronyms"))
                            if (item2.lower() not in lookup_ambigs_lower)]
                            ) #Check if wordchunk matches to any non-ambig terms
            if is_exact:
                #Print some notes
                if do_verbose:
                    print("Exact match to non-ambig. phrase. Marking true...")
                #
                #Store info for this true match
                list_results[ii] = {"bool":True,
                                    "info":{"matcher":None, "bool":True,
                                            "text_wordchunk":curr_chunk_text,
                                            "text_database":None}}
                #Skip ahead
                continue
            #

            #Extract representation of core meaning of current wordchunk
            tmp_res = self._extract_core_from_phrase(phrase_NLP=curr_chunk,
                                keyword_objs=keyword_objs_ambigs,
                                do_skip_useless=False,
                                do_verbose=do_verbose)
            curr_meaning = tmp_res["str_meaning"] #Str representation of meaning
            curr_inner_kw = tmp_res["keywords"] #Matched keywords
            #

            #Extract all ambig. phrases+substrings that match to this meaning
            set_matches_raw = [{"ind":jj, "text_database":list_text_ambigs[jj],
                                "text_wordchunk":curr_chunk_text,
                                "exp":list_exp_ambigs[jj],
                                "matcher":re.search(list_exp_ambigs[jj],
                                            curr_meaning, flags=re.IGNORECASE),
                                "bool":list_bool_ambigs[jj]}
                                for jj in range(0, num_ambigs)
                                if (list_kw_ambigs[jj] in curr_inner_kw)]
            set_matches = [item for item in set_matches_raw
                            if (item["matcher"] is not None)]
            #
            #Print some notes
            if do_verbose_deep:
                print("Set of matches assembled from ambig. database:")
                for item1 in set_matches_raw:
                    print(item1)
            #

            #Throw error if no match found
            if (len(set_matches) == 0):
                raise ValueError("Err: Unrecognized ambig. phrase:\n{0}"
                                .format(curr_chunk))
            #

            #Determine and extract best match (=match with shortest substring)
            best_set = sorted(set_matches,
                            key=(lambda w:len(w["matcher"][0])))[0]

            #Print some notes
            if do_verbose:
                print("Current wordchunk: {0}\nMeaning: {2}\nBest set: {1}-"
                        .format(curr_chunk, best_set, curr_meaning))
            #

            #Store the verdict for this best match
            list_status[ii] = best_set["bool"]

            #Exit method early since match found
            if do_verbose:
                print("Match found. Returning status...")
            #
            list_results[ii] = {"bool":best_set["bool"],
                    "info":{"matcher":best_set["matcher"],
                            "bool":best_set["bool"],
                            "text_wordchunk":best_set["text_wordchunk"],
                            "text_database":best_set["text_database"]}}
        #

        #Combine the results and return overall boolean match
        fin_result = {"bool":any([(item["bool"]) for item in list_results]),
                        "info":[item["info"] for item in list_results]}
        return fin_result
        #
    #

    ##Method: _cleanse_text()
    ##Purpose: Cleanse given (any length) string of extra whitespace, dashes, etc.
    def _cleanse_text(self, text, do_streamline_etal):
        """
        Method: _cleanse_text
        WARNING! This method is *not* meant to be used directly by users.
        Purpose:
         - Cleanse text of extra whitespace, punctuation, etc.
         - Replace paper citations (e.g. 'et al.') with uniform placeholder.
        """
        #Extract global punctuation expressions
        set_apostrophe = preset.set_apostrophe
        set_punctuation = preset.set_punctuation
        exp_punctuation = preset.exp_punctuation
        set_openbrackets = preset.set_openbrackets
        set_closebrackets = preset.set_closebrackets

        #Remove any starting punctuation
        text = re.sub(((r"^("+"|".join(exp_punctuation)+r")")),
                        "", text) #Remove starting punct.

        #Remove extra whitespace in general
        text = re.sub("  +", " ", text) #Removes spaces > length=1

        #Remove excessive whitespace around punctuation
        #For opening brackets
        tmp_exp_inner = ("\\" + "|\\".join(set_openbrackets))
        text = re.sub(("("+tmp_exp_inner+") ?"), r"\1", text)
        #For closing brackets and punctuation
        tmp_exp_inner = ("\\" + "|\\"
                            .join((set_closebrackets+set_punctuation)))
        text = re.sub((" ?("+tmp_exp_inner+")"), r"\1", text)
        #For apostrophes
        tmp_exp_inner = ("\\" + "|\\".join(set_apostrophe))
        text = re.sub((" ?("+tmp_exp_inner+") ?"), r"\1", text)
        #

        #Remove empty brackets and doubled-up punctuation
        #Extract ids of empty brackets and doubled-up punctuation
        ids_rem = [ii for ii in range(0, (len(text)-1))
                    if (((text[ii] in set_openbrackets)
                        and (text[ii+1] in set_closebrackets)) #Empty brackets
                    or ((text[ii] in set_punctuation)
                        and (text[ii+1] in set_punctuation)))] #Double punct.
        #Remove the characters (in reverse order!) at the identified ids
        for ii in sorted(ids_rem)[::-1]: #Reverse-sorted
            text = text[0:ii] + text[(ii+1):len(text)]
        #

        #Replace pesky "Author & Author (date)", et al., etc., wordage
        if do_streamline_etal:
            #Adapted from:
            # https://regex101.com/r/xssPEs/1
            # https://stackoverflow.com/questions/63632861/
            #                           python-regex-to-get-citations-in-a-paper
            bit_author = r"(?:[A-Z][A-Za-z'`-]+)"
            bit_etal = r"(?:et al\.?)"
            bit_additional = f"(?:,? (?:(?:and |& )?{bit_author}|{bit_etal}))"
            #Regular expressions for years (with or without brackets)
            exp_year_yesbrackets = (
                        r"( (\(|\[|\{)"
                        + r"([0-9]{4,4}|[0-9]{2,2})"
                        + r"((,|;) ?([0-9]{4,4}|[0-9]{2,2}))*"
                        + r"(\)|\]|\}))")
            exp_year_nobrackets = (
                        r" "
                        + r"([0-9]{4,4}|[0-9]{2,2})"
                        + r"((,|;) ?([0-9]{4,4}|[0-9]{2,2}))*")
            #Regular expressions for citations (with or without brackets)
            exp_cites_yesbrackets = (
                    r"(\(|\[|\{)"
                    +fr"{bit_author}{bit_additional}*{exp_year_nobrackets}"
                    +(r"((,|;) "
                        +fr"{bit_author}{bit_additional}*{exp_year_nobrackets}"
                    +r")*")
                    +r"(\)|\]|\})")
            exp_cites_nobrackets = (
                    fr"{bit_author}{bit_additional}*{exp_year_yesbrackets}")
            #
            #Replace not-bracketed citations or remove bracketed citations
            text = re.sub(exp_cites_yesbrackets, "", text)
            text = re.sub(exp_cites_nobrackets, preset.placeholder_author, text)
            #
            #Replace singular et al. (e.g. SingleAuthor et al.) wordage as well
            text = re.sub(r" et al\b\.?", "etal", text)
        #

        #Remove starting+ending whitespace
        text = text.lstrip().rstrip()

        #Return cleansed text
        return text
    #

    ##Method: _extract_core_from_phrase()
    ##Purpose: Extract core meaning (e.g., synsets) from given phrase
    def _extract_core_from_phrase(self, phrase_NLP, do_skip_useless, do_verbose=None, keyword_objs=None):
        """
        Method: _extract_core_from_phrase
        WARNING! This method is *not* meant to be used directly by users.
        Purpose:
         - Extract representative "meaning" (i.e., synsets) of given phrase.
        """
        #Set global variables
        num_words = len(phrase_NLP)
        if (do_verbose is None):
            do_verbose = self._get_info("do_verbose")
        if (keyword_objs is None):
            try:
                keyword_objs = [self._get_info("keyword_obj",
                                                do_flag_hidden=True)]
            except KeyError:
                keyword_objs = self._get_info("keyword_objs",
                                                do_flag_hidden=True)
        #
        #Print some notes
        if do_verbose:
            print("\n> Running _extract_core_from_phrase for phrase: {0}"
                    .format(phrase_NLP))
        #

        #Initialize containers of core information
        core_keywords = []
        core_synsets = []

        #Iterate through words within phrase
        for ii in range(0, num_words):
            curr_word = phrase_NLP[ii]
            #Print some notes
            if do_verbose:
                print("-\nLatest keywords: {0}".format(core_keywords))
                print("Latest synsets: {0}".format(core_synsets))
                print("-Now considering word: {0}".format(curr_word))
            #

            #Skip if this word is punctuation or possessive marker
            if (self._is_pos_word(word=curr_word, pos="PUNCTUATION")
                        or self._is_pos_word(word=curr_word, pos="POSSESSIVE")):
                #Print some notes
                if do_verbose:
                    print("Word is punctuation or possessive. Skipping.")
                #
                continue
            #

            #Store the keyword itself and skip ahead if this word is a keyword
            matched_kobjs = [item for item in keyword_objs
                            if (item.is_keyword(curr_word.text))]
            if (len(matched_kobjs) > 0): #If word is a keyword
                name_kobj = matched_kobjs[0].get_name() #Fetch name for kobj
                core_keywords.append(name_kobj.lower())
                core_synsets.append([name_kobj.lower()])
                #
                #Print some notes
                if do_verbose:
                    print("Word itself is keyword. Stored synset: {0}"
                            .format(core_synsets))
                #
                continue
            #

            #Store a representative synset and skip ahead if word is a numeral
            if bool(re.search(("^(ID)?[0-9]+"), curr_word.text,
                                flags=re.IGNORECASE)):
                tmp_rep = preset.string_numeral_ambig
                core_synsets.append([tmp_rep])
                #Print some notes
                if do_verbose:
                    print("Word itself is a numeral. Stored synset: {0}"
                            .format(core_synsets))
                #
                continue
            #

            #Ignore this word if not a relevant p.o.s.
            check_useless = self._is_pos_word(word=curr_word, pos="USELESS",
                                    keyword_objs=keyword_objs)
            check_adj = self._is_pos_word(word=curr_word, pos="ADJECTIVE")
            if (do_skip_useless and (check_useless and (not check_adj))):
                #Print some notes
                if do_verbose:
                    print("Word itself is useless. Skipping.")
                #
                continue
            #

            #Gather and store the noun synsets
            curr_synsets_raw = [item.name()
                            for item in wordnet.synsets(curr_word.text)
                            if (".n." in item.name())] #Noun synsets only
            #
            #If no synsets known, store word itself
            if (len(curr_synsets_raw) == 0):
                core_synsets.append([curr_word.text.lower()])
            #Otherwise, store synsets
            else:
                core_synsets += [curr_synsets_raw]
            #

            #Print some notes
            if do_verbose:
                print("-Done considering word: {0}".format(curr_word))
                print("Updated synsets: {0}".format(core_synsets))
                print("Updated keywords: {0}".format(core_keywords))
            #
        #

        #Throw an error if any empty strings passed as synsets
        if any([("" in item) for item in core_synsets]):
            raise ValueError("Err: Empty synset?\n{0}".format(core_synsets))
        #

        #Extract unique roots of sets of synsets
        exp_synset = preset.exp_synset
        core_roots = [np.unique([item2.split(".")[0]
                                if bool(re.search(exp_synset, item2))
                                else item2
                                for item2 in item1]).tolist()
                        for item1 in core_synsets]
        #

        #Convert core meaning into string representation
        str_meaning = " ".join([" ".join(item)
                                for item in core_roots]) #Long spaced string

        #Return the core components
        if do_verbose:
            print(("\n-\nPhrase '{0}':\nKeyword: {1}\nSynsets: {2}"
                    +"\nRoots: {3}\nString representation: {4}\n-\n")
                .format(phrase_NLP, core_keywords, core_synsets, core_roots,
                        str_meaning))
        #
        return {"keywords":core_keywords, "synsets":core_synsets,
                "roots":core_roots, "text":phrase_NLP.text,
                "str_meaning":str_meaning}
    #

    ##Method: _is_pos_conjoined()
    ##Purpose: Return boolean for if given word (NLP type word) is of conjoined given part of speech
    def _is_pos_conjoined(self, word, pos):
        """
        Method: _is_pos_conjoined
        WARNING! This method is *not* meant to be used directly by users.
        Purpose:
         - Determine if original part-of-speech (p.o.s.) for given word (if conjoined) matches given p.o.s.
        """
        #Check if this word is conjoined
        is_conjoined = self._is_pos_word(word=word, pos="CONJOINED")
        if (not is_conjoined): #Terminate early if not conjoined
            return False
        #

        #Check if this word has any previous nodes
        word_ancestors = list(word.ancestors)#All previous nodes leading to word
        if (len(word_ancestors) == 0): #Terminate early if no previous nodes
            return False
        #

        #Follow chain upward to find if original p.o.s. matches given p.o.s.
        for pre_node in word_ancestors:
            #Continue if previous word also conjoined
            if self._is_pos_word(word=pre_node, pos="CONJOINED"):
                continue
            #
            #Otherwise, check if original p.o.s. matches given p.o.s.
            return self._is_pos_word(word=pre_node, pos=pos)
        #

        #If no original p.o.s. found, throw an error
        raise ValueError("Err: No original p.o.s. for conjoined word {0}!\n{1}"
                            .format(word, word_ancestors))
    #

    ##Method: _is_pos_word()
    ##Purpose: Return boolean for if given word (NLP type word) is of given part of speech
    def _is_pos_word(self, word, pos, keyword_objs=None):
        """
        Method: _is_pos_word
        WARNING! This method is *not* meant to be used directly by users.
        Purpose:
         - Return if the given word (of NLP type) is of the given part-of-speech.
        """
        ##Load global variables
        word_i = word.i #Index
        word_dep = word.dep_ #dep label
        word_pos = word.pos_ #p.o.s. label
        word_tag = word.tag_ #tag label
        word_text = word.text #Text version of word
        word_ancestors = list(word.ancestors)#All previous nodes leading to word
        #

        ##Check if given word is of given part-of-speech
        #Identify roots
        if pos in ["ROOT"]:
            return (word_dep in preset.dep_root)
        #
        #Identify verbs
        elif pos in ["VERB"]:
            check_posaux = (word_pos in preset.pos_aux)
            check_root = self._is_pos_word(word=word, pos="ROOT")
            check_tag = (word_tag in preset.tag_verb_any)
            check_pos = (word_pos in preset.pos_verb)
            check_dep = (word_dep in preset.dep_verb)
            tag_approved = (preset.tag_verb_present + preset.tag_verb_past
                            + preset.tag_verb_future)
            check_approved = (word_tag in tag_approved)
            return ((((check_dep or check_root) and check_pos and check_tag)
                    or (check_root and check_posaux)) and check_approved)
        #
        #Identify useless words
        elif pos in ["USELESS"]:
            #Fetch keyword objects
            if (keyword_objs is None):
                try:
                    keyword_objs = [self._get_info("keyword_obj")]
                except KeyError:
                    keyword_objs = self._get_info("keyword_objs",
                                                    do_flag_hidden=True)
            #
            #Check p.o.s. components
            check_tag = (word_tag in preset.tag_useless)
            check_dep = (word_dep in preset.dep_useless)
            check_pos = (word_pos in preset.pos_useless)
            check_use = self._check_importance(word_text,
                                        version_NLP=word,
                                    keyword_objs=keyword_objs)["is_any"] #Useful
            check_root = self._is_pos_word(word=word, pos="ROOT")
            check_neg = self._is_pos_word(word=word, pos="NEGATIVE")
            check_subj = self._is_pos_word(word=word, pos="SUBJECT")
            return ((check_tag and check_dep and check_pos)
                and (not (check_use or check_neg or check_subj or check_root)))
        #
        #Identify subjects
        elif pos in ["SUBJECT"]:
            check_noun = self._is_pos_word(word=word, pos="NOUN")
            check_adj = self._is_pos_word(word=word, pos="ADJECTIVE")
            check_obj = self._is_pos_word(word=word, pos="BASE_OBJECT")
            #
            #Determine if to left of verb or root, if applicable
            is_leftofverb = False
            if (len(word_ancestors) > 0):
                tmp_verb = self._is_pos_word(word=word_ancestors[0], pos="VERB")
                tmp_root = self._is_pos_word(word=word_ancestors[0], pos="ROOT")
                if (tmp_verb or tmp_root):
                    is_leftofverb = (word in word_ancestors[0].lefts)
            #
            #Determine if conjoined to subject, if applicable
            is_conjsubj = self._is_pos_conjoined(word, pos=pos)
            check_dep = (word_dep in preset.dep_subject)
            return (((check_dep or is_conjsubj)
                        or ((check_noun or check_adj) and is_leftofverb))
                    and (not check_obj))
        #
        #Identify prepositions
        elif pos in ["PREPOSITION"]:
            check_dep = (word_dep in preset.dep_preposition)
            check_pos = (word_pos in preset.pos_preposition)
            check_tag = (word_tag in preset.tag_preposition)
            check_prepaux = ((word_dep in preset.dep_aux)
                            and (word_pos in preset.pos_aux)
                            and (check_tag)) #For e.g. mishandled 'to'
            return ((check_dep and check_pos and check_tag) or (check_prepaux))
        #
        #Identify base objects (so either direct or prep. objects)
        elif pos in ["BASE_OBJECT"]:
            check_dep = (word_dep in preset.dep_object)
            check_noun = self._is_pos_word(word=word, pos="NOUN")
            return (check_noun and check_dep)
        #
        #Identify direct objects
        elif pos in ["DIRECT_OBJECT"]:
            check_baseobj = self._is_pos_word(word=word, pos="BASE_OBJECT")
            is_conjdirobj = self._is_pos_conjoined(word, pos=pos)
            #Check preceding term is a verb
            check_afterprep = False
            check_afterverb = False
            for pre_node in word_ancestors:
                #If preceding preposition found first
                if self._is_pos_word(word=pre_node, pos="PREPOSITION"):
                    check_afterprep = True
                    break
                #If preceding verb found first
                elif self._is_pos_word(word=pre_node, pos="VERB"):
                    check_afterverb = True
                    break
            #
            return (((not check_afterprep) and (check_afterverb)
                        and (check_baseobj))
                    or is_conjdirobj)
        #
        #Identify prepositional objects
        elif pos in ["PREPOSITION_OBJECT"]:
            check_baseobj = self._is_pos_word(word=word, pos="BASE_OBJECT")
            is_conjprepobj = self._is_pos_conjoined(word, pos=pos)
            #Check if this word follows preposition
            check_objprep = False
            for pre_node in word_ancestors:
                #If preceding preposition found first
                if self._is_pos_word(word=pre_node, pos="PREPOSITION"):
                    pre_pre_node = list(pre_node.ancestors)[0]
                    #Ensure prepositional object instead of prep. subject
                    check_objprep = (not self._is_pos_word(word=pre_pre_node,
                                                        pos="SUBJECT"))
                    break
                #If preceding verb found first
                elif self._is_pos_word(word=pre_node, pos="VERB"):
                    check_objprep = False
                    break
            #
            return (is_conjprepobj or (check_baseobj and check_objprep))
        #
        #Identify prepositional subjects
        elif pos in ["PREPOSITION_SUBJECT"]:
            check_obj = self._is_pos_word(word=word, pos="BASE_OBJECT")
            is_conjprepsubj = self._is_pos_conjoined(word, pos=pos)
            #Check if this word follows preposition
            check_subjprep = False
            for pre_node in word_ancestors:
                #If preceding preposition found first
                if self._is_pos_word(word=pre_node, pos="PREPOSITION"):
                    pre_pre_node = list(pre_node.ancestors)[0]
                    #Ensure prepositional subject instead of prep. object
                    check_subjprep = self._is_pos_word(word=pre_pre_node,
                                                        pos="SUBJECT")
                    break
                #If preceding verb found first
                elif self._is_pos_word(word=pre_node, pos="VERB"):
                    check_subjprep = False
                    break
            #
            return (is_conjprepsubj or (check_obj and check_subjprep))
        #
        #Identify markers
        elif pos in ["MARKER"]:
            check_dep = (word_dep in preset.dep_marker)
            check_tag = (word_tag in preset.tag_marker)
            check_marker = (check_dep or check_tag)
            #Check if subject marker after non-root verb
            is_notroot = (len(word_ancestors) > 0)
            is_afterroot = (is_notroot and
                        self._is_pos_word(word=word_ancestors[0],pos="ROOT"))
            check_subjmark = False
            if ((is_notroot) and (not is_afterroot)):
                check_subj = self._is_pos_word(word=word, pos="SUBJECT")
                check_det = self._is_pos_word(word=word, pos="DETERMINANT")
                check_subjmark = (check_det and check_subj)
            #
            return ((check_marker or check_subjmark) and (not is_afterroot))
        #
        #Identify improper X-words (for improper sentences)
        elif pos in ["X"]:
            check_dep = (word_dep in preset.dep_xpos)
            check_pos = (word_pos in preset.pos_xpos)
            return (check_dep or check_pos)
        #
        #Identify conjoined words
        elif pos in ["CONJOINED"]:
            check_dep = (word_dep in preset.dep_conjoined)
            return (check_dep)
        #
        #Identify determinants
        elif pos in ["DETERMINANT"]:
            check_pos = (word_pos in preset.pos_determinant)
            check_tag = (word_tag in preset.tag_determinant)
            return (check_pos and check_tag)
        #
        #Identify aux
        elif pos in ["AUX"]:
            check_dep = (word_dep in preset.dep_aux)
            check_pos = (word_pos in preset.pos_aux)
            check_prep = (word_tag in preset.tag_preposition)
            check_num = (word_tag in preset.tag_number)
            #
            tags_approved = (preset.tag_verb_past + preset.tag_verb_present
                            + preset.tag_verb_future + preset.tag_verb_purpose)
            check_approved = (word_tag in tags_approved)
            #
            return ((check_dep and check_pos and check_approved)
                    and (not (check_prep or check_num)))
        #
        #Identify nouns
        elif pos in ["NOUN"]:
            check_pos = (word_pos in preset.pos_noun)
            return (check_pos)
        #
        #Identify pronouns
        elif pos in ["PRONOUN"]:
            check_tag = (word_tag in preset.tag_pronoun)
            check_pos = (word_pos in preset.pos_pronoun)
            return (check_tag or check_pos)
        #
        #Identify adjectives
        elif pos in ["ADJECTIVE"]:
            check_adjverb = ((word_dep in preset.dep_adjective)
                                and (word_pos in preset.pos_verb)
                                and (word_tag in preset.tag_verb_any))
            check_pos = (word_pos in preset.pos_adjective)
            check_tag = (word_tag in preset.tag_adjective)
            return (check_tag or check_pos or check_adjverb)
        #
        #Identify  conjunctions
        elif pos in ["CONJUNCTION"]:
            check_pos = (word_pos in preset.pos_conjunction)
            check_tag = (word_tag in preset.tag_conjunction)
            return (check_pos and check_tag)
        #
        #Identify passive verbs and aux
        elif pos in ["PASSIVE"]:
            check_dep = (word_dep in preset.dep_verb_passive)
            return check_dep
        #
        #Identify negative words
        elif pos in ["NEGATIVE"]:
            check_dep = (word_dep in preset.dep_negative)
            return check_dep
        #
        #Identify punctuation
        elif pos in ["PUNCTUATION"]:
            check_punct = (word_dep in preset.dep_punctuation)
            check_letter = bool(re.search(".*[a-z|0-9].*", word_text,
                                            flags=re.IGNORECASE))
            return (check_punct and (not check_letter))
        #
        #Identify punctuation
        elif pos in ["BRACKET"]:
            check_brackets = (word_tag in (preset.tag_brackets))
            return (check_brackets)
        #
        #Identify possessive markers
        elif pos in ["POSSESSIVE"]:
            check_possessive = (word_tag in preset.tag_possessive)
            return (check_possessive)
        #
        #Identify numbers
        elif pos in ["NUMBER"]:
            check_number = (word_pos in preset.pos_number)
            return (check_number)
        #
        #Otherwise, raise error if given pos is not recognized
        else:
            raise ValueError("Err: {0} is not a recognized part of speech."
                            .format(pos))
        #
    #

    ##Method: _process_database_ambig()
    ##Purpose: Process database of ambig. phrases into lookups and dictionary
    def _process_database_ambig(self, keyword_objs=None, do_verbose=False):
        """
        Method: _process_database_ambig
        WARNING! This method is *not* meant to be used directly by users.
        Purpose:
         - Process database of ambiguous keyword phrases into dictionary of keywords, regular expressions, boolean verdicts, etc.
        """
        #Load the ambig. phrase data
        lookup_ambigs = [str(item).lower() for item in
                    np.genfromtxt(preset.filepath_keywords_ambig,
                                comments="#", dtype=str)]
        data_ambigs = np.genfromtxt(preset.filepath_phrases_ambig,
                                comments="#", dtype=str, delimiter="\t"
                                )
        if (len(data_ambigs.shape) == 1): #If single row, reshape to 2D
            data_ambigs = data_ambigs.reshape(1, data_ambigs.shape[0])
        num_ambigs = data_ambigs.shape[0]
        #
        str_anymatch_ambig = preset.string_anymatch_ambig.lower()
        #
        ind_keyword = 0
        ind_phrase = 1
        ind_bool = 2
        #

        #Initialize containers for processed ambig. data
        list_kw_ambigs = []
        list_exp_ambigs = []
        list_bool_ambigs = []
        list_text_ambigs = []

        #Convert known ambig. phrase database into dict. of components
        for ii in range(0, num_ambigs):
            #Convert to NLP notation
            curr_text = data_ambigs[ii,ind_phrase]
            curr_NLP = nlp(str(curr_text))

            #Extract current boolean value
            if (data_ambigs[ii,ind_bool].lower().strip()
                            in ["true", "yes", "t", "y"]):
                curr_bool = True
            elif (data_ambigs[ii,ind_bool].lower().strip()
                            in ["false", "no", "f", "n"]):
                curr_bool = False
            else:
                raise ValueError("Err: {0}:{1} in ambig. database not bool!"
                                .format(ii, data_ambigs[ii,ind_bool]))
            #

            #Formulate current regular expression
            curr_roots = self._extract_core_from_phrase(phrase_NLP=curr_NLP,
                                do_verbose=do_verbose, do_skip_useless=False,
                                keyword_objs=keyword_objs)["roots"]
            curr_exp = (r"("
                        + r")( .*)* (".join([(r"(\b"+r"\b|\b".join(item)+r"\b)")
                                        for item in curr_roots])
                        + r")") #Convert to reg. exp. for substring search later
            #

            #Extract current keywords
            curr_kw_raw = data_ambigs[ii,ind_keyword].lower()
            if (str_anymatch_ambig == curr_kw_raw): #Match any keyword
                curr_kw = [item.lower() for item in lookup_ambigs]
            else: #Otherwise, store given keyword
                curr_kw = [curr_kw_raw]
            #

            #Store the extracted data for each keyword
            tmp_num = len(curr_kw)
            list_kw_ambigs += curr_kw
            list_exp_ambigs += [re.sub(str_anymatch_ambig, curr_kw[jj],
                                        curr_exp, flags=re.IGNORECASE)
                                for jj in range(0, tmp_num)]
            list_bool_ambigs += [curr_bool]*tmp_num
            list_text_ambigs += [curr_text]*tmp_num
        #

        #Gather all of the results into a dictionary
        dict_ambigs = {"lookup_ambigs":lookup_ambigs,
                        "all_kw_ambigs":list_kw_ambigs,
                        "all_exp_ambigs":list_exp_ambigs,
                        "all_bool_ambigs":list_bool_ambigs,
                        "all_text_ambigs":list_text_ambigs}
        #

        #Return the processed results
        return dict_ambigs
    #

    ##Method: _search_text()
    ##Purpose: Search text for given keywords and acronyms and return metric
    def _search_text(self, text, keyword_objs, do_verbose=False):
        """
        Method: _search_text
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Return boolean for whether or not given text contains keywords/acronyms from given keyword objects.
        """
        #Check if keywords and/or acronyms present in given text
        check_keywords = any([item.is_keyword(text) for item in keyword_objs])
        #
        #Print some notes
        if do_verbose:
            #Extract global variables
            keywords = [item2
                        for item1 in keyword_objs
                        for item2 in item1._get_info("keywords")]
            acronyms = [item2
                        for item1 in keyword_objs
                        for item2 in item1._get_info("acronyms")]
            #
            print("Completed _search_text().")
            print("Keywords={0}\nAcronyms={1}".format(keywords, acronyms))
            print("Boolean: {0}".format(check_keywords))
        #

        #Return boolean result
        return check_keywords
    #

    ##Method: _streamline_phrase()
    ##Purpose: Cleanse given (short) string of extra whitespace, dashes, etc, and replace websites, etc, with uniform placeholders.
    def _streamline_phrase(self, text):
        """
        Method: _streamline_phrase
        WARNING! This method is *not* meant to be used directly by users.
        Purpose:
         - Run _cleanse_text with citation streamlining.
         - Replace websites with uniform placeholder.
         - Replace some common science abbreviations that confuse external NLP package sentence parsing.
        """
        #Extract global variables
        dict_exp_abbrev = preset.dict_exp_abbrev

        #Remove any initial excessive whitespace
        text = self._cleanse_text(text=text, do_streamline_etal=True)

        #Replace annoying websites with placeholder
        text = re.sub(preset.exp_website, preset.placeholder_website, text)

        #Replace annoying <> inserts (e.g. html)
        text = re.sub(r"<[A-Z|a-z|/]+>", "", text)

        #Replace annoying abbreviations that confuse NLP sentence parser
        for key1 in dict_exp_abbrev:
            text = re.sub(key1, dict_exp_abbrev[key1], text)
        #

        #Replace annoying object numerical name notations
        #E.g.: HD 123456, 2MASS123-456
        text = re.sub(r"([A-Z]+) ?[0-9][0-9]+[A-Z|a-z]*((\+|-)[0-9][0-9]+)*",
                        r"\g<1>"+preset.placeholder_number, text)
        #E.g.: Kepler-123ab
        text = re.sub(r"([A-Z][a-z]+)( |-)?[0-9][0-9]+([A-Z|a-z])*",
                        r"\g<1> "+preset.placeholder_number, text)

        #Remove most obnoxious numeric ranges
        text = re.sub(r"~?[0-9]+([0-9]|\.)* ?- ?[0-9]+([0-9]|\.)*[A-Z|a-z]*\b",
                        "{0}".format(preset.placeholder_numeric), text)

        #Remove spaces between capital+numeric names
        text = re.sub(r"([A-Z]+) ([0-9]+)([0-9]|[a-z])+",
                        r"\1\2\3".format(preset.placeholder_numeric), text)

        #Remove any new excessive whitespace and punctuation spaces
        text = self._cleanse_text(text=text, do_streamline_etal=True)
        #

        #Return streamlined text
        return text
    #
#


##Class: Paper
class Keyword(_Base):
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
    ##Method: __init__
    ##Purpose: Initialize this class instance
    def __init__(self, keywords, acronyms=None, do_verbose=False):
        """
        Method: __init__
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Initialize instance of Keyword class.
        """
        #Initialize storage
        self._storage = {}
        self._store_info(do_verbose, "do_verbose")

        #Cleanse keywords of extra whitespace, punctuation, etc.
        keywords_clean = sorted([self._cleanse_text(text=phrase,
                                                    do_streamline_etal=True)
                            for phrase in keywords],
                            key=(lambda w:len(w)))[::-1] #Sort by desc. length
        #Store keywords
        self._store_info(keywords_clean, key="keywords")

        #Also cleanse+store acronyms, if given
        if (acronyms is not None):
            acronyms_mid = [re.sub(preset.exp_nopunct, "", item,
                                flags=re.IGNORECASE) for item in acronyms]
            #Remove all whitespace
            acronyms_mid = [re.sub(" ", "", item) for item in acronyms_mid]
            acronyms_clean = sorted(acronyms_mid,
                            key=(lambda w:len(w)))[::-1]#Sort by desc. length
            self._store_info(acronyms_clean, key="acronyms")
        else:
            self._store_info([], key="acronyms")
        #

        #Store representative name for this keyword object
        repr_name = (self._get_info("keywords")[::-1]+self._get_info("acronyms")
                    )[0] #Take shortest keyword or longest acronym as repr. name
        self._store_info(repr_name, key="name")
        #

        #Store regular expressions for keywords and acronyms
        exps_k = [(r"\b"+phrase+r"\b") for phrase in self._get_info("keywords")]
        if (acronyms is not None) and (len(acronyms) > 0):
            #Update acronyms to allow optional spaces
            acronyms_upd = [(r"(\.?)( ?)".join(item))
                            for item in self._get_info("acronyms")]
            #Build regular expression to recognize acronyms
            exp_a = (r"(^|[^\.])((\b"
                        + r"\b)|(\b".join([phrase for phrase in acronyms_upd])
                        + r"\b)(\.?))($|[^A-Z|a-z])") #Matches any acronym
        else:
            exp_a = None
        #
        self._store_info(exps_k, key="exps_keywords")
        self._store_info(exp_a, key="exp_acronyms")
        #

        #Close method
        return
    #

    ##Method: __str__
    ##Purpose: Generate string representation of this class instance
    def __str__(self):
        """
        Method: __str__
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Print instance of Keyword class.
        """
        ##Build string of characteristics of this instance
        print_str = ("Keyword Object:\n"
                    + "Name: {0}\n".format(self.get_name())
                    + "Keywords: {0}\n".format(self._get_info("keywords"))
                    + "Acronyms: {0}\n".format(self._get_info("acronyms"))
                    )

        ##Return the completed string for printing
        return print_str
    #

    ##Method: get_name
    ##Purpose: Fetch the representative name for this keyword object
    def get_name(self):
        """
        Method: get_name
        Purpose: Return representative name for this instance.
        Arguments: None
        Returns:
          - [str]
        """
        #Fetch and return representative name
        return self._get_info("name")
    #

    ##Method: is_keyword
    ##Purpose: Check if text matches to this keyword object
    def is_keyword(self, text, mode=None):
        """
        Method: is_keyword
        Purpose: Return whether or not the given text contains terms (keywords, acronyms) matching this instance.
        Arguments:
          - "text" [str]: The text to search within for terms
        Returns:
          - [bool]
        """
        #Fetch global variables
        exps_k = self._get_info("exps_keywords")
        exp_a = self._get_info("exp_acronyms")
        do_verbose = self._get_info("do_verbose")
        allowed_modes = [None, "keyword", "acronym"]
        #

        #Throw error is specified mode not recognized
        if ((mode is not None) and (mode.lower() not in allowed_modes)):
            raise ValueError("Err: Invalid mode '{0}'.\nAllowed modes: {1}"
                            .format(mode, allowed_modes))
        #

        #Check if this text contains keywords
        if ((mode is None) or (mode.lower() == "keyword")):
            check_keywords = any([bool(re.search(item, text,
                            flags=re.IGNORECASE)) for item in exps_k])
        else:
            check_keywords = False
        #
        #Check if this text contains acronyms
        if (((mode is None) or (mode.lower() == "acronym"))
                    and (exp_a is not None)):
            check_acronyms = bool(re.search(exp_a, text, flags=re.IGNORECASE))
        else:
            check_acronyms = False
        #

        #Print some notes
        if do_verbose:
            print("Keywords: {0}\nKeyword regex:\n{1}".format(keywords, exps_k))
            print("Acronyms: {0}\nAcronym regex:\n{1}".format(acronyms, exp_a))
            print("Keyword bool: {0}\nAcronym bool: {1}"
                    .format(check_keywords, check_acronyms))
        #

        #Return booleans
        return (check_acronyms or check_keywords)
    #

    ##Method: replace_keyword
    ##Purpose: Replace any text that matches to this keyword object
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
        #Fetch global variables
        exps_k = self._get_info("exps_keywords")
        exp_a = self._get_info("exp_acronyms")
        if (exp_a is not None):
            exps_a = [exp_a]
        else:
            exps_a = []
        #
        do_verbose = self._get_info("do_verbose")
        text_new = text
        #

        #Replace terms with given placeholder
        #For keywords
        for curr_exp in exps_k:
            text_new = re.sub(curr_exp, placeholder, text_new,
                            flags=re.IGNORECASE)
        #For acronyms: avoids matches to acronyms in larger acronyms
        for curr_exp in exps_a:
            #Group processing below prevents substitution of spaces around acr.
            str_tot = str(re.compile(curr_exp).groups) #Get # of last group
            text_new = re.sub(curr_exp, (r"\1"+placeholder+("\\"+str_tot)),
                            text_new, flags=re.IGNORECASE)
        #

        #Print some notes
        if do_verbose:
            print("Keywords: {0}\nKeyword regex:\n{1}".format(keywords, exps_k))
            print("Acronyms: {0}\nAcronym regex:\n{1}".format(acronyms, exp_a))
            print("Updated text: {0}".format(text_new))
        #

        #Return updated text
        return text_new
    #
#


##Class: Paper
class Paper(_Base):
    """
    Class: Paper
    Purpose:
        - Load in text.
        - Split text into sentences containing target terms, if any found.
        - Gather sentences into 'paragraph'.
    Initialization Arguments:
        - dict_ambigs [None or dict (default=None)]:
          - If None, will load and process external database of ambiguous mission phrases. If given, will use what is given.
        - do_check_truematch [bool]:
          - Whether or not to check that mission phrases found in text are known true vs. false matches. (E.g., 'Edwin Hubble' as false match for the Hubble Space Telescope).
        - keyword_objs [list of Keyword instances]:
          - Target missions; terms will be used to search the text.
        - text [str]:
          - The text to search.
        - do_verbose [bool (default=False)]:
          - Whether or not to print surface-level log information and tests.
        - do_verbose_deep [bool (default=False)]:
          - Whether or not to print inner log information and tests.
    """
    ##Method: __init__
    ##Purpose: Initialize this class instance
    def __init__(self, text, keyword_objs, do_check_truematch, dict_ambigs=None, do_verbose=False, do_verbose_deep=False):
        """
        Method: __init__
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Initialize instance of Paper class.
        """
        ##Initialize global storage variable
        self._storage = {} #Dictionary to hold all information

        #Store information about this paper
        text_original = text
        self._store_info(text_original, key="text_original") #Original text
        self._store_info(keyword_objs, key="keyword_objs") #Keyword groups
        self._store_info(do_verbose, "do_verbose")
        self._store_info(do_verbose_deep, "do_verbose_deep")
        self._store_info(do_check_truematch, key="do_check_truematch")

        ##Process ambig. phrase data, if not given
        if do_check_truematch:
            if (dict_ambigs is None):
                dict_ambigs = self._process_database_ambig()
            #
            lookup_ambigs = dict_ambigs["lookup_ambigs"]
            self._store_info(dict_ambigs, key="dict_ambigs")
            self._store_info(lookup_ambigs, key="lookup_ambigs")
        #

        #Preprocess the data
        #Cleanse extra whitespace, strange chars, etc.
        text_clean = self._streamline_phrase(text=text_original)
        #Split cleansed text into naive sentences
        text_clean_split = self._split_text(text=text_clean)

        #Store the preprocessed text
        self._store_info(text_clean, key="text_clean")
        self._store_info(text_clean_split, key="text_clean_split")

        #Close method
        return
    #

    ##Method: get_paragraphs()
    ##Purpose: Fetch paragraphs for given Keyword instances that were previously stored in this instance
    def get_paragraphs(self, keyword_objs=None):
        """
        Method: get_paragraphs
        Purpose: Fetch and return paragraphs previously assembled for given Keyword instances.
        Arguments:
          - "keyword_objs" [list of Keyword instances, or None (default=None)]: List of Keyword instances for which previously constructed paragraphs will be extracted.
        Returns:
          - dict:
            - keys = Representative names of the Keyword instances.
            - values = The paragraphs corresponding to the Keyword instances.
        """
        #Attempt to access previously extracted paragraphs
        try:
            dict_paragraphs = self._get_info("_paragraphs")
        #Throw an error if no paragraphs extracted yet
        except KeyError:
            errstring = ("Whoa there! Looks like you don't have any paragraphs "
                    +"stored in this class instance yet. Please run the method "
                    +"'process_paragraphs(...)' first.")
            raise ValueError(errstring)
        #

        #Extract paragraphs associated with keyword objects, if given
        if (keyword_objs is not None):
            paragraphs = {key.get_name():dict_paragraphs[key.get_name()]
                            for key in keyword_objs}
        #
        #Otherwise, return all paragraphs
        else:
            paragraphs = dict_paragraphs
        #

        #Return paragraphs
        return paragraphs
    #

    ##Method: process_paragraphs
    ##Purpose: Process paragraph that contains given keywords/verified acronyms
    def process_paragraphs(self, buffer=0, do_overwrite=False):
        """
        Method: process_paragraphs
        Purpose: Assemble collection of sentences (a 'paragraph') that contain references to target missions (as indicated by stored keyword objects).
        Arguments:
          - "buffer" [int (default=0)]: Number of +/- sentences around a sentence containing a target mission to include in the paragraph.
          - "do_overwrite" [bool (default=False)]: Whether or not to overwrite any previously extracted and stored paragraphs.
        Returns: None
        """
        #Extract clean, naively split paragraphs
        text_clean_split = self._get_info("text_clean_split")
        keyword_objs = self._get_info("keyword_objs")
        do_check_truematch = self._get_info("do_check_truematch")
        #

        #If overwrite not allowed, check if paragraphs already extracted+saved
        if (not do_overwrite):
            is_exist = True
            #Check for previously stored paragraphs
            try:
                self._get_info("_paragraphs")
            #Catch error raised if no paragraphs exist
            except KeyError:
                is_exist = False

            #Raise error if previously stored paragraphs after all
            if is_exist:
                errstring = ("Whoa there. You already have paragraphs "
                        +"stored in this class instance, and we don't want you "
                        +"to accidentally overwrite them!\nIf you DO want to "
                        +"overwrite previously extracted paragraphs, please "
                        +"rerun this method with do_overwrite=True.")
                raise ValueError(errstring)
        #

        #Extract paragraphs for each keyword
        dict_paragraphs = {} #Dictionary to hold paragraphs for keyword objects
        dict_setup = {"keyword_objs":keyword_objs, "buffer":buffer} #Parameters
        dict_results_ambig = {item.get_name():None
                                for item in keyword_objs} #To hold ambig. output
        dict_acronym_meanings = {item.get_name():None
                                for item in keyword_objs} #False acr. meanings
        for ii in range(0, len(keyword_objs)):
            #Extract all paragraphs containing keywords/verified acronyms
            tmp_res = self._extract_paragraph(keyword_obj=keyword_objs[ii],
                                                buffer=buffer)
            paragraphs = tmp_res["paragraph"]
            dict_results_ambig[keyword_objs[ii].get_name()
                                ] = tmp_res["ambig_matches"]
            dict_acronym_meanings[keyword_objs[ii].get_name()
                                ] = tmp_res["acronym_meanings"]
            #

            #Store the paragraphs under name of first given keyword
            dict_paragraphs[keyword_objs[ii].get_name()] = paragraphs
        #

        #Store the extracted paragraphs and setup information
        self._store_info(dict_paragraphs, "_paragraphs")
        self._store_info(dict_setup, "_paragraphs_setup")
        self._store_info(dict_results_ambig, "_results_ambig")
        self._store_info(dict_acronym_meanings, "_dict_acronym_meanings")

        #Close this method
        return
    #

    ##Method: _buffer_indices()
    ##Purpose: Apply +/- buffer to given list of indices
    def _buffer_indices(self, indices, buffer, max_index):
        """
        Method: _buffer_indices
        WARNING! This method is *not* meant to be used directly by users.
        Purpose:
          - Add a +/- buffer to each index in a set of indices.
          - Merge buffered indices that have overlapping buffers.
        """
        #
        #Build spans for extent of each buffered index
        spans_buffered_init = [[(ind-buffer), (ind+buffer)] for ind in indices]

        #Merge overlapping spans and truncate at min/max bounds
        spans_buffered_merged = []
        for ii in range(0, len(spans_buffered_init)):
            #First span starts at max([0, first_ind])
            if (ii == 0):
                spans_buffered_merged.append([
                            max([0, spans_buffered_init[0][0]]),
                            spans_buffered_init[0][1]])
                #Terminate loop early if latest span reaches maximum boundary
                if (spans_buffered_merged[-1][1] >= max_index):
                    break
                #
                #Otherwise, skip ahead
                continue
            #

            #For all other spans:
            #If overlap, expand previous span
            if (spans_buffered_merged[-1][1] >= spans_buffered_init[ii][0]):
                spans_buffered_merged[-1][1] = spans_buffered_init[ii][1]
            #
            #Otherwise, append new span
            else:
                spans_buffered_merged.append([spans_buffered_init[ii][0],
                                min([spans_buffered_init[ii][1], max_index])]
                                            ) #Written out to avoid shallow copy
            #

            #Terminate loop early if latest span reaches maximum boundary
            if (spans_buffered_merged[-1][1] >= max_index):
                break
            #
        #

        #Return the buffered index spans
        return spans_buffered_merged
    #

    ##Method: _extract_paragraph()
    ##Purpose: Search for paragraphs that contain target mission terms
    def _extract_paragraph(self, keyword_obj, buffer):
        """
        Method: _extract_paragraph
        WARNING! This method is *not* meant to be used directly by users.
        Purpose:
          - Extract sentences from stored text that contain target terms (based on Keyword instance).
          - Keep sentences that do not have false-matches to external ambiguous phrase database.
          - Buffer sentences if non-zero buffer given.
        """
        #Fetch global variables
        do_verbose = self._get_info("do_verbose")
        do_verbose_deep = self._get_info("do_verbose_deep")
        do_check_truematch = self._get_info("do_check_truematch")
        sentences = np.asarray(self._get_info("text_clean_split"))
        num_sentences = len(sentences)
        #
        #Load ambiguous phrases, if necessary
        if do_check_truematch:
            #Print some notes
            if do_verbose:
                print("do_check_truematch=True, so will verify ambig. phrases.")
            #
            #Load previously stored ambig. phrase data
            dict_ambigs = self._get_info("dict_ambigs")
            lookup_ambigs = dict_ambigs["lookup_ambigs"]
        #

        #Print some notes
        if do_verbose:
            print("Fetching inds of target sentences...")
        #

        #Get indices of sentences that contain any target mission terms
        #For keyword terms
        inds_with_keywords_init = [ii for ii in range(0, num_sentences)
                            if keyword_obj.is_keyword(sentences[ii],
                                                        mode="keyword")]
        #For acronym terms
        inds_with_acronyms = [ii for ii in range(0, num_sentences)
                            if keyword_obj.is_keyword(sentences[ii],
                                                        mode="acronym")]
        #
        #Print some notes
        if do_verbose:
            print(("Found:\n# of keyword sentences: {0}\n"
                +"# of acronym sentences: {1}...")
                .format(len(inds_with_keywords_init), len(inds_with_acronyms)))
        #

        #If only acronym terms found, run a check of possible false meanings
        if (len(inds_with_acronyms) > 0):
            acronym_meanings = self._verify_acronyms(keyword_obj=keyword_obj)
        #Otherwise, set empty
        else:
            acronym_meanings = None
        #

        #If requested, run a check for ambiguous phrases if any ambig. keywords
        if (do_check_truematch and any([keyword_obj.is_keyword(item)
                                        for item in lookup_ambigs])):
            #Print some notes
            if do_verbose:
                print("Verifying ambiguous phrases...")
            #

            #Run ambiguous phrase check on all sentences with keyword terms
            output_truematch = [{"ind":ind,
                        "result":self._check_truematch(text=sentences[ind],
                            keyword_objs=[keyword_obj],dict_ambigs=dict_ambigs)}
                        for ind in inds_with_keywords_init]
            #
            ambig_matches = [item["result"] for item in output_truematch]

            #Keep indices that have true matches
            inds_with_keywords_truematch = [item["ind"]
                                            for item in output_truematch
                                            if (item["result"]["bool"])]
            #

            #Print some notes
            if do_verbose:
                print("Done verifying ambiguous phrases.")
                print("Match output:\n{0}".format(output_truematch))
                print("Indices with true matches:\n{0}"
                        .format(inds_with_keywords_truematch))
                print("Keyword sentences with true matches:\n{0}"
                        .format(sentences[inds_with_keywords_truematch]))
            #
        #
        #Otherwise, set empty
        else:
            output_truematch = None
            ambig_matches = None
            inds_with_keywords_truematch = inds_with_keywords_init #Copy over
        #

        #Take note of if keywords and/or acronyms have matches
        dict_has_matches = {"keywords":(len(inds_with_keywords_truematch) > 0),
                            "acronyms":(len(inds_with_acronyms) > 0)}
        #

        #Pool together unique indices of sentences with keywords, acronyms
        inds_with_terms = list(set().union(inds_with_keywords_truematch,
                                            inds_with_acronyms))
        #

        #Determine buffered sentences, if requested
        if (buffer > 0):
            #Print some notes
            if do_verbose:
                print("Buffering sentences with buffer={0}...".format(buffer))
            #
            ranges_buffered = self._buffer_indices(indices=inds_with_terms,
                                    buffer=buffer, max_index=(num_sentences-1))
            sentences_buffered = [" ".join(sentences[item[0]:item[1]+1])
                                for item in ranges_buffered] #Combined sentences
            #Print some notes
            if do_verbose:
                print("Done buffering sentences.\nRanges = {0}."
                        .format(ranges_buffered))
            #
        #
        #Otherwise, just copy over previous indices
        else:
            ranges_buffered = None
            sentences_buffered = sentences[inds_with_terms].tolist()
        #

        #Return outputs
        return {"paragraph":sentences_buffered, "ambig_matches":ambig_matches,
                "acronym_meanings":acronym_meanings,
                "has_matches":dict_has_matches}
    #

    ##Method: _split_text()
    ##Purpose: Split text into sentences at assumed sentence breaks
    def _split_text(self, text):
        """
        Method: _split_text
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Split given text into sentences based on assumed sentence boundaries.
        """
        #Split by line breaks first
        text_lines = text.split("\n")
        #Split by sentences starting with brackets
        text_flat = [item for phrase in text_lines
                            for item in
                            re.split(preset.exp_splitbracketstarts, phrase)]
        #Split by sentences ending with brackets
        text_flat = [item for phrase in text_flat
                            for item in
                            re.split(preset.exp_splitbracketends, phrase)]
        #Then split by assumed sentence structure
        text_flat = [item for phrase in text_flat
                            for item in
                            re.split(preset.exp_splittext, phrase)]
        #Return the split text
        return text_flat
    #

    ##Method: _verify_acronyms()
    ##Purpose: Find possible meanings of acronyms in text
    def _verify_acronyms(self, keyword_obj):
        """
        Method: _verify_acronyms
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Extract all possible matches from the stored text to the acronyms of the Keyword instance.
        """
        #Fetch global variables
        acronyms = [item.upper() for item in keyword_obj._get_info("acronyms")]
        text = self._get_info("text_original")

        #Build regular expression for all acronyms
        list_exp = [(r"\b"
                    +(r"[a-z]+\b"+preset.exp_acronym_midwords).join(letterset)
                    +r"[a-z]+\b")
                    for letterset in acronyms]
        combined_exp = (r"(?:"+(r")|(?:".join(list_exp))+r")")

        #Search full text for possible acronym meanings
        matches = re.findall(combined_exp, text)
        #Throw error if any tuple entries found (e.g. loophole in regex)
        if any([(isinstance(item, tuple)) for item in matches]):
            raise ValueError("Err: Regex must have holes!\n{0}\n{1}"
                            .format(combined_exp, matches))
        #

        #Return all determined matches
        return matches
    #
#


##Class: Grammar
class Grammar(_Base):
    """
    Class: Grammar
    Purpose:
        - Load in text.
        - Extract 'paragraph' from text using Paper class and Keyword instance.
        - Convert paragraph into grammar tree structure.
        - Use grammar tree structure to simplify, streamline, and/or anonymize paragraph as directed by user.
    Initialization Arguments:
        - buffer [int (default=0)]:
          - Number of +/- sentences around a sentence containing a target mission to include in the paragraph.
        - dict_ambigs [None or dict (default=None)]:
          - If None, will load and process external database of ambiguous mission phrases. If given, will use what is given.
        - do_check_truematch [bool]:
          - Whether or not to check that mission phrases found in text are known true vs. false matches. (E.g., 'Edwin Hubble' as false match for the Hubble Space Telescope).
        - keyword_obj [Keyword instance]:
          - Target mission; terms will be used to search the text.
        - text [str]:
          - Text to process for target terms.
        - do_verbose [bool (default=False)]:
          - Whether or not to print surface-level log information and tests.
    """
    #

    ##Method: __init__
    ##Purpose: Initialize this class instance
    def __init__(self, text, keyword_obj, do_check_truematch, buffer=0, do_verbose=False, do_verbose_deep=False, dict_ambigs=None):
        """
        Method: __init__
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Initialize instance of Grammar class.
        """
        #Initialize storage for this class instance
        self._storage = {}
        #Store inputs for this instance
        self._store_info(text, key="text_original")
        self._store_info(keyword_obj, key="keyword_obj")
        self._store_info(buffer, key="buffer")
        self._store_info(do_verbose, key="do_verbose")
        self._store_info(do_verbose_deep, key="do_verbose_deep")
        #Print some notes
        if do_verbose:
            print("Initializing instance of Grammar class.")
        #

        #Process ambig. phrase data, if not given
        if (do_check_truematch) and (dict_ambigs is None):
            #Print some notes
            if do_verbose:
                print("Processing database of ambiguous phrases...")
            #
            dict_ambigs = self._process_database_ambig()
        #Otherwise, store empty placeholders
        else:
            #Print some notes
            if do_verbose:
                print("No ambiguous phrase processing requested.")
            #
            dict_ambigs = None
        #

        #Extract keyword paragraph from the text
        if do_verbose:
            print("Processing text using the Paper class...")
        #
        paper = Paper(text, keyword_objs=[keyword_obj], dict_ambigs=dict_ambigs,
                        do_check_truematch=do_check_truematch,
                        do_verbose=do_verbose, do_verbose_deep=do_verbose_deep)
        paper.process_paragraphs(buffer=buffer)
        self._store_info(paper, "paper")

        #Close the method
        if do_verbose:
            print("Text process and Paper instance stored.")
            print("Initialization of this Grammar class instance complete.")
        return
    #

    ##Method: get_modifs
    ##Purpose: Return modifs (modified paragraphs), modified to specified modes
    def get_modifs(self, which_modes=None, do_include_forest=False):
        """
        Method: get_modifs
        Purpose: Fetch the modified paragraphs ('modifs') previously assembled and stored within this instance.
        Arguments:
          - "which_modes" [list of str, or None (default=None)]: List of modes for which modifs will be extracted. If None, then all previously assembled and stored modifs will be returned.
        Returns:
          - dict:
            - keys = Names of the modes.
            - values = The modif (the modified paragraph) for each mode.
        """
        #Extract global variables
        forest = self._get_info("forest")
        dict_modifs_orig = self._get_info("modifs")
        do_verbose = self._get_info("do_verbose")
        #Extract all computed modes, if none specified
        if (which_modes is None):
            which_modes = [key for key in forest]
        #
        #Print some notes
        if do_verbose:
            print("\n> Running get_modifs() for modes: {0}".format(which_modes))
        #

        #Extract and return requested modifs
        dict_modifs = {key:dict_modifs_orig[key] for key in which_modes}
        #
        #Tack on grammar information if requested
        if (do_include_forest):
            dict_results = {"modifs":dict_modifs, "_forest":forest}
        else:
            dict_results = dict_modifs
        #
        #Print some notes
        if do_verbose:
            print("Fetched modifs: {0}".format(dict_modifs))
        #
        return dict_results
    #

    ##Method: run_modifications
    ##Purpose: Run submethods to convert paragraphs into custom grammar trees
    def run_modifications(self, which_modes=None):
        """
        Method: run_modifications
        Purpose: Parse paragraphs and process them into grammar structures using various modification schemes.
        Arguments: None
        Returns: None (internal storage updated)
        """
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        lookup_kobj = self._get_info("keyword_obj").get_name()
        if (which_modes is None):
            which_modes = preset.list_preset_modes
        paragraphs = self._get_info("paper").get_paragraphs()[lookup_kobj]
        #Print some notes
        if do_verbose:
            print("\n> Running run_modifications():")
        #

        ##Process the raw text into NLP-text using external NLP packages
        clusters_NLP = self._run_NLP(text=paragraphs)
        num_clusters = len(clusters_NLP) #Num. clusters of sentences
        #Print some notes
        if do_verbose:
            print("{0} NLP-processed clusters. Clusters:\n{1}"
                    .format(num_clusters, clusters_NLP))
        #

        ##Store containers and information
        #Initialize storage for the grammar tree
        ids_wordchunks = [None for ii in range(0, num_clusters)]
        forest = {mode:{ii:None for ii in range(0, num_clusters)}
                for mode in which_modes}
        dict_modifs = {mode:None for mode in which_modes}
        #
        #Store the info in this instance
        self._store_info(clusters_NLP, "clusters_NLP")
        self._store_info(num_clusters, "num_clusters")
        self._store_info(forest, "forest")
        self._store_info(dict_modifs, "modifs")
        self._store_info(ids_wordchunks, "_ids_wordchunks")
        #Print some notes
        if do_verbose:
            print("Internal storage for class instance initialized.\nClusters:")
            for ii in range(0, num_clusters):
                print("> {0}: '{1}'".format(ii, clusters_NLP[ii]))
            print("")
        #

        ##Build grammar structures for NLP-sentences in each cluster
        #Iterate through clusters
        for ii in range(0, num_clusters): #Iterate through NLP-sentences
            #Prepare variables and storage for current cluster
            curr_cluster = clusters_NLP[ii]
            num_sentences = len(curr_cluster)
            num_words = sum([len(item) for item in curr_cluster])
            #Print some notes
            if do_verbose:
                print("\n---------------\n")
                print("Building structure for cluster {2} ({1} words):\n{0}\n"
                        .format(curr_cluster, num_words, ii))
            #

            #Identify word chunks for this NLP-cluster
            ids_wordchunks[ii] = self._set_wordchunks(cluster_NLP=curr_cluster)
            #Print some notes
            if do_verbose:
                print("Word-chunks identified as:\n{0}\n"
                        .format(ids_wordchunks[ii]))
                print("Building grammar structure next...")
            #

            #Examine and store info for each word within this cluster
            curr_struct_verbs = {}
            curr_struct_words = {}
            curr_is_checked = np.zeros(num_words).astype(bool) #Checked words
            #Iterate through sentences within this cluster
            for jj in range(0, num_sentences):
                curr_sentence = curr_cluster[jj]
                #Print some notes
                if do_verbose:
                    print("Working on sentence #{1} of cluster #{0}:\n{2}"
                            .format(ii, jj, curr_sentence))
                #
                #Recursively navigate NLP-tree from the root
                self._recurse_NLP_categorization(node=curr_sentence.root,
                                storage_verbs=curr_struct_verbs,
                                storage_words=curr_struct_words,
                                i_cluster=ii, i_sentence=jj,
                                i_verb=curr_sentence.root.i,
                                chain_i_verbs=[], is_checked=curr_is_checked,
                                verb_side=None, i_headoftrail=None)
                #
                #Print some notes
                if do_verbose:
                    print("Grammar structure for current sentence complete!")
                    print("Sentence {0}: '{1}'".format(jj, curr_sentence))
                    print("Verb-struct.:\n{0}\n\n".format(curr_struct_verbs))
                    print("Word-struct.:")
                    for key1 in curr_struct_words:
                        print("- {0}={1}: {2}".format(
                                        curr_struct_words[key1]["word"].i,
                                        curr_struct_words[key1]["word"],
                                        curr_struct_words[key1]))
            #

            #Print some notes
            if do_verbose:
                print("\n---\nGrammar structure for this cluster complete!")
                print("Verb-struct.:\n{0}\n".format(curr_struct_verbs))
                print("Modifying structure based on given modes ({0})..."
                        .format(which_modes))
            #

            #Generate diff. versions of grammar structure (orig, trim, anon...)
            for curr_mode in which_modes:
                forest[curr_mode][ii] = self._modify_structure(mode=curr_mode,
                                                struct_verbs=curr_struct_verbs,
                                                struct_words=curr_struct_words)
            #
        #

        #Generate final modif across all clusters for each mode
        for curr_mode in which_modes:
            curr_modif = "\n".join([forest[curr_mode][ii]["text_updated"]
                                        for ii in range(0, num_clusters)])
            dict_modifs[curr_mode] = curr_modif
        #

        #Close the method
        if do_verbose:
            print("Modification of grammar structure complete.\n")
            for curr_mode in which_modes:
                print("Mod. structure for mode {0}:\n---\n".format(curr_mode))
                for ii in range(0, num_clusters):
                    print("\nCluster #{0}, mode {1}:".format(ii, curr_mode))
                    print("Updated text: {0}".format(
                                        forest[curr_mode][ii]["text_updated"]))
                    print("---")
            #
            print("\n---------------\n")
        #
        return
    #

    ##Method: _add_aux
    ##Purpose: Add aux to grammar structure
    def _add_aux(self, word, storage_verbs):
        """
        Method: _add_aux
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Characterize and store an aux word within grammar structure.
        """
        #Extract global variables
        do_verbose = self._get_info("do_verbose")
        type_verbs = storage_verbs["verbtype"]
        tenses_main = ["PAST", "PRESENT", "FUTURE"]
        word_tag = word.tag_
        word_dep = word.dep_
        #
        #Part-of-speech (pos) tag markers for tense of aux word
        tags_past = preset.tag_verb_past
        tags_present = preset.tag_verb_present
        tags_future = preset.tag_verb_future
        tags_purpose = preset.tag_verb_purpose
        deps_passive = preset.dep_verb_passive
        #Print some notes
        if do_verbose:
            print("\n> Running _add_aux!")
            print("Word: {0}\nInitial verb types: {1}".format(word, type_verbs))
        #

        #Determine if passive tense and store if applicable
        if ((word_dep in deps_passive) and ("PASSIVE" not in type_verbs)):
            type_verbs.append("PASSIVE")
        #

        #Determine main tense of this aux word
        if (word_tag in tags_past): #For past tense
            tense = "PAST"
        elif (word_tag in tags_present): #For present tense
            tense = "PRESENT"
        elif (word_tag in tags_future): #For future tense
            tense = "FUTURE"
        elif (word_tag in tags_purpose): #For purpose tense
            tense = "PURPOSE"
        else: #Raise error if tense not recognized
            raise ValueError(("Err: Tense {1} of word {0} unrecognized!\n{2}"
                                +"\ndep={3}, pos={4}, tag={5}")
                            .format(word, word_tag, word.sent, word.dep_,
                                    word.pos_, word.tag_))
        #

        #Store main tense of aux if no tenses so far
        is_updated = False
        if (not any([(item in tenses_main) for item in type_verbs])):
            type_verbs.append(tense) #Store aux tense
            is_updated = True #Mark verb types as updated
        #

        #Store purpose tense of aux if given
        if (not is_updated):
            if ((tense == "PURPOSE") and (tense not in type_verbs)):
                type_verbs.append(tense) #Store aux tense
                is_updated = True #Mark verb types as updated
        #

        #Update verb tenses if aux tense supercedes previous values
        if (not is_updated):
            #For past aux: supercedes present
            if ((tense == "PAST") and ("PRESENT" in type_verbs)):
                type_verbs.remove("PRESENT")
                type_verbs.append("PAST")
                is_updated = True #Mark verb types as updated
            #
            #For future aux: supercedes past, present
            elif ((tense == "FUTURE") and ("PRESENT" in type_verbs)):
                type_verbs.remove("PRESENT")
                type_verbs.append("FUTURE")
                is_updated = True #Mark verb types as updated
            elif ((tense == "FUTURE") and ("PAST" in type_verbs)):
                type_verbs.remove("PAST")
                type_verbs.append("FUTURE")
                is_updated = True #Mark verb types as updated
            #
        #

        #Exit the method
        if do_verbose:
            print("\n> Run of _add_aux complete.")
            print("Aux: {0}\nLatest verb types: {1}\n".format(word, type_verbs))
        #
        return
    #

    ##Method: _add_verb
    ##Purpose: Add verb to grammar structure
    def _add_verb(self, word):
        """
        Method: __init__
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Characterize and store a verb within grammar structure.
        """
        #Extract global variables
        tag_verb = word.tag_

        #Initialize dictionary to hold characteristics of this verb
        dict_verb = {"i_verb":word.i, "verb":word, "is_important":False,
                    "i_postverbs":[], "i_branchwords_all":[], "verbtype":[]}

        #Determine tense, etc. as types of this verb
        if tag_verb in preset.tag_verb_present:
            dict_verb["verbtype"].append("PRESENT")
        elif tag_verb in preset.tag_verb_past:
            dict_verb["verbtype"].append("PAST")
        elif tag_verb in preset.tag_verb_future:
            dict_verb["verbtype"].append("FUTURE")
        else:
            raise ValueError(("Err: Tag unrecognized for verb {0}: {1}\n{2}"
                                +"\ndep={3}, pos={4}, tag={5}")
                            .format(word, tag_verb, word.sent,
                                    word.dep_, word.pos_, word.tag_))
        #

        #Return initialized verb dictionary
        return dict_verb
    #

    ##Method: _add_word
    ##Purpose: Add general word to grammar structure
    def _add_word(self, node, i_verb, i_cluster, i_sentence, storage_verbs, storage_words, i_headoftrail):
        """
        Method: _add_word
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Characterize and store a word within grammar structure.
        """
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        text_wordchunk = self._get_wordchunk(node.i, i_sentence=i_sentence,
                                    i_cluster=i_cluster, do_text=True) #Text
        NLP_wordchunk = self._get_wordchunk(node.i, i_sentence=i_sentence,
                                    i_cluster=i_cluster, do_text=False) #NLP
        i_wordchunk = np.array([word.i for word in NLP_wordchunk]) #Just ids
        all_pos_mains = preset.special_pos_main
        trail_pos_main = preset.trail_pos_main
        ignore_pos_main = preset.ignore_pos_main
        #
        #Print some notes
        if do_verbose:
            print("\n> Running _add_word for node: {0}. Wordchunk: {1}."
                    .format(node, text_wordchunk))
        #

        ##Characterize some traits of entire phrase
        #Characterize importance
        res_importance = self._check_importance(text_wordchunk,
                                                version_NLP=NLP_wordchunk)
        #
        #Determine part-of-speech (pos) of main (current) word in wordchunk
        pos_main = None
        for check_pos in all_pos_mains:
            #Keep this pos if valid
            is_pos = self._is_pos_word(word=node, pos=check_pos) #Check this pos
            is_conj = self._is_pos_conjoined(word=node, pos=check_pos)#Conjoined
            if (is_pos or is_conj):
                #Throw error if pos already identified; should just be 1 valid
                if (pos_main is not None):
                #            and ("ROOT" not in [pos_main, check_pos])):
                    raise ValueError(("Err: Multi pos for {2}!: {0}, {1}\n{3}"
                                    +"\ndep={4}, pos={5}, tag={6}")
                                    .format(pos_main, check_pos, node,
                                    node.sent, node.dep_, node.pos_, node.tag_))
                #
                #Otherwise, store this pos
                pos_main = check_pos
        #
        #Throw error if no pos found and not marked to ignore
        if ((pos_main is None)
                    and (not any([(self._is_pos_word(word=node, pos=item))
                                    for item in ignore_pos_main]))):
            if do_verbose:
                print(("No p.o.s. recognized for word: {0} (so likely useless)."
                            +"\ndep={1}, pos={2}, tag={3}\nSentence: {4}")
                            .format(node, node.dep_, node.pos_, node.tag_,
                                    node.sent))
        #
        #Print some notes
        if do_verbose:
            print("Word {0} has pos={1}, importance={2}:."
                    .format(node, pos_main, res_importance))
        #

        ##Generate dictionary of characteristics for each word in chunk
        num_words = len(NLP_wordchunk)
        list_dict_words = [None]*num_words
        #Iterate through words
        for ww in range(0, num_words):
            word = NLP_wordchunk[ww] #Current word in wordchunk
            dict_word = {"i_clausechain":None, "i_clausetrail":None
                        } #Initialize dictionary to hold word information
            list_dict_words[ww] = dict_word #Store ahead of time
            #

            ##Characterize the general word
            #For word itself
            dict_word["word"] = word
            dict_word["wordchunk"] = NLP_wordchunk
            #
            #For importance
            #If important, mark as important
            if res_importance["is_any"]:
                dict_word["is_important"] = True
                dict_word["dict_importance"] = res_importance
                #Mark current verb as important as well
                storage_verbs["is_important"] = True
                if (i_verb in storage_words): #If exists already
                    storage_words[i_verb]["is_important"] = True
            else:
                dict_word["is_important"] = False
                dict_word["dict_importance"] = None
            #
            #For uselessness
            if self._is_pos_word(word, pos="USELESS"):
                dict_word["is_useless"] = True
            else:
                dict_word["is_useless"] = False
            #
            #Apply main part-of-speech (p.o.s.) to each word in wordchunk
            dict_word["pos_main"] = pos_main
            #
            #Additional aux characterization, if applicable
            if (pos_main in ["AUX"]):
                self._add_aux(word, storage_verbs=storage_verbs)
            #
        #
        #Print some notes
        if do_verbose:
            print("Characterized wordchunk '{1}' for word '{0}', with pos={2}."
                    .format(node, NLP_wordchunk, pos_main))
        #

        ##Update or append to the latest word trail
        #NOTE: This trail is for clauses...
        #      ...so that unimportant inner clauses can be trimmed later
        #Print some notes
        if do_verbose:
            print("Storing word chunk in an id-post-trail, if necessary...")
        #
        new_trail = None
        new_headoftrail = i_headoftrail
        i_main = [list_dict_words[ww]["word"].i for ww in range(0, num_words)
                    ].index(node.i) #Wordchunk index for main node
        #
        #If this word chunk necessitates a new trail
        if (pos_main in trail_pos_main):
            #Print some notes
            if do_verbose:
                print("Starting new trail from word: {0}"
                        .format(node))
            #
            #Initialize and fill new trail
            new_trail = [i_wordchunk[ww] for ww in range(0, num_words)]
            #
            #Store this trail in storage for the main word of this chunk
            list_dict_words[i_main]["i_clausetrail"] = new_trail
            #
            #Tack the main word onto the end of the previous trail, if necessary
            if (i_headoftrail is not None):
                storage_words[i_headoftrail]["i_clausetrail"].append(node.i)
                #
                #Copy instance of chain of clauses
                pre_chain = storage_words[i_headoftrail]["i_clausechain"]
                pre_chain.append(node.i)
                list_dict_words[i_main]["i_clausechain"] = pre_chain
            #
            #Otherwise, start tracking new chain
            else:
                list_dict_words[i_main]["i_clausechain"] = [node.i]
            #
            #Update the previous head of trail, regardless
            new_headoftrail = node.i
        #
        #Otherwise, tack entire chunk onto previous trail if exists
        elif (i_headoftrail is not None):
            #Print some notes
            if do_verbose:
                print("No new trail for word: {0}. Appending to previous trail."
                        .format(node))
            #
            for ww in range(0, num_words):
                storage_words[i_headoftrail]["i_clausetrail"].append(
                                                            i_wordchunk[ww])
        #
        #Otherwise, do nothing new
        else:
            #Print some notes
            if do_verbose:
                print("No new trail from word: {0}. Nothing new done."
                        .format(node))
            #
        #

        #Print some notes, if updates occurred
        if any([(item is not None) for item in [new_trail, i_headoftrail]]):
            if do_verbose:
                print("Updated or appended this word chunk to a post-trail.")
                print("Current main id, word: {0}, {1}".format(node.i, node))
                print("Latest trail chain: {0}"
                            .format(list_dict_words[i_main]["i_clausechain"]))
                print("New trail: {0}".format(new_trail))
                print("Head of previous trail: {0}".format(i_headoftrail))
                if (i_headoftrail is not None):
                    print("Updated previous trail: {0}"
                        .format(storage_words[i_headoftrail]["i_clausetrail"]))
                else:
                    print("No previous trail.")
        #

        ##Return word dictionaries
        if do_verbose:
            print("Run of _add_word complete.")
            print("Dictionaries per word:")
            for ww in range(0, num_words):
                print("{0}: {1}".format(NLP_wordchunk[ww], list_dict_words[ww]))
                print("-")
            print("\nLatest verb dictionary: {0}\n".format(storage_verbs))
        #
        return {"dict_words":list_dict_words, "i_headoftrail":new_headoftrail}
    #

    ##Method: _get_wordchunk()
    ##Purpose: Retrieve word chunk assigned the given id (index)
    def _get_wordchunk(self, index, i_sentence, i_cluster, do_text):
        """
        Method: _get_wordchunk
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Fetch the word chunk assigned to word at given index.
        """
        #Extract global variables
        cluster_NLP = self._get_info("clusters_NLP")[i_cluster]
        sentence_NLP = cluster_NLP[i_sentence]
        id_wordchunks = self._get_info("_ids_wordchunks")[i_cluster][i_sentence]
        if (i_sentence > 0):
            index_shifted = (index - sum([len(cluster_NLP[ii])
                                        for ii in range(0, (i_sentence-1+1))]))
        else:
            index_shifted = index
        #

        #Return singular word at this index if no word chunk found
        if (id_wordchunks[index_shifted] is None):
            phrase = [sentence_NLP[index_shifted]]
        #Otherwise, join all words within this word chunk
        else:
            inds = (id_wordchunks == id_wordchunks[index_shifted])
            phrase = np.asarray(sentence_NLP)[inds]
        #

        #Return NLP-word phrase or joined text, as requested
        if do_text: #Return joined text
            return " ".join([item.text for item in phrase]).replace(" - ", "-")
        else: #Return NLP-word form
            return phrase
        #
    #

    ##Method: _modify_structure
    ##Purpose: Modify given grammar structure, following specifications of the given mode
    def _modify_structure(self, struct_verbs, struct_words, mode):
        """
        Method: _modify_structure
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Modify given grammar structure using the specifications of the given mode.
        """
        #Extract global variables
        do_verbose = self._get_info("do_verbose")
        keyword_obj = self._get_info("keyword_obj")
        buffer = self._get_info("buffer")
        allowed_modifications = ["none", "skim", "trim", "anon"] #Implemented
        #

        #Initialize storage for modified versions of grammar structure
        num_words = len(struct_words)
        arr_is_keep = np.ones(num_words).astype(bool)
        arr_text_keep = np.array([struct_words[ii]["word"].text
                                for ii in range(0, num_words)])
        text_updated = " ".join(arr_text_keep) #Starting text
        #
        #Print some notes
        if do_verbose:
            print("\n> Running _modify_structure!")
            print("Number of words: {1}\nRequested mode: {0}"
                    .format(mode, num_words))
        #

        #Fetch the modifications assigned to this mode
        list_mods = (mode.lower().split("_"))
        #Throw error if any modifications not recognized
        if any([(item not in allowed_modifications) for item in list_mods]):
            raise ValueError(("Err: Looks like {0} is not a recognized mode."
                            +" It was split into these modifications:\n{2}\n"
                            +" Allowed modes consist of the following,"
                            +" joined by '_' signs:\n{1}")
                            .format(mode, allowed_modifications, list_mods))
        #

        #Set booleans for which modifications to apply
        do_skim = ("skim" in list_mods)
        do_trim = ("trim" in list_mods)
        do_anon = ("anon" in list_mods)

        #Throw error if a trimming mode was requested with a non-zero buffer
        if (do_trim and buffer > 0):
            raise ValueError(("Err: Mode {0} with 'trim' modification given"
                        +" with a non-zero buffer ({1}). This is not allowed"
                        +" because buffered sentences would likely be trimmed."
                        +" Please rerun with a different mode or buffer of 0.")
                        .format(mode, buffer))
        #

        #Print some notes
        if do_verbose:
            print("Allowed modifications: {0}".format(allowed_modifications))
            print("Assigned modifications: {0}".format(list_mods))
        #

        #Apply modifications
        #For skim: Remove useless words (like adjectives)
        if do_skim:
            #Print some notes
            if do_verbose:
                print("> Applying skim modifications...")
            #
            #Iterate through words
            for ii in range(0, num_words):
                #Remove useless words (e.g., adjectives)
                if struct_words[ii]["is_useless"]:
                    arr_is_keep[ii] = False
                    arr_text_keep[ii] = ""
            #
            #Update latest text with these updates
            text_updated = " ".join(arr_text_keep)
            #
            #Print some notes
            if do_verbose:
                print("skim modifications complete.\nUpdated text:\n{0}\n"
                        .format(text_updated))
            #
        #
        #For trim: Remove clauses without any important information/subclauses
        if do_trim:
            #Print some notes
            if do_verbose:
                print("> Applying trim modifications...")
                print("Iterating through clause chains...")
            #
            #Extract all clause chains
            list_chains = []
            for ii in range(0, num_words):
                curr_set = struct_words[ii]["i_clausechain"] #Current chain
                #Keep chain if not empty and if not already stored
                if ((curr_set is not None) and (curr_set not in list_chains)):
                    list_chains.append(curr_set)
            #
            #Iterate through chains
            for curr_chain_raw in list_chains:
                #Reverse chain order
                curr_chain = curr_chain_raw[::-1]
                #
                #Iterate through heads of clauses in this chain
                for curr_iclause in curr_chain:
                    curr_trail = np.asarray(
                                    struct_words[curr_iclause]["i_clausetrail"])
                    #Mark as unimportant if no important terms within
                    if not any([(struct_words[jj]["is_important"])
                                for jj in curr_trail]):
                        arr_is_keep[curr_trail] = False
                        arr_text_keep[curr_trail] = ""
                    #
                    #Print some notes
                    if do_verbose:
                        print("Considered clause {0} for this text.\nWords: {1}"
                                .format(curr_trail,
                                        [struct_words[jj]["word"]
                                        for jj in curr_trail]))
                        print("Latest is_keep values for these words:\n{0}"
                                .format(arr_is_keep[curr_trail]))
                    #
                #
            #
            #Update latest text with these updates
            text_updated = " ".join(arr_text_keep)
            #
            #Print some notes
            if do_verbose:
                print("trim modifications complete.\nUpdated text:\n{0}\n"
                        .format(text_updated))
            #
        #
        #For anon: Replace mission-specific terms with anonymous placeholder
        if do_anon:
            #Print some notes
            if do_verbose:
                print("> Applying anon modifications...")
            #
            placeholder_anon = preset.placeholder_anon
            #Update latest text with these updates
            text_updated = keyword_obj.replace_keyword(text=text_updated,
                                                placeholder=placeholder_anon)
            #
            #Print some notes
            if do_verbose:
                print("anon modifications complete.\nUpdated text:\n{0}\n"
                        .format(text_updated))
            #
        #

        #Cleanse the text to finalize it
        text_updated = self._streamline_phrase(text=text_updated)
        #

        #Build grammar structures using only kept words
        struct_verbs_updated = {key:struct_verbs[key] for key in struct_verbs
                                if (arr_is_keep[key])} #Copy kept verb storage
        struct_words_updated = {key:struct_words[key] for key in struct_words
                                if (arr_is_keep[key])} #Copy kept word storage
        #

        #Return dictionary containing the updated grammar structures
        if do_verbose:
            print("Run of _modify_structure() complete.")
            print(("Mode: {0}\nUpdated word structure: {1}\n"
                    +"Updated verb structure: {2}\nUpdated text: {3}")
                    .format(mode, struct_words_updated, struct_verbs_updated,
                            text_updated))
        #
        return {"mode":mode, "struct_verbs_updated":struct_verbs_updated,
                "struct_words_updated":struct_words_updated,
                "text_updated":text_updated, "arr_is_keep":arr_is_keep}
    #

    ##Method: _recurse_NLP_categorization
    ##Purpose: Recursively explore each word of NLP-sentence and categorize
    def _recurse_NLP_categorization(self, node, storage_verbs, storage_words, is_checked, i_cluster, i_sentence, i_verb, chain_i_verbs, verb_side, i_headoftrail):
        """
        Method: _recurse_NLP_categorization
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Recursively examine and store information for each word within an NLP-sentence.
        """
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        wordchunk = self._get_wordchunk(index=node.i, i_cluster=i_cluster,
                                        i_sentence=i_sentence, do_text=False)
        #Print some notes
        if do_verbose:
            print(("-"*60)+"\nCURRENT NODE ({1}): {0}".format(node, node.i))
            print("node.dep_ = {0}, node.pos_ = {1}, node tag = {2}"
                    .format(node.dep_, node.pos_, node.tag_))
            if len(list(node.ancestors)) != 0:
                print("Root: {0}".format(list(node.ancestors)[0]))
            print("Wordchunk: '{0}'".format(wordchunk))
            print("Lefts: {0}, Rights: {1}"
                    .format(list(node.lefts), list(node.rights)))
            print("Verb chain: {0}".format(chain_i_verbs))
            print("Check status of node: {0}".format(is_checked[node.i]))
        #


        ##Skip ahead if this word has already been checked
        if is_checked[node.i]:
            #Print some notes
            if do_verbose:
                print("This node has already been checked.  Skipping...")
            #
            #Go ahead and recurse through successors of this node
            #For left nodes
            for left_node in node.lefts:
                self._recurse_NLP_categorization(node=left_node,
                        storage_verbs=storage_verbs, verb_side=verb_side,
                        storage_words=storage_words, is_checked=is_checked,
                        i_cluster=i_cluster, i_sentence=i_sentence,
                        i_verb=i_verb, chain_i_verbs=chain_i_verbs.copy(),
                        i_headoftrail=i_headoftrail)
            #
            #For right nodes
            for right_node in node.rights:
                self._recurse_NLP_categorization(node=right_node,
                        storage_verbs=storage_verbs, verb_side=verb_side,
                        storage_words=storage_words, is_checked=is_checked,
                        i_cluster=i_cluster, i_sentence=i_sentence,
                        i_verb=i_verb, chain_i_verbs=chain_i_verbs.copy(),
                        i_headoftrail=i_headoftrail)
            #
            #Exit the method early
            return
        #


        ##Store characteristics of this word
        #For verbs vs. non-verbs
        is_verb = self._is_pos_word(node, pos="VERB")
        is_root = self._is_pos_word(node, pos="ROOT")
        #
        #If verb or root, create new storage for this verb and its info
        if is_verb:
            storage_verbs[node.i] = self._add_verb(node) #Verb storage
            #Iterate through previous verb chains
            for vv in chain_i_verbs: #Note that root will have empty chain
                #Tack on current verb
                storage_verbs[vv]["i_postverbs"].append(node.i)
            #
            chain_i_verbs.append(node.i) #Tack current verb onto verb chain
            i_verb = node.i #Update index of most recent verb
            verb_side = None #Reset tracking of side of verb
        #
        #Handle special case of incomplete sentences (e.g., root is noun)
        elif is_root:
            storage_verbs[node.i] = {"i_verb":node.i, "verb":node,
                                    "is_important":False, "i_postverbs":[],
                                    "i_branchwords_all":[], "verbtype":[]}
            #Iterate through previous verb chains
            for vv in chain_i_verbs: #Note that root will have empty chain
                #Tack on current verb
                storage_verbs[vv]["i_postverbs"].append(node.i)
            #
            chain_i_verbs.append(node.i) #Tack current verb onto verb chain
            i_verb = node.i #Update index of most recent verb
            verb_side = None #Reset tracking of side of verb
        #
        #Otherwise, store this word underneath latest verb
        else:
            storage_verbs[i_verb]["i_branchwords_all"].append(node.i)
        #
        #For general words
        dict_res = self._add_word(node,storage_verbs=storage_verbs[i_verb],
                                            storage_words=storage_words,
                                            i_verb=i_verb, i_cluster=i_cluster,
                                            i_sentence=i_sentence,
                                            i_headoftrail=i_headoftrail)
        list_maxed_dict_words = dict_res["dict_words"]
        i_headoftrail = dict_res["i_headoftrail"]
        #Set same maximum storage to all words in wordchunk
        for ww in range(0, len(wordchunk)):
            storage_words[wordchunk[ww].i] = list_maxed_dict_words[ww]
            is_checked[wordchunk[ww].i] = True
        #


        ##Check off this node and recurse through successors of this node
        #For left nodes
        for left_node in node.lefts:
            if verb_side is None:
                self._recurse_NLP_categorization(node=left_node,
                    storage_verbs=storage_verbs, storage_words=storage_words,
                    i_cluster=i_cluster, i_sentence=i_sentence,
                    i_verb=i_verb, verb_side="left",
                    chain_i_verbs=chain_i_verbs.copy(), is_checked=is_checked,
                    i_headoftrail=i_headoftrail)
            else:
                self._recurse_NLP_categorization(node=left_node,
                    storage_verbs=storage_verbs, storage_words=storage_words,
                    i_cluster=i_cluster, i_sentence=i_sentence,
                    i_verb=i_verb, verb_side=verb_side,
                    chain_i_verbs=chain_i_verbs.copy(), is_checked=is_checked,
                    i_headoftrail=i_headoftrail)
        #
        #For right nodes
        for right_node in node.rights:
            if verb_side is None:
                self._recurse_NLP_categorization(node=right_node,
                    storage_verbs=storage_verbs, storage_words=storage_words,
                    i_cluster=i_cluster, i_sentence=i_sentence,
                    i_verb=i_verb, verb_side="right",
                    chain_i_verbs=chain_i_verbs.copy(), is_checked=is_checked,
                    i_headoftrail=i_headoftrail)
            else:
                self._recurse_NLP_categorization(node=right_node,
                    storage_verbs=storage_verbs, storage_words=storage_words,
                    i_cluster=i_cluster, i_sentence=i_sentence,
                    i_verb=i_verb, verb_side=verb_side,
                    chain_i_verbs=chain_i_verbs.copy(), is_checked=is_checked,
                    i_headoftrail=i_headoftrail)
        #


        ##Exit the method
        return
    #

    ##Method: _run_NLP()
    ##Purpose: Run natural language processing (NLP) on text using external package
    def _run_NLP(self, text):
        """
        Method: _run_NLP
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Run external natural language processing NLP package on given text.
        """
        #Convert text (or clusters of sentences) into clusters of NLP objects
        #For block of text
        if isinstance(text, str):
            #Run external natural language processing (NLP) package
            clusters_NLP = [list(nlp(text).sents)]
        #For list of texts
        else:
            #Run NLP package for each sentence-cluster in paragraph
            clusters_NLP = [list(nlp(cluster).sents) for cluster in text]
        #

        #Return NLP clusters
        return clusters_NLP
    #

    ##Method: _set_wordchunks()
    ##Purpose: Group nouns into chunks as applicable (e.g., proper nouns)
    def _set_wordchunks(self, cluster_NLP):
        """
        Method: _set_wordchunks
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Assign words to noun chunks, as applicable.
        """
        #Extract global variables
        do_verbose = self._get_info("do_verbose")
        num_sentences = len(cluster_NLP)

        #Initialize container to hold chunk ids for each word in sentence
        ids_wordchunks = [None]*num_sentences
        entries_wordchunks = [None]*num_sentences #For checks of words

        #Print some notes
        if do_verbose:
            print("\n> Running _set_wordchunks()!")
            print("Assigning word chunks for the following cluster: {0}"
                    .format(cluster_NLP))
        #

        #Set individual id for root words; avoids weird nounroot wordchunk issue
        itrack = 0 #Index for tracking incremental increase in ids over cluster
        rshift = 0 #Accumulated index across all words in previous sentences
        for ii in range(0, num_sentences):
            #Extract current sentence
            curr_sentence = cluster_NLP[ii]
            nlp_nounchunks = curr_sentence.noun_chunks
            #Initialize container for current sentence ids
            curr_ids = np.array([None]*len(curr_sentence))
            curr_entries = [None]*len(curr_sentence)
            #
            ids_wordchunks[ii] = curr_ids
            entries_wordchunks[ii] = curr_entries
            #
            #Store root index separately to avoid weird noun-root issues
            curr_ids[curr_sentence.root.i - rshift] = itrack
            itrack += 1
            #

            #Iterate through noun chunks identified by external NLP package
            for nounchunk in nlp_nounchunks:
                #Iterate through words in this chunk
                for word in nounchunk:
                    curr_loc = (word.i - rshift)#Word index, shifted to sentence
                    #Skip words that are deemed useless
                    is_useless = self._is_pos_word(word, pos="USELESS")
                    if (is_useless):
                        if do_verbose:
                            print("Skipping {0} because it seems useless...."
                                    .format(word))
                        continue
                    #
                    #Otherwise, assign chunk id to word, if not already done so
                    if (curr_ids[curr_loc] is None):
                        curr_ids[curr_loc] = itrack
                        curr_entries[curr_loc] = word
                    #
                #
                #Increment id after each word chunk
                itrack += 1
            #

            #Update accumulated index across sentences completed so far
            rshift += len(curr_sentence)
        #

        #Print some notes about the established word chunks, if so desired
        if do_verbose:
            print("Run of _set_wordchunks() complete.")
            print("Cluster: {0}".format(cluster_NLP))
            print("Original NLP-generated word chunks for this cluster: {0}"
                    .format([list(item.noun_chunks) for item in cluster_NLP]))
            print("Final array of chunk ids: {0}".format(ids_wordchunks))
        #

        #Return the established ids
        return ids_wordchunks
    #
#


##Class: _Classifier
class _Classifier(_Base):
    """
    WARNING! This class is *not* meant to be used directly by users.
    -
    Class: _Classifier
    Purpose:
     - Container for common underlying methods used in Classifier_* classes.
     - Purely meant to be inherited by Classifier_* classes.
    -
    """
    ##Method: __init__
    ##Purpose: Initialize this class instance
    def __init__(self):
        """
        Method: __init__
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Initializes instance of _Classifier class.
        """
        #Nothing to see here - inheritance base
        pass
    #

    ##Method: classify_text
    ##Purpose: Base classification; overwritten by inheritance as needed
    def classify_text(self, text):
        """
        Method: classify_text
        WARNING! This method is a placeholder for inheritance by other classes.
        """
        #Nothing to see here - inheritance base
        pass
    #

    ##Method: generate_directory_TVT
    ##Purpose: Split a given dictionary of classified texts into directories containing training, validation, and testing datasets
    def generate_directory_TVT(self, dir_model, fraction_TVT, mode_TVT="uniform", filename_json=None, dict_texts=None, do_shuffle=True, seed=10, do_verbose=None):
        """
        Method: generate_directory_TVT
        Purpose: !!!
        """
        ##Load global variables
        name_folderTVT = [preset.folders_TVT["train"],
                        preset.folders_TVT["validate"],
                        preset.folders_TVT["test"]]
        #
        num_TVT = len(name_folderTVT)
        if (num_TVT != len(fraction_TVT)):
            raise ValueError("Err: fraction_TVT ({0}) needs {1} fractions."
                            .format(fraction_TVT, num_TVT))
        #
        if (do_verbose is None):
            do_verbose = self._get_info("do_verbose")
        #
        #Set random seed, if requested
        if do_shuffle:
            np.random.seed(seed)
        #
        #Print some notes
        if do_verbose:
            print("\n> Running generate_directory_TVT().")
            if do_shuffle:
                print("Random seed set to: {0}".format(seed))
        #

        ##Load in data based on file format
        #For data from .json file
        if (filename_json is not None):
            with openfile(filename_json, 'r') as openfile:
                dataset = json.load(openfile)
        #
        #For data from dictionary
        elif (dict_texts is not None):
            dataset = dict_texts
        #
        #Otherwise, throw error
        else:
            raise ValueError("Err: Please pass in either a .json file or a "
                            +"dictionary of pre-classified texts.")
        #

        ##Determine distribution of classes across dataset
        extracted_classes = [dataset[key]["class"] for key in dataset]
        counter_classes = collections.Counter(extracted_classes)
        name_classes = counter_classes.keys()
        #Print some notes
        if do_verbose:
            print("\nClass breakdown of given dataset:\n{0}\n"
                    .format(counter_classes))
        #

        ##Split indices of classes into training, validation, and testing (TVT)
        fraction_TVT = np.asarray(fraction_TVT) / sum(fraction_TVT) #Normalize
        #For mode where training sets should be uniform in size
        if (mode_TVT.lower() == "uniform"):
            min_count = min(counter_classes.values())
            dict_split = {key:(np.round((fraction_TVT * min_count))).astype(int)
                        for key in name_classes} #Partition per class per TVT
            #Update split to send remaining files into testing datasets
            for curr_key in name_classes:
                curr_max = counter_classes[curr_key]
                curr_used = (dict_split[curr_key][0] + dict_split[curr_key][1])
                dict_split[curr_key][2] = (curr_max - curr_used)
        #
        #For mode where training sets should use fraction of data available
        elif (mode_TVT.lower() == "available"):
            max_count = max(counter_classes.values())
            dict_split = {key:(np.round((fraction_TVT
                                        * counter_classes[key]))).astype(int)
                        for key in name_classes} #Partition per class per TVT
        #
        #Otherwise, throw error if mode not recognized
        else:
            raise ValueError(("Err: The given mode for generating the TVT"
                            +" directory {0} is invalid. Valid modes are: {1}")
                            .format(mode_TVT,
                                    ["uniform", "available"]))
        #
        #Print some notes
        if do_verbose:
            print("Fractions given for TVT split: {0}\nMode requested: {1}"
                    .format(fraction_TVT, mode_TVT))
            print("TVT partition per class:")
            for curr_key in name_classes:
                print("{0}: {1}".format(curr_key, dict_split[curr_key]))
        #

        ##Verify splits add up to original file count
        for curr_key in name_classes:
            if (counter_classes[curr_key] != sum(dict_split[curr_key])):
                raise ValueError("Err: Split did not use all data available!")
        #

        ##Prepare indices for extracting TVT sets per class
        dict_inds = {key:[None for ii in range(0, num_TVT)]
                    for key in name_classes}
        for curr_key in name_classes:
            #Fetch available indices
            curr_inds = np.arange(0, counter_classes[curr_key], 1)
            #Shuffle, if requested
            if do_shuffle:
                np.random.shuffle(curr_inds)
            #
            #Split out the indices
            i_start = 0 #Accumulated place within overarching index array
            for ii in range(0, num_TVT): #Iterate through TVT
                i_end = (i_start + dict_split[curr_key][ii]) #Ending point
                dict_inds[curr_key][ii] = curr_inds[i_start:i_end]
                i_start = i_end #Update latest starting place in array
            #
        #
        #Print some notes
        if do_verbose:
            print("\nIndices split per class, per TVT. Shuffling={0}."
                    .format(do_shuffle))
        #

        ##Build new directories to hold TVT (or throw error if exists)
        #Build model directory, if does not already exist
        if (not os.path.exists(dir_model)):
            os.mkdir(dir_model)
        #
        #Verify TVT directories do not already exist
        if any([os.path.exists(os.path.join(dir_model, item))
                    for item in name_folderTVT]):
            raise ValueError("Err: TVT directories exist in model directory"
                            +" and will not be overwritten. Please remove or"
                            +" change the given model directory (dir_model)."
                            +"\nCurrent dir_model: {0}".format(dir_model))
        #
        #Otherwise, make the directories
        for curr_folder in name_folderTVT:
            os.mkdir(os.path.join(dir_model, curr_folder))
            #Iterate through classes and create subfolder per class
            for curr_key in name_classes:
                os.mkdir(os.path.join(dir_model, curr_folder, curr_key))
        #
        #Print some notes
        if do_verbose:
            print("Created new directories for TVT files.\nStored in: {0}"
                    .format(dir_model))
        #

        ##Save texts to .txt files within class directories
        #Iterate through classes
        for curr_key in name_classes:
            #Fetch all texts within this class
            curr_texts = [dataset[key] for key in dataset
                            if (dataset[key]["class"] == curr_key)]
            #Save each text to assigned TVT
            for ii in range(0, num_TVT): #Iterate through TVT
                curr_filebase = os.path.join(dir_model, name_folderTVT[ii],
                                            curr_key) #TVT path
                #Iterate through texts assigned to this TVT
                for jj in dict_inds[curr_key][ii]:
                    curr_filename = "{0}_{1}_{2}".format("text", curr_key, jj)
                    if (curr_texts[jj]["id"] is not None): #Add id, if given
                        curr_filename += "_{0}".format(curr_texts[jj]["id"])
                    curr_filename += ".txt"
                    #Write this text to new file
                    self._write_text(text=curr_texts[jj]["text"],
                                    filepath=os.path.join(curr_filebase,
                                                            curr_filename))
                #
            #
        #
        #Print some notes
        if do_verbose:
            print("Files saved to new TVT directories.")
        #

        ##Verify that count of saved .txt files adds up to original data count
        for curr_key in name_classes:
            #Count items in this class across TVT directories
            curr_count = sum([len([item2 for item2 in
                        os.listdir(os.path.join(dir_model, item1, curr_key))
                        if (item2.endswith(".txt"))])
                        for item1 in name_folderTVT])
            #
            #Verify count
            if (curr_count != counter_classes[curr_key]):
                raise ValueError("Err: Unequal class count in {0}!\n{1} vs {2}"
                        .format(curr_key, curr_count,counter_classes[curr_key]))
            #
        #

        ##Exit the method
        if do_verbose:
            print("\nRun of generate_directory_TVT() complete.\n---\n")
        #
        return
    #

    ##Method: _load_text
    ##Purpose: Load text from given file
    def _load_text(self, filepath):
        """
        Method: _load_text
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Load text from a given filepath.
        """
        #Load text from file
        with open(filepath, 'r') as openfile:
            text = openfile.read()
        #Return the loaded text
        return text
    #

    ##Method: _process_text
    ##Purpose: Load text and process into modifs using Grammar class
    def _process_text(self, text, keyword_obj, which_mode, do_check_truematch, buffer=0, do_verbose=False):
        """
        Method: _process_text
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Process text into modifs using Grammar class.
        """
        #Generate and store instance of Grammar class for this text
        grammar = Grammar(text, keyword_obj=keyword_obj,
                                    do_check_truematch=do_check_truematch,
                                    do_verbose=do_verbose, buffer=buffer)
        grammar.run_modifications()
        self._store_info(grammar, "grammar")
        #
        #Fetch modifs and grammar information
        set_info = grammar.get_modifs(which_modes=[which_mode],
                                        do_include_forest=True)
        modif = set_info["modifs"][which_mode]
        forest = set_info["_forest"]
        #

        #Return all processed statements
        return {"modif":modif, "forest":forest}
    #

    ##Method: _write_text
    ##Purpose: Write given text to given filepath
    def _write_text(self, text, filepath):
        """
        Method: _write_text
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Write given text to given filepath.
        """
        #Write text to file
        with open(filepath, 'w') as openfile:
            openfile.write(text)
        #Exit the method
        return
    #
#


##Class: Classifier_ML
class Classifier_ML(_Classifier):
    """
    Class: Classifier_ML
    Purpose:
        - Train a machine learning model on text within a directory.
        - Use a trained machine learning model to classify given text.
    Initialization Arguments:
        - class_names [list of str]:
          - Names of the classes used in classification.
        - filepath_model [None or str (default=None)]:
          - Filepath of a model to load, or None to load no models.
        - fileloc_model [None or str (default=None)]:
          - File folder location of model-related information to load, or None to load no information.
        - do_verbose [bool (default=False)]:
          - Whether or not to print surface-level log information and tests.
    """
    ##Method: __init__
    ##Purpose: Initialize this class instance
    def __init__(self, class_names=None, filepath_model=None, fileloc_ML=None, do_verbose=False):
        """
        Method: __init__
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Initializes instance of Classifier_ML class.
        """
        ##Store information about this instance
        self._storage = {} #Dictionary to hold all information
        #Load model and related information, if given
        if (filepath_model is not None):
            load_dict = np.load(filepath_model, allow_pickle=True).item()
            #
            class_names = load_dict["class_names"]
            optimizer = tf_opt.create_optimizer(init_lr=load_dict["init_lr"],
                                num_train_steps=load_dict["num_steps_train"],
                                num_warmup_steps=load_dict["num_steps_warmup"],
                                optimizer_type=load_dict["type_optimizer"])
            model = tf.keras.models.load_model(fileloc_ML,
                                    custom_objects={config.ML_name_optimizer:
                                                    optimizer})
        #
        #Otherwise, store empty placeholder
        else:
            model = None
            load_dict = None
            class_names = None
        #
        #Store the model and related quantities
        self._store_info(class_names, "class_names")
        self._store_info(model, "model")
        self._store_info(load_dict, "dict_info")
        self._store_info(do_verbose, "do_verbose")

        #Exit the method
        return
    #

    ##Method: _build_ML
    ##Purpose: Build an empty ML model
    def _build_ML(self, ml_preprocessor, ml_encoder, frac_dropout, num_dense, activation_dense):
        """
        Method: _build_ML
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Build an empty, layered machine learning (ML) model.
        """
        #Assemble the layers for the empty model
        #NOTE: Structure is:
        #=Text Input -> Preprocessor -> Encoder -> Dropout layer -> Dense layer
        #Text input
        layer_input =tf.keras.layers.Input(shape=(),dtype=tf.string,name="text")
        #Preprocessor
        layer_preprocessor=tfhub.KerasLayer(ml_preprocessor,name="preprocessor")
        #Encoder
        inputs_encoder = layer_preprocessor(layer_input)
        layer_encoder = tfhub.KerasLayer(ml_encoder, trainable=True, name="encoder")
        outputs_encoder = layer_encoder(inputs_encoder)

        #Construct the overall model
        net = outputs_encoder["pooled_output"]
        net = tf.keras.layers.Dropout(frac_dropout)(net)
        net = tf.keras.layers.Dense(num_dense, activation=activation_dense,
                                    name="classifier")(net)
        #

        #Return the completed empty model
        return tf.keras.Model(layer_input, net)
    #

    ##Method: train_ML
    ##Purpose: Train and save an empty ML model
    def train_ML(self, dir_model, name_model, seed, do_verbose=None, do_return_model=False):
        """
        Method: train_ML
        Purpose: Build an empty machine learning (ML) model and train it.
        Arguments:
          - dir_model [str]:
            - File location containing directories of training, validation, and testing data text entries. Model will be saved here.
          - name_model [str]:
            - Base name for this model.
          - do_save [bool (default=False)]:
            - Whether or not to save model and related output.
          - seed [int]:
            - Seed for random number generation.
          - do_verbose [bool (default=False)]:
            - Whether or not to print surface-level log information and tests.
        Returns:
          - dict:
            - 'model': the model itself.
            - 'dict_history': outputs from model training.
            - 'accuracy': accuracy from model training.
            - 'loss': loss from model training.
        """
        #Load global variables
        dir_train = os.path.join(dir_model, preset.folders_TVT["train"])
        dir_validation = os.path.join(dir_model, preset.folders_TVT["validate"])
        dir_test = os.path.join(dir_model, preset.folders_TVT["test"])
        #
        savename_ML = (preset.tfoutput_prefix + name_model)
        savename_model = (name_model + ".npy")
        #
        if (do_verbose is None):
            do_verbose = self._get_info("do_verbose")
        #
        #Load in ML values
        label_mode = config.ML_label_model
        batch_size = config.ML_batch_size
        type_optimizer = config.ML_type_optimizer
        ml_model_key = config.ML_model_key
        frac_dropout = config.ML_frac_dropout
        frac_steps_warmup = config.ML_frac_steps_warmup
        num_epochs = config.ML_num_epochs
        init_lr = config.ML_init_lr
        activation_dense = config.ML_activation_dense
        #

        ##Throw error if model already exists
        if os.path.exists(os.path.join(dir_model, savename_model)):
            raise ValueError("Err: Model already exists, will not overwrite."
                            +"\n{0}, at {1}."
                            .format(savename_model, dir_model))
        #
        elif os.path.exists(os.path.join(dir_model, savename_ML)):
            raise ValueError("Err: ML output already exists, will not overwrite"
                            +".\n{0}, at {1}."
                            .format(savename_ML, dir_model))
        #

        ##Load in the training, validation, and testing datasets
        if do_verbose:
            print("Loading datasets...")
            print("Loading training data...")
        #
        #For training
        dataset_train_raw = tf.keras.preprocessing.text_dataset_from_directory(
                                dir_train, batch_size=batch_size,
                                label_mode=label_mode, seed=seed)
        class_names = dataset_train_raw.class_names
        num_dense = len(class_names)
        dataset_train = dataset_train_raw.cache().prefetch(
                                                buffer_size=tf.data.AUTOTUNE)
        #
        #For validation
        if do_verbose:
            print("Loading validation data...")
        dataset_validation = tf.keras.preprocessing.text_dataset_from_directory(
                                dir_validation, batch_size=batch_size,
                                label_mode=label_mode, seed=seed
                                ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        #
        #For testing
        if do_verbose:
            print("Loading testing data...")
        dataset_test = tf.keras.preprocessing.text_dataset_from_directory(
                                dir_test, batch_size=batch_size,
                                label_mode=label_mode, seed=seed
                                ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        #
        #Print some notes
        if do_verbose:
            print("Done loading datasets.")
        #

        ##Load in the ML model components
        if do_verbose:
            print("Loading ML model components...")
        #
        #Load the preprocessor
        ml_handle_preprocessor =preset.dict_ml_model_preprocessors[ml_model_key]
        ml_preprocessor = tfhub.KerasLayer(ml_handle_preprocessor)
        if do_verbose:
            print("Loaded ML preprocessor: {0}".format(ml_handle_preprocessor))
        #
        ml_handle_encoder = preset.dict_ml_model_encoders[ml_model_key]
        ml_encoder = tfhub.KerasLayer(ml_handle_encoder, trainable=True)
        if do_verbose:
            print("Loaded ML encoder: {0}".format(ml_handle_encoder))
            print("Done loading ML model components.")
        #

        ##Build an empty ML model
        if do_verbose:
            print("Building an empty ML model...")
            print("Dropout fraction: {0}".format(frac_dropout))
            print("Number of Dense layers: {0}".format(num_dense))
        #
        model = self._build_ML(ml_preprocessor=ml_preprocessor,
                        ml_encoder=ml_encoder, frac_dropout=frac_dropout,
                        num_dense=num_dense, activation_dense=activation_dense)
        #
        #Print some notes
        if do_verbose:
            print("Done building an empty ML model.")
        #

        ##Set up the loss, metric, and optimization functions
        if do_verbose:
            print("Setting up loss, metric, and optimization functions...")
        #
        init_loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = [tf.keras.metrics.CategoricalAccuracy("accuracy")]
        stepsize_epoch = tf.data.experimental.cardinality(dataset_train).numpy()
        #
        num_steps_train = stepsize_epoch*num_epochs
        num_steps_warmup = int(frac_steps_warmup * num_steps_train)
        #
        optimizer = tf_opt.create_optimizer(init_lr=init_lr,
                                            num_train_steps=num_steps_train,
                                            num_warmup_steps=num_steps_warmup,
                                            optimizer_type=type_optimizer)
        #
        #Print some notes
        if do_verbose:
            print("# of training steps: {0}\n# of warmup steps: {1}"
                    .format(num_steps_train, num_steps_warmup))
            print("Type of optimizer and initial lr: {0}, {1}"
                    .format(type_optimizer, init_lr))
        #

        ##Compile the model with the loss, metric, and optimization functions
        model.compile(optimizer=optimizer, loss=init_loss, metrics=metrics)
        if do_verbose:
            print("Done compiling loss, metric, and optimization functions.")
            print(model.summary())
        #

        ##Run and evaluate the model on the training and validation data
        if do_verbose:
            print("\nTraining the ML model...")
        history = model.fit(x=dataset_train, validation_data=dataset_validation,
                            epochs=num_epochs)
        #
        if do_verbose:
            print("\nTesting the ML model...")
        res_loss, res_accuracy = model.evaluate(dataset_test)
        #
        if do_verbose:
            print("\nDone training and testing the ML model!")
        #

        ##Save the model
        save_dict = {"loss":res_loss, "class_names":class_names,
                    "accuracy":res_accuracy, "init_lr":init_lr,
                    "num_epochs":num_epochs,
                    "num_steps_train":num_steps_train,
                    "num_steps_warmup":num_steps_warmup,
                    "type_optimizer":type_optimizer}
        model.save(os.path.join(dir_model, savename_ML),
                    include_optimizer=False)
        np.save(os.path.join(dir_model, savename_model), save_dict)
        #
        #Plot the results
        self._plot_ML(model=model, history=history.history, dict_info=save_dict,
                        folder_save=dir_model)
        #

        ##Below Section: Exit the method
        if do_verbose:
            print("\nTraining complete.")
        #
        if do_return_model:
            return model
        else:
            return
    #

    ##Method: _run_ML
    ##Purpose: Run trained ML model on given text
    def _run_ML(self, model, texts, do_verbose=False):
        """
        Method: _run_ML
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Use trained model to classify given text.
        """
        #Run the model on the given texts
        results = model.predict(texts)

        #Print some notes
        if do_verbose:
            for ii in range(0, len(texts)):
                print("{0}:\n{1}\n".format(texts[ii], results[ii]))
        #

        #Return results
        return results
    #

    ##Method: _plot_ML
    ##Purpose: Plot structure and results of ML model
    def _plot_ML(self, model, history, dict_info, folder_save):
        """
        Method: _plot_ML
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Plot recorded loss, accuracy, etc. for a trained model.
        """
        #Extract variables
        num_epochs = dict_info["num_epochs"]

        #For plot of loss and accuracy over time
        #For base plot
        e_arr = np.arange(0, num_epochs, 1)
        fig = plt.figure(figsize=(10,10))
        plt.tight_layout()
        #
        #For loss
        ax = plt.subplot(2, 1, 1)
        ax.set_title("Test loss: {0}\nTest accuracy: {1}"
                    .format(dict_info["loss"], dict_info["accuracy"]))
        ax.plot(e_arr, history["loss"], label="Loss: Training",
                color="blue", linewidth=4)
        ax.plot(e_arr, history["val_loss"], label="Loss: Validation",
                color="gray", linewidth=2)
        leg = ax.legend(loc="best", frameon=False)
        leg.set_alpha(0.5)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        #
        #For accuracy
        ax = plt.subplot(2, 1, 2)
        ax.plot(e_arr, history["accuracy"], label="Accuracy: Training",
                color="blue", linewidth=4)
        ax.plot(e_arr, history["val_accuracy"], label="Accuracy: Validation",
                color="gray", linewidth=2)
        leg = ax.legend(loc="best", frameon=False)
        leg.set_alpha(0.5)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        #
        #Save and close the plot
        plt.savefig(os.path.join(folder_save, "fig_model_lossandacc.png"))
        plt.close()

        #Exit the method
        return
    #

    ##Method: classify_text
    ##Purpose: Classify a single block of text
    def classify_text(self, text, threshold, keyword_obj=None, do_verbose=False, forest=None):
        """
        Method: classify_text
        Purpose: Classify given text using stored machine learning (ML) model.
        Arguments:
          - forest [None (default=None)]:
            - Unused - merely an empty placeholder for uniformity of classify_text across Classifier_* classes. Keep as None.
          - keyword_objs [list of Keyword instances, or None (default=None)]:
            - List of Keyword instances for which previously constructed paragraphs will be extracted.
          - threshold [str]:
            - The minimum uncertainty allowed to return a classification.
          - text [str]:
            - The text to classify.
          - do_verbose [bool (default=False)]:
            - Whether or not to print surface-level log information and tests.
        Returns:
          - dict:
            - 'verdict': the classification.
            - 'scores_comb': the final score.
            - 'scores_indiv': the individual scores.
            - 'uncertainty': the uncertainty of the classification.
        """
        #Load global variables
        list_classes = self._get_info("class_names") #dict_info")["class_names"]
        if (do_verbose is None):
            do_verbose = self._get_info("do_verbose")
        #
        #Print some notes
        if do_verbose:
            print("\nRunning classify_text for ML classifier:")
            print("Class names from model:\n{0}\n".format(list_classes))
        #

        #Cleanse the text
        text_clean = self._streamline_phrase(text)

        #Fetch and use stored model
        model = self._get_info("model")
        probs = np.asarray(model.predict([text_clean]))[0] #Uncertainties
        dict_uncertainty = {list_classes[ii]:probs[ii]
                            for ii in range(0, len(list_classes))}#Dict. version
        #

        #Determine best verdict
        max_ind = np.argmax(probs)
        max_verdict = list_classes[max_ind]

        #Return low-uncertainty verdict if below given threshold
        if ((threshold is not None) and (probs[max_ind] < threshold)):
            dict_results = preset.dictverdict_lowprob.copy()
            dict_results["uncertainty"] = dict_uncertainty
        #
        #Otherwise, generate dictionary of results
        else:
            dict_results = {"verdict":max_verdict, "scores_comb":None,
                            "scores_indiv":None, "uncertainty":dict_uncertainty}
        #

        #Print some notes
        if do_verbose:
            print("\nMethod classify_text for ML classifier complete!")
            print("Max verdict: {0}\n".format(max_verdict))
        #

        #Return dictionary of results
        return dict_results
    #
#


##Class: Classifier_Rules
class Classifier_Rules(_Classifier):
    """
    Class: Classifier_Rules
    Purpose:
        - Use an internal 'decision tree' to classify given text.
    Initialization Arguments:
        - which_classifs [list of str or None (default=None)]:
          - Names of the classes used in classification. If None, will load from bibcat_constants.py.
        - do_verbose [bool (default=False)]:
          - Whether or not to print surface-level log information and tests.
        - do_verbose_deep [bool (default=False)]:
          - Whether or not to print inner log information and tests.
    """
    ##Method: __init__
    ##Purpose: Initialize this class instance
    def __init__(self, which_classifs=None, do_verbose=False, do_verbose_deep=False):
        ##Initialize storage
        self._storage = {}
        #Store global variables
        #self._store_info(threshold, "threshold")
        #self._store_info(do_treetrim, "do_treetrim")
        #self._store_info(do_filler, "do_filler")
        #self._store_info(do_anoncites, "do_anoncites")
        self._store_info(do_verbose, "do_verbose")
        self._store_info(do_verbose_deep, "do_verbose_deep")
        if which_classifs is None:
            which_classifs = preset.list_default_verdicts_decisiontree
        self._store_info(which_classifs, "which_classifs")

        ##Assemble the fixed decision tree
        decision_tree = self._assemble_decision_tree()
        self._store_info(decision_tree, "decision_tree")

        ##Print some notes
        if do_verbose:
            print("> Initialized instance of Classifier_Rules class.")
            print("Internal decision tree has been assembled.")
            print("NOTE: Decision tree probabilities:\n{0}\n"
                    .format(decision_tree))
            #print("NOTE: {0} rules within the internal decision tree.\n"
            #        .format(len(decision_tree)))
        #

        ##Nothing to see here
        return
    #

    ##Method: _apply_decision_tree
    ##Purpose: Apply a decision tree to a 'nest' dictionary for some text
    def _apply_decision_tree(self, decision_tree, tree_nest):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        which_classifs = self._get_info("which_classifs")
        keys_main = preset.nest_keys_main
        keys_matter = [item for item in keys_main if (item.endswith("matter"))]
        bool_keyword = "is_keyword"
        prefix = "prob_"
        #
        dict_nest = tree_nest.copy()
        for key in keys_main: #Encapsulate any single values into tuples
            if isinstance(dict_nest[key], str) or (dict_nest[key] is None):
                dict_nest[key] = [tree_nest[key]]
            else:
                dict_nest[key] = tree_nest[key]
        #
        #Print some notes
        if do_verbose:
            print("Applying decision tree to the following nest:")
            print(dict_nest)
        #

        ##Reject this nest if no keywords within
        if (not any([bool_keyword in dict_nest[key] for key in keys_matter])):
            #Print some notes
            if do_verbose:
                print("No keywords remaining for this cleaned nest. Skipping.")
            return None
        #

        ##Find matching decision tree branch
        best_branch = None
        for key_tree in decision_tree:
            curr_branch = decision_tree[key_tree]
            #Determine if current branch matches
            is_match = True
            #Iterate through parameters
            for key_param in keys_main:
                #Skip ahead if current parameter allows any value ('is_any')
                if curr_branch[key_param] == "is_any":
                    continue
                #
                #Otherwise, check for exact matching values based on branch form
                elif isinstance(curr_branch[key_param], tuple):
                    #Check for exact matching values
                    if not np.array_equal(np.sort(curr_branch[key_param]),
                                        np.sort(dict_nest[key_param])):
                        is_match = False
                        break #Exit from this branch early
                #
                #Otherwise, check inclusive and excluded ('!') matching values
                elif isinstance(curr_branch[key_param], list):
                    #Check for included matching values
                    #if not all([(item in curr_branch[key_param])
                    #            for item in dict_nest[key_param]]):
                    if ((not all([(item in dict_nest[key_param])
                                for item in curr_branch[key_param]
                                if (not item.startswith("!"))]))
                            or (any([(item in dict_nest[key_param])
                                        for item in curr_branch[key_param]
                                        if (item.startswith("!"))]))):
                        is_match = False
                        break #Exit from this branch early
                #
                #Otherwise, check if any of allowed values contained
                elif isinstance(curr_branch[key_param], set):
                    #Check for any of matching values
                    #if not any([(item in curr_branch[key_param])
                    #            for item in dict_nest[key_param]]):
                    if not any([(item in dict_nest[key_param])
                                for item in curr_branch[key_param]]):
                        is_match = False
                        break #Exit from this branch early
                #
                #Otherwise, throw error if format not recognized
                else:
                    raise ValueError("Err: Invalid format for {0}!"
                                    .format(curr_branch))
                #
            #
            #Store this branch as match, if valid
            if is_match:
                best_branch = curr_branch
                #Print some notes
                if do_verbose:
                    print("\n- Found matching decision branch:\n{0}"
                            .format(best_branch))
                #
                break
            #Otherwise, carry on
            else:
                pass
            #
        #
        #Raise an error if no matching branch found
        if (not is_match) or (best_branch is None):
            raise ValueError("Err: No match found for {0}!".format(dict_nest))
        #

        ##Extract the probabilities from the branch
        dict_probs = {key:best_branch[prefix+key] for key in which_classifs}

        ##Return the final probabilities
        if do_verbose:
            print("Final scores computed for nest:\n{0}\n\n{1}\n\n{2}"
                    .format(dict_nest, dict_probs, best_branch))
        #
        return {"probs":dict_probs, "components":best_branch}
    #

    ##Method: _assemble_decision_tree
    ##Purpose: Assemble base of decision tree, with probabilities, that can be read from/expanded as full decision tree
    def _assemble_decision_tree(self):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        which_classifs = self._get_info("which_classifs")
        dict_possible_values = preset.dict_tree_possible_values
        #dict_valid_combos = preset.dict_tree_valid_value_combinations
        keys_matter = preset.nest_keys_matter
        key_verbtype = preset.nest_key_verbtype
        all_params = list(dict_possible_values.keys())
        prefix = "prob_"
        #


        ##Goal: Generate matrix of probabilities based on 'true' examples
        if True: #Just to make it easier to hide the example content display
            dict_examples_base = {}
            itrack = -1
            #
            #<know verb classes>
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":"is_any",
                "objectmatter":"is_any",
                "verbclass":{"know"},
                "verbtypes":"is_any",
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #<be,has verb classes>
            #'OBJ data has/are/had/were available in the archive.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":tuple([]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #'The stars have/are/had/were available in OBJ data.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple([]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #'We have/had OBJ data/Our rms is/was small for the OBJ data.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #’The data is/was from OBJ by Authorsetal.’
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple([]),
                "objectmatter":tuple(["is_keyword", "is_etal"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.5}
            #
            #’The data is from OBJ by Authorsetal.’
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple([]),
                "objectmatter":tuple(["is_keyword", "is_etal"]),
                "verbclass":{"be"},
                "verbtypes":{"PRESENT"},
                "prob_science":0.0,
                "prob_data_influenced":0.8,
                "prob_mention":0.6}
            #
            #’We know/knew the limits of the OBJ data.’
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"know"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.5}
            #
            #’!.’
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple([]),
                "objectmatter":tuple(["is_keyword", "is_pron_1st", "is_etal"]),
                "verbclass":{"know"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.25,
                "prob_mention":0.5}
            #
            #<science,plot verb classes>
            #'OBJ data shows/detects/showed/detected a trend.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":tuple([]),
                "verbclass":{"plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.4,
                "prob_data_influenced":0.5,
                "prob_mention":0.6}
            #
            #'People use/used OBJ data.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple([]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.4,
                "prob_data_influenced":0.5,
                "prob_mention":0.6}
            #
            #'We detect/plot/detected/plotted OBJ data.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #'This work detects/plots our OBJ data.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple([]),
                "objectmatter":tuple(["is_pron_1st", "is_keyword"]),
                "verbclass":{"plot", "science"},
                "verbtypes":{"PRESENT"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #'Figures 1-10 detect/plot/detected/plotted OBJ data.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_term_fig"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #'This work detects/plots the OBJ data in Figure 1.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple([]),
                "objectmatter":tuple(["is_term_fig", "is_keyword"]),
                "verbclass":{"plot", "science"},
                "verbtypes":{"PRESENT"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #'We detect/plot/detected/plotted the OBJ data in Figure 1.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_term_fig", "is_keyword"]),
                "verbclass":{"plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #'Authorsetal detect/plot/detected/plotted OBJ data.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_etal"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.1,
                "prob_mention":1.0}
            #
            #'Authorsetal detect/plot/detected/plotted the OBJ data in Fig 1.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_etal"]),
                "objectmatter":tuple(["is_term_fig", "is_keyword"]),
                "verbclass":{"plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":1.0,
                "prob_mention":0.5}
            #
            #'The data shows trends for the OBJ data by Authorsetal in Fig 1.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple([]),
                "objectmatter":tuple(["is_term_fig", "is_keyword", "is_etal"]),
                "verbclass":{"plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":1.0,
                "prob_mention":0.5}
            #
            #'The trend uses/plots/used/plotted the OBJ data from Authorsetal.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple([]),
                "objectmatter":tuple(["is_etal", "is_keyword"]),
                "verbclass":{"plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":1.0,
                "prob_mention":0.5}
            #
            #'Their OBJ observations detects/detected the star.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword", "is_pron_3rd"]),
                "objectmatter":tuple([]),
                "verbclass":{"science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.2,
                "prob_mention":0.6}
            #
            #'They plot/plotted/detect/detected stars in their OBJ data.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_pron_3rd"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.2,
                "prob_mention":0.6}
            #
            #'Their OBJ observations detects/detected the star.'
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword", "is_pron_3rd"]),
                "objectmatter":tuple([]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.2,
                "prob_mention":0.6}
            #
            #<Data-influenced stuff>
            #’We simulate/simulated the OBJ data.’
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.8,
                "prob_mention":0.4}
            #
            #’We simulate/simulated the OBJ data of Authorsetal.’
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword", "is_etal"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.8,
                "prob_mention":0.6}
            #
            #’Authorsetal simulate/simulated OBJ data in their study.’
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_etal"]),
                "objectmatter":tuple(["is_keyword", "is_pron_3rd"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #<All FUTURE verbs>
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":"is_any",
                "objectmatter":"is_any",
                "verbclass":"is_any",
                "verbtypes":["FUTURE"],
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #<All POTENTIAL verbs>
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":"is_any",
                "objectmatter":"is_any",
                "verbclass":"is_any",
                "verbtypes":["POTENTIAL"],
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #<All PURPOSE verbs>
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":"is_any",
                "objectmatter":"is_any",
                "verbclass":"is_any",
                "verbtypes":["PURPOSE"],
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #<All None verbs>
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":"is_any",
                "objectmatter":"is_any",
                "verbclass":{None},
                "verbtypes":"is_any",
                "prob_science":0.5,
                "prob_data_influenced":0.5,
                "prob_mention":0.5}
            #
            #<All nonverb verbs - usually captions>
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":"is_any",
                "objectmatter":["is_etal"],
                "verbclass":{"root_nonverb"},
                "verbtypes":"is_any",
                "prob_science":0.0,
                "prob_data_influenced":0.8,
                "prob_mention":0.5}
            #
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":["is_etal"],
                "objectmatter":"is_any",
                "verbclass":{"root_nonverb"},
                "verbtypes":"is_any",
                "prob_science":0.0,
                "prob_data_influenced":0.8,
                "prob_mention":0.5}
            #
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":"is_any",
                "objectmatter":"is_any",
                "verbclass":{"root_nonverb"},
                "verbtypes":"is_any",
                "prob_science":0.8,
                "prob_data_influenced":0.5,
                "prob_mention":0.5}
            #
            #
            #Stuff from missing branch output, now with '!' ability: 2023-05-30
            #is_key/is_proI/is_fig; is_etal/is_they
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":["is_pron_1st"],
                "objectmatter":{"is_etal", "is_pron_3rd"},
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.5,
                "prob_data_influenced":0.5,
                "prob_mention":0.5}
            #
            #is_key/is_proI/is_fig; is_proI, !is_etal, !is_they
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":{"is_keyword", "is_pron_1st", "is_term_fig"},
                "objectmatter":["is_pron_1st", "!is_etal", "!is_pron_3rd"],
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #is_key; is_fig, !is_etal, !is_they
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":["is_term_fig", "!is_etal", "!is_pron_3rd"],
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #is_key; is_etal, !is_proI, !is_fig
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":["is_etal", "!is_pron_1st", "!is_term_fig"],
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.5}
            #
            #is_key; is_they, !is_proI, !is_fig
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":["is_pron_3rd", "!is_pron_1st", "!is_term_fig"],
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.5,
                "prob_data_influenced":0.5,
                "prob_mention":0.5}
            #
            #is_fig; is_etal, !is_pron_1st
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_term_fig"]),
                "objectmatter":["is_etal", "!is_pron_1st"],
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.0,
                "prob_data_influenced":0.5,
                "prob_mention":0.5}
            #
            #is_fig, !is_etal, !is_they; is_etal, !is_pron_1st
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":["is_term_fig", "!is_etal", "!is_pron_3rd"],
                "objectmatter":["is_keyword", "!is_etal", "!is_pron_3rd"],
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #is_etal + is_any (not proI, fig), is_any (not proI, fig)
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":["is_etal", "!is_pron_1st", "!is_term_fig"],
                "objectmatter":["!is_pron_1st", "!is_term_fig"],
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #is_they + is_any (not proI, fig), is_any (not proI, fig)
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":["is_pron_3rd", "!is_pron_1st","!is_term_fig"],
                "objectmatter":["!is_pron_1st", "!is_term_fig"],
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #Stuff from missing branch output: 2023-05-25
            #Missing single stuff:
            #is_key; is_key
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #is_key; is_fig
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":tuple(["is_term_fig"]),
                "verbclass":{"be", "has", "plot", "science", "datainfluenced"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #is_etal; is_key
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_etal"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #Multi-combos and subj.obj. duplicates
            #is_key; is_proI, is_key
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":tuple(["is_pron_1st", "is_keyword"]),
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.5,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #is_key; is_proI, is_fig
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":tuple(["is_pron_1st", "is_term_fig"]),
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.5,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #is_key; is_proI, is_they + is_any
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":["is_pron_1st", "is_pron_3rd"],
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.5,
                "prob_data_influenced":0.5,
                "prob_mention":0.5}
            #
            #is_key; is_proI, is_etal + is_any
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":["is_pron_1st", "is_etal"],
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.5,
                "prob_data_influenced":0.5,
                "prob_mention":0.5}
            #
            #is_key; is_proI, is_etal + is_any
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":["is_pron_1st", "is_etal"],
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.5,
                "prob_data_influenced":0.5,
                "prob_mention":0.5}
            #
            #is_key; is_etal, is_key
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":tuple(["is_keyword", "is_etal"]),
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.5}
            #
            #is_key; is_etal, is_fig
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":tuple(["is_term_fig", "is_etal"]),
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.5}
            #
            #is_key; is_etal, is_they
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":tuple(["is_pron_3rd", "is_etal"]),
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.5}
            #
            #is_proI; is_proI, is_key
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_pron_1st", "is_keyword"]),
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.5,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #is_proI; is_etal, is_key
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_etal", "is_keyword"]),
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.0,
                "prob_data_influenced":0.5,
                "prob_mention":0.5}
            #
            #is_proI; is_they, is_key
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_pron_3rd", "is_keyword"]),
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.0,
                "prob_data_influenced":0.5,
                "prob_mention":0.5}
            #
            #is_proI; is_fig, is_key
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_term_fig", "is_keyword"]),
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.5,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #is_proI; is_proI, is_etal, is_key + is_any
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":["is_pron_1st", "is_keyword", "is_etal"],
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.5,
                "prob_data_influenced":0.5,
                "prob_mention":0.5}
            #
            #is_proI; is_proI, is_they, is_key + is_any
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":["is_pron_1st", "is_keyword", "is_pron_3rd"],
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.5,
                "prob_data_influenced":0.5,
                "prob_mention":0.5}
            #
            #is_proI; is_proI, is_fig, is_key + is_any
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":["is_pron_1st", "is_term_fig", "is_keyword"],
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.5,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #is_proI; is_etal, is_they, is_key + is_any
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":["is_etal", "is_keyword", "is_pron_3rd"],
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.5,
                "prob_data_influenced":0.5,
                "prob_mention":0.5}
            #
            #is_proI; is_etal, is_fig, is_key + is_any
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":["is_etal", "is_keyword", "is_term_fig"],
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.5,
                "prob_data_influenced":0.5,
                "prob_mention":0.5}
            #
            #is_proI; is_they, is_fig, is_key + is_any
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":["is_pron_3rd", "is_keyword", "is_term_fig"],
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.5,
                "prob_data_influenced":0.5,
                "prob_mention":0.5}
            #
            #is_etal; is_key + is_any
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_etal"]),
                "objectmatter":["is_keyword"], #tuple(["is_pron_1st", "is_keyword"]),
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.5}
            #
            #is_they; is_key + is_any
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_pron_3rd"]),
                "objectmatter":["is_keyword"], #tuple(["is_pron_1st", "is_keyword"]),
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.5,
                "prob_data_influenced":0.5,
                "prob_mention":0.5}
            #
            #
            #Stuff with empty subj/obj.matter: 2023-05-31
            #is_keyword, is_proI; None; plot, science
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st", "is_keyword"]),
                "objectmatter":tuple([]),
                "verbclass":{"plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.5,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #None; is_keyword; data-infl.; no-verbtype
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"plot"}, #{"be", "has", "plot", "science"},
                "verbtypes":tuple([]),
                "prob_science":0.5,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #None; is_keyword; data-infl.; no-verbtype
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple([]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"datainfluenced"}, #{"be", "has", "plot", "science"},
                "verbtypes":tuple([]),
                "prob_science":0.0,
                "prob_data_influenced":0.5,
                "prob_mention":0.0}
            #
            #None; is_keyword; !data-infl.; no-verbtype
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple([]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":["!datainfluenced"], #{"be", "has", "plot", "science"},
                "verbtypes":tuple([]),
                "prob_science":0.5,
                "prob_data_influenced":0.5,
                "prob_mention":0.5}
            #
            #None; is_etal, !is_proI, !is_fig; no-verbtype
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple([]),
                "objectmatter":["is_etal", "!is_pron_1st", "!is_term_fig"],
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":tuple([]),
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #!is_etal, !is_they; is_proI, !is_etal, !is_they
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":["!is_etal", "!is_pron_3rd"],
                "objectmatter":["is_pron_1st", "!is_etal", "!is_pron_3rd"],
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.5,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #!is_etal, !is_they; is_proI, !is_etal, !is_they
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":["!is_etal", "!is_pron_3rd"],
                "objectmatter":["is_term_fig", "!is_etal", "!is_pron_3rd"],
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.5,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #!is_etal, !is_they; is_proI, !is_etal, !is_they
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":["!is_etal", "!is_pron_3rd"],
                "objectmatter":["is_keyword", "!is_etal", "!is_pron_3rd"],
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.5,
                "prob_data_influenced":0.5,
                "prob_mention":0.5}
            #
            #No-verbtype, all-combos
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":"is_any",
                "objectmatter":"is_any",
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":tuple([]),
                "prob_science":0.5,
                "prob_data_influenced":0.5,
                "prob_mention":0.5}
            #
        #

        ##Generate final base tree with only target classifs and norm. probs.
        decision_tree = {}
        itrack = -1
        #Iterate through base examples
        for key_ex in dict_examples_base:
            curr_ex = dict_examples_base[key_ex]
            curr_denom = np.sum([curr_ex[(prefix+item)]
                                for item in which_classifs]) #Prob. normalizer
            curr_probs = {(prefix+item):(curr_ex[(prefix+item)]/curr_denom)
                            for item in which_classifs}
            #

            ##For main example
            #Extract all parameters and their values
            new_ex = {key:curr_ex[key] for key in all_params}
            #Normalize and store probabilities for target classifs
            new_ex.update(curr_probs)
            #Store this example
            itrack += 1
            decision_tree[itrack] = new_ex
            #

            ##For flipped subj. and obj. matter example
            """
            #Extract all parameters and their values
            new_ex = {key:curr_ex[key] for key in all_params
                    if (key not in keys_matter)}
            #Flip the subject and object terms
            new_ex["subjectmatter"] = curr_ex["objectmatter"]
            new_ex["objectmatter"] = curr_ex["subjectmatter"]
            #Normalize and store probabilities for target classifs
            new_ex.update(curr_probs)
            #Store this example
            itrack += 1
            decision_tree[itrack] = new_ex
            #"""

            ##For passive example
            #For general (not is_any, not set) case
            if ((curr_ex[key_verbtype] == "is_any")):
                #                    or (curr_ex[key_verbtype] is None)):
                pass #No passive counterpart necessary for is_any case
            #For set, add example+passive for each entry in set
            elif isinstance(curr_ex[key_verbtype], set):
                #Iterate through set entries
                for curr_val in curr_ex[key_verbtype]:
                    #Passive with original subj-obj
                    #Extract all parameters and their values
                    """
                    new_ex = {key:curr_ex[key] for key in all_params
                            if (key not in (keys_matter+[key_verbtype]))}
                    #Add in passive term for verbtypes
                    tmp_vals = [curr_val, "PASSIVE"]
                    new_ex[key_verbtype] = tmp_vals
                    #Flip the subject and object terms
                    new_ex["subjectmatter"] = curr_ex["subjectmatter"]
                    new_ex["objectmatter"] = curr_ex["objectmatter"]
                    #Normalize and store probabilities for target classifs
                    new_ex.update(curr_probs)
                    #Store this example
                    itrack += 1
                    decision_tree[itrack] = new_ex
                    #"""
                    #
                    #Passive with flipped subj-obj
                    #Extract all parameters and their values
                    new_ex = {key:curr_ex[key] for key in all_params
                            if (key not in (keys_matter+[key_verbtype]))}
                    #Add in passive term for verbtypes
                    tmp_vals = [curr_val, "PASSIVE"]
                    new_ex[key_verbtype] = tmp_vals
                    #Flip the subject and object terms
                    new_ex["subjectmatter"] = curr_ex["objectmatter"]
                    new_ex["objectmatter"] = curr_ex["subjectmatter"]
                    #Normalize and store probabilities for target classifs
                    new_ex.update(curr_probs)
                    #Store this example
                    itrack += 1
                    decision_tree[itrack] = new_ex
            else:
                #Extract all parameters and their values
                new_ex = {key:curr_ex[key] for key in all_params
                        if (key not in (keys_matter+[key_verbtype]))}
                #Add in passive term for verbtypes
                tmp_vals = list(curr_ex[key_verbtype]) + ["PASSIVE"]
                #Apply old data structure type to new expanded verbtype
                new_ex[key_verbtype]=type(curr_ex[key_verbtype])(tmp_vals)
                #Flip the subject and object terms
                new_ex["subjectmatter"] = curr_ex["objectmatter"]
                new_ex["objectmatter"] = curr_ex["subjectmatter"]
                #Normalize and store probabilities for target classifs
                new_ex.update(curr_probs)
                #Store this example
                itrack += 1
                decision_tree[itrack] = new_ex
        #

        #Return the assembled decision tree
        return decision_tree
    #

    ##Method: _categorize_verb
    ##Purpose: Categorize topic of given verb
    def _categorize_verb(self, i_verb, struct_words):
        ##Extract global variables
        verb_NLP = struct_words[i_verb]["word"]
        verb = verb_NLP.text
        do_verbose = self._get_info("do_verbose")
        list_category_names = preset.list_category_names
        list_category_synsets = preset.list_category_synsets
        list_category_threses = preset.list_category_threses
        max_hyp = preset.max_num_hypernyms
        #root_hypernyms = wordnet.synsets(verb, pos=wordnet.VERB)
        if max_hyp is None:
            root_hypernyms = wordnet.synsets(verb, pos=wordnet.VERB)
        else:
            root_hypernyms = wordnet.synsets(verb, pos=wordnet.VERB)[0:max_hyp]
        num_categories = len(list_category_synsets)

        ##Print some notes
        if do_verbose:
            print("\n> Running _categorize_verb().")
            print("Verb: {0}\nMax #hyp: {1}\nRoot hyp: {2}\nCategories: {3}\n"
                .format(verb, max_hyp, root_hypernyms, list_category_names))
        #

        ##Handle non-verb roots
        if (verb_NLP.dep_ in ["ROOT"]) and (verb_NLP.pos_ in ["NOUN"]):
            if do_verbose:
                print("Verb {0} is a root noun. Marking as such.")
            #
            return preset.category_nonverb_root
        #

        ##Handle specialty verbs
        #For 'be' verbs
        if any([(roothyp in preset.synsets_verbs_be)
                    for roothyp in root_hypernyms]):
            return "be"
        #For 'has' verbs
        elif any([(roothyp in preset.synsets_verbs_has)
                    for roothyp in root_hypernyms]):
            return "has"
        #

        ##Determine likely topical category for this verb
        score_alls = [None]*num_categories
        score_fins = [None]*num_categories
        pass_bools = [None]*num_categories
        #Iterate through the categories
        for ii in range(0, num_categories):
            score_alls[ii] = [roothyp.path_similarity(mainverb)
                        for mainverb in list_category_synsets[ii]
                        for roothyp in root_hypernyms]
            #Take max score, if present
            if len(score_alls[ii]) > 0:
                score_fins[ii] = max(score_alls[ii])
            else:
                score_fins[ii] = 0
            #Determine if this score passes any category thresholds
            pass_bools[ii] = (score_fins[ii] >= list_category_threses[ii])
        #

        ##Throw an error if no categories fit this verb well
        if not any(pass_bools):
            #raise ValueError("Err: No categories fit verb: {0}, {1}"
            #                    .format(verb, score_fins))
            if do_verbose:
                print("No categories fit verb: {0}, {1}\n"
                                .format(verb, score_fins))
            return None
        #

        ##Throw an error if this verb gives very similar top scores
        thres = preset.thres_category_fracdiff
        metric_close_raw = (np.abs(np.diff(np.sort(score_fins)[::-1]))
                            /max(score_fins))
        metric_close = metric_close_raw[0]
        if metric_close < thres:
            #Select most extreme verb with the max score
            tmp_max = max(score_fins)
            if score_fins[list_category_names.index("plot")] == tmp_max:
                tmp_extreme = "plot"
            elif score_fins[list_category_names.index("science")] == tmp_max:
                tmp_extreme = "science"
            else:
                raise ValueError("Reconsider extreme categories for scoring!")
            #
            #Print some notes
            if do_verbose:
                print("Multiple categories with max score: {0}: {1}\n{2}\n{3}"
                        .format(verb, root_hypernyms, score_fins,
                                list_category_names))
                print("Selecting most extreme verb: {0}\n".format(tmp_extreme))
            #Return the selected most-extreme score
            return tmp_extreme
            #raise ValueError("Err: Top scores too close: {0}: {1}\n{2}\n{3}."
            #    .format(verb, root_hypernyms, score_fins,list_category_names))

        ##Return the determined topical category with the best score
        best_category = list_category_names[np.argmax(score_fins)]
        #Print some notes
        if do_verbose:
            print("Best category: {0}\nScores: {1}"
                    .format(best_category, score_fins))
        #Return the best category
        return best_category
    #

    ##Method: _classify_statements
    ##Purpose: Classify a set of statements (rule approach)
    def _classify_statements(self, forest, threshold, do_verbose=None):
        ##Extract global variables
        if do_verbose is not None: #Override do_verbose if specified for now
            self._store_info(do_verbose, "do_verbose")
        #Load the fixed decision tree
        decision_tree = self._get_info("decision_tree")
        #Print some notes
        if do_verbose:
            print("\n> Running _classify_statements.")
            print("Forest word-trees:")
            for key in forest:
                print("{0}".format(forest[key]["modestruct_words_info"].keys()))
                print("{0}\n".format(forest[key]["modestruct_words_info"]))
        #

        ##Set the booleans for this statement dictionary
        forest_nests = self._make_nest_forest(forest)["main"]
        num_trees = len(forest_nests)

        ##Send each statement through the decision tree
        dict_scores = [] #None]*num_trees
        list_comps = [] #None]*num_trees
        for ii in range(0, num_trees):
            #Split this nest into unlinked components
            nests_unlinked = self._unlink_nest(forest_nests[ii])
            #Compute score for each component
            curr_scores = [] #[None]*len(nests_unlinked)
            curr_comps = []
            for jj in range(0, len(nests_unlinked)):
                tmp_stuff = self._apply_decision_tree(
                                            decision_tree=decision_tree,
                                            tree_nest=nests_unlinked[jj])
                if tmp_stuff is not None: #Store if not None
                    curr_scores.append(tmp_stuff["probs"])
                    curr_comps.append(tmp_stuff["components"])
            #
            #Combine the scores
            #dict_scores[ii] = self._combine_unlinked_scores(curr_scores)
            if (len(curr_scores) > 0):
                dict_scores.append(self._combine_unlinked_scores(curr_scores))
                list_comps.append(curr_comps)
        #

        ##Convert the tree scores into a set of verdicts
        resdict = self._convert_scorestoverdict(dict_scores_indiv=dict_scores,
                                                components=list_comps,
                                                threshold=threshold)

        ##Return the dictionary containing verdict, etc. for these statements
        return resdict
    #

    ##Method: classify_text
    ##Purpose: Classify full text based on its statements (rule approach)
    def classify_text(self, keyword_obj, threshold, do_check_truematch=True, which_mode=None, forest=None, text=None, buffer=0, do_verbose=None):
        #Fetch global variables
        if do_verbose is None:
            do_verbose = self._get_info("do_verbose")
        do_verbose_deep = self._get_info("do_verbose_deep")
        #do_filler = self._get_info("do_filler")
        #do_anoncites = self._get_info("do_anoncites")
        #Store/Override latest keyword object and None return of classif.
        self._store_info(keyword_obj, "keyword_obj")

        #Process the text into paragraphs and their statements
        if (forest is None):
            forest = self._process_text(text=text,
                        do_check_truematch=do_check_truematch,
                        keyword_obj=keyword_obj, do_verbose=do_verbose_deep,
                        buffer=buffer, which_mode=which_mode)["forest"]
        #

        #Extract verdict dictionary of statements for keyword object
        dict_results = self._classify_statements(forest, do_verbose=do_verbose,
                                                threshold=threshold)

        #Print some notes
        if do_verbose:
            print("Verdicts complete.")
            print("Verdict dictionary:\n{0}".format(dict_results))
            print("---")

        #Classify total text based on its verdicts across all statements
        #dict_verdicts = self._compile_verdicts(dict_results,
        #                                        do_verbose=do_verbose)

        #Return final verdicts
        return dict_results #dict_verdicts
    #

    ##Method: _combine_unlinked_scores
    ##Purpose: Add together scores from unlinked components of a nest
    def _combine_unlinked_scores(self, component_scores):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        #Print some notes
        if do_verbose:
            print("\n> Running _combine_unlinked_scores.")
            print("Considering set of scores:\n{0}".format(component_scores))
        #

        ##Verify all score keys are the same
        if any([(component_scores[0].keys() != item.keys())
                            for item in component_scores]):
            raise ValueError("Err: Unequal score keys?\n{0}"
                            .format(component_scores))
        #

        ##Combine the scores across all components
        keys_score = list(component_scores[0].keys())
        fin_scores = {key:0 for key in keys_score}
        tot_score = 0 #Total score for normalization purposes
        for ii in range(0, len(component_scores)):
            for key in component_scores[ii]:
                fin_scores[key] += component_scores[ii][key]
                tot_score += component_scores[ii][key]
        #

        ##Normalize the combined scores
        for key in fin_scores:
            if (tot_score == 0): #If empty score, record as 0
                fin_scores[key] = 0
            else: #Otherwise, normalize score
                fin_scores[key] /= tot_score
        #

        ##Return the combined scores
        if do_verbose:
            print("\n> Run of _combine_unlinked_scores complete!")
            print("Combined scores:\n{0}".format(fin_scores))
        #
        return fin_scores
    #

    ##Method: _convert_scorestoverdict
    ##Purpose: Convert set of decision tree scores into single verdict
    def _convert_scorestoverdict(self, dict_scores_indiv, components, threshold, max_diff_thres=0.10, max_diff_count=3, max_diff_verdicts=["science"]):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        #Print some notes
        if do_verbose:
            print("\n> Running _convert_scorestoverdict.")
            print("Individual components and score sets:")
            print("Probability threshold: {0}".format(threshold))
            for ii in range(0, len(components)):
                print("{0}\n{1}\n-".format(components[ii],
                                            dict_scores_indiv[ii]))
            #
        #

        ##Return empty verdict if empty scores
        #For completely empty scores
        if len(dict_scores_indiv) == 0:
            tmp_res = preset.dictverdict_error.copy()
            #Print some notes
            if do_verbose:
                print("\n-Empty scores; verdict: {0}".format(tmp_res))
            #
            return tmp_res
        #
        #Otherwise, remove Nones
        dict_scores_indiv = [item for item in dict_scores_indiv
                            if (item is not None)]
        all_keys = list(dict_scores_indiv[0].keys())
        #

        ##Calculate and store verdict value statistics from indiv. entries
        num_indiv = len(dict_scores_indiv)
        dict_results = {key:{"score_tot_unnorm":0, "count_max":0,
                                "count_tot":num_indiv}
                        for key in all_keys}
        for ii in range(0, num_indiv):
            curr_scores = dict_scores_indiv[ii]
            for curr_key in all_keys:
                tmp_unnorm = curr_scores[curr_key]
                #Determine if current key has max. score across all keys
                if (max_diff_thres is not None) and (tmp_unnorm > 0):
                    tmp_compare = all([(
                                    (np.abs(tmp_unnorm - curr_scores[other_key])
                                                / curr_scores[other_key])
                                        >= max_diff_thres)
                                    for other_key in all_keys
                                    if (other_key != curr_key)]
                                    ) #Check if rel. max. key by some threshold
                #
                else:
                    tmp_compare = False
                #

                #Increment count of sentences with max-valued verdict
                if tmp_compare:
                    dict_results[curr_key]["count_max"] += 1
                #
                #Increment count of statements in general
                #dict_results[curr_key]["count_tot"] += 1
                #
                #Increment unnorm. score count
                dict_results[curr_key]["score_tot_unnorm"] += tmp_unnorm
            #
        #

        #Normalize and store the scores
        denom = np.sum([dict_results[key]["score_tot_unnorm"]
                        for key in all_keys])
        for curr_key in all_keys:
            #Calculate and store normalized score
            tmp_score = (dict_results[curr_key]["score_tot_unnorm"] / denom)
            dict_results[curr_key]["score_tot_norm"] = tmp_score
        #
        list_scores_comb = [dict_results[key]["score_tot_norm"]
                            for key in all_keys]
        #

        #Gather final scores into set of uncertainties
        dict_uncertainties = {key:dict_results[key]["score_tot_norm"]
                                for key in dict_results}
        #

        #Print some notes
        if do_verbose:
            print("Indiv. scores without Nones:\n{0}".format(dict_scores_indiv))
            print("Normalizing denominator: {0}".format(denom))
            print("Full score set:")
            for curr_key in dict_results:
                print("{0}: {1}".format(curr_key, dict_results[curr_key]))
            print("Listed combined scores: {0}: {1}"
                    .format(all_keys, list_scores_comb))
        #

        ##Determine best verdict and associated probabilistic error
        is_found = False
        #For max sentence count:
        if (not is_found) and (max_diff_thres is not None):
            #Check allowed keys that fit max condition
            for curr_key in max_diff_verdicts:
                if (dict_results[curr_key]["count_max"] >= max_diff_count):
                    is_found = True
                    max_score = dict_results[curr_key]["score_tot_norm"]
                    max_verdict = curr_key
                    #
                    #Print some notes
                    if do_verbose:
                        print("\n-Max score: {0}".format(max_score))
                        print("Max verdict: {0}\n".format(max_verdict))
                    #
                    #Break from loop early if found
                    break
            #
        #

        #For max normalized total score:
        if (not is_found):
            max_ind = np.argmax(list_scores_comb)
            max_score = list_scores_comb[max_ind]
            max_verdict = all_keys[max_ind]
            #Print some notes
            if do_verbose:
                print("\n-Max score: {0}".format(max_score))
                print("Max verdict: {0}\n".format(max_verdict))
            #
            #Return verdict only if above given threshold probability
            if (threshold is not None) and (max_score < threshold):
                tmp_res = preset.dictverdict_lowprob.copy()
                tmp_res["scores_indiv"] = dict_scores_indiv
                tmp_res["uncertainty"] = dict_uncertainties
                tmp_res["components"] = components
                #Print some notes
                if do_verbose:
                    print("-No scores above probability threshold.")
                    print("Returning low-prob verdict:\n{0}".format(tmp_res))
                #
                return tmp_res
            #
            #Return low-prob verdict if multiple equal top probabilities
            elif (list_scores_comb.count(max_score) > 1):
                tmp_res = preset.dictverdict_lowprob.copy()
                tmp_res["scores_indiv"] = dict_scores_indiv
                tmp_res["uncertainty"] = dict_uncertainties
                tmp_res["components"] = components
                #Print some notes
                if do_verbose:
                    print("-Multiple top prob. scores.")
                    print("Returning low-prob verdict:\n{0}".format(tmp_res))
                #
                return tmp_res
            #
        #

        ##Assemble and return final verdict
        fin_res = {"verdict":max_verdict, "scores_indiv":dict_scores_indiv,
                "uncertainty":dict_uncertainties, "components":components}
        #
        #Print some notes
        if do_verbose:
            print("-Returning final verdict dictionary:\n{0}".format(fin_res))
        #
        return fin_res
    #

    ##Method: _find_missing_branches
    ##Purpose: Determine and return missing branches in decision tree
    def _find_missing_branches(self, do_verbose=None, cap_iter=3, print_freq=100):
        #Load global variables
        if (do_verbose is None):
            do_verbose = self._get_info("do_verbose")
        decision_tree = self._get_info("decision_tree")
        #
        if do_verbose:
            print("\n\n")
            print(" > Running _find_missing_branches()!")
            print("Loading all possible branch parameters...")
        #
        #Load all possible values
        all_possible_values = preset.dict_tree_possible_values
        solo_values = [None]
        all_subjmatters = [item for item in all_possible_values["subjectmatter"]
                            if (item is not None)]
        all_objmatters = [item for item in all_possible_values["objectmatter"]
                            if (item is not None)]
        all_verbclasses = [item for item in all_possible_values["verbclass"]
                            ] #No shallow copying!
        solo_verbtypes = ["PAST", "PRESENT", "FUTURE"]
        multi_verbtypes = [item for item in all_possible_values["verbtypes"]
                            if ((item is not None)
                                and (item not in solo_verbtypes))]
        #
        if do_verbose:
            print("Done loading all possible branch parameters.")
            print("Generating combinations of branch parameters...")
        #
        #Set caps for multi-parameter combinations
        if (cap_iter is not None):
            cap_subj = cap_iter
            cap_obj = cap_iter
            cap_vtypes = cap_iter
        else:
            cap_subj = len(all_subjmatters)
            cap_obj = len(all_objmatters)
            cap_vtypes = len(multi_verbtypes)
        #

        #Generate set of all possible individual multi-parameter combinations
        sub_subjmatters = [item for ii in range(1, (cap_subj+1))
                            for item in iterer.combinations(all_subjmatters,ii)]
        sub_objmatters = [item for ii in range(1, (cap_obj+1))
                            for item in iterer.combinations(all_objmatters, ii)]
        sub_multiverbtypes = [item for ii in range(1, (cap_vtypes+1))
                            for item in iterer.combinations(multi_verbtypes,ii)]
        #
        sub_allverbtypes = [(list(item_sub)+[item_solo])
                            for item_sub in sub_multiverbtypes
                            for item_solo in solo_verbtypes] #Fold in verb tense
        #

        #Fold in solo and required values as needed
        sets_verbclasses = all_verbclasses
        sets_subjmatters = (sub_subjmatters) # + [None])
        sets_objmatters = (sub_objmatters) # + [None])
        sets_allverbtypes = (sub_allverbtypes) # + [None])
        #
        if do_verbose:
            print("Combinations of branch parameters complete.")
        #

        #Generate set of all possible branches (all possible valid combos)
        list_branches = [{"subjectmatter":sets_subjmatters[aa],
                        "objectmatter":sets_objmatters[bb],
                        "verbclass":sets_verbclasses[cc],
                        "verbtypes":sets_allverbtypes[dd]}
                        for aa in range(0, len(sets_subjmatters))
                        for bb in range(0, len(sets_objmatters))
                        for cc in range(0, len(sets_verbclasses))
                        for dd in range(0, len(sets_allverbtypes))
                        if (((sets_subjmatters[aa] is not None)
                                and ("is_keyword" in sets_subjmatters[aa]))
                            or ((sets_objmatters[bb] is not None)
                                and ("is_keyword" in sets_objmatters[bb]))
                            ) #Must have keyword
                        ]
        #
        num_branches = len(list_branches)
        if do_verbose:
            print("{0} branches generated across all parameter combinations."
                    .format(num_branches))
            print("Extracting branches not covered by decision tree...")
        #

        #Collect branches that are missing from decision tree
        bools_is_missing = [None]*num_branches
        for ii in range(0, num_branches):
            curr_branch = list_branches[ii]
            #Apply decision tree and ensure valid output
            try:
                tmp_res = self._apply_decision_tree(decision_tree=decision_tree,
                                                    tree_nest=curr_branch)
                #
                bools_is_missing[ii] = False #Branch is covered
            #
            #Record this branch as missing if invalid output
            except:
                #Mark this branch as missing
                bools_is_missing[ii] = True
            #
            #Print some notes
            if (do_verbose and ((ii % print_freq) == 0)):
                print("{0} of {1} branches have been checked."
                        .format((ii+1), num_branches))
        #

        #Throw error if any branches not checked
        if (None in bools_is_missing):
            raise ValueError("Err: Branches not checked?")
        #

        #Gather and return missing branches
        if do_verbose:
            print("All branches checked.\n{0} of {1} missing."
                    .format(np.sum(bools_is_missing), num_branches))
            print("Run of _find_missing_branches() complete!")
        #
        return (np.asarray(list_branches)[np.asarray(bools_is_missing)])
    #

    ##Method: _make_nest_forest
    ##Purpose: Construct nest of bools, etc, for all verb-clauses, all sentences
    def _make_nest_forest(self, forest):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        if do_verbose:
            print("\n> Running _make_nest_forest.")
        #
        ##Initialize holder for nests
        num_trees = len(forest["none"])
        list_nests = [None]*num_trees
        list_nest_main = [None]*num_trees
        #Iterate through trees (sentences)
        for ii in range(0, num_trees):
            curr_struct_verbs = forest["none"][ii]["struct_verbs_updated"]
            curr_struct_words = forest["none"][ii]["struct_words_updated"]
            num_branches = len(curr_struct_verbs)
            curr_i_verbs = list(curr_struct_verbs.keys())
            curr_i_words = list(curr_struct_words.keys())
            list_nests[ii] = [None]*num_branches
            #Print some notes
            if do_verbose:
                print("\nCurrent tree: {0} ({0}-{1}).".format(ii, num_trees-1))
                print("\nVerb-tree keys: {0}".format(curr_struct_verbs.keys()))
                print("Verb-tree: {0}\n".format(curr_struct_verbs))
                print("\nWord-tree keys: {0}".format(curr_struct_words.keys()))
                print("Word-tree data: {0}\n".format(curr_struct_words))
            #
            #Iterate through branches (verb-clauses)
            for jj in range(0, num_branches):
                list_nests[ii][jj] = self._make_nest_verbclause(
                                                i_verb=curr_i_verbs[jj],
                                                struct_verbs=curr_struct_verbs,
                                                struct_words=curr_struct_words)
            #

            #Pull representative nest for this tree (at tree root)
            tmp_tomax = [len(curr_struct_verbs[jj]["i_postverbs"])
                                for jj in curr_i_verbs
                                if (curr_struct_verbs[jj]["is_important"])]
            if (len(tmp_tomax) > 0):
                id_main = np.argmax(tmp_tomax)
            #Otherwise, throw error
            else:
                raise ValueError("Err: Nothing important:\n{0}\n\n{1}\n\n{2}"
                        .format(curr_struct_words[list(curr_struct_words.keys())[0]]
                                    ["word"].sent, curr_struct_verbs,
                                    curr_struct_words))
            #
            list_nest_main[ii] = list_nests[ii][id_main]
            #Print some notes
            if do_verbose:
                print("Branch nests complete.")
                print("Individual nests:")
                for jj in range(0, num_branches):
                    print(list_nests[ii][jj])
                    print("")
                print("\nId of main nest: {0}".format(id_main))
            #
        #

        ##Return the completed nests
        return {"all":list_nests, "main":list_nest_main}
    #

    ##Method: _make_nest_verbclause
    ##Purpose: Construct nest of bools, etc, to describe a branch (verb-clause)
    def _make_nest_verbclause(self, struct_verbs, struct_words, i_verb):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        branch_verb = struct_verbs[i_verb]
        lookup_pos = preset.conv_pos_fromtreetonest
        ignore_pos = preset.nest_unimportant_pos
        target_bools = preset.nest_important_treebools
        #Print some notes
        if do_verbose:
            print("\n> Running _make_nest_verbclause.")
            print("Considering verb branch:\n{0}".format(branch_verb))
        #


        ##Build a nest for the given verb
        dict_nest = {"i_verb":i_verb, "subjectmatter":[], "objectmatter":[],
                    "verbtypes":branch_verb["verbtype"],
                    #"verbclass":self._categorize_verb(branch_verb["verb"].text),
                    "verbclass":self._categorize_verb(i_verb=i_verb,
                                                    struct_words=struct_words),
                    "link_verbclass":[], "link_verbtypes":[],
                    "link_subjectmatter":[], "link_objectmatter":[]}
        #

        #Iterate through words directly attached to this verb
        #wordchunk_ids = [item.i for item in struct_words[i_verb]["wordchunk"]]
        tmp_list = list(set((branch_verb["i_branchwords_all"])))
                            #+ struct_words[i_verb]["i_wordchunk"].tolist()))
                            #+ wordchunk_ids))
                        #) #Following words + words in wordchunk; cover all bases
        tmp_list = [item for item in tmp_list if ((item != i_verb) and (struct_words[item]["pos_main"] != "VERB"))]
        for ii in tmp_list:
            #Skip word if not stored (i.e., trimmed word for trimming scheme)
            if ii not in struct_words:
                if do_verbose:
                    print("Word {0} trimmed from word-tree, so skipping..."
                            .format(ii))
                #
                continue
            #

            #Pull current word info
            curr_info = struct_words[ii]
            curr_pos_raw = curr_info["pos_main"]
            if do_verbose:
                print("Considering word {0}.".format(curr_info["word"]))
                print("Has info: {0}.".format(curr_info))
            #

            #Skip if unimportant word
            if (not curr_info["is_important"]):
                if do_verbose:
                    print("Unimportant word. Skipping.")
                #
                continue
            #

            #Skip if unimportant pos
            if (curr_pos_raw in ignore_pos):
                if do_verbose:
                    print("Unimportant pos {0}. Skipping.".format(curr_pos_raw))
                #
                continue
            #

            #Otherwise, convert pos to accepted nest terminology
            try:
                curr_pos = lookup_pos[curr_pos_raw]
            except KeyError:
                #Print context for this error
                print("Upcoming KeyError! Context:")
                print("Word: {0}".format(curr_info["word"]))
                print("Word info: {0}".format(curr_info))
                tmp_sent = np.asarray(list(curr_info["word"].sent))
                print("Sentence: {0}".format(tmp_sent))
                tmp_chunk = struct_words[i_verb]["wordchunk"] #tmp_sent[curr_info["i_wordchunk"]]
                print("Wordchunk:")
                for aa in range(0, len(tmp_chunk)):
                    print("{0}: {1}: {2}, {3}, {4}"
                            .format(aa, tmp_chunk[aa], tmp_chunk[aa].dep_,
                                    tmp_chunk[aa].pos_, tmp_chunk[aa].tag_))
                print("Chosen main pos.: {0}".format(curr_pos_raw))
                #
                curr_pos = lookup_pos[curr_pos_raw]
            #

            #Store target booleans into this nest
            for is_item in target_bools:
                #If this boolean is True, store if not stored previously
                if (curr_info["dict_importance"][is_item]
                                    and (is_item not in dict_nest[curr_pos])):
                    dict_nest[curr_pos].append(is_item)
                #Otherwise, pass
                else:
                    pass
            #
        #

        #Print some notes
        if do_verbose:
            print("\nDone iterating through main (not linked) terms.")
            print("Current state of nest:\n{0}\n".format(dict_nest))
        #

        #Iterate through linked verbs
        for vv in branch_verb["i_postverbs"]:
            #Skip verb if not stored (i.e., trimmed for trimming scheme)
            if vv not in struct_words:
                if do_verbose:
                    print("Word {0} trimmed from word-tree, so skipping..."
                            .format(vv))
                #
                continue
            #
            #Store current verb class if not stored previously
            dict_nest["link_verbtypes"].append(struct_verbs[vv]["verbtype"])
            #link_verbclass = self._categorize_verb(struct_verbs[vv]["verb"].text)
            link_verbclass = self._categorize_verb(
                                            i_verb=struct_verbs[vv]["i_verb"],
                                            struct_words=struct_words)
            dict_nest["link_verbclass"].append(link_verbclass)
            #
            #Prepare temporary dictionary to merge with nest dictionary
            tmp_dict = {"link_subjectmatter":[], "link_objectmatter":[]}
            #
            #Iterate through words attached to linked verbs
            for ii in struct_verbs[vv]["i_branchwords_all"]:
                #Skip word if not stored (i.e., trimmed for trimming scheme)
                if ii not in struct_words:
                    if do_verbose:
                        print("Word {0} trimmed from word-tree, so skipping..."
                                .format(ii))
                    #
                    continue
                #
                #Pull current word info
                curr_info = struct_words[ii]
                curr_pos_raw = curr_info["pos_main"]

                #Skip if unimportant word
                if not curr_info["is_important"]:
                    continue
                #

                #Skip if unimportant pos
                if (curr_pos_raw in ignore_pos):
                    continue
                #

                #Otherwise, convert pos to accepted nest terminology
                curr_pos = lookup_pos[curr_pos_raw]

                #Store target booleans into this nest
                for is_item in target_bools:
                    curr_key = ("link_" + curr_pos)
                    #If this boolean is True, store if not stored previously
                    if (curr_info["dict_importance"][is_item]
                                and (is_item not in tmp_dict[curr_key])):
                        tmp_dict[curr_key].append(is_item)
                    #Otherwise, pass
                    else:
                        pass
                #
            #

            #Merge the dictionary for this clause into overall nest
            for key in tmp_dict:
                dict_nest[key].append(tmp_dict[key])
            #
        #

        ##Return the nest
        if do_verbose:
            print("Nest complete!\n\nVerb branch: {0}\n\nNest: {1}\n"
                    .format(branch_verb, dict_nest))
            print("Run of _make_nest_verbclause complete!\n---\n")
        #
        return dict_nest
    #

    ##Method: _unlink_nest
    ##Purpose: Split a nest into its main and linked components
    def _unlink_nest(self, nest):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        keys_main = preset.nest_keys_main
        keys_matter = [item for item in keys_main if (item.endswith("matter"))]
        keys_nonmatter = [item for item in keys_main
                            if (not item.endswith("matter"))]
        key_matter_obj = "objectmatter"
        prefix_link = preset.nest_prefix_link
        terms_superior = preset.nest_important_treebools_superior
        keys_linked_main = [(prefix_link+key) for key in keys_main]
        keys_linked_matter = [(prefix_link+key) for key in keys_matter]
        num_links = len(nest[keys_linked_matter[0]])
        bool_keyword = "is_keyword"
        bool_pronounI = "is_pron_1st"
        #Print some notes
        if do_verbose:
            print("\n> Running _unlink_nest.")
            print("Considering nest:\n{0}\n".format(nest))
        #

        ##Extract location of keyword in nest
        matters_with_keyword = [key for key in keys_matter
                                if (bool_keyword in nest[key])]
        matters_with_keyword += [key for key in keys_linked_matter
                                if any([(bool_keyword in nest[key][ii])
                                        for ii in range(0, num_links)])]
        matters_with_keyword = [item.replace(prefix_link,"")
                                for item in matters_with_keyword]#Rem. link mark
        matters_with_keyword = list(set(matters_with_keyword))
        #
        is_proandkey = False
        #

        ##Extract and merge components of nest
        components = []
        #Extract main component
        comp_main = {key:nest[key] for key in keys_main} #Main component
        main_matters = sorted([item for key in keys_matter
                                for item in nest[key]])
        #
        #Note if I-pronoun and keyword already paired
        if (bool_keyword in main_matters) and (bool_pronounI in main_matters):
            is_proandkey = True
        #

        #Merge in any linked components
        if any([(nest[key] not in [[], None]) for key in keys_linked_matter]):
            #Throw error if unequal number of components by matter
            if len(set([len(nest[key]) for key in keys_linked_matter])) != 1:
                raise ValueError("Err: Unequal num. of matter components?\n{0}"
                            .format([nest[key] for key in keys_linked_matter]))
            #

            #Extract and merge in each linked component
            for ii in range(0, num_links):
                #Extract current linked component
                curr_matters = sorted([item for key in keys_linked_matter
                                        for item in nest[key][ii]])
                #Print some notes
                if do_verbose:
                    print("Current main matters: {0}".format(main_matters))
                    print("Considering for linkage: {0}".format(curr_matters))
                #

                #Skip this component if no interesting terms
                if (len(curr_matters) == 0):
                    #Print some notes
                    if do_verbose:
                        print("No interesting terms, so skipping.")
                    #
                    continue
                #

                #Copy over keyword if only keyword present
                if (bool_keyword in curr_matters):
                    #Tack on keyword, if not done so already
                    if (bool_keyword not in main_matters):
                        comp_main[key_matter_obj].append(bool_keyword)
                        main_matters.append(bool_keyword) #Mark as included
                #

                #Copy over any precedent terms
                for term in curr_matters:
                    if (term in terms_superior):
                        if (term not in main_matters):
                            comp_main[key_matter_obj].append(term)
                            main_matters.append(term)
                        #Override main terms with non-matter terms, if 'I'-term
                        if ((bool_pronounI in curr_matters)
                                        and not (is_proandkey)):
                            for key in keys_nonmatter:
                                comp_main[key] = nest[(prefix_link+key)][ii]
                #Copy over any precedent terms
                #for term in curr_matters:
                #    if (term not in main_matters) and (term in terms_superior):
                #        comp_main[key_matter_obj].append(term)
                #        main_matters.append(term)
                #        #Override main terms with non-matter terms, if 'I'-term
                #        if (bool_pronounI in curr_matters):
                #            for key in keys_nonmatter:
                #                comp_main[key] = nest[(prefix_link+key)][ii]

                #Print some notes
                if do_verbose:
                    print("Done linking current term.")
                    print("Latest main matters: {0}\n".format(main_matters))
                #
            #
        #
        #Store the merged component
        components.append(comp_main)
        #

        ##Return the unlinked components of the nest
        if do_verbose:
            print("\nNest has been unlinked!\nComponents: {0}"
                    .format(components))
            print("Run of _unlink_nest complete!\n")
        #
        return components
    #
#


##Class: Operator
class Operator(_Base):
    """
    Class: Operator
    Purpose:
        - Run full workflow of text classification, from input text to internal text processing to output classification.
    Initialization Arguments:
        - classifier [Classifier_* instance]:
          - Classifier to use for classification.
        - keyword_objs [list of Keyword instances]:
          - Target missions; terms will be used to search the text.
        - mode [str]:
          - Mode of modification for generating modifs using the Grammar class.
        - load_check_truematch [bool (default=True)]:
          - Whether nor not to load external data for verifying ambiguous terms as true vs false matches to mission.
        - do_verbose [bool (default=False)]:
          - Whether or not to print surface-level log information and tests.
    """
    ##Method: __init__
    ##Purpose: Initialize this class instance
    def __init__(self, classifier, mode, keyword_objs, do_verbose, load_check_truematch=True, do_verbose_deep=False):
        """
        Method: __init__
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Initializes instance of Operator class.
        """
        #Initialize storage
        self._storage = {}
        self._store_info(classifier, "classifier")
        self._store_info(mode, "mode")
        self._store_info(keyword_objs, "keyword_objs")
        self._store_info(do_verbose, "do_verbose")
        self._store_info(load_check_truematch, "load_check_truematch")
        self._store_info(do_verbose_deep, "do_verbose_deep")

        #Load and process ambiguous (ambig.) data, if so requested
        if load_check_truematch:
            #Run method to load and process external ambig. database
            dict_ambigs =self._process_database_ambig(keyword_objs=keyword_objs)
            lookup_ambigs = dict_ambigs["lookup_ambigs"]
            #
            #Print some notes
            if do_verbose_deep:
                print("Loaded+Assembled data for ambiguous phrases.")
            #
        #
        #Otherwise, set empty placeholders
        else:
            dict_ambigs = None
            lookup_ambigs = None
        #
        #Store the processed data in this object instance
        self._store_info(dict_ambigs, "dict_ambigs")
        self._store_info(lookup_ambigs, "lookup_ambigs")
        #

        #Exit the method
        if do_verbose:
            print("Instance of Operator successfully initialized!")
            print("Keyword objects:")
            for ii in range(0, len(keyword_objs)):
                print("{0}: {1}".format(ii, keyword_objs[ii]))
        #
        return
    #

    ##Method: _fetch_keyword_object
    ##Purpose: Fetch a keyword object that matches the given lookup
    def _fetch_keyword_object(self, lookup, do_verbose=None, do_raise_emptyerror=True):
        """
        Method: _fetch_keyword_object
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Finds stored Keyword instance that matches to given lookup term.
        """
        #Load Global variables
        if (do_verbose is None):
            do_verbose = self._get_info("do_verbose")
        keyword_objs = self._get_info("keyword_objs")
        num_keyobjs = len(keyword_objs)
        #Print some notes
        if do_verbose:
            print("> Running _fetch_keyword_object() for lookup term {0}."
                    .format(lookup))
        #

        #Find keyword object that matches to given lookup term
        match = None
        for ii in range(0, num_keyobjs):
            #If current keyword object matches, record and stop loop
            if (keyword_objs[ii].is_keyword(lookup)):
                match = keyword_objs[ii]
                break
            #
        #

        #Throw error if no matching keyword object found
        if (match is None):
            errstr = "No matching keyword object for {0}.\n".format(lookup)
            errstr += "Available keyword objects are:\n"
            for ii in range(0, num_keyobjs):
                errstr += "{0}\n".format(keyword_objs[ii])
            #
            #Raise error if so requested
            if do_raise_emptyerror:
                raise ValueError(errstr)
            #Otherwise, return None
            else:
                return None
        #

        #Return the matching keyword object
        return match
    #

    ##Method: classify
    ##Purpose: Inspect text and either reject as false target or give class
    def classify(self, text, lookup, threshold, do_check_truematch, do_raise_innererror, buffer=0, do_verbose=None, do_verbose_deep=None):
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
        #Fetch global variables
        if (do_verbose is None):
            do_verbose = self._get_info("do_verbose")
        if (do_verbose_deep is None):
            do_verbose_deep = self._get_info("do_verbose_deep")
        #
        classifier = self._get_info("classifier")
        #

        #Fetch keyword object matching to the given keyword
        keyobj = self._fetch_keyword_object(lookup=lookup)
        if do_verbose:
            print("Best matching keyword object (keyobj) for keyword {0}:\n{1}"
                    .format(lookup, keyobj))
        #

        #Process text into modifs using Grammar class
        if do_verbose:
            print("\nPreprocessing and extracting modifs from the text...")
        output = self.process(text=text, do_check_truematch=do_check_truematch,
                                buffer=buffer, lookup=lookup,keyword_obj=keyobj,
                                do_verbose=do_verbose,
                                do_verbose_deep=do_verbose_deep)
        modif = output["modif"]
        forest = output["forest"]
        #
        #Print some notes
        if do_verbose:
            print("Text has been processed into modif.")
        #

        #Set rejected verdict if empty text
        if (modif.strip() == ""):
            if do_verbose:
                print("No text found matching keyword object.")
                print("Returning rejection verdict.")
            #
            dict_verdicts = preset.dictverdict_rejection.copy()
        #
        #Classify the text using stored classifier with raised error
        elif do_raise_innererror: #If True, allow raising of inner errors
            dict_verdicts = classifier.classify_text(text=modif,
                                            forest=forest,
                                            keyword_obj=keyobj,
                                            do_verbose=do_verbose, #_deep,
                                            threshold=threshold)
        #
        #Otherwise, run classification while ignoring inner errors
        else:
            #Try running classification
            try:
                dict_verdicts = classifier.classify_text(text=modif,
                                            forest=forest,
                                            keyword_obj=keyobj,
                                            do_verbose=do_verbose, #_deep,
                                            threshold=threshold)
            #
            #Catch certain exceptions and force-print some notes
            except (ValueError, KeyError, IndexError) as err:
                if True:
                    print("-")
                    print("The following err. was encountered in operate:")
                    print(err)
                    print("Error was noted. Continuing.")
                    print("-")
                #
                dict_verdicts = preset.dictverdict_error.copy()
        #

        #Return the verdict with modif included
        dict_verdicts["modif"] = modif
        return dict_verdicts
    #

    ##Method: process
    ##Purpose: Process text into modifs
    def process(self, text, do_check_truematch, buffer=0, lookup=None, keyword_obj=None, do_verbose=None, do_verbose_deep=None):
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
        #Fetch global variables
        if (do_verbose is None):
            do_verbose = self._get_info("do_verbose")
        if (do_verbose_deep is None):
            do_verbose_deep = self._get_info("do_verbose_deep")
        #
        mode = self._get_info("mode")
        dict_ambigs = self._get_info("dict_ambigs")
        #

        #Fetch keyword object matching to the given keyword
        if (keyword_obj is None):
            keyword_obj = self._fetch_keyword_object(lookup=lookup,
                                                    do_verbose=do_verbose_deep)
            if do_verbose:
                print("Best matching Keyword object for keyword {0}:\n{1}"
                        .format(lookup, keyword_obj))
        #

        #Process text into modifs using Grammar class
        if do_verbose:
            print("\nRunning Grammar on the text...")
        grammar = Grammar(text=text, keyword_obj=keyword_obj,
                            do_check_truematch=do_check_truematch,
                            dict_ambigs=dict_ambigs,
                            do_verbose=do_verbose_deep, buffer=buffer)
        grammar.run_modifications()
        output = grammar.get_modifs(do_include_forest=True)
        modif = output["modifs"][mode]
        forest = output["_forest"]
        #
        #Print some notes
        if do_verbose:
            print("Text has been processed into modifs.")
        #

        #Return the modif and internal processing output
        return {"modif":modif, "forest":forest}
    #

    ##Method: train_model_ML
    ##Purpose: Process text into modifs and then train ML model on the modifs
    def train_model_ML(self, dir_model, name_model, do_reuse_run, seed_TVT=10, seed_ML=8, filename_json=None, dict_texts=None, buffer=0, fraction_TVT=[0.8, 0.1, 0.1], mode_TVT="uniform", do_shuffle=True, print_freq=25, do_verbose=None, do_verbose_deep=None):
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
        #Fetch global variables
        if (do_verbose is None):
            do_verbose = self._get_info("do_verbose")
        if (do_verbose_deep is None):
            do_verbose_deep = self._get_info("do_verbose_deep")
        #
        classifier = self._get_info("classifier")
        folders_TVT = preset.folders_TVT
        savename_ML = (preset.tfoutput_prefix + name_model)
        savename_model = (name_model + ".npy")
        #
        #Print some notes
        if do_verbose:
            print("\n> Running train_model_ML()!")
        #

        #Throw error if invalid classifier given
        allowed_types = ["Classifier_ML"]
        if (type(classifier).__name__ not in allowed_types):
            raise ValueError("Err: Classifier ({0}) not allowed type ({1})"
                            .format(type(classifier), allowed_types))
        #

        #Preprocess texts into modifs and store in TVT directories
        #NOTE: TVT = training, validation, testing datasets
        is_exist = (os.path.exists(os.path.join(dir_model,folders_TVT["train"]))
            or os.path.exists(os.path.join(dir_model,folders_TVT["validate"])))
        #If TVT directories already exist, either print note or raise error
        if is_exist:
            #Print some notes
            if do_verbose:
                print("Previous training/validation directories already exist.")
            #
            #Skip ahead if previous data should be reused
            if do_reuse_run:
                print("Reusing the existing training/validation data in {0}."
                        .format(dir_model))
                pass
            #
            #Otherwise, raise error if not to reuse previous run data
            else:
                raise ValueError(("Err: Training/validation data already exists"
                                +" in {0}. Either delete it, or rerun method"
                                +" with do_reuse_run=True.").format(dir_model))
            #
        #
        #Otherwise, preprocess the text and store in TVT directories
        else:
            #Print some notes
            if do_verbose:
                print("Loading data from given .json file or dict. of texts...")
            #
            #Load in data based on file format
            #For data from .json file
            if (filename_json is not None):
                with openfile(filename_json, 'r') as openfile:
                    dataset = json.load(openfile)
            #
            #For data from dictionary
            elif (dict_texts is not None):
                dataset = dict_texts
            #
            #Otherwise, throw error
            else:
                raise ValueError("Err: Please pass in either a .json file to"
                                +" filename_json or a dictionary of"
                                +" pre-classified texts to dict_texts.")
            #
            #Print some notes
            if do_verbose:
                print("Text data has been loaded.")
                print("Processing text data into modifs...")
            #

            #Process each text within the database into a modif
            dict_modifs = {} #Container for modifs and text classification info
            i_track = 0
            num_data = len(dataset)
            for curr_key in dataset:
                old_dict = dataset[curr_key]
                #Extract modif for current text
                curr_modif = self.process(text=old_dict["text"],
                                    do_check_truematch=False, buffer=buffer,
                                    lookup=old_dict["mission"],
                                    keyword_obj=None,do_verbose=do_verbose_deep,
                                    do_verbose_deep=do_verbose_deep
                                    )["modif"]
                #
                #Store the modif and previous classification information
                new_dict = {"text":curr_modif, "class":old_dict["class"],
                            "id":old_dict["id"], "mission":old_dict["mission"]}
                dict_modifs[curr_key] = new_dict
                #
                #Increment count of modifs generated
                i_track += 1
                #
                #Print some notes at desired frequency
                if (do_verbose and ((i_track % print_freq) == 0)):
                    print("{0} of {1} total texts have been processed..."
                            .format(i_track, num_data))
                #
            #
            #Print some notes
            if do_verbose:
                print("Text data has been processed into modifs.")
                print("Storing the data in train+validate+test directories...")
            #

            #Store the modifs in new TVT directories
            classifier.generate_directory_TVT(dir_model=dir_model,
                            fraction_TVT=fraction_TVT, mode_TVT=mode_TVT,
                            filename_json=None, dict_texts=dict_modifs,
                            do_shuffle=do_shuffle, seed=seed_TVT,
                            do_verbose=do_verbose)
            #Print some notes
            if do_verbose:
                print("Train+validate+test directories created in {0}."
                        .format(dir_model))
            #
        #

        #Train a new machine learning (ML) model
        is_exist = (os.path.exists(os.path.join(dir_model, savename_model))
                    or os.path.exists(os.path.join(dir_model, savename_ML)))
        #If ML model or output already exists, either print note or raise error
        if is_exist:
            #Print some notes
            if do_verbose:
                print("ML model already exists for {0} in {1}."
                        .format(name_model, dir_model))
            #
            #Skip ahead if previous data should be reused
            if do_reuse_run:
                print("Reusing the existing ML model in {0}."
                        .format(dir_model))
                pass
            #
            #Otherwise, raise error if not to reuse previous run data
            else:
                raise ValueError(("Err: ML model/output already exists"
                                +" in {0}. Either delete it, or rerun method"
                                +" with do_reuse_run=True.").format(dir_model))
            #
        #
        #Otherwise, train new ML model on the TVT directories
        else:
            #Print some notes
            if do_verbose:
                print("Training new ML model on training data in {0}..."
                        .format(dir_model))
            #
            #Train new ML model
            model = classifier.train_ML(dir_model=dir_model,
                                    name_model=name_model, seed=seed_ML,
                                    do_verbose=do_verbose, do_return_model=True)
            #
            #Print some notes
            if do_verbose:
                print("New ML model trained and stored in {0}."
                        .format(dir_model))
            #
        #

        #Exit the method
        #Print some notes
        if do_verbose:
            print("Run of train_model_ML() complete!\n")
        #
        return
    #
#










#
