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
from mpl_toolkits.axes_grid1 import make_axes_locatable
#
#Internal presets/constants
import bibcat_config as config
#
import spacy
nlp = spacy.load(config.spacy_language_model)
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
                            if any([item.identify_keyword(curr_sent[ind].text
                                                            )["bool"]
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
                                        version_NLP=curr_sent[ii]
                                        )["bools"]["is_any"]
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
                                        version_NLP=curr_sent[ii]
                                        )["bools"]["is_any"]
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
                    print("All wordchunks so far: {0}\nNewest wordchunk: {1}"
                            .format(list_wordchunks, list_wordchunks[-1]))
                    print("pos_ values: {0}\ndep_ values: {1}\ntag_ values: {2}"
                            .format([item.pos_ for item in list_wordchunks[-1]],
                                [item.dep_ for item in list_wordchunks[-1]],
                                [item.tag_ for item in list_wordchunks[-1]]))
                #
            #
        #

        #Return the assembled wordchunks
        if do_verbose:
            print("Assembled keyword wordchunks:\n{0}".format(list_wordchunks))
        #
        return list_wordchunks
    #

    ##Method: _ax_confusion_matrix()
    ##Purpose: Plot rectangular confusion matrix for given data and labels
    def _ax_confusion_matrix(self, matr, ax, x_labels, y_labels, x_title, y_title, cbar_title, ax_title, is_norm, minmax_inds=None, cmap=plt.cm.BuPu, fontsize=16, ticksize=16, valsize=14, y_rotation=30, x_rotation=30):
        """
        Method: _ax_confusion_matrix
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Plots confusion matrix within given axis.
        """
        #Set global variables
        if is_norm:
            vmin = 0
            vmax = 1
        elif (minmax_inds is not None):
            vmin = 0 #None
            #Ignore non-target verdicts to avoid color spikes scaling if present
            tmpmatr = matr.copy()
            #if (config.verdict_rejection.lower() in x_labels):
            #    tmpmatr[:,x_labels.index(config.verdict_rejection.upper())] = -1
            #if (config.verdict_rejection.lower() in y_labels):
            #    tmpmatr[y_labels.index(config.verdict_rejection.upper()),:] = -1
            #Remove max scaling for non-target classifs along y-axis
            for yind in minmax_inds["y"]:
                #Remove non-target classifications from max consideration
                tmpmatr[yind,:] = -1
            #Remove max scaling for non-target classifs along x-axis
            for xind in minmax_inds["x"]:
                #Remove non-target classifications from max consideration
                tmpmatr[:,xind] = -1
            #
            vmax = tmpmatr.max() #None
        #
        else:
            vmin = 0
            vmax = matr.max()
        #

        #Plot the confusion matrix and colorbar
        image = ax.imshow(matr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)

        #Fetch the matrix dimensions
        ydim = matr.shape[0]
        xdim = matr.shape[1]

        #Write in the values
        for yy in range(0, ydim):
            for xx in range(0, xdim):
                #Set current text color based on background grayscale value
                if is_norm:
                    curr_gray = np.mean(cmap(matr[yy,xx])[0:3])
                else:
                    curr_gray = np.mean(cmap(matr[yy,xx] / vmax)[0:3])
                #
                if curr_gray <= 0.6:
                    curr_color = "white"
                else:
                    curr_color = "black"
                #

                #Write current text
                if is_norm:
                    plt.text(xx, yy, "{0:.3f}".format(matr[yy,xx]),
                            color=curr_color, horizontalalignment="center",
                            verticalalignment="center", fontsize=valsize)
                else:
                    plt.text(xx, yy, "{0:.0f}".format(matr[yy,xx]),
                            color=curr_color, horizontalalignment="center",
                            verticalalignment="center", fontsize=valsize)
            #
        #

        #Generate the colorbar
        cbarax = make_axes_locatable(ax).append_axes("right",size="5%",pad=0.05)
        cbar = plt.colorbar(image, cax=cbarax, extend="max")
        cbar.ax.tick_params(labelsize=valsize)

        #Set the tick and axis labels
        ax.tick_params(axis="both", which="both", labelsize=ticksize)
        ax.set_xticks(np.arange(0, xdim, 1))
        ax.set_xticklabels([item.title() for item in x_labels],
                            rotation=x_rotation)
        ax.set_yticks(np.arange(0, ydim, 1))
        ax.set_yticklabels([item.title() for item in y_labels],
                            rotation=y_rotation)
        ax.set_xlabel(x_title, fontsize=fontsize)
        ax.set_ylabel(y_title, fontsize=fontsize)

        #Set the subplot title
        ax.set_title("{0}\n{1}".format(ax_title, cbar_title), fontsize=fontsize)

        #Exit the method
        return
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
        tmp_res = self._search_text(text=text, keyword_objs=keyword_objs)
        dict_results["is_keyword"] = tmp_res["bool"]
        charspans_keyword = tmp_res["charspans"]
        #

        #Check for first-person pronouns, if requested
        if include_Ipronouns:
            list_pos_pronoun = config.pos_pronoun
            nlp_lookup_person = config.nlp_lookup_person
            check_pronounI = any([((item.pos_ in list_pos_pronoun) #Pronoun
                                and ("1" in item.morph.get(nlp_lookup_person)))
                                for item in version_NLP]) #Check if 1st-person
            dict_results["is_pron_1st"] = check_pronounI
        else: #Otherwise, remove pronoun contribution
            dict_results["is_pron_1st"] = False
        #

        #Check for special terms, if requested
        if include_terms:
            list_pos_pronoun = config.pos_pronoun
            nlp_lookup_person = config.nlp_lookup_person
            special_synsets_fig = config.special_synsets_fig
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
            exp = config.exp_etal_cleansed #Reg.ex. to find cleansed et al
            check_etal = bool(re.search(exp, text, flags=re.IGNORECASE))
            dict_results["is_etal"] = check_etal
        else: #Otherwise, remove term contribution
            dict_results["is_etal"] = False
        #

        #Store overall status of if any booleans set to True
        dict_results["is_any"] =any([dict_results[key] for key in dict_results])
        #

        #Return the booleans
        return {"bools":dict_results, "charspans_keyword":charspans_keyword}
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
        placeholder_number = config.placeholder_number
        text = re.sub(r"\(?\b[0-9]+\b\)?", placeholder_number, text_orig)

        #Print some notes
        if do_verbose:
            print(("\n> Running _check_truematch for text: '{0}'"
                    +"\nOriginal text: {1}\nLookups: {2}")
                    .format(text, text_orig, lookup_ambigs))
        #

        #Extract keyword objects that are potentially ambiguous
        keyword_objs_ambigs = [item1 for item1 in keyword_objs
                                if any([item1.identify_keyword(item2)["bool"]
                                        for item2 in lookup_ambigs])]
        #

        #Extract keyword identification information for each kobj
        dict_kobjinfo = {item._get_info("name"):item.identify_keyword(text)
                        for item in keyword_objs}

        #Return status as true match if non-ambig keywords match to text
        #if any([item1.identify_keyword(text)["bool"] for item1 in keyword_objs
        #        if (item1 not in keyword_objs_ambigs)]):
        if any([dict_kobjinfo[item1._get_info("name")]["bool"]
                    for item1 in keyword_objs
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
        #elif (not any([item.identify_keyword(text)["bool"]
        #                    for item in keyword_objs_ambigs])):
        elif (not any([dict_kobjinfo[item._get_info("name")]["bool"]
                            for item in keyword_objs_ambigs])):
            #Print some notes
            if do_verbose:
                print("Text matches no keywords at all. Returning false state.")
            #
            #Return status as false match
            return {"bool":False, "info":[{"matcher":None, "text_database":None,
                            "bool":False,
                            "text_wordchunk":"<No matching keywords at all.>"}]}
        #
        #First two checks confirm that ambig-keyword-obj. matches to text
        #
        #Return status as true match if no ambig. phrases match to text
        #elif (not any([bool(re.search((r"\b"+item2+r"\b"), text,
        #                                flags=re.IGNORECASE))
        #            for item1 in keyword_objs_ambigs
        #            for item2 in item1._get_info("ambig_words")])):
        #Return status as true match if any acronyms match
        elif (any([dict_kobjinfo[item._get_info("name")]["bool_acronym_only"]
                            for item in keyword_objs_ambigs])):
            #Print some notes
            if do_verbose:
                print("Text matches acronym. Returning true state.")
            #
            #Return status as true match
            return {"bool":True, "info":[{"matcher":None, "set":None,
                        "bool":True,
                        "text_wordchunk":"<Not ambig.>", "text_database":None}]}
        #
        #Return status as true match if any non-ambig. phrases match to text
        elif (any([bool(re.search((r"\b"+item2+r"\b"), text,
                                        flags=re.IGNORECASE))
                    for item1 in keyword_objs_ambigs
                    for item2 in item1._get_info("keywords")
                    if (item2 not in item1._get_info("ambig_words"))])):
            #Print some notes
            if do_verbose:
                print("Text matches unambiguous keyword. Returning true state.")
            #
            #Return status as true match
            return {"bool":True, "info":[{"matcher":None, "set":None,
                        "bool":True,
                        "text_wordchunk":"<Not ambig.>", "text_database":None}]}
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
                                +item1._get_info("acronyms_casesensitive")
                                +item1._get_info("acronyms_caseinsensitive"))
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
                #Raise a unique for-user error (using NotImplementedError)
                #Allows this exception to be uniquely caught elsewhere in code
                #Use-case isn't exactly what NotImplemented means, but that's ok
                #RuntimeError could also work but seems more for general use
                raise NotImplementedError(
                                    ("Err: Unrecognized ambig. phrase:\n{0}"
                                    +"\nTaken from this text snippet:\n{1}")
                                .format(curr_chunk, text))
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
    def _cleanse_text_old_2024_03_25_beforeetalupdates(self, text, do_streamline_etal):
        """
        Method: _cleanse_text
        WARNING! This method is *not* meant to be used directly by users.
        Purpose:
         - Cleanse text of extra whitespace, punctuation, etc.
         - Replace paper citations (e.g. 'et al.') with uniform placeholder.
        """
        #Extract global punctuation expressions
        set_apostrophe = config.set_apostrophe
        set_punctuation = config.set_punctuation
        exp_punctuation = config.exp_punctuation
        set_openbrackets = config.set_openbrackets
        set_closebrackets = config.set_closebrackets

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
            text = re.sub(exp_cites_nobrackets, config.placeholder_author, text)
            #
            #Replace singular et al. (e.g. SingleAuthor et al.) wordage as well
            #text = re.sub(r" et al\b\.?", "etal", text)
            text = re.sub(r"([A-Z][A-Z|a-z]+) et al\b\.?",
                            config.placeholder_author, text)
        #

        #Remove starting+ending whitespace
        text = text.lstrip().rstrip()

        #Return cleansed text
        return text
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
        set_apostrophe = config.set_apostrophe
        set_punctuation = config.set_punctuation
        exp_punctuation = config.exp_punctuation
        set_openbrackets = config.set_openbrackets
        set_closebrackets = config.set_closebrackets

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

        #Remove 'et al.' period phrasing (can mess up sentence splitter later)
        text = re.sub(r"\bet al\b\.", "et al", text)
        #

        #Replace pesky "Author & Author (date)", et al., etc., wordage
        if do_streamline_etal:
            #Adapted from:
            # https://regex101.com/r/xssPEs/1
            # https://stackoverflow.com/questions/63632861/
            #                           python-regex-to-get-citations-in-a-paper
            bit_author = r"(?:(\b[A-Z]\. )*[A-Z][A-Za-z'`-]+)"
            #x bit_author = r"(?:[A-Z][A-Za-z'`-]+)"
            bit_etal = r"(?:et al\.?)"
            #x bit_additional = f"(?:,? (?:(?:and |& )?{bit_author}|{bit_etal}))"
            #x bit_additional = f"(?: (?:(?:and |& )?{bit_author}|{bit_etal}))"
            bit_additional = f"(?: (?:(?:and |& ){bit_author}|{bit_etal}))"
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
            text = re.sub(exp_cites_nobrackets, config.placeholder_author, text)
            #
            #Replace singular et al. (e.g. SingleAuthor et al.) wordage as well
            #text = re.sub(r" et al\b\.?", "etal", text)
            text = re.sub(r"\b([A-Z]\. )*(\b[A-Z][A-Z|a-z]+) et al\b\.?",
                            config.placeholder_author, text)
            #

            #Collapse adjacent author terms
            text = re.sub(r"{0}((,|;|(,? and))( )+{0})+"
                                            .format(config.placeholder_author),
                            config.placeholder_author, text)
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
                            if (item.identify_keyword(curr_word.text)["bool"])]
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
                tmp_rep = config.string_numeral_ambig
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
        exp_synset = config.exp_synset
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

    ##Method: _is_pos_word()
    ##Purpose: Return boolean for if given word (NLP type word) is of given part of speech
    def _is_pos_word(self, word, pos, keyword_objs=None, ids_nounchunks=None, do_verbose=False, do_exclude_nounverbs=False):
        """
        Method: _is_pos_word
        WARNING! This method is *not* meant to be used directly by users.
        Purpose:
         - Return if the given word (of NLP type) is of the given part-of-speech.
        """
        ##Load global variables
        word_i = (word.i - word.sent.start) #Index
        word_dep = word.dep_ #dep label
        word_pos = word.pos_ #p.o.s. label
        word_tag = word.tag_ #tag label
        word_text = word.text #Text version of word
        word_ancestors = list(word.ancestors)#All previous nodes leading to word
        #word_conjoins = list(word.conjuncts)#All previous nodes leading to word
        #
        #Print some notes
        if do_verbose:
            print("Running _is_pos_word for: {0}".format(word))
            print("dep_: {0}\npos_: {1}\ntag_: {2}"
                    .format(word_dep, word_pos, word_tag))
            print("Node head: {0}\nSentence: {1}"
                    .format(word.head, word.sent))
            print("Node lefts: {0}\nNode rights: {1}"
                    .format(list(word.lefts), list(word.rights)))
        #

        ##Measure conjoined status ahead of time
        is_conjoined = False
        if (pos not in ["CONJOINED", "VERB", "ROOT"]):
            is_conjoined = self._is_pos_word(word=word, pos="CONJOINED")
        #

        """BLOCKED: 2024-03-27: Bad attempt at handling conjoined words.
        ##Check if given word is of given part-of-speech
        #Handle conjoined cases
        if is_conjoined:
            #Check if verb first and if root is verb (standard tree structure)
            isverb = self._is_pos_word(word=word, pos="VERB",
                    keyword_objs=keyword_objs, ids_nounchunks=ids_nounchunks,
                    do_verbose=do_verbose,
                    do_exclude_nounverbs=do_exclude_nounverbs)
            ispreverb = self._is_pos_word(word=word_conjoins[0], pos="VERB",
                    keyword_objs=keyword_objs, ids_nounchunks=ids_nounchunks,
                    do_verbose=do_verbose,
                    do_exclude_nounverbs=do_exclude_nounverbs)
            #
            #Point pos check to nearby node, if bad pairing from bad tree
            if ((not isverb) and ispreverb):
                word_neighbors = list(word_conjoins[0].rights) #Right of root
                #Scroll through neighbors until another noun found
                #for curr_neighbor in word_neighbors:
                    #Check if current neighbor is a noun
                #    isprenoun = self._is_pos_word(word=curr_neighbor,pos="NOUN",
                #            keyword_objs=keyword_objs,
                #            ids_nounchunks=ids_nounchunks,do_verbose=do_verbose,
                #            do_exclude_nounverbs=do_exclude_nounverbs)
                    #If noun, set this as root of conjoined set
                #    if isprenoun:
                #        conj_rootword = curr_neighbor
                        #Break out of loop early
                #        break
                    #
                #
                conj_rootword = word_neighbors[0]
            #
            #Point pos check to previous node, otherwise
            else:
                conj_rootword = word_conjoins[0]
            #
            #Call pos check on noted root of conjoining
            return self._is_pos_word(word=conj_rootword, pos=pos,
                        keyword_objs=keyword_objs,
                        ids_nounchunks=ids_nounchunks, do_verbose=do_verbose,
                        do_exclude_nounverbs=do_exclude_nounverbs)
            #
        #"""

        ##Check if given word is of given part-of-speech
        #Handle sufficiently clear-cut conjoined cases
        if is_conjoined:
            #Check if verb first and if root is verb (standard tree structure)
            isverb = self._is_pos_word(word=word, pos="VERB",
                    keyword_objs=keyword_objs, ids_nounchunks=ids_nounchunks,
                    do_verbose=do_verbose,
                    do_exclude_nounverbs=do_exclude_nounverbs)
            ispreverb = self._is_pos_word(word=word_ancestors[0], pos="VERB",
                    keyword_objs=keyword_objs, ids_nounchunks=ids_nounchunks,
                    do_verbose=do_verbose,
                    do_exclude_nounverbs=do_exclude_nounverbs)
            #
            #Point pos check for root node, if clear-cut conjoined case
            if ((isverb and ispreverb) or ((not isverb) and (not ispreverb))):
                return self._is_pos_word(word=word_ancestors[0], pos=pos,
                        keyword_objs=keyword_objs,
                        ids_nounchunks=ids_nounchunks, do_verbose=do_verbose,
                        do_exclude_nounverbs=do_exclude_nounverbs)
            #
        #
        #Identify roots
        if pos in ["ROOT"]:
            check_all = (word_dep in config.dep_root)
        #
        #Identify verbs
        elif pos in ["VERB"]:
            check_verb = (word_pos in ["VERB"]) #!!! Set in config. !!!
            check_root = self._is_pos_word(word=word, pos="ROOT")
            check_pos = (word_pos in ["AUX"])
            check_tag = (word_tag in config.tag_verb_any)
            check_auxverb = (check_tag
                                and (word_dep not in ["aux", "auxpass"])
                                and (check_pos))
            check_rootaux = (check_root and check_pos and check_tag)
            if do_exclude_nounverbs:
                check_notnoun = (ids_nounchunks[word_i] is None)
            else:
                check_notnoun = True
            #
            check_all = ((check_verb or check_rootaux or check_auxverb)
                            and check_notnoun)
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
            check_tag = (word_tag in config.tag_useless)
            check_dep = (word_dep in config.dep_useless)
            check_pos = (word_pos in config.pos_useless)
            check_use = self._check_importance(word_text,
                                        version_NLP=word,
                                    keyword_objs=keyword_objs
                                    )["bools"]["is_any"] #Useful
            check_root = self._is_pos_word(word=word, pos="ROOT")
            check_neg = self._is_pos_word(word=word, pos="NEGATIVE")
            check_subj = self._is_pos_word(word=word, pos="SUBJECT")
            check_all = ((check_tag and check_dep and check_pos)
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
                #tmp_root = self._is_pos_word(word=word_ancestors[0], pos="ROOT")
                #if (tmp_verb or tmp_root):
                if tmp_verb:
                    is_leftofverb = (word in word_ancestors[0].lefts)
            #
            #Determine if conjoined to subject, if applicable
            #is_conjsubj = self._is_pos_conjoined(word, pos=pos)
            is_root = self._is_pos_word(word=word, pos="ROOT")
            check_dep = (word_dep in config.dep_subject)
            check_all = ((
                        (check_noun and check_dep and is_leftofverb)
                        #or (is_conjsubj)
                        or (check_noun and is_root)
                        or ((check_noun or check_adj) and is_leftofverb))
                    and (not check_obj))
        #
        #Identify prepositions
        elif pos in ["PREPOSITION"]:
            check_dep = (word_dep in config.dep_preposition)
            check_pos = (word_pos in config.pos_preposition)
            check_tag = (word_tag in config.tag_preposition)
            check_prepaux = ((word_dep in config.dep_aux)
                            and (word_pos in config.pos_aux)
                            and (check_tag)) #For e.g. mishandled 'to'
            check_all = ((check_dep and check_pos and check_tag)
                        or (check_prepaux))
        #
        #Identify base objects (so either direct or prep. objects)
        elif pos in ["BASE_OBJECT"]:
            check_dep = (word_dep in config.dep_object)
            check_noun = self._is_pos_word(word=word, pos="NOUN")
            check_all = (check_noun and check_dep)
        #
        #Identify direct objects
        elif pos in ["DIRECT_OBJECT"]:
            check_noun = self._is_pos_word(word=word, pos="NOUN")
            check_baseobj = self._is_pos_word(word=word, pos="BASE_OBJECT")
            #is_conjprepobj = self._is_pos_conjoined(word, pos=pos)
            #Check if this word follows preposition
            check_objprep = False
            check_subjprep = False
            for pre_node in word_ancestors:
                #If preceding preposition found first
                if self._is_pos_word(word=pre_node, pos="PREPOSITION"):
                    check_hasof = (pre_node.text.lower() == "of")
                    pre_pre_node = list(pre_node.ancestors)[0]
                    #Ensure prepositional object instead of prep. 'of' subject
                    check_aftersubj = self._is_pos_word(word=pre_pre_node,
                                            pos="SUBJECT",
                                            ids_nounchunks=ids_nounchunks)
                    check_subjprep = (check_hasof and check_aftersubj)
                    check_objprep = (not check_subjprep)
                    break
                #If preceding verb found first
                elif self._is_pos_word(word=pre_node, pos="VERB"):
                    check_objprep = False
                    break
            #
            #check_all = (is_conjprepobj or (check_baseobj and check_objprep))
            check_all = ((check_baseobj and (not check_objprep)
                            and (not check_subjprep) and check_noun))
        #
        #Identify prepositional objects
        elif pos in ["PREPOSITIONAL_OBJECT"]:
            check_noun = self._is_pos_word(word=word, pos="NOUN")
            check_baseobj = self._is_pos_word(word=word, pos="BASE_OBJECT")
            #is_conjprepobj = self._is_pos_conjoined(word, pos=pos)
            #Check if this word follows preposition
            check_objprep = False
            for pre_node in word_ancestors:
                #If preceding preposition found first
                if self._is_pos_word(word=pre_node, pos="PREPOSITION"):
                    pre_pre_node = list(pre_node.ancestors)[0]
                    #Ensure prepositional object instead of prep. subject
                    check_objprep = (not self._is_pos_word(word=pre_pre_node,
                                            pos="SUBJECT",
                                            ids_nounchunks=ids_nounchunks))
                    break
                #If preceding verb found first
                elif self._is_pos_word(word=pre_node, pos="VERB"):
                    check_objprep = False
                    break
            #
            #check_all = (is_conjprepobj or (check_baseobj and check_objprep))
            check_all = ((check_baseobj and check_objprep and check_noun))
        #
        #Identify improper X-words (for improper sentences)
        elif pos in ["X"]:
            check_dep = (word_dep in config.dep_xpos)
            check_pos = (word_pos in config.pos_xpos)
            check_all = (check_dep or check_pos)
        #
        #Identify conjoined words
        elif pos in ["CONJOINED"]:
            check_conj = (word_dep in config.dep_conjoined)
            #check_appos = (word_dep in config.dep_appos)
            check_det = (word_tag in config.tag_determinant)
            #check_all = ((check_conj or check_appos) and (not check_det))
            check_all = (check_conj and (not check_det))
        #
        #Identify  conjunctions
        elif pos in ["CONJUNCTION"]:
            check_pos = (word_pos in config.pos_conjunction)
            check_tag = (word_tag in config.tag_conjunction)
            check_all = (check_pos and check_tag)
        #
        #Identify determinants
        elif pos in ["DETERMINANT"]:
            check_pos = (word_pos in config.pos_determinant)
            check_tag = (word_tag in config.tag_determinant)
            check_all = (check_pos and check_tag)
        #
        #Identify aux
        elif pos in ["AUX"]:
            check_dep = (word_dep in config.dep_aux)
            check_pos = (word_pos in config.pos_aux)
            check_prep = (word_tag in config.tag_preposition)
            check_num = (word_tag in config.tag_number)
            #
            tags_approved = (config.tag_verb_past + config.tag_verb_present
                            + config.tag_verb_future + config.tag_verb_purpose)
            check_approved = (word_tag in tags_approved)
            #
            check_all = ((check_dep and check_pos and check_approved)
                    and (not (check_prep or check_num)))
        #
        #Identify nouns
        elif pos in ["NOUN"]:
            check_pos = (word_pos in config.pos_noun)
            check_det = (word_tag in config.tag_determinant)
            check_all = (check_pos and (not check_det))
        #
        #Identify adjectives
        elif pos in ["ADJECTIVE"]:
            check_adjverb = ((word_dep in config.dep_adjective)
                                and (word_pos in config.pos_verb)
                                and (word_tag in config.tag_verb_any))
            check_pos = (word_pos in config.pos_adjective)
            check_tag = (word_tag in config.tag_adjective)
            check_all = (check_tag or check_pos or check_adjverb)
        #
        #Identify passive verbs and aux
        elif pos in ["PASSIVE"]:
            check_dep = (word_dep in config.dep_verb_passive)
            check_all = check_dep
        #
        #Identify negative words
        elif pos in ["NEGATIVE"]:
            check_dep = (word_dep in config.dep_negative)
            check_all = check_dep
        #
        #Identify punctuation
        elif pos in ["PUNCTUATION"]:
            check_punct = (word_dep in config.dep_punctuation)
            check_letter = bool(re.search(".*[a-z|0-9].*", word_text,
                                            flags=re.IGNORECASE))
            check_all = (check_punct and (not check_letter))
        #
        #Identify punctuation
        elif pos in ["BRACKET"]:
            check_brackets = (word_tag in (config.tag_brackets))
            check_all = (check_brackets)
        #
        #Identify possessive markers
        elif pos in ["POSSESSIVE"]:
            check_possessive = (word_tag in config.tag_possessive)
            check_all = (check_possessive)
        #
        #Identify numbers
        elif pos in ["NUMBER"]:
            check_number = (word_pos in config.pos_number)
            check_all = (check_number)
        #
        #Otherwise, raise error if given pos is not recognized
        else:
            raise ValueError("Err: {0} is not a recognized part of speech."
                            .format(pos))
        #

        #Print some notes
        if do_verbose:
            print("Is pos={0}? {1}\n-".format(pos, check_all))
        #Return the final verdict
        return check_all
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

    ##Method: _process_database_ambig()
    ##Purpose: Process database of ambig. phrases into lookups and dictionary
    def _process_database_ambig(self, keyword_objs=None, do_verbose=False):
        """
        Method: _process_database_ambig
        WARNING! This method is *not* meant to be used directly by users.
        Purpose:
         - Process database of ambiguous keyword phrases into dictionary of keywords, regular expressions, boolean verdicts, etc.
        """
        #Load the keywords
        if (keyword_objs is None):
            try:
                keyword_objs = [self._get_info("keyword_obj",
                                                do_flag_hidden=True)]
            except KeyError:
                keyword_objs = self._get_info("keyword_objs",
                                                do_flag_hidden=True)
        #
        #Load the ambig. phrase data
        #lookup_ambigs = [str(item).lower() for item in
        #            np.genfromtxt(config.KW_AMBIG,
        #                        comments="#", dtype=str)]
        lookup_ambigs =[item._get_info("name").lower() for item in keyword_objs
                        if (len(item._get_info("ambig_words")) > 0)]
        data_ambigs = np.genfromtxt(config.PHR_AMBIG,
                                comments="#", dtype=str, delimiter="\t"
                                )
        if (len(data_ambigs.shape) == 1): #If single row, reshape to 2D
            data_ambigs = data_ambigs.reshape(1, data_ambigs.shape[0])
        num_ambigs = data_ambigs.shape[0]
        #
        str_anymatch_ambig = config.string_anymatch_ambig.lower()
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
            #This regex version can be very slow... replaced with version below
            #curr_exp = (r"("
            #            + r")( .*)* (".join([(r"(\b"+r"\b|\b".join(item)+r"\b)")
            #                            for item in curr_roots])
            #            + r")") #Convert to reg. exp. for substring search later
            curr_exp = (r"("
                        + r") (\w+ )*(".join([(r"\b("+r"|".join(item)+r")\b")
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
        tmp_res = [item.identify_keyword(text) for item in keyword_objs]
        check_keywords = any([item["bool"] for item in tmp_res])
        charspans_keywords = []
        for ii in range(0, len(tmp_res)):
            charspans_keywords += tmp_res[ii]["charspans"]
        #
        #Print some notes
        if do_verbose:
            #Extract global variables
            keywords = [item2
                        for item1 in keyword_objs
                        for item2 in item1._get_info("keywords")]
            acronyms = [item2
                        for item1 in keyword_objs
                        for item2 in (item1._get_info("acronyms_casesensitive")
                                +item1._get_info("acronyms_caseinsensitive"))]
            #
            print("Completed _search_text().")
            print("Keywords={0}\nAcronyms={1}".format(keywords, acronyms))
            print("Boolean: {0}".format(check_keywords))
            print("Char. Spans: {0}".format(charspans_keywords))
        #

        #Return boolean result
        return {"bool":check_keywords, "charspans":charspans_keywords}
    #

    ##Method: _streamline_phrase()
    ##Purpose: Cleanse given (short) string of extra whitespace, dashes, etc, and replace websites, etc, with uniform placeholders.
    def _streamline_phrase(self, text, do_streamline_etal):
        """
        Method: _streamline_phrase
        WARNING! This method is *not* meant to be used directly by users.
        Purpose:
         - Run _cleanse_text with citation streamlining.
         - Replace websites with uniform placeholder.
         - Replace some common science abbreviations that confuse external NLP package sentence parsing.
        """
        #Extract global variables
        dict_exp_abbrev = config.dict_exp_abbrev

        #Remove any initial excessive whitespace
        text = self._cleanse_text(text=text,
                                do_streamline_etal=do_streamline_etal)

        """BLOCKED: 2024-03-25: Unnecessary initial text modification.
        #Replace annoying websites with placeholder
        text = re.sub(config.exp_website, config.placeholder_website, text)
        #"""

        #Replace annoying <> inserts (e.g. html)
        text = re.sub(r"<[A-Z|a-z|/]+>", "", text)

        #Replace annoying abbreviations that confuse NLP sentence parser
        for key1 in dict_exp_abbrev:
            text = re.sub(key1, dict_exp_abbrev[key1], text)
        #

        #Replace annoying object numerical name notations
        """BLOCKED: 2024-03-25: Unnecessary initial text modification.
        #E.g.: HD 123456, 2MASS123-456
        text = re.sub(r"([A-Z]+) ?[0-9][0-9]+[A-Z|a-z]*((\+|-)[0-9][0-9]+)*",
                        r"\g<1>"+config.placeholder_number, text)
        #E.g.: Kepler-123ab
        text = re.sub(r"([A-Z][a-z]+)( |-)?[0-9][0-9]+([A-Z|a-z])*",
                        r"\g<1> "+config.placeholder_number, text)

        #Remove most obnoxious numeric ranges
        text = re.sub(r"~?[0-9]+([0-9]|\.)* ?- ?[0-9]+([0-9]|\.)*[A-Z|a-z]*\b",
                        "{0}".format(config.placeholder_numeric), text)

        #Remove spaces between capital+numeric names
        text = re.sub(r"([A-Z]+) ([0-9]+)([0-9]|[a-z])+",
                        r"\1\2\3".format(config.placeholder_numeric), text)
        #
        #"""

        #Remove any new excessive whitespace and punctuation spaces
        text = self._cleanse_text(text=text,
                                    do_streamline_etal=do_streamline_etal)
        #

        #Return streamlined text
        return text
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


##Class: Keyword
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
    def __init__(self, keywords, acronyms_caseinsensitive, acronyms_casesensitive, banned_overlap, ambig_words, do_not_classify, do_verbose=False):
        """
        Method: __init__
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Initialize instance of Keyword class.
        """
        #Initialize storage
        self._storage = {}
        self._store_info(do_verbose, "do_verbose")
        self._store_info(banned_overlap, "banned_overlap")
        self._store_info(ambig_words, "ambig_words")
        self._store_info(do_not_classify, "do_not_classify")

        #Cleanse keywords of extra whitespace, punctuation, etc.
        keywords_clean = sorted([self._cleanse_text(text=phrase,
                                                    do_streamline_etal=True)
                            for phrase in keywords],
                            key=(lambda w:len(w)))[::-1] #Sort by desc. length
        #Store keywords
        self._store_info(keywords_clean, key="keywords")

        #Cleanse banned overlap of extra whitespace, punctuation, etc.
        banned_overlap_lowercase = [self._cleanse_text(text=phrase.lower(),
                                                    do_streamline_etal=True)
                                    for phrase in banned_overlap]
        #Store keywords
        self._store_info(banned_overlap_lowercase,
                        key="banned_overlap_lowercase")

        #Also cleanse+store acronyms, if given
        #For case-insensitive acronyms
        if (acronyms_caseinsensitive is not None):
            acronyms_mid = [re.sub(config.exp_nopunct, "", item,
                                flags=re.IGNORECASE)
                                for item in acronyms_caseinsensitive]
            #Remove all whitespace
            acronyms_mid = [re.sub(" ", "", item) for item in acronyms_mid]
            acronyms_clean = sorted(acronyms_mid,
                            key=(lambda w:len(w)))[::-1]#Sort by desc. length
            self._store_info(acronyms_clean, key="acronyms_caseinsensitive")
        else:
            self._store_info([], key="acronyms_caseinsensitive")
        #
        #For case-sensitive acronyms
        if (acronyms_casesensitive is not None):
            acronyms_mid = [re.sub(config.exp_nopunct, "", item,
                                flags=re.IGNORECASE)
                                for item in acronyms_casesensitive]
            #Remove all whitespace
            acronyms_mid = [re.sub(" ", "", item) for item in acronyms_mid]
            acronyms_clean = sorted(acronyms_mid,
                            key=(lambda w:len(w)))[::-1]#Sort by desc. length
            self._store_info(acronyms_clean, key="acronyms_casesensitive")
        else:
            self._store_info([], key="acronyms_casesensitive")
        #

        #Store representative name for this keyword object
        repr_name = (self._get_info("keywords")[::-1]
                                +self._get_info("acronyms_casesensitive")
                                +self._get_info("acronyms_caseinsensitive")
                    )[0] #Take shortest keyword or longest acronym as repr. name
        self._store_info(repr_name, key="name")
        #

        #Store regular expressions for keywords and acronyms
        exps_k = [(r"\b"+phrase+r"\b") for phrase in self._get_info("keywords")]
        #For case-insensitive acronyms
        if ((acronyms_caseinsensitive is not None)
                            and (len(acronyms_caseinsensitive) > 0)):
            #Update acronyms to allow optional spaces
            acronyms_upd = [(r"(\.?)( ?)".join(item))
                        for item in self._get_info("acronyms_caseinsensitive")]
            #Build regular expression to recognize acronyms
            exp_a_caseinsensitive = (
                        r"(^|[^\.])((\b"
                        + r"\b)|(\b".join([phrase for phrase in acronyms_upd])
                        + r"\b)(\.?))($|[^A-Z|a-z])") #Matches any acronym
        else:
            exp_a_caseinsensitive = None
        #
        #For case-sensitive acronyms
        if ((acronyms_casesensitive is not None)
                                        and (len(acronyms_casesensitive) > 0)):
            #Update acronyms to allow optional spaces
            acronyms_upd = [(r"(\.?)( ?)".join(item))
                        for item in self._get_info("acronyms_casesensitive")]
            #Build regular expression to recognize acronyms
            exp_a_casesensitive = (
                        r"(^|[^\.])((\b"
                        + r"\b)|(\b".join([phrase for phrase in acronyms_upd])
                        + r"\b)(\.?))($|[^A-Z|a-z])") #Matches any acronym
        else:
            exp_a_casesensitive = None
        #
        self._store_info(exps_k, key="exps_keywords")
        self._store_info(exp_a_caseinsensitive,
                        key="exp_acronyms_caseinsensitive")
        self._store_info(exp_a_casesensitive,
                        key="exp_acronyms_casesensitive")
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
                    + ("Acronyms (Case-Insensitive): {0}\n"
                            .format(self._get_info("acronyms_caseinsensitive")))
                    + ("Acronyms (Case-Sensitive): {0}\n"
                            .format(self._get_info("acronyms_casesensitive")))
                    + "Banned Overlap: {0}\n"
                                    .format(self._get_info("banned_overlap"))
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

    ##Method: identify_keyword
    ##Purpose: Check if text matches to this keyword object; return match inds
    def identify_keyword_old_2024_03_26_beforeupdatestobannedoverlaphandling(self, text, mode=None):
        """
        Method: identify_keyword
        Purpose: Return whether or not the given text contains terms (keywords, acronyms) matching this instance.
        Arguments:
          - "text" [str]: The text to search within for terms
        Returns:
          - [dict] containing:
            - "bool":[bool] - if any matches
            - "span":[tuple of 2 ints] - character span of matches
        """
        #Fetch global variables
        exps_k = self._get_info("exps_keywords")
        keywords = self._get_info("keywords")
        exp_a_yescase = self._get_info("exp_acronyms_casesensitive")
        exp_a_nocase = self._get_info("exp_acronyms_caseinsensitive")
        banned_overlap_lowercase = self._get_info("banned_overlap_lowercase")
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
            set_keywords = [list(re.finditer(item1, text, flags=re.IGNORECASE))
                            for item1 in exps_k
                            if (not any([(ban1 in text.lower())
                                    for ban1 in banned_overlap_lowercase]))]
            #set_keywords = [list(re.finditer(item1, text, flags=re.IGNORECASE))
            #                for item1 in exps_k
            #                if (not any([
            #                        (bool(re.finditer(item1, ban1,
            #                                            flags=re.IGNORECASE))
            #                            and (ban1 in text.lower()))
            #                        for ban1 in banned_overlap_lowercase]))]
            charspans_keywords = [(item2.start(), (item2.end()-1))
                                for item1 in set_keywords
                                for item2 in item1] #Char. span of matches
            check_keywords = any([(len(item) > 0)
                                for item in set_keywords]) #If any matches
        else:
            charspans_keywords = []
            check_keywords = False
        #
        #Check if this text contains case-sensitive acronyms
        if (((mode is None) or (mode.lower() == "acronym"))
                    and (exp_a_yescase is not None)):
            set_acronyms_yescase = list(re.finditer(exp_a_yescase, text))
            #set_acronyms_yescase_raw = [list(re.finditer(item1, text))
            #                for item1 in exp_a_yescase
            #                if (not any([
            #                    (bool(re.finditer(item1, ban1,
            #                                            flags=re.IGNORECASE))
            #                            and (ban1 in text.lower()))
            #                        for ban1 in banned_overlap_lowercase]))]
            #set_acronyms_yescase = [item2 for item1 in set_acronyms_yescase_raw
            #                        for item2 in item1]
            #
            charspans_acronyms_yescase = [(item.start(), (item.end()-1))
                                        for item in set_acronyms_yescase]
            check_acronyms_yescase = (len(set_acronyms_yescase) > 0)
        else:
            charspans_acronyms_yescase = []
            check_acronyms_yescase = False
        #
        #Check if this text contains case-insensitive acronyms
        if (((mode is None) or (mode.lower() == "acronym"))
                    and (exp_a_nocase is not None)):
            set_acronyms_nocase = list(re.finditer(exp_a_nocase, text,
                                                    flags=re.IGNORECASE))
            #set_acronyms_nocase_raw = [list(re.finditer(item1, text,
            #                                        flags=re.IGNORECASE))
            #                for item1 in exp_a_nocase
            #                if (not any([
            #                    (bool(re.finditer(item1, ban1,
            #                                            flags=re.IGNORECASE))
            #                            and (ban1 in text.lower()))
            #                        for ban1 in banned_overlap_lowercase]))]
            #set_acronyms_nocase = [item2 for item1 in set_acronyms_nocase_raw
            #                        for item2 in item1]
            #
            charspans_acronyms_nocase = [(item.start(), (item.end()-1))
                                        for item in set_acronyms_nocase]
            check_acronyms_nocase = (len(set_acronyms_nocase) > 0)
        else:
            charspans_acronyms_nocase = []
            check_acronyms_nocase = False
        #
        #Combine acronym results
        check_acronyms_all = (check_acronyms_nocase or check_acronyms_yescase)
        charspans_acronyms_all = (charspans_acronyms_nocase
                                    + charspans_acronyms_yescase)
        #

        #Print some notes
        if do_verbose:
            print("Keywords: {0}\nKeyword regex:\n{1}".format(keywords, exps_k))
            print("Acronyms (Case-Sensitive): {0}\nAcronym regex:\n{1}"
                    .format(acronyms_yescase, exp_a_yescase))
            print("Acronyms (Case-Insensitive): {0}\nAcronym regex:\n{1}"
                    .format(acronyms_nocase, exp_a_nocase))
            print("Keyword bool: {0}\nAcronym bool: {1}"
                    .format(check_keywords, check_acronyms_all))
            print("Keyword char. spans: {0}\nAcronym char. spans: {1}"
                    .format(charspans_keywords, charspans_acronyms_all))
        #

        #Return booleans
        return {"bool":(check_acronyms_all or check_keywords),
                "charspans":(charspans_keywords + charspans_acronyms_all),
                "bool_acronym_only":check_acronyms_all}
    #

    ##Method: identify_keyword
    ##Purpose: Check if text matches to this keyword object; return match inds
    def identify_keyword(self, text, mode=None):
        """
        Method: identify_keyword
        Purpose: Return whether or not the given text contains terms (keywords, acronyms) matching this instance.
        Arguments:
          - "text" [str]: The text to search within for terms
        Returns:
          - [dict] containing:
            - "bool":[bool] - if any matches
            - "span":[tuple of 2 ints] - character span of matches
        """
        #Fetch global variables
        exps_k = self._get_info("exps_keywords")
        keywords = self._get_info("keywords")
        exp_a_yescase = self._get_info("exp_acronyms_casesensitive")
        exp_a_nocase = self._get_info("exp_acronyms_caseinsensitive")
        banned_overlap_lowercase = self._get_info("banned_overlap_lowercase")
        do_verbose = self._get_info("do_verbose")
        allowed_modes = [None, "keyword", "acronym"]
        #

        #Throw error is specified mode not recognized
        if ((mode is not None) and (mode.lower() not in allowed_modes)):
            raise ValueError("Err: Invalid mode '{0}'.\nAllowed modes: {1}"
                            .format(mode, allowed_modes))
        #

        #Modify text (locally only) to block banned overlap
        text_mod = text
        for curr_ban in banned_overlap_lowercase:
            #Below replaces banned phrases with mask '#' string of custom length
            text_mod = re.sub(curr_ban, (lambda x: ("#" * len(x.group()))),
                                text_mod, flags=re.IGNORECASE)
        #

        #Check if this text contains keywords
        if ((mode is None) or (mode.lower() == "keyword")):
            set_keywords = [list(re.finditer(item1, text_mod,
                            flags=re.IGNORECASE)) for item1 in exps_k]
                            #if (not any([(ban1 in text.lower())
                            #        for ban1 in banned_overlap_lowercase]))]
            #set_keywords = [list(re.finditer(item1, text, flags=re.IGNORECASE))
            #                for item1 in exps_k
            #                if (not any([
            #                        (bool(re.finditer(item1, ban1,
            #                                            flags=re.IGNORECASE))
            #                            and (ban1 in text.lower()))
            #                        for ban1 in banned_overlap_lowercase]))]
            charspans_keywords = [(item2.start(), (item2.end()-1))
                                for item1 in set_keywords
                                for item2 in item1] #Char. span of matches
            check_keywords = any([(len(item) > 0)
                                for item in set_keywords]) #If any matches
        else:
            charspans_keywords = []
            check_keywords = False
        #
        #Check if this text contains case-sensitive acronyms
        if (((mode is None) or (mode.lower() == "acronym"))
                    and (exp_a_yescase is not None)):
            set_acronyms_yescase = list(re.finditer(exp_a_yescase, text_mod))
            #set_acronyms_yescase_raw = [list(re.finditer(item1, text))
            #                for item1 in exp_a_yescase
            #                if (not any([
            #                    (bool(re.finditer(item1, ban1,
            #                                            flags=re.IGNORECASE))
            #                            and (ban1 in text.lower()))
            #                        for ban1 in banned_overlap_lowercase]))]
            #set_acronyms_yescase = [item2 for item1 in set_acronyms_yescase_raw
            #                        for item2 in item1]
            #
            charspans_acronyms_yescase = [(item.start(), (item.end()-1))
                                        for item in set_acronyms_yescase]
            check_acronyms_yescase = (len(set_acronyms_yescase) > 0)
        else:
            charspans_acronyms_yescase = []
            check_acronyms_yescase = False
        #
        #Check if this text contains case-insensitive acronyms
        if (((mode is None) or (mode.lower() == "acronym"))
                    and (exp_a_nocase is not None)):
            set_acronyms_nocase = list(re.finditer(exp_a_nocase, text_mod,
                                                    flags=re.IGNORECASE))
            #set_acronyms_nocase_raw = [list(re.finditer(item1, text,
            #                                        flags=re.IGNORECASE))
            #                for item1 in exp_a_nocase
            #                if (not any([
            #                    (bool(re.finditer(item1, ban1,
            #                                            flags=re.IGNORECASE))
            #                            and (ban1 in text.lower()))
            #                        for ban1 in banned_overlap_lowercase]))]
            #set_acronyms_nocase = [item2 for item1 in set_acronyms_nocase_raw
            #                        for item2 in item1]
            #
            charspans_acronyms_nocase = [(item.start(), (item.end()-1))
                                        for item in set_acronyms_nocase]
            check_acronyms_nocase = (len(set_acronyms_nocase) > 0)
        else:
            charspans_acronyms_nocase = []
            check_acronyms_nocase = False
        #
        #Combine acronym results
        check_acronyms_all = (check_acronyms_nocase or check_acronyms_yescase)
        charspans_acronyms_all = (charspans_acronyms_nocase
                                    + charspans_acronyms_yescase)
        #

        #Print some notes
        if do_verbose:
            print("Keywords: {0}\nKeyword regex:\n{1}".format(keywords, exps_k))
            print("Acronyms (Case-Sensitive): {0}\nAcronym regex:\n{1}"
                    .format(acronyms_yescase, exp_a_yescase))
            print("Acronyms (Case-Insensitive): {0}\nAcronym regex:\n{1}"
                    .format(acronyms_nocase, exp_a_nocase))
            print("Keyword bool: {0}\nAcronym bool: {1}"
                    .format(check_keywords, check_acronyms_all))
            print("Keyword char. spans: {0}\nAcronym char. spans: {1}"
                    .format(charspans_keywords, charspans_acronyms_all))
        #

        #Return booleans
        return {"bool":(check_acronyms_all or check_keywords),
                "charspans":(charspans_keywords + charspans_acronyms_all),
                "bool_acronym_only":check_acronyms_all}
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
        exp_a_yescase = self._get_info("exp_acronyms_casesensitive")
        exp_a_nocase = self._get_info("exp_acronyms_caseinsensitive")
        #
        if (exp_a_yescase is not None):
            exps_a_yescase = [exp_a_yescase]
        else:
            exps_a_yescase = []
        #
        if (exp_a_nocase is not None):
            exps_a_nocase = [exp_a_nocase]
        else:
            exps_a_nocase = []
        #
        do_verbose = self._get_info("do_verbose")
        text_new = text
        #

        #Replace terms with given placeholder
        #For keywords
        for curr_exp in exps_k:
            text_new = re.sub(curr_exp, placeholder, text_new,
                            flags=re.IGNORECASE)
        #
        #For acronyms: avoids matches to acronyms in larger acronyms
        #For case-sensitive acronyms
        for curr_exp in exps_a_yescase:
            #Group processing below prevents substitution of spaces around acr.
            str_tot = str(re.compile(curr_exp).groups) #Get # of last group
            text_new = re.sub(curr_exp, (r"\1"+placeholder+("\\"+str_tot)),
                            text_new)
        #For case-insensitive acronyms
        for curr_exp in exps_a_nocase:
            #Group processing below prevents substitution of spaces around acr.
            str_tot = str(re.compile(curr_exp).groups) #Get # of last group
            text_new = re.sub(curr_exp, (r"\1"+placeholder+("\\"+str_tot)),
                            text_new, flags=re.IGNORECASE)
        #

        #Print some notes
        if do_verbose:
            print("Keyword regex:\n{0}".format(exps_k))
            print("Acronyms (Case-Sensitive) regex: {0}".format(exp_a_yescase))
            print("Acronyms (Case-Insensitive) regex: {0}".format(exp_a_nocase))
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
        text_clean = self._streamline_phrase(text=text_original,
                                        do_streamline_etal=False)
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
        #dict_acronym_meanings = {item.get_name():None
        #                        for item in keyword_objs} #False acr. meanings
        for ii in range(0, len(keyword_objs)):
            #Extract all paragraphs containing keywords/verified acronyms
            tmp_res = self._extract_paragraph(keyword_obj=keyword_objs[ii],
                                                buffer=buffer)
            paragraphs = tmp_res["paragraph"]
            dict_results_ambig[keyword_objs[ii].get_name()
                                ] = tmp_res["ambig_matches"]
            #dict_acronym_meanings[keyword_objs[ii].get_name()
            #                    ] = tmp_res["acronym_meanings"]
            #

            #Store the paragraphs under name of first given keyword
            dict_paragraphs[keyword_objs[ii].get_name()] = paragraphs
        #

        #Store the extracted paragraphs and setup information
        self._store_info(dict_paragraphs, "_paragraphs")
        self._store_info(dict_setup, "_paragraphs_setup")
        self._store_info(dict_results_ambig, "_results_ambig")
        #self._store_info(dict_acronym_meanings, "_dict_acronym_meanings")

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
        do_not_classify = keyword_obj._get_info("do_not_classify")
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
                            if keyword_obj.identify_keyword(sentences[ii],
                                                        mode="keyword")["bool"]]
        #For acronym terms
        inds_with_acronyms = [ii for ii in range(0, num_sentences)
                            if keyword_obj.identify_keyword(sentences[ii],
                                                        mode="acronym")["bool"]]
        #
        #Print some notes
        if do_verbose:
            print(("Found:\n# of keyword sentences: {0}\n"
                +"# of acronym sentences: {1}...")
                .format(len(inds_with_keywords_init), len(inds_with_acronyms)))
        #

        #If only acronym terms found, run a check of possible false meanings
        #if (len(inds_with_acronyms) > 0):
        #    acronym_meanings = self._verify_acronyms(keyword_obj=keyword_obj)
        #Otherwise, set empty
        #else:
        #    acronym_meanings = None
        #

        #If requested, run a check for ambiguous phrases if any ambig. keywords
        if ((not do_not_classify) and do_check_truematch and any([
                                    keyword_obj.identify_keyword(item)["bool"]
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
                #"acronym_meanings":acronym_meanings,
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
                            re.split(config.exp_splitbracketstarts, phrase)]
        #Split by sentences ending with brackets
        text_flat = [item for phrase in text_flat
                            for item in
                            re.split(config.exp_splitbracketends, phrase)]
        #Then split by assumed sentence structure
        text_flat = [item for phrase in text_flat
                            for item in
                            re.split(config.exp_splittext, phrase)]
        #Return the split text
        return text_flat
    #

    #BLOCKED: 2024-03-26: Passive feature of code; output not actually used.
    ##Method: _verify_acronyms()
    ##Purpose: Find possible meanings of acronyms in text
    def _verify_acronyms_old_2024_03_26_passivefeatureofcodewhereoutputnotactuallyused(self, keyword_obj):
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
                    +(r"[a-z]+\b"+config.exp_acronym_midwords).join(letterset)
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
        if ((do_check_truematch) and (dict_ambigs is None)):
            #Print some notes
            if do_verbose:
                print("Processing database of ambiguous phrases...")
            #
            dict_ambigs = self._process_database_ambig()
        #Otherwise, do nothing new
        else:
            #Print some notes
            if do_verbose:
                print("No ambiguous phrase processing requested.")
            #
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
            #which_modes = [key for key in forest]
            which_modes = [key for key in dict_modifs_orig]
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
    ##Purpose: Run submethods to convert paragraphs into custom grammar clauses
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
            which_modes = ["none"]
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
        #forest = {mode:{ii:None for ii in range(0, num_clusters)}
        #        for mode in which_modes}
        forest = [None for ii in range(0, num_clusters)]
        dict_modifs = {mode:None for mode in which_modes}
        #
        #Store the info in this instance
        self._store_info(clusters_NLP, "clusters_NLP")
        self._store_info(num_clusters, "num_clusters")
        self._store_info(forest, "forest")
        self._store_info(dict_modifs, "modifs")
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
            forest[ii] = [None for jj in range(0, num_sentences)]
            #Print some notes
            if do_verbose:
                print("\n---------------\n")
                print("Building structure for cluster {2} ({1} words):\n{0}\n"
                        .format(curr_cluster, num_words, ii))
            #

            #Iterate through sentences within this cluster
            for jj in range(0, num_sentences):
                curr_sentence = curr_cluster[jj]
                num_words = len(curr_sentence)
                #Print some notes
                if do_verbose:
                    print("Working on sentence #{1} of cluster #{0}:\n{2}"
                            .format(ii, jj, curr_sentence))
                #

                #Split sentences into dictionary of clauses
                tmp_res = self._generate_clauses_from_sentence(
                                                sentence_NLP=curr_sentence)
                curr_clauses = tmp_res["clauses"]
                list_pos_NLP = tmp_res["list_pos_NLP"]
                list_pos_clause = tmp_res["list_pos_clause"]
                list_iverbs = tmp_res["list_iverbs"]
                ids_conjoined = tmp_res["ids_conjoined"]
                list_iskeyword = tmp_res["list_iskeyword"]
                list_isuseless = tmp_res["list_isuseless"]
                ids_nounchunks = tmp_res["ids_nounchunks"]
                flags_nounchunks = tmp_res["flags_nounchunks"]

                #Print some notes
                if do_verbose:
                    print("Sentence {0}: '{1}'".format(jj, curr_sentence))
                    print("NLP p.o.s.:\n{0}".format(list_pos_NLP))
                    print("Clausal p.o.s.:\n{0}".format(list_pos_clause))
                    print("Conjoined ids:\n{0}".format(ids_conjoined))
                    print("Noun-chunk ids:\n{0}".format(ids_nounchunks))
                    print("Noun-chunks:")
                    for curr_ind in set(ids_nounchunks):
                        if (curr_ind is None):
                            continue
                        #
                        print("{0}: {1}\nFlags = {2}"
                            .format(curr_ind,
                            [item for item in curr_sentence
                            if (ids_nounchunks[item.i-curr_sentence.start]
                                == curr_ind)],
                            flags_nounchunks[curr_ind]))
                    #
                    print("Clauses:")
                    for curr_key in curr_clauses:
                        print("{0}: {1}".format(curr_key,
                                                curr_clauses[curr_key]))
                #

                #Store the clauses and information
                forest[ii][jj] = {"pos_NLP":list_pos_NLP,
                                    "pos_clause":list_pos_clause,
                                    "flags":flags_nounchunks,
                                    "ids_nounchunk":ids_nounchunks,
                                    "iverbs":list_iverbs,
                                    "iskeyword":list_iskeyword,
                                    "isuseless":list_isuseless,
                                    "clauses":curr_clauses}
                #
            #

            #Print some notes
            if do_verbose:
                print("\n---\nGrammar structure for this cluster complete!")
                print("Modifying structure based on given modes ({0})..."
                        .format(which_modes))
            #

            #Generate diff. versions of grammar structure (orig, trim, anon...)
            #for curr_mode in which_modes:
            #    forest[curr_mode][ii] = self._modify_structure(mode=curr_mode,
            #                                    struct_verbs=curr_struct_verbs,
            #                                    struct_words=curr_struct_words)
            #
        #

        #Generate final modif across all clusters for each mode
        for curr_mode in which_modes:
            #Assemble modif for each requested mode
            curr_modif = "\n".join([self._modify_structure(mode=curr_mode,
                                            cluster_NLP=clusters_NLP[ii],
                                            cluster_info=forest[ii]
                                            )["text_updated"]
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
                    print("Updated text: {0}".format(dict_modifs[curr_mode]))
                    print("---")
            #
            print("\n---------------\n")
        #

        #print(checkthatallkeywordwordsincluded_errifnot)
        return
    #

    ##Method: _add_word_to_clause()
    ##Purpose: Add word, such as sentence subjects, to corresponding verb clause
    def _add_word_to_clause(self, word, i_verb, sentence_NLP, clauses_text, clauses_ids, list_pos_NLP, list_pos_clause, list_iverbs, ids_conjoined, ids_nounchunks):
        """
        Method: _get_verb_connector
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Recursively examine and store information for each word within an NLP-sentence.
        """
        #Extract global variables
        do_verbose = self._get_info("do_verbose")
        i_shift = sentence_NLP.start
        i_word = (word.i - i_shift)
        roots = list(word.ancestors)
        dict_conv = config.dict_conv_pos
        #
        #Noun-chunk for this word
        new_chunk = self._get_nounchunk(word=word,
                                sentence_NLP=sentence_NLP,
                                ids_nounchunks=ids_nounchunks) #,
                                #do_make_new_if_none=True)
        #
        #Booleans
        is_conjoined = (self._is_pos_word(word, pos="CONJOINED")
                        and (len(roots) > 0)
                        and ((list_pos_clause[roots[0].i-i_shift] is None)
                            or self._is_pos_word(word,
                            pos=dict_conv[list_pos_clause[roots[0].i-i_shift]])
                        )) #Same p.o.s must be valid with root
        is_subject = self._is_pos_word(word, pos="SUBJECT",
                                        ids_nounchunks=ids_nounchunks)
        is_dirobject = self._is_pos_word(word, pos="DIRECT_OBJECT")
        is_prepobject = self._is_pos_word(word, pos="PREPOSITIONAL_OBJECT",
                                        ids_nounchunks=ids_nounchunks)
        is_aux = self._is_pos_word(word, pos="AUX")

        #Print some notes
        if do_verbose:
            print("Adding word '{0}' to verb-clauses.".format(word))
        #

        #Store word according to its p.o.s. as applicable
        #If this is a conjoined node, record+copy from start of conjoined chain
        if is_conjoined:
            #Iterate through roots and find start of conjoined chain
            for curr_root in roots:
                #If this root also conjoined, keep looking
                if self._is_pos_word(curr_root, pos="CONJOINED"):
                    continue
                #
                #Otherwise, if not conjoined, must be start of chain
                else:
                    root_conj = curr_root
                    #Break out of this search
                    break
                #
            #
            #Store the index of the start of this chain
            i_root_conj = (root_conj.i - i_shift)
            ids_conjoined[i_word] = i_root_conj
            #
            #Copy over traits of the conjoined root, if valid chunks
            if (new_chunk is not None):
                list_pos_NLP[i_word] = list_pos_NLP[i_root_conj]
                for curr_ind in new_chunk["ids"]:
                    list_pos_clause[curr_ind] = list_pos_clause[i_root_conj]
                #
                #Store in the same verb-clause as conjoined root, if relevant
                if (list_pos_clause[i_word] is not None):
                    pos_clause = list_pos_clause[i_word]
                    clauses_text[i_verb][pos_clause].append(new_chunk["text"])
                    #
                    if (pos_clause == "auxs"): #Singular entry
                        clauses_ids[i_verb][pos_clause] += new_chunk["ids"]
                    else: #List of entries
                        clauses_ids[i_verb][pos_clause].append(new_chunk["ids"])
                        clauses_ids[i_verb]["clause_nounchunks"].append(
                                                        new_chunk["chunk_id"])
            #
        #
        #For subjects
        elif (is_subject or is_dirobject or is_prepobject):
            #Do nothing if not noun
            if (new_chunk is None):
                return
            #
            #Set clausal part-of-speech
            if is_subject:
                pos_clause = "subjects"
            elif is_dirobject:
                pos_clause = "dir_objects"
            elif is_prepobject:
                pos_clause = "prep_objects"
            else:
                raise ValueError("Err: Missing condition?")
            #

            #Find verb index for this word so as to look up correct verb-clause
            #for curr_root in roots:
            #    #Look for nearest root verb
            #    if (self._is_pos_word(curr_root, pos="VERB",
            #                            ids_nounchunks=ids_nounchunks,
            #                            do_exclude_nounverbs=True)):
            #        #Extract index and stop search
            #        #i_verb = curr_root.i
            #        break #Exit the loop
            #
            #Fill in notes for this noun-chunk
            for curr_ind in new_chunk["ids"]:
                list_pos_clause[curr_ind] = pos_clause
            #
            #Add to verb-clause
            clauses_text[i_verb][pos_clause].append(new_chunk["text"])
            clauses_ids[i_verb][pos_clause].append(new_chunk["ids"])
            clauses_ids[i_verb]["clause_nounchunks"].append(
                                                        new_chunk["chunk_id"])
        #
        #For auxes
        elif is_aux:
            pos_clause = "auxs"
            #Find verb index for this word so as to look up correct verb-clause
            #for curr_root in roots:
            #    #Look for nearest root verb
            #    if (self._is_pos_word(curr_root, pos="VERB",
            #                            ids_nounchunks=ids_nounchunks,
            #                            do_exclude_nounverbs=True)):
            #        #Extract index and stop search
            #        i_verb = curr_root.i
            #        break #Exit the loop
            #
            #Fill in notes for this aux
            list_pos_clause[i_word] = pos_clause
            #
            #Add to verb-clause
            clauses_text[i_verb][pos_clause].append(word.text)
            clauses_ids[i_verb][pos_clause].append(i_word)
        #
        #Do nothing if not an important part-of-speech for clauses
        else:
            pass
        #

        #Exit the method; storage was implicitly updated
        return
    #

    ##Method: _assign_nounchunk_ids()
    ##Purpose: Assign ids to noun-chunks
    def _assign_nounchunk_ids(self, sentence_NLP, do_keyword_check):
        """
        Method: _assign_nounchunk_ids
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Characterize and store a word within grammar structure.
        """
        #Extract global variables
        do_verbose = self._get_info("do_verbose")
        keyword_obj = self._get_info("keyword_obj")
        num_words = len(sentence_NLP)
        i_shift = sentence_NLP.start
        idx_shift = sentence_NLP[0].idx

        #Fetch all NLP-identifieid noun-chunks for this sentence
        noun_chunks = list(sentence_NLP.noun_chunks)

        #Initialize container for noun-chunk ids
        ids_nounchunks = np.asarray([None]*len(sentence_NLP))

        #Iterate through noun-chunks and assign ids
        i_track = 0
        for ii in range(0, len(noun_chunks)):
            #Copy over previous noun-chunk id if separated by 'of'
            if ((ii > 0)
                    and (((noun_chunks[ii][0].i-i_shift)
                            - (noun_chunks[ii-1][-1].i-i_shift)) == 2)
                    and (sentence_NLP[noun_chunks[ii-1][-1].i-i_shift + 1
                                    ].text.lower() == "of")):
                curr_id = ids_nounchunks[noun_chunks[ii-1][0].i-i_shift] #Prev.
                #Assign id to 'of'
                curr_id = ids_nounchunks[noun_chunks[ii-1][-1].i-i_shift + 1
                                    ] = curr_id
            #Otherwise, set new chunk id
            else:
                curr_id = i_track
            #

            #Apply id to entire chunk
            for curr_word in noun_chunks[ii]:
                ids_nounchunks[curr_word.i-i_shift] = curr_id

            #Increment chunk counter
            i_track += 1
        #

        #Iterate through words and assign any missed nouns to noun-chunks
        for ii in range(0, num_words):
            #Skip words with noun-chunks already
            if (ids_nounchunks[ii] is not None):
                continue

            #Skip words that are not nouns
            if (not self._is_pos_word(sentence_NLP[ii], pos="NOUN")):
                continue

            #Chain together adjacent nouns
            i_end = None
            for jj in range((ii+1), num_words):
                if (not self._is_pos_word(sentence_NLP[jj], pos="NOUN")):
                    i_end = jj
                    break
            #
            #If last word in sentence is noun and in chunk, handle edge case
            if ((i_end is None) and (self._is_pos_word(
                                    sentence_NLP[num_words-1], pos="NOUN"))):
                i_end = num_words
            #
            #Set noun-chunk id for this latest chunk
            ids_nounchunks[ii:i_end] = i_track

            #Increment chunk counter
            i_track += 1
        #

        #Ensure that all keywords are included in noun-chunks
        check_spans = self._check_importance(text=sentence_NLP.text,
                                include_Ipronouns=False, include_terms=False,
                                include_etal=False, keyword_objs=[keyword_obj],
                                version_NLP=sentence_NLP
                                )["charspans_keyword"]
        #
        #Iterate through flagged keyword spans and ensure included in noun-chunk
        for ii in range(0, len(check_spans)):
            #Extract current keyword span
            curr_span = check_spans[ii]
            #Convert keyword character span into sentence-word ids; ignore punct
            curr_inds = [ind for ind in range(0, len(sentence_NLP))
                        if (any([item.isalnum() #Ignore any punctuation strings
                                for item in sentence_NLP[ind].text]) #Clunky
                        and ((sentence_NLP[ind].idx-idx_shift) >= curr_span[0])
                        and ((sentence_NLP[ind].idx-idx_shift) <=curr_span[1]))]
            #Check this span has assigned noun-chunks
            is_found = all([(ids_nounchunks[ind] is not None)
                            for ind in curr_inds])
            #
            #Throw error if span not found within any noun-chunks
            #if (not is_found):
            #    raise ValueError(
            #        ("Err: Keyword not in noun-chunks!!\n"
            #            +"{0}\n{1}\n{2}\n{3}\n{4}\n{5}")
            #                    .format(sentence_NLP, noun_chunks, check_spans,
            #                            curr_inds, ids_nounchunks, i_shift))
            #
            #If the above error is ever called, could use code below
            #But should be carefully tested first!
            #
            #Add/Overwrite as own noun-chunk if not fully in noun-chunk already
            if (not is_found):
                ids_nounchunks[min(curr_inds):(max(curr_inds)+1)] = i_track
                i_track += 1
        #

        #Print some notes
        if do_verbose:
            num_words = len(sentence_NLP)
            tmp_sets = [[sentence_NLP[ind2] for ind2 in range(0, num_words)
                        if (ids_nounchunks[ind2] == ind1)]
                        for ind1 in set(ids_nounchunks) if (ind1 is not None)]
            print("The following noun-chunks have been assigned:\n{0}\nFor: {1}"
                    .format(tmp_sets, sentence_NLP))

        #Return the assigned ids
        return ids_nounchunks
    #

    ##Method: _generate_clauses_from_sentence()
    ##Purpose: Extract verb-clauses from given NLP sentence
    def _generate_clauses_from_sentence(self, sentence_NLP):
        """
        Method: _recurse_NLP_categorization
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Recursively examine and store information for each word within an NLP-sentence.
        """
        #Extract global variables
        do_verbose = self._get_info("do_verbose")
        num_words = len(sentence_NLP)
        i_shift = sentence_NLP.start
        #Print some notes
        if do_verbose:
            print(("-"*60)+"\nCURRENT SENTENCE: {0}".format(sentence_NLP))
        #

        #Assign noun-chunks for this sentence
        ids_nounchunks = self._assign_nounchunk_ids(sentence_NLP,
                                                    do_keyword_check=True)
        #

        #Initialize storage to hold key sentence information
        list_pos_NLP = [None]*num_words
        list_pos_clause = [None]*num_words
        list_iverbs = [None]*num_words
        list_ischecked = np.zeros(num_words).astype(bool)
        list_isuseless = np.zeros(num_words).astype(bool)
        list_iskeyword = np.zeros(num_words).astype(bool)
        ids_conjoined = [None]*num_words
        if (len([item for item in ids_nounchunks if (item is not None)]) > 0):
            flags_nounchunks = [None]*max([(item+1) for item in ids_nounchunks
                                            if (item is not None)])
        else:
            flags_nounchunks = []
        #

        #Iterate through and identify ids of all clausal verbs
        inds_verbs = []
        for curr_word in sentence_NLP:
            #Get index of current word
            i_word = (curr_word.i - i_shift)

            #Determine if this word is a clausal verb
            is_verb = self._is_pos_word(curr_word, pos="VERB",
                                ids_nounchunks=ids_nounchunks,
                                do_exclude_nounverbs=True)
            is_root = self._is_pos_word(curr_word, pos="ROOT",
                                ids_nounchunks=ids_nounchunks
                                ) #Handle incomplete sentence case
            if (is_verb or is_root):
                inds_verbs.append(i_word)
            #
        #

        #Sort clausal verb indices by length of subtree (large to small)
        inds_verbs_ordered = inds_verbs.copy()
        inds_verbs_ordered.sort(key=lambda x:len(list(sentence_NLP[x].subtree)))
        inds_verbs_ordered = inds_verbs_ordered[::-1] #Descending subtree size

        #Initialize container for clauses
        dict_clauses_text = {}
        dict_clauses_ids = {}

        #Iterate through clausal verbs and initialize clausal containers
        i_track = 0
        for curr_iverb in inds_verbs_ordered:
            #Extract useful characteristics of current verb
            curr_verb = sentence_NLP[curr_iverb]

            #Initialize containers for this clause
            curr_dict_text = {"verb":curr_verb.text, "auxs":[], "subjects":[],
                                "dir_objects":[], "prep_objects":[],
                                "order":i_track, "verbtype":None}
            curr_dict_ids = {"verb":curr_iverb, "auxs":[], "subjects":[],
                                "dir_objects":[], "prep_objects":[],
                                "order":i_track, "verbtype":None,
                                "chain_iverbs":None, "clause_nounchunks":[]}
            #

            #Store these clausal containers
            dict_clauses_text[curr_iverb] = curr_dict_text
            dict_clauses_ids[curr_iverb] = curr_dict_ids

            #Increment clause counter
            i_track += 1
        #

        #Recursively navigate NLP-tree from the root and fill in clauses
        self._recurse_NLP_tree(node=sentence_NLP.root,
                i_verb=(sentence_NLP.root.i - i_shift),
                sentence_NLP=sentence_NLP, list_isuseless=list_isuseless,
                clauses_text=dict_clauses_text, clauses_ids=dict_clauses_ids,
                list_iskeyword=list_iskeyword, ids_conjoined=ids_conjoined,
                ids_nounchunks=ids_nounchunks,flags_nounchunks=flags_nounchunks,
                list_ischecked=list_ischecked, list_pos_NLP=list_pos_NLP,
                list_pos_clause=list_pos_clause, list_iverbs=list_iverbs)
        #

        #Set verb types using extracted verbs and aux words
        for curr_iverb in inds_verbs_ordered:
            #Fetch containers for this clause
            curr_dict_text = dict_clauses_text[curr_iverb]
            curr_dict_ids = dict_clauses_ids[curr_iverb]

            #Set the tense/type of verb (if verb and not e.g. root noun)
            if self._is_pos_word(sentence_NLP[curr_iverb], pos="VERB"):
                verbtype = self._tense_verb(i_verb=curr_iverb,
                                        sentence_NLP=sentence_NLP,
                                        i_auxs=curr_dict_ids["auxs"])
            else:
                verbtype = None
            curr_dict_text["verbtype"] = verbtype
            curr_dict_ids["verbtype"] = verbtype
        #

        #Print some notes
        if do_verbose:
            print("Clausal breakdown complete.\nClauses:")
            for curr_ind in inds_verbs_ordered:
                print("id={0}: verb={1}:\n{2}\n{3}\n-"
                        .format(curr_ind, sentence_NLP[curr_ind],
                                dict_clauses_ids[curr_ind],
                                dict_clauses_text[curr_ind]))
            print("---")
        #

        #Return the dictionary of clauses
        return {"clauses":{"text":dict_clauses_text, "ids":dict_clauses_ids},
                "list_pos_NLP":list_pos_NLP, "list_pos_clause":list_pos_clause,
                "ids_conjoined":ids_conjoined, "list_isuseless":list_isuseless,
                "list_iskeyword":list_iskeyword,"ids_nounchunks":ids_nounchunks,
                "flags_nounchunks":flags_nounchunks,
                "list_iverbs":list_iverbs}
    #

    ##Method: _get_nounchunk()
    ##Purpose: Retrieve noun chunk assigned the given id (index)
    def _get_nounchunk(self, word, sentence_NLP, ids_nounchunks): #, do_make_new_if_none=False):
        """
        Method: _get_nounchunks
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Recursively examine and store information for each word within an NLP-sentence.
        """
        #Extract global variables
        do_verbose = self._get_info("do_verbose")
        i_shift = sentence_NLP.start
        i_word = (word.i - i_shift) #Index of given word
        word_dep = word.dep_ #NLP dep. of given word
        chunkid = ids_nounchunks[i_word] #Id of current wordchunk

        #Return singular word if noun-esque but no noun-chunk for this word
        if (chunkid is None): # and
            #        (word_dep.endswith("subj") or word_dep.endswith("obj"))):
            #return None #{"text":word.text, "ids":[i_word], "chunk_id":None}
            #If new one requested, make new one
            #if do_make_new_if_none:
                #Set values for new chunk
            #    chunk_text = word.text
            #    word_ids = [i_word]
            #    chunkid = (max([item for item in ids_nounchunks
            #                    if (item is not None)])
            #                + 1) #Increment by one to start new chunk
            #    ids_nounchunks[i_word] = chunkid
                #Return the newly created noun-chunk information
            #    return {"text":chunk_text, "ids":word_ids, "chunk_id":chunkid}
            #
            #Otherwise, return None
            #else:
            #    return None
            #
            return None
        #

        #Extract text and ids of this chunk
        set_chunk = [item for item in sentence_NLP
                    if (ids_nounchunks[item.i-i_shift] == chunkid)]
        chunk_text = " ".join([item.text for item in set_chunk]) #Text version
        word_ids = [(item.i-i_shift) for item in set_chunk] #id version

        #Return extracted text and ids of chunk
        return {"text":chunk_text, "ids":word_ids, "chunk_id":chunkid}
    #

    ##Method: _modify_structure
    ##Purpose: Modify given grammar structure, following specifications of the given mode
    def _modify_structure(self, cluster_NLP, cluster_info, mode):
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
        num_sents = len(cluster_NLP)
        arrcluster_NLP = [np.asarray(item) for item in cluster_NLP]
        #

        #Initialize storage for modified versions of grammar structure
        arrs_is_keep = [np.ones(len(item)).astype(bool) for item in cluster_NLP]
        sents_updated = [item.text for item in cluster_NLP] #Init. sentences
        #
        #Print some notes
        if do_verbose:
            print("\n> Running _modify_structure!")
            print("Number of sentences in cluster: {1}\nRequested mode: {0}"
                    .format(mode, num_sents))
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
            #Iterate through sentences
            for ii in range(0, num_sents):
                curr_is_useless = cluster_info[ii]["isuseless"]
                #Remove words marked as useless
                arrs_is_keep[ii][curr_is_useless] = False
                #
                #Update latest text with these updates
                tmp_res = arrcluster_NLP[ii][arrs_is_keep[ii]]
                sents_updated[ii] = " ".join(item.text for item in tmp_res)
            #
            #Print some notes
            if do_verbose:
                print("skim modifications complete.\nUpdated text:\n{0}\n"
                        .format(sents_updated))
            #
        #
        #For trim: Remove clauses without any important information/subclauses
        if do_trim:
            #Print some notes
            if do_verbose:
                print("> Applying trim modifications...")
                print("Iterating through clause chains...")
            #
            #Iterate through sentences
            for ii in range(0, num_sents):
                curr_clauses_ids = cluster_info[ii]["clauses"]["ids"]
                #Sort clausal keys by verb hierarchical order
                #list_iverbs_sorted = list(curr_clauses_ids.keys())
                #list_iverbs_sorted.sort(key=lambda x
                #                            :curr_clauses_ids[x]["order"])
                list_iverbs_all = list(curr_clauses_ids.keys())

                #Fetch ids of noun-chunks flagged as important
                curr_flags = cluster_info[ii]["flags"]
                curr_imp_chunkids = [ind for ind in range(0, len(curr_flags))
                                        if ((curr_flags[ind] is not None)
                                            and (curr_flags[ind]["is_any"]))]

                #Fetch keys for clauses marked as important
                #list_iverbs_imp_raw = [ind for ind in list_iverbs_sorted
                list_iverbs_imp_raw = [ind for ind in list_iverbs_all
                                if any([(item in
                                    curr_clauses_ids[ind]["clause_nounchunks"])
                                    for item in curr_imp_chunkids])]

                #Now add in keys that precede imp. clauses in heirarchical order
                #list_iverbs_imp = [
                #        list_iverbs_sorted[0:list_iverbs_sorted.index(item)+1]
                #        for item in list_iverbs_imp_raw]
                list_iverbs_imp = [curr_clauses_ids[item]["chain_iverbs"]
                                    for item in list_iverbs_imp_raw
                                ] #Gather verbs that branch down to imp. verbs
                list_iverbs_imp = [item2 for item1 in list_iverbs_imp
                                    for item2 in item1] #Flatten
                list_iverbs_imp = set(list_iverbs_imp) #Take unique

                #Take note of important words within these spans
                list_word_iverbs = cluster_info[ii]["iverbs"]
                tmp_iskeep = np.array([True
                                if (list_word_iverbs[ind] in list_iverbs_imp)
                                else False
                                for ind in range(0, len(list_word_iverbs))])
                #

                #Remove any conjunctions if leading+trailing words not kept
                for jj in range(1, (len(tmp_iskeep)-1)):
                    is_conjunction = self._is_pos_word(cluster_NLP[ii][jj],
                                                        pos="CONJUNCTION")
                    is_punctuation = self._is_pos_word(cluster_NLP[ii][jj],
                                                        pos="PUNCTUATION")
                    if (is_conjunction or is_punctuation):
                        if ((not tmp_iskeep[jj-1]) or (not tmp_iskeep[jj+1])):
                            tmp_iskeep[jj] = False
                #

                #Merge notes of importance into global array of kept words
                arrs_is_keep[ii][~tmp_iskeep] = False
                #
                #Update latest text with these updates
                tmp_res = arrcluster_NLP[ii][arrs_is_keep[ii]]
                sents_updated[ii] = " ".join(item.text for item in tmp_res)
            #
            #Print some notes
            if do_verbose:
                print("trim modifications complete.\nUpdated text:\n{0}\n"
                        .format(sents_updated))
            #
        #
        #For anon: Replace mission-specific terms with anonymous placeholder
        if do_anon:
            #Print some notes
            if do_verbose:
                print("> Applying anon modifications...")
            #
            placeholder_anon = config.placeholder_anon
            #Update latest text with these updates
            for ii in range(0, num_sents):
                sents_updated[ii] = keyword_obj.replace_keyword(
                                                text=sents_updated[ii],
                                                placeholder=placeholder_anon)
                sents_updated[ii] = self._streamline_phrase(
                                            text=sents_updated[ii],
                                            do_streamline_etal=True)
            #
            #Print some notes
            if do_verbose:
                print("anon modifications complete.\nUpdated text:\n{0}\n"
                        .format(sents_updated))
            #
        #

        #Join and cleanse the text to finalize it
        text_updated = " ".join(sents_updated)
        text_updated = self._streamline_phrase(text=text_updated,
                                            do_streamline_etal=False)
        #

        #Return dictionary containing the updated grammar structures
        if do_verbose:
            print("Run of _modify_structure() complete.")
            print(("Mode: {0}\nUpdated text: {1}")
                    .format(mode, text_updated))
        #
        return {"mode":mode, "text_updated":text_updated,
                "arrs_is_keep":arrs_is_keep}
    #

    ##Method: _recurse_NLP_tree
    ##Purpose: Recursively explore each word of NLP-sentence and characterize
    def _recurse_NLP_tree(self, node, i_verb, sentence_NLP, clauses_text, clauses_ids, list_pos_NLP, list_pos_clause, list_iverbs, list_ischecked, list_iskeyword, list_isuseless, ids_conjoined, ids_nounchunks, flags_nounchunks):
        """
        Method: _get_verb_connector
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Recursively examine and store information for each word within an NLP-sentence.
        """
        #Extract global variables
        do_verbose = self._get_info("do_verbose")
        i_shift = sentence_NLP.start
        i_node = (node.i - i_shift)
        dep_node = node.dep_
        #

        #If verb index, update latest i_verb and fetch used subj. if needed
        if (i_node in clauses_text): #Only verb ids used to build clauses
            #Update i_verb chain with new verb
            if (node.i == sentence_NLP.root.i): #If sentence root, no chain yet
                clauses_ids[i_node]["chain_iverbs"] = [i_node]
            else: #Otherwise if later verb, copy over and add to pre. chain
                clauses_ids[i_node]["chain_iverbs"] = (
                        clauses_ids[i_verb]["chain_iverbs"] #Prev. chain
                        + [i_node]) #Tack on previous i_verb
            #
            #Update global verb id
            i_verb = i_node
            roots = list(node.ancestors)
            #
            #If no subjects to the left, pull subject from root as able
            tmp_bool = any([self._is_pos_word(item, pos="SUBJECT")
                            for item in node.lefts])
            #if ((not tmp_bool) and (len(roots) > 0)):
            #    if self._is_pos_word(roots[0], pos="NOUN"):
                    #Extract noun-chunk for this noun
            #        new_chunk = self._get_nounchunk(word=roots[0],
            #                                sentence_NLP=sentence_NLP,
            #                                ids_nounchunks=ids_nounchunks)
                    #
                    #Add to verb-clause
            #        clauses_text[i_verb]["subjects"].append(new_chunk["text"])
            #        clauses_ids[i_verb]["subjects"].append(new_chunk["ids"])
            #        clauses_ids[i_verb]["clause_nounchunks"].append(
            #                                            new_chunk["chunk_id"])
            #    elif self._is_pos_word(roots[0], pos="VERB"): #If conjoined
                    #Add to verb-clause
            #        clauses_text[i_verb]["subjects"] = clauses_text[
            #                                roots[0].i-i_shift]["subjects"]
            #        clauses_ids[i_verb]["subjects"] = clauses_ids[
            #                                roots[0].i-i_shift]["subjects"]
            #        tmp_list = list(set([ids_nounchunks[item[0]]
            #                    for item in clauses_ids[i_verb]["subjects"]]))
            #        clauses_ids[i_verb]["clause_nounchunks"] += tmp_list
            if ((not tmp_bool) and (len(roots) > 0)):
                for curr_root in roots:
                    if self._is_pos_word(curr_root, pos="NOUN"):
                        #Extract noun-chunk for this noun
                        new_chunk = self._get_nounchunk(word=curr_root,
                                                sentence_NLP=sentence_NLP,
                                                ids_nounchunks=ids_nounchunks)
                        #
                        #Add to verb-clause
                        clauses_text[i_verb]["subjects"].append(
                                                            new_chunk["text"])
                        clauses_ids[i_verb]["subjects"].append(new_chunk["ids"])
                        clauses_ids[i_verb]["clause_nounchunks"].append(
                                                        new_chunk["chunk_id"])
                        #
                        #Stop the search
                        break
                    elif ((curr_root.i-i_shift) in clauses_ids): #If verb
                        #Add to verb-clause
                        clauses_text[i_verb]["subjects"] = clauses_text[
                                                curr_root.i-i_shift]["subjects"]
                        clauses_ids[i_verb]["subjects"] = clauses_ids[
                                                curr_root.i-i_shift]["subjects"]
                        tmp_list = list(set([ids_nounchunks[item[0]]
                                for item in clauses_ids[i_verb]["subjects"]]))
                        clauses_ids[i_verb]["clause_nounchunks"] += tmp_list
                        #
                        #Stop the search
                        break
                    #
                #
            #
        #

        #Skip if this node has already been characterized
        if list_ischecked[i_node]:
            #Call this recursive method on branch nodes
            #For left nodes
            for left_node in node.lefts:
                self._recurse_NLP_tree(node=left_node,
                        sentence_NLP=sentence_NLP, i_verb=i_verb,
                        list_pos=list_pos, ids_conjoined=ids_conjoined,
                        ids_nounchunks=ids_nounchunks,
                        list_iskeyword=list_iskeyword,
                        list_isuseless=list_isuseless,
                        flags_nounchunks=flags_nounchunks,
                        list_ischecked=list_ischecked)
            #
            #For right nodes
            for right_node in node.rights:
                self._recurse_NLP_tree(node=right_node,
                        sentence_NLP=sentence_NLP, i_verb=i_verb,
                        list_pos=list_pos, ids_conjoined=ids_conjoined,
                        ids_nounchunks=ids_nounchunks,
                        list_iskeyword=list_iskeyword,
                        list_isuseless=list_isuseless,
                        flags_nounchunks=flags_nounchunks,
                        list_ischecked=list_ischecked)
            #

            #Exit the method early
            return
        #

        #Characterize current node
        list_pos_NLP[i_node] = node.pos_ #Store current NLP part-of-speech
        list_iverbs[i_node] = i_verb
        #

        #Add this word to corresponding verb-clause as needed, if not verb
        if (i_node not in clauses_text): #Do not add verbs again
            #Do not add to clause if already added
            tmp_ind = ids_nounchunks[i_node]
            #if ((tmp_ind is not None) and
            #        (tmp_ind not in clauses_ids[i_verb]["clause_nounchunks"])):
            if ((tmp_ind is None) or
                    (tmp_ind not in clauses_ids[i_verb]["clause_nounchunks"])):
                self._add_word_to_clause(word=node, clauses_text=clauses_text,
                        clauses_ids=clauses_ids, sentence_NLP=sentence_NLP,
                        ids_conjoined=ids_conjoined, list_pos_NLP=list_pos_NLP,
                        list_pos_clause=list_pos_clause,list_iverbs=list_iverbs,
                        i_verb=i_verb, ids_nounchunks=ids_nounchunks)

        #Store importance flags for associated noun-chunk, if applicable
        if (ids_nounchunks[i_node] is not None):
            #Fetch id of current chunk
            curr_chunkid = ids_nounchunks[i_node]

            #Characterize if this noun-chunk not already characterized
            if (curr_chunkid not in flags_nounchunks):
                #Fetch noun-chunk and set flags for this noun-chunk
                curr_set_nounchunk = self._get_nounchunk(word=node,
                                                sentence_NLP=sentence_NLP,
                                                ids_nounchunks=ids_nounchunks)
                tmp_res = self._set_nounchunk_flags(sentence_NLP=sentence_NLP,
                                            dict_nounchunk=curr_set_nounchunk)
                #
                #Store the fetched flags
                flags_nounchunks[curr_chunkid] = tmp_res["flags"]
                #Mark off any important words
                for curr_impid in tmp_res["ids_keywords"]:
                    list_iskeyword[curr_impid] = True
        #

        #Mark word as useless (e.g., if some adverb) as applicable
        list_isuseless[i_node] = (self._is_pos_word(node, pos="USELESS")
                                and (not list_iskeyword[i_node]))

        #Mark this node as characterized
        list_ischecked[i_node] = True

        #Call this recursive method on branch nodes
        #For left nodes
        for left_node in node.lefts:
            self._recurse_NLP_tree(node=left_node, sentence_NLP=sentence_NLP,
                    list_pos_NLP=list_pos_NLP, list_pos_clause=list_pos_clause,
                    list_iverbs=list_iverbs, ids_conjoined=ids_conjoined,
                    ids_nounchunks=ids_nounchunks,list_iskeyword=list_iskeyword,
                    flags_nounchunks=flags_nounchunks,
                    list_isuseless=list_isuseless, i_verb=i_verb,
                    clauses_text=clauses_text, clauses_ids=clauses_ids,
                    list_ischecked=list_ischecked)
        #
        #For right nodes
        for right_node in node.rights:
            self._recurse_NLP_tree(node=right_node, sentence_NLP=sentence_NLP,
                    list_pos_NLP=list_pos_NLP, list_pos_clause=list_pos_clause,
                    list_iverbs=list_iverbs, ids_conjoined=ids_conjoined,
                    ids_nounchunks=ids_nounchunks,list_iskeyword=list_iskeyword,
                    flags_nounchunks=flags_nounchunks, i_verb=i_verb,
                    list_isuseless=list_isuseless,
                    clauses_text=clauses_text, clauses_ids=clauses_ids,
                    list_ischecked=list_ischecked)
        #

        #Exit the method
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

    ##Method: _set_nounchunk_flags()
    ##Purpose: Set importance flags and mark keywords for a given noun-chunk
    def _set_nounchunk_flags(self, dict_nounchunk, sentence_NLP):
        #Set global variables
        text_nounchunk = dict_nounchunk["text"]
        ids_in_nounchunk = dict_nounchunk["ids"]
        abschar_chunkstart = sentence_NLP[ids_in_nounchunk[0]].idx
        kobj = self._get_info("keyword_obj")

        #Set importance flags and fetch char spans of kw. for this noun-chunk
        tmp_res = self._check_importance(text_nounchunk, keyword_objs=[kobj],
                                        version_NLP=sentence_NLP)
        flags = tmp_res["bools"]
        charspans_keywords = tmp_res["charspans_keyword"]

        #Determine which words fall within any keyword character spans
        ids_wordsthatarekeywords = []
        if (len(charspans_keywords) > 0):
            #Iterate through ids of each word in noun-chunk
            for curr_id in ids_in_nounchunk:
                #Get relative char. span of current word
                char_wordstart = (sentence_NLP[curr_id].idx
                                    - abschar_chunkstart) #Rel. char start ind
                char_wordend = (
                        (sentence_NLP[curr_id].idx+len(sentence_NLP[curr_id]))
                                    - abschar_chunkstart) #Rel. char end ind

                #Check for intersects with flagged keywords
                is_intersects = [((char_wordstart >= item[0])
                                    and (char_wordend <= item[1]))
                                for item in charspans_keywords]

                #Store this id if intersection with flagged keywords
                if any(is_intersects):
                    ids_wordsthatarekeywords.append(curr_id)
            #
        #

        #Return the flags and ids
        return {"flags":flags, "ids_keywords":ids_wordsthatarekeywords}
    #

    ##Method: _tense_verb()
    ##Purpose: Set verb tense using verb and auxs
    def _tense_verb(self, i_verb, i_auxs, sentence_NLP):
        """
        Method: _tense_verb
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Run external natural language processing NLP package on given text.
        """
        #Set global variables
        ordered_poss_values = ["FUTURE", "PURPOSE", "PRESENT", "PAST"
                                ] #Allowed verb types in priority order
        tags_past = config.tag_verb_past
        tags_present = config.tag_verb_present
        tags_future = config.tag_verb_future
        tags_purpose = config.tag_verb_purpose

        #Initialize containers for extracted information
        type_verbs = []

        #Iterate through verbs and auxs
        for curr_id in ([i_verb] + i_auxs):
            #Set current word traits
            curr_word = sentence_NLP[curr_id]
            word_dep = curr_word.dep_
            word_tag = curr_word.tag_

            #Determine if passive tense and store if applicable
            #if (word_dep in config.deps_passive):
            #    type_verbs.append("PASSIVE")
            #

            #Determine main tense of this word
            if (word_tag in tags_past): #For past tense
                tense = "PAST"
            elif (word_tag in tags_present): #For present tense
                tense = "PRESENT"
            elif (word_tag in tags_future): #For future tense
                tense = "FUTURE"
            elif (word_tag in tags_purpose): #For purpose tense
                tense = "PURPOSE"
            else: #Raise error if tense not recognized
                raise ValueError(("Err: Tag {4} of word {0} unknown!\n{1}"
                                +"\ndep={2}, pos={3}, tag={4}\nRoots: {5}")
                            .format(curr_word, sentence_NLP, word_dep,
                                    curr_word.pos_, word_tag,
                                    list(curr_word.ancestors)))

            #Store current tense
            type_verbs.append(tense)
        #

        #Set priority verb type
        type_fin = None
        for curr_type in ordered_poss_values:
            #If current type matches, keep it and stop search
            if (curr_type in type_verbs):
                type_fin = curr_type
                break
        #Throw error if no match found
        if (type_fin is None):
            raise ValueError(("Err: No match found!\nWord: {0}\n{1}"
                            +"\ndep={2}, pos={3}, tag={4}")
                        .format(curr_word, sentence_NLP, word_dep, word_pos,
                                word_tag))

        #Return finalized verb type
        return type_fin
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
    def generate_directory_TVT(self, dir_model, fraction_TVT, dict_texts, mode_TVT="uniform", do_shuffle=True, seed=10, do_verbose=None):
        """
        Method: generate_directory_TVT
        Purpose: !!!
        """
        ##Load global variables
        dataset = dict_texts
        filepath_dictinfo = config.path_TVTinfo
        name_folderTVT = [config.folders_TVT["train"],
                        config.folders_TVT["validate"],
                        config.folders_TVT["test"]]
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

        ##Load all unique paper identifiers (the bibcodes) from across texts
        unique_bibcodes = np.unique(
                            [dataset[key]["bibcode"] for key in dataset
                            ]).tolist()
        #Print some notes
        if do_verbose:
            print("\nDataset contains {0} papers with {1} unique bibcodes.\n"
                    .format(len(dataset), len(unique_bibcodes)))
        #

        ##Load all unique classifs across all texts
        count_classes = collections.Counter([dataset[key]["class"]
                                            for key in dataset])
        unique_classes = count_classes.keys()
        #Print some notes
        if do_verbose:
            print("\nClass breakdown of given dataset:\n{0}\n"
                    .format(count_classes))
        #

        ##Invert dataset so that classifs are stored under bibcode keys
        dict_bibcode_classifs = {key:[] for key in unique_bibcodes}
        dict_bibcode_textids = {key:[] for key in unique_bibcodes}
        for curr_key in dataset:
            curr_id = curr_key
            curr_classif = dataset[curr_key]["class"]
            curr_bibcode = dataset[curr_key]["bibcode"]
            #Store id for this bibcode
            dict_bibcode_textids[curr_bibcode].append(curr_key)
            #Store this classif for this bibcode, if not already stored
            if (curr_classif not in dict_bibcode_classifs[curr_bibcode]):
                dict_bibcode_classifs[curr_bibcode].append(curr_classif) #Store
            #
        #
        #Record the number of processed text ids for later
        num_textids = sum([len(dict_bibcode_textids[key])
                            for key in unique_bibcodes])
        #
        #Print some notes
        if do_verbose:
            print("\nGiven dataset inverted. Bibcode count: {0}"
                    .format(len(dict_bibcode_classifs)))
        #

        ##Label bibcodes with representative classif
        #Separate bibcodes based on number of unique associated classifs
        all_bibcodes_oneclassif = {key1:[key2 for key2 in unique_bibcodes
                                    if (dict_bibcode_classifs[key2] == [key1])]
                                    for key1 in unique_classes}
        all_bibcodes_multiclassif = [key for key in unique_bibcodes
                                    if (len(dict_bibcode_classifs[key]) > 1)]
        #
        #Print some notes
        if do_verbose:
            print(("\nNumber of processed text ids: {2}"
                    +"\nNumber of bibcodes with single unique classif: {0}"
                    +"\nNumber of bibcodes with multiple classifs: {1}"
                    +"\nNumber of bibcodes with multiple text ids: {3}")
                    .format(sum([len(all_bibcodes_oneclassif[key])
                                        for key in unique_classes]),
                            len(all_bibcodes_multiclassif),
                            num_textids,
                            len([key for key in dict_bibcode_textids
                                    if (len(dict_bibcode_textids[key]) > 1)])
            ))
        #
        #Throw an error if separated bibcode count does not equal original count
        tmp_check = (sum([len(all_bibcodes_oneclassif[key])
                            for key in unique_classes])
                        + len(all_bibcodes_multiclassif))
        if (tmp_check != len(unique_bibcodes)):
            raise ValueError("Err: Invalid separation of bibcodes occurred.\n"
                            +"{0} + {1} != {2}"
                            .format(len([all_bibcodes_oneclassif[key]
                                            for key in unique_classes]),
                                    len(all_bibcodes_multiclassif),
                                    len(unique_bibcodes)))
        #
        #Shuffle the list of multi-classif bibcodes, if requested
        if do_shuffle:
            np.random.shuffle(all_bibcodes_multiclassif)
        #
        #Partition multi-classif bibcodes into representative classif lists
        all_bibcodes_partitioned_lists = all_bibcodes_oneclassif.copy()
        all_bibcodes_partitioned_counts = {
                                    key:len(all_bibcodes_partitioned_lists[key])
                                    for key in unique_classes
                                    } #Starting count of bibcodes per repr.clf.
        #Iterate through multi-classif bibcodes
        for curr_bibcode in all_bibcodes_multiclassif:
            curr_classifs = dict_bibcode_classifs[curr_bibcode]
            #Determine which classif has least number of bibcodes so far
            curr_counts = [all_bibcodes_partitioned_counts[key]
                            for key in curr_classifs] #Current bibcode counts
            #Store this bibcode under representative classif with min. count
            ind_min = np.argmin(curr_counts) #Index of classif with min. count
            min_classif = curr_classifs[ind_min]
            all_bibcodes_partitioned_lists[min_classif
                                            ].append(curr_bibcode) #Append bibc.
            #Update count of this representative classif
            all_bibcodes_partitioned_counts[min_classif] += 1
        #
        #Throw an error if any bibcodes are missing/duplicated in partition
        tmp_check = sum(all_bibcodes_partitioned_counts.values())
        if (tmp_check != len(unique_bibcodes)):
            raise ValueError("Err: Bibcode count does not match original"
                            +" bibcodes.\n{0}\nvs. {1}"
                            .format(tmp_check, len(unique_bibcodes)))
        tmp_check = [item for key in all_bibcodes_partitioned_lists
                    for item in all_bibcodes_partitioned_lists[key]]#Part. bibc.
        if (sorted(tmp_check) != sorted(unique_bibcodes)):
            raise ValueError("Err: Bibcode partitioning does not match original"
                            +" bibcodes.\n{0}\nvs. {1}"
                            .format(tmp_check, unique_bibcodes))
        #
        #Print some notes
        if do_verbose:
            print("\nBibcode partitioning of representative classifs.:\n{0}\n"
                    .format(all_bibcodes_partitioned_counts))
        #

        ##Split bibcode counts into TVT sets: training, validation, testing =TVT
        valid_modes = ["uniform", "available"]
        fraction_TVT = np.asarray(fraction_TVT) / sum(fraction_TVT) #Normalize
        #For mode where training sets should be uniform in size
        if (mode_TVT.lower() == "uniform"):
            min_count = min(all_bibcodes_partitioned_counts.values())
            dict_split = {key:(np.round((fraction_TVT * min_count))).astype(int)
                        for key in unique_classes}#Partition per classif per TVT
            #Update split to send remaining bibcodes into testing datasets
            for curr_key in unique_classes:
                curr_max = all_bibcodes_partitioned_counts[curr_key]
                curr_used = (dict_split[curr_key][0] + dict_split[curr_key][1])
                dict_split[curr_key][2] = (curr_max - curr_used)
        #
        #For mode where training sets should use fraction of data available
        elif (mode_TVT.lower() == "available"):
            max_count = max(all_bibcodes_partitioned_counts.values())
            dict_split = {key:(np.round((fraction_TVT
                                * all_bibcodes_partitioned_counts[key]))
                                ).astype(int)
                        for key in unique_classes} #Partition per class per TVT
        #
        #Otherwise, throw error if mode not recognized
        else:
            raise ValueError(("Err: The given mode for generating the TVT"
                            +" directory {0} is invalid. Valid modes are: {1}")
                            .format(mode_TVT, valid_modes))
        #
        #Print some notes
        if do_verbose:
            print("Fractions given for TVT split: {0}\nMode requested: {1}"
                    .format(fraction_TVT, mode_TVT))
            print("Target TVT partition for bibcodes:")
            for curr_key in unique_classes:
                print("{0}: {1}".format(curr_key, dict_split[curr_key]))
        #
        #Verify splits add up to original file count
        for curr_key in unique_classes:
            if (all_bibcodes_partitioned_counts[curr_key]
                                                != sum(dict_split[curr_key])):
                raise ValueError("Err: Split did not use all data available!")
        #

        ##Prepare indices for extracting TVT sets per class
        dict_bibcodes_perTVT = {key:[None for ii in range(0, num_TVT)]
                                for key in unique_classes}
        for curr_key in unique_classes:
            #Fetch bibcodes represented by this class
            curr_list = np.asarray(all_bibcodes_partitioned_lists[curr_key])
            #
            #Fetch available indices to select these bibcodes
            curr_inds = np.arange(0,all_bibcodes_partitioned_counts[curr_key],1)
            #Shuffle, if requested
            if do_shuffle:
                np.random.shuffle(curr_inds)
            #
            #Split out the bibcodes
            i_start = 0 #Accumulated place within overarching index array
            for ii in range(0, num_TVT): #Iterate through TVT
                i_end = (i_start + dict_split[curr_key][ii]) #Ending point
                dict_bibcodes_perTVT[curr_key][ii] = curr_list[
                                                    curr_inds[i_start:i_end]]
                i_start = i_end #Update latest starting place in array
            #
        #
        #Throw an error if any bibcodes not accounted for
        tmp_check = [item2 for key in dict_bibcodes_perTVT
                    for item1 in dict_bibcodes_perTVT[key]
                    for item2 in item1] #All bibcodes used
        if (not np.array_equal(np.sort(tmp_check), np.sort(unique_bibcodes))):
            raise ValueError("Err: Split bibcodes do not match up with"
                            +" original bibcodes:\n\n{0}\nvs.\n{1}"
                        .format(np.sort(tmp_check), np.sort(unique_bibcodes)))
        #
        #Print some notes
        if do_verbose:
            print("\nIndices split per bibcode, per TVT. Shuffling={0}."
                    .format(do_shuffle))
            print("Number of indices per class, per TVT:")
            for curr_key in unique_classes:
                print("{0}: {1}".format(curr_key,
                        [len(item) for item in dict_bibcodes_perTVT[curr_key]]))
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
            for curr_key in unique_classes:
                os.mkdir(os.path.join(dir_model, curr_folder, curr_key))
        #
        #Print some notes
        if do_verbose:
            print("Created new directories for TVT files.\nStored in: {0}"
                    .format(dir_model))
        #

        ##Save texts to .txt files within class directories
        dict_info = {}
        saved_filenames = [None]*num_textids
        used_bibcodes = [None]*len(unique_bibcodes)
        used_textids = [None]*num_textids
        all_texts_partitioned_counts = {key1:{key2:0 for key2 in unique_classes}
                                        for key1 in name_folderTVT
                                        } #Count of texts per TVT, per classif
        #Iterate through classes
        i_track = 0
        i_bibcode = 0
        curr_key = None
        for repr_key in unique_classes:
            #Save each text to assigned TVT
            for ind_TVT in range(0, num_TVT): #Iterate through TVT
                #Iterate through bibcodes assigned to this TVT
                for curr_bibcode in dict_bibcodes_perTVT[repr_key][ind_TVT]:
                    #Throw error if used bibcode
                    if (curr_bibcode in used_bibcodes):
                        raise ValueError("Err: Used bibcode {0} in {1}:TVT={2}"
                                        .format(curr_bibcode, repr_key,ind_TVT))
                    #
                    #Record assigned TVT and representative classif for bibcode
                    dict_info[curr_bibcode] = {"repr_class":repr_key,
                                        "folder_TVT":name_folderTVT[ind_TVT],
                                        "storage":{}}
                    #
                    #Iterate through texts associated with this bibcode
                    for curr_textid in dict_bibcode_textids[curr_bibcode]:
                        curr_data = dataset[curr_textid] #Data for this text
                        act_key = curr_data["class"] #Actual class of text id
                        curr_filename = "{0}_{1}_{2}".format("text", act_key,
                                                            curr_textid)
                        #
                        if (curr_data["id"] is not None): #Add id
                            curr_filename += "_{0}".format(curr_data["id"])
                        #
                        #Throw error if used text id
                        if (curr_textid in used_textids):
                            raise ValueError("Err: Used text id: {0}"
                                            .format(curr_textid))
                        #
                        #Throw error if not unique filename
                        if (curr_filename in saved_filenames):
                            raise ValueError("Err: Non-unique filename: {0}"
                                            .format(curr_filename))
                        #
                        #Write this text to new file
                        curr_filebase = os.path.join(dir_model,
                                            name_folderTVT[ind_TVT],
                                            act_key)#TVT path; use actual class!
                        self._write_text(text=curr_data["text"],
                                        filepath=os.path.join(curr_filebase,
                                                        (curr_filename+".txt")))
                        #
                        #Store record of this storage in info dictionary
                        dict_info[curr_bibcode]["storage"][curr_textid
                                ] = {"filename":curr_filename, "class":act_key,
                                    "mission":dataset[curr_textid]["mission"]}
                        #
                        #Increment count of texts in this classif. and TVT dir.
                        all_texts_partitioned_counts[name_folderTVT[ind_TVT]][
                                                    act_key] += 1
                        used_textids[i_track] = curr_textid #Check off text id
                        saved_filenames[i_track] = curr_filename #Check off file
                        i_track += 1 #Increment place in list of filenames
                    #
                #
                #Store and increment count of used bibcodes
                used_bibcodes[i_bibcode] = curr_bibcode
                i_bibcode += 1
                #
            #
        #
        #Throw an error if any bibcodes not accounted for in partitioning
        tmp_check = len(dict_info)
        if (tmp_check != len(unique_bibcodes)):
            raise ValueError("Err: Count of bibcodes does not match original"
                            +" bibcode count.\n{0} vs. {1}"
                            .format(tmp_check, len(unique_bibcodes)))
        #
        #Throw an error if any text ids not accounted for in partitioning
        tmp_check = sum([sum(all_texts_partitioned_counts[key].values())
                        for key in all_texts_partitioned_counts])
        if (tmp_check != num_textids):
            raise ValueError("Err: Count of text ids does not match original"
                            +" text id count.\n{0}\nvs. {1}\n{2}"
                            .format(tmp_check, num_textids,
                                    all_texts_partitioned_counts))
        #
        #Throw an error if not enough filenames saved
        if (None in saved_filenames):
            tmp_check = len([item for item in saved_filenames
                            if (item is not None)])
            raise ValueError("Err: Only subset of filenames saved: {0} vs {1}."
                            .format(tmp_check, num_textids))
        #
        #Print some notes
        if do_verbose:
            print("\nFiles saved to new TVT directories.")
            print("Final partition of texts across classes and TVT dirs.:\n{0}"
                    .format(all_texts_partitioned_counts))
        #

        ##Save the dictionary of TVT bibcode partitioning to its own file
        #tmp_filesave = filepath_dictinfo
        np.save(filepath_dictinfo, dict_info)
        #Print some notes
        if do_verbose:
            print("Dictionary of TVT bibcode partitioning info saved at: {0}."
                    .format(filepath_dictinfo))
        #

        ##Verify that count of saved .txt files adds up to original data count
        for curr_key in unique_classes:
            #Count items in this class across TVT directories
            curr_count = sum([len([item2 for item2 in
                        os.listdir(os.path.join(dir_model, item1, curr_key))
                        if (item2.endswith(".txt"))])
                        for item1 in name_folderTVT])
            #
            #Verify count
            if (curr_count != count_classes[curr_key]):
                raise ValueError("Err: Unequal class count in {0}!\n{1} vs {2}"
                        .format(curr_key, curr_count, count_classes[curr_key]))
            #
        #

        ##Exit the method
        if do_verbose:
            print("\nRun of generate_directory_TVT() complete.\n---\n")
        #
        return
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
        use_these_modes = list(set([which_mode, "none"]))
        grammar = Grammar(text, keyword_obj=keyword_obj,
                            do_check_truematch=do_check_truematch,
                            do_verbose=do_verbose, buffer=buffer)
        grammar.run_modifications(which_modes=use_these_modes)
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
    def __init__(self, filepath_model=None, fileloc_ML=None, do_verbose=False):
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
        dir_train = os.path.join(dir_model, config.folders_TVT["train"])
        dir_validation = os.path.join(dir_model, config.folders_TVT["validate"])
        dir_test = os.path.join(dir_model, config.folders_TVT["test"])
        #
        savename_ML = (config.tfoutput_prefix + name_model)
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
        ml_handle_preprocessor =config.dict_ml_model_preprocessors[ml_model_key]
        ml_preprocessor = tfhub.KerasLayer(ml_handle_preprocessor)
        if do_verbose:
            print("Loaded ML preprocessor: {0}".format(ml_handle_preprocessor))
        #
        ml_handle_encoder = config.dict_ml_model_encoders[ml_model_key]
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
    def classify_text(self, text, do_check_truematch=None, keyword_obj=None, do_verbose=False, forest=None):
        """
        Method: classify_text
        Purpose: Classify given text using stored machine learning (ML) model.
        Arguments:
          - forest [None (default=None)]:
            - Unused - merely an empty placeholder for uniformity of classify_text across Classifier_* classes. Keep as None.
          - keyword_objs [list of Keyword instances, or None (default=None)]:
            - List of Keyword instances for which previously constructed paragraphs will be extracted.
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
        text_clean = self._streamline_phrase(text, do_streamline_etal=False)

        #Fetch and use stored model
        model = self._get_info("model")
        probs = np.asarray(model.predict([text_clean]))[0] #Uncertainties
        dict_uncertainty = {list_classes[ii]:probs[ii]
                            for ii in range(0, len(list_classes))}#Dict. version
        #

        #Determine best verdict
        max_ind = np.argmax(probs)
        max_verdict = list_classes[max_ind]

        #Generate dictionary of results
        dict_results = {"verdict":max_verdict, "scores_comb":None,
                            "scores_indiv":None, "uncertainty":dict_uncertainty}
        #

        #Print some notes
        if do_verbose:
            print("\nMethod classify_text for ML classifier complete!")
            print("Max verdict: {0}\n".format(max_verdict))
            print("Uncertainties: {0}\n".format(dict_uncertainty))
        #

        #Return dictionary of results
        return dict_results
    #
#


##Class: Classifier_Rules
class matrix_Classifier_Rules(_Classifier):
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
        self._store_info(do_verbose, "do_verbose")
        self._store_info(do_verbose_deep, "do_verbose_deep")
        if (which_classifs is None):
            which_classifs = config.list_default_verdicts_decisiontree
        self._store_info(which_classifs, "class_names")

        ##Assemble the trained scoring matrix
        quick_obj = Keyword(
                    keywords=["Hubble", "Hubble Telescope", "Hubble Space Telescope"],
                    acronyms=["HST", "HT"],
                    banned_overlap=["Hubble Legacy Archive"])
        scoring_matrix = self._assemble_scoring_matrix(keyword_obj=quick_obj, do_check_truematch=True)
        #
        self._store_info(scoring_matrix, "scoring_matrix")

        ##Print some notes
        if do_verbose:
            print("> Initialized instance of Classifier_Rules class.")
            print("Internal decision tree has been assembled.")
            print("NOTE: Decision tree probabilities:\n{0}\n"
                    .format(scoring_matrix))
        #

        ##Nothing to see here
        return
    #

    ##Method: _apply_scoring_matrix
    ##Purpose: Apply a scoring matrix to a 'nest' dictionary for some text
    def _apply_scoring_matrix(self, scoring_matrix, rule):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        which_classifs = self._get_info("class_names")
        keys_main = config.nest_keys_main
        keys_matter = [item for item in keys_main if (item.endswith("matter"))]
        bool_keyword = "is_keyword"
        prefix = "prob_"
        #
        rule_mod = rule.copy()
        for curr_key in keys_main: #Encapsulate any single values into lists
            if (isinstance(rule[curr_key], str) or (rule[curr_key] is None)):
                rule_mod[curr_key] = [rule[curr_key]]
            else:
                rule_mod[curr_key] = rule[curr_key]
        #
        #Print some notes
        if do_verbose:
            print("Applying decision tree to the following rule:")
            print(rule_mod)
        #

        ##Ignore this rule if no keywords within
        if (not any([bool_keyword in rule_mod[key] for key in keys_matter])):
            #Print some notes
            if do_verbose:
                print("No keywords in this rule. Skipping.")
            return None
        #

        ##Convert the rule into a matrix
        matr_rule = self._convert_rules_into_matrix(rules=[rule_mod])

        ##Apply the scoring matrix and order the scores
        dict_scores_raw = {key:np.dot(matr_rule, scoring_matrix[key])
                            for key in which_classifs}
        #Shift scores to be nonzero
        minval = min(dict_scores_raw.values())
        if (minval < 0):
            dict_scores_raw = {key:(dict_scores_raw[key]+np.abs(minval))
                                for key in which_classifs}
        #Normalize the scores
        totval = sum(dict_scores_raw.values())
        dict_scores = {key:
                        ((dict_scores_raw[key]/totval) if (totval > 0) else 0)
                        for key in which_classifs}
        #

        ##Return the final scores
        if do_verbose:
            print("Final scores computed for rule:\n{0}\n\n{1}" #\n\n{2}"
                    .format(rule_mod, dict_scores)) #, best_branch))
        #
        return {"scores":dict_scores, "components":rule_mod,
                "matr_rule":matr_rule}
    #

    ##Method: _assemble_scoring_matrix
    ##Purpose: Assemble matrix to use for scoring rules
    def _assemble_scoring_matrix(self, keyword_obj, do_check_truematch, buffer=0, which_mode="none"):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        do_verbose_deep = self._get_info("do_verbose_deep")
        which_classifs = self._get_info("class_names")
        dict_all = config.dict_tree_possible_values
        prefix = "score_"
        #Print some notes
        if do_verbose:
            print("Assembling scoring matrix...")
        #

        #Build in-place dictionary of texts just for now
        """
        dict_texts = [
        #Future tense
            {"text":"We will use HST data in a future paper.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"They will use HST data in a future paper.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Authors et al. will use HST data in a future paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
        #Present tense
            {"text":"We use HST data.", "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"We used HST data.", "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"They use HST data.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"They used HST data.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"We simulate HST data.", "class":None,
            "scores":{"science":0, "data_influenced":1, "mention":0}},
            {"text":"They simulate HST data.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Authors et al. simulate HST data.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"The HST data was simulated by Authors et al.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0.8, "mention":0.2}},
            {"text":"Authors et al. use HST data.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"HST data is cool.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Our work analyzes the HST data of Authors et al.",
                    "class":None,
            "scores":{"science":0, "data_influenced":1, "mention":0}},
        #Rules with figures
            {"text":"Figure 3 is of HST data.", "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"Figure 3 plots the HST data.", "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"The trends in the HST data are shown in Figure 3.",
                    "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"The data observed by HST are plotted in Figure 3.",
                    "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"The data plotted in Figure 3 was observed by HST.",
                    "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"Figure 3 plots the HST data from Authors etal.",
                    "class":None,
            "scores":{"science":0, "data_influenced":1, "mention":0}},
        #More rules
            {"text":"The trends are analyzed in our HST data.", "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"Our analysis is of HST data.", "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"We know that HST is useful.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"HST is known for its beautiful images.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"This telescope is HST.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"This image was from HST.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"This image complimented HST.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            #
            {"text":"The trends in the HST data are plotted in Figure 3.",
                    "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"Authors et al. analyzed the HST data.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Authors et al. plotted the HST data.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Authors et al. simulated the HST data.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Authors et al. simulated, plotted, and analyzed the HST data.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"They analyzed the HST data.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"They plotted the HST data.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"They simulated the HST data.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"They simulated and plotted and analyzed the HST data.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            #
            {"text":"Trends are analyzed in the HST paper by Athors et al..",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Trends were analyzed in the HST paper by Athors et al..",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Trends are presented in the HST paper by Athors et al..",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Trends were presented in the HST paper by Athors et al..",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Trends are analyzed in their HST paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Trends were analyzed in their HST paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Trends are presented in their HST paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Trends were presented in their HST paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            #
            {"text":"Our bright HST images shows the power of HST.",
                    "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"Our observations of HST data are analyzed.",
                    "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"We already presented that HST data in Authors et al..",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"HST is neat.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Our analysis is for HST data.",
                    "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"Stars were plotted from the HST data.",
                    "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            #
            {"text":"HST will revolutionize the industry.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"We will observe HST data in a future paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"We will present HST data in a future paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"They will observe HST data in a future paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"They will present HST data in a future paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Authors et al. will observe HST data in a future paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Authors et al. will present HST data in a future paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            #
            {"text":"Authors et al. will present HST data in a future paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"HST orbits around.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Our future HST observations will be compared in our future HST paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Our future HST observations will be presented in our future HST paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Our future HST observations will be analyzed in our future HST paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            #
        #
        ]
        """

        #Build in-place dictionary of set blanket-rules for now
        dict_blanketrules = {}
        itrack = 0
        #
        if True:
            #Blanket 'know' ruling
            itrack += 1
            dict_blanketrules[str(itrack)] = {
                "subjectmatter":"is_any",
                "objectmatter":"is_any",
                "verbclass":{"know"},
                "verbtypes":"is_any",
                (prefix+"science"):0.0,
                (prefix+"data_influenced"):0.0,
                (prefix+"mention"):1.0}
            #
            #Blanket future-verb ruling
            itrack += 1
            dict_blanketrules[str(itrack)] = {
                "subjectmatter":"is_any",
                "objectmatter":"is_any",
                "verbclass":"is_any",
                "verbtypes":["FUTURE"],
                (prefix+"science"):0.0,
                (prefix+"data_influenced"):0.0,
                (prefix+"mention"):1.0}
            #
            #'be/has' verb
            #HST is/has/was/had an observatory.
            itrack += 1
            dict_blanketrules[str(itrack)] = {
                "subjectmatter":["is_keyword"],
                "objectmatter":[],
                "verbclass":{"be", "has"},
                "verbtypes":["PRESENT"],
                (prefix+"science"):0.0,
                (prefix+"data_influenced"):0.0,
                (prefix+"mention"):1.0}
            itrack += 1
            dict_blanketrules[str(itrack)] = {
                "subjectmatter":[],
                "objectmatter":["is_keyword"],
                "verbclass":{"be", "has"},
                "verbtypes":["PRESENT"],
                (prefix+"science"):0.0,
                (prefix+"data_influenced"):0.0,
                (prefix+"mention"):1.0}
            #Our images are/were from HST / We have/had HST data.
            itrack += 1
            dict_blanketrules[str(itrack)] = {
                "subjectmatter":["is_pron_1st"],
                "objectmatter":["is_keyword"],
                "verbclass":{"be", "has"},
                "verbtypes":["PRESENT", "PAST"],
                (prefix+"science"):1.0,
                (prefix+"data_influenced"):0.0,
                (prefix+"mention"):0.0}
            #
            #Their images are/were from HST / They have/had HST data.
            itrack += 1
            dict_blanketrules[str(itrack)] = {
                "subjectmatter":["is_pron_3rd"],
                "objectmatter":["is_keyword"],
                "verbclass":{"be", "has"},
                "verbtypes":["PRESENT", "PAST"],
                (prefix+"science"):0.0,
                (prefix+"data_influenced"):0.0,
                (prefix+"mention"):1.0}
            #
            #Authors' images are/were from HST / Authors have/had HST data.
            itrack += 1
            dict_blanketrules[str(itrack)] = {
                "subjectmatter":["is_etal"],
                "objectmatter":["is_keyword"],
                "verbclass":{"be", "has"},
                "verbtypes":["PRESENT", "PAST"],
                (prefix+"science"):0.0,
                (prefix+"data_influenced"):0.0,
                (prefix+"mention"):1.0}
            #
            #'plot' verb
            #The trend is shown in the HST data.
            itrack += 1
            dict_blanketrules[str(itrack)] = {
                "subjectmatter":[],
                "objectmatter":["is_keyword"],
                "verbclass":{"plot"},
                "verbtypes":["PRESENT", "PAST"],
                (prefix+"science"):1.0,
                (prefix+"data_influenced"):0.0,
                (prefix+"mention"):0.0}
            #'science/plot' verb
            #We use/plot HST data.
            itrack += 1
            dict_blanketrules[str(itrack)] = {
                "subjectmatter":["is_pron_1st"],
                "objectmatter":["is_keyword"],
                "verbclass":{"science", "plot"},
                "verbtypes":["PRESENT", "PAST"],
                (prefix+"science"):1.0,
                (prefix+"data_influenced"):0.0,
                (prefix+"mention"):0.0}
            #They use/plot HST data.
            itrack += 1
            dict_blanketrules[str(itrack)] = {
                "subjectmatter":["is_pron_3rd"],
                "objectmatter":["is_keyword"],
                "verbclass":{"science", "plot"},
                "verbtypes":["PRESENT", "PAST"],
                (prefix+"science"):0.0,
                (prefix+"data_influenced"):0.0,
                (prefix+"mention"):1.0}
            #Authors use/plot HST data.
            itrack += 1
            dict_blanketrules[str(itrack)] = {
                "subjectmatter":["is_etal"],
                "objectmatter":["is_keyword"],
                "verbclass":{"science", "plot"},
                "verbtypes":["PRESENT", "PAST"],
                (prefix+"science"):0.0,
                (prefix+"data_influenced"):0.0,
                (prefix+"mention"):1.0}
            #
            #'datainfl.' verb
            #We simulate HST data.
            itrack += 1
            dict_blanketrules[str(itrack)] = {
                "subjectmatter":["is_pron_1st"],
                "objectmatter":["is_keyword"],
                "verbclass":{"data_influenced"},
                "verbtypes":["PRESENT", "PAST"],
                (prefix+"science"):0.0,
                (prefix+"data_influenced"):1.0,
                (prefix+"mention"):0.0}
            #They simulate HST data.
            itrack += 1
            dict_blanketrules[str(itrack)] = {
                "subjectmatter":["is_pron_3rd"],
                "objectmatter":["is_keyword"],
                "verbclass":{"data_influenced"},
                "verbtypes":["PRESENT", "PAST"],
                (prefix+"science"):0.0,
                (prefix+"data_influenced"):0.0,
                (prefix+"mention"):1.0}
            #Authors simulate HST data.
            itrack += 1
            dict_blanketrules[str(itrack)] = {
                "subjectmatter":["is_etal"],
                "objectmatter":["is_keyword"],
                "verbclass":{"data_influenced"},
                "verbtypes":["PRESENT", "PAST"],
                (prefix+"science"):0.0,
                (prefix+"data_influenced"):0.0,
                (prefix+"mention"):1.0}
        #

        #Initialize container for input matrix of rules and classif vectors
        input_matr = []
        classif_vec = {key:[] for key in which_classifs}

        #Convert blanket-rules into matrix rows
        for curr_rulekey in dict_blanketrules:
            curr_blanketrule = dict_blanketrules[curr_rulekey]

            #Extract all rule combinations and handle 'is_any' cases
            #For subject matter
            if (curr_blanketrule["subjectmatter"] == "is_any"):
                curr_subjs = [[item] for item in dict_all["subjectmatter"]]
            else:
                curr_subjs = [curr_blanketrule["subjectmatter"]]
            #
            #For object matter
            if (curr_blanketrule["objectmatter"] == "is_any"):
                curr_objs = [[item] for item in dict_all["objectmatter"]]
            else:
                curr_objs = [curr_blanketrule["objectmatter"]]
            #
            #For verbclass
            if (curr_blanketrule["verbclass"] == "is_any"):
                curr_vclass = [[item] for item in dict_all["verbclass"]]
            else:
                curr_vclass = [[item] for item in curr_blanketrule["verbclass"]]
            #
            #For verbtype
            if (curr_blanketrule["verbtypes"] == "is_any"):
                curr_vtypes = [[item] for item in dict_all["verbtypes"]]
            elif (isinstance(curr_blanketrule["verbtypes"], list)):
                curr_vtypes = [[item] for item in curr_blanketrule["verbtypes"]]
            else:
                curr_vtypes = list(curr_blanketrule["verbtypes"])
            #

            #Get list of all combinations
            list_all = [curr_subjs, curr_objs, curr_vclass, curr_vtypes]
            combos = list(iterer.product(*list_all))

            #Store each combination as a rule
            for curr_combo in combos:
                curr_rule = {"subjectmatter":curr_combo[0],
                            "objectmatter":curr_combo[1],
                            "verbclass":curr_combo[2],
                            "verbtypes":curr_combo[3]}
                #
                #Convert rules into vector and store them
                curr_vector = self._convert_rules_into_matrix([curr_rule])
                input_matr.append(curr_vector)
                #
                #Convert actual classifs into answer vector and store them
                for curr_classkey in which_classifs:
                    classif_vec[curr_classkey].append(
                                        curr_blanketrule[prefix+curr_classkey])
            #
        #

        #Reformat into matrices
        input_matr = np.asarray(input_matr).astype(float)
        for curr_key in classif_vec:
            classif_vec[curr_key] = np.asarray(classif_vec[curr_key]
                                                ).astype(float).T
        #

        #Solve for scoring matrix (least squares)
        scoring_matrix = {key:np.linalg.lstsq(input_matr, classif_vec[key])[0]
                            for key in which_classifs}

        #Return the results
        if do_verbose:
            print("Scoring matrix has been assembled:\n{0}\n-\n"
                    .format(scoring_matrix))
        return scoring_matrix
    #

    ##Method: _assemble_scoring_matrix
    ##Purpose: Assemble matrix to use for scoring rules
    def x_assemble_scoring_matrix(self, keyword_obj, do_check_truematch, buffer=0, which_mode="none"):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        do_verbose_deep = self._get_info("do_verbose_deep")
        which_classifs = self._get_info("class_names")
        #Print some notes
        if do_verbose:
            print("Assembling scoring matrix...")
        #

        #Build in-place dictionary of texts just for now
        """
        dict_texts = [
        #Future tense
            {"text":"We will use HST data in a future paper.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"They will use HST data in a future paper.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Authors et al. will use HST data in a future paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
        #Present tense
            {"text":"We use HST data.", "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"We used HST data.", "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"They use HST data.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"They used HST data.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"We simulate HST data.", "class":None,
            "scores":{"science":0, "data_influenced":1, "mention":0}},
            {"text":"They simulate HST data.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Authors et al. simulate HST data.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"The HST data was simulated by Authors et al.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0.8, "mention":0.2}},
            {"text":"Authors et al. use HST data.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"HST data is cool.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Our work analyzes the HST data of Authors et al.",
                    "class":None,
            "scores":{"science":0, "data_influenced":1, "mention":0}},
        #Rules with figures
            {"text":"Figure 3 is of HST data.", "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"Figure 3 plots the HST data.", "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"The trends in the HST data are shown in Figure 3.",
                    "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"The data observed by HST are plotted in Figure 3.",
                    "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"The data plotted in Figure 3 was observed by HST.",
                    "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"Figure 3 plots the HST data from Authors etal.",
                    "class":None,
            "scores":{"science":0, "data_influenced":1, "mention":0}},
        #More rules
            {"text":"The trends are analyzed in our HST data.", "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"Our analysis is of HST data.", "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"We know that HST is useful.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"HST is known for its beautiful images.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"This telescope is HST.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"This image was from HST.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"This image complimented HST.", "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            #
            {"text":"The trends in the HST data are plotted in Figure 3.",
                    "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"Authors et al. analyzed the HST data.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Authors et al. plotted the HST data.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Authors et al. simulated the HST data.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Authors et al. simulated, plotted, and analyzed the HST data.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"They analyzed the HST data.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"They plotted the HST data.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"They simulated the HST data.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"They simulated and plotted and analyzed the HST data.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            #
            {"text":"Trends are analyzed in the HST paper by Athors et al..",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Trends were analyzed in the HST paper by Athors et al..",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Trends are presented in the HST paper by Athors et al..",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Trends were presented in the HST paper by Athors et al..",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Trends are analyzed in their HST paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Trends were analyzed in their HST paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Trends are presented in their HST paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Trends were presented in their HST paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            #
            {"text":"Our bright HST images shows the power of HST.",
                    "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"Our observations of HST data are analyzed.",
                    "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"We already presented that HST data in Authors et al..",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"HST is neat.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Our analysis is for HST data.",
                    "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            {"text":"Stars were plotted from the HST data.",
                    "class":None,
            "scores":{"science":1, "data_influenced":0, "mention":0}},
            #
            {"text":"HST will revolutionize the industry.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"We will observe HST data in a future paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"We will present HST data in a future paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"They will observe HST data in a future paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"They will present HST data in a future paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Authors et al. will observe HST data in a future paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Authors et al. will present HST data in a future paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            #
            {"text":"Authors et al. will present HST data in a future paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"HST orbits around.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Our future HST observations will be compared in our future HST paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Our future HST observations will be presented in our future HST paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            {"text":"Our future HST observations will be analyzed in our future HST paper.",
                    "class":None,
            "scores":{"science":0, "data_influenced":0, "mention":1}},
            #
        #
        ]
        """

        #Build in-place dictionary of set rules for now
        itrack = 0
        if True:
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":"is_any",
                "objectmatter":"is_any",
                "verbclass":{"know"},
                "verbtypes":"is_any",
                "score_science":0.0,
                "score_data_influenced":0.0,
                "score_mention":1.0}
        #

        #Initialize container for input matrix of rules and classif vectors
        input_matr = []
        classif_vec = {key:[] for key in which_classifs}

        #Convert texts into rules
        for ii in range(0, len(dict_texts)):
            #Extract current values
            text = dict_texts[ii]["text"]
            if ("scores" in dict_texts[ii]):
                act_classifs = dict_texts[ii]["scores"]
            else:
                act_classifs = {key:0 for key in which_classifs}
                if (dict_texts[ii]["class"] not in act_classifs):
                    raise ValueError("Err: Bad class pass: {0}\n{1}"
                            .format(dict_texts[ii]["class"], which_classifs))
                act_classifs[dict_texts[ii]["class"]] = 1
            #

            #Grammar processing
            forest = self._process_text(text=text,
                        do_check_truematch=do_check_truematch,
                        keyword_obj=keyword_obj, do_verbose=do_verbose_deep,
                        buffer=buffer, which_mode=which_mode)["forest"]
            #

            #Iterate through and score clauses
            list_scores = [[] for jj in range(0, len(forest))]
            list_components = [[] for jj in range(0, len(forest))]
            for jj in range(0, len(forest)):
                for kk in range(0, len(forest[jj])):
                    curr_info = forest[jj][kk]
                    #Convert current clause into rules
                    curr_rules_raw = [item for key in curr_info["clauses"]["text"]
                            for item in
                            self._convert_clause_into_rule(
                            clause_text=curr_info["clauses"]["text"][key],
                            clause_ids=curr_info["clauses"]["ids"][key],
                            flags_nounchunks=curr_info["flags"],
                            ids_nounchunks=curr_info["ids_nounchunk"])]
                    #
                    #Merge rules
                    curr_rules = [self._merge_rules(rules=curr_rules_raw)]
                    #
                    #Convert rules into vector and store them
                    curr_vector = self._convert_rules_into_matrix(curr_rules)
                    input_matr.append(curr_vector)
                    #
                    #Convert actual classifs into answer vector and store them
                    for curr_key in which_classifs:
                        classif_vec[curr_key].append(act_classifs[curr_key])
                #
            #
        #

        #Reformat into matrices
        input_matr = np.asarray(input_matr).astype(float)
        for curr_key in classif_vec:
            classif_vec[curr_key] = np.asarray(classif_vec[curr_key]
                                                ).astype(float).T
        #

        #Solve for scoring matrix (least squares)
        scoring_matrix = {key:np.linalg.lstsq(input_matr, classif_vec[key])[0]
                            for key in which_classifs}

        #Return the results
        if do_verbose:
            print("Scoring matrix has been assembled:\n{0}\n-\n"
                    .format(scoring_matrix))
        return scoring_matrix
    #

    ##Method: _assemble_decision_tree
    ##Purpose: Assemble base of decision tree, with probabilities, that can be read from/expanded as full decision tree
    def x_assemble_decision_tree(self):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        which_classifs = self._get_info("class_names")
        dict_possible_values = config.dict_tree_possible_values
        #dict_valid_combos = config.dict_tree_valid_value_combinations
        keys_matter = config.nest_keys_matter
        key_verbtype = config.nest_key_verbtype
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
                "prob_science":0.9,
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
                "prob_data_influenced":0.4,
                "prob_mention":0.5}
            #
            #’The data is from OBJ by Authorsetal.’
            #itrack += 1
            #dict_examples_base[str(itrack)] = {
            #    "subjectmatter":tuple([]),
            #    "objectmatter":tuple(["is_keyword", "is_etal"]),
            #    "verbclass":{"be"},
            #    "verbtypes":{"PRESENT"},
            #    "prob_science":0.0,
            #    "prob_data_influenced":0.8,
            #    "prob_mention":0.6}
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
                "prob_science":0.9,
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
                "prob_science":0.9,
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
                "prob_mention":0.0}
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
                "prob_mention":0.0}
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
                "prob_data_influenced":1.0,
                "prob_mention":0.0}
            #
            #’We simulate/simulated the OBJ data of Authorsetal.’
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword", "is_etal"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":1.0,
                "prob_mention":0.0}
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
            #is_key; is_fig, !is_etal, !is_they
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":["is_term_fig", "!is_etal", "!is_pron_3rd"],
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.9,
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
                "prob_science":0.9,
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
                "prob_science":0.9,
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
            #<Keyword figure be/has/plot/science/datainfl. by etal and pron_3rd>
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword", "is_term_fig"]),
                "objectmatter":["is_etal"],
                "verbclass":"is_any",
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.0,
                "prob_data_influenced":1.0,
                "prob_mention":0.0}
            #
            #<Keyword figure be/has/plot/science/datainfl. by etal and pron_3rd>
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword", "is_term_fig"]),
                "objectmatter":["is_pron_3rd"],
                "verbclass":"is_any",
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.0,
                "prob_data_influenced":0.9,
                "prob_mention":0.1}
            #
            #<Keyword figure be/has/plot/science/datainfl. by etal and pron_3rd>
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword", "is_term_fig"]),
                "objectmatter":["is_etal", "is_pron_3rd"],
                "verbclass":"is_any",
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.0,
                "prob_data_influenced":1.0,
                "prob_mention":0.1}
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

            ##For passive example
            #For general (not is_any, not set) case
            if ((curr_ex[key_verbtype] == "is_any")):
                #                    or (curr_ex[key_verbtype] is None)):
                pass #No passive counterpart necessary for is_any case
            #For set, add example+passive for each entry in set
            #elif isinstance(curr_ex[key_verbtype], set):
                #Iterate through set entries
            #    for curr_val in curr_ex[key_verbtype]:
                    #Passive with flipped subj-obj
                    #Extract all parameters and their values
            #        new_ex = {key:curr_ex[key] for key in all_params
            #                if (key not in (keys_matter+[key_verbtype]))}
                    #Add in passive term for verbtypes
            #        tmp_vals = [curr_val, "PASSIVE"]
            #        new_ex[key_verbtype] = tmp_vals
                    #Flip the subject and object terms
            #        new_ex["subjectmatter"] = curr_ex["objectmatter"]
            #        new_ex["objectmatter"] = curr_ex["subjectmatter"]
                    #Normalize and store probabilities for target classifs
            #        new_ex.update(curr_probs)
                    #Store this example
            #        itrack += 1
            #        decision_tree[itrack] = new_ex
            else:
                #Extract all parameters and their values
                new_ex = {key:curr_ex[key] for key in all_params
                        if (key not in (keys_matter+[key_verbtype]))}
                #Add in passive term for verbtypes
                tmp_vals = list(curr_ex[key_verbtype]) #+ ["PASSIVE"]
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
    def _categorize_verb(self, verb):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        list_category_names = config.list_category_names
        list_category_synsets = config.list_category_synsets
        list_category_threses = config.list_category_threses
        max_hyp = config.max_num_hypernyms
        #root_hypernyms = wordnet.synsets(verb, pos=wordnet.VERB)
        if (max_hyp is None):
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

        ##Handle specialty verbs
        #For 'be' verbs
        if any([(roothyp in config.synsets_verbs_be)
                    for roothyp in root_hypernyms]):
            return "be"
        #For 'has' verbs
        elif any([(roothyp in config.synsets_verbs_has)
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
            if do_verbose:
                print("No categories fit verb: {0}, {1}\n"
                                .format(verb, score_fins))
            return None
        #

        ##Throw an error if this verb gives very similar top scores
        thres = config.thres_category_fracdiff
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
    def _classify_statements(self, forest, do_verbose=None):
        #Extract global variables
        if do_verbose is not None: #Override do_verbose if specified for now
            self._store_info(do_verbose, "do_verbose")
        #Load the fixed decision tree
        scoring_matrix = self._get_info("scoring_matrix")
        #Print some notes
        if do_verbose:
            print("\n> Running _classify_statements.")
            #print("Forest word-trees:")
            #for key1 in forest:
            #    for key2 in forest[key1]:
            #        print("Key1, Key2: {0}, {1}".format(key1, key2))
            #        print("{0}".format(forest[key1][key2
            #                            ]["struct_words_updated"].keys()))
            #        print("{0}\n-".format(forest[key1][key2
            #                            ]["struct_words_updated"]))
        #

        #Iterate through and score clauses
        list_scores = [[] for ii in range(0, len(forest))]
        list_components = [[] for ii in range(0, len(forest))]
        for ii in range(0, len(forest)):
            for jj in range(0, len(forest[ii])):
                curr_info = forest[ii][jj]
                #Convert current clause into rules
                curr_rules_raw = [item for key in curr_info["clauses"]["text"]
                            for item in
                            self._convert_clause_into_rule(
                            clause_text=curr_info["clauses"]["text"][key],
                            clause_ids=curr_info["clauses"]["ids"][key],
                            flags_nounchunks=curr_info["flags"],
                            ids_nounchunks=curr_info["ids_nounchunk"])]
                #
                #Merge rules
                curr_rules = [self._merge_rules(rules=curr_rules_raw)]
                #
                #Fetch and store score of each rule
                tmp_res = [self._apply_scoring_matrix(rule=item,
                                                scoring_matrix=scoring_matrix)
                                for item in curr_rules]
                list_scores[ii] += [item["scores"] for item in tmp_res
                                    if (item is not None)]
                list_components[ii] += [item["components"] for item in tmp_res
                                    if (item is not None)]
        #

        #Combine scores across clauses
        #comb_score = self._combine_scores(list_scores)
        comb_score = [item2 for item1 in list_scores for item2 in item1]

        #Convert final score into verdict and other information
        results = self._convert_score_to_verdict(comb_score)
        results["indiv_scores"] = list_scores
        results["indiv_components"] = list_components

        ##Return the dictionary containing verdict, etc. for these statements
        return results
    #

    ##Method: classify_text
    ##Purpose: Classify full text based on its statements (rule approach)
    def classify_text(self, keyword_obj, do_check_truematch, which_mode=None, forest=None, text=None, buffer=0, do_verbose=None):
        #Fetch global variables
        if do_verbose is None:
            do_verbose = self._get_info("do_verbose")
        do_verbose_deep = self._get_info("do_verbose_deep")
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
        dict_results = self._classify_statements(forest, do_verbose=do_verbose)

        #Print some notes
        if do_verbose:
            print("Verdicts complete.")
            print("Verdict dictionary:\n{0}".format(dict_results))
            print("---")

        #Return final verdicts
        return dict_results #dict_verdicts
    #

    ##Method: _convert_scorestoverdict
    ##Purpose: Convert set of decision tree scores into single verdict
    def x_convert_score_to_verdict(self, dict_scores_indiv, max_diff_thres=0.10, max_diff_count=3, max_diff_verdicts=["science"]):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        #Print some notes
        if do_verbose:
            print("\n> Running _convert_scorestoverdict.")
            print("Individual components and score sets:")
            for ii in range(0, len(dict_scores_indiv)):
                print("{0}\n-".format(#components[ii],
                                            dict_scores_indiv[ii]))
            #
        #

        ##Return empty verdict if empty scores
        #For completely empty scores
        if len(dict_scores_indiv) == 0:
            tmp_res = config.dictverdict_error.copy()
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
                                    ) #Check if rel. max. key by some thres.
                #
                else:
                    tmp_compare = False
                #

                #Increment count of sentences with max-valued verdict
                if tmp_compare:
                    dict_results[curr_key]["count_max"] += 1
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

        #Gather final scores into set of error
        dict_error = {key:dict_results[key]["score_tot_norm"]
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

        ##Establish uncertainties from scores
        dict_uncertainties = {all_keys[ii]:list_scores_comb[ii]
                                for ii in range(0, len(all_keys))}
        #

        ##Determine best verdict and associated probabilistic error
        is_found = False
        #For max sentence count:
        if (not is_found) and (max_diff_thres is not None):
            #Check allowed keys that fit max condition
            for curr_key in max_diff_verdicts:
                if (dict_results[curr_key]["count_max"] >= max_diff_count):
                    is_found = True
                    max_score = 1 #dict_results[curr_key]["score_tot_norm"]
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
            #Return low-prob verdict if multiple equal top probabilities
            if (list_scores_comb.count(max_score) > 1):
                tmp_res = config.dictverdict_lowprob.copy()
                tmp_res["scores_indiv"] = dict_scores_indiv
                tmp_res["uncertainty"] = dict_uncertainties
                #tmp_res["components"] = components
                #Print some notes
                if do_verbose:
                    print("-Multiple top prob. scores.")
                    print("Returning low-prob verdict:\n{0}".format(tmp_res))
                #
                return tmp_res
            #
        #

        ##Establish uncertainty for max verdict from scores
        dict_uncertainties[max_verdict] = max_score
        #

        ##Assemble and return final verdict
        fin_res = {"verdict":max_verdict, "scores_indiv":dict_scores_indiv,
                "uncertainty":dict_uncertainties, #"components":components,
                "norm_error":dict_error}
        #
        #Print some notes
        if do_verbose:
            print("-Returning final verdict dictionary:\n{0}".format(fin_res))
        #
        return fin_res
    #

    ##Method: _convert_rules_into_matrix
    ##Purpose: Convert list of rules into single matrix
    def _convert_rules_into_matrix(self, rules):
        #Establish all rows of scoring matrices (sorted order for consistency)
        dict_all = config.dict_tree_possible_values
        order_keys = sorted(list(dict_all.keys()))
        order_len = [len(dict_all[key]) for key in order_keys] #Value count

        #Initialize holder of rule-vector
        vector = np.asarray([None]*sum(order_len))

        #Build binary vector of whether or not values are in this rule
        i_track = 0
        for ii in range(0, len(order_keys)):
            #Extract values for current rule key
            curr_key = order_keys[ii]
            curr_len = order_len[ii]
            curr_allvalues = sorted(dict_all[curr_key]) #Sorted order
            #
            #Set boolean vector of which key's values present in each rule
            curr_binaries_raw = [[(1 if val in item[curr_key] else 0)
                                for val in curr_allvalues] for item in rules]
            #
            #Sum the binaries across the rules and store them
            curr_binaries = np.sum(curr_binaries_raw, axis=0)
            vector[i_track:(i_track+curr_len)] = curr_binaries #Store binaries
            #
            #Increment tracker for placement in vector
            i_track += curr_len
        #

        #Ensure vector completely filled in
        if (None in vector):
            raise ValueError("Err: Parsing problem, Nones in vector.\n{0}\n{1}"
                            .format(vector, rules))
        #

        #Return vector
        return vector
    #

    ##Method: _convert_scorestoverdict
    ##Purpose: Convert set of decision tree scores into single verdict
    def _convert_score_to_verdict(self, dict_scores_indiv, max_diff_thres=0.10, max_diff_count=3, max_diff_verdicts=["science"], thres_override_acceptance=np.inf, order_override_acceptance=["science", "data_influenced"]):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        #Print some notes
        if do_verbose:
            print("\n> Running _convert_scorestoverdict.")
            print("Individual components and score sets:")
            for ii in range(0, len(dict_scores_indiv)):
                print("{0}\n-".format(#components[ii],
                                            dict_scores_indiv[ii]))
            #
        #

        ##Return empty verdict if empty scores
        #For completely empty scores
        if len(dict_scores_indiv) == 0:
            tmp_res = config.dictverdict_error.copy()
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
                                    ) #Check if rel. max. key by some thres.
                #
                else:
                    tmp_compare = False
                #

                #Increment count of sentences with max-valued verdict
                if tmp_compare:
                    dict_results[curr_key]["count_max"] += 1
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
            dict_results[curr_key]["score_tot_fin"] = tmp_score
        #

        #Override combined scores with absolutes if requested
        tmp_max_key = None
        if ((thres_override_acceptance is not None)
                            and (order_override_acceptance is not None)):
            for curr_key in order_override_acceptance:
                if any([(item[curr_key] >= thres_override_acceptance)
                                for item in dict_scores_indiv]):
                    #Override scores
                    tmp_max_key = curr_key
                    break
        #
        if (tmp_max_key is not None):
            for curr_key in all_keys:
                if (curr_key == tmp_max_key):
                    dict_results[curr_key]["score_tot_fin"] = 1.0
                else:
                    dict_results[curr_key]["score_tot_fin"] = 0.0
        #

        #Combined score storage
        list_scores_comb = [dict_results[key]["score_tot_fin"]
                            for key in all_keys]
        #

        #Gather final scores into set of error
        dict_error = {key:dict_results[key]["score_tot_norm"]
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

        ##Establish uncertainties from scores
        dict_uncertainties = {all_keys[ii]:list_scores_comb[ii]
                                for ii in range(0, len(all_keys))}
        #

        ##Determine best verdict and associated probabilistic error
        is_found = False
        #For max sentence count:
        if (not is_found) and (max_diff_thres is not None):
            #Check allowed keys that fit max condition
            for curr_key in max_diff_verdicts:
                if (dict_results[curr_key]["count_max"] >= max_diff_count):
                    is_found = True
                    max_score = 1 #dict_results[curr_key]["score_tot_norm"]
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
            #Return low-prob verdict if multiple equal top probabilities
            if (list_scores_comb.count(max_score) > 1):
                tmp_res = config.dictverdict_lowprob.copy()
                tmp_res["scores_indiv"] = dict_scores_indiv
                tmp_res["uncertainty"] = dict_uncertainties
                #tmp_res["components"] = components
                #Print some notes
                if do_verbose:
                    print("-Multiple top prob. scores.")
                    print("Returning low-prob verdict:\n{0}".format(tmp_res))
                #
                return tmp_res
            #
        #

        ##Establish uncertainty for max verdict from scores
        dict_uncertainties[max_verdict] = max_score
        #

        ##Assemble and return final verdict
        fin_res = {"verdict":max_verdict, "scores_indiv":dict_scores_indiv,
                "uncertainty":dict_uncertainties, #"components":components,
                "norm_error":dict_error}
        #
        #Print some notes
        if do_verbose:
            print("-Returning final verdict dictionary:\n{0}".format(fin_res))
        #
        return fin_res
    #

    ##Method: _find_missing_branches
    ##Purpose: Determine and return missing branches in decision tree
    def _find_missing_branches(self, do_verbose=None, cap_iter=None, print_freq=100):
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
        all_possible_values = config.dict_tree_possible_values
        solo_values = [None]
        all_subjmatters = [item for item in all_possible_values["subjectmatter"]
                            if (item is not None)]
        all_objmatters = [item for item in all_possible_values["objectmatter"]
                            if (item is not None)]
        all_verbclasses = [item for item in all_possible_values["verbclass"]
                            ] #No shallow copying!
        solo_verbtypes = ["PAST", "PRESENT", "FUTURE", "PURPOSE"]
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
            #cap_vtypes = cap_iter
        else:
            cap_subj = 2 #len(all_subjmatters)
            cap_obj = 2 #len(all_objmatters)
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
        if (cap_vtypes > 0):
            sub_allverbtypes = [(list(item_sub)+[item_solo])
                            for item_sub in sub_multiverbtypes
                            for item_solo in solo_verbtypes] #Fold in verb tense
        else:
            sub_allverbtypes = all_possible_values["verbtypes"]
        #

        #Fold in solo and required values as needed
        sets_verbclasses = all_verbclasses
        sets_subjmatters = (sub_subjmatters) # + [None])
        sets_objmatters = (sub_objmatters) # + [None])
        sets_allverbtypes = (sub_allverbtypes) # + [None])
        #
        if do_verbose:
            print("Combinations of branch parameters complete.")
            print(("Verb classes: {0}\nSubj.matter: {1}\nObj. matter: {2}"
                    +"\nVerbtypes: {3}")
                    .format(sets_verbclasses, sets_subjmatters,
                            sets_objmatters, sets_allverbtypes))
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
                                                    rule=curr_branch)
                #
                bools_is_missing[ii] = False #Branch is covered
            #
            #Record this branch as missing if invalid output
            except NotImplementedError:
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

    ##Method: _convert_clause_into_rule
    ##Purpose: Convert Grammar-made clause into rule for rule-based classifier
    def _convert_clause_into_rule(self, clause_text, clause_ids, flags_nounchunks, ids_nounchunks):
        """
        Method: _convert_clause_into_rule
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Process text into modifs using Grammar class.
        """
        #Set global variables
        do_verbose = self._get_info("do_verbose")

        #Fetch all sets of subject flags via their ids
        num_subj = len(clause_text["subjects"])
        sets_subjmatter = [] #[None]*num_subj #Container for subject flags
        for ii in range(0, num_subj): #Iterate through subjects
            id_chunk = ids_nounchunks[clause_ids["subjects"][ii][0]]
            sets_subjmatter += [key
                                    for key in flags_nounchunks[id_chunk]
                                    if ((flags_nounchunks[id_chunk][key])
                                        and (key != "is_any"))
                                    ] #Store all subj. flags for this clause
        #
        #If no subjects, set container to empty
        if (num_subj == 0):
            sets_subjmatter = []
        #
        sets_subjmatter = list(set(sets_subjmatter))

        #Fetch all sets of object flags via their ids
        tmp_list = (clause_ids["dir_objects"] + clause_ids["prep_objects"])
        num_obj = len(tmp_list)
        sets_objmatter = [] #[None]*num_obj #Container for object flags
        for ii in range(0, num_obj): #Iterate through objects
            id_chunk = ids_nounchunks[tmp_list[ii][0]]
            sets_objmatter += [key
                                    for key in flags_nounchunks[id_chunk]
                                    if ((flags_nounchunks[id_chunk][key])
                                        and (key != "is_any"))
                                    ] #Store all obj. flags for this clause
        #
        #If no objects, set container to empty
        if (num_obj == 0):
            sets_objmatter = []
        #
        sets_objmatter = list(set(sets_objmatter))

        #Set verb class
        #verbclass = self._categorize_verb(verb=clause_text["verb"])
        verbclass_raw = self._categorize_verb(verb=clause_text["verb"])
        if (verbclass_raw is None):
            verbclass = []
        elif isinstance(verbclass_raw, str):
            verbclass = [verbclass_raw]
        else:
            verbclass = verbclass_raw
        #

        #Set verb type
        #verbtype = clause_text["verbtype"]
        verbtype_raw = clause_text["verbtype"]
        if (verbtype_raw is None):
            verbtype = []
        elif isinstance(verbtype_raw, str):
            verbtype = [verbtype_raw]
        else:
            verbtype = verbtype_raw
        #

        #Set rules from combinations of subj. and obj.matter
        rules = [{"subjectmatter":sets_subjmatter,
                            "objectmatter":sets_objmatter,
                            "verbclass":verbclass, "verbtypes":verbtype}]
        #

        #Print some notes
        if do_verbose:
            print("\n> Run of _convert_clause_into_rule complete.")
            print("Orginal clause:\n{0}".format(clause_text))
            print("Extracted rules:")
            for ii in range(0, len(rules)):
                print("{0}: {1}\n-".format(ii, rules[ii]))

        #Return the assembled rules
        return rules
    #

    ##Method: _convert_clause_into_rule
    ##Purpose: Convert Grammar-made clause into rule for rule-based classifier
    def x_convert_clause_into_rule(self, clause_text, clause_ids, flags_nounchunks, ids_nounchunks):
        """
        Method: _convert_clause_into_rule
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Process text into modifs using Grammar class.
        """
        #Set global variables
        do_verbose = self._get_info("do_verbose")

        #Fetch all sets of subject flags via their ids
        num_subj = len(clause_text["subjects"])
        sets_subjmatter = [None]*num_subj #Container for subject flags
        for ii in range(0, num_subj): #Iterate through subjects
            id_chunk = ids_nounchunks[clause_ids["subjects"][ii][0]]
            sets_subjmatter[ii] = [key
                                    for key in flags_nounchunks[id_chunk]
                                    if ((flags_nounchunks[id_chunk][key])
                                        and (key != "is_any"))
                                    ] #Store all subj. flags for this clause
        #
        #If no subjects, set container to empty
        if (num_subj == 0):
            sets_subjmatter = [[]]
        #

        #Fetch all sets of object flags via their ids
        tmp_list = (clause_ids["dir_objects"] + clause_ids["prep_objects"])
        num_obj = len(tmp_list)
        sets_objmatter = [None]*num_obj #Container for object flags
        for ii in range(0, num_obj): #Iterate through objects
            id_chunk = ids_nounchunks[tmp_list[ii][0]]
            sets_objmatter[ii] = [key
                                    for key in flags_nounchunks[id_chunk]
                                    if ((flags_nounchunks[id_chunk][key])
                                        and (key != "is_any"))
                                    ] #Store all obj. flags for this clause
        #
        #If no objects, set container to empty
        if (num_obj == 0):
            sets_objmatter = [[]]
        #

        #Set verb class
        #verbclass = self._categorize_verb(verb=clause_text["verb"])
        verbclass_raw = self._categorize_verb(verb=clause_text["verb"])
        if (verbclass_raw is None):
            verbclass = []
        elif isinstance(verbclass_raw, str):
            verbclass = [verbclass_raw]
        else:
            verbclass = verbclass_raw
        #

        #Set verb type
        #verbtype = clause_text["verbtype"]
        verbtype_raw = clause_text["verbtype"]
        if (verbtype_raw is None):
            verbtype = []
        elif isinstance(verbtype_raw, str):
            verbtype = [verbtype_raw]
        else:
            verbtype = verbtype_raw
        #

        #Set rules from combinations of subj. and obj.matter
        rules = []
        for ii in range(0, len(sets_subjmatter)):
            for jj in range(0, len(sets_objmatter)):
                curr_rule = {"subjectmatter":sets_subjmatter[ii],
                            "objectmatter":sets_objmatter[jj],
                            "verbclass":verbclass, "verbtypes":verbtype}
                #

                #Store rules
                rules.append(curr_rule)
            #
        #

        #Print some notes
        if do_verbose:
            print("\n> Run of _convert_clause_into_rule complete.")
            print("Orginal clause:\n{0}".format(clause_text))
            print("Extracted rules:")
            for ii in range(0, len(rules)):
                print("{0}: {1}\n-".format(ii, rules[ii]))

        #Return the assembled rules
        return rules
    #

    ##Method: _merge_rules
    ##Purpose: Merge list of rules into one rule
    def _merge_rules(self, rules):
        #Extract global variables
        keys_matter = config.nest_keys_matter
        #Extract keys of rules
        list_keys = rules[0].keys()
        #Merge rule across all keys
        merged_rule_raw = {key:[] for key in list_keys}
        for ii in range(0, len(rules)):
            #Merge if any important terms in this rule
            if any([(len(rules[ii][key]) > 0) for key in keys_matter]):
                for curr_key in list_keys:
                    merged_rule_raw[curr_key] += rules[ii][curr_key]
        #
        #Keep only unique values
        merged_rule = {key:list(set(merged_rule_raw[key])) for key in list_keys}
        #
        #Return merged rule
        return merged_rule
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
    def __init__(self, do_final_blanketrule=True, which_classifs=None, do_verbose=False, do_verbose_deep=False):
        ##Initialize storage
        self._storage = {}
        #Store global variables
        self._store_info(do_verbose, "do_verbose")
        self._store_info(do_verbose_deep, "do_verbose_deep")
        if (which_classifs is None):
            which_classifs = config.list_default_verdicts_decisiontree
        self._store_info(which_classifs, "class_names")

        ##Assemble the fixed decision tree
        decision_tree = self._assemble_decision_tree(do_final_blanketrule)
        self._store_info(decision_tree, "decision_tree")

        ##Print some notes
        if do_verbose:
            print("> Initialized instance of Classifier_Rules class.")
            print("Internal decision tree has been assembled.")
            print("NOTE: Decision tree probabilities:\n{0}\n"
                    .format(decision_tree))
        #

        ##Nothing to see here
        return
    #

    ##Method: _apply_decision_tree
    ##Purpose: Apply a decision tree to a 'nest' dictionary for some text
    def _apply_decision_tree(self, decision_tree, rule):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        which_classifs = self._get_info("class_names")
        keys_main = config.nest_keys_main
        keys_matter = [item for item in keys_main if (item.endswith("matter"))]
        bool_keyword = "is_keyword"
        prefix = "prob_"
        #
        rule_mod = rule.copy()
        for curr_key in keys_main: #Encapsulate any single values into lists
            if (isinstance(rule[curr_key], str) or (rule[curr_key] is None)):
                rule_mod[curr_key] = [rule[curr_key]]
            else:
                rule_mod[curr_key] = rule[curr_key]
        #
        #Print some notes
        if do_verbose:
            print("Applying decision tree to the following rule:")
            print(rule_mod)
        #

        ##Ignore this rule if no keywords within
        if (not any([bool_keyword in rule_mod[key] for key in keys_matter])):
            #Print some notes
            if do_verbose:
                print("No keywords in this rule. Skipping.")
            return None
        #

        ##Find matching decision tree branch
        best_branch = None
        best_id = None
        #for key_tree in decision_tree:
        tmp_list = sorted(list(decision_tree.keys()))
        for key_tree in tmp_list:
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
                                        np.sort(rule_mod[key_param])):
                        is_match = False
                        break #Exit from this branch early
                #
                #Otherwise, check inclusive and excluded ('!') matching values
                elif isinstance(curr_branch[key_param], list):
                    #Check for included matching values
                    if ((not all([(item in rule_mod[key_param])
                                for item in curr_branch[key_param]
                                if (not item.startswith("!"))]))
                            or (any([(item in rule_mod[key_param])
                                        for item in curr_branch[key_param]
                                        if (item.startswith("!"))]))):
                        is_match = False
                        break #Exit from this branch early
                #
                #Otherwise, check if any of allowed values contained
                elif isinstance(curr_branch[key_param], set):
                    #Check for any of matching values
                    if not any([(item in rule_mod[key_param])
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
                best_id = key_tree
                #Print some notes
                if do_verbose:
                    print("\n- Found matching decision branch:\n{0}: {1}"
                            .format(best_id, best_branch))
                #
                break
            #Otherwise, carry on
            else:
                pass
            #
        #
        #Raise an error if no matching branch found
        if ((not is_match) or (best_branch is None)):
            raise NotImplementedError("Err: No match found for {0}!"
                                        .format(rule_mod))
        #

        ##Extract the scores from the branch
        #dict_scores = {key:best_branch[prefix+key] for key in which_classifs}
        dict_scores_raw ={key:best_branch[prefix+key] for key in which_classifs}
        #Normalize
        tmp_tot = sum([dict_scores_raw[key] for key in dict_scores_raw])
        dict_scores = {key:(dict_scores_raw[key]/tmp_tot)
                        if (tmp_tot > 0) else 0 #Avoid division by zero
                        for key in dict_scores_raw}
        #

        ##Return the final scores
        if do_verbose:
            print("Final scores computed for rule:\n{0}\n\n{1}\n\n{2}"
                    .format(rule_mod, dict_scores, best_branch))
        #
        return {"scores":dict_scores, "components":best_branch,
                "id_branch":best_id}
    #

    ##Method: _assemble_decision_tree
    ##Purpose: Assemble base of decision tree, with probabilities, that can be read from/expanded as full decision tree
    def x_trynewbuild_2024_03_05a_assemble_decision_tree(self, do_final_blanketrule):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        which_classifs = self._get_info("class_names")
        dict_possible_values = config.dict_tree_possible_values
        #dict_valid_combos = config.dict_tree_valid_value_combinations
        keys_matter = config.nest_keys_matter
        key_verbtype = config.nest_key_verbtype
        all_params = list(dict_possible_values.keys())
        prefix = "prob_"
        strformat = "{0:02d}"
        #


        ##Goal: Generate matrix of probabilities based on 'true' examples
        if True: #Just to make it easier to hide the example content display
            dict_examples_base = {}
            itrack = -1
            #
            ##Keyword-only rules.
            #
            #
            #
            ##Keyword, figure rules.
            #Figure 1 analyze/analyzed/plot/plotted the HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_term_fig"]),
                "verbclass":{"plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.9,
                "prob_data_influenced":0.0,
                "prob_mention":0.1}
            #
            #
            #
            ##Keyword, etal. rules.
            #Authors analyze/analyzed/plot/plotted HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_etal"]),
                "verbclass":{"plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #
            #
            ##Keyword, pron.3rd rules.
            #
            #
            #
            ##Keyword, pron.1st, fig. rules.
            #We analyze/analyzed/plot/plotted HST data in Figure 1.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_term_fig", "is_pron_1st"]),
                "verbclass":{"plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.9,
                "prob_data_influenced":0.0,
                "prob_mention":0.1}
            #
            #
            #
            #Catch all for now.
            if do_final_blanketrule:
                itrack += 1
                dict_examples_base[strformat.format(itrack)] = {
                    "allmatter":"is_any",
                    "verbclass":"is_any",
                    "verbtypes":"is_any",
                    "prob_science":0.0,
                    "prob_data_influenced":0.0,
                    "prob_mention":0.0}
            #
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

            ##For passive example
            """
            #For general (not is_any, not set) case
            if ((curr_ex[key_verbtype] == "is_any")):
                #                    or (curr_ex[key_verbtype] is None)):
                pass #No passive counterpart necessary for is_any case
            #For set, add example+passive for each entry in set
            #elif isinstance(curr_ex[key_verbtype], set):
                #Iterate through set entries
            #    for curr_val in curr_ex[key_verbtype]:
                    #Passive with flipped subj-obj
                    #Extract all parameters and their values
            #        new_ex = {key:curr_ex[key] for key in all_params
            #                if (key not in (keys_matter+[key_verbtype]))}
                    #Add in passive term for verbtypes
            #        tmp_vals = [curr_val, "PASSIVE"]
            #        new_ex[key_verbtype] = tmp_vals
                    #Flip the subject and object terms
            #        new_ex["subjectmatter"] = curr_ex["objectmatter"]
            #        new_ex["objectmatter"] = curr_ex["subjectmatter"]
                    #Normalize and store probabilities for target classifs
            #        new_ex.update(curr_probs)
                    #Store this example
            #        itrack += 1
            #        decision_tree[itrack] = new_ex
            else:
                #Extract all parameters and their values
                new_ex = {key:curr_ex[key] for key in all_params
                        if (key not in (keys_matter+[key_verbtype]))}
                #Add in passive term for verbtypes
                tmp_vals = list(curr_ex[key_verbtype]) #+ ["PASSIVE"]
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
            """
        #

        #Return the assembled decision tree
        return decision_tree
    #

    ##Method: _assemble_decision_tree
    ##Purpose: Assemble base of decision tree, with probabilities, that can be read from/expanded as full decision tree
    def _assemble_decision_tree(self, do_final_blanketrule):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        which_classifs = self._get_info("class_names")
        dict_possible_values = config.dict_tree_possible_values
        #dict_valid_combos = config.dict_tree_valid_value_combinations
        keys_matter = config.nest_keys_matter
        key_verbtype = config.nest_key_verbtype
        all_params = list(dict_possible_values.keys())
        prefix = "prob_"
        strformat = "{0:02d}"
        #


        ##Goal: Generate matrix of probabilities based on 'true' examples
        if True: #Just to make it easier to hide the example content display
            dict_examples_base = {}
            itrack = -1
            #
            #<know verb classes>
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":"is_any",
                "verbclass":{"know"},
                "verbtypes":"is_any",
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #<future verb classes>
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":"is_any",
                "verbclass":"is_any",
                "verbtypes":["FUTURE"],
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #<purpose verb classes>
            #itrack += 1
            #dict_examples_base[strformat.format(itrack)] = {
            #    "allmatter":"is_any",
            #    "verbclass":"is_any",
            #    "verbtypes":["PURPOSE"],
            #    "prob_science":0.3,
            #    "prob_data_influenced":0.0,
            #    "prob_mention":0.7}
            #
            #
            #
            #<Empty keyword statements>
            #HST is/has/was/had cool images.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #HST presents/observes cool objects.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #HST simulations simulate cool objects.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #
            #
            #<Empty keyword+ statements>
            #HST data.
            #itrack += 1
            #dict_examples_base[strformat.format(itrack)] = {
            #    "allmatter":tuple(["is_keyword"]),
            #    "objectmatter":tuple([]),
            #    "verbclass":tuple([]),
            #    "verbtypes":tuple([]),
            #    "prob_science":0.8,
            #    "prob_data_influenced":0.0,
            #    "prob_mention":0.2}
            #Figure 1's HST data presents/analyzes/presented/analyzed the trend.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_term_fig"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.9,
                "prob_data_influenced":0.1,
                "prob_mention":0.0}
            #Their HST study presents/analyzes/presented/analyzed a trend.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_pron_3rd"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #Authors' HST study presents/analyzes/presented/analyzed a trend.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_etal"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #
            #
            #<Empty verb statements>
            #HST.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword"]),
                "verbclass":tuple([]),
                "verbtypes":tuple([]),
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #HST orbits/orbitted nearby.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #E.g.: "The HST data adheres to the trend in the HST image in Figure 1."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_term_fig"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.8,
                "prob_data_influenced":0.1,
                "prob_mention":0.1}
            #
            #E.g.: "Our approach is guided by the HST data rms."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_1st", "is_keyword"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.9,
                "prob_data_influenced":0.0,
                "prob_mention":0.1}
            #E.g.: "Their HST data appeared in the archive."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_3rd", "is_keyword"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #E.g.: "Author's HST data appeared in the archive."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_etal", "is_keyword"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #E.g.: "Our analysis considers our HST data."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_1st", "is_keyword"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.8,
                "prob_data_influenced":0.1,
                "prob_mention":0.1}
            #
            #E.g.: "Our analysis considers their HST data."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_1st", "is_keyword", "is_pron_3rd"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.2,
                "prob_data_influenced":0.6,
                "prob_mention":0.2}
            #
            #E.g.: "Our analysis considers Author's HST data."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_1st", "is_keyword", "is_etal"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.1,
                "prob_data_influenced":0.8,
                "prob_mention":0.1}
            #
            #E.g.: "Author's analysis considers Author's HST data."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_etal"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #E.g.: "Our HST observations echo the HST trend."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_pron_1st"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.8,
                "prob_data_influenced":0.1,
                "prob_mention":0.1}
            #E.g.: "Their HST observations echo the HST trend."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_pron_3rd"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #E.g.: "HST data is considered in our Figure 1."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_pron_1st", "is_term_fig"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.8,
                "prob_data_influenced":0.1,
                "prob_mention":0.1}
            #
            #E.g.: "They/Author1 noted the trend in Author2's HST data, as plotted in Figure 1."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_etal", "is_term_fig"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #
            #
            #<be/has verb classes>, singular matter
            #Our analysis is/has/was/had HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_1st", "is_keyword"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.9,
                "prob_data_influenced":0.0,
                "prob_mention":0.1}
            #Their analysis is/has/was/had HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_3rd", "is_keyword"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #Authors' analysis is/has/was/had HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_etal", "is_keyword"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #Figure 1 is/has/was/had HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_term_fig", "is_keyword"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.9,
                "prob_data_influenced":0.0,
                "prob_mention":0.1}
            #
            #
            #
            #<be/has verb classes>, multiple matter
            #Our analysis is/has/was/had neat.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_1st", "is_keyword"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.8,
                "prob_data_influenced":0.1,
                "prob_mention":0.1}
            #
            #Figure 1 analysis is/has/was/had neat images.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_term_fig", "is_keyword"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.8,
                "prob_data_influenced":0.1,
                "prob_mention":0.1}
            #
            #The HST analysis is/has/was/had our HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_pron_1st"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.9,
                "prob_data_influenced":0.0,
                "prob_mention":0.1}
            #Their analysis is/has/was/had their HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_pron_3rd"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #Authors' analysis is/has/was/had Authors HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_etal"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #Our analysis is/has/was/had their HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_1st", "is_keyword", "is_pron_3rd"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.4,
                "prob_data_influenced":0.6,
                "prob_mention":0.0}
            #Our analysis is/has/was/had Authors HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_1st", "is_keyword", "is_etal"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.2,
                "prob_data_influenced":0.8,
                "prob_mention":0.0}
            #
            #We is/has/was/had HST data in Figure 1.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_1st", "is_keyword", "is_term_fig"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.9,
                "prob_data_influenced":0.05,
                "prob_mention":0.05}
            #
            #They is/has/was/had our HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_3rd", "is_keyword", "is_pron_1st"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.3,
                "prob_data_influenced":0.3,
                "prob_mention":0.4}
            #Authors is/has/was/had our HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_etal", "is_keyword", "is_pron_1st"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #Their analysis is/has/was/had on the HST data in Figure 1.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_3rd", "is_keyword", "is_term_fig"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #Authors' analysis is/has/was/had on the HST data in Figure 1.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_etal", "is_keyword", "is_term_fig"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.9,
                "prob_mention":0.3}
            #
            #
            #
            #<be/has verb classes>, many matter
            #Our analysis of their work is/has/was/had our HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_1st", "is_pron_3rd", "is_keyword"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.8,
                "prob_data_influenced":0.1,
                "prob_mention":0.1}
            #Their HST observations are/were/has/had in Authors et al..
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_pron_3rd", "is_etal"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #
            #
            #<science/plot verb classes>, singular matter
            #We analyze/plot/analyzed/plotted HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_1st", "is_keyword"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #They analyze/plot/analyzed/plotted HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_3rd", "is_keyword"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #Authors analyze/plot/analyzed/plotted HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_etal", "is_keyword"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #
            #
            #We analyze/plot/analyzed/plotted their HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_1st", "is_keyword", "is_pron_3rd"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.2,
                "prob_data_influenced":0.9,
                "prob_mention":0.0}
            #We analyze/plot/analyzed/plotted Authors HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_1st", "is_keyword", "is_etal"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.2,
                "prob_data_influenced":0.9,
                "prob_mention":0.0}
            #
            #We analyze/plot/analyzed/plotted HST data in Figure 1.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_1st", "is_keyword", "is_term_fig"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #We analyze/plot/analyzed/plotted our HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_pron_1st"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #They analyze/plot/analyzed/plotted our HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_3rd", "is_keyword", "is_pron_1st"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #Authors analyze/plot/analyzed/plotted our HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_etal", "is_keyword", "is_pron_1st"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #
            #
            #<science/plot verb classes>, many matter
            #Authors analyze/analyzed/plot/plotted HST data in their work.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_etal", "is_keyword", "is_pron_3rd"]),
                "verbclass":{"be", "has", "science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #Figure 1 analyze/analyzed/plot/plotted HST data from Authors.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_etal", "is_keyword", "is_term_fig"]),
                "verbclass":{"be", "has", "science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.2,
                "prob_data_influenced":0.5,
                "prob_mention":0.3}
            #
            #
            #
            #<datainfl. verb classes>, singular matter
            #We simulate/simulated HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_1st", "is_keyword"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":1.0,
                "prob_mention":0.0}
            #Figure 1 simulate/simulated HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_term_fig", "is_keyword"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":1.0,
                "prob_mention":0.0}
            #They simulate/simulated HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_3rd", "is_keyword"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #Authors simulate/simulated HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_etal", "is_keyword"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #
            #
            #<datainfl. verb classes>, multiple matter
            #Our HST program simulates/simulated HD 123456.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_pron_1st"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":1.0,
                "prob_mention":0.0}
            #
            #HST data is simulated/was simulated in our Figure 1.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_term_fig", "is_pron_1st"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.3,
                "prob_data_influenced":0.9,
                "prob_mention":0.0}
            #
            #They simulate/simulated their HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_pron_3rd"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #Authors simulate/simulated Authors' HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_etal"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #We simulate/simulated their HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_1st", "is_keyword", "is_pron_3rd"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":1.0,
                "prob_mention":0.0}
            #We simulate/simulated Authors' HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_1st", "is_keyword", "is_etal"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":1.0,
                "prob_mention":0.0}
            #
            #We simulate/simulated HST data in Figure 1.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_1st", "is_keyword", "is_term_fig"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.3,
                "prob_data_influenced":0.9,
                "prob_mention":0.0}
            #
            #They simulate/simulated our HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_pron_3rd", "is_keyword", "is_pron_1st"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.3,
                "prob_mention":0.9}
            #Authors simulate/simulated our HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_etal", "is_keyword", "is_pron_1st"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.3,
                "prob_mention":0.9}
            #
            #
            #
            #<data-influenced verb classes>, many matter
            #Authors simulate/simulated HST data in their work.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_etal", "is_keyword", "is_pron_3rd"]),
                "verbclass":{"be", "has", "datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.4,
                "prob_mention":0.8}
            #
            #
            #
            #Rules with triple-matter
            #The HST targets is/has in our Figure 1.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_pron_1st", "is_term_fig"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.9,
                "prob_data_influenced":0.1,
                "prob_mention":0.0}
            #The HST targets show/analyze in our Figure 1.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_pron_1st", "is_term_fig"]),
                "verbclass":{"plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #The Figure 1 HST targets is/has/show/analyze the HST data trend.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_term_fig"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.9,
                "prob_data_influenced":0.0,
                "prob_mention":0.1}
            #Their HST targets is/has/show/analyze the HST data trend.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_pron_3rd"]),
                "verbclass":{"be", "has", "plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #The Author HST targets is/has/show/analyze the HST data trend.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "allmatter":tuple(["is_keyword", "is_etal"]),
                "verbclass":{"be", "has", "plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #
            #
            """
            #Expandable-rules with triple-matter
            #The trend is/was/has/had/plots/plotted/analyzes/analyzed for the Authors' HST data in Figure 1.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":["is_etal", "is_keyword", "is_term_fig"],
                "verbclass":{"be", "has", "plot", "science", "datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.3,
                "prob_data_influenced":0.9,
                "prob_mention":0.0}
            #
            #
            #
            #Blanket is_pron_1st combinations
            #We/Figure 1 analyze/plot/analyzed/plotted our HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":["is_pron_1st", "is_term_fig", "!is_pron_3rd", "!is_etal", "is_keyword"],
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #<Pron.1st involvement with keyword figure
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":["is_pron_1st", "!is_etal", "!is_pron_3rd"],
                "objectmatter":["is_keyword", "is_pron_1st", "!is_etal", "!is_pron_3rd"],
                "verbclass":{"science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #<Pron.1st involvement with keyword figure
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":["is_pron_1st", "!is_etal", "!is_pron_3rd"],
                "objectmatter":["is_keyword", "is_term_fig", "!is_etal", "!is_pron_3rd"],
                "verbclass":["!datainfluenced"], #{"be", "has", "plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #
            #
            #Blanket is_term_fig combinations
            #<Figure involvement with keyword>.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":["is_term_fig", "!is_pron_3rd", "!is_etal"],
                "objectmatter":["is_keyword", "!is_pron_3rd", "!is_etal"],
                "verbclass":{"be", "has", "plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #
            #
            #Blanket is_etal combinations
            #<Author involvement with keyword figure>.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":["is_etal"],
                "objectmatter":["is_keyword", "is_term_fig"],
                "verbclass":{"be", "has", "plot", "science", "datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.9,
                "prob_mention":0.3}
            #
            #Authors analyze/plot/analyzed/plotted Authors HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":["is_pron_3rd", "is_etal", "!is_pron_1st", "!is_term_fig"],
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.3,
                "prob_mention":0.9}
            #
            #<Author involvement with keyword data>.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":["is_etal", "is_keyword", "!is_pron_1st", "!is_term_fig"],
                "objectmatter":tuple([]),
                "verbclass":{"be", "has", "plot", "science", "datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.3,
                "prob_mention":0.9}
            #<Author involvement with keyword object>.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":["is_etal", "!is_pron_1st", "!is_term_fig"],
                "objectmatter":["is_keyword", "!is_pron_1st", "!is_term_fig"],
                "verbclass":{"be", "has", "plot", "science", "datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.3,
                "prob_mention":0.9}
            #<Author object involvement with keyword object>.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":["!is_pron_1st", "!is_term_fig"],
                "objectmatter":["is_keyword", "is_etal", "!is_pron_1st", "!is_term_fig"],
                "verbclass":{"be", "has", "plot", "science", "datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.3,
                "prob_mention":0.9}
            #
            #<Author object involvement with keyword object>.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":["is_etal", "!is_pron_1st", "!is_term_fig"],
                "objectmatter":["is_keyword", "!is_pron_1st", "!is_term_fig"],
                "verbclass":{"be", "has", "plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #
            #
            #Blanket is_pron_3rd combinations
            #<Author object involvement with keyword object>.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":["is_pron_3rd", "!is_etal", "!is_pron_1st", "!is_term_fig"],
                "objectmatter":["is_keyword", "!is_etal", "!is_pron_1st", "!is_term_fig"],
                "verbclass":{"be", "has", "plot", "science", "datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.5,
                "prob_data_influenced":0.5,
                "prob_mention":0.5}
            #
            #"""
            #
            #Catch all for now.
            if do_final_blanketrule:
                itrack += 1
                dict_examples_base[strformat.format(itrack)] = {
                    "allmatter":"is_any",
                    "verbclass":"is_any",
                    "verbtypes":"is_any",
                    "prob_science":0.0,
                    "prob_data_influenced":0.0,
                    "prob_mention":0.0}
            #
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

            ##For passive example
            """
            #For general (not is_any, not set) case
            if ((curr_ex[key_verbtype] == "is_any")):
                #                    or (curr_ex[key_verbtype] is None)):
                pass #No passive counterpart necessary for is_any case
            #For set, add example+passive for each entry in set
            #elif isinstance(curr_ex[key_verbtype], set):
                #Iterate through set entries
            #    for curr_val in curr_ex[key_verbtype]:
                    #Passive with flipped subj-obj
                    #Extract all parameters and their values
            #        new_ex = {key:curr_ex[key] for key in all_params
            #                if (key not in (keys_matter+[key_verbtype]))}
                    #Add in passive term for verbtypes
            #        tmp_vals = [curr_val, "PASSIVE"]
            #        new_ex[key_verbtype] = tmp_vals
                    #Flip the subject and object terms
            #        new_ex["subjectmatter"] = curr_ex["objectmatter"]
            #        new_ex["objectmatter"] = curr_ex["subjectmatter"]
                    #Normalize and store probabilities for target classifs
            #        new_ex.update(curr_probs)
                    #Store this example
            #        itrack += 1
            #        decision_tree[itrack] = new_ex
            else:
                #Extract all parameters and their values
                new_ex = {key:curr_ex[key] for key in all_params
                        if (key not in (keys_matter+[key_verbtype]))}
                #Add in passive term for verbtypes
                tmp_vals = list(curr_ex[key_verbtype]) #+ ["PASSIVE"]
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
            """
        #

        #Return the assembled decision tree
        return decision_tree
    #

    ##Method: _assemble_decision_tree
    ##Purpose: Assemble base of decision tree, with probabilities, that can be read from/expanded as full decision tree
    def old_beforemattercomb_assemble_decision_tree(self, do_final_blanketrule):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        which_classifs = self._get_info("class_names")
        dict_possible_values = config.dict_tree_possible_values
        #dict_valid_combos = config.dict_tree_valid_value_combinations
        keys_matter = config.nest_keys_matter
        key_verbtype = config.nest_key_verbtype
        all_params = list(dict_possible_values.keys())
        prefix = "prob_"
        strformat = "{0:02d}"
        #


        ##Goal: Generate matrix of probabilities based on 'true' examples
        if True: #Just to make it easier to hide the example content display
            dict_examples_base = {}
            itrack = -1
            #
            #<know verb classes>
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":"is_any",
                "objectmatter":"is_any",
                "verbclass":{"know"},
                "verbtypes":"is_any",
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #<future verb classes>
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":"is_any",
                "objectmatter":"is_any",
                "verbclass":"is_any",
                "verbtypes":["FUTURE"],
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #<purpose verb classes>
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":"is_any",
                "objectmatter":"is_any",
                "verbclass":"is_any",
                "verbtypes":["PURPOSE"],
                "prob_science":0.5,
                "prob_data_influenced":0.5,
                "prob_mention":0.5}
            #
            #
            #
            #<Empty keyword statements>
            #HST is/has/was/had cool images.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":tuple([]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.1,
                "prob_data_influenced":0.1,
                "prob_mention":0.8}
            #HST presents/observes cool objects.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":tuple([]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.5,
                "prob_data_influenced":0.5,
                "prob_mention":0.5}
            #HST simulations simulate cool objects.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":tuple([]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.5,
                "prob_data_influenced":0.5,
                "prob_mention":0.5}
            #
            #
            #
            #<Empty keyword+ statements>
            #HST data.
            #itrack += 1
            #dict_examples_base[strformat.format(itrack)] = {
            #    "subjectmatter":tuple(["is_keyword"]),
            #    "objectmatter":tuple([]),
            #    "verbclass":tuple([]),
            #    "verbtypes":tuple([]),
            #    "prob_science":0.8,
            #    "prob_data_influenced":0.0,
            #    "prob_mention":0.2}
            #Figure 1's HST data presents/analyzes/presented/analyzed the trend.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword", "is_term_fig"]),
                "objectmatter":tuple([]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #Their HST study presents/analyzes/presented/analyzed a trend.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword", "is_pron_3rd"]),
                "objectmatter":tuple([]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.3,
                "prob_mention":0.9}
            #Authors' HST study presents/analyzes/presented/analyzed a trend.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword", "is_etal"]),
                "objectmatter":tuple([]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.3,
                "prob_mention":0.9}
            #
            #HST targets present/analyze/presented/analyzed in the HST observations.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.7,
                "prob_data_influenced":0.0,
                "prob_mention":0.3}
            #HST targets are simulated/were simulated in the HST observations.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.7,
                "prob_data_influenced":0.0,
                "prob_mention":0.3}
            #
            #
            #
            #<Empty verb statements>
            #HST.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":tuple([]),
                "verbclass":tuple([]),
                "verbtypes":tuple([]),
                "prob_science":0.5,
                "prob_data_influenced":0.5,
                "prob_mention":0.5}
            #
            #HST orbits/orbitted nearby.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":tuple([]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.1,
                "prob_data_influenced":0.1,
                "prob_mention":0.8}
            #
            #HST orbits/orbitted nearby.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.1,
                "prob_data_influenced":0.1,
                "prob_mention":0.8}
            #
            #E.g.: "The HST data adheres to the trend in the HST image in Figure 1."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":tuple(["is_keyword", "is_term_fig"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.8,
                "prob_data_influenced":0.1,
                "prob_mention":0.1}
            #
            #E.g.: "Figure 1's error bars are guided by the HST data rms."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_term_fig"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #E.g.: "Our approach is guided by the HST data rms."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.7,
                "prob_data_influenced":0.4,
                "prob_mention":0.0}
            #E.g.: "Our HST data appeared in the archive."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st", "is_keyword"]),
                "objectmatter":tuple([]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.9,
                "prob_data_influenced":0.0,
                "prob_mention":0.1}
            #E.g.: "Our HST data appeared in the archive."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_term_fig", "is_keyword"]),
                "objectmatter":tuple([]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.9,
                "prob_data_influenced":0.0,
                "prob_mention":0.1}
            #E.g.: "Their HST data appeared in the archive."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_3rd", "is_keyword"]),
                "objectmatter":tuple([]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.0,
                "prob_data_influenced":0.3,
                "prob_mention":0.9}
            #E.g.: "Author's HST data appeared in the archive."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_etal", "is_keyword"]),
                "objectmatter":tuple([]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.0,
                "prob_data_influenced":0.3,
                "prob_mention":0.9}
            #
            #E.g.: "Our analysis considers our HST data."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword", "is_pron_1st"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #E.g.: "Our analysis considers their HST data."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword", "is_pron_3rd"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.1,
                "prob_data_influenced":0.8,
                "prob_mention":0.1}
            #
            #E.g.: "Our analysis considers Author's HST data."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword", "is_etal"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.1,
                "prob_data_influenced":0.8,
                "prob_mention":0.1}
            #
            #E.g.: "Author's analysis considers Author's HST data."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_etal"]),
                "objectmatter":tuple(["is_keyword", "is_etal"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #E.g.: "Our HST observations echo the HST trend."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword", "is_pron_1st"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #E.g.: "Their HST observations echo the HST trend."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword", "is_pron_3rd"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #E.g.: "Author's HST observations echo the HST trend."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword", "is_etal"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #E.g.: "HST data is considered in our Figure 1."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":tuple(["is_keyword", "is_pron_1st", "is_term_fig"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #E.g.: "Their/Author's error bars are guided by the HST data rms."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":{"is_etal", "is_pron_3rd"},
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.0,
                "prob_data_influenced":0.3,
                "prob_mention":0.9}
            #E.g.: "They/Author1 noted the trend in Author2's HST data, as plotted in Figure 1."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":{"is_etal", "is_pron_3rd"},
                "objectmatter":tuple(["is_keyword", "is_etal", "is_term_fig"]),
                "verbclass":tuple([]),
                "verbtypes":[],
                "prob_science":0.0,
                "prob_data_influenced":0.35,
                "prob_mention":0.85}
            #
            #
            #
            #<be/has verb classes>, singular matter
            #E.g.: "HST is short for 'Hubble Space Telescope'."
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #Our analysis is/has/was/had HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #Their analysis is/has/was/had HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_3rd"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #Authors' analysis is/has/was/had HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_etal"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #Figure 1 is/has/was/had HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_term_fig"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #
            #
            #<be/has verb classes>, multiple matter
            #Our analysis is/has/was/had neat.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st", "is_keyword"]),
                "objectmatter":tuple([]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.8,
                "prob_data_influenced":0.1,
                "prob_mention":0.1}
            #
            #Figure 1 analysis is/has/was/had neat images.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_term_fig", "is_keyword"]),
                "objectmatter":tuple([]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.8,
                "prob_data_influenced":0.1,
                "prob_mention":0.1}
            #
            #The HST analysis is/has/was/had our HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":tuple(["is_keyword", "is_pron_1st"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #Our analysis is/has/was/had our HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword", "is_pron_1st"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #Their analysis is/has/was/had their HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_3rd"]),
                "objectmatter":tuple(["is_keyword", "is_pron_3rd"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #Authors' analysis is/has/was/had Authors HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_etal"]),
                "objectmatter":tuple(["is_keyword", "is_etal"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #Our analysis is/has/was/had their HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword", "is_pron_3rd"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.3,
                "prob_data_influenced":0.9,
                "prob_mention":0.0}
            #Our analysis is/has/was/had Authors HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword", "is_etal"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.3,
                "prob_data_influenced":0.9,
                "prob_mention":0.0}
            #
            #We is/has/was/had HST data in Figure 1.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword", "is_term_fig"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #They is/has/was/had our HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_3rd"]),
                "objectmatter":tuple(["is_keyword", "is_pron_1st"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.3,
                "prob_mention":0.9}
            #Authors is/has/was/had our HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_etal"]),
                "objectmatter":tuple(["is_keyword", "is_pron_1st"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.3,
                "prob_mention":0.9}
            #
            #Their analysis is/has/was/had on the HST data in Figure 1.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_3rd"]),
                "objectmatter":tuple(["is_keyword", "is_term_fig"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.9,
                "prob_mention":0.3}
            #Authors' analysis is/has/was/had on the HST data in Figure 1.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_etal"]),
                "objectmatter":tuple(["is_keyword", "is_term_fig"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.9,
                "prob_mention":0.3}
            #
            #
            #
            #<be/has verb classes>, many matter
            #Our analysis of their work is/has/was/had our HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st", "is_pron_3rd"]),
                "objectmatter":tuple(["is_keyword", "is_pron_1st"]),
                "verbclass":{"be", "has"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.8,
                "prob_data_influenced":0.1,
                "prob_mention":0.1}
            #
            #
            #
            #<science/plot verb classes>, singular matter
            #We analyze/plot/analyzed/plotted HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #They analyze/plot/analyzed/plotted HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_3rd"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #Authors analyze/plot/analyzed/plotted HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_etal"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #Figure 1 analyzes/plots/analyzed/plotted HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_term_fig"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #
            #
            #<science/plot verb classes>, multiple matter
            #Our HST program presents/presented/observes/observed HD 123456.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword", "is_pron_1st"]),
                "objectmatter":tuple([]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #They analyze/plot/analyzed/plotted their HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_3rd"]),
                "objectmatter":tuple(["is_keyword", "is_pron_3rd"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #Authors analyze/plot/analyzed/plotted Authors HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_etal"]),
                "objectmatter":tuple(["is_keyword", "is_etal"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #We analyze/plot/analyzed/plotted their HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword", "is_pron_3rd"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.2,
                "prob_data_influenced":0.9,
                "prob_mention":0.0}
            #We analyze/plot/analyzed/plotted Authors HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword", "is_etal"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.2,
                "prob_data_influenced":0.9,
                "prob_mention":0.0}
            #
            #We analyze/plot/analyzed/plotted HST data in Figure 1.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword", "is_term_fig"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #We analyze/plot/analyzed/plotted our HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword", "is_pron_1st"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #They analyze/plot/analyzed/plotted our HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_3rd"]),
                "objectmatter":tuple(["is_keyword", "is_pron_1st"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.3,
                "prob_mention":0.9}
            #Authors analyze/plot/analyzed/plotted our HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_etal"]),
                "objectmatter":tuple(["is_keyword", "is_pron_1st"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.3,
                "prob_mention":0.9}
            #
            #
            #
            #<datainfl. verb classes>, singular matter
            #We simulate/simulated HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":1.0,
                "prob_mention":0.0}
            #Figure 1 simulate/simulated HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_term_fig"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":1.0,
                "prob_mention":0.0}
            #They simulate/simulated HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_3rd"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #Authors simulate/simulated HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_etal"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #
            #
            #<datainfl. verb classes>, multiple matter
            #Our HST program simulates/simulated HD 123456.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword", "is_pron_1st"]),
                "objectmatter":tuple([]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":1.0,
                "prob_mention":0.0}
            #
            #HST data is simulated/was simulated in our Figure 1.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":tuple(["is_term_fig", "is_pron_1st"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.3,
                "prob_data_influenced":0.9,
                "prob_mention":0.0}
            #
            #HST data is simulated/was simulated in our paper.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":tuple(["is_keyword", "is_pron_1st"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.3,
                "prob_data_influenced":0.9,
                "prob_mention":0.0}
            #
            #We simulate/simulated our HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword", "is_pron_1st"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":1.0,
                "prob_mention":0.0}
            #They simulate/simulated their HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_3rd"]),
                "objectmatter":tuple(["is_keyword", "is_pron_3rd"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #Authors simulate/simulated Authors' HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_etal"]),
                "objectmatter":tuple(["is_keyword", "is_etal"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #We simulate/simulated their HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword", "is_pron_3rd"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":1.0,
                "prob_mention":0.0}
            #We simulate/simulated Authors' HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword", "is_etal"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":1.0,
                "prob_mention":0.0}
            #
            #We simulate/simulated HST data in Figure 1.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword", "is_term_fig"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.3,
                "prob_data_influenced":0.9,
                "prob_mention":0.0}
            #
            #They simulate/simulated our HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_pron_3rd"]),
                "objectmatter":tuple(["is_keyword", "is_pron_1st"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.3,
                "prob_mention":0.9}
            #Authors simulate/simulated our HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_etal"]),
                "objectmatter":tuple(["is_keyword", "is_pron_1st"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.3,
                "prob_mention":0.9}
            #
            #
            #
            #Rules with triple-matter
            #The HST targets is/has/show/analyze in our Figure 1.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":tuple(["is_pron_1st", "is_term_fig"]),
                "verbclass":{"be", "has", "plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #The Figure 1 HST targets is/has/show/analyze the HST data trend.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword", "is_pron_1st"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"be", "has", "plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #The Figure 1 HST targets is/has/show/analyze the HST data trend.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword", "is_term_fig"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"be", "has", "plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #Their HST targets is/has/show/analyze the HST data trend.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword", "is_pron_3rd"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"be", "has", "plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.3,
                "prob_mention":0.9}
            #The Author HST targets is/has/show/analyze the HST data trend.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword", "is_etal"]),
                "objectmatter":tuple(["is_keyword"]),
                "verbclass":{"be", "has", "plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.3,
                "prob_mention":0.9}
            #
            #
            #
            #Expandable-rules with triple-matter
            #The trend is/was/has/had/plots/plotted/analyzes/analyzed for the Authors' HST data in Figure 1.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":[],
                "objectmatter":["is_etal", "is_keyword", "is_term_fig"],
                "verbclass":{"be", "has", "plot", "science", "datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.3,
                "prob_data_influenced":0.9,
                "prob_mention":0.0}
            #
            #
            #
            #Blanket is_pron_1st combinations
            #We/Figure 1 analyze/plot/analyzed/plotted our HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":["is_pron_1st", "is_term_fig", "!is_pron_3rd", "!is_etal"],
                "objectmatter":tuple(["is_keyword", "is_pron_1st"]),
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #<Pron.1st involvement with keyword figure
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":["is_pron_1st", "!is_etal", "!is_pron_3rd"],
                "objectmatter":["is_keyword", "is_pron_1st", "!is_etal", "!is_pron_3rd"],
                "verbclass":{"science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #<Pron.1st involvement with keyword figure
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":["is_pron_1st", "!is_etal", "!is_pron_3rd"],
                "objectmatter":["is_keyword", "is_term_fig", "!is_etal", "!is_pron_3rd"],
                "verbclass":["!datainfluenced"], #{"be", "has", "plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #
            #
            #Blanket is_term_fig combinations
            #<Figure involvement with keyword>.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":["is_term_fig", "!is_pron_3rd", "!is_etal"],
                "objectmatter":["is_keyword", "!is_pron_3rd", "!is_etal"],
                "verbclass":{"be", "has", "plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":1.0,
                "prob_data_influenced":0.0,
                "prob_mention":0.0}
            #
            #
            #
            #Blanket is_etal combinations
            #<Author involvement with keyword figure>.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":["is_etal"],
                "objectmatter":["is_keyword", "is_term_fig"],
                "verbclass":{"be", "has", "plot", "science", "datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.9,
                "prob_mention":0.3}
            #
            #Authors analyze/plot/analyzed/plotted Authors HST data.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":["is_pron_3rd", "is_etal", "!is_pron_1st", "!is_term_fig"],
                "verbclass":{"science", "plot"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.3,
                "prob_mention":0.9}
            #
            #<Author involvement with keyword data>.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":["is_etal", "is_keyword", "!is_pron_1st", "!is_term_fig"],
                "objectmatter":tuple([]),
                "verbclass":{"be", "has", "plot", "science", "datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.3,
                "prob_mention":0.9}
            #<Author involvement with keyword object>.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":["is_etal", "!is_pron_1st", "!is_term_fig"],
                "objectmatter":["is_keyword", "!is_pron_1st", "!is_term_fig"],
                "verbclass":{"be", "has", "plot", "science", "datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.3,
                "prob_mention":0.9}
            #<Author object involvement with keyword object>.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":["!is_pron_1st", "!is_term_fig"],
                "objectmatter":["is_keyword", "is_etal", "!is_pron_1st", "!is_term_fig"],
                "verbclass":{"be", "has", "plot", "science", "datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.3,
                "prob_mention":0.9}
            #
            #<Author object involvement with keyword object>.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":["is_etal", "!is_pron_1st", "!is_term_fig"],
                "objectmatter":["is_keyword", "!is_pron_1st", "!is_term_fig"],
                "verbclass":{"be", "has", "plot", "science"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":0.0,
                "prob_mention":1.0}
            #
            #
            #
            #Blanket is_pron_3rd combinations
            #<Author object involvement with keyword object>.
            itrack += 1
            dict_examples_base[strformat.format(itrack)] = {
                "subjectmatter":["is_pron_3rd", "!is_etal", "!is_pron_1st", "!is_term_fig"],
                "objectmatter":["is_keyword", "!is_etal", "!is_pron_1st", "!is_term_fig"],
                "verbclass":{"be", "has", "plot", "science", "datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.5,
                "prob_data_influenced":0.5,
                "prob_mention":0.5}
            #
            #
            #
            #Catch all for now.
            if do_final_blanketrule:
                itrack += 1
                dict_examples_base[strformat.format(itrack)] = {
                    "subjectmatter":"is_any",
                    "objectmatter":"is_any",
                    "verbclass":"is_any",
                    "verbtypes":"is_any",
                    "prob_science":0.5,
                    "prob_data_influenced":0.5,
                    "prob_mention":0.5}
            #
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

            ##For passive example
            #For general (not is_any, not set) case
            if ((curr_ex[key_verbtype] == "is_any")):
                #                    or (curr_ex[key_verbtype] is None)):
                pass #No passive counterpart necessary for is_any case
            #For set, add example+passive for each entry in set
            #elif isinstance(curr_ex[key_verbtype], set):
                #Iterate through set entries
            #    for curr_val in curr_ex[key_verbtype]:
                    #Passive with flipped subj-obj
                    #Extract all parameters and their values
            #        new_ex = {key:curr_ex[key] for key in all_params
            #                if (key not in (keys_matter+[key_verbtype]))}
                    #Add in passive term for verbtypes
            #        tmp_vals = [curr_val, "PASSIVE"]
            #        new_ex[key_verbtype] = tmp_vals
                    #Flip the subject and object terms
            #        new_ex["subjectmatter"] = curr_ex["objectmatter"]
            #        new_ex["objectmatter"] = curr_ex["subjectmatter"]
                    #Normalize and store probabilities for target classifs
            #        new_ex.update(curr_probs)
                    #Store this example
            #        itrack += 1
            #        decision_tree[itrack] = new_ex
            else:
                #Extract all parameters and their values
                new_ex = {key:curr_ex[key] for key in all_params
                        if (key not in (keys_matter+[key_verbtype]))}
                #Add in passive term for verbtypes
                tmp_vals = list(curr_ex[key_verbtype]) #+ ["PASSIVE"]
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

    ##Method: _assemble_decision_tree
    ##Purpose: Assemble base of decision tree, with probabilities, that can be read from/expanded as full decision tree
    def old_assemble_decision_tree(self):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        which_classifs = self._get_info("class_names")
        dict_possible_values = config.dict_tree_possible_values
        #dict_valid_combos = config.dict_tree_valid_value_combinations
        keys_matter = config.nest_keys_matter
        key_verbtype = config.nest_key_verbtype
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
                "prob_science":0.9,
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
                "prob_data_influenced":0.4,
                "prob_mention":0.5}
            #
            #’The data is from OBJ by Authorsetal.’
            #itrack += 1
            #dict_examples_base[str(itrack)] = {
            #    "subjectmatter":tuple([]),
            #    "objectmatter":tuple(["is_keyword", "is_etal"]),
            #    "verbclass":{"be"},
            #    "verbtypes":{"PRESENT"},
            #    "prob_science":0.0,
            #    "prob_data_influenced":0.8,
            #    "prob_mention":0.6}
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
                "prob_science":0.9,
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
                "prob_science":0.9,
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
                "prob_mention":0.0}
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
                "prob_mention":0.0}
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
                "prob_data_influenced":1.0,
                "prob_mention":0.0}
            #
            #’We simulate/simulated the OBJ data of Authorsetal.’
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword", "is_etal"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":1.0,
                "prob_mention":0.0}
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
            #is_key; is_fig, !is_etal, !is_they
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":["is_term_fig", "!is_etal", "!is_pron_3rd"],
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.9,
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
                "prob_science":0.9,
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
                "prob_science":0.9,
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
            #<Keyword figure be/has/plot/science/datainfl. by etal and pron_3rd>
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword", "is_term_fig"]),
                "objectmatter":["is_etal"],
                "verbclass":"is_any",
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.0,
                "prob_data_influenced":1.0,
                "prob_mention":0.0}
            #
            #<Keyword figure be/has/plot/science/datainfl. by etal and pron_3rd>
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword", "is_term_fig"]),
                "objectmatter":["is_pron_3rd"],
                "verbclass":"is_any",
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.0,
                "prob_data_influenced":0.9,
                "prob_mention":0.1}
            #
            #<Keyword figure be/has/plot/science/datainfl. by etal and pron_3rd>
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword", "is_term_fig"]),
                "objectmatter":["is_etal", "is_pron_3rd"],
                "verbclass":"is_any",
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.0,
                "prob_data_influenced":1.0,
                "prob_mention":0.1}
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

            ##For passive example
            #For general (not is_any, not set) case
            if ((curr_ex[key_verbtype] == "is_any")):
                #                    or (curr_ex[key_verbtype] is None)):
                pass #No passive counterpart necessary for is_any case
            #For set, add example+passive for each entry in set
            #elif isinstance(curr_ex[key_verbtype], set):
                #Iterate through set entries
            #    for curr_val in curr_ex[key_verbtype]:
                    #Passive with flipped subj-obj
                    #Extract all parameters and their values
            #        new_ex = {key:curr_ex[key] for key in all_params
            #                if (key not in (keys_matter+[key_verbtype]))}
                    #Add in passive term for verbtypes
            #        tmp_vals = [curr_val, "PASSIVE"]
            #        new_ex[key_verbtype] = tmp_vals
                    #Flip the subject and object terms
            #        new_ex["subjectmatter"] = curr_ex["objectmatter"]
            #        new_ex["objectmatter"] = curr_ex["subjectmatter"]
                    #Normalize and store probabilities for target classifs
            #        new_ex.update(curr_probs)
                    #Store this example
            #        itrack += 1
            #        decision_tree[itrack] = new_ex
            else:
                #Extract all parameters and their values
                new_ex = {key:curr_ex[key] for key in all_params
                        if (key not in (keys_matter+[key_verbtype]))}
                #Add in passive term for verbtypes
                tmp_vals = list(curr_ex[key_verbtype]) #+ ["PASSIVE"]
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
    def _categorize_verb(self, verb):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        list_category_names = config.list_category_names
        list_category_synsets = config.list_category_synsets
        list_category_threses = config.list_category_threses
        max_hyp = config.max_num_hypernyms
        #root_hypernyms = wordnet.synsets(verb, pos=wordnet.VERB)
        if (max_hyp is None):
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

        ##Handle specialty verbs
        #For 'be' verbs
        if any([(roothyp in config.synsets_verbs_be)
                    for roothyp in root_hypernyms]):
            return "be"
        #For 'has' verbs
        elif any([(roothyp in config.synsets_verbs_has)
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
            if do_verbose:
                print("No categories fit verb: {0}, {1}\n"
                                .format(verb, score_fins))
            return None
        #

        ##Throw an error if this verb gives very similar top scores
        thres = config.thres_category_fracdiff
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
    def _classify_statements(self, forest, do_verbose=None):
        #Extract global variables
        if do_verbose is not None: #Override do_verbose if specified for now
            self._store_info(do_verbose, "do_verbose")
        #Load the fixed decision tree
        decision_tree = self._get_info("decision_tree")
        #Print some notes
        if do_verbose:
            print("\n> Running _classify_statements.")
            #print("Forest word-trees:")
            #for key1 in forest:
            #    for key2 in forest[key1]:
            #        print("Key1, Key2: {0}, {1}".format(key1, key2))
            #        print("{0}".format(forest[key1][key2
            #                            ]["struct_words_updated"].keys()))
            #        print("{0}\n-".format(forest[key1][key2
            #                            ]["struct_words_updated"]))
        #

        #Iterate through and score clauses
        list_scores = [[] for ii in range(0, len(forest))]
        list_components = [[] for ii in range(0, len(forest))]
        for ii in range(0, len(forest)):
            for jj in range(0, len(forest[ii])):
                curr_info = forest[ii][jj]
                #Convert current clause into rules
                curr_rules_raw = [item for key in curr_info["clauses"]["text"]
                            for item in
                            self._convert_clause_into_rule(
                            clause_text=curr_info["clauses"]["text"][key],
                            clause_ids=curr_info["clauses"]["ids"][key],
                            flags_nounchunks=curr_info["flags"],
                            ids_nounchunks=curr_info["ids_nounchunk"])]
                curr_rules = [self._merge_rules(curr_rules_raw)]
                #Fetch and store score of each rule
                tmp_res = [self._apply_decision_tree(rule=item,
                                                decision_tree=decision_tree)
                                for item in curr_rules]
                list_scores[ii] += [item["scores"] for item in tmp_res
                                    if (item is not None)]
                list_components[ii] += [item["components"] for item in tmp_res
                                    if (item is not None)]
        #

        #Combine scores across clauses
        #comb_score = self._combine_scores(list_scores)
        comb_score = [item2 for item1 in list_scores for item2 in item1]

        #Convert final score into verdict and other information
        results = self._convert_score_to_verdict(comb_score)
        results["indiv_scores"] = list_scores
        results["indiv_components"] = list_components

        ##Return the dictionary containing verdict, etc. for these statements
        return results
    #

    ##Method: classify_text
    ##Purpose: Classify full text based on its statements (rule approach)
    def classify_text(self, keyword_obj, do_check_truematch, which_mode=None, forest=None, text=None, buffer=0, do_verbose=None):
        #Fetch global variables
        if do_verbose is None:
            do_verbose = self._get_info("do_verbose")
        do_verbose_deep = self._get_info("do_verbose_deep")
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
        dict_results = self._classify_statements(forest, do_verbose=do_verbose)

        #Print some notes
        if do_verbose:
            print("Verdicts complete.")
            print("Verdict dictionary:\n{0}".format(dict_results))
            print("---")

        #Return final verdicts
        return dict_results #dict_verdicts
    #

    ##Method: _convert_scorestoverdict
    ##Purpose: Convert set of decision tree scores into single verdict
    def x_convert_score_to_verdict(self, dict_scores_indiv, max_diff_thres=0.10, max_diff_count=3, max_diff_verdicts=["science"]):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        #Print some notes
        if do_verbose:
            print("\n> Running _convert_scorestoverdict.")
            print("Individual components and score sets:")
            for ii in range(0, len(dict_scores_indiv)):
                print("{0}\n-".format(#components[ii],
                                            dict_scores_indiv[ii]))
            #
        #

        ##Return empty verdict if empty scores
        #For completely empty scores
        if len(dict_scores_indiv) == 0:
            tmp_res = config.dictverdict_error.copy()
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
                                    ) #Check if rel. max. key by some thres.
                #
                else:
                    tmp_compare = False
                #

                #Increment count of sentences with max-valued verdict
                if tmp_compare:
                    dict_results[curr_key]["count_max"] += 1
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

        #Gather final scores into set of error
        dict_error = {key:dict_results[key]["score_tot_norm"]
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

        ##Establish uncertainties from scores
        dict_uncertainties = {all_keys[ii]:list_scores_comb[ii]
                                for ii in range(0, len(all_keys))}
        #

        ##Determine best verdict and associated probabilistic error
        is_found = False
        #For max sentence count:
        if (not is_found) and (max_diff_thres is not None):
            #Check allowed keys that fit max condition
            for curr_key in max_diff_verdicts:
                if (dict_results[curr_key]["count_max"] >= max_diff_count):
                    is_found = True
                    max_score = 1 #dict_results[curr_key]["score_tot_norm"]
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
            #Return low-prob verdict if multiple equal top probabilities
            if (list_scores_comb.count(max_score) > 1):
                tmp_res = config.dictverdict_lowprob.copy()
                tmp_res["scores_indiv"] = dict_scores_indiv
                tmp_res["uncertainty"] = dict_uncertainties
                #tmp_res["components"] = components
                #Print some notes
                if do_verbose:
                    print("-Multiple top prob. scores.")
                    print("Returning low-prob verdict:\n{0}".format(tmp_res))
                #
                return tmp_res
            #
        #

        ##Establish uncertainty for max verdict from scores
        dict_uncertainties[max_verdict] = max_score
        #

        ##Assemble and return final verdict
        fin_res = {"verdict":max_verdict, "scores_indiv":dict_scores_indiv,
                "uncertainty":dict_uncertainties, #"components":components,
                "norm_error":dict_error}
        #
        #Print some notes
        if do_verbose:
            print("-Returning final verdict dictionary:\n{0}".format(fin_res))
        #
        return fin_res
    #

    ##Method: _convert_scorestoverdict
    ##Purpose: Convert set of decision tree scores into single verdict
    #def _convert_score_to_verdict(self, dict_scores_indiv, max_diff_thres=0.10, max_diff_count=3, max_diff_verdicts=["science"], thres_override_acceptance=1.0, order_override_acceptance=["science"], "data_influenced"]):
    #If this doesn't work, revert back and use split uncertainty thresholds for rule-based classifs (e.g., more lenient for mentions)
    def _convert_score_to_verdict(self, dict_scores_indiv, count_for_override=1, thres_override_acceptance=1.0, thres_indiv_score=0.7, weight_for_override=0.75, uncertainty_for_override=0.80, order_override_acceptance=["science", "data_influenced"], dict_weights={"science":1, "data_influenced":1, "mention":1}): #, "data_influenced"]):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        #Print some notes
        if do_verbose:
            print("\n> Running _convert_scorestoverdict.")
            print("Individual components and score sets:")
            for ii in range(0, len(dict_scores_indiv)):
                print("{0}\n-".format(#components[ii],
                                            dict_scores_indiv[ii]))
            #
        #

        ##Return empty verdict if empty scores
        #For completely empty scores
        if len(dict_scores_indiv) == 0:
            tmp_res = config.dictverdict_error.copy()
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
        #Initialize container for accumulated scores
        dict_results = {key:{"score_tot_unnorm":0, "count_thiskey":0}
                        for key in all_keys}
        #
        #Iterate through individual scores and accumulate them
        for ii in range(0, num_indiv):
            curr_scores = dict_scores_indiv[ii]
            #Iterate through classif scores and accumulate total score
            for curr_key in all_keys:
                tmp_unnorm = curr_scores[curr_key]
                #Increment unnorm. score count
                dict_results[curr_key]["score_tot_unnorm"] += tmp_unnorm
            #
            #Determine and store classif with max-score, if above threshold
            curr_maxkey = max(curr_scores, key=curr_scores.get) #Max of indiv.
            if (curr_scores[curr_maxkey] >= thres_indiv_score): #If above thres.
                dict_results[curr_maxkey]["count_thiskey"] += 1
            #
        #

        #Normalize and store the scores
        denom = np.sum([dict_results[key]["score_tot_unnorm"]
                        for key in all_keys])
        denom_weighted = np.sum([(dict_results[key]["score_tot_unnorm"]
                                    * dict_weights[key])
                                for key in all_keys])
        for curr_key in all_keys:
            #Calculate and store normalized score
            if (denom > 0):
                tmp_score = (dict_results[curr_key]["score_tot_unnorm"] / denom)
                tmp_score_weighted = (dict_results[curr_key]["score_tot_unnorm"]
                                        * dict_weights[curr_key]
                                        / denom_weighted)
            else:
                tmp_score = 0
                tmp_score_weighted = 0
            #
            dict_results[curr_key]["score_tot_norm"] = tmp_score
            dict_results[curr_key]["score_tot_fin"] = tmp_score
            dict_results[curr_key]["score_tot_weighted"] = tmp_score_weighted
        #

        #Normalize the count of individual scores with max verdicts as well
        denom = sum([dict_results[key]["count_thiskey"] for key in all_keys])
        for curr_key in all_keys:
            #Calculate and store normalized count
            if (denom > 0):
                dict_results[curr_key]["count_norm"] = (
                            dict_results[curr_key]["count_thiskey"] / denom)
            else:
                dict_results[curr_key]["count_norm"] = 0
        #

        #Override combined scores with absolutes if requested
        tmp_max_key = None
        if ((thres_override_acceptance is not None)
                            and (order_override_acceptance is not None)):
            for curr_key in order_override_acceptance:
                if (sum([(item[curr_key] >= thres_override_acceptance)
                                for item in dict_scores_indiv])
                            >= count_for_override):
                    #Override scores
                    tmp_max_key = curr_key
                    break
        #
        if (tmp_max_key is not None):
            for curr_key in all_keys:
                if (curr_key == tmp_max_key):
                    dict_results[curr_key]["score_tot_fin"] = 1.0
                else:
                    dict_results[curr_key]["score_tot_fin"] = 0.0
        #

        #Combined score storage
        list_scores_comb = [dict_results[key]["score_tot_fin"]
                            for key in all_keys]
        #

        #Gather final scores into set of error
        dict_error = {key:dict_results[key]["score_tot_norm"]
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

        ##Establish uncertainties from scores
        #Uncertainty component 1: normalized count
        #Uncertainty component 2: normalized total score
        #
        if (tmp_max_key is None):
            #Average for the overall uncertainty, if no override
            #dict_uncertainties = {
            #        key:np.mean([dict_results[key]["count_norm"],
            #                        dict_results[key]["score_tot_fin"]])
            #        for key in all_keys}
            #dict_uncertainties = {
            #        key:dict_results[key]["count_norm"]
            #        for key in all_keys}
            #dict_uncertainties = {
            #        key:dict_results[key]["score_tot_fin"]
            #        for key in all_keys}
            dict_uncertainties = {
                    key:dict_results[key]["score_tot_weighted"]
                    for key in all_keys}
            #dict_uncertainties = {
            #        key:np.mean([dict_results[key]["count_norm"],
            #                    dict_results[key]["score_tot_weighted"]])
            #        for key in all_keys}
        #
        #Otherwise, splice in fixed max. uncertainty
        else:
            #Set fixed uncertainty for overriden classif.
            dict_uncertainties = {tmp_max_key:uncertainty_for_override}
            #Scale other uncertainties to mesh with fixed override uncertainty
            tmp_subset = sum([dict_results[key]["score_tot_fin"]
                                for key in all_keys
                                if (key != tmp_max_key)])
            #Splice in scaled uncertainties
            if (tmp_subset != 0): #Avoid division by zero
                tmp_dict = {key:(dict_results[key]["score_tot_fin"]
                                                / tmp_subset
                                                * (1-uncertainty_for_override))
                                            for key in all_keys
                                            if (key != tmp_max_key)}
            else: #If zero denominator, just set to 0
                tmp_dict = {key:0 for key in all_keys if (key != tmp_max_key)}
            dict_uncertainties.update(tmp_dict)
        #
        #
        #Block: 2024-03-02a
        #Weight the score metric more if absolute override was applied
        #if (tmp_max_key is None):
        #    tmp_weight = 0.5
        #else:
        #    tmp_weight = weight_for_override
        #
        #Average for the overall uncertainty
        #dict_uncertainties = {
        #        key:np.average([dict_results[key]["count_norm"],
        #                        dict_results[key]["score_tot_fin"]],
        #                        weights=[(1-tmp_weight), tmp_weight])
        #        for key in all_keys}
        #
        #
        #
        #dict_uncertainties = {all_keys[ii]:list_scores_comb[ii]
        #                        for ii in range(0, len(all_keys))}
        #
        #Uncertainty components: percentage of sentences that are counted as verdict; the normalized scores
        #Example: text with: 3 mentions (80% prob), 1 science (40% prob as mention)
        #So maybe unnorm. certainty should be 75% and 80%?

        ##Determine best verdict and associated probabilistic error
        #is_found = False

        #For max normalized total score:
        #if (not is_found):
        max_ind = np.argmax(list_scores_comb)
        #max_score = list_scores_comb[max_ind]
        max_verdict = all_keys[max_ind]
        #Print some notes
        if do_verbose:
            #print("\n-Max score: {0}".format(max_score))
            print("Max verdict: {0}\n".format(max_verdict))
        #
        #Return low-prob verdict if multiple equal top probabilities
        #if (list_scores_comb.count(max_score) > 1):
        if (list_scores_comb.count(list_scores_comb[max_ind]) > 1):
            tmp_res = config.dictverdict_lowprob.copy()
            tmp_res["scores_indiv"] = dict_scores_indiv
            tmp_res["uncertainty"] = dict_uncertainties
            #tmp_res["components"] = components
            #Print some notes
            if do_verbose:
                print("-Multiple top prob. scores.")
                print("Returning low-prob verdict:\n{0}".format(tmp_res))
            #
            return tmp_res
        #

        ##Establish uncertainty for max verdict from scores
        #dict_uncertainties[max_verdict] = max_score
        #

        ##Assemble and return final verdict
        fin_res = {"verdict":max_verdict, "scores_indiv":dict_scores_indiv,
                "uncertainty":dict_uncertainties, #"components":components,
                "norm_error":dict_error}
        #
        #Print some notes
        if do_verbose:
            print("-Returning final verdict dictionary:\n{0}".format(fin_res))
        #
        return fin_res
    #

    ##Method: _convert_scorestoverdict
    ##Purpose: Convert set of decision tree scores into single verdict
    #def _convert_score_to_verdict(self, dict_scores_indiv, max_diff_thres=0.10, max_diff_count=3, max_diff_verdicts=["science"], thres_override_acceptance=1.0, order_override_acceptance=["science"], "data_influenced"]):
    def old_beforeupdatestoscoringapproach_convert_score_to_verdict(self, dict_scores_indiv, count_for_override=2, thres_override_acceptance=1.0, order_override_acceptance=["science"]): #, "data_influenced"]):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        #Print some notes
        if do_verbose:
            print("\n> Running _convert_scorestoverdict.")
            print("Individual components and score sets:")
            for ii in range(0, len(dict_scores_indiv)):
                print("{0}\n-".format(#components[ii],
                                            dict_scores_indiv[ii]))
            #
        #

        ##Return empty verdict if empty scores
        #For completely empty scores
        if len(dict_scores_indiv) == 0:
            tmp_res = config.dictverdict_error.copy()
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
                """
                #Determine if current key has max. score across all keys
                if (max_diff_thres is not None) and (tmp_unnorm > 0):
                    tmp_compare = all([(
                                    (np.abs(tmp_unnorm - curr_scores[other_key])
                                                / curr_scores[other_key])
                                        >= max_diff_thres)
                                    for other_key in all_keys
                                    if (other_key != curr_key)]
                                    ) #Check if rel. max. key by some thres.
                #
                else:
                    tmp_compare = False
                #

                #Increment count of sentences with max-valued verdict
                if tmp_compare:
                    dict_results[curr_key]["count_max"] += 1
                #"""
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
            dict_results[curr_key]["score_tot_fin"] = tmp_score
        #

        #Override combined scores with absolutes if requested
        tmp_max_key = None
        if ((thres_override_acceptance is not None)
                            and (order_override_acceptance is not None)):
            for curr_key in order_override_acceptance:
                if (sum([(item[curr_key] >= thres_override_acceptance)
                                for item in dict_scores_indiv])
                            >= count_for_override):
                    #Override scores
                    tmp_max_key = curr_key
                    break
        #
        if (tmp_max_key is not None):
            for curr_key in all_keys:
                if (curr_key == tmp_max_key):
                    dict_results[curr_key]["score_tot_fin"] = 1.0
                else:
                    dict_results[curr_key]["score_tot_fin"] = 0.0
        #

        #Combined score storage
        list_scores_comb = [dict_results[key]["score_tot_fin"]
                            for key in all_keys]
        #

        #Gather final scores into set of error
        dict_error = {key:dict_results[key]["score_tot_norm"]
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

        ##Establish uncertainties from scores
        dict_uncertainties = {all_keys[ii]:list_scores_comb[ii]
                                for ii in range(0, len(all_keys))}
        #

        ##Determine best verdict and associated probabilistic error
        is_found = False
        """
        #For max sentence count:
        if (not is_found) and (max_diff_thres is not None):
            #Check allowed keys that fit max condition
            for curr_key in max_diff_verdicts:
                if (dict_results[curr_key]["count_max"] >= max_diff_count):
                    is_found = True
                    max_score = 1 #dict_results[curr_key]["score_tot_norm"]
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
        #"""

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
            #Return low-prob verdict if multiple equal top probabilities
            if (list_scores_comb.count(max_score) > 1):
                tmp_res = config.dictverdict_lowprob.copy()
                tmp_res["scores_indiv"] = dict_scores_indiv
                tmp_res["uncertainty"] = dict_uncertainties
                #tmp_res["components"] = components
                #Print some notes
                if do_verbose:
                    print("-Multiple top prob. scores.")
                    print("Returning low-prob verdict:\n{0}".format(tmp_res))
                #
                return tmp_res
            #
        #

        ##Establish uncertainty for max verdict from scores
        dict_uncertainties[max_verdict] = max_score
        #

        ##Assemble and return final verdict
        fin_res = {"verdict":max_verdict, "scores_indiv":dict_scores_indiv,
                "uncertainty":dict_uncertainties, #"components":components,
                "norm_error":dict_error}
        #
        #Print some notes
        if do_verbose:
            print("-Returning final verdict dictionary:\n{0}".format(fin_res))
        #
        return fin_res
    #

    ##Method: _find_missing_branches
    ##Purpose: Determine and return missing branches in decision tree
    def _find_missing_branches(self, do_verbose=None, cap_iter=None, print_freq=100):
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
        all_possible_values = config.dict_tree_possible_values
        solo_values = [None]
        all_matters = [item for item in all_possible_values["allmatter"]
                            if (item is not None)]
        all_verbclasses = [item for item in all_possible_values["verbclass"]
                            ] #No shallow copying!
        solo_verbtypes = ["PAST", "PRESENT", "FUTURE", "PURPOSE"]
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
            cap_matter = cap_iter
            #cap_vtypes = cap_iter
        else:
            cap_matter = 3 #len(all_subjmatters)
            cap_vtypes = len(multi_verbtypes)
        #

        #Generate set of all possible individual multi-parameter combinations
        sub_allmatters = [item for ii in range(1, (cap_matter+1))
                            for item in iterer.combinations(all_matters,ii)]
        sub_multiverbtypes = [item for ii in range(1, (cap_vtypes+1))
                            for item in iterer.combinations(multi_verbtypes,ii)]
        #
        if (cap_vtypes > 0):
            sub_allverbtypes = [(list(item_sub)+[item_solo])
                            for item_sub in sub_multiverbtypes
                            for item_solo in solo_verbtypes] #Fold in verb tense
        else:
            sub_allverbtypes = all_possible_values["verbtypes"]
        #

        #Fold in solo and required values as needed
        sets_verbclasses = all_verbclasses
        sets_allmatters = (sub_allmatters) # + [None])
        sets_allverbtypes = (sub_allverbtypes) # + [None])
        #
        if do_verbose:
            print("Combinations of branch parameters complete.")
            print(("Verb classes: {0}\nAll.matter: {1}\nVerbtypes: {2}")
                    .format(sets_verbclasses, sets_allmatters,
                            sets_allverbtypes))
        #

        #Generate set of all possible branches (all possible valid combos)
        list_branches = [{"allmatter":sets_allmatters[aa],
                        "verbclass":sets_verbclasses[cc],
                        "verbtypes":sets_allverbtypes[dd]}
                        for aa in range(0, len(sets_allmatters))
                        #for bb in range(0, len(sets_objmatters))
                        for cc in range(0, len(sets_verbclasses))
                        for dd in range(0, len(sets_allverbtypes))
                        if (((sets_allmatters[aa] is not None)
                                and ("is_keyword" in sets_allmatters[aa]))
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
        used_branch_ids = []
        for ii in range(0, num_branches):
            curr_branch = list_branches[ii]
            #Apply decision tree and ensure valid output
            try:
                tmp_res = self._apply_decision_tree(decision_tree=decision_tree,
                                                    rule=curr_branch)
                #
                bools_is_missing[ii] = False #Branch is covered
                used_branch_ids.append(tmp_res["id_branch"])
            #
            #Record this branch as missing if invalid output
            except NotImplementedError:
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

        #Streamline list of used branch ids
        used_branch_ids = sorted(list(set(used_branch_ids)))

        #Gather and return missing branches
        if do_verbose:
            print("All branches checked.\n{0} of {1} missing.\nUsed ids: {2}"
                    .format(np.sum(bools_is_missing), num_branches,
                            used_branch_ids))
            print("Run of _find_missing_branches() complete!")
        #
        return {"missing":
                (np.asarray(list_branches)[np.asarray(bools_is_missing)]),
                "unused":[decision_tree[key]
                            for key in sorted(list(decision_tree.keys()))
                            if (key not in used_branch_ids)]}
    #

    ##Method: _find_missing_branches
    ##Purpose: Determine and return missing branches in decision tree
    def old_beforecombinedmatter_find_missing_branches(self, do_verbose=None, cap_iter=None, print_freq=100):
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
        all_possible_values = config.dict_tree_possible_values
        solo_values = [None]
        all_subjmatters = [item for item in all_possible_values["subjectmatter"]
                            if (item is not None)]
        all_objmatters = [item for item in all_possible_values["objectmatter"]
                            if (item is not None)]
        all_verbclasses = [item for item in all_possible_values["verbclass"]
                            ] #No shallow copying!
        solo_verbtypes = ["PAST", "PRESENT", "FUTURE", "PURPOSE"]
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
            #cap_vtypes = cap_iter
        else:
            cap_subj = 2 #len(all_subjmatters)
            cap_obj = 2 #len(all_objmatters)
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
        if (cap_vtypes > 0):
            sub_allverbtypes = [(list(item_sub)+[item_solo])
                            for item_sub in sub_multiverbtypes
                            for item_solo in solo_verbtypes] #Fold in verb tense
        else:
            sub_allverbtypes = all_possible_values["verbtypes"]
        #

        #Fold in solo and required values as needed
        sets_verbclasses = all_verbclasses
        sets_subjmatters = (sub_subjmatters) # + [None])
        sets_objmatters = (sub_objmatters) # + [None])
        sets_allverbtypes = (sub_allverbtypes) # + [None])
        #
        if do_verbose:
            print("Combinations of branch parameters complete.")
            print(("Verb classes: {0}\nSubj.matter: {1}\nObj. matter: {2}"
                    +"\nVerbtypes: {3}")
                    .format(sets_verbclasses, sets_subjmatters,
                            sets_objmatters, sets_allverbtypes))
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
                                                    rule=curr_branch)
                #
                bools_is_missing[ii] = False #Branch is covered
            #
            #Record this branch as missing if invalid output
            except NotImplementedError:
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

    ##Method: _convert_clause_into_rule
    ##Purpose: Convert Grammar-made clause into rule for rule-based classifier
    def _convert_clause_into_rule(self, clause_text, clause_ids, flags_nounchunks, ids_nounchunks):
        """
        Method: _convert_clause_into_rule
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Process text into modifs using Grammar class.
        """
        #Set global variables
        do_verbose = self._get_info("do_verbose")

        #Fetch all sets of subject flags via their ids
        num_subj = len(clause_text["subjects"])
        sets_subjmatter = [] #[None]*num_subj #Container for subject flags
        for ii in range(0, num_subj): #Iterate through subjects
            id_chunk = ids_nounchunks[clause_ids["subjects"][ii][0]]
            sets_subjmatter += [key
                                    for key in flags_nounchunks[id_chunk]
                                    if ((flags_nounchunks[id_chunk][key])
                                        and (key != "is_any"))
                                    ] #Store all subj. flags for this clause
        #
        #If no subjects, set container to empty
        if (num_subj == 0):
            sets_subjmatter = []
        #
        sets_subjmatter = list(set(sets_subjmatter))

        #Fetch all sets of object flags via their ids
        tmp_list = (clause_ids["dir_objects"] + clause_ids["prep_objects"])
        num_obj = len(tmp_list)
        sets_objmatter = [] #[None]*num_obj #Container for object flags
        for ii in range(0, num_obj): #Iterate through objects
            id_chunk = ids_nounchunks[tmp_list[ii][0]]
            sets_objmatter += [key
                                    for key in flags_nounchunks[id_chunk]
                                    if ((flags_nounchunks[id_chunk][key])
                                        and (key != "is_any"))
                                    ] #Store all obj. flags for this clause
        #
        #If no objects, set container to empty
        if (num_obj == 0):
            sets_objmatter = []
        #
        sets_objmatter = list(set(sets_objmatter))

        #Combine the subject and object matter
        sets_allmatter = list(set((sets_subjmatter + sets_objmatter)))

        #Set verb class
        #verbclass = self._categorize_verb(verb=clause_text["verb"])
        verbclass_raw = self._categorize_verb(verb=clause_text["verb"])
        if (verbclass_raw is None):
            verbclass = []
        elif isinstance(verbclass_raw, str):
            verbclass = [verbclass_raw]
        else:
            verbclass = verbclass_raw
        #

        #Set verb type
        #verbtype = clause_text["verbtype"]
        verbtype_raw = clause_text["verbtype"]
        if (verbtype_raw is None):
            verbtype = []
        elif isinstance(verbtype_raw, str):
            verbtype = [verbtype_raw]
        else:
            verbtype = verbtype_raw
        #

        #Set rules from combinations of subj. and obj.matter
        rules = [{"allmatter":sets_allmatter,
                            #"objectmatter":sets_objmatter,
                            "verbclass":verbclass, "verbtypes":verbtype}]
        #

        #Print some notes
        if do_verbose:
            print("\n> Run of _convert_clause_into_rule complete.")
            print("Orginal clause:\n{0}".format(clause_text))
            print("Extracted rules:")
            for ii in range(0, len(rules)):
                print("{0}: {1}\n-".format(ii, rules[ii]))

        #Return the assembled rules
        return rules
    #

    ##Method: _convert_clause_into_rule
    ##Purpose: Convert Grammar-made clause into rule for rule-based classifier
    def x_convert_clause_into_rule(self, clause_text, clause_ids, flags_nounchunks, ids_nounchunks):
        """
        Method: _convert_clause_into_rule
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Process text into modifs using Grammar class.
        """
        #Set global variables
        do_verbose = self._get_info("do_verbose")

        #Fetch all sets of subject flags via their ids
        num_subj = len(clause_text["subjects"])
        sets_subjmatter = [None]*num_subj #Container for subject flags
        for ii in range(0, num_subj): #Iterate through subjects
            id_chunk = ids_nounchunks[clause_ids["subjects"][ii][0]]
            sets_subjmatter[ii] = [key
                                    for key in flags_nounchunks[id_chunk]
                                    if ((flags_nounchunks[id_chunk][key])
                                        and (key != "is_any"))
                                    ] #Store all subj. flags for this clause
        #
        #If no subjects, set container to empty
        if (num_subj == 0):
            sets_subjmatter = [[]]
        #

        #Fetch all sets of object flags via their ids
        tmp_list = (clause_ids["dir_objects"] + clause_ids["prep_objects"])
        num_obj = len(tmp_list)
        sets_objmatter = [None]*num_obj #Container for object flags
        for ii in range(0, num_obj): #Iterate through objects
            id_chunk = ids_nounchunks[tmp_list[ii][0]]
            sets_objmatter[ii] = [key
                                    for key in flags_nounchunks[id_chunk]
                                    if ((flags_nounchunks[id_chunk][key])
                                        and (key != "is_any"))
                                    ] #Store all obj. flags for this clause
        #
        #If no objects, set container to empty
        if (num_obj == 0):
            sets_objmatter = [[]]
        #

        #Set verb class
        #verbclass = self._categorize_verb(verb=clause_text["verb"])
        verbclass_raw = self._categorize_verb(verb=clause_text["verb"])
        if (verbclass_raw is None):
            verbclass = []
        elif isinstance(verbclass_raw, str):
            verbclass = [verbclass_raw]
        else:
            verbclass = verbclass_raw
        #

        #Set verb type
        #verbtype = clause_text["verbtype"]
        verbtype_raw = clause_text["verbtype"]
        if (verbtype_raw is None):
            verbtype = []
        elif isinstance(verbtype_raw, str):
            verbtype = [verbtype_raw]
        else:
            verbtype = verbtype_raw
        #

        #Set rules from combinations of subj. and obj.matter
        rules = []
        for ii in range(0, len(sets_subjmatter)):
            for jj in range(0, len(sets_objmatter)):
                curr_rule = {"subjectmatter":sets_subjmatter[ii],
                            "objectmatter":sets_objmatter[jj],
                            "verbclass":verbclass, "verbtypes":verbtype}
                #

                #Store rules
                rules.append(curr_rule)
            #
        #

        #Print some notes
        if do_verbose:
            print("\n> Run of _convert_clause_into_rule complete.")
            print("Orginal clause:\n{0}".format(clause_text))
            print("Extracted rules:")
            for ii in range(0, len(rules)):
                print("{0}: {1}\n-".format(ii, rules[ii]))

        #Return the assembled rules
        return rules
    #

    ##Method: _merge_rules
    ##Purpose: Merge list of rules into one rule
    def _merge_rules(self, rules):
        #Extract global variables
        keys_matter = config.nest_keys_matter
        #Extract keys of rules
        list_keys = rules[0].keys()
        #Merge rule across all keys
        merged_rule_raw = {key:[] for key in list_keys}
        for ii in range(0, len(rules)):
            #Merge if any important terms in this rule
            if any([(len(rules[ii][key]) > 0) for key in keys_matter]):
                for curr_key in list_keys:
                    merged_rule_raw[curr_key] += rules[ii][curr_key]
        #
        #Keep only unique values
        merged_rule = {key:list(set(merged_rule_raw[key])) for key in list_keys}
        #
        #Return merged rule
        return merged_rule
    #
#


##Class: Classifier_Rules
class orig_Classifier_Rules(_Classifier):
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
        self._store_info(do_verbose, "do_verbose")
        self._store_info(do_verbose_deep, "do_verbose_deep")
        if (which_classifs is None):
            which_classifs = config.list_default_verdicts_decisiontree
        self._store_info(which_classifs, "class_names")

        ##Assemble the fixed decision tree
        decision_tree = self._assemble_decision_tree()
        self._store_info(decision_tree, "decision_tree")

        ##Print some notes
        if do_verbose:
            print("> Initialized instance of Classifier_Rules class.")
            print("Internal decision tree has been assembled.")
            print("NOTE: Decision tree probabilities:\n{0}\n"
                    .format(decision_tree))
        #

        ##Nothing to see here
        return
    #

    ##Method: _apply_decision_tree
    ##Purpose: Apply a decision tree to a 'nest' dictionary for some text
    def _apply_decision_tree(self, decision_tree, rule):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        which_classifs = self._get_info("class_names")
        keys_main = config.nest_keys_main
        keys_matter = [item for item in keys_main if (item.endswith("matter"))]
        bool_keyword = "is_keyword"
        prefix = "prob_"
        #
        rule_mod = rule.copy()
        for curr_key in keys_main: #Encapsulate any single values into lists
            if (isinstance(rule[curr_key], str) or (rule[curr_key] is None)):
                rule_mod[curr_key] = [rule[curr_key]]
            else:
                rule_mod[curr_key] = rule[curr_key]
        #
        #Print some notes
        if do_verbose:
            print("Applying decision tree to the following rule:")
            print(rule_mod)
        #

        ##Ignore this rule if no keywords within
        if (not any([bool_keyword in rule_mod[key] for key in keys_matter])):
            #Print some notes
            if do_verbose:
                print("No keywords in this rule. Skipping.")
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
                                        np.sort(rule_mod[key_param])):
                        is_match = False
                        break #Exit from this branch early
                #
                #Otherwise, check inclusive and excluded ('!') matching values
                elif isinstance(curr_branch[key_param], list):
                    #Check for included matching values
                    if ((not all([(item in rule_mod[key_param])
                                for item in curr_branch[key_param]
                                if (not item.startswith("!"))]))
                            or (any([(item in rule_mod[key_param])
                                        for item in curr_branch[key_param]
                                        if (item.startswith("!"))]))):
                        is_match = False
                        break #Exit from this branch early
                #
                #Otherwise, check if any of allowed values contained
                elif isinstance(curr_branch[key_param], set):
                    #Check for any of matching values
                    if not any([(item in rule_mod[key_param])
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
            raise NotImplementedError("Err: No match found for {0}!"
                                        .format(rule_mod))
        #

        ##Extract the scores from the branch
        dict_scores = {key:best_branch[prefix+key] for key in which_classifs}

        ##Return the final scores
        if do_verbose:
            print("Final scores computed for rule:\n{0}\n\n{1}\n\n{2}"
                    .format(rule_mod, dict_scores, best_branch))
        #
        return {"scores":dict_scores, "components":best_branch}
    #

    ##Method: _assemble_decision_tree
    ##Purpose: Assemble base of decision tree, with probabilities, that can be read from/expanded as full decision tree
    def _assemble_decision_tree(self):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        which_classifs = self._get_info("class_names")
        dict_possible_values = config.dict_tree_possible_values
        #dict_valid_combos = config.dict_tree_valid_value_combinations
        keys_matter = config.nest_keys_matter
        key_verbtype = config.nest_key_verbtype
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
                "prob_science":0.9,
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
                "prob_data_influenced":0.4,
                "prob_mention":0.5}
            #
            #’The data is from OBJ by Authorsetal.’
            #itrack += 1
            #dict_examples_base[str(itrack)] = {
            #    "subjectmatter":tuple([]),
            #    "objectmatter":tuple(["is_keyword", "is_etal"]),
            #    "verbclass":{"be"},
            #    "verbtypes":{"PRESENT"},
            #    "prob_science":0.0,
            #    "prob_data_influenced":0.8,
            #    "prob_mention":0.6}
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
                "prob_science":0.9,
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
                "prob_science":0.9,
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
                "prob_mention":0.0}
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
                "prob_mention":0.0}
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
                "prob_data_influenced":1.0,
                "prob_mention":0.0}
            #
            #’We simulate/simulated the OBJ data of Authorsetal.’
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_pron_1st"]),
                "objectmatter":tuple(["is_keyword", "is_etal"]),
                "verbclass":{"datainfluenced"},
                "verbtypes":{"PRESENT", "PAST"},
                "prob_science":0.0,
                "prob_data_influenced":1.0,
                "prob_mention":0.0}
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
            #is_key; is_fig, !is_etal, !is_they
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword"]),
                "objectmatter":["is_term_fig", "!is_etal", "!is_pron_3rd"],
                "verbclass":"is_any", #{"be", "has", "plot", "science"},
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.9,
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
                "prob_science":0.9,
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
                "prob_science":0.9,
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
            #<Keyword figure be/has/plot/science/datainfl. by etal and pron_3rd>
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword", "is_term_fig"]),
                "objectmatter":["is_etal"],
                "verbclass":"is_any",
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.0,
                "prob_data_influenced":1.0,
                "prob_mention":0.0}
            #
            #<Keyword figure be/has/plot/science/datainfl. by etal and pron_3rd>
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword", "is_term_fig"]),
                "objectmatter":["is_pron_3rd"],
                "verbclass":"is_any",
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.0,
                "prob_data_influenced":0.9,
                "prob_mention":0.1}
            #
            #<Keyword figure be/has/plot/science/datainfl. by etal and pron_3rd>
            itrack += 1
            dict_examples_base[str(itrack)] = {
                "subjectmatter":tuple(["is_keyword", "is_term_fig"]),
                "objectmatter":["is_etal", "is_pron_3rd"],
                "verbclass":"is_any",
                "verbtypes":{"PAST", "PRESENT"},
                "prob_science":0.0,
                "prob_data_influenced":1.0,
                "prob_mention":0.1}
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

            ##For passive example
            #For general (not is_any, not set) case
            if ((curr_ex[key_verbtype] == "is_any")):
                #                    or (curr_ex[key_verbtype] is None)):
                pass #No passive counterpart necessary for is_any case
            #For set, add example+passive for each entry in set
            #elif isinstance(curr_ex[key_verbtype], set):
                #Iterate through set entries
            #    for curr_val in curr_ex[key_verbtype]:
                    #Passive with flipped subj-obj
                    #Extract all parameters and their values
            #        new_ex = {key:curr_ex[key] for key in all_params
            #                if (key not in (keys_matter+[key_verbtype]))}
                    #Add in passive term for verbtypes
            #        tmp_vals = [curr_val, "PASSIVE"]
            #        new_ex[key_verbtype] = tmp_vals
                    #Flip the subject and object terms
            #        new_ex["subjectmatter"] = curr_ex["objectmatter"]
            #        new_ex["objectmatter"] = curr_ex["subjectmatter"]
                    #Normalize and store probabilities for target classifs
            #        new_ex.update(curr_probs)
                    #Store this example
            #        itrack += 1
            #        decision_tree[itrack] = new_ex
            else:
                #Extract all parameters and their values
                new_ex = {key:curr_ex[key] for key in all_params
                        if (key not in (keys_matter+[key_verbtype]))}
                #Add in passive term for verbtypes
                tmp_vals = list(curr_ex[key_verbtype]) #+ ["PASSIVE"]
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
    def _categorize_verb(self, verb):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        list_category_names = config.list_category_names
        list_category_synsets = config.list_category_synsets
        list_category_threses = config.list_category_threses
        max_hyp = config.max_num_hypernyms
        #root_hypernyms = wordnet.synsets(verb, pos=wordnet.VERB)
        if (max_hyp is None):
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

        ##Handle specialty verbs
        #For 'be' verbs
        if any([(roothyp in config.synsets_verbs_be)
                    for roothyp in root_hypernyms]):
            return "be"
        #For 'has' verbs
        elif any([(roothyp in config.synsets_verbs_has)
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
            if do_verbose:
                print("No categories fit verb: {0}, {1}\n"
                                .format(verb, score_fins))
            return None
        #

        ##Throw an error if this verb gives very similar top scores
        thres = config.thres_category_fracdiff
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
    def _classify_statements(self, forest, do_verbose=None):
        #Extract global variables
        if do_verbose is not None: #Override do_verbose if specified for now
            self._store_info(do_verbose, "do_verbose")
        #Load the fixed decision tree
        decision_tree = self._get_info("decision_tree")
        #Print some notes
        if do_verbose:
            print("\n> Running _classify_statements.")
            #print("Forest word-trees:")
            #for key1 in forest:
            #    for key2 in forest[key1]:
            #        print("Key1, Key2: {0}, {1}".format(key1, key2))
            #        print("{0}".format(forest[key1][key2
            #                            ]["struct_words_updated"].keys()))
            #        print("{0}\n-".format(forest[key1][key2
            #                            ]["struct_words_updated"]))
        #

        #Iterate through and score clauses
        list_scores = [[] for ii in range(0, len(forest))]
        list_components = [[] for ii in range(0, len(forest))]
        for ii in range(0, len(forest)):
            for jj in range(0, len(forest[ii])):
                curr_info = forest[ii][jj]
                #Convert current clause into rules
                curr_rules = [item for key in curr_info["clauses"]["text"]
                            for item in
                            self._convert_clause_into_rule(
                            clause_text=curr_info["clauses"]["text"][key],
                            clause_ids=curr_info["clauses"]["ids"][key],
                            flags_nounchunks=curr_info["flags"],
                            ids_nounchunks=curr_info["ids_nounchunk"])]
                #Fetch and store score of each rule
                tmp_res = [self._apply_decision_tree(rule=item,
                                                decision_tree=decision_tree)
                                for item in curr_rules]
                list_scores[ii] += [item["scores"] for item in tmp_res
                                    if (item is not None)]
                list_components[ii] += [item["components"] for item in tmp_res
                                    if (item is not None)]
        #

        #Combine scores across clauses
        #comb_score = self._combine_scores(list_scores)
        comb_score = [item2 for item1 in list_scores for item2 in item1]

        #Convert final score into verdict and other information
        results = self._convert_score_to_verdict(comb_score)
        results["indiv_scores"] = list_scores
        results["indiv_components"] = list_components

        ##Return the dictionary containing verdict, etc. for these statements
        return results
    #

    ##Method: classify_text
    ##Purpose: Classify full text based on its statements (rule approach)
    def classify_text(self, keyword_obj, do_check_truematch, which_mode=None, forest=None, text=None, buffer=0, do_verbose=None):
        #Fetch global variables
        if do_verbose is None:
            do_verbose = self._get_info("do_verbose")
        do_verbose_deep = self._get_info("do_verbose_deep")
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
        dict_results = self._classify_statements(forest, do_verbose=do_verbose)

        #Print some notes
        if do_verbose:
            print("Verdicts complete.")
            print("Verdict dictionary:\n{0}".format(dict_results))
            print("---")

        #Return final verdicts
        return dict_results #dict_verdicts
    #

    ##Method: _convert_scorestoverdict
    ##Purpose: Convert set of decision tree scores into single verdict
    def x_convert_score_to_verdict(self, dict_scores_indiv, max_diff_thres=0.10, max_diff_count=3, max_diff_verdicts=["science"]):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        #Print some notes
        if do_verbose:
            print("\n> Running _convert_scorestoverdict.")
            print("Individual components and score sets:")
            for ii in range(0, len(dict_scores_indiv)):
                print("{0}\n-".format(#components[ii],
                                            dict_scores_indiv[ii]))
            #
        #

        ##Return empty verdict if empty scores
        #For completely empty scores
        if len(dict_scores_indiv) == 0:
            tmp_res = config.dictverdict_error.copy()
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
                                    ) #Check if rel. max. key by some thres.
                #
                else:
                    tmp_compare = False
                #

                #Increment count of sentences with max-valued verdict
                if tmp_compare:
                    dict_results[curr_key]["count_max"] += 1
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

        #Gather final scores into set of error
        dict_error = {key:dict_results[key]["score_tot_norm"]
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

        ##Establish uncertainties from scores
        dict_uncertainties = {all_keys[ii]:list_scores_comb[ii]
                                for ii in range(0, len(all_keys))}
        #

        ##Determine best verdict and associated probabilistic error
        is_found = False
        #For max sentence count:
        if (not is_found) and (max_diff_thres is not None):
            #Check allowed keys that fit max condition
            for curr_key in max_diff_verdicts:
                if (dict_results[curr_key]["count_max"] >= max_diff_count):
                    is_found = True
                    max_score = 1 #dict_results[curr_key]["score_tot_norm"]
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
            #Return low-prob verdict if multiple equal top probabilities
            if (list_scores_comb.count(max_score) > 1):
                tmp_res = config.dictverdict_lowprob.copy()
                tmp_res["scores_indiv"] = dict_scores_indiv
                tmp_res["uncertainty"] = dict_uncertainties
                #tmp_res["components"] = components
                #Print some notes
                if do_verbose:
                    print("-Multiple top prob. scores.")
                    print("Returning low-prob verdict:\n{0}".format(tmp_res))
                #
                return tmp_res
            #
        #

        ##Establish uncertainty for max verdict from scores
        dict_uncertainties[max_verdict] = max_score
        #

        ##Assemble and return final verdict
        fin_res = {"verdict":max_verdict, "scores_indiv":dict_scores_indiv,
                "uncertainty":dict_uncertainties, #"components":components,
                "norm_error":dict_error}
        #
        #Print some notes
        if do_verbose:
            print("-Returning final verdict dictionary:\n{0}".format(fin_res))
        #
        return fin_res
    #

    ##Method: _convert_scorestoverdict
    ##Purpose: Convert set of decision tree scores into single verdict
    def _convert_score_to_verdict(self, dict_scores_indiv, max_diff_thres=0.10, max_diff_count=3, max_diff_verdicts=["science"], thres_override_acceptance=1.0, order_override_acceptance=["science", "data_influenced"]):
        ##Extract global variables
        do_verbose = self._get_info("do_verbose")
        #Print some notes
        if do_verbose:
            print("\n> Running _convert_scorestoverdict.")
            print("Individual components and score sets:")
            for ii in range(0, len(dict_scores_indiv)):
                print("{0}\n-".format(#components[ii],
                                            dict_scores_indiv[ii]))
            #
        #

        ##Return empty verdict if empty scores
        #For completely empty scores
        if len(dict_scores_indiv) == 0:
            tmp_res = config.dictverdict_error.copy()
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
                                    ) #Check if rel. max. key by some thres.
                #
                else:
                    tmp_compare = False
                #

                #Increment count of sentences with max-valued verdict
                if tmp_compare:
                    dict_results[curr_key]["count_max"] += 1
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
            dict_results[curr_key]["score_tot_fin"] = tmp_score
        #

        #Override combined scores with absolutes if requested
        tmp_max_key = None
        if ((thres_override_acceptance is not None)
                            and (order_override_acceptance is not None)):
            for curr_key in order_override_acceptance:
                if any([(item[curr_key] >= thres_override_acceptance)
                                for item in dict_scores_indiv]):
                    #Override scores
                    tmp_max_key = curr_key
                    break
        #
        if (tmp_max_key is not None):
            for curr_key in all_keys:
                if (curr_key == tmp_max_key):
                    dict_results[curr_key]["score_tot_fin"] = 1.0
                else:
                    dict_results[curr_key]["score_tot_fin"] = 0.0
        #

        #Combined score storage
        list_scores_comb = [dict_results[key]["score_tot_fin"]
                            for key in all_keys]
        #

        #Gather final scores into set of error
        dict_error = {key:dict_results[key]["score_tot_norm"]
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

        ##Establish uncertainties from scores
        dict_uncertainties = {all_keys[ii]:list_scores_comb[ii]
                                for ii in range(0, len(all_keys))}
        #

        ##Determine best verdict and associated probabilistic error
        is_found = False
        #For max sentence count:
        if (not is_found) and (max_diff_thres is not None):
            #Check allowed keys that fit max condition
            for curr_key in max_diff_verdicts:
                if (dict_results[curr_key]["count_max"] >= max_diff_count):
                    is_found = True
                    max_score = 1 #dict_results[curr_key]["score_tot_norm"]
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
            #Return low-prob verdict if multiple equal top probabilities
            if (list_scores_comb.count(max_score) > 1):
                tmp_res = config.dictverdict_lowprob.copy()
                tmp_res["scores_indiv"] = dict_scores_indiv
                tmp_res["uncertainty"] = dict_uncertainties
                #tmp_res["components"] = components
                #Print some notes
                if do_verbose:
                    print("-Multiple top prob. scores.")
                    print("Returning low-prob verdict:\n{0}".format(tmp_res))
                #
                return tmp_res
            #
        #

        ##Establish uncertainty for max verdict from scores
        dict_uncertainties[max_verdict] = max_score
        #

        ##Assemble and return final verdict
        fin_res = {"verdict":max_verdict, "scores_indiv":dict_scores_indiv,
                "uncertainty":dict_uncertainties, #"components":components,
                "norm_error":dict_error}
        #
        #Print some notes
        if do_verbose:
            print("-Returning final verdict dictionary:\n{0}".format(fin_res))
        #
        return fin_res
    #

    ##Method: _find_missing_branches
    ##Purpose: Determine and return missing branches in decision tree
    def _find_missing_branches(self, do_verbose=None, cap_iter=None, print_freq=100):
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
        all_possible_values = config.dict_tree_possible_values
        solo_values = [None]
        all_subjmatters = [item for item in all_possible_values["subjectmatter"]
                            if (item is not None)]
        all_objmatters = [item for item in all_possible_values["objectmatter"]
                            if (item is not None)]
        all_verbclasses = [item for item in all_possible_values["verbclass"]
                            ] #No shallow copying!
        solo_verbtypes = ["PAST", "PRESENT", "FUTURE", "PURPOSE"]
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
            #cap_vtypes = cap_iter
        else:
            cap_subj = 2 #len(all_subjmatters)
            cap_obj = 2 #len(all_objmatters)
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
        if (cap_vtypes > 0):
            sub_allverbtypes = [(list(item_sub)+[item_solo])
                            for item_sub in sub_multiverbtypes
                            for item_solo in solo_verbtypes] #Fold in verb tense
        else:
            sub_allverbtypes = all_possible_values["verbtypes"]
        #

        #Fold in solo and required values as needed
        sets_verbclasses = all_verbclasses
        sets_subjmatters = (sub_subjmatters) # + [None])
        sets_objmatters = (sub_objmatters) # + [None])
        sets_allverbtypes = (sub_allverbtypes) # + [None])
        #
        if do_verbose:
            print("Combinations of branch parameters complete.")
            print(("Verb classes: {0}\nSubj.matter: {1}\nObj. matter: {2}"
                    +"\nVerbtypes: {3}")
                    .format(sets_verbclasses, sets_subjmatters,
                            sets_objmatters, sets_allverbtypes))
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
                                                    rule=curr_branch)
                #
                bools_is_missing[ii] = False #Branch is covered
            #
            #Record this branch as missing if invalid output
            except NotImplementedError:
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

    ##Method: _convert_clause_into_rule
    ##Purpose: Convert Grammar-made clause into rule for rule-based classifier
    def _convert_clause_into_rule(self, clause_text, clause_ids, flags_nounchunks, ids_nounchunks):
        """
        Method: _convert_clause_into_rule
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Process text into modifs using Grammar class.
        """
        #Set global variables
        do_verbose = self._get_info("do_verbose")

        #Fetch all sets of subject flags via their ids
        num_subj = len(clause_text["subjects"])
        sets_subjmatter = [None]*num_subj #Container for subject flags
        for ii in range(0, num_subj): #Iterate through subjects
            id_chunk = ids_nounchunks[clause_ids["subjects"][ii][0]]
            sets_subjmatter[ii] = [key
                                    for key in flags_nounchunks[id_chunk]
                                    if ((flags_nounchunks[id_chunk][key])
                                        and (key != "is_any"))
                                    ] #Store all subj. flags for this clause
        #
        #If no subjects, set container to empty
        if (num_subj == 0):
            sets_subjmatter = [[]]
        #

        #Fetch all sets of object flags via their ids
        tmp_list = (clause_ids["dir_objects"] + clause_ids["prep_objects"])
        num_obj = len(tmp_list)
        sets_objmatter = [None]*num_obj #Container for object flags
        for ii in range(0, num_obj): #Iterate through objects
            id_chunk = ids_nounchunks[tmp_list[ii][0]]
            sets_objmatter[ii] = [key
                                    for key in flags_nounchunks[id_chunk]
                                    if ((flags_nounchunks[id_chunk][key])
                                        and (key != "is_any"))
                                    ] #Store all obj. flags for this clause
        #
        #If no objects, set container to empty
        if (num_obj == 0):
            sets_objmatter = [[]]
        #

        #Set verb characteristics
        verbclass = self._categorize_verb(verb=clause_text["verb"])
        verbtype = clause_text["verbtype"]

        #Set rules from combinations of subj. and obj.matter
        rules = []
        for ii in range(0, len(sets_subjmatter)):
            for jj in range(0, len(sets_objmatter)):
                curr_rule = {"subjectmatter":sets_subjmatter[ii],
                            "objectmatter":sets_objmatter[jj],
                            "verbclass":verbclass, "verbtypes":verbtype}
                #

                #Store rules
                rules.append(curr_rule)
            #
        #

        #Print some notes
        if do_verbose:
            print("\n> Run of _convert_clause_into_rule complete.")
            print("Orginal clause:\n{0}".format(clause_text))
            print("Extracted rules:")
            for ii in range(0, len(rules)):
                print("{0}: {1}\n-".format(ii, rules[ii]))

        #Return the assembled rules
        return rules
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
        - name [str (default="operator")]:
          - A unique name for this Operator.
        - load_check_truematch [bool (default=True)]:
          - Whether nor not to load external data for verifying ambiguous terms as true vs false matches to mission.
        - do_verbose [bool (default=False)]:
          - Whether or not to print surface-level log information and tests.
    """
    ##Method: __init__
    ##Purpose: Initialize this class instance
    def __init__(self, classifier, mode, keyword_objs, do_verbose, name="operator", load_check_truematch=True, do_verbose_deep=False):
        """
        Method: __init__
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Initializes instance of Operator class.
        """
        #Throw an error if an unallowed character in name
        if ("|" in name):
            raise ValueError("Please do not use the reserved character '|'"
                            +" in the name for your Operator.")
        #

        #Initialize storage
        self._storage = {}
        self._store_info(classifier, "classifier")
        self._store_info(mode, "mode")
        self._store_info(name, "name")
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
            if (keyword_objs[ii].identify_keyword(lookup)["bool"]):
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
    ##Purpose: Inspect text and either reject as false target or give classifications
    def classify(self, text, lookup, do_check_truematch, do_raise_innererror, modif=None, forest=None, buffer=0, do_verbose=None, do_verbose_deep=None):
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
        keyobj = self._fetch_keyword_object(lookup=lookup,do_verbose=do_verbose)
        if do_verbose:
            print("Best matching keyword object (keyobj) for keyword {0}:\n{1}"
                    .format(lookup, keyobj))
        #

        #Process text into modifs using Grammar class, if modif not given
        if (modif is None):
            if do_verbose:
                print("\nPreprocessing and extracting modifs from the text...")
            #
            if do_raise_innererror: #Allow stop from any exceptions encountered
                output = self.process(text=text,
                                do_check_truematch=do_check_truematch,
                                buffer=buffer, lookup=lookup,keyword_obj=keyobj,
                                do_verbose=do_verbose,
                                do_verbose_deep=do_verbose_deep)
            else: #Otherwise, print any exceptions and keep moving forward
                try:
                    output = self.process(text=text,
                                do_check_truematch=do_check_truematch,
                                buffer=buffer, lookup=lookup,keyword_obj=keyobj,
                                do_verbose=do_verbose,
                                do_verbose_deep=do_verbose_deep)
                #
                #Catch any exceptions and force-print some notes
                except Exception as err:
                    dict_verdicts = config.dictverdict_error.copy()
                    print("-\nThe following err. was encountered in operate:")
                    print(repr(err))
                    print("Error was noted. Returning error as verdict.\n-")
                    dict_verdicts["modif"] = ("<PROCESSING ERROR:\n{0}>"
                                                .format(repr(err)))
                    dict_verdicts["modif_none"] = None
                    return dict_verdicts
            #
            #Fetch the generated output
            modif = output["modif"]
            modif_none = output["modif_none"]
            forest = output["forest"]
            #
            #Print some notes
            if do_verbose:
                print("Text has been processed into modif.")
        #
        #Otherwise, use given modif
        else:
            #Print some notes
            if do_verbose:
                print("Modif given. No text processing will be done.")
            #
            modif_none = None
            pass
        #

        #Set not-classified verdict if flagged for no classification
        if (keyobj._get_info("do_not_classify")):
            dict_verdicts = config.dictverdict_donotclassify.copy()
        #
        #Set rejected verdict if empty text
        elif (modif.strip() == ""):
            if do_verbose:
                print("No text found matching keyword object.")
                print("Returning rejection verdict.")
            #
            dict_verdicts = config.dictverdict_rejection.copy()
        #
        #Classify the text using stored classifier with raised error
        elif do_raise_innererror: #If True, allow raising of inner errors
            dict_verdicts = classifier.classify_text(text=modif,
                                    do_check_truematch=do_check_truematch,
                                    forest=forest, keyword_obj=keyobj,
                                    do_verbose=do_verbose)
        #
        #Otherwise, run classification while ignoring inner errors
        else:
            #Try running classification
            try:
                dict_verdicts = classifier.classify_text(text=modif,
                                    do_check_truematch=do_check_truematch,
                                    forest=forest, keyword_obj=keyobj,
                                    do_verbose=do_verbose)
            #
            #Catch certain exceptions and force-print some notes
            except Exception as err:
                dict_verdicts = config.dictverdict_error.copy()
                print("-\nThe following err. was encountered in operate:")
                print(repr(err))
                print("Error was noted. Continuing.\n-")
        #

        #Return the verdict with modif included
        dict_verdicts["modif"] = modif
        dict_verdicts["modif_none"] = modif_none
        return dict_verdicts
    #

    ##Method: classify_set
    ##Purpose: Classify set of texts as false target or give classifications
    def classify_set(self, texts, do_check_truematch, do_raise_innererror, modifs=None, forests=None, buffer=0, print_freq=25, do_verbose=None, do_verbose_deep=None):
        """
        Method: classify_set
        Purpose:
          - !
          - Accept text and process it into modifs (using Grammar, Paper classes).
          - Classify that text (using stored classifier).
        Arguments:
          - !
          - do_verbose [bool (default=False)]:
            - Whether or not to print surface-level log information and tests.
          - do_verbose_deep [bool (default=False)]:
            - Whether or not to print inner log information and tests.
        Returns:
          - List of dicts, each containing:
            - 'modif': the modif.
            - 'verdict': the classification.
            - 'scores_comb': the final score.
            - 'scores_indiv': the individual scores.
            - 'uncertainty': the uncertainty of the classification.
        """
        ##Fetch global variables
        if (do_verbose is None):
            do_verbose = self._get_info("do_verbose")
        if (do_verbose_deep is None):
            do_verbose_deep = self._get_info("do_verbose_deep")
        #
        all_kobjs = self._get_info("keyword_objs")
        num_kobjs = len(all_kobjs)
        #
        #Throw error if both texts and modifs given
        if ((texts is not None) and (modifs is not None)):
            raise ValueError("Err: texts OR modifs should be given, not both.")
        elif (texts is not None):
            num_texts = len(texts)
        elif (modifs is not None):
            num_texts = len(modifs)
        #
        #Print some notes
        if do_verbose:
            print("\n> Running classify_set()!")
        #

        ##Classify every text against every mission
        list_results = [None]*num_texts
        curr_text = None
        curr_modif = None
        curr_forest = None
        #Iterate through texts
        for ii in range(0, num_texts):
            curr_dict = {} #Dictionary to hold set of results for this text
            list_results[ii] = curr_dict #Store this dictionary
            #
            #Extract current text if given in raw (not processed) form
            if (texts is not None):
                curr_text = texts[ii] #Current text to classify
            #
            #Extract current modifs and forests if already processed text
            if (modifs is not None):
                curr_modif = modifs[ii]
            #
            if (forests is not None):
                curr_forest = forests[ii]
            #
            #Iterate through keyword objects
            for jj in range(0, num_kobjs):
                curr_kobj = all_kobjs[jj]
                curr_name = curr_kobj._get_info("name")
                #Classify current text for current mission
                curr_result = self.classify(text=curr_text, lookup=curr_name,
                                        modif=curr_modif, forest=curr_forest,
                                        buffer=buffer,
                                        do_check_truematch=do_check_truematch,
                                        do_raise_innererror=do_raise_innererror,
                                        do_verbose=do_verbose_deep)
                #
                #Store current result
                curr_dict[curr_name] = curr_result
            #
            #Print some notes at given frequency, if requested
            if (do_verbose and ((ii % print_freq) == 0)):
                print("Classification for text #{0} of {1} complete..."
                        .format((ii+1), num_texts))
        #

        ##Return the classification results
        if do_verbose:
            print("\nRun of classify_set() complete!\n")
        #
        return list_results
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
        use_these_modes = list(set([mode, "none"]))
        grammar = Grammar(text=text, keyword_obj=keyword_obj,
                            do_check_truematch=do_check_truematch,
                            dict_ambigs=dict_ambigs,
                            do_verbose=do_verbose_deep, buffer=buffer)
        grammar.run_modifications(which_modes=use_these_modes)
        output = grammar.get_modifs(do_include_forest=True)
        modif = output["modifs"][mode]
        modif_none = output["modifs"]["none"] #Include unmodified vers. as well
        forest = output["_forest"]
        #
        #Print some notes
        if do_verbose:
            print("Text has been processed into modifs.")
        #

        #Return the modif and internal processing output
        return {"modif":modif, "modif_none":modif_none, "forest":forest}
    #

    ##Method: train_model_ML
    ##Purpose: Process text into modifs and then train ML model on the modifs
    def train_model_ML(self, dir_model, name_model, do_reuse_run, dict_texts, mapper, do_check_truematch, seed_TVT=10, seed_ML=8, buffer=0, fraction_TVT=[0.8, 0.1, 0.1], mode_TVT="uniform", do_shuffle=True, print_freq=25, do_verbose=None, do_verbose_deep=None):
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
          - do_check_truematch [bool]:
            - Whether or not to check that mission phrases found in text are known true vs. false matches. (E.g., 'Edwin Hubble' as false match for the Hubble Space Telescope).
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
        dataset = dict_texts
        classifier = self._get_info("classifier")
        folders_TVT = config.folders_TVT
        savename_ML = (config.tfoutput_prefix + name_model)
        savename_model = (name_model + ".npy")
        filepath_dicterrors = config.path_modiferrors
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
            str_err = None #Placeholder
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
                print("Processing text data into modifs...")
            #

            #Process each text within the database into a modif
            dict_errors = {} #Container for modifs with caught processing errors
            dict_modifs = {} #Container for modifs and text classification info
            i_track = 0
            i_skipped = 0
            str_err = ""
            num_data = len(dataset)
            for curr_key in dataset:
                old_dict = dataset[curr_key]
                masked_class = mapper[old_dict["class"].lower()]
                #
                #Extract modif for current text
                if do_check_truematch: #Catch and print unknown ambig. phrases
                    try:
                        curr_res = self.process(text=old_dict["text"],
                                    do_check_truematch=do_check_truematch,
                                    buffer=buffer, lookup=old_dict["mission"],
                                    keyword_obj=None,do_verbose=do_verbose_deep,
                                    do_verbose_deep=do_verbose_deep
                                    )
                    except NotImplementedError as err:
                        curr_str = (("\n-\n"
                                    +"Printing Error:\nID: {0}\nBibcode: {1}\n"
                                    +"Mission: {2}\nMasked class: {3}\n")
                                    .format(old_dict["id"], old_dict["bibcode"],
                                            old_dict["mission"], masked_class))
                        curr_str += ("The following err. was encountered"
                                    +" in train_model_ML:\n")
                        curr_str += repr(err)
                        curr_str += "\nError was noted. Skipping this paper.\n-"
                        print(curr_str) #Print current error
                        #
                        #Store this error-modif
                        err_dict = {"text":curr_str,
                            "class":masked_class, #Mask class
                            "id":old_dict["id"], "mission":old_dict["mission"],
                            "forest":None, "bibcode":old_dict["bibcode"]}
                        if (old_dict["bibcode"] not in dict_errors):
                            dict_errors[old_dict["bibcode"]] = {}
                        dict_errors[old_dict["bibcode"]][curr_key] = err_dict
                        #
                        str_err += curr_str #Tack this error onto full string
                        i_skipped += 1 #Increment count of skipped papers
                        continue
                #
                else: #Otherwise, run without ambig. phrase check
                    curr_res = self.process(text=old_dict["text"],
                                    do_check_truematch=do_check_truematch,
                                    buffer=buffer, lookup=old_dict["mission"],
                                    keyword_obj=None,do_verbose=do_verbose_deep,
                                    do_verbose_deep=do_verbose_deep
                                    )
                #
                #Store the modif and previous classification information
                new_dict = {"text":curr_res["modif"],
                            "class":masked_class, #Mask class
                            "id":old_dict["id"], "mission":old_dict["mission"],
                            "forest":curr_res["forest"],
                            "bibcode":old_dict["bibcode"]}
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
                print("{0} texts have been processed into modifs."
                        .format(i_track))
                if do_check_truematch:
                    print("{0} texts skipped due to unknown ambig. phrases."
                            .format(i_skipped))
                print("Storing the data in train+validate+test directories...")
            #

            #Store the modifs in new TVT directories
            classifier.generate_directory_TVT(dir_model=dir_model,
                            fraction_TVT=fraction_TVT, mode_TVT=mode_TVT,
                            dict_texts=dict_modifs, do_shuffle=do_shuffle,
                            seed=seed_TVT, do_verbose=do_verbose)
            #Save the modifs with caught processing errors
            np.save(filepath_dicterrors, dict_errors)
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

        #Exit the method with error string
        #Print some notes
        if do_verbose:
            print("Run of train_model_ML() complete!\nError string returned.")
        #
        return str_err
    #
#


##Class: Performance
class Performance(_Base):
    """
    Class: Performance
    Purpose:
        - !
    Initialization Arguments:
        - !
    """
    ##Method: __init__
    ##Purpose: Initialize this class instance
    def __init__(self, do_verbose=False, do_verbose_deep=False):
        """
        Method: __init__
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Initializes instance of Performance class.
        """
        #Initialize storage
        self._storage = {}
        self._store_info(do_verbose, "do_verbose")
        self._store_info(do_verbose_deep, "do_verbose_deep")
        #

        #Exit the method
        if do_verbose:
            print("Instance of Performance successfully initialized!")
        #
        return
    #

    ##Method: _combine_performance_across_evaluations
    ##Purpose: Combine measured evaluations per operator across those operators so as to investigate performance across combined operators
    def _combine_performance_across_evaluations(self, evaluations, titles, mappers, do_verbose=None):
        """
        Method: _combine_performance_across_evaluations
        WARNING! This method is *not* meant to be used directly by users.
        Purpose: Combines evaluations, measured per operator, across those operators, so as to investigate the performance across combined operators.

        """
        #Fetch global variables
        if (do_verbose is None):
            do_verbose = self._get_info("do_verbose")
        #
        #Print some notes
        if do_verbose:
            print("\n> Running _combine_performance_across_evaluations()!")
            print("Combining evaluations across these operators: {0}"
                    .format(titles))
        #

        #Fetch and cross-check actual and measured classifs
        act_classnames = evaluations[titles[0]]["act_classnames"]
        meas_classnames = evaluations[titles[0]]["meas_classnames"]
        #
        #Throw error if any differing classif names
        check_act = [evaluations[key]["act_classnames"] for key in titles]
        check_meas = [evaluations[key]["meas_classnames"] for key in titles]
        if any([(item != act_classnames) for item in check_act]):
            raise ValueError("Err: Different actual classifs:\n{0} vs. {1}"
                            .format(act_classnames, check_act))
        if any([(item != meas_classnames) for item in check_meas]):
            raise ValueError("Err: Different measured classifs:\n{0} vs. {1}"
                            .format(meas_classnames, check_meas))
        #

        #Fetch all possible two-part combinations of operators
        list_pairs = list(iterer.combinations(sorted(titles), 2))
        #Print some notes
        if do_verbose:
            print("All possible operator combinations: {0}".format(list_pairs))
        #

        #Initialize container for all combined evaluations
        dict_combined = {}

        #Iterate through operator combinations
        for curr_pair in list_pairs:
            curr_combname = "{0}|{1}".format(curr_pair[0], curr_pair[1]).lower()
            #Print some notes
            if do_verbose:
                print("Considering combination: {0}".format(curr_pair))

            #Initialize container for current combined evaluations
            meas_opnames_0 =["{0}_{1}".format(curr_pair[0].lower(),item.lower())
                                for item in meas_classnames]
            meas_opnames_1 =["{0}_{1}".format(curr_pair[1].lower(),item.lower())
                                for item in meas_classnames]
            curr_dict = {"{0}_{1}".format(curr_combname, key):
                            {"counters":
                            {item0:{item1:0
                                    for item1 in meas_opnames_1}
                            for item0 in meas_opnames_0}
                            }
                        for key in act_classnames}
            #
            #Include _total counter to maintain general evaluation format
            for curr_key1 in curr_dict:
                for curr_key2 in curr_dict[curr_key1]["counters"]:
                    curr_dict[curr_key1]["counters"][curr_key2]["_total"] = 0
            #

            #Verify same actual results across operators
            if (evaluations[curr_pair[0]]["actual_results"]
                        != evaluations[curr_pair[1]]["actual_results"]):
                raise ValueError("Err: Different actual results for operators.")
            #

            #Iterate through results
            act_results = evaluations[curr_pair[0]]["actual_results"]
            meas_results_0 = evaluations[curr_pair[0]]["measured_results_new"]
            meas_results_1 = evaluations[curr_pair[1]]["measured_results_new"]
            for ii in range(0, len(act_results)):
                #Iterate through missions
                for curr_mission in act_results[ii]["missions"]:
                    #Fetch current actual results
                    curr_act_raw = act_results[ii]["missions"][
                                                curr_mission]["class"].lower()
                    if (curr_act_raw in mappers[0]):
                        curr_act = mappers[0][curr_act_raw]
                    else:
                        curr_act = curr_act_raw
                    curr_act = curr_act.lower().replace("_","")
                    #
                    #Fetch current measured results
                    curr_meas_0 = (meas_results_0[ii][curr_mission]["verdict"]
                                    .lower().replace("_",""))
                    curr_meas_1 = (meas_results_1[ii][curr_mission]["verdict"]
                                    .lower().replace("_",""))
                    #
                    #Update counters with latest results
                    curr_actkey = "{0}_{1}".format(curr_combname, curr_act)
                    curr_key0="{0}_{1}".format(curr_pair[0].lower(),curr_meas_0)
                    curr_key1="{0}_{1}".format(curr_pair[1].lower(),curr_meas_1)
                    curr_dict[curr_actkey]["counters"][curr_key0][curr_key1] +=1
                    curr_dict[curr_actkey]["counters"][curr_key0]["_total"] += 1
                #
            #
            #Store welded operator-class names
            for curr_actkey in curr_dict:
                curr_dict[curr_actkey]["act_classnames"] = meas_opnames_0
                curr_dict[curr_actkey]["meas_classnames"] = meas_opnames_1
            #
            #Store combined evaluations for current pair
            dict_combined[curr_combname] = curr_dict
        #

        #Return the combined evaluations
        if do_verbose:
            print("Run of _combine_performance_across_evaluations() complete!")
        #
        return dict_combined
    #

    ##Method: evaluate_performance_basic
    ##Purpose: Evaluate the basic performance of the internal classifier on a test set of data
    def evaluate_performance_basic(self, operators, dicts_texts, mappers, thresholds, buffers, is_text_processed, do_verify_truematch, filepath_output, do_raise_innererror, do_reuse_run, target_classifs=None, target_classifs_comb=None, do_save_evaluation=False, do_save_misclassif=False, minmax_exclude_classifs=None, filename_root="performance_confmatr_basic", fileroot_evaluation=None, fileroot_misclassif=None, figcolor="white", figsize=(20, 20), figsize_comb=(60,30), fontsize=16, hspace=None, cmap_abs=plt.cm.BuPu, cmap_norm=plt.cm.PuRd, print_freq=25, do_verbose=None, do_verbose_deep=None):
        """
        Method: evaluate_performance_basic
        Purpose:
          - Evaluate the performance of the internally stored classifier on a test set of data.
        Arguments:
          - !
          - do_verbose [bool (default=False)]:
            - Whether or not to print surface-level log information and tests.
          - do_verbose_deep [bool (default=False)]:
            - Whether or not to print inner log information and tests.
        Returns:
          - dict:
            - !
        """
        #Fetch global variables
        if (do_verbose is None):
            do_verbose = self._get_info("do_verbose")
        if (do_verbose_deep is None):
            do_verbose_deep = self._get_info("do_verbose_deep")
        #
        #Print some notes
        if do_verbose:
            print("\n> Running evaluate_performance_basic()!")
            print("Generating classifications for the given operators...")
        #

        ##Run classifier within each operator
        dict_classifications = self._generate_classifications_set(
                        operators=operators,
                        dicts_texts=dicts_texts, mappers=mappers,
                        buffers=buffers, is_text_processed=is_text_processed,
                        do_reuse_run=do_reuse_run,
                        do_verify_truematch=do_verify_truematch,
                        do_raise_innererror=do_raise_innererror,
                        do_save_evaluation=do_save_evaluation,
                        filepath_output=filepath_output,
                        fileroot_evaluation=fileroot_evaluation,
                        print_freq=print_freq,
                        do_verbose=do_verbose, do_verbose_deep=do_verbose_deep)
        #
        #Print some notes
        if do_verbose:
            print("\nClassifications generated.")
            print("Evaluating classifications...")
        #

        ##Evaluate classifier within each operator
        dict_evaluations = self._generate_performance_counters(
                        operators=operators, thresholds=thresholds,
                        mappers=mappers, classifications=dict_classifications,
                        filepath_output=filepath_output,
                        do_save_misclassif=do_save_misclassif,
                        fileroot_misclassif=fileroot_misclassif,
                        print_freq=print_freq, do_verbose=do_verbose,
                        do_verbose_deep=do_verbose_deep)
        #
        #Print some notes
        if do_verbose:
            print("\nEvaluations generated.")
            print("Plotting confusion matrices...")
        #

        ##Plot grids of confusion matrices for classifier performance
        titles = [item._get_info("name") for item in operators]
        list_evaluations = [dict_evaluations[item] for item in titles]
        #
        #For performance calculated across all possible classifications
        tmp_filename = "{0}_classifier_all.png".format(filename_root)
        self.plot_performance_confusion_matrix(
                        list_evaluations=list_evaluations, list_titles=titles,
                        filepath_plot=filepath_output,
                        minmax_exclude_classifs=minmax_exclude_classifs,
                        filename_plot=tmp_filename, figcolor=figcolor,
                        figsize=figsize, fontsize=fontsize, hspace=hspace,
                        cmap_abs=cmap_abs, cmap_norm=cmap_norm,
                        which_norm="row")
        #
        #For performance calculated across target classifications
        if (target_classifs is not None):
            tmp_filename = "{0}_classifier_targets.png".format(filename_root)
            self.plot_performance_confusion_matrix(
                        list_evaluations=list_evaluations, list_titles=titles,
                        filepath_plot=filepath_output,
                        target_act_classifs=target_classifs,
                        target_meas_classifs=target_classifs,
                        minmax_exclude_classifs=minmax_exclude_classifs,
                        filename_plot=tmp_filename, figcolor=figcolor,
                        figsize=figsize, fontsize=fontsize, hspace=hspace,
                        cmap_abs=cmap_abs, cmap_norm=cmap_norm,
                        which_norm="row")
        #

        ##Plot grids of confusion matrices for combined operator performance
        dict_combined = self._combine_performance_across_evaluations(
                                evaluations=dict_evaluations, titles=titles,
                                mappers=mappers, do_verbose=do_verbose)
        for curr_comb in dict_combined:
            list_evaluations = [dict_combined[curr_comb][key]
                                for key in dict_combined[curr_comb]]
            titles_comb = [key for key in dict_combined[curr_comb]]
            #
            y_title = curr_comb.split("|")[0]
            x_title = curr_comb.split("|")[1]
            #
            #For performance calculated across all possible classifications
            tmp_filename = ("{0}_operator_all_{1}.png"
                            .format(filename_root, curr_comb.replace("|","vs")))
            self.plot_performance_confusion_matrix(
                        list_evaluations=list_evaluations,
                        y_title=y_title, x_title=x_title,
                        filepath_plot=filepath_output, list_titles=titles_comb,
                        minmax_exclude_classifs=minmax_exclude_classifs,
                        filename_plot=tmp_filename, figcolor=figcolor,
                        figsize=figsize_comb, fontsize=fontsize, hspace=hspace,
                        cmap_abs=cmap_abs, cmap_norm=cmap_norm,
                        which_norm="all")
            #
            #For performance calculated across target classifications
            if (target_classifs_comb is not None):
                tmp_filename = ("{0}_operator_targets_{1}.png"
                            .format(filename_root, curr_comb.replace("|","vs")))
                target_act_classifs_comb = [
                                "{0}_{1}".format(curr_comb.split("|")[0], item)
                                        for item in target_classifs_comb]
                target_meas_classifs_comb = [
                                "{0}_{1}".format(curr_comb.split("|")[1], item)
                                        for item in target_classifs_comb]
                self.plot_performance_confusion_matrix(
                        list_evaluations=list_evaluations,
                        y_title=y_title, x_title=x_title,
                        filepath_plot=filepath_output, list_titles=titles_comb,
                        target_act_classifs=target_act_classifs_comb,
                        target_meas_classifs=target_meas_classifs_comb,
                        minmax_exclude_classifs=minmax_exclude_classifs,
                        filename_plot=tmp_filename, figcolor=figcolor,
                        figsize=figsize_comb, fontsize=fontsize, hspace=hspace,
                        cmap_abs=cmap_abs, cmap_norm=cmap_norm,
                        which_norm="all")
        #

        ##Exit the method
        if do_verbose:
            print("Confusion matrices have been plotted at:\n{0}"
                    .format(filepath_output))
            print("\nRun of evaluate_performance_basic() complete!")
        #
        return
    #

    ##Method: evaluate_performance_uncertainty
    ##Purpose: Evaluate the performance of the internal classifier on a test set of data as a function of uncertainty
    def evaluate_performance_uncertainty(self, operators, dicts_texts, mappers, threshold_arrays, buffers, is_text_processed, do_verify_truematch, filepath_output, do_raise_innererror, do_reuse_run, target_classifs=None, do_save_evaluation=False, filename_root="performance_grid_uncertainty", fileroot_evaluation=None, figcolor="white", figsize=(20, 20), fontsize=16, ticksize=14, tickwidth=3, tickheight=5, colors=["tomato", "dodgerblue", "purple", "dimgray", "silver", "darkgoldenrod", "darkgreen", "green", "cyan"], alphas=([0.75]*10), linestyles=["-", "-", "-", "--", "--", "--", ":", ":", ":"], linewidths=([3]*10), markers=(["o"]*10), alpha_match=0.5, color_match="black", linestyle_match="-", linewidth_match=8, marker_match="*", print_freq=25, do_verbose=None, do_verbose_deep=None):
        """
        Method: evaluate_performance_uncertainty
        Purpose:
          - Evaluate the performance of the internally stored classifier on a test set of data as a function of uncertainty.
        Arguments:
          - !
          - do_verbose [bool (default=False)]:
            - Whether or not to print surface-level log information and tests.
          - do_verbose_deep [bool (default=False)]:
            - Whether or not to print inner log information and tests.
        Returns:
          - dict:
            - !
        """
        #Fetch global variables
        num_ops = len(operators)
        if (do_verbose is None):
            do_verbose = self._get_info("do_verbose")
        if (do_verbose_deep is None):
            do_verbose_deep = self._get_info("do_verbose_deep")
        #
        #Print some notes
        if do_verbose:
            print("\n> Running evaluate_performance_uncertainty()!")
            print("Generating classifications for operators...")
        #

        #Run classifier for each operator at each uncertainty level
        dict_classifications = self._generate_classifications_set(
                        operators=operators,
                        dicts_texts=dicts_texts, mappers=mappers,
                        buffers=buffers, is_text_processed=is_text_processed,
                        do_reuse_run=do_reuse_run,
                        do_verify_truematch=do_verify_truematch,
                        do_raise_innererror=do_raise_innererror,
                        do_save_evaluation=do_save_evaluation,
                        filepath_output=filepath_output,
                        fileroot_evaluation=fileroot_evaluation,
                        print_freq=print_freq,
                        do_verbose=do_verbose, do_verbose_deep=do_verbose_deep)
        #
        #Print some notes
        if do_verbose:
            print("\nClassifications generated.")
            print("Evaluating classifications...")
        #

        ##Evaluate classifier within each operator
        num_thres = len(threshold_arrays[0])
        list_evaluations_raw = [None]*num_thres
        for ii in range(0, num_thres):
            #Print some notes
            if do_verbose:
                print("Threshold #{0} of {1}:".format((ii+1), num_thres))
            #
            #Generate the evaluation for this threshold
            curr_threses = [item[ii] for item in threshold_arrays]
            list_evaluations_raw[ii] = self._generate_performance_counters(
                        operators=operators, thresholds=curr_threses,
                        mappers=mappers, classifications=dict_classifications,
                        filepath_output=filepath_output,
                        do_save_misclassif=False, fileroot_misclassif=None,
                        print_freq=print_freq, do_verbose=do_verbose,
                        do_verbose_deep=do_verbose_deep)
        #
        #Print some notes
        if do_verbose:
            print("\nEvaluations generated.")
            print("Plotting performance with respect to uncertainty...")
        #

        ##Plot grids of classifier performance as function of uncertainty
        titles = [item._get_info("name") for item in operators]
        list_evaluations = [[list_evaluations_raw[ii][item]
                                for ii in range(0, num_thres)]
                            for item in titles] #Change hierarchy
        #

        #For performance across all possible classifiers
        list_actlabels = [sorted(item[0]["act_classnames"])
                            for item in list_evaluations]
        list_measlabels = [sorted(item[0]["meas_classnames"])
                            for item in list_evaluations]
        fig_suptitle = "Performance vs. Uncertainty\nAll Classifs."
        self.plot_performance_vs_metric(list_xs=threshold_arrays,
                list_evaluations=list_evaluations,list_actlabels=list_actlabels,
                list_measlabels=list_measlabels, titles=titles,
                filepath_plot=filepath_output, xlabel="Uncertainty Threshold",
                ylabel="Count of Classifications", fig_suptitle=fig_suptitle,
                filename_plot="{0}_all.png".format(filename_root),
                figcolor=figcolor, figsize=figsize, fontsize=fontsize,
                ticksize=ticksize, tickwidth=tickwidth, tickheight=tickheight,
                colors=colors, alphas=alphas, linestyles=linestyles,
                linewidths=linewidths, markers=markers, alpha_match=alpha_match,
                color_match=color_match, linestyle_match=linestyle_match,
                linewidth_match=linewidth_match, marker_match=marker_match,
                do_verbose=do_verbose, do_verbose_deep=do_verbose_deep)
        #

        #For performance across just target classifiers
        if (target_classifs is not None):
            list_actlabels = [[item2 for item2 in
                                    sorted(item1[0]["act_classnames"])
                                    if (item2 in target_classifs)]
                                for item1 in list_evaluations]
            list_measlabels = [[item2 for item2 in
                                    sorted(item1[0]["meas_classnames"])
                                    if (item2 in target_classifs)]
                                for item1 in list_evaluations]
            fig_suptitle = ("Performance vs. Uncertainty\nTarget Classifs Only:"
                            +"{0}".format(target_classifs))
            self.plot_performance_vs_metric(list_xs=threshold_arrays,
                list_evaluations=list_evaluations,list_actlabels=list_actlabels,
                list_measlabels=list_measlabels, titles=titles,
                xlabel="Uncertainty Threshold",
                ylabel="Count of Classifications",
                fig_suptitle=fig_suptitle, filepath_plot=filepath_output,
                filename_plot="{0}_targets.png".format(filename_root),
                figcolor=figcolor, figsize=figsize, fontsize=fontsize,
                ticksize=ticksize, tickwidth=tickwidth, tickheight=tickheight,
                colors=colors, alphas=alphas, linestyles=linestyles,
                linewidths=linewidths, markers=markers, alpha_match=alpha_match,
                color_match=color_match, linestyle_match=linestyle_match,
                linewidth_match=linewidth_match, marker_match=marker_match,
                do_verbose=do_verbose, do_verbose_deep=do_verbose_deep)
        #

        #Print some notes
        if do_verbose:
            print("Results have been plotted at:\n{0}"
                    .format(filepath_output))
        #

        #Exit the method
        if do_verbose:
            print("\nRun of evaluate_performance_uncertainty() complete!")
        #
        return
    #

    ##Method: _generate_classification
    ##Purpose: Generate performance evaluation of full classification pipeline (text to rejection/verdict) for singular operator
    def _generate_classification(self, operator, dict_texts, buffer, is_text_processed, do_verify_truematch, do_raise_innererror, do_save_evaluation, filepath_save, print_freq=25, do_verbose=False, do_verbose_deep=False):
        """
        Method: _generate_classification
        Purpose:
          - !
        Arguments:
          - !
          - do_verbose [bool (default=False)]:
            - Whether or not to print surface-level log information and tests.
          - do_verbose_deep [bool (default=False)]:
            - Whether or not to print inner log information and tests.
        Returns:
          - dict:
            - !
        """
        #Print some notes
        if do_verbose:
            print("\n> Running _generate_classification()!")
            print("Operator: {0}".format(operator._get_info("name")))
        #

        ##Use operator to classify the set of texts and measure performance
        #Unpack the classified information for this operator
        curr_keys = list(dict_texts.keys()) #All keys for accessing texts
        curr_actdicts = [dict_texts[curr_keys[jj]]
                        for jj in range(0, len(curr_keys))] #Forced order
        #Load in as either raw or preprocessed data
        if is_text_processed: #If given text preprocessed
            curr_texts = None
            curr_modifs = [dict_texts[curr_keys[jj]]["text"]
                            for jj in range(0, len(curr_keys))]
            curr_forests = [dict_texts[curr_keys[jj]]["forest"]
                            for jj in range(0, len(curr_keys))]
        else: #If given text needs to be preprocessed
            curr_texts = [dict_texts[curr_keys[jj]]["text"]
                            for jj in range(0, len(curr_keys))]
            curr_modifs = None
            curr_forests = None
        #

        #Classify texts with current operator
        curr_results = operator.classify_set(texts=curr_texts,
                            modifs=curr_modifs, forests=curr_forests,
                            buffer=buffer,
                            do_check_truematch=do_verify_truematch,
                            do_raise_innererror=do_raise_innererror,
                            print_freq=print_freq, do_verbose=do_verbose,
                            do_verbose_deep=do_verbose_deep)
        #
        #Print some notes
        if do_verbose:
            print("Classification complete for this operator.")
        #

        #Store the evaluation
        dict_evaluation = {"actual_results":curr_actdicts,
                            "measured_results":curr_results}
        #

        ##Save the evaluation components, if so requested
        if do_save_evaluation:
            np.save(filepath_save, dict_evaluation)
            #Print some notes
            if do_verbose:
                print("\nEvaluation saved at: {0}".format(filepath_save))
        #

        ##Return the evaluation components
        if do_verbose:
            print("\nRun of _generate_classification() complete!")
        #
        return dict_evaluation
    #

    ##Method: _generate_classifications
    ##Purpose: Generate performance evaluation of full classification pipeline (text to rejection/verdict) for set of operators
    def _generate_classifications_set(self, operators, dicts_texts, mappers, buffers, is_text_processed, do_verify_truematch, do_raise_innererror, do_reuse_run, do_save_evaluation=False, filepath_output=None, fileroot_evaluation=None, print_freq=25, do_verbose=False, do_verbose_deep=False):
        """
        Method: _generate_classifications_set
        Purpose:
          - !
        Arguments:
          - !
          - do_verbose [bool (default=False)]:
            - Whether or not to print surface-level log information and tests.
          - do_verbose_deep [bool (default=False)]:
            - Whether or not to print inner log information and tests.
        Returns:
          - dict:
            - !
        """
        ##Fetch global variables
        if (do_verbose is None):
            do_verbose = self._get_info("do_verbose")
        if (do_verbose_deep is None):
            do_verbose_deep = self._get_info("do_verbose_deep")
        #
        names_ops = [item._get_info("name") for item in operators]
        num_ops = len(operators)
        if (do_save_evaluation or do_reuse_run):
            list_save_filepaths = [os.path.join(filepath_output,
                                        (fileroot_evaluation
                                        +"_{0}.npy".format(item)))
                                    for item in names_ops]
        #

        #Throw error if operators do not have unique names
        if (len(set(names_ops)) !=num_ops):
            raise ValueError("Err: Please give each operator a unique name."
                        +"\nCurrently, the names are:\n{0}"
                        .format([item._get_info("name") for item in operators]))
        #
        #Print some notes
        if do_verbose:
            print("\n> Running _generate_classifications_set()!")
            print("Iterating through Operators to classify each set of text...")
        #

        #Iterate through operators and either load or generate evaluations
        dict_evaluations = {}
        for ii in range(0, num_ops):
            #Load info for current operator
            curr_op = operators[ii] #Current operator
            curr_name = names_ops[ii]
            curr_data = dicts_texts[ii]
            #
            ##Load any pre-existing evaluation, if so requested
            if (do_reuse_run and (os.path.exists(list_save_filepaths[ii]))):
                #Print some notes
                if do_verbose:
                    print("Previous eval. exists at {0}\nLoading that eval..."
                            .format(list_save_filepaths[ii]))
                #
                dict_evaluations[curr_name] = np.load(list_save_filepaths[ii],
                                                    allow_pickle=True).item()
                #
            #
            ##Otherwise, generate and store new evaluation
            else:
                #Print some notes
                if do_verbose:
                    print("Classifying with Operator #{0}: {1}..."
                            .format(ii, curr_name))
                #
                dict_evaluations[curr_name] = self._generate_classification(
                        operator=curr_op, dict_texts=curr_data,
                        buffer=buffers[ii], is_text_processed=is_text_processed,
                        do_verify_truematch=do_verify_truematch,
                        do_raise_innererror=do_raise_innererror,
                        do_save_evaluation=do_save_evaluation,
                        filepath_save=list_save_filepaths[ii],
                        print_freq=print_freq, do_verbose=do_verbose,
                        do_verbose_deep=do_verbose_deep)
        #

        ##Return the evaluation components
        if do_verbose:
            print("\nRun of _generate_classifications_set() complete!")
        #
        return dict_evaluations
    #

    ##Method: _generate_classifications
    ##Purpose: Generate performance evaluation of full classification pipeline (text to rejection/verdict)
    def old_2024_03_04_before_split_evals_generate_classifications(self, operators, dicts_texts, mappers, buffers, is_text_processed, do_verify_truematch, do_raise_innererror, do_reuse_run, do_save_evaluation=False, filepath_output=None, fileroot_evaluation=None, print_freq=25, do_verbose=False, do_verbose_deep=False):
        """
        Method: _generate_classifications
        Purpose:
          - !
        Arguments:
          - !
          - do_verbose [bool (default=False)]:
            - Whether or not to print surface-level log information and tests.
          - do_verbose_deep [bool (default=False)]:
            - Whether or not to print inner log information and tests.
        Returns:
          - dict:
            - !
        """
        ##Fetch global variables
        if (do_verbose is None):
            do_verbose = self._get_info("do_verbose")
        if (do_verbose_deep is None):
            do_verbose_deep = self._get_info("do_verbose_deep")
        #
        num_ops = len(operators)
        if (do_save_evaluation or do_reuse_run):
            save_filepath = os.path.join(filepath_output,
                                        (fileroot_evaluation+".npy"))
        #

        #Throw error if operators do not have unique names
        if (len(set([item._get_info("name") for item in operators])) !=num_ops):
            raise ValueError("Err: Please give each operator a unique name."
                        +"\nCurrently, the names are:\n{0}"
                        .format([item._get_info("name") for item in operators]))
        #
        #Print some notes
        if do_verbose:
            print("\n> Running _generate_classifications()!")
            print("Iterating through Operators to classify each set of text...")
        #

        ##Load any pre-existing evaluation, if so requested
        if (do_reuse_run and (os.path.exists(save_filepath))):
            #Print some notes
            if do_verbose:
                print("Previous evaluation exists at {0}\nLoading that eval..."
                        .format(save_filepath))
            #
            dict_evaluations = np.load(save_filepath, allow_pickle=True).item()
            return dict_evaluations
        #

        ##Use each operator to classify the set of texts and measure performance
        dict_evaluations = {item._get_info("name"):None for item in operators}
        for ii in range(0, num_ops):
            curr_op = operators[ii] #Current operator
            curr_name = curr_op._get_info("name")
            curr_data = dicts_texts[ii]
            #
            #Print some notes
            if do_verbose:
                print("Classifying with Operator #{0}...".format(ii))
            #

            #Unpack the classified information for this operator
            curr_keys = list(curr_data.keys()) #All keys for accessing texts
            curr_actdicts = [curr_data[curr_keys[jj]]
                            for jj in range(0, len(curr_keys))] #Forced order
            #Load in as either raw or preprocessed data
            if is_text_processed: #If given text preprocessed
                curr_texts = None
                curr_modifs = [curr_data[curr_keys[jj]]["text"]
                                for jj in range(0, len(curr_keys))]
                curr_forests = [curr_data[curr_keys[jj]]["forest"]
                                for jj in range(0, len(curr_keys))]
            else: #If given text needs to be preprocessed
                curr_texts = [curr_data[curr_keys[jj]]["text"]
                                for jj in range(0, len(curr_keys))]
                curr_modifs = None
                curr_forests = None
            #

            #Classify texts with current operator
            curr_results = curr_op.classify_set(texts=curr_texts,
                                modifs=curr_modifs, forests=curr_forests,
                                buffer=buffers[ii],
                                do_check_truematch=do_verify_truematch,
                                do_raise_innererror=do_raise_innererror,
                                print_freq=print_freq,
                                do_verbose=do_verbose,
                                do_verbose_deep=do_verbose_deep)
            #
            #Print some notes
            if do_verbose:
                print("Classification complete for Operator #{0}.".format(ii))
                print("Generating the performance counter...")
            #

            #Store the current results
            dict_evaluations[curr_name] = {"actual_results":curr_actdicts,
                                        "measured_results":curr_results}
            #

            #Print some notes
            if do_verbose:
                print("All work complete for Operator #{0}.".format(ii))
            #
        #
        #Print some notes
        if do_verbose:
            print("!")
        #

        ##Save the evaluation components, if so requested
        if do_save_evaluation:
            #tmp_filepath = os.path.join(filepath_output,
            #                            (fileroot_evaluation+".npy"))
            #np.save(tmp_filepath, dict_evaluations)
            np.save(save_filepath, dict_evaluations)
            #
            #Print some notes
            if do_verbose:
                print("\nEvaluation saved at: {0}".format(save_filepath))
        #

        ##Return the evaluation components
        if do_verbose:
            print("\nRun of _generate_classifications() complete!")
        #
        return dict_evaluations
    #

    ##Method: _generate_performance_counter
    ##Purpose: Generate performance counter for set of measured classifications vs actual classifications
    def _generate_performance_counters(self, operators, mappers, classifications, thresholds=None, filepath_output=None, do_save_misclassif=None, fileroot_misclassif=None, print_freq=25, do_verbose=False, do_verbose_deep=False):
        """
        Method: _generate_performance_counter
        Purpose:
          - !
        Arguments:
          - !
          - do_verbose [bool (default=False)]:
            - Whether or not to print surface-level log information and tests.
          - do_verbose_deep [bool (default=False)]:
            - Whether or not to print inner log information and tests.
        Returns:
          - dict:
            - !
        """
        #Fetch global variables
        if (do_verbose is None):
            do_verbose = self._get_info("do_verbose")
        if (do_verbose_deep is None):
            do_verbose_deep = self._get_info("do_verbose_deep")
        #
        num_ops = len(operators)
        #

        #Iterate through operators
        dict_evaluations = {}
        for ii in range(0, num_ops):
            #Extract values for current operator
            curr_op = operators[ii]
            curr_mapper = mappers[ii]
            curr_opname = curr_op._get_info("name")
            curr_actdicts = classifications[curr_opname]["actual_results"]
            curr_measdicts_orig=classifications[curr_opname]["measured_results"]
            curr_measdicts_new = [None]*len(curr_measdicts_orig)
            num_texts = len(curr_measdicts_orig)
            meas_classifs = curr_op._get_info("classifier"
                                                )._get_info("class_names")
            #Print some notes
            if do_verbose:
                print("\n> Running _generate_performance_counter() for: {0}"
                        .format(curr_opname))
            #

            #Initialize container for counters and misclassifications
            if (curr_mapper is not None): #Use mask classifications, if given
                act_classnames_raw = list(set(
                                [item for item in list(curr_mapper.values())]))
                meas_classnames_raw = list(set(
                                [item for item in list(curr_mapper.values())]))
            else: #Otherwise, use internal classifications
                act_classnames_raw = meas_classifs
                meas_classnames_raw = meas_classifs
            #
            #Extend measured allowed class names to include low-uncertainty, etc.
            act_classnames = (act_classnames_raw + [config.verdict_rejection])
            meas_classnames = (meas_classnames_raw + config.list_other_verdicts)
            #
            #Streamline the class names
            act_classnames = [item.lower().replace("_","")
                                for item in act_classnames]
            meas_classnames = [item.lower().replace("_","")
                                for item in meas_classnames]
            #
            #Form containers
            dict_counters = {act_key:{meas_key:0 for meas_key in meas_classnames}
                            for act_key in act_classnames}
            dict_misclassifs = {}
            #
            #Print some notes
            if do_verbose:
                print("Accumulating performance over {0} texts."
                        .format(num_texts))
                print("Actual class names: {0}\nMeasured class names: {1}"
                        .format(act_classnames, meas_classnames))
            #

            #Count up classifications from given texts and classifications
            i_misclassif = 0 #Count of misclassifications
            for jj in range(0, num_texts):
                curr_actdict = curr_actdicts[jj]
                curr_measdict_orig = curr_measdicts_orig[jj]
                curr_measdict_new = {key:None for key in curr_measdict_orig}
                curr_measdicts_new[jj] = curr_measdict_new #For updated verdicts
                #
                #Iterate through missions that were considered
                for curr_key in curr_measdict_orig:
                    lookup = curr_key
                    res_lowprob = config.dictverdict_lowprob.copy()["verdict"]
                    #

                    #Extract actual classif
                    curr_actval = curr_actdict["missions"][lookup]["class"]
                    if ((curr_mapper is not None)
                                and (curr_actval.lower() in curr_mapper)):
                        #Map to masked value if so requested
                        curr_actval = curr_mapper[curr_actval.lower()]
                    #
                    curr_actval = curr_actval.lower().replace("_","")

                    #Extract measured classif and apply any thresholds
                    curr_measval_raw = curr_measdict_orig[lookup]["verdict"]
                    #curr_measval = curr_measval_raw.lower().replace("_","")
                    #If no threshold given, use original verdict
                    if ((thresholds is None) or (thresholds[ii] is None)):
                        if ((curr_mapper is not None)
                                and (curr_measval_raw.lower() in curr_mapper)):
                            curr_measval = curr_mapper[curr_measval_raw.lower()]
                        #
                        else:
                            curr_measval = curr_measval_raw
                        curr_measval = curr_measval.lower().replace("_","")
                    #
                    #Otherwise, apply threshold
                    else:
                        #Fetch uncertainties (higher -> more certain)
                        tmppass = curr_measdict_orig[lookup]["uncertainty"]
                        if (tmppass is not None):
                            max_verdict = max(tmppass, key=tmppass.get)
                            max_val = tmppass[max_verdict] #Uncertainty of verd.
                            #
                            #Fetch current threshold
                            curr_thres = thresholds[ii] #Same thres. for all
                            #
                            #Apply current threshold
                            if (max_val < curr_thres): #If below thres.
                                curr_measval = res_lowprob #Return low prob.
                            else:
                                curr_measval = max_verdict #Return verdict
                        #
                        else: #If no uncertainty, means was not classified
                            curr_measval = curr_measval_raw #Return orig. verd.
                        #
                        curr_measval = curr_measval.lower().replace("_","")
                    #

                    #Store the updated verdict for use later
                    curr_measdict_new[lookup] = {"verdict":curr_measval}

                    #Increment current counter
                    dict_counters[curr_actval][curr_measval] += 1

                    #If misclassification, take note
                    if (curr_actval != curr_measval):
                        curr_info = {"act_classif":curr_actval,
                        "meas_classif":curr_measval,
                        "bibcode":curr_actdict["bibcode"],
                        "mission":lookup, "id":curr_actdict["id"],
                        "modif":curr_measdict_orig[lookup]["modif"],
                        "modif_none":curr_measdict_orig[lookup]["modif_none"]}
                        #
                        dict_misclassifs[str(i_misclassif)] = curr_info
                        #Increment count of misclassifications
                        i_misclassif += 1
                    #
                #
            #

            #Save the misclassified cases, if so requested
            if do_save_misclassif:
                #Print some notes
                if do_verbose:
                    print("Saving misclassifications...")
                #
                #Build string of misclassification information
                list_str = [("Internal ID: {0}\nBibcode: {1}\nMission: {2}\n"
                            +"Actual Classification: {3}\n"
                            +"Measured Classification: {4}\n"
                            +"Modif:\n''\n{5}\n''\n-\n"
                            +"Base Paragraph:\n''\n{6}\n''\n-\n")
                            .format(dict_misclassifs[key]["id"],
                                    dict_misclassifs[key]["bibcode"],
                                    dict_misclassifs[key]["mission"],
                                    dict_misclassifs[key]["act_classif"],
                                    dict_misclassifs[key]["meas_classif"],
                                    dict_misclassifs[key]["modif"],
                                    dict_misclassifs[key]["modif_none"])
                            for key in dict_misclassifs]
                #
                str_misclassif = "\n-----\n".join(list_str) #Combined string

                #Save the full string of misclassifications
                tmp_filename = ("{0}_{1}.txt"
                                    .format(fileroot_misclassif, curr_opname))
                tmp_filepath = os.path.join(filepath_output, tmp_filename)
                self._write_text(text=str_misclassif, filepath=tmp_filepath)
                #
                #Print some notes
                if do_verbose:
                    print("\nMisclassifications for {1} saved at: {0}".format(tmp_filepath, curr_opname))
            #

            #Compute internal counter totals
            for key_1 in dict_counters:
                curr_sum = sum([dict_counters[key_1][key_2]
                                for key_2 in dict_counters[key_1]])
                dict_counters[key_1]["_total"] = curr_sum
            #
            #Print some notes
            if do_verbose:
                print("\n-\nPerformance counter generated:")
                for key_1 in dict_counters:
                    print("Actual {0} total: {1}"
                            .format(key_1, dict_counters[key_1]["_total"]))
                    for key_2 in dict_counters[key_1]:
                        print("Actual {0} vs Measured {1}: {2}"
                                .format(key_1, key_2, dict_counters[key_1][key_2]))
            #

            #Return the counters and misclassifications
            if do_verbose:
                print("\n-\n\nRun of _generate_performance_counter() complete!")
            #
            dict_evaluations[curr_opname] = {"counters":dict_counters,
                "misclassifs":dict_misclassifs, "act_classnames":act_classnames,
                "meas_classnames":meas_classnames,
                "actual_results":curr_actdicts,
                "measured_results_orig":curr_measdicts_orig,
                "measured_results_new":curr_measdicts_new}
        #

        #Return the results from all operators
        return dict_evaluations
    #

    ##Method: plot_performance_confusion_matrix
    ##Purpose: Plot confusion matrix for given performance counters
    def plot_performance_confusion_matrix(self, list_evaluations, list_titles, filepath_plot, filename_plot, which_norm, target_act_classifs=None, target_meas_classifs=None, minmax_exclude_classifs=None, y_title="Actual", x_title="Classification", figsize=(20, 6), figcolor="white", fontsize=16, hspace=None, cmap_abs=plt.cm.BuPu, cmap_norm=plt.cm.PuRd, do_verbose=None, do_verbose_deep=None):
        """
        Method: plot_performance_confusion_matrix
        Purpose:
          - !
        Arguments:
          - !
          - do_verbose [bool (default=False)]:
            - Whether or not to print surface-level log information and tests.
          - do_verbose_deep [bool (default=False)]:
            - Whether or not to print inner log information and tests.
        Returns:
          - dict:
            - !
        """
        ##Fetch global variables
        if (do_verbose is None):
            do_verbose = self._get_info("do_verbose")
        if (do_verbose_deep is None):
            do_verbose_deep = self._get_info("do_verbose_deep")
        #
        num_evals = len(list_evaluations)
        #Print some notes
        if do_verbose:
            print("\n> Running plot_performance_confusion_matrix()!")
        #

        ##Prepare the base figure
        fig = plt.figure(figsize=figsize)
        fig.set_facecolor(figcolor)
        nrow = 2
        ncol = num_evals

        ##Plot confusion matrix for each evaluation
        for ii in range(0, num_evals):
            #Fetch actual classifs
            if (target_act_classifs is not None): #Show only specific classifs
                act_classifs = sorted([item.lower()
                                    for item in target_act_classifs])
            else: #Show all allowed classifs
                act_classifs = sorted([item.lower() for item in
                                    list_evaluations[ii]["act_classnames"]])
            #Fetch measured classifs
            if (target_meas_classifs is not None): #Show only specific classifs
                meas_classifs = sorted([item.lower()
                                    for item in target_meas_classifs])
            else: #Show all allowed classifs
                meas_classifs = sorted([item.lower() for item in
                                    list_evaluations[ii]["meas_classnames"]])
            #
            num_act = len(act_classifs)
            num_meas = len(meas_classifs)

            #Set indices of classifs to exclude from conf. matrix color scaling
            if (minmax_exclude_classifs is not None):
                excl_classifs=[item.lower() for item in minmax_exclude_classifs]
                minmax_inds = {"x":[], "y":[]}
                #For actual classifs to exclude from color scaling
                for jj in range(0, num_act):
                    if (act_classifs[jj].lower() in excl_classifs):
                        minmax_inds["y"].append(jj)
                #For measured classifs to exclude from color scaling
                for jj in range(0, num_meas):
                    if (meas_classifs[jj].lower() in excl_classifs):
                        minmax_inds["x"].append(jj)
            #
            else:
                minmax_inds = None
            #

            #Initialize container for current confusion matrix
            confmatr_abs = np.ones(shape=(num_act, num_meas))*np.nan #Unnorm.
            confmatr_norm = np.ones(shape=(num_act, num_meas))*np.nan#Normalized

            #Fetch counters from current evaluation
            curr_counts = list_evaluations[ii]["counters"]

            #Accumulate the confusion matrices
            for yy in range(0, num_act): #Iterate through actual classifs
                #Iterate through measured classifs
                for xx in range(0, num_meas):
                    #For the unnormalized confusion matrix
                    curr_val = curr_counts[act_classifs[yy]][meas_classifs[xx]]
                    confmatr_abs[yy,xx] = curr_val
                #
            #
            #For the normalized confusion matrix
            act_classifs_ylabel = [None]*num_act #For y-axis labels
            for yy in range(0, num_act): #Iterate through actual classifs
                #Fetch counts of target subset and of all possible classifs
                row_total = np.sum([curr_counts[act_classifs[yy]][key]
                                    for key in meas_classifs])
                clf_total = curr_counts[act_classifs[yy]]["_total"]#Classif cnt.
                #Write row vs total counts into label for this row
                act_classifs_ylabel[yy] = ("{0}\nRow={1}, Tot.={2}: {3:.0f}%"
                                .format(act_classifs[yy], row_total, clf_total,
                                        (row_total/clf_total*100)))
                #Iterate through measured classifs
                for xx in range(0, num_meas):
                    if (which_norm == "row"):
                        confmatr_norm[yy,xx] = (confmatr_abs[yy,xx] / row_total)
                    elif (which_norm == "all"):
                        confmatr_norm[yy,xx] = (confmatr_abs[yy,xx]
                                                / confmatr_abs.sum())
                    else:
                        raise ValueError("Err: Unrecognized norm. scheme: {0}"
                                        .format(which_norm))
                #
            #

            #Plot the current set of confusion matrices
            #For the unnormalized matrix
            ax = fig.add_subplot(nrow, ncol, (ii+1))
            self._ax_confusion_matrix(matr=confmatr_abs, ax=ax,
                x_labels=meas_classifs, y_labels=act_classifs_ylabel,
                y_title=y_title, x_title=x_title,
                cbar_title="Absolute Count", minmax_inds=minmax_inds,
                ax_title="{0}".format(list_titles[ii]), cmap=cmap_abs,
                fontsize=fontsize, is_norm=False)
            #
            #For the normalized matrix
            ax = fig.add_subplot(nrow, ncol, (ii+ncol+1))
            self._ax_confusion_matrix(matr=confmatr_norm, ax=ax,
                x_labels=meas_classifs, y_labels=act_classifs_ylabel,
                y_title=y_title, x_title=x_title,
                cbar_title="Normalized Fraction", minmax_inds=minmax_inds,
                ax_title="{0}".format(list_titles[ii]), cmap=cmap_norm,
                fontsize=fontsize, is_norm=True)
            #
        #

        #Save and close the figure
        plt.tight_layout()
        if (hspace is not None):
            plt.subplots_adjust(hspace=hspace)
        plt.savefig(os.path.join(filepath_plot, filename_plot))
        plt.close()

        #Exit the method
        if do_verbose:
            print("\nRun of plot_performance_confusion_matrix() complete!")
        #
        return
    #

    ##Method: plot_performance_vs_metric
    ##Purpose: Plot performance as a function of given metric
    def plot_performance_vs_metric(self, list_xs, list_evaluations, list_actlabels, list_measlabels, titles, xlabel, ylabel, filepath_plot, filename_plot, fig_suptitle, figcolor="white", figsize=(20, 20), fontsize=16, ticksize=14, tickwidth=3, tickheight=5, colors=["tomato", "dodgerblue", "purple", "dimgray", "silver", "darkgoldenrod", "darkgreen", "green", "cyan"], alphas=([0.75]*10), linestyles=["-", "-", "-", "--", "--", "--", ":", ":", ":"], linewidths=([3]*10), markers=(["o"]*10), alpha_match=0.5, color_match="black", linestyle_match="-", linewidth_match=8, marker_match="*", do_verbose=None, do_verbose_deep=None):
        """
        Method: plot_performance_vs_metric
        Purpose:
          - !
        Arguments:
          - !
          - do_verbose [bool (default=False)]:
            - Whether or not to print surface-level log information and tests.
          - do_verbose_deep [bool (default=False)]:
            - Whether or not to print inner log information and tests.
        Returns:
          - dict:
            - !
        """
        #Prepare base figure
        num_evals = len(list_evaluations)
        fig = plt.figure(figsize=figsize)
        fig.set_facecolor(figcolor)
        nrow = num_evals
        ncol = max([len(item[0]["act_classnames"])
                    for item in list_evaluations])
        #

        #Iterate through operators (one row per operator)
        for ii in range(0, num_evals):
            curr_xs = list_xs[ii]
            curr_eval = list_evaluations[ii]
            curr_actlabels = list_actlabels[ii]
            curr_measlabels = list_measlabels[ii]
            #Iterate through current actual classifs
            for jj in range(0, len(curr_actlabels)):
                curr_act = curr_actlabels[jj]
                #Prepare subplot
                ax0 = fig.add_subplot(nrow, ncol, ((ii*ncol)+jj+1))
                #Iterate through measured classifs
                for kk in range(0, len(curr_measlabels)):
                    curr_meas = curr_measlabels[kk]
                    curr_ys = [curr_eval[zz]["counters"][curr_act][curr_meas]
                                for zz in range(0, len(curr_xs))
                                ] #Current count of act. vs. meas. classifs
                    #Plot results as function of uncertainty
                    ax0.plot(curr_xs, curr_ys, alpha=alphas[kk],
                            color=colors[kk], linewidth=linewidths[kk],
                            linestyle=linestyles[kk], marker=markers[kk],
                            label=curr_meas)
                    #Highlight correct answers
                    if (curr_act == curr_meas):
                        ax0.plot(curr_xs, curr_ys, alpha=alpha_match,
                                color=colors[kk], linewidth=linewidth_match,
                                linestyle=linestyle_match, marker=marker_match)
                        ax0.plot(curr_xs, curr_ys, alpha=0.5,
                                color="black", linewidth=2, linestyle="-")
                #
                #Label the subplot
                ax0.set_xlabel(xlabel, fontsize=fontsize)
                ax0.set_ylabel(ylabel, fontsize=fontsize)
                ax0.set_title("{0}: {1} Texts".format(titles[ii], curr_act),
                                fontsize=fontsize)
                ax0.tick_params(width=tickwidth, size=tickheight,
                                labelsize=ticksize, direction="in")
                #
                #Add legend, if last subplot in row
                if (jj == (len(curr_actlabels)-1)):
                    ax0.legend(loc="best", frameon=False,
                                prop={"size":fontsize})
            #
        #
        #Save and close the figure
        fig.suptitle(fig_suptitle, fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(os.path.join(filepath_plot, filename_plot))
        plt.close()
    #
#










#
