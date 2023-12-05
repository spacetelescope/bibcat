###FILE: bibcat_parameters.py
###PURPOSE: Temporary container for parameters.

#Import necessary packages
import bibcat_classes as bibcat
import bibcat_constants as preset

#Mission parameters
keyword_obj_HST = bibcat.Keyword(
            keywords=["Hubble", "Hubble Telescope", "Hubble Space Telescope"],
            acronyms=["HST", "HT"])
keyword_obj_TESS = bibcat.Keyword(
            keywords=["Transiting Exoplanet Survey Satellite"],
            acronyms=["TESS"])
keyword_obj_JWST = bibcat.Keyword(
            keywords=["James Webb Space Telescope", "James Webb Telescope", "Webb Space Telescope", "Webb Telescope"],
            acronyms=["JWST", "JST", "JT"])
keyword_obj_Kepler = bibcat.Keyword(
            keywords=["KEPLER"],
            acronyms=[])
keyword_obj_PanSTARRS = bibcat.Keyword(
            keywords=["Panoramic Survey Telescope and Rapid Response System", "Pan-STARRS", "Pan-STARRS1"],
            acronyms=["PanSTARRS", "PanSTARRS1", "PS1"])
keyword_obj_GALEX = bibcat.Keyword(
            keywords=["Galaxy Evolution Explorer"],
            acronyms=["GALEX"])
keyword_obj_K2 = bibcat.Keyword(
            keywords=["K2"],
            acronyms=[])
keyword_obj_HLA = bibcat.Keyword(
            keywords=["Hubble Legacy Archive"],
            acronyms=["HLA"])
#
all_kobjs = [keyword_obj_HST, keyword_obj_JWST, keyword_obj_TESS, keyword_obj_Kepler, keyword_obj_PanSTARRS, keyword_obj_GALEX, keyword_obj_K2, keyword_obj_HLA]
#

#Class parameters
map_papertypes = {"science":"science", "mention":"mention", "supermention":"mention", "data_influenced":"data_influenced", "unresolved_grey":"other", "unresolved_gray":"other", "engineering":"other", "instrument":"other"}
#for curr_class in preset.list_other_verdicts:
#    map_papertypes[curr_class] = curr_class
#
