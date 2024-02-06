"""
:title: parameters.py

This module contains keyword parameters for bibcat such as MAST mission names.

"""

from core import keyword

# Mission parameters
keyword_obj_HST = keyword.Keyword(
    keywords=["Hubble", "Hubble Telescope", "Hubble Space Telescope"],
    acronyms=["HST", "HT"],
    banned_overlap=["Hubble Legacy Archive"],
)
keyword_obj_TESS = keyword.Keyword(
    keywords=["Transiting Exoplanet Survey Satellite"], acronyms=["TESS"], banned_overlap=[]
)
keyword_obj_JWST = keyword.Keyword(
    keywords=["James Webb Space Telescope", "James Webb Telescope", "Webb Space Telescope", "Webb Telescope"],
    acronyms=["JWST", "JST", "JT"],
    banned_overlap=[],
)
keyword_obj_Kepler = keyword.Keyword(keywords=["Kepler"], acronyms=[], banned_overlap=[])
keyword_obj_PanSTARRS = keyword.Keyword(
    keywords=["Panoramic Survey Telescope and Rapid Response System", "Pan-STARRS", "Pan-STARRS1"],
    acronyms=["PanSTARRS", "PanSTARRS1", "PS1"],
    banned_overlap=[],
)
keyword_obj_GALEX = keyword.Keyword(keywords=["Galaxy Evolution Explorer"], acronyms=["GALEX"], banned_overlap=[])
keyword_obj_K2 = keyword.Keyword(keywords=["K2"], acronyms=[], banned_overlap=[])
keyword_obj_HLA = keyword.Keyword(keywords=["Hubble Legacy Archive"], acronyms=["HLA"], banned_overlap=[])

all_kobjs = [
    keyword_obj_HST,
    keyword_obj_JWST,
    keyword_obj_TESS,
    keyword_obj_Kepler,
    keyword_obj_PanSTARRS,
    keyword_obj_GALEX,
    keyword_obj_K2,
    keyword_obj_HLA,
]

# Classification parameters
allowed_classifications = ["SCIENCE", "DATA_INFLUENCED", "MENTION", "SUPERMENTION"]
map_papertypes = {
    "science": "science",
    "mention": "mention",
    "supermention": "mention",
    "data_influenced": "data_influenced",
    "unresolved_grey": "other",
    "unresolved_gray": "other",
    "engineering": "other",
    "instrument": "other",
}
#
