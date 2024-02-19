"""
:title: parameters.py

This module contains keyword parameters for bibcat such as MAST mission names.

"""

from bibcat.core import keyword

# Mission parameters
kobj_hubble = keyword.Keyword(
    keywords=["Hubble", "Hubble Telescope", "Hubble Space Telescope"],
    acronyms=["HST", "HT"],
    banned_overlap=["Hubble Legacy Archive"],
)
kobj_tess = keyword.Keyword(keywords=["Transiting Exoplanet Survey Satellite"], acronyms=["TESS"], banned_overlap=[])
kobj_jwst = keyword.Keyword(
    keywords=["James Webb Space Telescope", "James Webb Telescope", "Webb Space Telescope", "Webb Telescope"],
    acronyms=["JWST", "JST", "JT"],
    banned_overlap=[],
)
kobj_kepler = keyword.Keyword(keywords=["Kepler"], acronyms=[], banned_overlap=[])
kobj_panstarrs = keyword.Keyword(
    keywords=["Panoramic Survey Telescope and Rapid Response System", "Pan-STARRS", "Pan-STARRS1"],
    acronyms=["PanSTARRS", "PanSTARRS1", "PS1"],
    banned_overlap=[],
)
kobj_galex = keyword.Keyword(keywords=["Galaxy Evolution Explorer"], acronyms=["GALEX"], banned_overlap=[])
kobj_k2 = keyword.Keyword(keywords=["K2"], acronyms=[], banned_overlap=[])
kobj_hla = keyword.Keyword(keywords=["Hubble Legacy Archive"], acronyms=["HLA"], banned_overlap=[])

all_kobjs = [
    kobj_hubble,
    kobj_jwst,
    kobj_tess,
    kobj_kepler,
    kobj_panstarrs,
    kobj_galex,
    kobj_k2,
    kobj_hla,
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

# test Keyword-object lookups
test_dict_lookup_kobj = {
    "Hubble": kobj_hubble,
    "Kepler": kobj_kepler,
    "K2": kobj_k2,
    "HLA": kobj_hla,
}

# test Keyword-object lookups
test_list_lookup_kobj = [kobj_hubble, kobj_kepler, kobj_k2]
