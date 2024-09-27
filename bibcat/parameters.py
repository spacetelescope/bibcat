"""
:title: parameters.py

This module contains keyword parameters for bibcat such as MAST mission names.

"""

from bibcat.core import keyword

# Mission parameters
kobj_hubble = keyword.Keyword(
    keywords=["Hubble", "Hubble Telescope", "Hubble Space Telescope"],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=["HST", "HT"],
    do_not_classify=False,
    banned_overlap=["Hubble Legacy Archive"],
    ambig_words=["Hubble"],
)
kobj_jwst = keyword.Keyword(
    keywords=["James Webb Space Telescope", "James Webb Telescope", "Webb Space Telescope", "Webb Telescope"],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=["JWST", "JST", "JT"],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=[],
)
kobj_roman = keyword.Keyword(
    keywords=["Nancy Grace Roman Space Telescope", "Nancy Roman Telescope", "Roman Space Telescope", "Roman Telescope"],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=["Roman", "RST", "RT"],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=["Roman"],
)
kobj_hla = keyword.Keyword(
    keywords=["Hubble Legacy Archive"],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=["HLA"],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=[],
)
# need to revisit
kobj_hsc = keyword.Keyword(
    keywords=["Hubble Source Catalog"],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=["HSC"],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=[],
)
kobj_tess = keyword.Keyword(
    keywords=["Transiting Exoplanet Survey Satellite"],
    acronyms_casesensitive=["TESS"],
    acronyms_caseinsensitive=[],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=[],
)
kobj_kepler = keyword.Keyword(
    keywords=["Kepler", "Kepler Mission"],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=["KEPLER"],
    do_not_classify=False,
    banned_overlap=["Kepler K2"],
    ambig_words=["Kepler"],
)
# need to revisit
kobj_k2 = keyword.Keyword(
    keywords=["K2", "K2 Mission"],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=[],
    do_not_classify=False,
    banned_overlap=["Kepler K2"],
    ambig_words=["K2"],
)
kobj_galex = keyword.Keyword(
    keywords=["Galaxy Evolution Explorer"],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=["GALEX"],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=[],
)
kobj_panstarrs = keyword.Keyword(
    keywords=[
        "Panoramic Survey Telescope and Rapid Response System",
        "Pan-STARRS",
        "PanSTARRS",
        "Pan-STARRS1",
        "PanSTARRS1",
    ],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=["PanSTARRS", "PanSTARRS1", "PS1"],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=[],
)
# all missions below needs revisiting
kobj_fuse = keyword.Keyword(
    keywords=["Far Ultraviolet Spectroscopic Explorer"],
    acronyms_casesensitive=["FUSE"],
    acronyms_caseinsensitive=[],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=[],
)
kobj_iue = keyword.Keyword(
    keywords=["International Ultraviolet Explorer"],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=["IUE"],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=[],
)
kobj_hut = keyword.Keyword(
    keywords=["Hopkins Ultraviolet Telescope", "HUT"],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=[],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=[],
)
kobj_uit = keyword.Keyword(
    keywords=["Ultraviolet Imaging Telescope", "UIT"],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=[],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=[],
)
kobj_wuppe = keyword.Keyword(
    keywords=["Wisconsin Ultraviolet Photo-Polarimetry Experiment"],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=["WUPPE"],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=[],
)
kobj_befs = keyword.Keyword(
    keywords=["Berkeley Extreme and Far-UV Spectrometer", "Berkeley Extreme and Far UV Spectrometer"],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=["WUPPE"],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=[],
)
kobj_tues = keyword.Keyword(
    keywords=["Tubingen Echelle Spectrograph"],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=[],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=[],
)
kobj_imaps = keyword.Keyword(
    keywords=["Interstellar Medium Absorption Profile Spectrograph", "IMAPS"],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=[],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=[],
)
kobj_euve = keyword.Keyword(
    keywords=["Extreme Ultraviolet Explorer"],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=["EUVE"],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=[],
)
kobj_first = keyword.Keyword(
    keywords=["Very Large Array First", "VLA FIRST"],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=[],
    do_not_classify=True,
    banned_overlap=[],
    ambig_words=["first"],
)
kobj_copernicus = keyword.Keyword(
    keywords=["Orbiting Astronomical Observatory", "Copernicus Mission", "Copernicus Satellite"],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=[],
    do_not_classify=True,
    banned_overlap=[],
    ambig_words=["Corpernicus"],
)
all_kobjs = [
    kobj_hubble,
    kobj_jwst,
    kobj_roman,
    kobj_hla,
    kobj_hsc,
    kobj_tess,
    kobj_kepler,
    kobj_k2,
    kobj_galex,
    kobj_panstarrs,
    kobj_fuse,
    kobj_iue,
    kobj_hut,
    kobj_uit,
    kobj_wuppe,
    kobj_befs,
    kobj_tues,
    kobj_imaps,
    kobj_euve,
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
