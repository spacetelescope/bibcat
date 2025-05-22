"""
:title: parameters.py

User-specified variables that describe, e.g., the user's target missions (e.g., HST, JWST, TESS) and allowed classification types (e.g., science, mention, data-influenced).

This module contains keyword parameters for bibcat such as MAST mission names.  It also gathers Keyword objects and classification types into lists for use by the user.  For easy reference, the Keyword class initialization parameters are noted below.

Parameters
----------
acronyms_caseinsensitive : {empty list, list of strings}
    List of acronyms that can describe the mission; capitalization is not preserved (e.g., "HST" and "hst" are treated in the same manner).  Punctuation should be omitted (as it is handled internally within the code).
acronyms_casesensitive : {empty list, list of strings}
    List of acronyms that can describe the mission; capitalization is preserved (e.g., "STScI").  Punctuation should be omitted (as it is handled internally within the code).
ambig_words : {empty list, list of strings}
    Phrases for which the user requests false positive checks to be done against the internal database of false positives.  E.g., "Hubble" can be found in the mission phrase "Hubble Telescope" and also in the false positive (i.e., non-mission) phrase "Hubble constant".  By specifying "Hubble" as a false positive phrase for the Hubble mission, the code knows to internally check phrases in the text with "Hubble" against the internal false positive database and procedure.
banned_overlap : {empty list, list of strings}
    Phrases that overlap with the target mission keywords but should not be treated as the same mission.  E.g., "Hubble Legacy Archive" can be a distinct mission from "Hubble"; therefore "Hubble Legacy Archive" is banned overlap for the Hubble mission, to avoid matching "Hubble Legacy Archive" to a Keyword instance for HST.
do_not_classify : bool
    If True, text for the mission will be processed, extracted, and presented to the user, but not classified.  This can be useful for missions for which only human classification is desired.  This can also be useful for missions for which false positives are too difficult to automatically screen out (e.g., "K2", which can be a mission and also a stellar spectral type).
do_verbose : bool = False
    If True, will print statements and internal reports within applicable methods while the code is running.
keywords : {empty list, list of strings}
    List of full phrases that name the mission (e.g., "Hubble Space Telescope").  Not case-sensitive.
"""

from bibcat.core import keyword

# Mission parameters
kobj_befs = keyword.Keyword(
    keywords=[
        "Berkeley Extreme and Far UV Spectrometer",
        "Berkeley Spectrometer",
    ],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=["BEFS"],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=[],
)
kobj_copernicus = keyword.Keyword(
    keywords=["Orbiting Astronomical Observatory", "Copernicus Mission", "Copernicus Satellite", "Copernicus"],
    acronyms_casesensitive=["OAO3", "OAO 3"],
    acronyms_caseinsensitive=[],
    do_not_classify=True,
    banned_overlap=[],
    ambig_words=["Copernicus"],
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
    keywords=[
        "Very Large Array Faint Images of the Radio Sky at Twenty Centimeters Survey",
        "Very Large Array Faint Images of the Radio Sky at Twenty cm",
        "Very Large Array First",
        "Faint Images of the Radio Sky at Twenty cm",
        "Faint Images of the Radio Sky at Twenty Centimeters",
    ],
    acronyms_casesensitive=["FIRST", "VLA FIRST"],
    acronyms_caseinsensitive=[],
    do_not_classify=True,
    banned_overlap=[],
    ambig_words=[],
)
kobj_fuse = keyword.Keyword(
    keywords=["Far Ultraviolet Spectroscopic Explorer"],
    acronyms_casesensitive=["FUSE"],
    acronyms_caseinsensitive=[],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=[],
)
kobj_galex = keyword.Keyword(
    keywords=["Galaxy Evolution Explorer"],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=["GALEX"],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=[],
)
kobj_hubble = keyword.Keyword(
    keywords=[
        "Advanced Camera for Surveys",
        "Cosmic Origin Spectrograph",
        "Goddard High Resolution",
        "Hubble",
        "Hubble Frontier Field",
        "Hubble Frontier Fields",
        "Hubble Telescope",
        "Hubble Space Telescope",
        "Hubble Legacy Archive",
        "Hubble Source Catalog",
    ],
    acronyms_casesensitive=[
        "COS",
        "GOODS",
    ],
    acronyms_caseinsensitive=[
        "FGS",
        "GHRS",
        "HFF",
        "HST",
        "HLA",
        "HUDF",
        "NICMOS",
        "STIS",
        "WFPC",
        "WFPC1",
        "WFPC2",
        "WFPC 1",
    ],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=["Hubble"],
)
kobj_hut = keyword.Keyword(
    keywords=["Hopkins Ultraviolet Telescope"],
    acronyms_casesensitive=["HUT"],
    acronyms_caseinsensitive=[],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=[],
)
kobj_jwst = keyword.Keyword(
    keywords=[
        "James Webb Space Telescope",
        "James Webb Telescope",
        "Next Generation Space Telescope",
        "Webb Space Telescope",
        "Webb Telescope",
    ],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=["FGS", "JWST", "NIRCam", "NIRSpec", "NIRISS", "MIRI", "NGST"],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=[],
)
kobj_imaps = keyword.Keyword(
    keywords=["Interstellar Medium Absorption Profile Spectrograph"],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=["IMAPS"],
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
# need to revisit
kobj_k2 = keyword.Keyword(
    keywords=["K2", "K2 Mission"],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=["KTWOCANDELS"],
    do_not_classify=False,
    banned_overlap=["Kepler K2"],
    ambig_words=["K2"],
)
kobj_kepler = keyword.Keyword(
    keywords=["Kepler", "Kepler Mission"],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=[],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=["Kepler"],
)
kobj_panstarrs = keyword.Keyword(
    keywords=[
        "Panoramic Survey Telescope and Rapid Response System",
    ],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=[
        "PS1",
        "PanSTARS",
        "PanSTARRS",
        "Pan STARRS",
        "PanSTARRS1",
        "PanSTARRS 1",
        "Pan STARRS1",
        "Pan STARRS 1",
    ],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=[],
)
kobj_roman = keyword.Keyword(
    keywords=[
        "Nancy Grace Roman Space Telescope",
        "Nancy Roman Telescope",
        "Roman Space Telescope",
        "Roman Telescope",
        "Wide-Field Infrared Survey Telescope",
        "Wide Field Infrared Survey Telescope",
        "Roman",
    ],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=["RST", "WFIRST", "NGRST"],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=["Roman"],
)
kobj_tess = keyword.Keyword(
    keywords=["Transiting Exoplanet Survey Satellite"],
    acronyms_casesensitive=["TESS"],
    acronyms_caseinsensitive=[],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=[],
)
kobj_tues = keyword.Keyword(
    keywords=["Tubingen Echelle Spectrograph"],
    acronyms_casesensitive=["TUES"],
    acronyms_caseinsensitive=[],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=[],
)
kobj_uit = keyword.Keyword(
    keywords=["Ultraviolet Imaging Telescope"],
    acronyms_casesensitive=["UIT"],
    acronyms_caseinsensitive=[],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=[],
)
kobj_wuppe = keyword.Keyword(
    keywords=["Wisconsin Ultraviolet Photo Polarimetry Experiment"],
    acronyms_casesensitive=[],
    acronyms_caseinsensitive=["WUPPE"],
    do_not_classify=False,
    banned_overlap=[],
    ambig_words=[],
)
all_kobjs = [
    kobj_befs,
    kobj_copernicus,
    kobj_euve,
    kobj_first,
    kobj_fuse,
    kobj_galex,
    kobj_hubble,
    kobj_hut,
    kobj_imaps,
    kobj_iue,
    kobj_jwst,
    kobj_k2,
    kobj_kepler,
    kobj_panstarrs,
    kobj_roman,
    kobj_tess,
    kobj_tues,
    kobj_uit,
    kobj_wuppe,
]
