"""
:title: test_check_truematch.py

Testing the _check_truematch methods of the Paper class.
"""

import pytest
import spacy
from nltk.corpus import wordnet  # type: ignore

from bibcat import config
from bibcat.core import paper
from bibcat.core import parameters as params
from bibcat.core.parameters import kobj_copernicus, kobj_hubble, kobj_k2, kobj_kepler


@pytest.fixture(scope="session", autouse=True)
def nlp_model():
    return spacy.load(config.grammar.spacy_language_model)


# Keyword-object lookups
test_dict_lookup_kobj = {
    "Hubble": kobj_hubble,
    "Kepler": kobj_kepler,
    "K2": kobj_k2,
    "Copernicus": kobj_copernicus,
}
test_list_lookup_kobj = [kobj_hubble, kobj_kepler, kobj_k2, kobj_copernicus]


@pytest.fixture(scope="module")
def paper_setup():
    """
    Shared setup fixture for tests
    """
    testpaper = paper.Paper(text="", keyword_objs=[params.kobj_hubble], do_check_truematch=False)
    dict_ambigs = testpaper._process_database_ambig(keyword_objs=test_list_lookup_kobj)
    return testpaper, dict_ambigs


# -------------------------------
# Test _check_truematch
# -------------------------------
@pytest.mark.parametrize(
    "phrase,lookup,expected",
    [
        ("small Hubble constant", "Kepler", False),
        ("small Hubble's constant", "Kepler", False),
        ("Edwin Hubble's papers", "Hubble", False),
        ("high S/N Hubble image", "Hubble", True),
        ("HST observatory", "Hubble", True),
        ("H.S.T. observatory", "Hubble", True),
        ("Hubble calibrated images", "Hubble", True),
        ("Hubble's calibrated data", "Hubble", True),
        ("Hubble's pretty spectra", "Hubble", True),
        ("Edwin Hubble's analysis", "Hubble", False),
        ("A Hubble constant data", "Hubble", False),
        ("Hubble et al. 2000", "Hubble", False),
        ("Hubbleetal 2000", "Hubble", False),
        ("Hubble and more data.", "Hubble", True),
        ("Kepler fields.", "Kepler", True),
        ("Kepler velocities.", "Kepler", False),
        ("Kepler velocity fields.", "Kepler", False),
        ("Kepler rotation velocity fields.", "Kepler", False),
        ("that Kepler data velocity.", "Kepler", True),
        ("true Kepler planets", "Kepler", False),
        ("those Kepler radii", "Kepler", False),
        ("Keplerian orbits", "Kepler", False),
        ("Kepler's law", "Kepler", False),
        ("Kepler observations", "Kepler", True),
        ("K2 database", "K2", True),
        ("K2-123 star", "K2", False),
        ("K2 stars", "K2", False),
        ("Copernicus satellite", "Copernicus", True),
        ("Copernicus model", "Copernicus", False),
    ],
)
def test_check_truematch(paper_setup, phrase, lookup, expected):
    testpaper, dict_ambigs = paper_setup
    curr_kobjs = [test_dict_lookup_kobj[lookup]]
    test_res = testpaper._check_truematch(
        text=phrase,
        keyword_objs=curr_kobjs,
        dict_ambigs=dict_ambigs,
    )
    assert test_res["bool"] == expected


# -------------------------------
# Test _early_false_no_keyword_match
# -------------------------------
@pytest.mark.parametrize(
    "phrase,lookup,expected",
    [
        ("small Hubble constant", "Kepler", False),
        ("small Hubble's constant", "Kepler", False),
        ("Edwin Hubble's papers", "Hubble", None),
        ("high S/N Hubble image", "Hubble", None),
        ("HST observatory", "Hubble", None),
        ("H.S.T. observatory", "Hubble", None),
        ("Hubble et al. 2000", "Hubble", None),
        ("Hubbleetal 2000", "Hubble", False),
    ],
)
def test_early_false_no_keyword_match(paper_setup, phrase, lookup, expected):
    testpaper, dict_ambigs = paper_setup
    curr_kobjs = [test_dict_lookup_kobj[lookup]]
    setup_data = testpaper._setup_check_truematch_vars(
        text=phrase,
        keyword_objs=curr_kobjs,
        dict_ambigs=dict_ambigs,
    )
    test_res = testpaper._early_false_no_keyword_match(setup_data)
    if test_res is not None:
        test_res = test_res["bool"]
    assert test_res == expected


# -------------------------------
# Test _early_true_acronym_match
# -------------------------------
@pytest.mark.parametrize(
    "phrase,lookup,expected",
    [
        ("small Hubble constant", "Kepler", None),
        ("small Hubble's constant", "Kepler", None),
        ("Edwin Hubble's papers", "Hubble", None),
        ("high S/N Hubble image", "Hubble", None),
        ("HST observatory", "Hubble", True),
        ("H.S.T. observatory", "Hubble", True),
        ("Figure 1 plots the Hubble Space Telescope (HST) observations.", "Hubble", True),
        ("We summarize our Hubble results next.", "Hubble", None),
    ],
)
def test_early_true_acronym_match(paper_setup, phrase, lookup, expected):
    testpaper, dict_ambigs = paper_setup
    curr_kobjs = [test_dict_lookup_kobj[lookup]]
    setup_data = testpaper._setup_check_truematch_vars(
        text=phrase,
        keyword_objs=curr_kobjs,
        dict_ambigs=dict_ambigs,
    )
    test_res = testpaper._early_true_acronym_match(setup_data)
    if test_res is not None:
        test_res = test_res["bool"]
    assert test_res == expected


# -------------------------------
# Test _early_true_non_ambig_phrases
# -------------------------------
@pytest.mark.parametrize(
    "phrase,lookup,expected",
    [
        (
            "The Hubble Space Telescope is a really neat telescope that is often referred to as Hubble or HST.",
            "Hubble",
            True,
        ),
        (
            "Once the transit is observable, we will observe the target in a timely fashion without delay and at high resolution using either JWST or Hubble.",
            "Hubble",
            None,
        ),
        ("Edwin Hubble's papers", "Hubble", None),
        ("high S/N Hubble image", "Hubble", None),
        ("HST observatory", "Hubble", None),
        ("H.S.T. observatory", "Hubble", None),
        ("Figure 1 plots the Hubble Space Telescope (HST) observations.", "Hubble", True),
        ("We summarize our Hubble results next.", "Hubble", None),
    ],
)
def test_early_true_non_ambig_phrases(paper_setup, phrase, lookup, expected):
    testpaper, dict_ambigs = paper_setup
    curr_kobjs = [test_dict_lookup_kobj[lookup]]
    setup_data = testpaper._setup_check_truematch_vars(
        text=phrase,
        keyword_objs=curr_kobjs,
        dict_ambigs=dict_ambigs,
    )
    test_res = testpaper._early_true_non_ambig_phrases(setup_data)
    if test_res is not None:
        test_res = test_res["bool"]
    assert test_res == expected


# -------------------------------
# Test _assemble_keyword_wordchunks_wrapper
# -------------------------------
@pytest.mark.parametrize(
    "phrase,lookup,expected",
    [
        ("Hubble calibrated images", "Hubble", ["Hubble calibrated images"]),
        ("Hubble's calibrated data", "Hubble", ["Hubble's calibrated data"]),
        ("Hubble and more data.", "Hubble", ["Hubble"]),
        ("Kepler fields.", "Kepler", ["Kepler fields"]),
        ("We summarize our Hubble results next.", "Hubble", ["our Hubble results"]),
        ("The Kepler images and Kepler plots indicate a correlation.", "Kepler", ["Kepler images", "Kepler plots"]),
        (
            "Table 1 then gives the measured Kepler velocity data and Kepler planets.",
            "Kepler",
            ["measured Kepler velocity data", "Kepler planets"],
        ),
    ],
)
def test_assemble_keyword_wordchunks_wrapper(paper_setup, phrase, lookup, expected):
    testpaper, dict_ambigs = paper_setup
    curr_kobjs = [test_dict_lookup_kobj[lookup]]
    setup_data = testpaper._setup_check_truematch_vars(
        text=phrase,
        keyword_objs=curr_kobjs,
        dict_ambigs=dict_ambigs,
    )
    test_res = testpaper._assemble_keyword_wordchunks_wrapper(setup_data)
    test_res = [repr(item) for item in test_res]
    assert test_res == expected


# -------------------------------
# Test _early_true_exact_wordchunk
# -------------------------------
@pytest.mark.parametrize(
    "phrase,lookup,expected",
    [
        ("Hubble calibrated images", "Hubble", None),
        ("Hubble's calibrated data", "Hubble", None),
        ("Hubble and more data.", "Hubble", True),
        ("Kepler fields.", "Kepler", None),
        ("We summarize our Hubble results next.", "Hubble", None),
        ("The Kepler images and Kepler plots indicate a correlation.", "Kepler", None),
        ("Table 1 then gives the measured Kepler velocity data and Kepler planets.", "Kepler", None),
    ],
)
def test_early_true_exact_wordchunk(paper_setup, phrase, lookup, expected):
    testpaper, dict_ambigs = paper_setup
    curr_kobjs = [test_dict_lookup_kobj[lookup]]
    setup_data = testpaper._setup_check_truematch_vars(
        text=phrase,
        keyword_objs=curr_kobjs,
        dict_ambigs=dict_ambigs,
    )
    list_wordchunks = testpaper._assemble_keyword_wordchunks_wrapper(setup_data)
    test_res = testpaper._early_true_exact_wordchunk(list_wordchunks, setup_data)
    if test_res is not None:
        test_res = test_res["bool"]
    assert test_res == expected


# -------------------------------
# Test _consider_wordchunk
# -------------------------------
@pytest.mark.parametrize(
    "phrase,lookup,expected",
    [
        ("Hubble's calibrated data", "Hubble", [("Hubble's calibrated data", True, "AnyMission data")]),
        ("Hubble's pretty spectra", "Hubble", [("Hubble's pretty spectra", True, "AnyMission spectrum")]),
        ("Hubble et al. 2000", "Hubble", [("Hubble et al", False, "AnyMission et al")]),
        (
            "Table 1 then gives the measured Kepler velocity data and Kepler planets.",
            "Kepler",
            [
                ("measured Kepler velocity data", False, "Kepler velocity"),
                ("Kepler planets", False, "Kepler planet"),
            ],
        ),
        (
            "The Kepler images and Kepler plots indicate a correlation.",
            "Kepler",
            [
                ("Kepler images", True, "AnyMission image"),
                ("Kepler plots", True, "AnyMission plot"),
            ],
        ),
    ],
)
def test_consider_wordchunk(paper_setup, phrase, lookup, expected):
    testpaper, dict_ambigs = paper_setup
    curr_kobjs = [test_dict_lookup_kobj[lookup]]
    setup_data = testpaper._setup_check_truematch_vars(
        text=phrase,
        keyword_objs=curr_kobjs,
        dict_ambigs=dict_ambigs,
    )
    list_wordchunks = testpaper._assemble_keyword_wordchunks_wrapper(setup_data)
    list_results = [testpaper._consider_wordchunk(chunk, setup_data) for chunk in list_wordchunks]
    test_res = [
        (repr(chunk), result["info"][0]["bool"], result["info"][0]["text_database"])
        for chunk, result in zip(list_wordchunks, list_results)
    ]
    assert test_res == expected


# -------------------------------
# Test _setup_consider_wordchunk
# -------------------------------
@pytest.mark.parametrize(
    "phrase,lookup,expected",
    [
        (
            "Hubble's calibrated data",
            "Hubble",
            [("Hubble's calibrated data", "hubble calibrated data datum", ["hubble"])],
        ),
        (
            "Edwin Hubble's papers",
            "Hubble",
            [("Edwin Hubble's papers", "edwin hubble composition document newspaper paper", ["hubble"])],
        ),
        ("Hubble et al. 2000", "Hubble", [("Hubble et al", "hubble et alabama aluminum", ["hubble"])]),
        (
            "Kepler velocity fields.",
            "Kepler",
            [
                (
                    "Kepler velocity fields",
                    "kepler speed airfield battlefield discipline field fields plain playing_field sphere",
                    ["kepler"],
                )
            ],
        ),
        (
            "We summarize our Hubble results next.",
            "Hubble",
            [("our Hubble results", "our hubble consequence result resultant_role solution", ["hubble"])],
        ),
    ],
)
def test_setup_consider_wordchunk(paper_setup, phrase, lookup, expected):
    testpaper, dict_ambigs = paper_setup
    curr_kobjs = [test_dict_lookup_kobj[lookup]]
    setup_data = testpaper._setup_check_truematch_vars(
        text=phrase,
        keyword_objs=curr_kobjs,
        dict_ambigs=dict_ambigs,
    )
    list_wordchunks = testpaper._assemble_keyword_wordchunks_wrapper(setup_data)
    list_results = [testpaper._setup_consider_wordchunk(chunk, setup_data) for chunk in list_wordchunks]
    test_res = [(repr(chunk), result[0], result[1]) for chunk, result in zip(list_wordchunks, list_results)]
    assert test_res == expected


# -------------------------------
# Test _extract_ambig_phrases_substrings
# -------------------------------
@pytest.mark.parametrize(
    "phrase,lookup,expected",
    [
        ("Edwin Hubble's papers", "Hubble", {"Edwin Hubble's papers": {"matches": 1, "meanings": 1}}),
        ("A Hubble constant data", "Hubble", {"Hubble constant data": {"matches": 1, "meanings": 2}}),
        ("Hubble et al. 2000", "Hubble", {"Hubble et al": {"matches": 0, "meanings": 1}}),
        ("Kepler velocity fields.", "Kepler", {"Kepler velocity fields": {"matches": 0, "meanings": 2}}),
        (
            "Table 1 then gives the measured Kepler velocity data and Kepler planets.",
            "Kepler",
            {
                "measured Kepler velocity data": {"matches": 0, "meanings": 2},
                "Kepler planets": {"matches": 1, "meanings": 2},
            },
        ),
    ],
)
def test_extract_ambig_phrases_substrings(paper_setup, phrase, lookup, expected):
    testpaper, dict_ambigs = paper_setup
    curr_kobjs = [test_dict_lookup_kobj[lookup]]
    setup_data = testpaper._setup_check_truematch_vars(
        text=phrase,
        keyword_objs=curr_kobjs,
        dict_ambigs=dict_ambigs,
    )
    list_wordchunks = testpaper._assemble_keyword_wordchunks_wrapper(setup_data)
    test_res = {}
    for chunk in list_wordchunks:
        meaning, inner_kw = testpaper._setup_consider_wordchunk(chunk, setup_data)
        matches = testpaper._extract_ambig_phrases_substrings(
            setup_data.list_exp_exact_ambigs, "matches", chunk.text, meaning, inner_kw, setup_data
        )
        meanings = testpaper._extract_ambig_phrases_substrings(
            setup_data.list_exp_meaning_ambigs, "meanings", chunk.text, meaning, inner_kw, setup_data
        )
        test_res[chunk.text] = {"matches": len(matches), "meanings": len(meanings)}
    assert test_res == expected


# -------------------------------
# Test _assemble_consider_wordchunk_results
# -------------------------------
@pytest.mark.parametrize(
    "phrase,lookup,expected",
    [
        (
            "Edwin Hubble's papers",
            "Hubble",
            {"Edwin Hubble's papers": ["matcher", "bool", "text_wordchunk", "text_database"]},
        ),
        (
            "A Hubble constant data",
            "Hubble",
            {"Hubble constant data": ["matcher", "bool", "text_wordchunk", "text_database"]},
        ),
        ("Hubble et al. 2000", "Hubble", {"Hubble et al": ["matcher", "bool", "text_wordchunk", "text_database"]}),
        (
            "Kepler velocity fields.",
            "Kepler",
            {"Kepler velocity fields": ["matcher", "bool", "text_wordchunk", "text_database"]},
        ),
        (
            "Table 1 then gives the measured Kepler velocity data and Kepler planets.",
            "Kepler",
            {
                "measured Kepler velocity data": ["matcher", "bool", "text_wordchunk", "text_database"],
                "Kepler planets": ["matcher", "bool", "text_wordchunk", "text_database"],
            },
        ),
    ],
)
def test_assemble_consider_wordchunk_results(paper_setup, phrase, lookup, expected):
    testpaper, dict_ambigs = paper_setup
    curr_kobjs = [test_dict_lookup_kobj[lookup]]
    setup_data = testpaper._setup_check_truematch_vars(
        text=phrase,
        keyword_objs=curr_kobjs,
        dict_ambigs=dict_ambigs,
    )
    list_wordchunks = testpaper._assemble_keyword_wordchunks_wrapper(setup_data)
    test_res = {}
    for chunk in list_wordchunks:
        meaning, inner_kw = testpaper._setup_consider_wordchunk(chunk, setup_data)
        set_matches = testpaper._extract_ambig_phrases_substrings(
            setup_data.list_exp_exact_ambigs, "matches", chunk.text, meaning, inner_kw, setup_data
        ) or testpaper._extract_ambig_phrases_substrings(
            setup_data.list_exp_meaning_ambigs, "meanings", chunk.text, meaning, inner_kw, setup_data
        )
        list_results = testpaper._assemble_consider_wordchunk_results(set_matches, chunk, meaning, setup_data)
        test_res[chunk.text] = list(list_results["info"][0].keys())
    assert test_res == expected
