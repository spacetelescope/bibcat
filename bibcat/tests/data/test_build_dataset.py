import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import bibcat.data.build_dataset as bd
from bibcat.utils.utils import save_json_file


def test_file_exists(tmp_path: Path) -> None:
    f1 = tmp_path / "file1.txt"
    f2 = tmp_path / "file2.txt"

    (f1).touch()
    assert bd.file_exists([f1, f2]) == os.path.isfile(f1)
    assert not (f2).exists()


def test_save_text_file(tmp_path: Path) -> None:
    bibcodes = ["bibcode1", "bibcode2"]
    tmp_file = tmp_path / "temp_file.txt"

    # Save bibcodes to a text file, this tests save_test_file() too.
    bd.save_text_file(tmp_file, bibcodes)

    assert tmp_file.exists(), "File was not created."
    # Load bibcodes back from the file for verification
    with tmp_file.open(mode="r") as file:
        loaded_bibcodes = file.read().splitlines()
    assert loaded_bibcodes == bibcodes, "loaded bibcodes do not match the input bibcodes."


@pytest.fixture
def mockjson(tmp_path: Path):
    # Create mock papertext and papertrack files using mock dictionaries for test
    papertext: list[dict] = [
        {
            "bibcode": "2021MNRAS.500.3002B",
            "pubdate": "2021-01-00",
            "abstract": "This is a JWST abstract.",
            "body": "This is a JWST science paper.",
        },
        {
            "bibcode": "2023ApJ.200.4008B",
            "pubdate": "2023-01-00",
            "abstract": "This is a HST abstract.",
            "body": "This is a HST mention paper.",
        },
    ]

    papertrack: list[dict] = [
        {
            "bibcode": "2021MNRAS.500.3002B",
            "searches": [
                {"search_key": "jwst", "ignored": False},
            ],
            "class_missions": [{"mission": "JWST", "paper_type": "SCIENCE"}],
        },
        {
            "bibcode": "2024AA.345.1052C",
            "searches": [
                {"search_key": "kepler", "ignored": False},
            ],
            "class_missions": [{"mission": "Kepler", "paper_type": "MENTION"}],
        },
    ]

    trimmed_papertext: list[dict] = [
        {
            "bibcode": "2021MNRAS.500.3002B",
            "pubdate": "2021-01-00",
            "body": "This is a JWST science paper.",
        },
        {
            "bibcode": "2023ApJ.200.4008B",
            "pubdate": "2023-01-00",
            "body": "This is a HST mention paper.",
        },
    ]
    combined_dataset: list[dict] = [
        {
            "bibcode": "2021MNRAS.500.3002B",
            "pubdate": "2021-01-00",
            "body": "This is a JWST science paper.",
            "class_missions": {"JWST": {"bibcode": "2021MNRAS.500.3002B", "papertype": "SCIENCE"}},
            "is_ignored_jwst": False,
        }
    ]
    path_papertext = tmp_path / "papertext.json"
    path_papertrack = tmp_path / "papertrack.json"

    yield (
        path_papertext,
        papertext,
        path_papertrack,
        papertrack,
        trimmed_papertext,
        combined_dataset,
    )


def test_build_datasets(mockjson: Any) -> None:
    (
        mock_path_papertext,
        expected_papertext,
        mock_path_papertrack,
        expected_papertrack,
        expected_trimmed_papertext,
        expected_combined_dataset,
    ) = mockjson

    save_json_file(mock_path_papertext, expected_papertext)
    save_json_file(mock_path_papertrack, expected_papertrack)
    assert mock_path_papertext.exists(), f"mock papertext file was not created."
    assert mock_path_papertrack.exists(), f"mock papertrack file was not created."

    loaded_papertext, loaded_papertrack = bd.load_datasets(mock_path_papertext, mock_path_papertrack)
    assert loaded_papertext == expected_papertext, "loaded papertext data does not match the expected output"
    assert loaded_papertrack == expected_papertrack, "loaded papertrack data does not match the expected output"

    trimmed_papertext = bd.trim_dict(loaded_papertext, ["bibcode", "pubdate", "body"])
    assert trimmed_papertext == expected_trimmed_papertext, "trimmed papertext doesn't match the expected output"

    combined_dataset, bibcodes_notin_papertext, bibcodes_notin_papertrack, papertext_index_notin_papertrack = (
        bd.combine_datasets(trimmed_papertext, loaded_papertrack)
    )
    assert combined_dataset == expected_combined_dataset, "combinned dataset doesn't match the expected output"
    np.testing.assert_array_equal(
        bibcodes_notin_papertext,
        ["2024AA.345.1052C"],
        err_msg="Bibcodes not in papertext do not match expected values.",
    )
    np.testing.assert_array_equal(
        bibcodes_notin_papertrack,
        ["2023ApJ.200.4008B"],
        err_msg="Bibcodes not in papertrack do not match expected values.",
    )
    assert papertext_index_notin_papertrack == [1], "papertext data not in papertrack data does not match the expected"
