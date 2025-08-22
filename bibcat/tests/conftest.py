import importlib
import json
import os
import tempfile

import pytest

from bibcat import setup_paths
from bibcat.core.config import get_config, get_default_config


def pytest_sessionstart(session):
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Set the environment variable
    os.environ["BIBCAT_OUTPUT_DIR"] = temp_dir


@pytest.fixture(scope="session", autouse=True)
def setenv(tmp_path_factory):
    """globally set the bibcat envvars"""
    root = tmp_path_factory.mktemp("temp")
    output = root / "output"
    cc = root / "config"
    data = root / "data"
    ops = root / "ops"
    for d in [output, cc, data, ops]:
        d.mkdir(parents=True, exist_ok=True)

    os.environ["BIBCAT_OUTPUT_DIR"] = str(output)
    os.environ["BIBCAT_CONFIG_DIR"] = str(cc)
    os.environ["BIBCAT_DATA_DIR"] = str(data)
    os.environ["BIBCAT_OPSDATA_DIR"] = str(ops)


@pytest.fixture()
def setconfig(mocker):
    """fixture to patch the global config object in a particular module directory"""
    cfg = setup_paths(get_default_config())

    def _setconfig(moddir):
        mocker.patch(moddir, new=cfg)
        return cfg

    yield _setconfig


@pytest.fixture()
def reconfig(mocker):
    """fixture to patch the global config object in a particular module directory"""
    cfg = setup_paths(get_config())

    def _setconfig(moddir):
        mocker.patch(moddir, new=cfg)
        return cfg

    yield _setconfig


@pytest.fixture()
def fixconfig(reconfig, mocker, monkeypatch):
    """fixture factory to patch a config object in a particular module

    this sets a temporary output directory, updates the config object for
    the given module, e.g. bibcat.llm.io.config, and returns the new config object for use
    in tests.
    """

    def _fixconfig(root, configmod):
        cfg = f"{configmod}.config"
        ss = reconfig(cfg)
        monkeypatch.setenv("BIBCAT_DATA_DIR", root)
        mocker.patch(cfg, new=ss)
        mod = importlib.import_module(configmod)
        return mod.config

    yield _fixconfig


# fmt: off
example_request = {'method': 'POST', 'url': '/v1/responses', 'body': {'model': 'gpt-4.1-mini', 'instructions': 'please check this', 'input': 'what is this?', 'text': {'format': {'type': 'json_schema', 'strict': True, 'name': 'InfoModel', 'schema': {'$defs': {'MissionEnum': {'enum': ['HST', 'JWST', 'Roman', 'TESS', 'KEPLER', 'K2', 'GALEX', 'PanSTARRS', 'FUSE', 'IUE', 'HUT', 'UIT', 'WUPPE', 'BEFS', 'TUES', 'IMAPS', 'EUVE'], 'title': 'MissionEnum', 'type': 'string'}, 'MissionInfo': {'description': 'Pydantic model for a mission entry', 'properties': {'mission': {'description': 'The name of the mission.', 'enum': ['HST', 'JWST', 'Roman', 'TESS', 'KEPLER', 'K2', 'GALEX', 'PanSTARRS', 'FUSE', 'IUE', 'HUT', 'UIT', 'WUPPE', 'BEFS', 'TUES', 'IMAPS', 'EUVE'], 'title': 'MissionEnum', 'type': 'string'}, 'papertype': {'description': 'The type of paper you think it is', 'enum': ['SCIENCE', 'MENTION'], 'title': 'PapertypeEnum', 'type': 'string'}, 'quotes': {'description': 'A list of exact quotes from the paper that support your reason', 'items': {'type': 'string'}, 'title': 'Quotes', 'type': 'array'}, 'reason': {'description': 'A short sentence summarizing your reasoning for classifying this mission + papertype', 'title': 'Reason', 'type': 'string'}, 'confidence': {'description': 'Two float values representing confidence for SCIENCE and MENTION. Must sum to 1.0.', 'items': {'type': 'number'}, 'title': 'Confidence', 'type': 'array'}}, 'required': ['mission', 'papertype', 'quotes', 'reason', 'confidence'], 'title': 'MissionInfo', 'type': 'object', 'additionalProperties': False}, 'PapertypeEnum': {'description': 'Enumeration of paper types for classification', 'enum': ['SCIENCE', 'MENTION'], 'title': 'PapertypeEnum', 'type': 'string'}}, 'description': 'Pydantic model for the parsed response from the LLM', 'properties': {'notes': {'description': 'all your notes and thoughts you have written down during your process', 'title': 'Notes', 'type': 'string'}, 'missions': {'description': 'a list of your identified missions', 'items': {'$ref': '#/$defs/MissionInfo'}, 'title': 'Missions', 'type': 'array'}}, 'required': ['notes', 'missions'], 'title': 'InfoModel', 'type': 'object', 'additionalProperties': False}}}}}


@pytest.fixture()
def batchfile(tmp_path):
    """fixture for creating a batch file"""

    def _make_batch(n_samp):
        path = tmp_path / "batch.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for i in range(1, n_samp + 1):
                data = {"custom_id": f"bc_{i:03}"} | example_request
                f.write(json.dumps(data) + "\n")
        return path

    return _make_batch
