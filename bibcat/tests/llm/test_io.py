import re

import pytest

from bibcat import config
from bibcat.llm.io import get_file, get_llm_prompt, get_source

# expected data
data = [
    {"bibcode": "1234567890", "body": "This is the body of the source dataset."},
    {"bibcode": "0987654321", "body": "This is the body of the second source dataset."},
]


@pytest.fixture()
def make_paper(tmp_path):
    """fixture factory to create a temporary paper file"""
    d = tmp_path / "papers"
    d.mkdir()
    p = d / "new_paper.pdf"
    p.write_text("This is the body of the paper.")
    yield str(p)


# TODO - clean up this config mocking stuff with what it's conftest
@pytest.fixture()
def ss(reconfig):
    """fixture to set the config object"""
    yield reconfig("bibcat.llm.io.config")


@pytest.fixture()
def make_tempfile(monkeypatch, tmp_path):
    """fixture factory to create a temporary custom prompt file"""

    def _make_temp(prompt, content):
        d = tmp_path / "llm"
        d.mkdir()
        p = d / f"llm_{prompt}_prompt.txt"
        p.write_text(content)
        monkeypatch.setitem(config.llms, f"llm_{prompt}_prompt", str(p))
        return str(p)

    yield _make_temp


@pytest.mark.parametrize(
    "bibcode, index, body_only, exp",
    [
        ("1234567890", None, False, "body of the source dataset"),
        (None, 1, False, "body of the second source dataset"),
        (None, 1, True, "body of the second source dataset"),
    ],
    ids=["bibcode", "index", "index_body_only"],
)
def test_get_source(mocker, bibcode, index, body_only, exp):
    """test we can get source data"""
    mocker.patch("bibcat.llm.io.load_source_dataset", return_value=data)

    result = get_source(bibcode=bibcode, index=index, body_only=body_only)
    assert isinstance(result, dict) or isinstance(result, str)
    if body_only:
        assert exp in result
    else:
        assert exp in result["body"]


@pytest.mark.parametrize(
    "filepath, bibcode, index, expfile",
    [
        (True, None, None, ".*new_paper.pdf"),  # Test case 1: filepath is provided
        (None, "1234567890", None, "temp_.*_1234567890.json"),  # Test case 2: bibcode is provided
        (None, None, 1, "temp_.*_0987654321.json"),  # Test case 3: index is provided
    ],
    ids=["filepath", "bibcode", "index"],
)
def test_get_file(mocker, make_paper, filepath, bibcode, index, expfile):
    """test we can get a file"""
    mocker.patch("bibcat.llm.io.load_source_dataset", return_value=data)

    if filepath is not None:
        filepath = str(make_paper)

    result = get_file(filepath=filepath, bibcode=bibcode, index=index)
    assert re.search(expfile, result)


@pytest.mark.parametrize(
    "prompt, exp",
    [
        ("user", "Carefully follow these two instructions to classify papers for MAST bibliometric record-keeping"),
        ("agent", "You are an assistant with expertise in astronomical bibliographic and library systems"),
    ],
)
def test_default_get_llm_prompt(prompt, exp):
    """test we get the correct prompt from the default config"""
    result = get_llm_prompt(prompt)
    assert isinstance(result, str)
    assert exp in result


@pytest.mark.parametrize(
    "prompt, exp",
    [("user", "Tell me about this paper."), ("agent", "Read the paper given and extract some info.")],
    ids=["user", "agent"],
)
def test_custom_config_llm_prompt(mocker, ss, monkeypatch, prompt, exp):
    """test we get the correct prompt from a custom config"""
    # mock the config data dir, so we can test the config prompts
    monkeypatch.setenv("BIBCAT_DATA_DIR", "")
    mocker.patch("bibcat.llm.io.config", new=ss)
    from bibcat.llm.io import config

    monkeypatch.setitem(config.llms, f"{prompt}_prompt", exp)

    result = get_llm_prompt(prompt)
    assert isinstance(result, str)
    assert exp in result


@pytest.mark.parametrize(
    "prompt, content, exp",
    [
        ("user", "Hello.  Please tell me about this paper.", "tell me about this paper."),
        (
            "agent",
            "You are an information extractor.  Read the paper given and extract some info. Return something good",
            "Read the paper given and extract some info.",
        ),
    ],
    ids=["user", "agent"],
)
def test_custom_file_llm_prompt(make_tempfile, prompt, content, exp):
    """test we get the correct prompt from a custom file"""
    make_tempfile(prompt, content)
    result = get_llm_prompt(prompt)
    assert isinstance(result, str)
    assert exp in result


def test_get_llm_fail():
    """test we fail correctly"""
    with pytest.raises(ValueError, match='Prompt type must be either "user" or "agent"'):
        get_llm_prompt("bad_prompt")
