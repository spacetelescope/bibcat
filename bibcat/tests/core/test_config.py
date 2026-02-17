from pathlib import Path
from typing import Any, Dict, Generator

import pytest
import yaml  # type: ignore

from bibcat.core.config import get_config, get_default_config


def test_get_config_nouser() -> None:
    """test we can get the default config"""
    cc = get_default_config()
    assert cc["logging"]["verbose"] == False
    assert cc.logging.verbose == False


@pytest.fixture
def fakeyaml(tmp_path: Path) -> Generator[Path, Any, None]:
    data: Dict[Any, Any]
    data = {"logging": {"level": "CUSTOM", "verbose": True}}
    data["stuff"] = {"a": 1, "b": 2, "c": {"hello": "there", "yes": "no"}}

    path = tmp_path / "bibcat_config.yaml"
    with open(path, "w") as f:
        f.write(yaml.dump(data))

    yield path


def test_user_config(monkeypatch: pytest.MonkeyPatch, fakeyaml: Any) -> None:
    """test we can read in a custom config"""
    monkeypatch.setenv("BIBCAT_CONFIG_DIR", str(fakeyaml.parent))
    cc = get_config()
    assert cc["logging"]["level"] == "CUSTOM"
    assert cc["logging"]["verbose"] == True
    assert cc.stuff.b == 2
    assert cc.stuff.c.hello == "there"
