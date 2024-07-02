

import os
import tempfile

import pytest

from bibcat import setup_paths
from bibcat.core.config import get_default_config


def pytest_sessionstart(session):
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Set the environment variable
    os.environ['BIBCAT_OUTPUT_DIR'] = temp_dir


@pytest.fixture(scope='session', autouse=True)
def setenv(tmp_path_factory):
    """ globally set the bibcat envvars """
    root = tmp_path_factory.mktemp("temp")
    output = root / "output"
    cc = root / "config"
    data = root / "data"
    ops = root / "ops"
    for d in [output, cc, data, ops]:
        d.mkdir(parents=True, exist_ok=True)

    os.environ['BIBCAT_OUTPUT_DIR'] = str(output)
    os.environ['BIBCAT_CONFIG_DIR'] = str(cc)
    os.environ['BIBCAT_DATA_DIR'] = str(data)
    os.environ['BIBCAT_OPSDATA_DIR'] = str(ops)


@pytest.fixture()
def setconfig(mocker):
    """ fixture to patch the global config object in a particular module directory """
    cfg = setup_paths(get_default_config())
    def _setconfig(moddir):
        mocker.patch(moddir, new=cfg)
        return cfg
    yield _setconfig

