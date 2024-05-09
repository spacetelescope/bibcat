
import pytest
import yaml

from bibcat.core.config import get_config


def test_get_config_nouser():
    """ test we can get the default config """
    cc = get_config()
    assert cc['input'] == 'a'
    assert cc.input == 'a'


@pytest.fixture
def fakeyaml(tmp_path):
    data = {'model': 'new_model'}
    data['stuff'] = {'a': 1, 'b': 2, 'c': {'hello': 'there', 'yes': 'no'}}

    path = tmp_path / "bibcat_config.yaml"
    with open(path, 'w') as f:
        f.write(yaml.dump(data))

    yield path


def test_user_config(monkeypatch, fakeyaml):
    """ test we can read in a custom config """
    monkeypatch.setenv('BIBCAT_CONFIG_DIR', str(fakeyaml.parent))
    cc = get_config()
    assert cc['input'] == 'a'
    assert cc['model'] != 'b'
    assert cc['model'] == 'new_model'
    assert cc.stuff.c.hello == 'there'


