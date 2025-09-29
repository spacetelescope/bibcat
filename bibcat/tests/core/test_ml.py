import pytest

from bibcat import config
from bibcat.core.classifiers.ml import (
    MachineLearningClassifier,
    TensorFlow,
    select_library,
)


@pytest.fixture()
def ss(setconfig):
    """fixture to set the ml config object to the default one"""
    setconfig("bibcat.core.classifiers.ml.config")


def test_select_library():
    """test we get the right classifier"""
    classifier = select_library("tensorflow")
    assert issubclass(classifier, TensorFlow)

    classifier = select_library("junk")
    assert classifier is None


def test_defaut_tensorflow(ss):
    """test the default tensorflow init"""
    tt = TensorFlow()
    assert tt.model_type == "bert"
    assert tt.model_key == "small_bert/bert_en_uncased_L-4_H-512_A-8"
    assert tt.loaded is False
    assert tt.savename_ML == "tfoutput_tf_bert_run"


def test_default_ml():
    """test the default ml init"""
    mm = MachineLearningClassifier()
    assert isinstance(mm.model, TensorFlow)
    assert mm.model.loaded is False


def test_new_tensorflow_model_from_config(monkeypatch):
    """test we can load a new model from the config"""
    monkeypatch.setitem(config.output, "name_model", "tf_roberta_run")
    monkeypatch.setitem(config.ml, "ML_model_type", "roberta")
    monkeypatch.setitem(config.ml, "ML_model_key", "roberta_encased")

    tt = TensorFlow()
    assert tt.model_type == "roberta"
    assert tt.model_key == "roberta_encased"
    assert tt.loaded is False
    assert tt.savename_ML == "tfoutput_tf_roberta_run"


def test_new_tensorflow_model_from_init(ss):
    """test we can load a new model from the init"""
    tt = TensorFlow(model_type="roberta", model_key="roberta_encased")
    assert tt.model_type == "roberta"
    assert tt.model_key == "roberta_encased"
    assert tt.loaded is False
    assert tt.savename_ML == "tfoutput_tf_bert_run"


@pytest.mark.parametrize(
    "name, key, err",
    [
        ("junk", "junk", "Model type junk not found in config.ml"),
        ("bert", "junk", "Model key junk not found in config.ml.bert"),
    ],
    ids=["bad_model", "bad_key"],
)
def test_new_tf_fails(name, key, err):
    """test we can't load a new model from the init"""
    with pytest.raises(KeyError, match=err):
        TensorFlow(model_type=name, model_key=key)
