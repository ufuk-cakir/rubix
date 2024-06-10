import pytest
from rubix.config.user import UserConfig
import json


# Tests for UserConfig
def test_userconfig_getitem():
    config = UserConfig({"data": {"args": {"arg1": "value1"}}})

    assert config["data/args/arg1"] == "value1"

    with pytest.raises(KeyError):
        _ = config["data/args/arg2"]

    with pytest.raises(KeyError):
        _ = config["data/wrongkey"]

    with pytest.raises(TypeError):
        _ = config[123]


def test_userconfig_get():
    config = UserConfig({"data": {"args": {"arg1": "value1"}}})

    assert config.get("data/args/arg1") == "value1"
    assert config.get("data/args/arg2") is None
    assert config.get("data/args/arg2", "default") == "default"


def test_userconfig_contains_typeerror():
    config = UserConfig(
        {"data": {"args": {"arg1": "value1", "arg2": ["not", "a", "dict"]}}}
    )

    # Check if __contains__ returns False for non-string key types
    assert not (123 in config)
    assert not (["data", "args", "arg1"] in config)

    # Check if __contains__ handles TypeError when encountering non-dict during traversal
    assert not ("data/args/arg2/somekey" in config)


def test_userconfig_contains():
    config = UserConfig({"data": {"args": {"arg1": "value1"}}})

    assert "data/args/arg1" in config
    assert "data/args/arg2" not in config
    assert "data/wrongkey" not in config
    assert 123 not in config


def test_userconfig_str_repr():
    config_dict = {"data": {"args": {"arg1": "value1"}}}
    config = UserConfig(config_dict)
    assert str(config) == json.dumps(config_dict, indent=4)
    assert repr(config) == f"UserConfig({json.dumps(config_dict, indent=4)})"
