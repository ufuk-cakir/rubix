import json
from functools import reduce


class UserConfig:
    def __init__(self, config: dict):
        self.config = config

    # def __getitem__(self, key):
    #     keys = key.split("/")
    #
    #     value = self.config
    #     for k in keys:
    #         # Check if it exists, otherwise raise error
    #         if k not in value:
    #             raise KeyError(f"Key {k} not found in config located at {key}: {value}")
    #         value = value[k]
    #     return value
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, key):
        # key can have / to access nested keys
        # e.g. key = "data/args"
        # returns self.config["data"]["args"]
        # or key = "data/args/arg1"
        # returns self.config["data"]["args"]["arg1"]
        if not isinstance(key, str):
            raise TypeError(
                f"Key must be a string, but got {type(key).__name__}: {key}"
            )
        keys = key.split("/")
        try:
            return reduce(lambda d, k: d[k], keys, self.config)
        except KeyError:
            raise KeyError(f"Key {key} not found in config")

    def __contains__(self, key):
        if not isinstance(key, str):
            return False

        keys = key.split("/")
        try:
            reduce(lambda d, k: d[k], keys, self.config)
            return True
        except KeyError:
            return False
        except TypeError:
            return False

    def __str__(self):
        return json.dumps(self.config, indent=4)

    def __repr__(self):
        return f"UserConfig({json.dumps(self.config, indent=4)})"
