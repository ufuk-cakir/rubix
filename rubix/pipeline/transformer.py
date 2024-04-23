from abc import ABC, abstractmethod


class TransformerFactoryBase(ABC):

    def __init__(self, cfg: dict):
        self.config = cfg

    def __getitem__(self, key):
        return self.config[key]

    @abstractmethod
    def create(self):
        pass
