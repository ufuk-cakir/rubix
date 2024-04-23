from abc import ABC, abstractmethod


class TransformerFactoryBase(ABC):

    @abstractmethod
    def create(self):
        pass
