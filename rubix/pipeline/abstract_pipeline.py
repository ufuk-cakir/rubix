from abc import ABC, abstractmethod
from .transformer import compiled_transformer, expression_transformer


class AbstractPipeline(ABC):
    def __init__(self, cfg: dict):
        self.config = cfg
        self._pipeline = []
        self._names = []
        self.transformers = {}
        self.expression = None

    def assemble(self):
        if len(self.transformers) == 0:
            raise RuntimeError("no registered transformers")
        self.build_pipeline()
        self.build_expression()

    @property
    def pipeline(self):
        return dict(zip(self._names, self._pipeline))

    def register_transformer(self, cls: list):
        if cls.__name__ in self.transformers:
            raise ValueError("Error, a class of this name is already present")
        self.transformers[cls.__name__] = cls

    def get_jaxpr(self, *args, static_args: list = []):
        return expression_transformer(*args, static_args=static_args)(self.expression)

    def compile_expression(self, static_args=[], static_kwargs=[]):
        return compiled_transformer(
            static_args=static_args, static_kwargs=static_kwargs
        )(self.expression)

    def compile_element(self, name: str, static_args=[], static_kwargs=[]):
        return compiled_transformer(
            static_args=static_args, static_kwargs=static_kwargs
        )(self.transformers[name])

    def get_jaxpr_for_element(self, name: str, *args, static_args: list = []):
        return expression_transformer(*args, static_args=static_args)(
            self.transformers[name]
        )

    @abstractmethod
    def build_pipeline(self):
        pass

    @abstractmethod
    def build_expression(self):
        pass

    @abstractmethod
    def apply(self):
        pass
