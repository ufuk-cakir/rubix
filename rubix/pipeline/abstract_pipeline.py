from abc import ABC, abstractmethod


class AbstractPipeline(ABC):
    def __init__(self, cfg: dict):
        self.config = cfg
        self._pipeline = []
        self._names = []
        self.transformers = {}

    def _build_pipeline(self):

        # find the starting point and look into corrections
        start_func = None
        start_name = None
        for key, node in self.config["Transformers"].items():

            if "name" not in node:
                raise ValueError(
                    "Error, each node of a pipeline must have a config node"
                )

            # if node["active"] is False:
            #     continue

            if node["depends_on"] is None and start_name is None:
                start_func = self.transformers[node["name"]](**node["args"]).create()
                start_name = key
            elif node["depends_on"] is None and start_name is not None:
                raise ValueError("There can only be one starting point.")
            else:
                continue

        self._pipeline = [
            start_func,
        ]

        self._names = [
            start_name,
        ]

        self.update_pipeline(start_name)

    @property
    def pipeline(self):
        return dict(zip(self._names, self._pipeline))

    def register_transformer(self, cls):
        if cls.__name__ in self.transformers:
            raise ValueError("Error, a class of this name is already present")
        self.transformers[cls.__name__] = cls

    def assemble(self):
        if len(self.transformers) == 0:
            raise RuntimeError("no registered transformers")
        self._build_pipeline()
        self.expression = self.build_expression()

    @abstractmethod
    def apply(self, input):
        pass

    @abstractmethod
    def build_expression(self):
        pass

    @abstractmethod
    def update_pipeline(self, current_name: str):
        pass
