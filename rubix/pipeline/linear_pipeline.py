from . import abstract_pipeline as apl
from .transformer import bound_transformer, compiled_transformer
from jax.tree_util import Partial


class LinearTransformerPipeline(apl.AbstractPipeline):
    def __init__(self, cfg: dict):
        super().__init__(cfg)

    def update_pipeline(self, current_name: str):
        for key, node in self.config["Transformers"].items():
            if current_name == node["depends_on"]:
                func = bound_transformer(**node["args"])(
                    self.transformers[node["name"]]
                )
                self._pipeline.append(func)
                self._names.append(key)
                self.update_pipeline(key)

    def build_pipeline(self):

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

                start_func = bound_transformer(**node["args"])(
                    self.transformers[node["name"]]
                )

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

        # use update pipeline to recursively add nodes. Here, this is done only linearly
        self.update_pipeline(start_name)

    def build_expression(self):

        def expr(input, pipeline=[]):
            res = input
            for f in pipeline:
                res = f(res)
            return res

        # FIXME: does this actually create a pure function or is self._pipeline 
        # still a reference? 
        self.expression = Partial(expr, pipeline=self._pipeline)

    def apply(self, static_args=[], static_kwargs=[], *args):
        return compiled_transformer(
            static_args=static_args, static_kwargs=static_kwargs
        )(self.expression)(args)
