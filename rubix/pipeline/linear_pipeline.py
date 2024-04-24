from . import abstract_pipeline as apl
import jax

# README: extract the expression building and make it more 'functional', 
# such that the pipeline is an argument and not a captured reference in a closure


class LinearTransformerPipeline(apl.AbstractPipeline):
    def __init__(self, cfg: dict):
        super().__init__(cfg)

    def update_pipeline(self, current_name: str):
        for key, node in self.config["Transformers"].items():
            if current_name == node["depends_on"]:
                self._pipeline.append(
                    self.transformers[node["name"]](**node["args"]).create()
                )
                self._names.append(key)
                self.update_pipeline(key)

    def build_expression(self):
        # this probably should be a separate function
        def expr(input):
            res = input
            for f in self._pipeline:
                res = f(res)
            return res

        return expr

    def apply(self, input, with_jit: bool = True):
        if with_jit:
            expr = jax.jit(self.expression)
            return expr(input)
        else:
            return self.expression(input)
