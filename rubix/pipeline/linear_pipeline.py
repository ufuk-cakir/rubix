from . import abstract_pipeline as apl
from .transformer import bound_transformer, compiled_transformer
from jax.tree_util import Partial
from copy import deepcopy


class LinearTransformerPipeline(apl.AbstractPipeline):
    """
    LinearTransformerPipeline An implementation of a data transformation
    pipeline in the form of a simple, 1-D chain of composed functions in which
    each function uses the output of the function before it as arguments.

    Parameters
    ----------
    apl : Abstract base class for all pipeline implementations
    """

    def __init__(self, cfg: dict):
        """
        __init__ Build a new LinearTransformerPipeline instance

        This does only set up all the necessary things to build the pipeline.
        The pipeline itself has to be created after registering transformers
        to use and calling `assemble`.

        Parameters
        ----------
        cfg : dict
            Read yaml configuration file for the pipeline
        """
        super().__init__(cfg)

    def update_pipeline(self, current_name: str):
        """
        update_pipeline add a new pipeline node with name 'current_name' to
            the pipeline, taking into account internal linear dependencies.
            Mostly used internally for adding nodes one by one.

        Parameters
        ----------
        current_name : Name of the node to add

        """
        for key, node in self.config["Transformers"].items():
            if current_name == node["depends_on"]:
                func = bound_transformer(**node["args"])(
                    self.transformers[node["name"]]
                )
                self._pipeline.append(func)
                self._names.append(key)
                self.update_pipeline(key)

    def build_pipeline(self):
        """
        build_pipeline Build up the pipeline from the internally stored
        configuration.
        This only works when all transformers the pipeline is composed of have
        been registered with it.

        Raises
        ------
        ValueError
            When a config node has no `name` element that names a transformer
            function.
        ValueError
            When `depends_on` element of a node is None in more than one node
            of the config.
        """
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
        """
        build_expression Compose the assembled pipeline into a single
        expression that has the same signature as the first element of the
        pipeline.

        """

        def expr(input, pipeline=[]):
            res = input
            for f in pipeline:
                res = f(res)
            return res

        # deepcopy is needed to isolate the expr-function instance from the class,
        # since in principle it's a closure that pulls in the surrounding scope,
        # which includes `self`
        self.expression = Partial(expr, pipeline=deepcopy(self._pipeline))

    def apply(self, static_args=[], static_kwargs=[], *args):
        """
        apply Apply the pipeline to a set of input arguments that match the
            signature of the first method in the pipeline with static
            (keyword) arguments that are not traced. First applies the jax jit
            to the pre-assembled pipeline, then applies the result to the
            arguments.
        Parameters
        ----------
        static_args : list, optional
            List of positional argument indices that should not be traced,
            by default []
        static_kwargs : list, optional
            list of keyword arguments that should not be traced, by default []

        Returns
        -------
        object
            Result of the application of the pipeline to the provided input.

        Raises
        ------
        ValueError
            _description_
        """
        if len(args) == 0:
            raise ValueError("Cannot apply the pipeline to an empty list of arguments")

        if self.compiled_expression is None:
            self.compile_expression(
                static_args=static_args, static_kwargs=static_kwargs
            )

        return self.compiled_expression(self.expression)(args)
