from abc import ABC, abstractmethod

from jax import jit
from jax import make_jaxpr
from jax.tree_util import Partial


def bound_transformer(**kwargs):

    def transformer_wrap(kernel):
        return Partial(kernel, **kwargs)

    return transformer_wrap


def compiled_transformer(
    static_args: list = [],
    static_kwargs: list = [],
    **kwargs,
):

    def transformer_wrap(kernel):

        return jit(
            Partial(kernel, **kwargs),
            static_argnums=static_args,
            static_argnames=static_kwargs,
        )

    return transformer_wrap


def expression_transformer(
    *args,
    static_args: list = [],
):

    def transformer_wrap(kernel):
        if len(args) > 0:
            return make_jaxpr(kernel, static_argnums=static_args)(*args)
        else:
            return make_jaxpr(kernel, static_argnums=static_args)

    return transformer_wrap
