from abc import ABC, abstractmethod

from jax import jit
from jax import make_jaxpr
from jax.tree_util import Partial


def bound_transformer(**kwargs):
    """
    bound_transformer Create a jax.Partial function from an input
    function and given keyword arguments.
    """
    def transformer_wrap(kernel):
        return Partial(kernel, **kwargs)

    return transformer_wrap


def compiled_transformer(
    static_args: list = [],
    static_kwargs: list = [],
    **kwargs,
):
    """
    compiled_transformer Create a precompiled function with jax with given 
    untraced arguments and keyword arguments from an input function

    Parameters
    ----------
    static_args : list, optional
        Indices of static, i.e., untraced arguments, by default []
    static_kwargs : list, optional
        Names of static, i.e., untraced, keyword arguments, by default []
    """
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
    """
    expression_transformer Create a jax intermediate expression with given 
    untraced arguments from a function. 

    Parameters
    ----------
    static_args : list, optional
        Indices of static, i.e., untraced arguments to the function, by default []
    """
    def transformer_wrap(kernel):
        if len(args) > 0:
            return make_jaxpr(kernel, static_argnums=static_args)(*args)
        else:
            return make_jaxpr(kernel, static_argnums=static_args)

    return transformer_wrap
