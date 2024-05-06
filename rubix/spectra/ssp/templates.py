""" This module contains the supported templates for the SSP grid. """

from .factory import get_ssp_template

BruzualCharlot2003 = get_ssp_template("BruzualCharlot2003")


__all__ = ["BruzualCharlot2003"]

