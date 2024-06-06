""" This module contains the supported templates for the SSP grid. """

from .factory import get_ssp_template

BruzualCharlot2003 = get_ssp_template("BruzualCharlot2003")
MaStar_CB19_SLOG_1_5 = get_ssp_template("Mastar_CB19_SLOG_1_5")


__all__ = ["BruzualCharlot2003", "Mastar_CB19_SLOG_1_5"]