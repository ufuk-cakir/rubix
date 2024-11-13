import os

# TODO is this okay? Or should we use env variables?

RUBIX_PATH = os.path.dirname(os.path.abspath(__file__))

TEMPLATE_PATH = os.path.join(RUBIX_PATH, "spectra", "ssp", "templates")

FILTERS_PATH = os.path.join(RUBIX_PATH, "telescope", "filters", "data")
