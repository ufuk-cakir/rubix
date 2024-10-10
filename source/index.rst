.. rubix documentation master file, created by
   sphinx-quickstart on Thu Oct 10 13:33:35 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

RUBIX documentation
===================

RUBIX is a tested and modular Open Source tool developed in JAX, designed to forward model IFU cubes of galaxies from cosmological hydrodynamical simulations.
The code automatically parallelizes computations across multiple GPUs, demonstrating performance
improvements over state-of-the-art codes. For further details see the publications or the documentation of the individual functions.

Currently the following functionalities are provided:
- Generate mock IFU flux cubes for stars from IllustrisTNG50
- Use different stellar population synthesis models
- Use MUSE as telescope instrument

Currently the code is under development and is not yet all functionality is available.
We are working on adding more features and improving the code, espectially we work on the following features:
- Adding support for more simulations
- Adding support for more telescopes
- Adding gas emission lines and gas continuum
- Adding dust attenuation
- Adding support for gradient calculation

If you are interested in contributing to the code or have ideas for further features, please contact us.
If you use the code in your research, please cite the following paper: ???

Publications about RUBIX:
- [1] Fast GPU-Powered and Auto-Differentiable Forward Modeling of IFU Data Cubes - U. Çakır, A. Schaible and T. Buck (NeurIPS 2024)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   rubix
   rubix.pipeline
   rubix.core
   rubix.cosmology
   rubix.telescope
   rubix.telescope.noise
   rubix.galaxy
   rubix.galaxy.input_handler
   rubix.spectra
   rubix.spectra.ssp
