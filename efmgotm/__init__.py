"""efmgotm -- run and analyse GOTM water-column simulations for 37EFM.

The General Ocean Turbulence Model (GOTM, https://gotm.net) solves the
one-dimensional, vertical equations for a water column with a turbulence
closure.  This package wraps three exercises of the course Environmental
Fluid Mechanics (TU/e):

    couette   turbulent Couette flow driven by a surface stress
    lake      mixing of a stratified lake by surface cooling and wind
    estuary   tidal estuarine circulation with a salinity gradient

Typical use::

    import efmgotm
    ds = efmgotm.run('lake', {'surface.fluxes.heat.constant_value': -50,
                              'temperature.NN': 1e-4}, name='cooling')
    ds.temp.plot(x='time')

``run`` writes a ``gotm.yaml`` in a run directory, calls the ``gotm``
executable (installed with conda from conda-forge) and returns the
output as an xarray Dataset with 1-D vertical coordinates.
"""
from .run import run, load, find_gotm, case_config, available_cases, CASES
from . import analysis

__version__ = "0.1.0"
