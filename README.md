# GOTM_exercises — water-column simulations for 37EFM

Course material for **37EFM Environmental Fluid Mechanics** (Eindhoven
University of Technology). Three numerical exercises with the General
Ocean Turbulence Model (GOTM, https://gotm.net), a one-dimensional
model of the vertical structure of a water column with a turbulence
closure ($k$-$\varepsilon$ here). The model is run from Python and its
output is analysed in Jupyter notebooks.

| Notebook | Exercise |
|---|---|
| `notebooks/01_couette.ipynb` | Turbulent Couette flow: spin-up, law of the wall, turbulent viscosity |
| `notebooks/02_lake_convection.ipynb` | Convection in a lake cooled at the surface, with and without stratification |
| `notebooks/03_lake_wind.ipynb` | Wind mixing of a stratified lake, with and without a heat flux |
| `notebooks/04_estuary.ipynb` | Tidal estuarine circulation: spectrum, net outflow, tidally averaged profile |

The questions are on the exercise sheets handed out in the course. The
notebooks make the plots those questions need and leave the
interpretation to you.

## Installation

GOTM itself is a conda-forge package, prebuilt for Windows, Linux and
macOS, so no compiler is needed. With Miniconda or Anaconda installed:

```bash
git clone https://github.com/37EFM/GOTM_exercises.git
cd GOTM_exercises
conda env create -f environment.yml
conda activate efmgotm
jupyter lab notebooks
```

On Windows, run these in the *Anaconda Prompt* (or *Miniconda Prompt*).
Creating the environment takes a few minutes the first time. To check
that the model is found, run `gotm --version` in the activated
environment.

## How a run works

```python
import efmgotm
ds = efmgotm.run('lake', {'surface.fluxes.heat.constant_value': -5,
                          'temperature.NN': 1e-4})
ds.temp.plot(x='hours', y='z')
```

`run` takes a case name (`couette`, `lake`, `estuary`) and a dictionary
of changes to its base configuration. It writes a complete `gotm.yaml`
in a run directory (`runs/<case>_<name>/`), calls the `gotm` executable
there, and opens the netCDF output as an `xarray.Dataset`. The log of
the run is in `gotm.log` next to the output.

The base configurations are in `efmgotm/cases/*.yaml`, with comments.
A key such as `surface.fluxes.heat.constant_value` is the path to a
setting in that file. GOTM has many more settings than the ones listed
there; `gotm --write_yaml full.yaml --detail full` in a terminal
writes a configuration with every setting and its description, and
any of them can be passed to `run` in the same way.

The Dataset has the vertical coordinates `z` (layer centres, negative
upward, 0 at the surface) and `zi` (layer interfaces, where turbulence
quantities such as `tke`, `eps`, `num` and `NN` live), and the time
coordinates `time`, `seconds`, `hours` and `days`. Everything that
xarray offers works on it: `ds.u.sel(z=-5, method='nearest')`,
`ds.u.isel(time=-1)`, `ds.u.plot(x='hours')`.

`efmgotm.analysis` holds the theoretical expressions of the exercise
sheets (law of the wall, convective and wind-mixed layer depths, the
estuarine circulation profile) and a few diagnostics (mixed-layer
depth, depth average, tidal average, spectrum).

## The cases

**Couette** (`couette.yaml`): depth 10 m, surface stress 1.027 Pa, rough
bottom with $h_{0b} = 0.1$ m, 100 layers, 10 s time step, 1 day, no
rotation, no stratification. This is the `couette` scenario of the GOTM
distribution.

**Lake** (`lake.yaml`): depth 20 m, 100 layers, 60 s time step, 7 days,
linear temperature stratification set by `temperature.NN`, constant
salinity, prescribed surface heat flux and wind stress, no rotation.
The equation of state is linear with $\alpha = 2.6 \times 10^{-4}$
K$^{-1}$, $\rho_0 = 1000$ kg/m$^3$ and $c_p = 4150$ J/(kg K), the
constants of the exercise sheet.

**Estuary** (`estuary.yaml`): depth 15 m, 50 layers, 36 s time step, 29
days. An M2 tide with a depth-averaged velocity amplitude of 0.5 m/s on
top of a river outflow of $-0.02$ m/s, a constant along-channel
salinity gradient of $-2 \times 10^{-4}$ psu/m with salinity advection
and relaxation, and a small cross-channel wind stress. Linear equation
of state with $\beta = 7.7 \times 10^{-4}$ psu$^{-1}$ and
$\rho_0 = 1027$ kg/m$^3$. This is the `estuary` scenario of the GOTM
distribution without its surface-elevation signal and S2 tide.

The cases are the GOTM 5 namelist setups used in earlier years,
translated to the GOTM 6 configuration format.

## Repository layout

```
efmgotm/run.py        run, load, case_config, find_gotm
efmgotm/analysis.py   theory curves and diagnostics
efmgotm/cases/        couette.yaml, lake.yaml, estuary.yaml
notebooks/            one notebook per exercise (run directories appear in notebooks/runs/)
tests/                checks that need the gotm executable are skipped when it is absent
```

## References

- Burchard, H., *Applied Turbulence Modelling in Marine Waters*,
  Springer, 2002.
- Umlauf, L., Burchard, H. and Bolding, K., GOTM: Source code and test
  case documentation, https://gotm.net.
- Umlauf, L. and Burchard, H., A generic length-scale equation for
  geophysical turbulence models, *J. Mar. Res.* 61, 235–265, 2003.

## Licence

MIT, see `LICENSE`. GOTM is distributed under the GNU General Public
License by its authors.
