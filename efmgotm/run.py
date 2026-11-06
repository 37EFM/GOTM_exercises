"""Running GOTM from Python."""
from __future__ import annotations

import copy
import os
import shutil
import subprocess
from importlib import resources
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

CASES = ("couette", "lake", "estuary")


def available_cases():
    return list(CASES)


class _Loader(yaml.SafeLoader):
    """SafeLoader that reads only true/false as booleans.

    GOTM uses the words ``off`` and ``on`` as option values; YAML 1.1 would
    turn them into booleans.
    """


_Loader.yaml_implicit_resolvers = {
    k: [(tag, regexp) for tag, regexp in v if tag != "tag:yaml.org,2002:bool"]
    for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()}
import re as _re
_Loader.add_implicit_resolver("tag:yaml.org,2002:bool", _re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
                              list("tTfF"))


def load_yaml(text):
    """Parse GOTM-style YAML text into a nested dict.

    A bare ``*`` (GOTM's wildcard for "all variables") is not valid YAML;
    it is quoted before parsing and written back bare by :func:`dump_yaml`.
    """
    text = _re.sub(r"(:\s*)\*\s*$", r"\1'*'", text, flags=_re.M)
    return yaml.load(text, Loader=_Loader)


def case_config(case):
    """The base configuration of a case as a nested dict (from ``cases/<case>.yaml``)."""
    if case not in CASES:
        raise ValueError(f"unknown case {case!r}; choose from {CASES}")
    text = resources.files("efmgotm").joinpath(f"cases/{case}.yaml").read_text()
    return load_yaml(text)


def find_gotm():
    """Path of the GOTM executable.

    Looks at the environment variable ``GOTM_EXE``, then on the PATH (which
    includes the active conda environment).  Raises a helpful error if
    nothing is found.
    """
    exe = os.environ.get("GOTM_EXE")
    if exe and Path(exe).exists():
        return exe
    exe = shutil.which("gotm")
    if exe:
        return exe
    raise FileNotFoundError(
        "GOTM executable not found. Install it in the active conda environment with\n"
        "    conda install -c conda-forge gotm\n"
        "or set the environment variable GOTM_EXE to the path of the executable.")


def set_nested(config, key, value):
    """Set ``config['a']['b']['c'] = value`` from the dotted key ``'a.b.c'``."""
    parts = key.split(".")
    d = config
    for p in parts[:-1]:
        if p not in d or d[p] is None:
            d[p] = {}
        d = d[p]
    d[parts[-1]] = value


def get_nested(config, key, default=None):
    d = config
    for p in key.split("."):
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def _output_name(config):
    return next(iter(config["output"]))


def _scalar(v):
    """Format a scalar the way GOTM's YAML reader expects (no quotes)."""
    import datetime
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (datetime.datetime, datetime.date)):
        return v.strftime("%Y-%m-%d %H:%M:%S")
    if v is None:
        return ""
    if isinstance(v, float):
        return repr(v)
    return str(v)


def dump_yaml(config, indent=0):
    """Write a nested dict as YAML without quotes (GOTM's reader rejects them)."""
    lines = []
    pad = " " * indent
    for key, value in config.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines.append(dump_yaml(value, indent + 3))
        elif isinstance(value, list):
            lines.append(f"{pad}{key}:")
            for item in value:
                if isinstance(item, dict):
                    first = True
                    for k2, v2 in item.items():
                        prefix = f"{pad}- " if first else f"{pad}  "
                        lines.append(f"{prefix}{k2}: {_scalar(v2)}")
                        first = False
                else:
                    lines.append(f"{pad}- {_scalar(item)}")
        else:
            lines.append(f"{pad}{key}: {_scalar(value)}")
    return "\n".join(l for l in lines if l != "")


def run(case, settings=None, name=None, workdir="runs", rerun=True, quiet=True,
        gotm=None, output_variables=None):
    """Run a case with modified settings and return its output.

    Parameters
    ----------
    case : 'couette', 'lake' or 'estuary'
    settings : dict
        Changes to the base configuration, with dotted keys, for example
        ``{'surface.fluxes.tx.constant_value': 0.01, 'time.stop': '2005-01-03 00:00:00'}``.
        Run ``efmgotm.case_config(case)`` to see the base configuration, or
        read ``efmgotm/cases/<case>.yaml``.
    name : str
        Name of the run; the run directory is ``<workdir>/<case>_<name>``.
        Default: derived from the settings.
    workdir : str or Path
        Where run directories are created (relative to the current directory).
    rerun : bool
        If False and the output file exists, do not run again; just load it.
    quiet : bool
        Hide GOTM's screen output (it is saved to ``gotm.log`` in any case).
    gotm : str
        Path of the executable; default :func:`find_gotm`.
    output_variables : list of str
        Replace the output variable list of the case (``['*']`` for all).

    Returns
    -------
    xarray.Dataset, see :func:`load`.
    """
    settings = dict(settings or {})
    config = copy.deepcopy(case_config(case))
    for key, value in settings.items():
        set_nested(config, key, value)
    if output_variables is not None:
        config["output"][_output_name(config)]["variables"] = [{"source": v} for v in output_variables]
    if name is None:
        name = "_".join(f"{k.split('.')[-2] if k.endswith('constant_value') else k.split('.')[-1]}={v}"
                        for k, v in settings.items()) or "default"
        name = name.replace(" ", "").replace("/", "")
    rundir = Path(workdir) / f"{case}_{name}"
    rundir.mkdir(parents=True, exist_ok=True)
    ncfile = rundir / f"{_output_name(config)}.nc"
    if not rerun and ncfile.exists():
        return load(ncfile)
    (rundir / "gotm.yaml").write_text(dump_yaml(config) + "\n")
    exe = gotm or find_gotm()
    with open(rundir / "gotm.log", "w") as log:
        proc = subprocess.run([exe], cwd=rundir, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not ncfile.exists():
        tail = (rundir / "gotm.log").read_text().splitlines()[-15:]
        raise RuntimeError(f"GOTM failed in {rundir} (exit code {proc.returncode}):\n" + "\n".join(tail))
    if not quiet:
        print((rundir / "gotm.log").read_text())
    ds = load(ncfile)
    ds.attrs["run_directory"] = str(rundir)
    ds.attrs["settings"] = str(settings)
    return ds


def load(ncfile):
    """Open a GOTM output file as a tidy xarray Dataset.

    GOTM writes every field with singleton ``lat`` and ``lon`` dimensions and
    with the vertical coordinates ``z`` and ``zi`` depending on time (they
    move with the surface elevation).  Here the singleton dimensions are
    dropped and, when the grid does not move, ``z`` and ``zi`` are made
    1-D so that ``ds.u.plot(x='time')`` just works.  The extra coordinate
    ``hours`` and ``days`` give the elapsed time.
    """
    ds = xr.open_dataset(ncfile).squeeze(drop=True)
    for c in ("z", "zi"):
        if c in ds.coords and "time" in ds[c].dims:
            first = ds[c].isel(time=0)
            if np.allclose(ds[c].values, first.values, atol=1e-6):
                ds = ds.assign_coords({c: first.drop_vars("time", errors="ignore")})
    t = ds["time"]
    seconds = (t - t[0]) / np.timedelta64(1, "s")
    ds = ds.assign_coords(seconds=("time", seconds.values),
                          hours=("time", seconds.values / 3600.0),
                          days=("time", seconds.values / 86400.0))
    return ds
