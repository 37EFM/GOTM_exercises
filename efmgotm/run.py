"""Running GOTM from Python."""
from __future__ import annotations

import copy
import os
import shutil
import subprocess
import tempfile
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
    text = _re.sub(r"(:\s*)\*(\s*(#.*)?)$", r"\1'*'\2", text, flags=_re.M)
    return yaml.load(text, Loader=_Loader)


def _case_text(case):
    if case not in CASES:
        raise ValueError(f"unknown case {case!r}; choose from {CASES}, or give the path of a .yaml file")
    return resources.files("efmgotm").joinpath(f"cases/{case}.yaml").read_text()


def _is_file(case):
    return isinstance(case, Path) or str(case).endswith((".yaml", ".yml"))


def case_config(case):
    """The base configuration of a case as a nested dict.

    ``case`` is a course case (``'couette'``, ``'lake'``, ``'estuary'``, read
    from ``efmgotm/cases/<case>.yaml``) or the path of a configuration file,
    such as one written by :func:`copy_case`.
    """
    if _is_file(case):
        path = Path(case)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (from {Path.cwd()})")
        return load_yaml(path.read_text())
    return load_yaml(_case_text(case))


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


_CONFIG_VERSIONS = {}


def config_version(exe):
    """Configuration version that a GOTM executable reads.

    The GOTM 6.0.x releases on conda-forge, which the course uses and the
    case files are written for, read version 6; GOTM built from the current
    source reads version 7.  ``gotm --version`` reports it on its
    ``Config:`` line.
    """
    exe = str(exe)
    if exe not in _CONFIG_VERSIONS:
        try:
            p = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60)
            out = p.stdout + p.stderr                   # conda builds print it on stderr
        except (OSError, subprocess.SubprocessError):
            out = ""
        m = _re.search(r"Config:\s*(\d+)", out)
        _CONFIG_VERSIONS[exe] = int(m.group(1)) if m else 6
    return _CONFIG_VERSIONS[exe]


# settings introduced after version 6; a version 6 executable rejects them
_NEW_IN_VERSION7 = [
    "grid.fixed_grid", "bottom.calc_bottom_stress", "physical_constants.rotation_period",
    "temperature.type", "salinity.type", "surface.swr.type",
    "surface.longwave_radiation.type", "surface.longwave_radiation.constant_value",
]


_OUTPUT_RENAMED_IN_VERSION7 = {"rho_p": "rho"}
_OUTPUT_ONLY_IN_VERSION7 = {"idpdx", "idpdy", "nucl", "PSTK"}


def to_version6(config):
    """Translate a version 7 configuration into version 6, for GOTM 6.0.x.

    This undoes what GOTM's ``update_setup.py`` does between the two:

    - ``equation_of_state`` becomes ``eq_state``.  The reference density
      ``rho0`` moves to ``physical_constants.rho_0`` and the specific heat
      ``cp`` to ``physical_constants.cp``.  The dimensionless expansion
      coefficients become coefficients in kg/m^3 per unit:
      ``dtr0 = -rho0 alpha`` and ``dsr0 = rho0 beta``, so that
      rho = rho0 [1 - alpha (T - T0) + beta (S - S0)] in both versions.
    - ``mimic_3d.int_pressure.gradients.*`` becomes ``mimic_3d.int_press.*``.
    - The output variable ``rho_p`` is called ``rho``, the name the course
      uses.  ``idpdx`` and ``idpdy`` are not available and are dropped.

    Version 7 also interprets temperature and salinity as TEOS-10
    conservative temperature and absolute salinity, so the two versions
    agree to a few per cent, not exactly.
    """
    c = copy.deepcopy(config)
    c["version"] = 6

    eos = c.pop("equation_of_state", None)
    if eos is not None:
        method = eos.get("method", "full_teos-10")
        lin = dict(eos.get("linear") or {})
        rho0 = float(eos.get("rho0", 1027.0))
        old = {}
        if method == "linear_custom":
            old["form"] = "linear_custom"
            linear = {k: lin[k] for k in ("T0", "S0", "p0") if k in lin}
            linear["dtr0"] = -rho0 * float(lin.get("alpha", 2.0e-4))   # version 7 defaults
            linear["dsr0"] = rho0 * float(lin.get("beta", 7.5e-4))
            old["linear"] = linear
        elif method == "linear_teos-10":
            old["form"] = "linear"
            old["linear"] = {k: lin[k] for k in ("T0", "S0", "p0") if k in lin}
        else:
            old["form"] = "full"
        c["eq_state"] = old
        constants = c.setdefault("physical_constants", {})
        constants["rho_0"] = rho0
        if "cp" in lin:
            constants["cp"] = lin["cp"]

    mimic = c.get("mimic_3d") or {}
    ip = mimic.pop("int_pressure", None)
    if ip is not None:
        kind = ip.pop("type", "gradients")
        if kind not in ("gradients", "none", None):
            raise ValueError(f"mimic_3d.int_pressure.type = {kind} needs GOTM with configuration version 7")
        ip.pop("plume", None)
        old = dict(ip.pop("gradients", None) or {})
        old.update(ip)                              # s_adv, t_adv
        mimic["int_press"] = old

    # output variables: rho_p is called rho in version 6; the others do not exist there
    for category in (c.get("output") or {}).values():
        variables = category.get("variables") if isinstance(category, dict) else None
        if variables:
            kept = []
            for v in variables:
                src = v.get("source")
                if src in _OUTPUT_ONLY_IN_VERSION7:
                    continue
                kept.append({**v, "source": _OUTPUT_RENAMED_IN_VERSION7.get(src, src)})
            category["variables"] = kept

    for key in _NEW_IN_VERSION7:
        *parents, last = key.split(".")
        node = c
        for k in parents:
            node = node.get(k) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(last, None)
    return c


def to_version7(config):
    """Translate a version 6 configuration into version 7, for GOTM built from source.

    The inverse of :func:`to_version6`: ``eq_state`` becomes
    ``equation_of_state`` with the dimensionless coefficients
    ``alpha = -dtr0 / rho_0`` and ``beta = dsr0 / rho_0``, the reference
    density and specific heat move into it from ``physical_constants``,
    ``mimic_3d.int_press.*`` becomes ``mimic_3d.int_pressure.gradients.*``,
    and the output variable ``rho`` is called ``rho_p``.
    """
    c = copy.deepcopy(config)
    c["version"] = 7

    constants = c.get("physical_constants") or {}
    rho0 = float(constants.pop("rho_0", 1027.0))
    cp = constants.pop("cp", None)
    if "physical_constants" in c and not constants:
        del c["physical_constants"]
    eos = c.pop("eq_state", None) or {}
    form = eos.get("form", "full")
    lin = dict(eos.get("linear") or {})
    new = {"rho0": rho0}
    linear = {k: lin[k] for k in ("T0", "S0", "p0") if k in lin}
    if form == "linear_custom":
        new["method"] = "linear_custom"
        linear["alpha"] = -float(lin.get("dtr0", -0.17)) / rho0     # version 6 defaults
        linear["beta"] = float(lin.get("dsr0", 0.78)) / rho0
    elif form == "linear":
        new["method"] = "linear_teos-10"
    else:
        new["method"] = "full_teos-10"
    if cp is not None:
        linear["cp"] = cp
    if linear:
        new["linear"] = linear
    c["equation_of_state"] = new

    mimic = c.get("mimic_3d") or {}
    ip = mimic.pop("int_press", None)
    if ip is not None:
        gradients = {k: ip.pop(k) for k in ("dtdx", "dtdy", "dsdx", "dsdy") if k in ip}
        mimic["int_pressure"] = {"type": "gradients", "gradients": gradients, **ip}

    renamed = {old: new for new, old in _OUTPUT_RENAMED_IN_VERSION7.items()}
    for category in (c.get("output") or {}).values():
        variables = category.get("variables") if isinstance(category, dict) else None
        if variables:
            category["variables"] = [{**v, "source": renamed.get(v.get("source"), v.get("source"))}
                                     for v in variables]
    return c


def for_executable(config, exe):
    """The configuration in the version that the executable ``exe`` reads."""
    wanted, have = config_version(exe), config.get("version", 6)
    if have == 7 and wanted == 6:
        return to_version6(config)
    if have == 6 and wanted == 7:
        return to_version7(config)
    return config


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


def _write_full(exe, rundir):
    """Let GOTM write ``gotm_full.yaml``: every setting, with its meaning and default."""
    try:
        subprocess.run([exe, "--write_yaml", "gotm_full.yaml", "--detail", "full"], cwd=rundir,
                       capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        pass
    return Path(rundir) / "gotm_full.yaml"


def _case_choices(case):
    """The settings a course case chooses, as (dotted key, value, comment) from its file."""
    rows, stack = [], []
    for line in _case_text(case).splitlines():
        m = _re.match(r"^( *)([A-Za-z0-9_]+):\s*([^#]*?)\s*(#\s*(.*))?$", line)
        if not m:
            continue
        depth = len(m.group(1)) // 3
        stack[depth:] = [m.group(2)]
        if stack[0] in ("version", "title", "output") or not m.group(3):
            continue
        rows.append((".".join(stack), m.group(3), m.group(5) or ""))
    return rows


def copy_case(case, path=None, overwrite=False, gotm=None):
    """Write a course case with every GOTM setting to a file you can edit and run.

    GOTM writes the complete configuration: every setting it has, with the
    values of the course case, and on each line a comment giving its
    meaning, units, allowed values and default.  The header of the file
    lists the settings that the course case chooses; everything else is
    GOTM's default.  Edit the file (in JupyterLab, double-click it in the
    file browser) and run it with ``efmgotm.run('my_lake.yaml')``.

    Parameters
    ----------
    case : 'couette', 'lake' or 'estuary'
    path : str or Path
        Where to write it; default ``my_<case>.yaml`` in the current directory.
    overwrite : bool
        Replace an existing file (your edits in it are lost).

    Returns
    -------
    Path of the file.
    """
    path = Path(path or f"my_{case}.yaml")
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists already; pick another name or pass overwrite=True")
    exe = gotm or find_gotm()
    config = for_executable(case_config(case), exe)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "gotm.yaml").write_text(dump_yaml(config) + "\n")
        full = _write_full(exe, tmp)
        if not full.exists():
            raise RuntimeError(f"{exe} --write_yaml failed")
        body = [l for l in full.read_text().splitlines() if not l.startswith("#")]
    head = [f"# The course case '{case}' with every setting of GOTM (configuration version {config['version']}).",
            "# The comment on each line gives the meaning of the setting, its units, the allowed",
            "# values and GOTM's default.  Edit the values and run the file with",
            f"#     efmgotm.run('{path.name}')",
            "#"]
    if config["version"] == case_config(case)["version"]:
        rows = _case_choices(case)
        width = max(len(f"{k}: {v}") for k, v, _ in rows)
        head += ["# The course case sets the following; everything else is GOTM's default:", "#"]
        head += [(f"#   {k}: {v}".ljust(width + 6) + f"  {c}").rstrip() for k, v, c in rows]
        head += ["#"]
    path.write_text("\n".join(head + body) + "\n")
    return path


def run(case, settings=None, name=None, workdir="runs", rerun=True, quiet=True,
        gotm=None, output_variables=None):
    """Run a case with modified settings and return its output.

    Parameters
    ----------
    case : 'couette', 'lake', 'estuary', or the path of a .yaml file
        A course case, or a configuration file of your own, for instance
        one written by :func:`copy_case` and edited.
    settings : dict
        Changes to the base configuration, with dotted keys, for example
        ``{'surface.fluxes.tx.constant_value': 0.01, 'time.stop': '2005-01-03 00:00:00'}``.
        Every setting is listed, with its meaning and default, in the file
        ``gotm_full.yaml`` that each run leaves in its run directory, and in
        the file that :func:`copy_case` writes.
    name : str
        Name of the run; the run directory is ``<workdir>/<case>_<name>``
        (for a file, its name without ``.yaml`` takes the place of the case).
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
    label = Path(case).stem if _is_file(case) else case
    rundir = Path(workdir) / f"{label}_{name}"
    rundir.mkdir(parents=True, exist_ok=True)
    ncfile = rundir / f"{_output_name(config)}.nc"
    if not rerun and ncfile.exists():
        return load(ncfile)
    exe = gotm or find_gotm()
    config = for_executable(config, exe)       # GOTM built from source reads version 7
    (rundir / "gotm.yaml").write_text(dump_yaml(config) + "\n")
    _write_full(exe, rundir)                   # every setting, with its meaning and default
    with open(rundir / "gotm.log", "w") as log:
        proc = subprocess.run([exe], cwd=rundir, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not ncfile.exists():
        tail = (rundir / "gotm.log").read_text().splitlines()[-15:]
        if proc.returncode in (0xC0000135, -0x3FFFFECB):     # Windows: STATUS_DLL_NOT_FOUND
            tail.append("Windows could not load a library that gotm.exe needs, usually the netCDF one.\n"
                        "In the Anaconda Prompt, run\n"
                        "    conda activate efmgotm\n"
                        '    conda install -c conda-forge "netcdf-fortran<4.6.3"\n'
                        "and restart the notebook kernel.")
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
    if "rho_p" in ds and "rho" not in ds:          # GOTM 7 calls it rho_p
        ds = ds.rename({"rho_p": "rho"})
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
