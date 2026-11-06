"""Run with ``pytest``.  Tests that execute GOTM are skipped if it is not installed."""
import numpy as np
import pytest

import efmgotm
import importlib
from efmgotm import analysis
runmod = importlib.import_module("efmgotm.run")


def test_case_configs_parse():
    for case in efmgotm.available_cases():
        cfg = efmgotm.case_config(case)
        assert cfg["version"] == 7
        assert cfg["surface"]["fluxes"]["method"] == "off"     # not a boolean


def test_dump_roundtrip_keeps_gotm_words():
    cfg = efmgotm.case_config("couette")
    text = runmod.dump_yaml(cfg)
    assert "method: off" in text and "'" not in text and '"' not in text
    assert "source: *" in text
    assert runmod.load_yaml(text) == cfg


def test_set_nested():
    cfg = {"a": {"b": 1}}
    runmod.set_nested(cfg, "a.c.d", 2)
    assert cfg == {"a": {"b": 1, "c": {"d": 2}}}


def test_estuarine_profile_boundary_conditions():
    z = np.linspace(-15, 0, 301)
    u = analysis.estuarine_profile(z, 15.0, q=-0.3, dbdx=-1.5e-6, A=3e-3)
    assert abs(u[0]) < 1e-12                       # no slip at the bottom
    assert abs(np.trapezoid(u, z) - (-0.3)) < 1e-3  # integrates to q


def test_law_of_the_wall_zero_at_z0():
    assert analysis.law_of_the_wall(0.0, 0.03, 0.003) == 0.0


@pytest.fixture(scope="module")
def gotm_available():
    try:
        efmgotm.find_gotm()
    except FileNotFoundError:
        pytest.skip("gotm executable not installed")


def test_couette_log_layer(gotm_available, tmp_path):
    ds = efmgotm.run("couette", {"time.stop": "2005-01-01 06:00:00"}, name="test", workdir=tmp_path)
    assert ds.u.dims == ("time", "z")
    u = ds.u.isel(time=-1).values
    assert np.all(np.diff(u) > 0)                  # monotonic profile
    u_star = float(ds.u_taub.isel(time=-1))
    assert 0.02 < u_star < 0.05
    z_above = ds.z.values + 10.0
    sel = (z_above > 0.2) & (z_above < 2.0)
    slope = np.polyfit(np.log(z_above[sel]), u[sel], 1)[0]
    assert abs(u_star / slope - 0.4) < 0.1         # von Karman constant recovered


def test_lake_cooling_mixes(gotm_available, tmp_path):
    ds = efmgotm.run("lake", {"surface.fluxes.heat.constant_value": -50, "temperature.NN": 1e-4,
                              "time.stop": "2005-01-02 00:00:00"}, name="test", workdir=tmp_path)
    T0 = ds.temp.isel(time=0).values
    assert abs((T0[-1] - T0[0]) / 20.0 - 1e-4 / (2.6e-4 * 9.81)) < 2e-3   # dT/dz = NN/(alpha g)
    mld = analysis.mixed_layer_depth(ds, "temp", 0.01)
    assert float(mld[-1]) > float(mld[5]) > 0
