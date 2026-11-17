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
        assert cfg["version"] == 6                       # GOTM 6.0.x, as on conda-forge
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


def test_version_translations():
    """GOTM built from source reads configuration version 7; the cases are version 6."""
    c6 = efmgotm.case_config("estuary")
    c7 = runmod.to_version7(c6)
    assert c7["version"] == 7
    assert "eq_state" not in c7 and c7["equation_of_state"]["method"] == "linear_custom"
    assert c7["equation_of_state"]["rho0"] == 1027.0 and "physical_constants" not in c7
    assert np.isclose(c7["equation_of_state"]["linear"]["beta"], 7.7e-4)
    assert np.isclose(c7["equation_of_state"]["linear"]["alpha"], 1.7e-4)
    ip = c7["mimic_3d"]["int_pressure"]
    assert ip["type"] == "gradients" and ip["gradients"]["dsdx"]["constant_value"] == -2.0e-4
    assert ip["s_adv"] is True
    sources = [v["source"] for v in c7["output"]["estuary"]["variables"]]
    assert "rho_p" in sources and "rho" not in sources
    # and back
    back = runmod.to_version6(c7)
    assert back["eq_state"]["linear"]["dsr0"] == pytest.approx(0.79079)
    assert back["mimic_3d"]["int_press"]["dsdx"] == c6["mimic_3d"]["int_press"]["dsdx"]
    assert back["physical_constants"] == c6["physical_constants"]
    # the original is untouched
    assert "eq_state" in efmgotm.case_config("estuary")


def test_case_choices_listed():
    rows = dict((k, v) for k, v, _ in runmod._case_choices("lake"))
    assert rows["surface.fluxes.heat.constant_value"] == "-5.0"
    assert rows["temperature.two_layer.t_s"] == "20.0"
    assert not any(k.startswith("output") for k in rows)


def test_copy_case_and_run_file(gotm_available, tmp_path):
    path = efmgotm.copy_case("lake", tmp_path / "my_lake.yaml")
    text = path.read_text()
    assert "default=" in text
    if runmod.config_version(efmgotm.find_gotm()) == 6:          # the version the cases are written in
        assert "#   surface.fluxes.heat.constant_value: -5.0" in text
    with pytest.raises(FileExistsError):
        efmgotm.copy_case("lake", path)
    cfg = efmgotm.case_config(path)
    assert cfg["surface"]["fluxes"]["heat"]["constant_value"] == -5.0
    ds = efmgotm.run(path, {"time.stop": "2005-01-01 12:00:00"}, name="test", workdir=tmp_path)
    rundir = tmp_path / "my_lake_test"
    assert (rundir / "gotm_full.yaml").exists() and "rho" in ds
