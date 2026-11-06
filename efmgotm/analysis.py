"""Small helpers for the exercises: theory curves, averages and spectra.

Everything works on the xarray Datasets returned by :func:`efmgotm.run`.
Depth coordinates are negative upward (``z = 0`` at the surface,
``z = -H`` at the bottom), as in GOTM.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

g = 9.81


# ----------------------------------------------------------------------
# boundary layers
# ----------------------------------------------------------------------
def law_of_the_wall(z_above_bottom, u_star, z0, kappa=0.4):
    """Logarithmic velocity profile u = (u_*/kappa) ln((z + z0)/z0)."""
    z_above_bottom = np.asarray(z_above_bottom, float)
    return u_star / kappa * np.log((z_above_bottom + z0) / z0)


def eddy_viscosity_log_layer(z_above_boundary, u_star, z0, kappa=0.4):
    """Eddy viscosity of the log layer, nu_t = kappa u_* (z + z0)."""
    return kappa * u_star * (np.asarray(z_above_boundary, float) + z0)


def friction_velocity(stress, rho0=1027.0):
    """u_* = sqrt(|tau| / rho0)."""
    return np.sqrt(np.abs(stress) / rho0)


# ----------------------------------------------------------------------
# lake: convection and wind mixing
# ----------------------------------------------------------------------
def convective_velocity(Q, H, alpha=2.6e-4, rho0=1000.0, cp=4150.0):
    """Deardorff convective velocity w_* = (alpha g |Q| H / (rho0 cp))^(1/3)."""
    return (alpha * g * abs(Q) * H / (rho0 * cp)) ** (1.0 / 3.0)


def convective_depth_unstratified(t, Q, H, alpha=2.6e-4, rho0=1000.0, cp=4150.0):
    """Growth of the convective layer without stratification, h = w_* t."""
    return convective_velocity(Q, H, alpha, rho0, cp) * np.asarray(t, float)


def convective_depth_stratified(t, Q, N2, alpha=2.6e-4, rho0=1000.0, cp=4150.0, factor=6.0):
    """Convective layer depth in a linearly stratified lake,

        h = sqrt(factor * alpha g |Q| t / (rho0 cp N^2)).

    The exercise sheet gives ``factor = 6``.  A heat budget in which the
    surface cooling erodes the linear stratification without overshoot
    (the mixed layer takes the temperature of its base) gives ``factor = 2``;
    try both.
    """
    return np.sqrt(factor * alpha * g * abs(Q) * np.asarray(t, float) / (rho0 * cp * N2))


def wind_mixed_depth(t, tau, N2, rho0=1000.0):
    """Scaling of the wind-mixed layer depth, h ~ u_* sqrt(t / N)."""
    u_star = friction_velocity(tau, rho0)
    return u_star * np.sqrt(np.asarray(t, float) / np.sqrt(N2))


def mixed_layer_depth(ds, var="temp", threshold=0.02):
    """Depth at which ``var`` first departs from its surface value by ``threshold``.

    Returns a DataArray against time (positive depth in metres).
    """
    prof = ds[var]
    zdim = "z" if "z" in prof.dims else "zi"
    surf = prof.isel({zdim: -1})
    departed = np.abs(prof - surf) > threshold
    # z increases upward; the first departure from the top is the deepest index that is still mixed
    z = ds[zdim]
    mixed = xr.where(departed, -np.inf, z)      # where departed: -inf, else z
    # the mixed layer bottom is the lowest z that is not departed, searching from the top down
    out = []
    for it in range(prof.sizes["time"]):
        d = departed.isel(time=it).values
        zz = z.values if z.ndim == 1 else z.isel(time=it).values
        k = len(d) - 1
        while k > 0 and not d[k]:
            k -= 1
        out.append(-zz[k] if d.any() else -zz[0])
    return xr.DataArray(out, coords={"time": ds.time}, dims="time", name="mixed_layer_depth")


# ----------------------------------------------------------------------
# estuary
# ----------------------------------------------------------------------
def estuarine_profile(z, H, q, dbdx, A):
    """Classical tidally averaged estuarine circulation (Hansen & Rattray type),

        u = 3q/(2H) (1 - (z/H)^2) - H^3/(48 A) db/dx (1 - 9 (z/H)^2 - 8 (z/H)^3),

    with ``z`` from -H (bottom) to 0 (surface), ``q`` the river discharge per
    unit width [m^2/s], ``dbdx`` the along-channel buoyancy gradient [1/s^2]
    and ``A`` a constant eddy viscosity [m^2/s].
    """
    s = np.asarray(z, float) / H
    return 3 * q / (2 * H) * (1 - s ** 2) - H ** 3 / (48 * A) * dbdx * (1 - 9 * s ** 2 - 8 * s ** 3)


def buoyancy_gradient(dsdx, beta=7.7e-4):
    """db/dx = -g beta ds/dx for a linear equation of state."""
    return -g * beta * dsdx


def tidal_average(da, period=44714.0, cycles=1, end=None):
    """Average over the last ``cycles`` tidal periods (or the ``cycles`` before ``end``)."""
    t = da["seconds"] if "seconds" in da.coords else (da.time - da.time[0]) / np.timedelta64(1, "s")
    t_end = float(t[-1]) if end is None else float(end)
    sel = (t > t_end - cycles * period) & (t <= t_end)
    return da.isel(time=np.where(sel.values)[0]).mean("time")


def depth_average(da, ds):
    """Depth average of a layer-centred field weighted by the layer thickness ``h``."""
    h = ds["h"]
    return (da * h).sum("z") / h.sum("z")


def spectrum(da, dt=None):
    """One-sided power spectrum of a time series; returns (frequency [1/s], power)."""
    y = np.asarray(da.values, float)
    y = y - y.mean()
    if dt is None:
        t = da["seconds"].values
        dt = float(t[1] - t[0])
    n = len(y)
    p = np.abs(np.fft.rfft(y)) ** 2 / n
    f = np.fft.rfftfreq(n, dt)
    return f, p
