"""
1D enthalpy-method solver for an ice cap melted from below by volcanic heat.

Governing equation (conservation of energy, 1D, no advection):

    dE/dt = d/dz ( k dT/dz )

    E  volumetric enthalpy            [J m^-3]   (E = 0 is ice at the melting point)
    T  temperature                    [K]
    k  thermal conductivity           [W m^-1 K^-1]
    z  height above the bed           [m]        (z = 0 bed, z = H ice surface)
    t  time                           [s]

Heat flux is q = -k dT/dz [W m^-2], positive upward.

Equation of state T(E):
    E < 0          ice          T = T_m + E / (rho c_i)
    0 <= E <= rho L  ice + water  T = T_m,  melt fraction phi = E / (rho L)
    E > rho L      water        T = T_m + (E - rho L) / (rho c_w)

Discretization: finite volume on N equal cells, unknown E_j at each cell
center, forward-Euler (explicit) time stepping. The melt front is never
tracked directly; it is wherever the melt fraction goes from 1 to 0.

Assumptions: one density for ice and water (no volume change), meltwater
stays in place (no drainage or convection), constant melting point T_m.

This module uses only numpy so it can run from a script or a notebook.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

SECONDS_PER_YEAR = 365.25 * 24 * 3600  # [s]


@dataclass
class Params:
    """All model inputs, in SI units."""

    # Geometry and grid
    H: float = 200.0          # ice thickness [m]
    N: int = 200              # number of grid cells [-]

    # Boundary conditions
    T_s: float = 263.15       # surface temperature, fixed [K]
    basal_bc: str = "flux"    # "flux" (fixed q_b) or "temperature" (fixed T_b)
    q_b: float = 5.0          # basal heat flux into the ice [W m^-2]
    T_b: float = 278.15       # basal temperature [K]

    # Initial condition: None means a straight line from T_m at the bed
    # to T_s at the surface. A number means uniform ice at that temperature.
    T0: Optional[float] = None  # [K]

    # Run length and output
    t_end: float = 100 * SECONDS_PER_YEAR  # total simulated time [s]
    n_out: int = 100                       # number of saved snapshots [-]

    # Material properties
    rho: float = 917.0        # density of ice and water [kg m^-3]
    c_i: float = 2100.0       # heat capacity of ice [J kg^-1 K^-1]
    c_w: float = 4186.0       # heat capacity of water [J kg^-1 K^-1]
    k_i: float = 2.1          # conductivity of ice [W m^-1 K^-1]
    k_w: float = 0.56         # conductivity of water [W m^-1 K^-1]
    L: float = 3.34e5         # latent heat of fusion [J kg^-1]
    T_m: float = 273.15       # melting temperature [K]

    # Fraction of the stability limit used for the time step [-]
    safety: float = 0.25


@dataclass
class Result:
    """Model output. Arrays with a time axis have one row per snapshot."""

    z: np.ndarray               # cell-center heights [m], shape (N,)
    t: np.ndarray               # snapshot times [s], shape (n_out+1,)
    T: np.ndarray               # temperature [K], shape (n_out+1, N)
    phi: np.ndarray             # melt fraction [-], shape (n_out+1, N)
    melt_thickness: np.ndarray  # total meltwater thickness [m], shape (n_out+1,)
    energy_stored: np.ndarray   # total enthalpy per unit area [J m^-2]
    energy_in_base: np.ndarray  # cumulative heat in through the bed [J m^-2]
    energy_out_top: np.ndarray  # cumulative heat out through the surface [J m^-2]
    dt: float                   # time step used [s]
    n_steps: int                # number of time steps taken [-]


# ---------------------------------------------------------------------------
# Equation of state: converting between enthalpy and temperature
# ---------------------------------------------------------------------------

def temperature_from_enthalpy(E, p):
    """Temperature T(E) [K] from volumetric enthalpy E [J m^-3]."""
    E_melt = p.rho * p.L  # enthalpy needed to fully melt one cell [J m^-3]

    # Start every cell at the melting point (correct for partly melted cells).
    T = np.full_like(E, p.T_m, dtype=float)

    # Solid ice: all enthalpy below zero is sensible heat of ice.
    ice = E < 0
    T[ice] = p.T_m + E[ice] / (p.rho * p.c_i)

    # Liquid water: enthalpy above rho*L is sensible heat of water.
    water = E > E_melt
    T[water] = p.T_m + (E[water] - E_melt) / (p.rho * p.c_w)

    return T


def enthalpy_from_temperature(T, p):
    """Enthalpy E(T) [J m^-3]. T <= T_m is ice, T > T_m is water."""
    T = np.asarray(T, dtype=float)
    E_ice = p.rho * p.c_i * (T - p.T_m)
    E_water = p.rho * p.L + p.rho * p.c_w * (T - p.T_m)
    return np.where(T <= p.T_m, E_ice, E_water)


def melt_fraction(E, p):
    """Melt fraction phi [-]: 0 for ice, 1 for water, between in a mixed cell."""
    return np.clip(E / (p.rho * p.L), 0.0, 1.0)


def conductivity(E, p):
    """Cell conductivity k [W m^-1 K^-1], blended linearly by melt fraction."""
    phi = melt_fraction(E, p)
    return (1.0 - phi) * p.k_i + phi * p.k_w


# ---------------------------------------------------------------------------
# Grid, initial condition, and time step
# ---------------------------------------------------------------------------

def make_grid(p):
    """Cell size dz [m] and cell-center heights z [m]."""
    dz = p.H / p.N
    z = (np.arange(p.N) + 0.5) * dz  # centers at dz/2, 3dz/2, ...
    return dz, z


def initial_enthalpy(p, z):
    """Initial enthalpy profile E(z, t=0) [J m^-3]. Starts as all ice."""
    if p.T0 is None:
        # Linear profile: T_m at the bed, T_s at the surface.
        T_init = p.T_m + (p.T_s - p.T_m) * z / p.H
    else:
        T_init = np.full_like(z, p.T0)
    return enthalpy_from_temperature(T_init, p)


def stable_time_step(p, dz):
    """
    Largest safe explicit time step [s].

    Forward Euler on the diffusion equation needs dt <= dz^2 / (2 kappa) in
    the interior. The boundary cells see their boundary temperature only
    dz/2 away, which tightens the limit to dz^2 / (3 kappa). We use the
    larger of the ice and water diffusivities, times a safety factor below 1/3.
    """
    kappa_ice = p.k_i / (p.rho * p.c_i)    # [m^2 s^-1]
    kappa_water = p.k_w / (p.rho * p.c_w)  # [m^2 s^-1]
    kappa_max = max(kappa_ice, kappa_water)
    return p.safety * dz**2 / kappa_max


# ---------------------------------------------------------------------------
# Fluxes: the  k dT/dz  part of the PDE, evaluated on cell faces
# ---------------------------------------------------------------------------

def compute_fluxes(E, p, dz):
    """
    Heat flux q = -k dT/dz [W m^-2] on all N+1 cell faces, positive upward.

    Face 0 is the bed, face N is the ice surface, and face j sits between
    cells j-1 and j.
    """
    T = temperature_from_enthalpy(E, p)
    k = conductivity(E, p)
    q = np.zeros(p.N + 1)

    # Interior faces: q = -k dT/dz with dT/dz = (T_right - T_left) / dz.
    # k on a face is the harmonic mean of its two cells, so heat flows
    # correctly across the sharp jump in k between ice and water.
    k_face = 2.0 * k[:-1] * k[1:] / (k[:-1] + k[1:])
    q[1:-1] = -k_face * (T[1:] - T[:-1]) / dz

    # Bed boundary (face 0).
    if p.basal_bc == "flux":
        # Fixed heat flux (Neumann): the volcano supplies q_b directly.
        q[0] = p.q_b
    elif p.basal_bc == "temperature":
        # Fixed temperature (Dirichlet): the bed is held at T_b, which is
        # dz/2 below the center of the first cell.
        q[0] = -k[0] * (T[0] - p.T_b) / (dz / 2)
    else:
        raise ValueError("basal_bc must be 'flux' or 'temperature'")

    # Surface boundary (face N): fixed temperature T_s, dz/2 above the
    # center of the top cell.
    q[-1] = -k[-1] * (p.T_s - T[-1]) / (dz / 2)

    return q


# ---------------------------------------------------------------------------
# Time step: the  dE/dt  part of the PDE
# ---------------------------------------------------------------------------

def step(E, p, dz, dt):
    """
    Advance enthalpy by one forward-Euler step. Returns (E_new, q).

    Finite-volume form of dE/dt = -dq/dz: each cell gains the heat flowing
    in through its bottom face and loses the heat flowing out through its
    top face.
    """
    q = compute_fluxes(E, p, dz)
    dE_dt = -(q[1:] - q[:-1]) / dz  # [J m^-3 s^-1]
    return E + dt * dE_dt, q


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(p=None):
    """Run the model from t = 0 to p.t_end and return a Result."""
    if p is None:
        p = Params()

    dz, z = make_grid(p)
    E = initial_enthalpy(p, z)

    # Choose the number of steps so the run ends exactly at t_end and the
    # snapshots fall exactly on step boundaries.
    dt_max = stable_time_step(p, dz)
    steps_per_out = max(1, int(np.ceil(p.t_end / (p.n_out * dt_max))))
    n_steps = steps_per_out * p.n_out
    dt = p.t_end / n_steps

    # Storage for snapshots (row 0 is the initial condition).
    t_out = np.zeros(p.n_out + 1)
    E_out = np.zeros((p.n_out + 1, p.N))
    in_out = np.zeros(p.n_out + 1)
    top_out = np.zeros(p.n_out + 1)
    E_out[0] = E

    # Running totals of heat crossing the boundaries [J m^-2].
    energy_in_base = 0.0
    energy_out_top = 0.0

    for n in range(1, n_steps + 1):
        E, q = step(E, p, dz, dt)

        # Bookkeeping for the energy budget: heat through the bed and the
        # surface during this step, using the same fluxes the step used.
        energy_in_base += q[0] * dt
        energy_out_top += q[-1] * dt

        if n % steps_per_out == 0:
            i = n // steps_per_out
            t_out[i] = n * dt
            E_out[i] = E
            in_out[i] = energy_in_base
            top_out[i] = energy_out_top

    phi_out = melt_fraction(E_out, p)
    return Result(
        z=z,
        t=t_out,
        T=temperature_from_enthalpy(E_out, p),
        phi=phi_out,
        melt_thickness=phi_out.sum(axis=1) * dz,  # s(t) = sum(phi) dz
        energy_stored=E_out.sum(axis=1) * dz,
        energy_in_base=in_out,
        energy_out_top=top_out,
        dt=dt,
        n_steps=n_steps,
    )


if __name__ == "__main__":
    # Quick demo with the default parameters.
    res = run()
    print(f"time step: {res.dt / 86400:.2f} days, {res.n_steps} steps")
    print(f"melt thickness after {res.t[-1] / SECONDS_PER_YEAR:.0f} yr: "
          f"{res.melt_thickness[-1]:.2f} m")
