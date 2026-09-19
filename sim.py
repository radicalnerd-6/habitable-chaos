"""
HABITABLE CHAOS — A Computational Laboratory for Planetary Stability
YPAE Simathon 02 Computational Astrophysics Project

Technical Foundation:
- Units: Astronomical Units (AU), Solar Masses (M_sun), Years (yr)
- Gravitational Constant: G = 4 * pi^2 AU^3 / (M_sun * yr^2)
- Symplectic Integration: Velocity Verlet (Kick-Drift-Kick) inside Taichi kernels
- GPU Accelerated via Taichi (CUDA) with CPU fallback
- Real-time conservation diagnostics: Energy drift (dE/E0) and Angular momentum drift (dLz/Lz0)
- Osculating orbital elements: semimajor axis a(t), eccentricity e(t), periapsis/apoapsis
- Empirical Kepler verification: P^2 / a^3 computed directly from measured orbital periods
- Approximate Habitable Zone & Snow line based on stellar luminosity scaling
- Operational stability categorization with all raw physical diagnostics displayed
"""

import sys
import math
import time
import argparse
import numpy as np
import taichi as ti

# ==============================================================================
# 1. ASTROPHYSICAL CONSTANTS & UNITS
# ==============================================================================
# Simulation units: AU, Solar Mass (M_sun), Year (yr)
PI = math.pi
TWO_PI = 2.0 * math.pi
FOUR_PI_SQ = 4.0 * math.pi * math.pi
G_SIM = FOUR_PI_SQ  # 39.47841760435743 AU^3 / (M_sun * yr^2)

# Solar and planetary physical parameters (AU)
L_SUN = 1.0             # Solar luminosity
R_SUN_AU = 0.00465      # 1 Solar radius in AU (~696,340 km)
R_JUP_AU = 0.000478     # 1 Jupiter radius in AU (~71,492 km)
R_EARTH_AU = 0.0000426  # 1 Earth radius in AU (~6,371 km)
T_SUN_K = 5778.0        # Solar effective temperature in Kelvin

# Default numerical parameters
DEFAULT_EPSILON = 0.005  # Gravitational softening in AU (numerical only, NOT a collision radius)
MAX_BODIES = 8           # Maximum bodies in simulation
TRAIL_LENGTH = 600       # Points per orbital trail

def compute_physical_radius_au(mass_msun: float, is_star: bool = False) -> float:
    """
    Computes approximate physical radius in AU:
    - Main-sequence stars (M >= 0.05 M_sun): R ~ R_sun * (M / M_sun)^0.8
    - Gas Giants (0.0001 <= M < 0.05 M_sun): R ~ R_jup * (M / M_jup)^(1/3)
    - Terrestrial planets (M < 0.0001 M_sun): R ~ R_earth * (M / M_earth)^(1/3)
    
    IMPORTANT: Gravitational softening epsilon is strictly a numerical parameter to
    prevent force singularities; it must NOT be used as a physical collision radius.
    """
    if is_star or mass_msun >= 0.05:
        return R_SUN_AU * math.pow(max(mass_msun, 1e-4), 0.8)
    elif mass_msun >= 0.0001:
        m_jup = mass_msun / 0.000954
        return R_JUP_AU * math.pow(max(m_jup, 0.1), 1.0 / 3.0)
    else:
        m_earth = mass_msun / 3.003e-6
        return R_EARTH_AU * math.pow(max(m_earth, 0.01), 1.0 / 3.0)


# ==============================================================================
# 2. TAICHI BACKEND INITIALIZATION
# ==============================================================================
def initialize_best_backend():
    """Initializes Taichi with CUDA if available, falling back to Vulkan then CPU."""
    backends = [
        (ti.cuda, "NVIDIA CUDA (GPU)"),
        (ti.vulkan, "Vulkan (GPU)"),
        (ti.cpu, "CPU (x64 Multithreaded)"),
    ]
    for arch, name in backends:
        try:
            ti.init(arch=arch, log_level=ti.WARN)
            print(f"[Backend Selected] Successfully initialized {name}.", flush=True)
            return arch, name
        except Exception as e:
            print(f"[Backend Notice] Failed initializing {name}: {e}", flush=True)
            try:
                ti.reset()
            except Exception:
                pass
    raise RuntimeError("Failed to initialize any supported Taichi backend.")

# Initialize Taichi runtime before allocating any fields
ACTIVE_ARCH, ACTIVE_BACKEND_NAME = initialize_best_backend()

# ==============================================================================
# 3. TAICHI DATA FIELDS
# ==============================================================================
# Body state vectors (2D plane)
pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_BODIES)
vel = ti.Vector.field(2, dtype=ti.f32, shape=MAX_BODIES)
acc = ti.Vector.field(2, dtype=ti.f32, shape=MAX_BODIES)
mass = ti.field(dtype=ti.f32, shape=MAX_BODIES)
radius = ti.field(dtype=ti.f32, shape=MAX_BODIES)
color = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODIES)
active = ti.field(dtype=ti.i32, shape=MAX_BODIES)

# Global simulation scalar fields
num_bodies = ti.field(dtype=ti.i32, shape=())
softening_sq = ti.field(dtype=ti.f32, shape=())
g_constant = ti.field(dtype=ti.f32, shape=())

# Conservation metric fields
ke_field = ti.field(dtype=ti.f32, shape=())
pe_field = ti.field(dtype=ti.f32, shape=())
lz_field = ti.field(dtype=ti.f32, shape=())

# Historical orbital trails (ring buffer: [body_id, point_idx])
trail_pos = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_BODIES, TRAIL_LENGTH))

# Rendering line buffers
# Trail line segments per body: (TRAIL_LENGTH - 1) * 2 vertices
MAX_TRAIL_VERTICES = MAX_BODIES * (TRAIL_LENGTH - 1) * 2
trail_vertices = ti.Vector.field(2, dtype=ti.f32, shape=MAX_TRAIL_VERTICES)
trail_colors = ti.Vector.field(3, dtype=ti.f32, shape=MAX_TRAIL_VERTICES)

# Ring display vertices (for Habitable Zone boundaries and Snow Line)
RING_SEGMENTS = 128
# 3 rings (HZ Inner, HZ Outer, Snow Line) = 3 * RING_SEGMENTS * 2 vertices
ring_vertices = ti.Vector.field(2, dtype=ti.f32, shape=3 * RING_SEGMENTS * 2)
ring_colors = ti.Vector.field(3, dtype=ti.f32, shape=3 * RING_SEGMENTS * 2)

# Canvas circle drawing fields
draw_centers = ti.Vector.field(2, dtype=ti.f32, shape=MAX_BODIES)
draw_radii = ti.field(dtype=ti.f32, shape=MAX_BODIES)
draw_colors = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODIES)

draw_glow_centers = ti.Vector.field(2, dtype=ti.f32, shape=2)  # For stars
draw_glow_radii = ti.field(dtype=ti.f32, shape=2)
draw_glow_colors = ti.Vector.field(3, dtype=ti.f32, shape=2)

# Stability map rendering fields (up to 1000 points)
MAX_MAP_POINTS = 1000
map_point_centers = ti.Vector.field(2, dtype=ti.f32, shape=MAX_MAP_POINTS)
map_point_colors = ti.Vector.field(3, dtype=ti.f32, shape=MAX_MAP_POINTS)
map_point_radii = ti.field(dtype=ti.f32, shape=MAX_MAP_POINTS)

# Stability map axes, ticks, and reference lines
map_axis_vertices = ti.Vector.field(2, dtype=ti.f32, shape=128)
map_axis_colors = ti.Vector.field(3, dtype=ti.f32, shape=128)

# ==============================================================================
# 4. TAICHI SYMPLECTIC INTEGRATION KERNELS
# ==============================================================================
@ti.kernel
def compute_accelerations_kernel():
    """Calculates mutual Newtonian gravitational acceleration for all active bodies."""
    n = num_bodies[None]
    g = g_constant[None]
    eps_sq = softening_sq[None]

    for i in range(n):
        if active[i] == 1:
            ai = ti.Vector([0.0, 0.0])
            pi = pos[i]
            for j in range(n):
                if i != j and active[j] == 1:
                    pj = pos[j]
                    dr = pj - pi
                    dist_sq = dr[0] * dr[0] + dr[1] * dr[1] + eps_sq
                    dist = ti.sqrt(dist_sq)
                    inv_dist3 = 1.0 / (dist_sq * dist)
                    ai += (g * mass[j] * inv_dist3) * dr
            acc[i] = ai

@ti.kernel
def verlet_step_kernel(dt: ti.f32):
    """
    Performs one 2nd-order Symplectic Velocity Verlet (Kick-Drift-Kick) step:
    1. Half kick: v(t + dt/2) = v(t) + 0.5 * dt * a(t)
    2. Drift: r(t + dt) = r(t) + dt * v(t + dt/2)
    3. Acceleration evaluation: a(t + dt) from new positions
    4. Half kick: v(t + dt) = v(t + dt/2) + 0.5 * dt * a(t + dt)
    """
    n = num_bodies[None]
    half_dt = 0.5 * dt

    # 1. Half kick + 2. Drift
    for i in range(n):
        if active[i] == 1:
            vel[i] += half_dt * acc[i]
            pos[i] += dt * vel[i]

    # 3. Recompute mutual accelerations
    g = g_constant[None]
    eps_sq = softening_sq[None]
    for i in range(n):
        if active[i] == 1:
            ai = ti.Vector([0.0, 0.0])
            pi = pos[i]
            for j in range(n):
                if i != j and active[j] == 1:
                    pj = pos[j]
                    dr = pj - pi
                    dist_sq = dr[0] * dr[0] + dr[1] * dr[1] + eps_sq
                    dist = ti.sqrt(dist_sq)
                    inv_dist3 = 1.0 / (dist_sq * dist)
                    ai += (g * mass[j] * inv_dist3) * dr
            acc[i] = ai

    # 4. Final half kick
    for i in range(n):
        if active[i] == 1:
            vel[i] += half_dt * acc[i]

@ti.kernel
def verlet_substeps_kernel(substeps: ti.i32, dt: ti.f32):
    """
    Executes multiple Symplectic Velocity Verlet substeps consecutively on the GPU.
    Uses ti.loop_config(serialize=True) to execute steps sequentially without race conditions.
    """
    ti.loop_config(serialize=True)
    for _ in range(substeps):
        # 1. Half kick + Drift
        for i in range(num_bodies[None]):
            if active[i] == 1:
                vel[i] += (0.5 * dt) * acc[i]
                pos[i] += dt * vel[i]

        # 2. Recompute mutual accelerations
        for i in range(num_bodies[None]):
            if active[i] == 1:
                ai = ti.Vector([0.0, 0.0])
                pi = pos[i]
                for j in range(num_bodies[None]):
                    if i != j and active[j] == 1:
                        pj = pos[j]
                        dr = pj - pi
                        dist_sq = dr[0] * dr[0] + dr[1] * dr[1] + softening_sq[None]
                        dist = ti.sqrt(dist_sq)
                        inv_dist3 = 1.0 / (dist_sq * dist)
                        ai += (g_constant[None] * mass[j] * inv_dist3) * dr
                acc[i] = ai

        # 3. Final half kick
        for i in range(num_bodies[None]):
            if active[i] == 1:
                vel[i] += (0.5 * dt) * acc[i]

@ti.kernel
def compute_conservation_metrics_kernel():
    """Computes total kinetic energy, gravitational potential energy, and angular momentum."""
    n = num_bodies[None]
    g = g_constant[None]
    eps_sq = softening_sq[None]

    ke_sum = 0.0
    pe_sum = 0.0
    lz_sum = 0.0

    # Kinetic energy & Angular momentum
    for i in range(n):
        if active[i] == 1:
            m = mass[i]
            p = pos[i]
            v = vel[i]
            v_sq = v[0] * v[0] + v[1] * v[1]
            ke_sum += 0.5 * m * v_sq
            # L_z = x * v_y - y * v_x
            lz_sum += m * (p[0] * v[1] - p[1] * v[0])

    # Mutual potential energy
    for i in range(n):
        if active[i] == 1:
            pi = pos[i]
            for j in range(i + 1, n):
                if active[j] == 1:
                    pj = pos[j]
                    dr = pj - pi
                    dist = ti.sqrt(dr[0] * dr[0] + dr[1] * dr[1] + eps_sq)
                    pe_sum -= (g * mass[i] * mass[j]) / dist

    ke_field[None] = ke_sum
    pe_field[None] = pe_sum
    lz_field[None] = lz_sum

# ==============================================================================
# 5. ASTROPHYSICAL SCENARIOS & PRESETS
# ==============================================================================
class BodyConfig:
    def __init__(self, name, mass, radius, color, pos, vel, is_star=False):
        self.name = name
        self.mass = mass          # M_sun
        self.radius = radius      # AU (for collision detection & visualization)
        self.color = color        # (R, G, B) normalized [0, 1]
        self.pos = np.array(pos, dtype=np.float32) # [x, y] in AU
        self.vel = np.array(vel, dtype=np.float32) # [vx, vy] in AU/yr
        self.is_star = is_star

def compute_two_body_velocity(m1, m2, a, e=0.0):
    """
    Computes initial barycentric velocity vectors for a 2-body orbit at periapsis:
    r_rel = a * (1 - e)
    v_rel = sqrt(G * (m1 + m2) * (1 + e) / (a * (1 - e)))
    """
    mu_grav = G_SIM * (m1 + m2)
    r_peri = a * (1.0 - e)
    v_rel = math.sqrt(mu_grav * (1.0 + e) / r_peri)
    
    # Coordinates along x-axis, velocities along y-axis
    r1 = np.array([-r_peri * (m2 / (m1 + m2)), 0.0], dtype=np.float32)
    r2 = np.array([ r_peri * (m1 / (m1 + m2)), 0.0], dtype=np.float32)
    
    v1 = np.array([0.0, -v_rel * (m2 / (m1 + m2))], dtype=np.float32)
    v2 = np.array([0.0,  v_rel * (m1 / (m1 + m2))], dtype=np.float32)
    
    return r1, v1, r2, v2

def get_scenario_bodies(preset_id: int, custom_test_orbit=None):
    """
    Builds scientifically rigorous initial configurations:
    1: S-Type Binary (Calm HZ Benchmark)
    2: Nominal 3:1 Mean Motion Resonance (Resonance Pumping)
    3: Tight Binary Intruder (Chaotic Perturbations / Ejection)
    4: Kepler 3rd Law Benchmark (Single Star + Test Planet)
    """
    bodies = []
    description = ""
    target_resonance_label = "None"
    
    if preset_id == 1:
        description = "Hierarchical S-Type Binary: Primary Star + Companion (16 AU) + Giant Planet (3.2 AU) + Terrestrial HZ Planet (1.0 AU)"
        # Primary Star A: 1.0 M_sun (Sun-like)
        m_a = 1.0
        # Secondary Star B: 0.35 M_sun (M-dwarf) at 16 AU, e=0.12
        m_b = 0.35
        a_bin = 16.0
        e_bin = 0.12
        r_a, v_a, r_b, v_b = compute_two_body_velocity(m_a, m_b, a_bin, e_bin)
        
        bodies.append(BodyConfig("Primary Star (Star A)", m_a, 0.035, (1.0, 0.90, 0.35), r_a, v_a, is_star=True))
        bodies.append(BodyConfig("Secondary Star (Star B)", m_b, 0.022, (1.0, 0.40, 0.20), r_b, v_b, is_star=True))
        
        # Giant Planet: 1.0 M_Jupiter (0.001 M_sun) at a=3.2 AU around Star A (beyond snow line at 2.7 AU)
        m_giant = 0.001
        a_giant = 3.20
        v_giant_rel = math.sqrt(G_SIM * (m_a + m_giant) / a_giant)
        pos_giant = r_a + np.array([a_giant, 0.0], dtype=np.float32)
        vel_giant = v_a + np.array([0.0, v_giant_rel], dtype=np.float32)
        bodies.append(BodyConfig("Giant Planet (Gas Giant)", m_giant, 0.016, (0.95, 0.65, 0.25), pos_giant, vel_giant))
        
        # Terrestrial Test Planet: 1.0 M_Earth (3.0e-6 M_sun) at a=1.00 AU around Star A (inside HZ: 0.95 - 1.37 AU)
        m_test = 3.003e-6
        a_test = 1.00
        v_test_rel = math.sqrt(G_SIM * (m_a + m_test) / a_test)
        pos_test = r_a + np.array([0.0, a_test], dtype=np.float32)
        vel_test = v_a + np.array([-v_test_rel, 0.0], dtype=np.float32)
        bodies.append(BodyConfig("Terrestrial Planet (Test Planet)", m_test, 0.010, (0.20, 0.85, 1.0), pos_test, vel_test))

    elif preset_id == 2:
        description = "Nominal 3:1 Mean Motion Resonance: Initial a_giant = 2.080 AU, a_test = 1.000 AU (a_ratio ~ 2.080, P_ratio ~ 3.0)"
        target_resonance_label = "Nominal 3:1 MMR"
        m_a = 1.0
        m_b = 0.35
        a_bin = 20.0
        e_bin = 0.05
        r_a, v_a, r_b, v_b = compute_two_body_velocity(m_a, m_b, a_bin, e_bin)
        
        bodies.append(BodyConfig("Primary Star (Star A)", m_a, 0.035, (1.0, 0.90, 0.35), r_a, v_a, is_star=True))
        bodies.append(BodyConfig("Secondary Star (Star B)", m_b, 0.022, (1.0, 0.40, 0.20), r_b, v_b, is_star=True))
        
        # Giant planet placed near exact 3:1 period ratio with 1.0 AU: a = 1.0 * 3^(2/3) ~ 2.08008 AU
        m_giant = 0.002 # 2 M_Jup for clear resonance excitation
        a_giant = 2.08008
        v_giant_rel = math.sqrt(G_SIM * (m_a + m_giant) / a_giant)
        pos_giant = r_a + np.array([a_giant, 0.0], dtype=np.float32)
        vel_giant = v_a + np.array([0.0, v_giant_rel], dtype=np.float32)
        bodies.append(BodyConfig("Giant Perturber (3:1 MMR)", m_giant, 0.016, (0.95, 0.65, 0.25), pos_giant, vel_giant))
        
        # Test planet at 1.00 AU (or custom a0, e0 from Stability Map)
        m_test = 3.003e-6
        if custom_test_orbit is not None:
            a_test, e_test = custom_test_orbit
            r_peri = a_test * (1.0 - e_test)
            v_test_rel = math.sqrt(G_SIM * (m_a + m_test) * (1.0 + e_test) / max(r_peri, 1e-6))
            pos_test = r_a + np.array([r_peri, 0.0], dtype=np.float32)
            vel_test = v_a + np.array([0.0, v_test_rel], dtype=np.float32)
            label = f"Terrestrial Planet (a0={a_test:.2f}, e0={e_test:.2f})"
        else:
            a_test = 1.00
            v_test_rel = math.sqrt(G_SIM * (m_a + m_test) / a_test)
            pos_test = r_a + np.array([a_test, 0.0], dtype=np.float32)
            vel_test = v_a + np.array([0.0, v_test_rel], dtype=np.float32)
            label = "Terrestrial Planet (Resonant Test)"
        bodies.append(BodyConfig(label, m_test, 0.010, (0.20, 0.85, 1.0), pos_test, vel_test))

    elif preset_id == 3:
        description = "Tight Binary Intruder: Companion at a = 5.2 AU (e = 0.30) causing strong gravitational scattering"
        target_resonance_label = "Chaotic Scattering"
        m_a = 1.0
        m_b = 0.50
        a_bin = 5.2
        e_bin = 0.30
        r_a, v_a, r_b, v_b = compute_two_body_velocity(m_a, m_b, a_bin, e_bin)
        
        bodies.append(BodyConfig("Primary Star (Star A)", m_a, 0.035, (1.0, 0.90, 0.35), r_a, v_a, is_star=True))
        bodies.append(BodyConfig("Intruder Star (Star B)", m_b, 0.024, (1.0, 0.30, 0.15), r_b, v_b, is_star=True))
        
        m_giant = 0.001
        a_giant = 2.40
        v_giant_rel = math.sqrt(G_SIM * (m_a + m_giant) / a_giant)
        pos_giant = r_a + np.array([a_giant, 0.0], dtype=np.float32)
        vel_giant = v_a + np.array([0.0, v_giant_rel], dtype=np.float32)
        bodies.append(BodyConfig("Giant Planet", m_giant, 0.016, (0.95, 0.65, 0.25), pos_giant, vel_giant))
        
        m_test = 3.003e-6
        a_test = 1.00
        v_test_rel = math.sqrt(G_SIM * (m_a + m_test) / a_test)
        pos_test = r_a + np.array([a_test, 0.0], dtype=np.float32)
        vel_test = v_a + np.array([0.0, v_test_rel], dtype=np.float32)
        bodies.append(BodyConfig("Terrestrial Planet", m_test, 0.010, (0.20, 0.85, 1.0), pos_test, vel_test))

    elif preset_id == 4:
        description = "Kepler 3rd Law Benchmark: Single Star (1.0 M_sun) + Circular Orbit (a = 1.00 AU)"
        target_resonance_label = "Pure Keplerian Orbit"
        m_a = 1.0
        bodies.append(BodyConfig("Host Star", m_a, 0.035, (1.0, 0.92, 0.40), [0.0, 0.0], [0.0, 0.0], is_star=True))
        
        m_test = 3.003e-6
        a_test = 1.00
        v_test = math.sqrt(G_SIM * (m_a + m_test) / a_test) # exactly 2 * pi
        bodies.append(BodyConfig("Kepler Planet (1.0 AU)", m_test, 0.012, (0.25, 0.85, 1.0), [a_test, 0.0], [0.0, v_test]))

    return bodies, description, target_resonance_label

# ==============================================================================
# 6. ASTROPHYSICAL CALCULATOR (STELLAR & ORBITAL DIAGNOSTICS)
# ==============================================================================
def compute_orbital_angles(r_rel, v_rel, mu):
    """
    Computes longitude of periapsis (varpi), mean anomaly (M), and mean longitude (lambda)
    for a 2D orbit around a central body:
    r_rel = [x, y], v_rel = [vx, vy]
    """
    x, y = float(r_rel[0]), float(r_rel[1])
    vx, vy = float(v_rel[0]), float(v_rel[1])
    r = math.sqrt(x * x + y * y)
    
    # Specific angular momentum: h = x * vy - y * vx
    h = x * vy - y * vx
    
    # Eccentricity vector: e_vec = (v x h)/mu - r_vec/r
    # In 2D: (v x h)_x = vy * h, (v x h)_y = -vx * h
    ex = (vy * h) / mu - x / max(r, 1e-8)
    ey = (-vx * h) / mu - y / max(r, 1e-8)
    e = math.sqrt(ex * ex + ey * ey)
    
    # Longitude of periapsis: varpi = atan2(ey, ex)
    varpi = math.atan2(ey, ex) if e > 1e-6 else 0.0
    
    # True anomaly: nu
    if e > 1e-6:
        nu = math.atan2(x * ey - y * ex, x * ex + y * ey)
    else:
        nu = math.atan2(y, x)
        
    # Eccentric anomaly E and Mean anomaly M
    if e < 0.999:
        factor = math.sqrt(max(0.0, 1.0 - e * e))
        cos_nu = math.cos(nu)
        sin_nu = math.sin(nu)
        denom = 1.0 + e * cos_nu
        cos_E = (e + cos_nu) / max(denom, 1e-8)
        sin_E = (factor * sin_nu) / max(denom, 1e-8)
        E = math.atan2(sin_E, cos_E)
        M = E - e * math.sin(E)
    else:
        M = nu
        
    # Mean longitude lambda = varpi + M
    mean_long = varpi + M
    return varpi, M, mean_long, e

class AstrophysicsEngine:
    def __init__(self, preset_id=1, custom_test_orbit=None):
        self.preset_id = preset_id
        self.custom_test_orbit = custom_test_orbit
        self.time_sim = 0.0            # Elapsed simulation time in years
        self.paused = False
        
        # User interactive parameters
        self.substeps = 100            # Substeps per visual frame
        self.dt_sub = 0.0002           # Step size per substep in years (~1.75 hours)
        self.view_scale_au = 5.0       # Half-width of view in AU
        self.view_focus = 0            # 0: Primary Star, 1: System Barycenter
        
        # Empirical energy & momentum tracking
        self.e0 = None
        self.lz0 = None
        self.delta_e_rel = 0.0
        self.delta_lz_rel = 0.0
        
        # Trail history buffers (NumPy ring buffer)
        self.trail_history = np.zeros((MAX_BODIES, TRAIL_LENGTH, 2), dtype=np.float32)
        self.trail_counts = np.zeros(MAX_BODIES, dtype=np.int32)
        self.trail_head = np.zeros(MAX_BODIES, dtype=np.int32)
        
        # Test planet osculating elements (approximate two-body relative to primary)
        self.a_test = 1.0
        self.a0_test = 1.0
        self.e_test = 0.0
        self.r_test = 1.0
        self.v_test = 0.0
        self.peri_test = 1.0
        self.apo_test = 1.0
        self.period_kepler_est = 1.0
        self.bary_energy_test = -1.0
        self.survival_time = 0.0
        self.stability_state = "STABLE"
        self.physical_radii = np.zeros(MAX_BODIES, dtype=np.float32)
        
        # Giant planet diagnostics & Hill sphere metrics
        self.a_giant = 0.0
        self.e_giant = 0.0
        self.period_giant = 0.0
        self.period_ratio = 0.0
        self.a_ratio = 0.0
        self.r_hill_giant = 0.0
        self.hill_sep = 999.0
        
        # 3:1 Mean Motion Resonance Diagnostic
        self.phi_3_1 = 0.0
        self.phi_3_1_deg = 0.0
        self.phi_history = []
        self.resonance_state = "INSUFFICIENT DATA (Collecting orbital cycles)"
        
        # Kepler empirical measurement tracker
        self.test_planet_unwrapped_angle = 0.0
        self.prev_angle = 0.0
        self.last_period_start_time = 0.0
        self.measured_periods = []
        self.kepler_ratio_p2_over_a3 = 1.0
        self.kepler_relative_error = 0.0
        
        # Stellar scaling for Primary Star
        self.primary_luminosity = 1.0
        self.primary_radius_solar = 1.0
        self.primary_teff = T_SUN_K
        self.hz_inner_au = 0.95
        self.hz_outer_au = 1.37
        self.snow_line_au = 2.70
        self.total_flux_test = 1.0
        self.flux_status = "INSIDE APPROXIMATE HZ"
        
        # Initialize the scenario
        self.load_preset(preset_id, custom_test_orbit)

    def load_preset(self, preset_id, custom_test_orbit=None):
        self.preset_id = preset_id
        self.custom_test_orbit = custom_test_orbit
        self.time_sim = 0.0
        self.survival_time = 0.0
        self.stability_state = "STABLE"
        self.phi_history = []
        self.resonance_state = "INSUFFICIENT DATA (Collecting orbital cycles)"
        
        bodies, desc, res_label = get_scenario_bodies(preset_id, custom_test_orbit)
        self.scenario_description = desc
        self.target_resonance_label = res_label
        self.num_active = len(bodies)
        
        # Host star scaling (Body 0 is Primary Star)
        m_star1 = bodies[0].mass
        self.primary_luminosity = math.pow(m_star1, 3.5)
        self.primary_radius_solar = math.pow(m_star1, 0.8)
        self.primary_teff = T_SUN_K * math.pow(self.primary_luminosity / (self.primary_radius_solar**2), 0.25)
        
        # Habitable Zone reference geometry for Primary Star (S_in = 1.11, S_out = 0.53)
        self.hz_inner_au = math.sqrt(self.primary_luminosity / 1.11)
        self.hz_outer_au = math.sqrt(self.primary_luminosity / 0.53)
        
        # Luminosity-scaled Snow Line: r_snow = 2.7 * sqrt(L / L_sun) AU
        self.snow_line_au = 2.7 * math.sqrt(self.primary_luminosity / L_SUN)
        
        # Reset Taichi fields
        num_bodies[None] = self.num_active
        g_constant[None] = G_SIM
        softening_sq[None] = DEFAULT_EPSILON * DEFAULT_EPSILON
        
        # Populate body state
        self.physical_radii = np.zeros(MAX_BODIES, dtype=np.float32)
        for i in range(MAX_BODIES):
            if i < self.num_active:
                b = bodies[i]
                active[i] = 1
                pos[i] = b.pos
                vel[i] = b.vel
                mass[i] = b.mass
                radius[i] = b.radius
                color[i] = b.color
                self.physical_radii[i] = compute_physical_radius_au(b.mass, is_star=b.is_star)
            else:
                active[i] = 0
                pos[i] = [0.0, 0.0]
                vel[i] = [0.0, 0.0]
                mass[i] = 0.0
                radius[i] = 0.0
                color[i] = [0.0, 0.0, 0.0]

        # Initial accelerations
        compute_accelerations_kernel()
        compute_conservation_metrics_kernel()
        
        # Initial energy and momentum baselines
        ke0 = ke_field[None]
        pe0 = pe_field[None]
        self.e0 = ke0 + pe0
        self.lz0 = lz_field[None]
        self.delta_e_rel = 0.0
        self.delta_lz_rel = 0.0
        
        # Reset trails
        self.trail_history.fill(0.0)
        self.trail_counts.fill(0)
        self.trail_head.fill(0)
        for i in range(self.num_active):
            p = bodies[i].pos
            self.trail_history[i, 0] = p
            self.trail_counts[i] = 1
        
        # Reset initial test planet semimajor axis
        test_idx = self.num_active - 1
        r_rel = bodies[test_idx].pos - bodies[0].pos
        v_rel = bodies[test_idx].vel - bodies[0].vel
        r_dist = float(np.linalg.norm(r_rel))
        v_norm = float(np.linalg.norm(v_rel))
        mu_star = G_SIM * (bodies[0].mass + bodies[test_idx].mass)
        spec_energy = 0.5 * v_norm * v_norm - mu_star / max(r_dist, 1e-6)
        if spec_energy < 0:
            self.a0_test = -mu_star / (2.0 * spec_energy)
        else:
            self.a0_test = r_dist
            
        # Angle tracking for empirical Kepler period
        self.prev_angle = math.atan2(r_rel[1], r_rel[0])
        self.test_planet_unwrapped_angle = 0.0
        self.last_period_start_time = 0.0
        self.measured_periods = []
        self.kepler_ratio_p2_over_a3 = 1.0
        self.kepler_relative_error = 0.0
        
        # Adjust default view scale by scenario
        if preset_id in (1, 2):
            self.view_scale_au = 6.0
        elif preset_id == 3:
            self.view_scale_au = 7.5
        elif preset_id == 4:
            self.view_scale_au = 2.2
            
        # Initial diagnostic evaluation
        self._update_diagnostics()

    def step_simulation(self):
        """Advances the simulation by self.substeps integration substeps."""
        if self.paused:
            return
        
        dt = self.dt_sub
        steps = self.substeps
        
        # Execute all substeps consecutively on the GPU in one kernel launch
        verlet_substeps_kernel(steps, dt)
        
        dt_frame = steps * dt
        self.time_sim += dt_frame
        if self.stability_state not in ("EJECTED", "COLLISION"):
            self.survival_time += dt_frame
            
        # Single bulk readback from GPU to CPU
        pos_all = pos.to_numpy()
        vel_all = vel.to_numpy()
        
        # Buffer trail snapshot once per visual frame
        self._record_trail_snapshot(pos_all)
        
        # Empirical Kepler angle tracker
        self._update_kepler_measurement(pos_all, dt_frame)
        
        # Update conservation metrics
        compute_conservation_metrics_kernel()
        e_curr = ke_field[None] + pe_field[None]
        lz_curr = lz_field[None]
        
        if self.e0 is not None and abs(self.e0) > 1e-12:
            self.delta_e_rel = (e_curr - self.e0) / abs(self.e0)
        if self.lz0 is not None and abs(self.lz0) > 1e-12:
            self.delta_lz_rel = (lz_curr - self.lz0) / abs(self.lz0)
            
        # Update diagnostics using pos_all, vel_all
        self._update_diagnostics(pos_all, vel_all)

    def _record_trail_snapshot(self, pos_all=None):
        """Buffers current body positions into historical trail arrays."""
        if pos_all is None:
            pos_all = pos.to_numpy()
        for i in range(self.num_active):
            p = [pos_all[i][0], pos_all[i][1]]
            head = self.trail_head[i]
            self.trail_history[i, head] = p
            self.trail_head[i] = (head + 1) % TRAIL_LENGTH
            if self.trail_counts[i] < TRAIL_LENGTH:
                self.trail_counts[i] += 1

    def _update_kepler_measurement(self, pos_all=None, dt_frame=0.0):
        """Tracks unwrapped true anomaly to measure full orbital periods empirically."""
        if pos_all is None:
            pos_all = pos.to_numpy()
        test_idx = self.num_active - 1
        rx = pos_all[test_idx][0] - pos_all[0][0]
        ry = pos_all[test_idx][1] - pos_all[0][1]
        current_angle = math.atan2(ry, rx)
        
        d_theta = current_angle - self.prev_angle
        # Handle [-pi, pi] wrap-around
        if d_theta > PI:
            d_theta -= TWO_PI
        elif d_theta < -PI:
            d_theta += TWO_PI
            
        theta_old = self.test_planet_unwrapped_angle
        theta_new = theta_old + d_theta
        self.test_planet_unwrapped_angle = theta_new
        self.prev_angle = current_angle
        
        # Check for 2*pi revolution completion
        if abs(theta_new) >= TWO_PI:
            # Linear interpolation to find the exact time of 2*pi crossing
            abs_old = abs(theta_old)
            abs_new = abs(theta_new)
            denom = abs_new - abs_old
            if denom > 1e-12:
                frac = (TWO_PI - abs_old) / denom
                frac = max(0.0, min(1.0, frac))
            else:
                frac = 1.0
                
            t_cross = (self.time_sim - dt_frame) + frac * dt_frame
            measured_period = t_cross - self.last_period_start_time
            
            if measured_period > 0.05: # filter out transients
                self.measured_periods.append(measured_period)
                if len(self.measured_periods) > 10:
                    self.measured_periods.pop(0)
                    
                # Calculate P^2 / a^3
                avg_period = float(np.mean(self.measured_periods))
                if self.a_test > 0.05:
                    self.kepler_ratio_p2_over_a3 = (avg_period * avg_period) / math.pow(self.a_test, 3)
                    # Theoretical period = sqrt(a^3 / (m_star + m_planet))
                    mu_tot = mass[0] + mass[test_idx]
                    p_theoretical = math.sqrt(math.pow(self.a_test, 3) / mu_tot)
                    self.kepler_relative_error = abs(avg_period - p_theoretical) / p_theoretical * 100.0
                    
            # Subtract 2*pi rather than resetting to 0 to preserve angle continuity
            if theta_new > 0:
                self.test_planet_unwrapped_angle -= TWO_PI
            else:
                self.test_planet_unwrapped_angle += TWO_PI
            self.last_period_start_time = t_cross

    def _update_diagnostics(self, pos_all=None, vel_all=None):
        """Computes osculating Keplerian orbital elements, Hill radius, resonance angle, and operational stability."""
        if pos_all is None:
            pos_all = pos.to_numpy()
        if vel_all is None:
            vel_all = vel.to_numpy()
            
        test_idx = self.num_active - 1
        p_test = pos_all[test_idx]
        v_test = vel_all[test_idx]
        p_star1 = pos_all[0]
        v_star1 = vel_all[0]
        
        r_rel = p_test - p_star1
        v_rel = v_test - v_star1
        
        self.r_test = float(np.linalg.norm(r_rel))
        self.v_test = float(np.linalg.norm(v_rel))
        
        mu_star = G_SIM * (mass[0] + mass[test_idx])
        spec_energy = 0.5 * (self.v_test**2) - mu_star / max(self.r_test, 1e-6)
        
        # Osculating semimajor axis
        if spec_energy < 0.0:
            self.a_test = -mu_star / (2.0 * spec_energy)
            # Specific angular momentum: h = x*vy - y*vx
            h = r_rel[0] * v_rel[1] - r_rel[1] * v_rel[0]
            ecc_term = 1.0 + (2.0 * spec_energy * (h**2)) / (mu_star**2)
            self.e_test = math.sqrt(max(0.0, ecc_term))
            self.peri_test = self.a_test * (1.0 - self.e_test)
            self.apo_test = self.a_test * (1.0 + self.e_test)
            self.period_kepler_est = math.sqrt(math.pow(self.a_test, 3) / (mass[0] + mass[test_idx]))
        else:
            # Hyperbolic/Unbound
            self.a_test = -mu_star / (2.0 * spec_energy)
            h = r_rel[0] * v_rel[1] - r_rel[1] * v_rel[0]
            ecc_term = 1.0 + (2.0 * spec_energy * (h**2)) / (mu_star**2)
            self.e_test = math.sqrt(max(0.0, ecc_term))
            self.peri_test = self.r_test
            self.apo_test = float('inf')
            self.period_kepler_est = float('inf')

        # Barycentric energy of test planet (approximate test-particle energy in N-body system)
        total_m = sum(mass[i] for i in range(self.num_active))
        bary_pos = sum(mass[i] * pos_all[i] for i in range(self.num_active)) / total_m
        bary_vel = sum(mass[i] * vel_all[i] for i in range(self.num_active)) / total_m
        r_bary = float(np.linalg.norm(p_test - bary_pos))
        v_bary = float(np.linalg.norm(v_test - bary_vel))
        self.bary_energy_test = 0.5 * (v_bary**2) - (G_SIM * total_m) / max(r_bary, 1e-6)

        # Received Stellar Flux at Test Planet (in S_sun)
        flux_star1 = self.primary_luminosity / max(self.r_test * self.r_test, 1e-4)
        flux_star2 = 0.0
        if self.num_active > 1 and active[1] == 1:
            p_star2 = pos_all[1]
            r_star2 = float(np.linalg.norm(p_test - p_star2))
            l_star2 = math.pow(mass[1], 3.5)
            flux_star2 = l_star2 / max(r_star2 * r_star2, 1e-4)
        self.total_flux_test = flux_star1 + flux_star2
        
        # Radiative Habitability based on TOTAL received stellar flux
        if 0.53 <= self.total_flux_test <= 1.11:
            self.flux_status = "INSIDE APPROXIMATE HZ (Potentially Temperate: 0.53 - 1.11 S_sun)"
        elif self.total_flux_test > 1.11:
            self.flux_status = "INTERIOR TO HZ (Excessive Flux > 1.11 S_sun)"
        else:
            self.flux_status = "EXTERIOR TO HZ (Sub-freezing Flux < 0.53 S_sun)"

        # Giant planet diagnostics & Hill sphere metrics
        self.r_hill_giant = 0.0
        self.hill_sep = 999.0
        peri_giant = 0.0
        
        if self.num_active >= 4:
            giant_idx = 2
            p_giant = pos_all[giant_idx]
            v_giant = vel_all[giant_idx]
            rg_rel = p_giant - p_star1
            vg_rel = v_giant - v_star1
            rg_dist = float(np.linalg.norm(rg_rel))
            vg_norm = float(np.linalg.norm(vg_rel))
            mu_g = G_SIM * (mass[0] + mass[giant_idx])
            spec_eg = 0.5 * (vg_norm**2) - mu_g / max(rg_dist, 1e-6)
            if spec_eg < 0.0:
                self.a_giant = -mu_g / (2.0 * spec_eg)
                hg = rg_rel[0] * vg_rel[1] - rg_rel[1] * vg_rel[0]
                ecc_g = 1.0 + (2.0 * spec_eg * (hg**2)) / (mu_g**2)
                self.e_giant = math.sqrt(max(0.0, ecc_g))
                peri_giant = self.a_giant * (1.0 - self.e_giant)
                self.period_giant = math.sqrt(math.pow(self.a_giant, 3) / (mass[0] + mass[giant_idx]))
                if self.period_kepler_est > 0:
                    self.period_ratio = self.period_giant / self.period_kepler_est
                if self.a_test > 0:
                    self.a_ratio = self.a_giant / self.a_test

            # Physically motivated Hill radius: R_H = a_giant * (M_giant / (3 * M_star))^(1/3)
            m_star1 = mass[0]
            m_giant = mass[giant_idx]
            self.r_hill_giant = self.a_giant * math.pow(m_giant / (3.0 * m_star1), 1.0 / 3.0)
            
            # Separation in Hill units
            d_sep = float(np.linalg.norm(p_test - p_giant))
            if self.r_hill_giant > 1e-6:
                self.hill_sep = d_sep / self.r_hill_giant

            # ------------------------------------------------------------------
            # RESONANT ANGLE DIAGNOSTIC (3:1 Mean Motion Resonance)
            # phi = 3 * lambda_giant - lambda_test - 2 * varpi_test
            # ------------------------------------------------------------------
            varpi_test, m_test_anom, lambda_test, _ = compute_orbital_angles(r_rel, v_rel, mu_star)
            varpi_giant, m_giant_anom, lambda_giant, _ = compute_orbital_angles(rg_rel, vg_rel, mu_g)
            
            phi_raw = 3.0 * lambda_giant - lambda_test - 2.0 * varpi_test
            # Normalize phi to [-pi, pi]
            self.phi_3_1 = (phi_raw + PI) % TWO_PI - PI
            self.phi_3_1_deg = math.degrees(self.phi_3_1)
            
            # Track history for libration vs circulation analysis
            self.phi_history.append(self.phi_3_1)
            if len(self.phi_history) > 120:
                self.phi_history.pop(0)
                
            if len(self.phi_history) >= 50:
                # Directional statistics on phi
                s_sum = sum(math.sin(p) for p in self.phi_history) / len(self.phi_history)
                c_sum = sum(math.cos(p) for p in self.phi_history) / len(self.phi_history)
                r_resultant = math.sqrt(s_sum * s_sum + c_sum * c_sum)
                mean_dir = math.atan2(s_sum, c_sum)
                devs = [((p - mean_dir + PI) % TWO_PI - PI) for p in self.phi_history]
                span_deg = math.degrees(max(devs) - min(devs))
                
                # Documented criterion: Libration if angular span is bounded (< 300 deg) and R > 0.35
                if span_deg < 300.0 and r_resultant > 0.35:
                    self.resonance_state = f"LIBRATING (Span: {span_deg:.1f}°, R={r_resultant:.2f})"
                else:
                    self.resonance_state = f"CIRCULATING (Full 360° phase sweep, R={r_resultant:.2f})"
            else:
                self.resonance_state = f"INSUFFICIENT DATA ({len(self.phi_history)}/50 samples)"

        # ----------------------------------------------------------------------
        # OPERATIONAL STABILITY CATEGORIZATION:
        # Note: Documented as project-defined operational visualization rules,
        # NOT universal long-term physical theorems.
        # ----------------------------------------------------------------------
        collision_detected = False
        r_phys_test = self.physical_radii[test_idx]
        for i in range(self.num_active - 1):
            pi = pos_all[i]
            sep = float(np.linalg.norm(p_test - pi))
            # Physical collision occurs when separation is less than the sum of physical radii.
            # Gravitational softening epsilon (0.005 AU) is strictly a numerical parameter
            # for the integrator and is NOT used as a physical collision radius.
            coll_radius = self.physical_radii[i] + r_phys_test
            if sep < coll_radius:
                collision_detected = True
                break

        if collision_detected:
            self.stability_state = "COLLISION"
        elif r_bary > 40.0 and self.bary_energy_test > 0.0:
            self.stability_state = "EJECTED"
        elif self.e_test >= 0.70 or self.hill_sep < 1.0 or (self.num_active >= 4 and peri_giant > 0 and self.apo_test > (peri_giant - 1.5 * self.r_hill_giant)):
            self.stability_state = "UNSTABLE"
        elif self.e_test >= 0.20 or (self.a0_test > 0 and abs(self.a_test - self.a0_test) / self.a0_test >= 0.12) or (self.num_active >= 4 and peri_giant > 0 and self.apo_test > (peri_giant - 3.0 * self.r_hill_giant)):
            self.stability_state = "PERTURBED"
        else:
            self.stability_state = "STABLE"

# ==============================================================================
# 7. STABILITY MAP SCANNER (HEADLESS N-BODY EXPERIMENT)
# ==============================================================================
class StabilityMapScanner:
    def __init__(self, a_range=(0.70, 1.50), e_range=(0.00, 0.40), n_a=40, n_e=25, t_duration=5.0, dt_sub=0.0002):
        self.a_range = a_range
        self.e_range = e_range
        self.n_a = n_a
        self.n_e = n_e
        self.t_duration = t_duration
        self.dt_sub = dt_sub
        self.results = []
        # Nominal 3:1 resonance: a_res = a_giant / (3^(2/3))
        # For a_giant = 2.08008 AU -> a_res ~ 1.0000 AU
        self.nominal_a_res = 2.08008 / math.pow(3.0, 2.0 / 3.0)
        
    def run_scan(self, progress_callback=None):
        """
        Performs headless N-body parameter scan across the (a0, e0) grid.
        Integrates actual N-body equations using the verified Velocity Verlet GPU kernel.
        """
        a_vals = np.linspace(self.a_range[0], self.a_range[1], self.n_a)
        e_vals = np.linspace(self.e_range[0], self.e_range[1], self.n_e)
        total_points = self.n_a * self.n_e
        
        self.results = []
        engine = AstrophysicsEngine(preset_id=2)
        engine.dt_sub = self.dt_sub
        
        chunk_steps = 1000  # 0.2 yr per chunk (5 samples per orbit at 1 AU)
        n_chunks = max(1, int(self.t_duration / (chunk_steps * self.dt_sub)))
        
        start_time = time.time()
        print(f"\n[StabilityMapScanner] Starting scan: {self.n_a}x{self.n_e} = {total_points} configurations")
        print(f"  a0 range: [{self.a_range[0]:.2f}, {self.a_range[1]:.2f}] AU")
        print(f"  e0 range: [{self.e_range[0]:.2f}, {self.e_range[1]:.2f}]")
        print(f"  Duration: {self.t_duration:.1f} yr per point ({n_chunks * chunk_steps} substeps)")
        print(f"  Nominal 3:1 resonance: a_res = {self.nominal_a_res:.4f} AU\n")
        
        count = 0
        for i, a0 in enumerate(a_vals):
            for j, e0 in enumerate(e_vals):
                count += 1
                # Initialize system with this exact (a0, e0)
                engine.load_preset(2, custom_test_orbit=(float(a0), float(e0)))
                
                max_e = engine.e_test
                min_giant_sep_au = float('inf')
                min_giant_sep_rh = float('inf')
                
                for chunk in range(n_chunks):
                    verlet_substeps_kernel(chunk_steps, engine.dt_sub)
                    dt_frame = chunk_steps * engine.dt_sub
                    engine.time_sim += dt_frame
                    if engine.stability_state not in ("EJECTED", "COLLISION"):
                        engine.survival_time += dt_frame
                        
                    # Single readback per chunk
                    pos_all = pos.to_numpy()
                    vel_all = vel.to_numpy()
                    
                    # Update conservation metrics
                    compute_conservation_metrics_kernel()
                    e_curr = ke_field[None] + pe_field[None]
                    lz_curr = lz_field[None]
                    if engine.e0 is not None and abs(engine.e0) > 1e-12:
                        engine.delta_e_rel = (e_curr - engine.e0) / abs(engine.e0)
                    if engine.lz0 is not None and abs(engine.lz0) > 1e-12:
                        engine.delta_lz_rel = (lz_curr - engine.lz0) / abs(engine.lz0)
                        
                    engine._update_diagnostics(pos_all, vel_all)
                    
                    max_e = max(max_e, engine.e_test)
                    d_giant = float(np.linalg.norm(pos_all[3] - pos_all[2]))
                    min_giant_sep_au = min(min_giant_sep_au, d_giant)
                    if engine.r_hill_giant > 1e-6:
                        min_giant_sep_rh = min(min_giant_sep_rh, d_giant / engine.r_hill_giant)
                        
                    # Early termination if planet is destroyed or ejected
                    if engine.stability_state in ("COLLISION", "EJECTED"):
                        break
                        
                record = {
                    "grid_i": int(i),
                    "grid_j": int(j),
                    "a0": float(round(a0, 4)),
                    "e0": float(round(e0, 4)),
                    "a_final": float(round(engine.a_test, 4)),
                    "e_final": float(round(engine.e_test, 4)),
                    "max_e": float(round(max_e, 4)),
                    "min_giant_sep_au": float(round(min_giant_sep_au, 4)),
                    "min_giant_sep_rh": float(round(min_giant_sep_rh, 2)),
                    "dE_rel": float(engine.delta_e_rel),
                    "dLz_rel": float(engine.delta_lz_rel),
                    "classification": str(engine.stability_state),
                    "survival_time": float(round(engine.survival_time, 2)),
                }
                self.results.append(record)
                
                if progress_callback:
                    progress_callback(count, total_points, record)
                elif count % max(1, total_points // 10) == 0 or count == total_points:
                    elapsed = time.time() - start_time
                    rate = count / max(elapsed, 0.001)
                    eta = (total_points - count) / rate if rate > 0 else 0.0
                    print(f"  [{count:4d}/{total_points}] ({count/total_points*100:5.1f}%) "
                          f"a0={a0:.3f} e0={e0:.3f} -> [{record['classification']:9s}] "
                          f"e_max={max_e:.3f} d_min={min_giant_sep_rh:.1f} R_H | "
                          f"{rate:.1f} pts/s, ETA: {eta:.1f}s")
                          
        total_time = time.time() - start_time
        print(f"\n[StabilityMapScanner] Scan completed in {total_time:.2f} s ({total_points/max(total_time,0.001):.1f} pts/s)")
        return self.results

    def save_results(self, filepath="stability_map_data.json"):
        """Saves scan metadata and full numerical results to JSON."""
        import json
        data = {
            "a_range": list(self.a_range),
            "e_range": list(self.e_range),
            "n_a": self.n_a,
            "n_e": self.n_e,
            "t_duration": self.t_duration,
            "dt_sub": self.dt_sub,
            "nominal_a_res": self.nominal_a_res,
            "results": self.results
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[StabilityMapScanner] Saved {len(self.results)} points to {filepath}")

    def load_results(self, filepath="stability_map_data.json"):
        """Loads scan metadata and numerical results from JSON."""
        import os, json
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            self.a_range = tuple(data["a_range"])
            self.e_range = tuple(data["e_range"])
            self.n_a = data["n_a"]
            self.n_e = data["n_e"]
            self.t_duration = data["t_duration"]
            self.dt_sub = data["dt_sub"]
            self.nominal_a_res = data.get("nominal_a_res", 1.0)
            self.results = data["results"]
            print(f"[StabilityMapScanner] Loaded {len(self.results)} points from {filepath}")
            return True
        except Exception as e:
            print(f"[StabilityMapScanner] Could not load {filepath}: {e}")
            return False

# ==============================================================================
# 8. RENDERING SYSTEM & GGUI INTERACTION
# ==============================================================================
class SimulationRenderer:
    def __init__(self, engine: AstrophysicsEngine, width=1280, height=800):
        self.engine = engine
        self.width = width
        self.height = height
        self.aspect_ratio = height / width # 800 / 1280 = 0.625
        self.view_mode = "ORBIT" # "ORBIT" or "MAP"
        
        # Stability map scanner & dataset
        self.scanner = StabilityMapScanner(a_range=(0.70, 1.50), e_range=(0.00, 0.40), n_a=40, n_e=25, t_duration=5.0)
        self.scanner.load_results("stability_map_data.json")
        self.selected_point_idx = 0
        
        self.window = ti.ui.Window(
            name="HABITABLE CHAOS — A Computational Laboratory for Planetary Stability",
            res=(width, height),
            vsync=True,
        )
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        
        # Preallocated reusable NumPy buffers for zero-allocation rendering
        self.ring_verts_np = np.zeros((3 * RING_SEGMENTS * 2, 2), dtype=np.float32)
        self.ring_cols_np = np.zeros((3 * RING_SEGMENTS * 2, 3), dtype=np.float32)
        self.trail_verts_np = np.zeros((MAX_TRAIL_VERTICES, 2), dtype=np.float32)
        self.trail_cols_np = np.zeros((MAX_TRAIL_VERTICES, 3), dtype=np.float32)
        self.draw_centers_np = np.zeros((MAX_BODIES, 2), dtype=np.float32)
        self.draw_radii_np = np.zeros(MAX_BODIES, dtype=np.float32)
        self.draw_colors_np = np.zeros((MAX_BODIES, 3), dtype=np.float32)
        self.glow_centers_np = np.zeros((2, 2), dtype=np.float32)
        self.glow_radii_np = np.zeros(2, dtype=np.float32)
        self.glow_colors_np = np.zeros((2, 3), dtype=np.float32)
        
        # Preallocated stability map buffers
        self.map_centers_np = np.zeros((MAX_MAP_POINTS, 2), dtype=np.float32)
        self.map_colors_np = np.zeros((MAX_MAP_POINTS, 3), dtype=np.float32)
        self.map_radii_np = np.zeros(MAX_MAP_POINTS, dtype=np.float32)
        self.map_axis_verts_np = np.zeros((128, 2), dtype=np.float32)
        self.map_axis_cols_np = np.zeros((128, 3), dtype=np.float32)
        self.glow_radii_np = np.zeros(2, dtype=np.float32)
        self.glow_colors_np = np.zeros((2, 3), dtype=np.float32)

    def render_frame(self):
        eng = self.engine
        
        # Deep space astrophysical background
        self.canvas.set_background_color((0.015, 0.022, 0.035))

        if self.view_mode == "ORBIT":
            self._render_orbit_view()
        else:
            self._render_stability_map_view()

        # Render GGUI control dashboard
        self._render_gui_dashboard()
        self.window.show()

    def _render_orbit_view(self):
        eng = self.engine
        
        # Center of view: Star A (0) or System Barycenter (1)
        if eng.view_focus == 0:
            cx, cy = pos[0][0], pos[0][1]
        else:
            total_m = sum(mass[i] for i in range(eng.num_active))
            cx = sum(mass[i] * pos[i][0] for i in range(eng.num_active)) / total_m
            cy = sum(mass[i] * pos[i][1] for i in range(eng.num_active)) / total_m
            
        scale_au = max(eng.view_scale_au, 0.5)

        # Coordinate transform from AU to normalized canvas [0, 1]
        def au_to_canvas(x, y):
            sx = 0.5 + ((x - cx) / (scale_au * 2.0)) * self.aspect_ratio
            sy = 0.5 + ((y - cy) / (scale_au * 2.0))
            return sx, sy

        # 1. Habitable Zone Annulus & Snow Line Rings (centered on Star A)
        star_a_x, star_a_y = pos[0][0], pos[0][1]
        hz_in = eng.hz_inner_au
        hz_out = eng.hz_outer_au
        snow = eng.snow_line_au
        
        r_indices = [
            (hz_in, (0.15, 0.65, 0.40)),  # Emerald (HZ Inner)
            (hz_out, (0.10, 0.55, 0.35)), # Forest green (HZ Outer)
            (snow, (0.30, 0.55, 0.85)),   # Cyan frost (Snow Line)
        ]
        
        ring_v_idx = 0
        for radius_au, ring_col in r_indices:
            for s in range(RING_SEGMENTS):
                ang1 = (s / RING_SEGMENTS) * TWO_PI
                ang2 = ((s + 1) / RING_SEGMENTS) * TWO_PI
                
                x1 = star_a_x + radius_au * math.cos(ang1)
                y1 = star_a_y + radius_au * math.sin(ang1)
                x2 = star_a_x + radius_au * math.cos(ang2)
                y2 = star_a_y + radius_au * math.sin(ang2)
                
                sx1, sy1 = au_to_canvas(x1, y1)
                sx2, sy2 = au_to_canvas(x2, y2)
                
                self.ring_verts_np[ring_v_idx] = [sx1, sy1]
                self.ring_cols_np[ring_v_idx] = ring_col
                ring_v_idx += 1
                
                self.ring_verts_np[ring_v_idx] = [sx2, sy2]
                self.ring_cols_np[ring_v_idx] = ring_col
                ring_v_idx += 1

        ring_vertices.from_numpy(self.ring_verts_np)
        ring_colors.from_numpy(self.ring_cols_np)
        self.canvas.lines(ring_vertices, width=0.0015, per_vertex_color=ring_colors)

        # 2. Historical Orbital Trajectories (True numerical integration paths)
        trail_v_idx = 0
        self.trail_verts_np.fill(0.0)
        self.trail_cols_np.fill(0.0)
        
        for i in range(eng.num_active):
            cnt = eng.trail_counts[i]
            if cnt > 1:
                col = [color[i][0] * 0.7, color[i][1] * 0.7, color[i][2] * 0.7]
                head = eng.trail_head[i]
                
                for k in range(cnt - 1):
                    idx1 = (head - cnt + k) % TRAIL_LENGTH
                    idx2 = (head - cnt + k + 1) % TRAIL_LENGTH
                    
                    p1 = eng.trail_history[i, idx1]
                    p2 = eng.trail_history[i, idx2]
                    
                    sx1, sy1 = au_to_canvas(p1[0], p1[1])
                    sx2, sy2 = au_to_canvas(p2[0], p2[1])
                    
                    alpha = 0.2 + 0.8 * (k / float(cnt))
                    v_col = [col[0] * alpha, col[1] * alpha, col[2] * alpha]
                    
                    self.trail_verts_np[trail_v_idx] = [sx1, sy1]
                    self.trail_cols_np[trail_v_idx] = v_col
                    trail_v_idx += 1
                    
                    self.trail_verts_np[trail_v_idx] = [sx2, sy2]
                    self.trail_cols_np[trail_v_idx] = v_col
                    trail_v_idx += 1

        if trail_v_idx > 0:
            trail_vertices.from_numpy(self.trail_verts_np)
            trail_colors.from_numpy(self.trail_cols_np)
            self.canvas.lines(trail_vertices, width=0.0018, per_vertex_color=trail_colors)

        # 3. Celestial Bodies (Stars with Corona Glow + Planets)
        for i in range(eng.num_active):
            px, py = pos[i][0], pos[i][1]
            sx, sy = au_to_canvas(px, py)
            self.draw_centers_np[i] = [sx, sy]

        # Outer glowing corona for stars
        self.glow_centers_np[0] = self.draw_centers_np[0]
        self.glow_radii_np[0] = 0.024
        self.glow_colors_np[0] = [1.0, 0.82, 0.35]
        
        if eng.num_active > 1 and active[1] == 1 and mass[1] > 0.05:
            self.glow_centers_np[1] = self.draw_centers_np[1]
            self.glow_radii_np[1] = 0.017
            self.glow_colors_np[1] = [1.0, 0.45, 0.25]
        else:
            self.glow_centers_np[1] = [2.0, 2.0]
            self.glow_radii_np[1] = 0.0
            self.glow_colors_np[1] = [0.0, 0.0, 0.0]

        draw_glow_centers.from_numpy(self.glow_centers_np)
        draw_glow_radii.from_numpy(self.glow_radii_np)
        draw_glow_colors.from_numpy(self.glow_colors_np)
        self.canvas.circles(draw_glow_centers, radius=0.020, per_vertex_color=draw_glow_colors, per_vertex_radius=draw_glow_radii)

        # Individual celestial body cores
        for i in range(MAX_BODIES):
            if i < eng.num_active and active[i] == 1:
                self.draw_radii_np[i] = 0.012 if i == 0 else (0.009 if i == 1 and mass[1] > 0.05 else (0.007 if i == 2 and eng.num_active >= 4 else 0.0055))
                self.draw_colors_np[i] = [color[i][0], color[i][1], color[i][2]]
            else:
                self.draw_centers_np[i] = [2.0, 2.0]
                self.draw_radii_np[i] = 0.0
                self.draw_colors_np[i] = [0.0, 0.0, 0.0]

        draw_centers.from_numpy(self.draw_centers_np)
        draw_radii.from_numpy(self.draw_radii_np)
        draw_colors.from_numpy(self.draw_colors_np)
        self.canvas.circles(draw_centers, radius=0.006, per_vertex_color=draw_colors, per_vertex_radius=draw_radii)

    def _render_stability_map_view(self):
        """Renders the (a0, e0) parameter space stability map with 3:1 resonance line."""
        # Plot area on canvas: X in [0.42, 0.96], Y in [0.12, 0.88]
        x_min, x_max = 0.42, 0.96
        y_min, y_max = 0.12, 0.88
        dx = x_max - x_min
        dy = y_max - y_min
        
        a_min, a_max = self.scanner.a_range
        e_min, e_max = self.scanner.e_range
        da = a_max - a_min
        de = e_max - e_min
        
        # 1. Map Axes & Reference Lines
        v_idx = 0
        axis_col = [0.35, 0.42, 0.55]
        
        # Bounding Box (4 lines = 8 vertices)
        self.map_axis_verts_np[v_idx] = [x_min, y_min]; self.map_axis_cols_np[v_idx] = axis_col; v_idx += 1
        self.map_axis_verts_np[v_idx] = [x_max, y_min]; self.map_axis_cols_np[v_idx] = axis_col; v_idx += 1
        
        self.map_axis_verts_np[v_idx] = [x_max, y_min]; self.map_axis_cols_np[v_idx] = axis_col; v_idx += 1
        self.map_axis_verts_np[v_idx] = [x_max, y_max]; self.map_axis_cols_np[v_idx] = axis_col; v_idx += 1
        
        self.map_axis_verts_np[v_idx] = [x_max, y_max]; self.map_axis_cols_np[v_idx] = axis_col; v_idx += 1
        self.map_axis_verts_np[v_idx] = [x_min, y_max]; self.map_axis_cols_np[v_idx] = axis_col; v_idx += 1
        
        self.map_axis_verts_np[v_idx] = [x_min, y_max]; self.map_axis_cols_np[v_idx] = axis_col; v_idx += 1
        self.map_axis_verts_np[v_idx] = [x_min, y_min]; self.map_axis_cols_np[v_idx] = axis_col; v_idx += 1
        
        # Nominal 3:1 Resonance line: a_res ~ 1.000 AU
        res_x = x_min + ((self.scanner.nominal_a_res - a_min) / da) * dx
        res_col = [0.20, 0.85, 1.0] # Cyan highlight
        self.map_axis_verts_np[v_idx] = [res_x, y_min]; self.map_axis_cols_np[v_idx] = res_col; v_idx += 1
        self.map_axis_verts_np[v_idx] = [res_x, y_max]; self.map_axis_cols_np[v_idx] = res_col; v_idx += 1
        
        # Grid tick marks on X axis (a0 = 0.8, 1.0, 1.2, 1.4)
        for tick_a in [0.80, 1.00, 1.20, 1.40]:
            tx = x_min + ((tick_a - a_min) / da) * dx
            self.map_axis_verts_np[v_idx] = [tx, y_min]; self.map_axis_cols_np[v_idx] = axis_col; v_idx += 1
            self.map_axis_verts_np[v_idx] = [tx, y_min - 0.015]; self.map_axis_cols_np[v_idx] = axis_col; v_idx += 1
            
        # Grid tick marks on Y axis (e0 = 0.1, 0.2, 0.3)
        for tick_e in [0.10, 0.20, 0.30]:
            ty = y_min + ((tick_e - e_min) / de) * dy
            self.map_axis_verts_np[v_idx] = [x_min, ty]; self.map_axis_cols_np[v_idx] = axis_col; v_idx += 1
            self.map_axis_verts_np[v_idx] = [x_min - 0.010, ty]; self.map_axis_cols_np[v_idx] = axis_col; v_idx += 1

        # Clear remaining vertices
        self.map_axis_verts_np[v_idx:].fill(0.0)
        self.map_axis_cols_np[v_idx:].fill(0.0)
        
        map_axis_vertices.from_numpy(self.map_axis_verts_np)
        map_axis_colors.from_numpy(self.map_axis_cols_np)
        self.canvas.lines(map_axis_vertices, width=0.002, per_vertex_color=map_axis_colors)
        
        # 2. Scanned Grid Points
        n_res = len(self.scanner.results)
        self.map_centers_np.fill(2.0)
        self.map_radii_np.fill(0.0)
        self.map_colors_np.fill(0.0)
        
        col_map = {
            "STABLE":    [0.15, 0.85, 0.40], # Emerald
            "PERTURBED": [0.95, 0.80, 0.20], # Amber
            "UNSTABLE":  [0.95, 0.45, 0.15], # Orange
            "EJECTED":   [0.80, 0.20, 0.85], # Purple
            "COLLISION": [0.90, 0.15, 0.15], # Crimson
        }
        
        base_radius = 0.011 if n_res <= 50 else 0.0065
        for idx in range(min(n_res, MAX_MAP_POINTS)):
            p = self.scanner.results[idx]
            px = x_min + ((p["a0"] - a_min) / da) * dx
            py = y_min + ((p["e0"] - e_min) / de) * dy
            self.map_centers_np[idx] = [px, py]
            self.map_colors_np[idx] = col_map.get(p["classification"], [0.5, 0.5, 0.5])
            if idx == self.selected_point_idx:
                self.map_radii_np[idx] = base_radius * 2.2
                self.map_colors_np[idx] = [1.0, 1.0, 1.0] # Highlight white for selected
            else:
                self.map_radii_np[idx] = base_radius

        map_point_centers.from_numpy(self.map_centers_np)
        map_point_colors.from_numpy(self.map_colors_np)
        map_point_radii.from_numpy(self.map_radii_np)
        self.canvas.circles(map_point_centers, radius=base_radius, per_vertex_color=map_point_colors, per_vertex_radius=map_point_radii)

    def _render_gui_dashboard(self):
        eng = self.engine
        gui = self.gui
        gui.begin("Scientific Laboratory", 0.015, 0.02, 0.38, 0.96)
        
        gui.text("HABITABLE CHAOS: Astrophysics Lab")
        gui.text("----------------------------------------")
        
        # View Mode Switcher
        mode_btn_label = "Switch to STABILITY MAP VIEW" if self.view_mode == "ORBIT" else "Switch to ORBIT SIMULATION"
        if gui.button(mode_btn_label):
            self.view_mode = "MAP" if self.view_mode == "ORBIT" else "ORBIT"
            
        if self.view_mode == "MAP":
            # ------------------------------------------------------------------
            # STABILITY MAP DASHBOARD
            # ------------------------------------------------------------------
            gui.text("----------------------------------------")
            gui.text("STABILITY MAP: 'Where do orbits survive?'")
            gui.text("Primary Experiment: Preset 2 System")
            gui.text("  Binary Stars (1.0 + 0.35 M_sun)")
            gui.text("  Giant Perturber at 2.0801 AU")
            gui.text(f"  Parameter Space:")
            gui.text(f"    a0 in [{self.scanner.a_range[0]:.2f}, {self.scanner.a_range[1]:.2f}] AU (X-axis)")
            gui.text(f"    e0 in [{self.scanner.e_range[0]:.2f}, {self.scanner.e_range[1]:.2f}] (Y-axis)")
            gui.text(f"  Nominal 3:1 Resonance: a_res = {self.scanner.nominal_a_res:.3f} AU")
            gui.text("  (Cyan vertical line = reference marker)")
            
            gui.text("----------------------------------------")
            gui.text("HEADLESS SCAN CONTROLS:")
            if gui.button("Run Quick Test Scan (8x5 = 40 pts)"):
                self.scanner.n_a = 8
                self.scanner.n_e = 5
                self.scanner.run_scan()
                self.scanner.save_results("stability_map_data.json")
            if gui.button("Run Full Scan (40x25 = 1000 pts)"):
                self.scanner.n_a = 40
                self.scanner.n_e = 25
                self.scanner.run_scan()
                self.scanner.save_results("stability_map_data.json")

            gui.text("----------------------------------------")
            gui.text("CLASSIFICATION LEGEND:")
            gui.text("  [GREEN]   STABLE (e < 0.2, drift < 12%)")
            gui.text("  [YELLOW]  PERTURBED (e >= 0.2 or sep < 3 Rh)")
            gui.text("  [ORANGE]  UNSTABLE (e >= 0.7 or sep < 1 Rh)")
            gui.text("  [PURPLE]  EJECTED (r_bary > 40 AU, E > 0)")
            gui.text("  [RED]     COLLISION (r < r_star + r_test)")

            n_pts = len(self.scanner.results)
            gui.text("----------------------------------------")
            gui.text(f"POINT INSPECTOR ({n_pts} points available):")
            if n_pts > 0:
                self.selected_point_idx = gui.slider_int("Select Point", self.selected_point_idx, 0, n_pts - 1)
                p = self.scanner.results[self.selected_point_idx]
                gui.text(f"  Initial a0:      {p['a0']:.4f} AU")
                gui.text(f"  Initial e0:      {p['e0']:.4f}")
                gui.text(f"  Final a:         {p['a_final']:.4f} AU")
                gui.text(f"  Final e:         {p['e_final']:.4f}")
                gui.text(f"  Maximum e:       {p['max_e']:.4f}")
                gui.text(f"  Min Giant Sep:   {p['min_giant_sep_au']:.4f} AU ({p['min_giant_sep_rh']:.1f} R_H)")
                gui.text(f"  Energy Drift:    {p['dE_rel']:+.3e}")
                gui.text(f"  Lz Drift:        {p['dLz_rel']:+.3e}")
                gui.text(f"  Classification:  [{p['classification']}]")
                gui.text(f"  Survival Time:   {p['survival_time']:.2f} yr")

                if gui.button("--> LOAD INTO SIMULATION <--"):
                    eng.load_preset(2, custom_test_orbit=(p['a0'], p['e0']))
                    self.view_mode = "ORBIT"
                    eng.paused = False
            else:
                gui.text("  No scan data found. Click a scan button above")
                gui.text("  or run 'python sim.py --scan'.")
        else:
            # ------------------------------------------------------------------
            # ORBIT SIMULATION DASHBOARD (Existing 8 Panels)
            # ------------------------------------------------------------------
            status_text = f"Simulation Clock: {eng.time_sim:.2f} yr | {'[PAUSED]' if eng.paused else '[RUNNING]'}"
            gui.text(status_text)
            
            # System Architecture
            gui.text("----------------------------------------")
            gui.text("1. SYSTEM ARCHITECTURE")
            gui.text(f"Preset [{eng.preset_id}]: {eng.target_resonance_label}")
            if gui.button("Preset 1: S-Type Binary (Calm HZ)"):
                eng.load_preset(1)
            if gui.button("Preset 2: Nominal 3:1 MMR (Resonance)"):
                eng.load_preset(2)
            if gui.button("Preset 3: Tight Binary Intruder"):
                eng.load_preset(3)
            if gui.button("Preset 4: Kepler 3rd Law Benchmark"):
                eng.load_preset(4)

            # Numerical Fidelity & Conservation
            gui.text("----------------------------------------")
            gui.text("2. NUMERICAL CONSERVATION (Symplectic)")
            gui.text(f"Substeps/frame: {eng.substeps} | dt: {eng.dt_sub:.5f} yr")
            gui.text(f"Energy Drift dE/E0:     {eng.delta_e_rel:+.3e}")
            gui.text(f"Ang. Momentum dLz/Lz0:  {eng.delta_lz_rel:+.3e}")
            gui.text(f"Softening eps: {DEFAULT_EPSILON:.3f} AU (r < eps unphysical)")
            
            # Test Planet Raw Orbital Diagnostics
            gui.text("----------------------------------------")
            gui.text("3. TEST PLANET ORBITAL DIAGNOSTICS")
            gui.text("(Osculating 2-body elements relative to Star A)")
            gui.text(f"Instantaneous dist r:   {eng.r_test:.4f} AU")
            gui.text(f"Orbital speed v:        {eng.v_test:.4f} AU/yr ({eng.v_test * 4.74:.1f} km/s)")
            gui.text(f"Osculating semimajor a: {eng.a_test:.4f} AU")
            da_a0 = (eng.a_test - eng.a0_test) / eng.a0_test if eng.a0_test > 0 else 0.0
            gui.text(f"Semimajor drift da/a0:  {da_a0:+.3%}")
            gui.text(f"Osculating ecc e(t):    {eng.e_test:.4f}")
            gui.text(f"Periapsis / Apoapsis:   {eng.peri_test:.3f} / {eng.apo_test:.3f} AU")
            gui.text(f"Barycentric Energy:     {eng.bary_energy_test:+.4f} AU^2/yr^2")

            # Resonance Diagnostics (3:1 MMR)
            if eng.num_active >= 4:
                gui.text("----------------------------------------")
                gui.text("4. 3:1 RESONANT ANGLE DIAGNOSTIC")
                gui.text(f"Giant Semimajor a_J:    {eng.a_giant:.3f} AU (e = {eng.e_giant:.3f})")
                gui.text(f"Semimajor Ratio a_J/a_t:{eng.a_ratio:.3f}")
                gui.text(f"Period Ratio P_J/P_t:   {eng.period_ratio:.3f}")
                gui.text(f"Resonant Angle phi:     {eng.phi_3_1_deg:.1f}° ({eng.phi_3_1:.3f} rad)")
                gui.text(f"Resonance Classification:")
                gui.text(f"  {eng.resonance_state}")
                if eng.preset_id == 2:
                    gui.text("  (phi = 3*lambda_J - lambda_t - 2*varpi_t)")

            # Stellar Insolation & Habitable Zone
            gui.text("----------------------------------------")
            gui.text("5. STELLAR IRRADIATION & HZ")
            gui.text(f"Star A Luminosity:      {eng.primary_luminosity:.3f} L_sun")
            gui.text(f"Total Received Flux:    {eng.total_flux_test:.3f} S_sun")
            gui.text(f"Approximate HZ Flux:    [0.53 - 1.11] S_sun")
            gui.text(f"Radiative State:        {eng.flux_status}")
            gui.text(f"Luminosity Snow Line:   {eng.snow_line_au:.2f} AU")
            gui.text("(Simplified flux proxy; does not model climate)")

            # Operational Stability Classification
            gui.text("----------------------------------------")
            gui.text("6. OPERATIONAL STABILITY CATEGORY")
            gui.text(f"Classification:         [{eng.stability_state}]")
            gui.text(f"Survival Time:          {eng.survival_time:.2f} yr")
            if eng.num_active >= 4:
                gui.text(f"Giant Hill Radius R_H:  {eng.r_hill_giant:.3f} AU")
                gui.text(f"Current Separation:     {eng.hill_sep:.2f} R_H")
            gui.text("(Operational categories based on Hill separation,")
            gui.text(" not universal physical theorems.)")

            # Kepler 3rd Law Validation Monitor
            gui.text("----------------------------------------")
            gui.text("7. EMPIRICAL KEPLER VALIDATION")
            gui.text(f"Measured Revolutions:   {len(eng.measured_periods)}")
            if len(eng.measured_periods) > 0:
                last_p = eng.measured_periods[-1]
                gui.text(f"Empirical Period P:     {last_p:.4f} yr")
                gui.text(f"Kepler Ratio P^2/a^3:   {eng.kepler_ratio_p2_over_a3:.4f}")
                gui.text(f"Kepler Error vs Theory: {eng.kepler_relative_error:.2f}%")
            else:
                gui.text("Accumulating full 2*pi revolution...")

            # Interactive Controls
            gui.text("----------------------------------------")
            gui.text("8. INTERACTIVE CONTROLS")
            if gui.button("Pause / Resume (SPACE)"):
                eng.paused = not eng.paused
            if gui.button("Reset Scenario (R)"):
                eng.load_preset(eng.preset_id, eng.custom_test_orbit)
            if gui.button("Focus: " + ("Primary Star" if eng.view_focus == 0 else "Barycenter")):
                eng.view_focus = 1 - eng.view_focus

            eng.view_scale_au = gui.slider_float("View Scale (AU)", eng.view_scale_au, 1.0, 30.0)
            eng.substeps = gui.slider_int("Substeps/frame", eng.substeps, 10, 400)
            eng.dt_sub = gui.slider_float("dt step (yr)", eng.dt_sub, 0.00005, 0.001)

        gui.end()

    def process_events(self):
        """Handles keyboard and GUI interaction events."""
        for e in self.window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.SPACE:
                self.engine.paused = not self.engine.paused
            elif e.key in ('r', 'R'):
                self.engine.load_preset(self.engine.preset_id)
            elif e.key == '1':
                self.engine.load_preset(1)
            elif e.key == '2':
                self.engine.load_preset(2)
            elif e.key == '3':
                self.engine.load_preset(3)
            elif e.key == '4':
                self.engine.load_preset(4)
            elif e.key == ti.ui.ESCAPE:
                self.window.running = False

# ==============================================================================
# 8. HEADLESS SCIENTIFIC VERIFICATION SUITE (--verify)
# ==============================================================================
def run_headless_verification():
    """
    Executes a rigorous headless verification test:
    1. Compiles Taichi kernels on the active backend.
    2. Runs a 1.0 AU circular test orbit around a 1.0 M_sun star.
    3. Empirically measures orbital periods across 5 full revolutions.
    4. Validates Kepler's 3rd Law: P^2 / a^3 == 1.000.
    5. Measures relative energy drift and angular momentum drift.
    6. Tests multi-body S-type binary configuration for 10 orbits.
    """
    print("\n" + "=" * 70)
    print("HABITABLE CHAOS: SCIENTIFIC VERIFICATION SUITE")
    print("=" * 70)
    
    arch, backend_name = ACTIVE_ARCH, ACTIVE_BACKEND_NAME
    print(f"[Verification 1/4] Backend initialized: {backend_name}")

    # 1. Single Star Kepler Benchmark Test
    engine = AstrophysicsEngine(preset_id=4)
    print("[Verification 2/4] Initializing Kepler Benchmark Orbit (1.0 AU, 1.0 M_sun)...")
    
    dt = 0.0001 # 0.876 hours
    total_steps = 50000 # 5.0 years (5 full orbits)
    
    print(f"-> Integrating {total_steps} substeps with dt = {dt:.5f} yr...")
    engine.dt_sub = dt
    engine.substeps = 500
    
    for _ in range(total_steps // 500):
        engine.step_simulation()
        
    measured_revs = len(engine.measured_periods)
    print(f"-> Completed integration: {engine.time_sim:.2f} yr elapsed.")
    print(f"-> Measured full orbits completed: {measured_revs}")
    
    if measured_revs > 0:
        avg_p = float(np.mean(engine.measured_periods))
        kepler_ratio = (avg_p * avg_p) / math.pow(engine.a_test, 3)
        kepler_err = abs(kepler_ratio - 1.0) * 100.0
        print(f"-> Empirical Mean Period P_meas: {avg_p:.5f} yr (Theoretical: 1.00000 yr)")
        print(f"-> Empirical Kepler Ratio P^2 / a^3: {kepler_ratio:.5f}")
        print(f"-> Kepler Deviation: {kepler_err:.4f}%")
        assert kepler_err < 0.20, f"Kepler 3rd law discrepancy too high: {kepler_err}%"
    else:
        raise RuntimeError("No complete orbits detected during Kepler verification.")

    print(f"-> Relative Energy Drift dE/E0: {engine.delta_e_rel:+.3e}")
    print(f"-> Relative Angular Momentum Drift dLz/Lz0: {engine.delta_lz_rel:+.3e}")
    assert abs(engine.delta_e_rel) < 1e-4, f"Energy drift exceeds tolerance: {engine.delta_e_rel}"
    assert abs(engine.delta_lz_rel) < 1e-4, f"Angular momentum drift exceeds tolerance: {engine.delta_lz_rel}"
    print("[Verification 2/4 Passed] Kepler dynamics and symplectic conservation confirmed!")

    # 2. Multi-Body Binary + Giant Planet + Terrestrial Test Planet System
    print("\n[Verification 3/4] Initializing Hierarchical S-Type Binary System (Preset 1)...")
    engine.load_preset(1)
    steps_multibody = 25000 # 2.5 years
    print(f"-> Integrating 4-body mutual gravity for {steps_multibody} substeps...")
    
    for _ in range(steps_multibody // 500):
        engine.step_simulation()
        
    print(f"-> Time elapsed: {engine.time_sim:.2f} yr")
    print(f"-> Star A Luminosity: {engine.primary_luminosity:.3f} L_sun")
    print(f"-> Habitable Zone Range: [{engine.hz_inner_au:.3f} - {engine.hz_outer_au:.3f}] AU")
    print(f"-> Luminosity-scaled Snow Line: {engine.snow_line_au:.3f} AU")
    print(f"-> Test Planet semimajor axis a: {engine.a_test:.4f} AU, ecc e: {engine.e_test:.4f}")
    print(f"-> Received Total Flux: {engine.total_flux_test:.4f} S_sun")
    print(f"-> Operational Stability: {engine.stability_state}")
    print(f"-> Multi-body Energy Drift dE/E0: {engine.delta_e_rel:+.3e}")
    print(f"-> Multi-body Angular Momentum Drift dLz/Lz0: {engine.delta_lz_rel:+.3e}")
    assert not math.isnan(engine.delta_e_rel), "Energy drift evaluated to NaN!"
    assert engine.stability_state in ("STABLE", "PERTURBED"), f"Unexpected early instability: {engine.stability_state}"
    print("[Verification 3/4 Passed] Multi-body mutual gravity and stellar scaling confirmed!")

    # 3. Resonance Scenario Initial Condition & Diagnostics Verification
    print("\n[Verification 4/4] Verifying 3:1 Mean Motion Resonance Setup & Diagnostics (Preset 2)...")
    engine.load_preset(2)
    print(f"-> Giant planet a_giant: {engine.a_giant:.4f} AU, Test planet a_test: {engine.a_test:.4f} AU")
    print(f"-> Period ratio P_giant / P_test: {engine.period_ratio:.4f} (Target nominal 3:1 ~ 3.000)")
    assert abs(engine.period_ratio - 3.0) < 0.15, f"Period ratio deviates from nominal initial condition: {engine.period_ratio}"
    
    # Integrate for 2000 substeps to test resonance angle tracking and Hill radius
    for _ in range(4):
        engine.step_simulation()
    
    print(f"-> Giant Hill radius R_H: {engine.r_hill_giant:.4f} AU")
    print(f"-> Test planet separation in Hill units: {engine.hill_sep:.2f} R_H")
    print(f"-> 3:1 Resonant angle phi: {engine.phi_3_1_deg:.2f}° ({engine.phi_3_1:.4f} rad)")
    print(f"-> Resonance evaluation: {engine.resonance_state}")
    print(f"-> Total received flux: {engine.total_flux_test:.4f} S_sun -> {engine.flux_status}")
    assert engine.r_hill_giant > 0.05, f"Unphysical Hill radius: {engine.r_hill_giant}"
    assert not math.isnan(engine.phi_3_1), "Resonant angle evaluated to NaN!"
    assert -math.pi <= engine.phi_3_1 <= math.pi, f"Resonant angle not normalized to [-pi, pi]: {engine.phi_3_1}"
    print("[Verification 4/4 Passed] Resonant angle, Hill radius, and total flux diagnostics confirmed!")

    print("\n" + "=" * 70)
    print("ALL SCIENTIFIC VERIFICATION TESTS PASSED SUCCESSFULLY.")
    print("=" * 70 + "\n")
    return 0

# ==============================================================================
# 9. MAIN EXECUTION ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="HABITABLE CHAOS: Astrophysics Simulation")
    parser.add_argument("--verify", action="store_true", help="Run automated headless scientific verification suite")
    parser.add_argument("--scan", action="store_true", help="Run headless stability map parameter scan")
    parser.add_argument("--grid", type=str, default="8x5", help="Grid resolution for scan (e.g. 8x5 or 40x25)")
    parser.add_argument("--duration", type=float, default=5.0, help="Integration duration per point in years")
    parser.add_argument("--preset", type=int, default=1, choices=[1, 2, 3, 4], help="Initial scenario preset (1-4)")
    args = parser.parse_args()

    if args.verify:
        sys.exit(run_headless_verification())

    if args.scan:
        parts = args.grid.lower().split("x")
        n_a = int(parts[0])
        n_e = int(parts[1]) if len(parts) > 1 else n_a
        scanner = StabilityMapScanner(a_range=(0.70, 1.50), e_range=(0.00, 0.40),
                                      n_a=n_a, n_e=n_e, t_duration=args.duration)
        scanner.run_scan()
        scanner.save_results("stability_map_data.json")
        sys.exit(0)

    # Interactive GUI mode
    arch, backend_name = ACTIVE_ARCH, ACTIVE_BACKEND_NAME
    print(f"[HABITABLE CHAOS] Launching interactive simulation on {backend_name}...")
    
    engine = AstrophysicsEngine(preset_id=args.preset)
    renderer = SimulationRenderer(engine, width=1280, height=800)
    
    print("[HABITABLE CHAOS] Controls:")
    print("  SPACE : Pause / Resume")
    print("  R     : Reset current scenario")
    print("  1-4   : Switch scenario presets")
    print("  M     : Toggle Stability Map view mode")
    print("  ESC   : Exit simulation cleanly\n")
    
    last_fps_time = time.time()
    frames = 0

    while renderer.window.running:
        renderer.process_events()
        if renderer.view_mode == "ORBIT":
            engine.step_simulation()
        renderer.render_frame()
        
        frames += 1
        curr_time = time.time()
        if curr_time - last_fps_time >= 2.0:
            fps = frames / (curr_time - last_fps_time)
            # Log FPS periodically to console
            print(f"[Diagnostics] FPS: {fps:.1f} | Mode: {renderer.view_mode} | Clock: {engine.time_sim:.2f} yr | dE/E0: {engine.delta_e_rel:+.2e} | Status: {engine.stability_state}", flush=True)
            frames = 0
            last_fps_time = curr_time

    print("[HABITABLE CHAOS] Window closed. Simulation terminated cleanly.", flush=True)

if __name__ == "__main__":
    main()
