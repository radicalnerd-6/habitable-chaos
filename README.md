# Habitable Chaos — A Computational Laboratory for Planetary Stability

Habitable Chaos is a GPU-accelerated N-body computational astrophysics laboratory that investigates where planetary orbits survive under complex gravitational perturbations, rather than displaying precomputed or idealized orbital paths. Built with Python and Taichi, it couples symplectic numerical integration with real-time dynamical diagnostics, stellar radiative flux calculations, and a systematic 1,000-configuration computational stability map.

---

## 1. What is Habitable Chaos?

The central scientific question of the project is:
> **"Where do planetary orbits survive when gravity, eccentricity, and resonances compete?"**

Rather than relying on central-force approximations or isolated two-body Keplerian ellipses, Habitable Chaos integrates the full equations of mutual Newtonian motion for multi-body systems. The laboratory combines:
- **Mutual Newtonian N-body gravity**: Every celestial body exerts gravitational forces on every other body.
- **Symplectic Velocity-Verlet integration**: A second-order Hamiltonian-preserving integrator with bounded secular energy drift.
- **Binary-star architectures**: S-type hierarchical stellar configurations (primary star with an exterior stellar companion).
- **Giant-planet perturbations**: Massive perturbers capable of driving eccentricity excitation, close encounters, and orbital scattering.
- **Real-time orbital diagnostics**: Osculating orbital elements, Hill-sphere separations, 3:1 Mean Motion Resonance arguments, and empirical Kepler third-law validation.
- **Computational Stability Map**: A systematic headless parameter-space scan over initial semimajor axis and eccentricity ($a_0, e_0$).

---

## 2. The Physics

### Gravitational Equations of Motion
Mutual gravitational acceleration on body $i$ from all other active bodies $j$ is computed via:

$$\mathbf{a}_i = G \sum_{j \ne i} \frac{m_j (\mathbf{r}_j - \mathbf{r}_i)}{\left(|\mathbf{r}_j - \mathbf{r}_i|^2 + \epsilon^2\right)^{3/2}}$$

### Numerical Integration (Velocity-Verlet / Kick-Drift-Kick)
To preserve symplectic geometric structure and avoid artificial secular energy dissipation, the system advances via:

$$\mathbf{v}_i\left(t + \frac{\Delta t}{2}\right) = \mathbf{v}_i(t) + \frac{1}{2} \Delta t \, \mathbf{a}_i(t)$$

$$\mathbf{r}_i(t + \Delta t) = \mathbf{r}_i(t) + \Delta t \, \mathbf{v}_i\left(t + \frac{\Delta t}{2}\right)$$

$$\mathbf{a}_i(t + \Delta t) = \mathbf{a}\left(\mathbf{r}_1(t + \Delta t), \dots, \mathbf{r}_N(t + \Delta t)\right)$$

$$\mathbf{v}_i(t + \Delta t) = \mathbf{v}_i\left(t + \frac{\Delta t}{2}\right) + \frac{1}{2} \Delta t \, \mathbf{a}_i(t + \Delta t)$$

### Simulation Units
The simulation adopts natural astronomical units:
- **Length**: Astronomical Units ($\text{AU}$)
- **Mass**: Solar Masses ($M_\odot$)
- **Time**: Years ($\text{yr}$)
- **Gravitational Constant**: $G = 4\pi^2 \approx 39.4784176\text{ AU}^3 / (M_\odot \cdot \text{yr}^2)$

Under these units, a $1.0\,M_\odot$ star orbited by a test particle at $1.0\text{ AU}$ has an orbital period of exactly $P = 1.0\text{ yr}$ and an orbital speed of $v = 2\pi\text{ AU/yr} \approx 29.78\text{ km/s}$.

### Gravitational Softening
Gravitational softening is set to $\epsilon = 0.005\text{ AU}$ ($\sim 748,000\text{ km} \approx 1.07\,R_\odot$). Close encounters at separations $r \lesssim \epsilon$ are numerically smoothed to prevent singular accelerations and step-size collapse. Encounters below this scale are **numerically softened rather than physically resolved**. Physical collisions are evaluated independently using the physical radii of the bodies, ensuring numerical softening is never conflated with physical collision geometry.

---

## 3. Scientific Diagnostics

- **Conservation Diagnostics**:
  - Total mechanical energy $E(t) = T(t) + V(t)$ and relative energy drift $\Delta E / E_0 = (E(t) - E_0) / |E_0|$.
  - Total angular momentum $L_z(t) = \sum_i m_i (x_i v_{y,i} - y_i v_{x,i})$ and relative drift $\Delta L_z / L_{z0}$.
- **Osculating Orbital Elements**:
  - Instantaneous two-body elements computed relative to the host star: semimajor axis $a(t)$, eccentricity $e(t)$, periapsis $r_{\text{peri}}$, apoapsis $r_{\text{apo}}$, longitude of periapsis $\varpi$, and mean longitude $\lambda$.
- **Kepler's Third Law Validation**:
  - Measures orbital periods empirically via unwrapped true-anomaly crossing times ($2\pi$ radians) using linear interpolation, computing $P^2 / a^3$ without hardcoding theoretical periods.
- **Hill-Radius Stability Criterion**:
  - Giant planet Hill sphere:
    $$R_H = a_{\text{giant}} \left(\frac{M_{\text{giant}}}{3 M_{\star}}\right)^{1/3}$$
  - Close-encounter threshold: planetary separation evaluated in units of $R_H$.
- **3:1 Resonant-Angle Diagnostic**:
  - Tracks the resonant argument:
    $$\phi = 3\lambda_{\text{giant}} - \lambda_{\text{test}} - 2\varpi_{\text{test}}$$
  - The simulation **explicitly distinguishes the nominal period-ratio location ($P_J / P_t \approx 3$) from true resonance**. Libration vs. circulation is evaluated from the circular distribution of $\phi(t)$ over time using directional statistics (mean resultant vector length $R$ and angular deviation span).
- **Total Received Stellar Flux & Radiative Habitability**:
  - Evaluates instantaneous insolation from all stellar components:
    $$F_{\text{total}} = \frac{L_A}{r_A^2} + \frac{L_B}{r_B^2} \quad [S_\odot]$$
  - Approximate radiative habitability corresponds to $F_{\text{total}} \in [0.53, 1.11]\,S_\odot$ (runaway greenhouse to maximum greenhouse limits). Clearly labeled as an approximate instantaneous radiative proxy, not a full planetary climate model.

---

## 4. Stability Map Experiment: "Where Do Planetary Orbits Survive?"

The central scientific experiment tests test planet survival in the S-type binary system with a giant perturber (Preset 2: Star A $1.0\,M_\odot$, Star B $0.35\,M_\odot$ at $20\text{ AU}$, Giant Perturber $0.002\,M_\odot$ at $a = 2.0801\text{ AU}$).

### Parameter Grid & Setup
- **Initial Semimajor Axis ($a_0$)**: $0.70 - 1.50\text{ AU}$ ($40$ columns)
- **Initial Eccentricity ($e_0$)**: $0.00 - 0.40$ ($25$ rows)
- **Total Configurations**: $40 \times 25 = 1,000$ independent N-body integrations
- **Integration Duration**: $5.0\text{ yr}$ per configuration ($25,000$ integration substeps at $dt = 0.0002\text{ yr}$)
- **Execution**: Headless on NVIDIA CUDA GPU ($960.81\text{ s}$ total runtime, $\approx 1.0\text{ pts/s}$)

### Numerical Results (1,000 Configurations)
- **STABLE**: **`439` points (43.9%)** — Dominates lower eccentricities ($e_0 \le 0.18$) and inner orbits.
- **PERTURBED**: **`484` points (48.4%)** — Moderate eccentricity growth or close approaches within $3.0\,R_H$.
- **UNSTABLE**: **`77` points (7.7%)** — Strong perturbation zone where orbits penetrate within $1.0\,R_H$ of the giant or reach $e \ge 0.70$.
- **COLLISION**: **`0` points**
- **EJECTED**: **`0` points**

### Transition Boundaries
- **Eccentricity Boundary**: A sharp transition occurs at **$e_0 \approx 0.20 - 0.22$**. Below $e_0 = 0.20$, $75-100\%$ of orbits remain STABLE. At $e_0 \ge 0.2333$, $0\%$ of configurations remain STABLE across all semimajor axes.
- **Semimajor-Axis Boundary**: The strongest transition toward instability occurs across **$a_0 \in [1.29, 1.44]\text{ AU}$**, where the STABLE fraction drops from $48\%$ to $12\%$, and UNSTABLE configurations rise from $4\%$ to $36\%$ due to Hill sphere encroachment.

> [!NOTE]
> These classifications are **strictly 5-year numerical outcomes** and do not represent claims of permanent stability over astronomical timescales ($10^5 - 10^8\text{ yr}$).

---

## 5. The 3:1 Resonance Region

For the giant planet orbit at $a_{\text{giant}} = 2.08008\text{ AU}$, the nominal 3:1 Keplerian resonance occurs at:

$$a_{\text{res}} = \frac{a_{\text{giant}}}{3^{2/3}} \approx 1.0000\text{ AU}$$

In the stability map across the corridor $a_0 \in [0.95, 1.05]\text{ AU}$ ($125$ configurations):
- **STABLE**: `63` ($50.4\%$)
- **PERTURBED**: `62` ($49.6\%$)
- **UNSTABLE**: `0` ($0.0\%$)

### Dynamical Behavior Near Resonance
- **Zero Gross Instability at 5 Years**: The resonance corridor does not exhibit rapid disruption or ejection within 5 years.
- **Enhanced Eccentricity Excitation**: Configurations immediately interior to the nominal resonance ($a_0 \approx 0.946 - 0.967\text{ AU}$) show measurable eccentricity excitation ($\max(e_{\max} - e_0) \approx +0.0096$), which is **5 to 15 times higher** than in deep interior non-resonant columns ($a_0 = 0.70\text{ AU}$). Mean semimajor-axis drift $\langle |\Delta a|/a_0 \rangle \approx 0.0021$ is also elevated.

---

## 6. Numerical Validation

The simulation includes an automated headless verification suite testing the numerical engine against known analytic baselines:

```
======================================================================
HABITABLE CHAOS: SCIENTIFIC VERIFICATION SUITE
======================================================================
[Verification 1/4] Backend initialized: NVIDIA CUDA (GPU)
[Verification 2/4] Kepler Benchmark Orbit (1.0 AU, 1.0 M_sun):
  -> Empirical Mean Period P_meas: 1.00009 yr (Theoretical: 1.00000 yr)
  -> Empirical Kepler Ratio P^2 / a^3: 1.00012
  -> Kepler Deviation: 0.0122%
  -> Relative Energy Drift dE/E0: +2.050e-05
  -> Relative Angular Momentum Drift dLz/Lz0: +1.022e-05
[Verification 3/4] Multi-body Hierarchical S-Type Binary (Preset 1):
  -> Multi-body Energy Drift dE/E0: +1.487e-05
  -> Multi-body Angular Momentum Drift dLz/Lz0: +6.155e-06
  -> Operational Stability: STABLE
[Verification 4/4] 3:1 Mean Motion Resonance Setup & Diagnostics (Preset 2):
  -> Period ratio P_giant / P_test: 2.9970
  -> Giant Hill radius R_H: 0.1817 AU
  -> 3:1 Resonant angle phi: 163.37 deg (2.8514 rad)
======================================================================
ALL SCIENTIFIC VERIFICATION TESTS PASSED SUCCESSFULLY.
======================================================================
```

Across the full 1,000-point stability scan:
- **Maximum Energy Drift**: $\max |\Delta E / E_0| = \mathbf{3.48 \times 10^{-5}}$
- **Maximum Angular Momentum Drift**: $\max |\Delta L_z / L_{z0}| = \mathbf{1.63 \times 10^{-5}}$

---

## 7. Running the Simulation

### Prerequisites
- **Python**: 3.12 (recommended)
- **Dependencies**: Taichi and NumPy

Install dependencies directly:
```powershell
pip install taichi numpy
```
*(Or inside a virtual environment:)*
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install taichi numpy
```

### Execution Modes
1. **Interactive Simulation & Stability Map GUI**:
   ```powershell
   python sim.py
   ```
2. **Headless Scientific Verification Suite**:
   ```powershell
   python sim.py --verify
   ```
3. **Headless Stability Map Parameter Scan**:
   ```powershell
   # Quick test scan (8 x 5 = 40 configurations)
   python sim.py --scan --grid 8x5 --duration 5.0

   # Full scientific scan (40 x 25 = 1,000 configurations)
   python sim.py --scan --grid 40x25 --duration 5.0
   ```

### Backend Support
The simulation initializes Taichi with **NVIDIA CUDA (GPU)** acceleration when an NVIDIA GPU is available. If CUDA is absent, it falls back automatically to **Vulkan (GPU)** and then to multithreaded **CPU (x64)** execution.

---

## 8. Interactive Controls

The interactive window runs at $1280 \times 800$ resolution with vsync:

| Key / Control | Action |
| :--- | :--- |
| `SPACE` | Pause / Resume simulation |
| `R` | Reset current scenario to initial state |
| `1` | Switch to Preset 1: S-Type Binary (Calm HZ Benchmark) |
| `2` | Switch to Preset 2: Nominal 3:1 MMR (Resonance Pumping) |
| `3` | Switch to Preset 3: Tight Binary Intruder (Chaotic Scattering) |
| `4` | Switch to Preset 4: Kepler 3rd Law Benchmark (Single Star) |
| `ESC` | Exit simulation cleanly |

### Stability Map GUI Interaction
- **View Toggle**: Click the top dashboard button (**"Switch to STABILITY MAP VIEW"** / **"Switch to ORBIT SIMULATION"**) to toggle between the dynamic orbit view and the $(a_0, e_0)$ parameter map.
- **Reference Line**: Nominal 3:1 resonance line marked in cyan at $a_{\text{res}} = 1.000\text{ AU}$.
- **Point Inspector**: Select any grid point via the slider to inspect initial $(a_0, e_0)$, final $(a, e)$, $e_{\max}$, minimum giant separation ($d/R_H$), energy drift, and stability classification.
- **Load into Simulation**: Click **"--> LOAD INTO SIMULATION <--"** to immediately load the selected initial condition into the live N-body engine and observe its dynamic trajectory.

---

## 9. Scientific Limitations & Approximations

1. **Finite Integration Horizon ($t = 5.0\text{ yr}$)**:
   The stability map reflects 5-year numerical integration. Orbits classified as STABLE or PERTURBED are not guaranteed to be permanently stable over secular or astronomical timescales ($10^4 - 10^7\text{ yr}$).
2. **Coplanar 2D Dynamics**:
   All orbital inclinations are set to $i = 0^\circ$. Three-dimensional mutual inclinations (e.g., Kozai-Lidov resonance cycles) are not modeled.
3. **Gravitational Softening Scale**:
   Encounters with separations $r < 0.005\text{ AU}$ are smoothed by $\epsilon$ to prevent numerical singularities; close-range tidal effects and collision dynamics below this scale are not physically resolved.
4. **Radiative Proxy vs. Climate Modeling**:
   Habitability is evaluated strictly through total instantaneous stellar flux ($F_{\text{total}} \in [0.53, 1.11]\,S_\odot$). This does not model planetary albedo, greenhouse gas feedbacks, atmospheric heat transport, rotation period, or obliquity.
5. **Instantaneous Osculating Elements**:
   Orbital elements ($a, e, \varpi, \lambda$) are instantaneous two-body Keplerian projections relative to the primary star; in strongly perturbed regimes, they represent instantaneous velocity/position vectors rather than invariant orbital trajectories.

---

## 10. Reproducibility

Every result in the stability map is generated from deterministic numerical integrations of the initial state vectors and stored in [`stability_map_data.json`](file:///d:/YPAE/stability_map_data.json). The scan initial conditions, numerical parameters, and classification criteria are reproducible by running `python sim.py --scan`.

---

## 11. Project Structure

```
d:\YPAE\
├── sim.py                     # Main simulation, physics kernels, renderer, scanner, and verification suite
├── stability_map_data.json    # Complete numerical dataset for the 1,000-point stability scan
├── temporary_render_test.py   # Render buffer benchmarking and test script
├── .gitignore                 # Git ignore rules for virtual environments and caches
└── README.md                  # Scientific documentation and reproduction guide
```
