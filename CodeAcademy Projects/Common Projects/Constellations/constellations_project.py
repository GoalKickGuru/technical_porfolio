# ═══════════════════════════════════════════════════════════════════
# Extended Constellations Project
# — Beyond the Codecademy "Orion" starter into real astrometry —
#
# Background (from Wikipedia – Star Position):
#
#   Star positions are described by the equatorial coordinate system:
#
#     • Right Ascension (α): the "longitude" of the sky, measured in
#       hours (0 h – 24 h) eastward along the celestial equator.
#       It maps to Earth's rotation: 1 hour of RA ≈ 15° of sky rotation.
#
#     • Declination (δ): the "latitude" of the sky, measured in degrees
#       from +90° (north celestial pole) to −90° (south celestial pole),
#       with 0° at the celestial equator.
#
#     • Distance (d): usually given in light-years (ly) or parsecs (pc).
#       This third coordinate lets us convert angular positions on the
#       apparent celestial sphere into true 3-D Cartesian positions.
#
#   Conversion from spherical (α, δ, d) → Cartesian (x, y, z):
#
#       x = d · cos(δ) · cos(α)
#       y = d · cos(δ) · sin(α)
#       z = d · sin(δ)
#
#   Star positions drift slowly due to:
#     1. Precession (Earth's axis wobble, ~26 000 yr cycle)
#     2. Nutation (small superimposed wobble, 18.6 yr cycle)
#     3. Proper motion (individual stars' real tangential drift)
#     4. Aberration & parallax (observer-dependent effects)
#
#   Catalogues compile these positions: Hipparcos (~118 000 stars),
#   Tycho-2 (~2.5 million), SAO (~250 000), Bonner Durchmusterung
#   (~325 000, 1859–1863).
# ═══════════════════════════════════════════════════════════════════

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (registers 3-D projection)
from matplotlib.gridspec import GridSpec


# ───────────────────────────────────────────────────────────────────
# 1. STAR DATA
#    Each star is a dict with name, right_ascension (hours),
#    declination (degrees), distance (light-years), apparent
#    magnitude (brightness), and Bayer designation.
#
#    Sources: Simbad / Hipparcos catalogue values, rounded.
# ───────────────────────────────────────────────────────────────────

# --- Orion (original Codecademy data, kept for reference) ---
# The original project used abstract x,y,z. We keep them so the
# learner can compare, but we ALSO provide real (α, δ, d) data.
orion_x_old = [-0.41,  0.57,  0.07,  0.00, -0.29, -0.32, -0.50, -0.23, -0.23]
orion_y_old = [ 4.12,  7.71,  2.36,  9.10, 13.35,  8.13,  7.19, 13.25, 13.43]
orion_z_old = [ 2.06,  0.84,  1.56,  2.07,  2.36,  1.72,  0.66,  1.25,  1.38]

# Real Orion star data: RA (h), Dec (°), Distance (ly), Vmag, Name
orion_stars = [
    # name,        RA(h)   Dec(°)   Dist(ly)  Vmag   Bayer
    ("Betelgeuse",  5.919,  7.407,   548.0,   0.42,  "α Ori"),
    ("Rigel",       5.242, -8.202,   863.0,   0.18,  "β Ori"),
    ("Bellatrix",   5.418,  6.350,   250.0,   1.64,  "γ Ori"),
    ("Mintaka",     5.533, -0.300,   1200.0,  2.23,  "δ Ori"),
    ("Alnilam",     5.603, -1.202,   1342.0,  1.69,  "ε Ori"),
    ("Alnitak",     5.679, -1.943,   1260.0,  1.74,  "ζ Ori"),
    ("Saiph",       5.796, -9.670,    650.0,  2.09,  "κ Ori"),
    ("Meissa",      5.589,  9.931,   1100.0,  3.39,  "λ Ori"),
    ("Hatsya",      5.592, -5.391,   1340.0,  3.71,  "ι Ori"),  # iota Orionis
]

# --- Ursa Major (Big Dipper) ---
ursa_major_stars = [
    ("Dubhe",   11.062,  61.751,   123.0,  1.79, "α UMa"),
    ("Merak",   11.030,  56.382,    79.0,  2.37, "β UMa"),
    ("Phecda",  11.897,  53.694,    84.0,  2.44, "γ UMa"),
    ("Megrez",  12.257,  57.032,    81.0,  3.31, "δ UMa"),
    ("Alioth",  12.900,  55.960,    81.0,  1.77, "ε UMa"),
    ("Mizar",   13.398,  54.926,    83.0,  2.04, "ζ UMa"),
    ("Alkaid",  13.792,  49.313,   104.0,  1.86, "η UMa"),
]

# --- Cassiopeia (W shape) ---
cassiopeia_stars = [
    ("Caph",     0.153,  59.150,   228.0,  2.27, "β Cas"),
    ("Schedar",  0.675,  56.537,   228.0,  2.24, "α Cas"),
    ("Gamma Cas", 0.945, 60.717,   550.0,  2.47, "γ Cas"),
    ("Ruchbah",  1.430,  60.235,    99.0,  2.68, "δ Cas"),
    ("Segin",    1.907,  63.670,   441.0,  3.38, "ε Cas"),
]


# ───────────────────────────────────────────────────────────────────
# 2. COORDINATE CONVERSION
#    Convert (RA_hours, Dec_deg, Distance_ly) → Cartesian (x, y, z)
#
#    RA is in hours → multiply by 15 to get degrees, then radians.
#    Dec is already in degrees → radians directly.
# ───────────────────────────────────────────────────────────────────

def ra_dec_to_cartesian(ra_hours, dec_deg, distance_ly):
    """
    Convert equatorial coordinates to 3-D Cartesian coordinates.

    Parameters
    ----------
    ra_hours : float   Right Ascension in hours (0–24)
    dec_deg  : float   Declination in degrees (-90 to +90)
    distance_ly : float  Distance from Earth in light-years

    Returns
    -------
    tuple (x, y, z) in light-years
    """
    alpha = math.radians(ra_hours * 15.0)   # RA: hours → degrees → radians
    delta = math.radians(dec_deg)            # Dec: degrees → radians
    d = distance_ly

    x = d * math.cos(delta) * math.cos(alpha)
    y = d * math.cos(delta) * math.sin(alpha)
    z = d * math.sin(delta)
    return x, y, z


def stars_to_xyz(star_list):
    """Convert a list of star dicts/tuples into arrays of x, y, z, mag, names."""
    xs, ys, zs, mags, names, bayers = [], [], [], [], [], []
    for name, ra, dec, dist, vmag, bayer in star_list:
        x, y, z = ra_dec_to_cartesian(ra, dec, dist)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        mags.append(vmag)
        names.append(name)
        bayers.append(bayer)
    return (np.array(xs), np.array(ys), np.array(zs),
            np.array(mags), names, bayers)


# ───────────────────────────────────────────────────────────────────
# 3. MAGITUDE → MARKER SIZE
#    Brighter stars (lower magnitude) → larger markers.
#    Vmag ranges roughly from −1.5 (Sirius) to +6 (barely visible).
# ───────────────────────────────────────────────────────────────────

def mag_to_size(mag_array, min_size=20, max_size=300):
    """
    Map apparent magnitudes to marker sizes.

    Lower (brighter) magnitude → larger marker.
    """
    mag_min, mag_max = mag_array.min(), mag_array.max()
    if mag_max == mag_min:
        return np.full_like(mag_array, (min_size + max_size) / 2, dtype=float)
    normalized = (mag_max - mag_array) / (mag_max - mag_min)  # 0→dim, 1→bright
    return min_size + normalized * (max_size - min_size)


# ───────────────────────────────────────────────────────────────────
# 4. PROPER MOTION SIMULATION
#    Stars don't stay fixed—they drift across the sky over millennia.
#    We simulate proper motion by adding a small velocity vector
#    to each star's Cartesian position, scaled by time (years).
# ───────────────────────────────────────────────────────────────────

def simulate_proper_motion(star_list, years=50_000):
    """
    Simulate where stars will be (approximately) after `years` years.

    We generate small random proper-motion vectors (milliarcseconds/yr)
    scaled to plausible astrometric values. In real research you'd
    use measured μ_α, μ_δ from the Hipparcos catalogue.

    Returns the displaced (x, y, z) arrays.
    """
    xs, ys, zs, mags, names, bayers = stars_to_xyz(star_list)
    # Typical proper motion: 10–100 mas/yr.  Over 50 kyr that shifts
    # positions by a fraction of a light-year at typical distances.
    np.random.seed(42)
    pm_mas_per_yr = np.random.uniform(10, 80, size=len(xs))
    pm_ly_per_yr  = pm_mas_per_yr * 1e-3 / 206265.0 * np.sqrt(xs**2 + ys**2 + zs**2) * 0 + pm_mas_per_yr * 1e-6
    # Direction: random unit vectors
    directions = np.random.randn(len(xs), 3)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    displacement = directions * pm_ly_per_yr[:, None] * years
    new_xs = xs + displacement[:, 0]
    new_ys = ys + displacement[:, 1]
    new_zs = zs + displacement[:, 2]
    return new_xs, new_ys, new_zs


# ───────────────────────────────────────────────────────────────────
# 5. VISUALIZATION
# ───────────────────────────────────────────────────────────────────

NIGHT_BG   = "#01010f"
STAR_COLOR = "#ffe9a8"
GRID_COLOR = "#1a1a3a"

constellations = {
    "Orion":      orion_stars,
    "Ursa Major": ursa_major_stars,
    "Cassiopeia": cassiopeia_stars,
}


# --- 5a. Original 2-D scatter (x, y) — baseline from Codecademy ---
fig0, ax0 = plt.subplots(figsize=(8, 6))
ax0.scatter(orion_x_old, orion_y_old, s=120, c="white", edgecolors="steelblue")
ax0.set_title("Orion — Original 2D (x, y) from Codecademy Data")
ax0.set_xlabel("x")
ax0.set_ylabel("y")
ax0.grid(True, alpha=0.3)
plt.tight_layout()


# --- 5b. 2-D scatter from real RA/Dec (projected on sky) ---
fig1, ax1 = plt.subplots(figsize=(10, 6))
for cname, stars in constellations.items():
    ras  = [s[1] for s in stars]
    decs = [s[2] for s in stars]
    mags = np.array([s[4] for s in stars])
    sizes = mag_to_size(mags)
    ax1.scatter(ras, decs, s=sizes, label=cname,
                c=STAR_COLOR if cname == "Orion" else None,
                edgecolors="white", linewidth=0.5)
    # Label each star
    for s in stars:
        ax1.annotate(s[0], (s[1], s[2]),
                     textcoords="offset points", xytext=(5, 5),
                     fontsize=7, color="#aaa", alpha=0.8)
ax1.set_facecolor(NIGHT_BG)
ax1.invert_xaxis()  # RA increases eastward (left) in sky views
ax1.set_xlabel("Right Ascension (hours)")
ax1.set_ylabel("Declination (degrees)")
ax1.set_title("Night Sky Projection — Three Constellations")
ax1.legend(facecolor=GRID_COLOR, edgecolor=GRID_COLOR, labelcolor="white")
plt.tight_layout()


# --- 5c. 3-D scatter from real Cartesian coordinates — Orion ---
orion_data = stars_to_xyz(orion_stars)
ox, oy, oz, omags, onames, _ = orion_data
osizes = mag_to_size(omags)

fig2 = plt.figure(figsize=(9, 8))
ax2 = fig2.add_subplot(111, projection="3d")
ax2.scatter(ox, oy, oz, s=osizes, c=STAR_COLOR, edgecolors="white",
            linewidth=0.3, depthshade=True)
for i, name in enumerate(onames):
    ax2.text(ox[i], oy[i], oz[i], name, fontsize=7, color="#ccc")
ax2.set_xlabel("X (ly)")
ax2.set_ylabel("Y (ly)")
ax2.set_zlabel("Z (ly)")
ax2.set_title("Orion — True 3D Positions (Cartesian from RA/Dec/Distance)")
ax2.set_facecolor(NIGHT_BG)
# Dark grid
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
plt.tight_layout()


# --- 5d. Night-sky styled 2D + 3D side by side ---
fig3 = plt.figure(figsize=(16, 6))
gs = GridSpec(1, 2, figure=fig3)

# 2D night sky
ax3a = fig3.add_subplot(gs[0, 0])
ax3a.set_facecolor(NIGHT_BG)
ax3a.scatter(ox, oy, s=osizes, c=STAR_COLOR, alpha=0.9)
for i, name in enumerate(onames):
    ax3a.annotate(name, (ox[i], oy[i]), textcoords="offset points",
                  xytext=(6, 6), fontsize=7, color="#888")
ax3a.set_title("Orion — 2D Night Sky")
ax3a.set_xlabel("X (ly)")
ax3a.set_ylabel("Y (ly)")
ax3a.tick_params(colors="#666")

# 3D night sky
ax3b = fig3.add_subplot(gs[0, 1], projection="3d")
ax3b.set_facecolor(NIGHT_BG)
ax3b.scatter(ox, oy, oz, s=osizes, c=STAR_COLOR, alpha=0.9)
ax3b.set_title("Orion — 3D Night Sky")
ax3b.set_xlabel("X (ly)")
ax3b.set_ylabel("Y (ly)")
ax3b.set_zlabel("Z (ly)")
ax3b.xaxis.pane.fill = False
ax3b.yaxis.pane.fill = False
ax3b.zaxis.pane.fill = False
ax3b.tick_params(colors="#666")

plt.tight_layout()


# --- 5e. All three constellations in 3D ---
fig4 = plt.figure(figsize=(10, 8))
ax4 = fig4.add_subplot(111, projection="3d")
ax4.set_facecolor(NIGHT_BG)

colors_map = {"Orion": "#ffe9a8", "Ursa Major": "#9bc4ff", "Cassiopeia": "#ff9bb3"}

for cname, stars in constellations.items():
    sx, sy, sz, smags, snames, _ = stars_to_xyz(stars)
    ssizes = mag_to_size(smags)
    ax4.scatter(sx, sy, sz, s=ssizes, c=colors_map[cname],
                label=cname, edgecolors="white", linewidth=0.3, alpha=0.9)

ax4.set_xlabel("X (ly)")
ax4.set_ylabel("Y (ly)")
ax4.set_zlabel("Z (ly)")
ax4.set_title("Three Constellations — True 3D Positions")
ax4.legend(facecolor=GRID_COLOR, edgecolor=GRID_COLOR, labelcolor="white")
for axis in [ax4.xaxis, ax4.yaxis, ax4.zaxis]:
    axis.pane.fill = False
plt.tight_layout()


# --- 5f. Proper Motion: Orion now vs. 50,000 years from now ---
future_x, future_y, future_z = simulate_proper_motion(orion_stars, years=50_000)

fig5 = plt.figure(figsize=(14, 6))
gs2 = GridSpec(1, 2, figure=fig5)

# Current
ax5a = fig5.add_subplot(gs2[0, 0])
ax5a.set_facecolor(NIGHT_BG)
ax5a.scatter(ox, oy, s=osizes, c=STAR_COLOR, alpha=0.9)
ax5a.set_title("Orion — Present Day (2D projection)")
ax5a.set_xlabel("X (ly)")
ax5a.set_ylabel("Y (ly)")

# Future
ax5b = fig5.add_subplot(gs2[0, 1])
ax5b.set_facecolor(NIGHT_BG)
ax5b.scatter(future_x, future_y, s=osizes, c="#9bc4ff", alpha=0.9)
ax5b.set_title("Orion — ~50 000 Years From Now")
ax5b.set_xlabel("X (ly)")
ax5b.set_ylabel("Y (ly)")

# Draw arrows from current → future
for i in range(len(ox)):
    ax5a.annotate("", xy=(future_x[i], future_y[i]), xytext=(ox[i], oy[i]),
                  arrowprops=dict(arrowstyle="->", color="#555", lw=0.8))

plt.tight_layout()


# --- 5g. Celestial sphere schematic ---
# Show how RA/Dec map onto a sphere, using Orion stars' angular positions.
fig6, ax6 = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})
ax6.set_facecolor(NIGHT_BG)

# Draw a faint celestial sphere wireframe
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 20)
R = 100  # arbitrary radius for the sphere
sphere_x = R * np.outer(np.cos(u), np.sin(v))
sphere_y = R * np.outer(np.sin(u), np.sin(v))
sphere_z = R * np.outer(np.ones_like(u), np.cos(v))
ax6.plot_wireframe(sphere_x, sphere_y, sphere_z, color=GRID_COLOR, alpha=0.15)

# Plot Orion stars on the sphere surface (normalized to R)
for name, ra, dec, dist, vmag, bayer in orion_stars:
    alpha_r = math.radians(ra * 15)
    delta_r = math.radians(dec)
    sx = R * math.cos(delta_r) * math.cos(alpha_r)
    sy = R * math.cos(delta_r) * math.sin(alpha_r)
    sz = R * math.sin(delta_r)
    size = mag_to_size(np.array([vmag]))[0]
    ax6.scatter(sx, sy, sz, s=size, c=STAR_COLOR, edgecolors="white", zorder=5)
    ax6.text(sx, sy, sz, f" {name}", fontsize=6, color="#aaa")

ax6.set_title("Orion Stars on the Celestial Sphere (angular positions)")
ax6.set_xlabel("X")
ax6.set_ylabel("Y")
ax6.set_zlabel("Z")
for axis in [ax6.xaxis, ax6.yaxis, ax6.zaxis]:
    axis.pane.fill = False

plt.tight_layout()


# ───────────────────────────────────────────────────────────────────
# 6. PRINT ANALYTICAL SUMMARY
# ───────────────────────────────────────────────────────────────────

print("=" * 60)
print("  EXTENDED CONSTELLATIONS — ANALYTICAL SUMMARY")
print("=" * 60)

for cname, stars in constellations.items():
    xs, ys, zs, mags, names, _ = stars_to_xyz(stars)
    dists = np.array([s[3] for s in stars])
    print(f"\n  ▸ {cname}")
    print(f"    Stars: {len(stars)}")
    print(f"    Distance range: {dists.min():.0f} – {dists.max():.0f} ly "
          f"(span: {dists.max()-dists.min():.0f} ly)")
    print(f"    Magnitude range: {mags.min():.2f} (brightest) – "
          f"{mags.max():.2f} (dimmest)")
    # Spatial spread
    span_xyz = np.ptp(np.column_stack([xs, ys, zs]), axis=0)
    print(f"    Spatial extent: Δx={span_xyz[0]:.0f}  Δy={span_xyz[1]:.0f}"
          f"  Δz={span_xyz[2]:.0f} ly")

print("\n" + "=" * 60)
print("  Coordinate note:")
print("  RA is measured in hours (1h = 15°). The conversion")
print("  multiplies RA×15 to get degrees before converting to radians.")
print("  The 3D Cartesian coordinates preserve TRUE interstellar")
print("  distances, revealing that the 'constellation' shape we")
print("  see from Earth is a projection—stars are at vastly")
print("  different depths along our line of sight.")
print("=" * 60)

plt.show()