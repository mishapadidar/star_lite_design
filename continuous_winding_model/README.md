# Continuous winding model of the coils

This folder models each Star_Lite coil as it is actually wound: one continuous
cable, not 18 closed loops. The existing model, `finite_build/hex_finite_build_coil.py`,
places 18 closed filaments in a perfect hexagonal pattern around the spine. A real
cable also has turn-to-turn crossovers, a layer-to-layer transition and two leads.
None of these can be represented by closed loops. This model adds them, without
elasticity. If the effect on the field turns out to matter, the elastic relaxation
of `elastic_finite_build/` can be applied on top of this geometry later.

We use design A, `designs/designA/designA_after_scaled.json`, and a cable minor
radius $r = 0.0108$ m (winding pack radius $0.054/5$).

## How the coil is wound

- **Spine.** The steel spine is the design curve $\boldsymbol\gamma(t)$, $t\in[0,1)$,
  and is treated as exact.
- **Starting lead.** The cable starts in the first slot of the inner layer at $t_0$.
  The starting lead points straight out of the pack, perpendicular to the spine.
- **Turns.** The cable follows the spine once around. When its tip gets back to the
  start of the turn, it steps a little clockwise or counterclockwise, to start the
  next turn right next to the previous one. When a layer is full it steps up to the
  next layer.
- **Slot order.** Slots are filled in angular order. The direction is not
  standardized, so it is a parameter per layer. We think both layers were wound in
  the same direction.
- **Going around the lead.** The outer turns go around the starting lead (not over
  it: it points straight out).
- **Finishing lead.** The end of the cable finishes within a couple of cm of the
  exit point of the starting lead. The leads are side by side, not twisted
  together (the cable is thick).

## Geometry (`winding.py`)

All 18 winds: 6 in the inner layer, the climb, and 12 in the outer layer, wound in
the same direction. We added one step at a time and checked each in ParaView.

The cable is laid point by point in the order it is wound, one point per spine grid
point (`n_per_turn` = 1000 per turn, about 3 mm apart). A point's position in the
normal plane is its angle $\psi$ around the spine, in the centroid frame
$\mathbf n, \mathbf b$ (the frame of `create_hexagonal_filament_grid`) measured from
the first slot in the winding direction, and its distance $\rho$ from the spine.

- **Resting.** The winding tension pulls the cable onto what is beneath it: $\rho$
  is the smallest distance $\ge 2r$ at which it touches neither the spine nor the
  cable of the layers below. Inner-layer cable sits at $\rho = 2r$. Outer-layer
  cable sits at $2\sqrt3\,r$ in a groove and at $4r$ on top of an inner cable.
- **Starting lead.** A straight rod from the first slot at $t_0$, perpendicular to
  the spine, in the direction `lead_angle` (default: straight out from the slot).
- **Contact.** At each point the cable may sit anywhere in the free interval of
  angles around where it is, between the cable of its own layer behind it and the
  cable ahead of it, laid one turn or more before. If the cable behind moves into
  its place, it is pushed forward to the first clear angle. Each wind therefore
  lies against the previous one and steps over where the previous wind steps.
  There is no prescribed step location or shape.
- **The lead.** A cable passes the lead on the side its centre is on as it gets
  there. The first wind, and the outer wind whose slot is on top of it, are pushed
  past it on the forward side, touching it. A cable just before it stays on the
  near side: near $t_0$ the lead bounds its angle from above.
- **Squeezed by the lead.** The lead takes up one slot of the outer layer where it
  comes out, so near $t_0$ the outer layer has 12 cables for 11 slots. The outer
  cables are pushed forward one after the other, and the last ones, with no room
  left, ride up over their neighbours (a local third layer).
- **Climbing.** When the last inner wind is pushed by the previous wind's step and
  there is no room left in the full inner layer, it climbs. It starts to lift off
  $3\ell$ before that point; from there on the inner layer is beneath it rather
  than holding it in its slot. It rides up over the previous wind's step and settles into
  the outer slot `outer_start_slot`; by default that is the groove between the last
  and the first inner slot.
- **Bending.** The tension pulls the cable toward its slot, within the bounds above,
  and its stiffness resists bending. With $y = 2r\psi$, each turn minimizes
  $\sum (\ell^4 y''^2 + (y - y_\text{slot})^2)$ within the bounds, where $\ell$ is
  `bend_length`. A larger $\ell$ means the cable starts moving over earlier before
  it touches an obstacle. Once a wind has been pushed more than half a slot
  forward, it is pulled toward the next slot.
- **Tension.** $\rho$ is not pulled into every dip of the surface beneath: the
  cable spans dips like a taut rope, its free spans curving at most
  1/`sag_radius` (20 cm). It is then smoothed for bending like the angle.
- **Finishing lead.** The last wind runs on, riding over whatever is in its way,
  until `lead_gap` ($2r$) before the starting lead. The end curves out of the pack
  and over to the starting lead, and runs alongside it, parallel and touching
  (one cable diameter behind it along the spine). The circuit closes between the
  far ends.
- **Current.** The cable carries $I/18$. The path is a closed polyline
  (`PolylineCurve`), so `BiotSavart` uses the midpoint rule on each segment. The
  bounded smoothing is polished to the exact minimizer, so the cable has no
  node-to-node jitter (which spoils smooth quadrature rules on the path).

## Visualization (`visualize_winding.py`)

Writes to `output/<coil and parameters>/`:

- `unrolled.png`: top, each wind as a band one cable wide, plotted as angle
  around the spine against arc length near $t_0$; bands touch where the winds
  touch. Bottom, the distance from the spine.
- `cross_sections.png` and `slice_sweep.mp4`: the cross section near $t_0$.
- `winding.mp4`: the coil being wound.
- `overlap.png`: where cables come closer than $2r$ in 3D.
- ParaView files: `cable.vtu`, `cable_layer1.vtu` (with the starting lead),
  `cable_layer2.vtu` (with the finishing lead), `leads.vtu`, `spine.vtu`, and
  `overlaps.vtu` (only if there are overlaps).

## Parameters

| Parameter | Default | Range to explore | Note |
|:-|:-|:-|:-|
| Number of winds `n_winds` | 3 | up to 18 | |
| First outer slot `outer_start_slot` | 11 | 0–11 | 30° per slot from the first inner slot; odd = groove |
| Winding direction `direction` | +1 | ±1 | counterclockwise in $(\mathbf n, \mathbf b)$ is +1 |
| First slot `start_slot` | 0 | 0–5 | in angular order |
| Start position $t_0$ | 0 | [0, 1) | where the leads are; see TODO |
| Bend length $\ell$ | 1 cm | 0.5–3 cm | how early the cable moves over |
| Pack twist $\alpha$ | 0 | e.g. 0–π/3 | rotation of the hexagon about the spine |
| Lead angle | first slot | [0, 2π) | direction of the starting lead in the normal plane |
| Lead length | 1 m | 0.5–2 m | |
| Lead gap `lead_gap` | $2r$ | | finishing lead behind the starting lead, along the spine |
| Lead exit radius | $6r$ | | where the finishing lead comes alongside the starting lead |
| Sag radius `sag_radius` | 20 cm | | how much a free span of cable sags |
| Points per turn | 1000 | | 3 mm spacing |

Each physical coil was wound separately. So the parameters can differ from coil to
coil, and the winding breaks the stellarator symmetry and field-period symmetry of
the design. `continuous_winding_coils` takes either one parameter dict for every
coil or one dict per coil. On the stellarator-symmetry copies (flipped
`RotatedCurve`s), the centroid frame is mirrored. The same `direction` value
therefore means the opposite physical direction there.

## Field analysis

| Script | What it does |
|:-|:-|
| `wind_coils.py` | winds the 6 coils of design A (18 winds, 1 m leads) into `output/designA/coil{i}.npz` |
| `field_models.py` | the coil models, current scaling, closed field lines, exact Boozer surfaces, continuation, QS error |
| `analyze_winding.py` | axis, X-point, $\iota$, QS error and Boozer surface of each model, and their differences |
| `quadrature_study.py` | convergence of the Biot–Savart quadrature with the number of points per turn |
| `resolution_study.py` | convergence of the exact Boozer surface with its resolution |
| `compare_pack.py` | the pack against the ideal hexagon, along each coil |
| `forces.py` | $\mathbf J\times\mathbf B$ forces on the cables |
| `export_hall_fit.py` | the Hall-probe T coil (coil index 5) of each model as input for Georg's `fit_custom_coil.py` |
| `hall_probe_repeatability.py` | noise and drift of the Hall-probe map, from the points the probe passed twice |
| `hall_probe_pose_drift.py` | refits the T coil's pose on each face of the Hall-probe map, to test whether it moved |

Coil models:

- **filament**: the design's coils, one filament per coil carrying the coil current $I$.
- **hex build**: the hexagonal finite build (`finite_build/hex_finite_build_coil.py`),
  18 closed filaments per coil, each carrying $I/18$.
- **hex build (polyline)**: the same filaments as polylines on the continuous
  winding's spine grid (1000 points per turn), so its difference with the continuous
  winding is the winding alone, not the discretization.
- **continuous**: one cable per coil carrying $I/18$, leads included. The path is
  closed by a straight piece (about 3 cm) between the far ends of the 1 m leads, which
  stands in for the power supply, so the leads are a hairpin carrying $\pm I/18$.

**Currents.** The design's currents of the configuration (configuration 0: ±41,077 A
in coils 0–3, −27,738 A in coils 4–5), scaled in each model by one factor so that the
mean $|\mathbf B|$ on that model's own magnetic axis is 0.0875 T, as in the design
(`utils/axis_field_strength_penalty.py`). In configuration 0 the factors are 1
(filament), 1.00038 (hex build) and 1.0017 (continuous; its axis is 1 mm off). A
common factor changes no field line, $\iota$ or Boozer surface.

**Quadrature.** `BiotSavart` weights all points of a curve equally: the trapezoid rule
on the Fourier curves of the filament and hex build (160 points per coil, as in the
design), the midpoint rule on each segment of the continuous winding's polyline
(1000 points per turn, 3 mm).

**Surfaces.** The continuous winding breaks the stellarator symmetry and the
field-period symmetry, so everything is computed on the full torus. The axis and the
X-point are closed field lines over one toroidal turn (`find_x_point` with
`nfp=1`, 101 points). The Boozer surfaces are exact Boozer surfaces (Newton;
`BoozerSurface` without `constraint_weight`), with `nfp=1` and no stellarator
symmetry, at the volume of the design's surface. A resolution of $k$ per field
period is `mpol` $= k$, `ntor` $= 2k$ on the full torus, on $(4k+1)\times(2k+1)$
collocation points. The continuous winding's surface is found by continuation: the
current is moved from the hex build (polyline) to the continuous winding in steps
(fractions 0.25, 0.5, 0.75, 1), each solve started from the last. Only the last solve,
all the current in the continuous winding, is the result.

## Results (design A, current configuration 0, logged 2026-10-28)

**Summary.** The three coil models give nearly the same field at the plasma.

- The hex build changes $\iota$ by +3.4e-3 (+1.8%) from the filament, $|\mathbf B|$ on
  the axis by ±2.3e-3 pointwise, the QS error by −0.14%, and moves the axis and X-point
  by at most 0.5 mm and 1.2 mm.
- The continuous winding then changes $\iota$ by −7e-4, $|\mathbf B|$ on the axis by
  ±5e-4, the QS error by +0.5%, and moves the axis and X-point by at most 0.9 mm and
  0.65 mm.
- The numerical errors are orders of magnitude smaller than these differences: about
  1e-5 in $\mathbf B$ from the quadrature and 1e-10 in $\iota$ from the Boozer
  surface resolution. Field-line tracing reproduces the Boozer $\iota$ to 1e-6.

### Field of the three models (`analyze_winding.py --k 12`)

| Model | $\iota$ (exact Boozer surface, $k = 12$) | QS error | mean $\lvert\mathbf B\rvert$ on its own axis, design currents |
|:-|-:|-:|-:|
| filament | 0.1883161 | 3.5485e-2 | 0.087500 T |
| hex build | 0.1917483 | 3.5436e-2 | 0.087467 T |
| hex build (polyline) | 0.1917416 | 3.5437e-2 | |
| continuous | 0.1910366 | 3.5613e-2 | 0.087350 T |
| design's stored surface (a spurious solution, see below) | 0.188245 | 3.5504e-2 | |

| | hex build vs filament | continuous vs hex build | continuous vs filament |
|:-|-:|-:|-:|
| $\Delta\iota$ | +3.43e-3 | −7.12e-4 | +2.72e-3 |
| axis shift, mean / max | 0.27 / 0.50 mm | 0.78 / 0.91 mm | 0.90 / 1.09 mm |
| axis shift, direction | 0.10 mm out in $R$, ±0.43 mm in $Z$ | 0.73 mm out, 0.17–0.47 mm down | 0.83 mm out |
| X-point shift, mean / max | 0.88 / 1.16 mm | 0.55 / 0.65 mm | 1.11 / 1.42 mm |
| X-point shift, direction | 0.47 mm out, 0.55–0.92 mm down | 0.51 mm out, 0.18–0.30 mm up | 0.98 mm out, 0.35–0.71 mm down |
| mean $\lvert\mathbf B\rvert$ on the design axis | +3.8e-6 | +1.5e-5 | +1.9e-5 |
| pointwise $\lvert\mathbf B\rvert$ on the design axis | −2.1e-3 … +2.3e-3 | −5.5e-4 … +4.3e-4 | −2.4e-3 … +2.3e-3 |
| QS error change | −0.14% | +0.50% | +0.36% |
| max distance between Boozer surfaces | 1.28 mm | 1.13 mm | 1.66 mm |

The X-point is the upper X-point line ($Z$ = 8–21 cm); its stellarator-symmetric
image below moves the opposite way in $Z$. The $|\mathbf B|$ rows use the design
currents. Scaled to 0.0875 T on its own axis, the hex build's currents go up by
1.0004 and the continuous winding's by 1.0017, which changes no field line, $\iota$
or surface.

- **The hex build's $\Delta\iota$ is real.** Field-line tracing (300 toroidal turns,
  weighted Birkhoff average) on each model's own surface, with no Boozer solve,
  gives $\iota$ = 0.1883166 (filament) and 0.1917465 (hex build).
  `finite_build/analyze_finite_build.py` compared the mean $|\mathbf B|$ on the axis
  (4e-6) and the X-point shift (1.16 mm). But its Boozer solves (`mpol` = `ntor` = 10,
  started from the design's surface) both land on the spurious branch. That gives
  $\Delta\iota$ = +5.7e-4 (not printed) and a QS error change of +1.5%, instead of
  +3.4e-3 and −0.14%.
- **The discretization of the continuous winding does not matter.** The hex build
  as polylines like the continuous winding's differs from the hex build by 2 µm on
  the axis and 7e-6 in $\iota$.
- **The continuous winding moves the axis almost uniformly outward.** Splitting it
  up (earlier run, mean axis shift):
  - the continuous winding on coils 0–3 alone gives 0.68 mm, on coils 4–5 alone
    0.28 mm;
  - the hex build plus two loops of $I/18$ around the spine per coil gives 0.35 mm.
    These are the steps of the two layers, both wound the same way;
  - the lead hairpins alone give 0.08 mm;
  - the rest is the reshaped pack near the leads and coil 4's bunching.

  Winding the two layers in opposite directions would cancel the loops.

### Accuracy of the computation

**Biot–Savart quadrature** (`quadrature_study.py`, `output/designA/quadrature_study.png`;
max relative error of $\mathbf B$ on the filament model's axis and X-point against a
much finer discretization):

- The filament is at machine precision from 128 points per coil, the hex build from
  256 (1e-11 at the design's 160).
- The continuous winding converges much more slowly. Its path has sharp features near
  the leads: 90° corners where the leads leave the pack, a 1.25 cm bend where the
  outer layer crosses the starting lead, and a 6 mm bend in the finishing lead.
  - One spline sampled uniformly (trapezoid rule): 1e-4 at 160 points per turn,
    3e-6 at 1024.
  - The spline split at the corners, with 4-point Gauss–Legendre panels: 1e-4 at
    160, 1e-5 at 362, 5e-7 at 1024.
  - The same with panels no longer than half the local bend radius: 1e-5 at
    110–150 points per turn, 1e-6 at 515. With a quarter of the bend radius, 2e-6
    at about 240.
- The polyline used for the results (1019 points per turn, midpoint rule): 9e-6 on the
  axis and 1.1e-5 on the X-point. That is about 2% of the continuous winding's effect
  on $|\mathbf B|$.
- While setting this up, two bugs were found and fixed.
  - The cable's distance from the spine jittered by 6 µm from point to point, because
    the bounded smoothing stopped short of its minimizer. The polish now in
    `smooth_above` fixes it; rewinding moved the cable by at most 0.17 mm and
    $\iota$ by 4e-6.
  - The spline resampling inserted midpoints into every segment longer than the
    median.

**Boozer surface resolution** (`resolution_study.py`,
`output/designA/resolution_study_config0.png`; $k$ = `mpol` = `ntor` per field
period, from 10 to 20).

- Every model is converged at $k = 10$. $\iota$ is within 1e-8 of its $k = 20$ value
  at $k = 10$, about 1e-10 at $k = 12$, and 1e-12 at $k = 16$.
- The continuous winding's $\iota$ is 0.1910366 at every $k$. The continuation took
  4 steps.
- But at $k = 10$ the collocation equations have a second, spurious solution next to
  the true one. It satisfies the equations at the collocation points only: its Boozer
  residual between them is 4.7e-5 (filament), against 2.7e-6 for the true solution.
- The design's stored surface is that spurious solution. Compared with the true
  surface at the same volume:
  - $\iota$ = 0.188245 instead of 0.188316;
  - 1.5 mm away;
  - QS error 3.5504e-2 instead of 3.5485e-2.
- Started from the design's surface, the $k = 10$ solve stays on the spurious branch.
  The hex build gives $\iota$ = 0.18882, 12 mm from its true surface ($\iota$ =
  0.19175), and the continuation to the continuous winding fails. Started from the
  $k = 11$ surfaces, it finds the true ones.
- From $k = 11$ up, starting from the design's surface finds the true solution. So
  `analyze_winding.py` uses $k = 12$.

**Error budget.**

| | numerical error | difference between models |
|:-|-:|-:|
| $\mathbf B$ (relative) | 1e-5 (quadrature) | 5e-4 to 2.4e-3 |
| $\iota$ | 1e-10 (resolution), 1e-6 (tracing check) | 7e-4 to 3.4e-3 |
| axis, X-point | unchanged from 101 to 151 points per closed field line; 2 µm from the polyline discretization | 0.3 to 1.4 mm |

### Pack shape (`compare_pack.py`, `output/designA/pack_deviation.png`)

More than 25 cm from the leads, coil 0's cables are within 0.021 mm of the ideal
hexagon. Coil 4 bunches by up to 4.5 mm (outer winds 12–18) at its two sharpest bends
($\sigma \approx \pm 130$ cm, bend radius 9.5 cm). There the centroid frame twists by
up to 5.7 rad/m (median 0.1), and the ideal hexagon itself overlaps by up to 1.1 mm.

### Forces (`forces.py`, `output/designA/forces.png`)

The forces tighten the pack and press it against the spine.

- Coil 0: 252 N/m per cable (median) and 2.3 kN/m on the pack (mean, up to
  3.5 kN/m). The pinch toward the spine is −126 N/m on the inner layer and −295 N/m
  on the outer. Only 3% of the cable is pulled away from the spine, by at most
  57 N/m.
- Coil 4: 120 N/m per cable and 1.3 kN/m on the pack (up to 2.4 kN/m). At most
  79 N/m pulls outward, on 5.7% of the cable.
- For comparison, the coil's weight is 14 N/m per metre of cable, or about 250 N/m
  per metre of spine.

### Comparison with the measured field

**The data and fit.**
- Arsenne mapped the field of one T coil (Georg's "coil 6" = designA coil index 5; index 4 is its mirror twin) with the 16-chip OCTO-BEE Hall probe: a 24 Sept 2026 map at 250 A, and held-out traverses on 26 Aug 2026 at 300 A.
- Georg fitted a Biot–Savart model to it. Report: https://claude.ai/artifact/BfA5qNTZ3Sfjhr4SWvPrrL.
- His data and code are in `../../hall_probe_mapping/` (outside the repo; start with `HANDOFF_Misha.md`).
- His `analysis/fit_custom_coil.py` runs his rigid fit with any winding:
  - the coil's three rotations about the taped bottom mount;
  - one rotation and three gains per chip;
  - the steel spine at μr 1000, or no steel.
- Residual: vector residual as a percentage of |B|, rms over chips and components.

| T coil model (250 A per conductor) | field vs Georg's pack at the chips | Sept, steel | Aug (held out), steel | Sept, no steel | Aug, no steel | gain |
|:-|-:|-:|-:|-:|-:|-:|
| Georg's pack (6 + 12 cables round the 22 mm spine) | | 1.54 % | 2.83 % | 2.20 % | 3.33 % | 1.107 |
| filament | 0.29 % | 1.62 % | 2.89 % | 2.30 % | 3.42 % | 1.105 |
| hex build, $r$ = 10.8 mm | 0.07 % | 1.54 % | 2.82 % | 2.19 % | 3.31 % | 1.108 |
| hex build, $r$ = 9.5 mm | 0.07 % | 1.56 % | 2.83 % | 2.21 % | 3.33 % | 1.107 |
| continuous winding, our leads | 0.83 % | 1.69 % | 2.94 % | 2.30 % | 3.41 % | 1.110 |

**What the fits show.**
- **Finite pack:** improves on the filament by about 0.1 percentage points. That is small next to the steel spine (0.7 points), Georg's flexed-coil fit (another 0.9 points) and the unexplained 8 % gain.
- **Continuous winding:** fits worse.
  - At its crossover, where every wind steps to its next position and the leads leave, it differs from the hex build by up to 400 µT (3.3 %).
  - In our model that place is the arbitrary start of the design curve, about 11 cm from the nearest probe points. The real coil's crossover is somewhere else.
  - The fit is sensitive to where the crossover is, so the photos (or a scan over its position) should fix it.
- **Gain:** 1.105–1.114 in every model. No winding model touches the 8 % excess.
- The excess follows the chip axes, not the rig axes. Chip x axes read about 10 % high and chip z axes about 6 %, whichever way they point (Georg's gain table), so at least that split is in the sensors.

**Noise and drift** (`hall_probe_repeatability.py`, raw rows):
- **Noise:** 2.7 µT rms per 10 ms reading (1.8–4.9 µT by chip), about 0.3 µT per 5 mm bin. Negligible.
- **Crossings:** the x-swept lines cross the y-swept lines at the box edges, so the probe passed 171 points twice, 34–173 min apart, with every chip in the same orientation.
- **Change between passes:** the readings differ by 35 µT rms per component (60 µT rms vector, up to 305 µT), 0.8 % of |B|.
- **It looks like movement:** one flange shift plus a gain change per crossing explains most of it (14 µT left).
- **The shift follows the time order.** Relative to the faces swept last (x0 and x250, at 141–206 min):
  - the faces swept in the first hour (y0, y250) are offset by +2.1 to +3.1 mm in rig x;
  - the faces swept in the second hour (z70, z270) by −0.5 to +0.9 mm.
- So relative to the probe, the coil (hung from the crane) crept about 3 mm in x over the first 1.5 h.
- **Per-face refit** (`hall_probe_pose_drift.py`, Georg's pack with steel, chips held at the September fit):
  - three rotations per face about the pinned mount lower the residual by at most 0.2 points per face;
  - with the mount free, a single face does not pin the pose.
- **Conclusion:** motion during the scan is a secondary part of the 1.5 % rigid residual. It is about the size of the 0.6 % flexed-coil residual, so part of that fit's bend may be motion.

## TODO

- [ ] Figure out the location of the leads on each built coil (from the photos):
  the start position $t_0$ along the spine and the lead angle, for every coil.
- [ ] Figure out the flange location on each built coil (from the photos).
- [ ] Outer layer winding direction: ask the winders. Wind 7 can start in the
  groove between winds 6 and 1 and go on toward 1 and 2 (the model now), or go the
  other way toward 5 and 6.
- [ ] Check against the photos: where the last wind rides over the others (a local
  third layer next to the leads) and how the finishing lead comes out.
- [ ] Round the corners where the leads leave the pack (now 90° kinks) and the 6 mm
  bend of the finishing lead to the cable's real minimum bend radius.
- [ ] Update `check_winding.py` to the new `continuous_winding` arguments.
- [x] Compare the filament, hex build and continuous models with the Hall-probe map of
  coil 6 (Georg's fit, see above).
- [ ] Locate the T coil's crossover and leads (photos, or a scan of the fit over the
  crossover position along the coil) and refit the continuous winding.
- [ ] Settle the 8 % gain independently: a clamp meter on a lead, and one chip in a
  known field.
- [ ] Use the built pack's geometry: Ø19 mm cables on a Ø22 mm spine (pack Ø98 mm),
  not the design's $r$ = 10.8 mm with a spine as wide as a cable (pack Ø108 mm).
- [ ] Current configurations 1 and 2.
- [ ] Parameter sweeps: layer directions, start slots, $t_0$, bend length,
  twist, lead angle and length.
- [ ] Check the parameters against the build photos.
- [ ] If the effect matters: elastic relaxation (`elastic_finite_build/`).
