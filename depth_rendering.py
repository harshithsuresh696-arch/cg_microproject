"""
Comparative 3D Depth Rendering System
Z-Buffer  vs  Painter's Algorithm
=====================================

Both algorithms are rendered by the SAME software rasterizer (NumPy), so the
only difference between the two panels is how hidden surfaces are resolved:

  * Z-Buffer     : per-pixel depth test against a stored depth buffer.
  * Painter's    : sort triangles far -> near, then just overwrite pixels.

Run:
    pip install pygame numpy matplotlib
    python depth_rendering.py                 # interactive comparison window
    python depth_rendering.py --benchmark     # console table + CSV + chart (for the poster)
    python depth_rendering.py --scene 2       # start on a given scene (1-3)

Controls (interactive window):
    1 / 2 / 3     switch scene  (1 intersecting cubes, 2 cyclic overlap, 3 stress test)
    SPACE         pause / resume auto-rotation
    LEFT / RIGHT  rotate scene (yaw)          UP / DOWN   tilt scene (pitch)
    + / -         more / fewer cubes in the stress test
    C             toggle back-face culling
    S             Painter's sort key: centroid depth <-> farthest-vertex depth
    D             show the Z-buffer as a grayscale depth map (left panel)
    P             save a screenshot (depth_render_XXXX.png) - handy for the A3 poster
    ESC / Q       quit
"""

import argparse
import csv
import itertools
import math
import os
import sys
import time
from collections import deque

import numpy as np

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
W, H = 320, 240            # internal render resolution of each panel
SCALE = 2                  # on-screen upscale factor
FOCAL = 300.0              # perspective focal length (pixels)
CAM_DIST = 7.0             # camera distance from scene origin
NEAR = 0.2
BG = np.array([18, 20, 28], dtype=np.uint8)
LIGHT = np.array([-0.3, 0.5, -0.8])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


_CUBE_V = np.array(list(itertools.product([-1, 1], repeat=3)), dtype=float) * 0.5
_CUBE_QUADS = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
               [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]


def make_cube(center, size, euler, color):
    """Return (12,3,3) triangles with outward-facing winding, plus (12,3) colors."""
    tris = []
    for a, b, c, d in _CUBE_QUADS:
        tris.append([_CUBE_V[a], _CUBE_V[b], _CUBE_V[c]])
        tris.append([_CUBE_V[a], _CUBE_V[c], _CUBE_V[d]])
    tris = np.array(tris)
    # make every triangle wind so its normal points away from the cube centre
    for i in range(len(tris)):
        p, q, r = tris[i].copy()
        n = np.cross(q - p, r - p)
        if np.dot(n, (p + q + r) / 3.0) < 0:
            tris[i] = np.array([p, r, q])
    R = rot_z(euler[2]) @ rot_y(euler[1]) @ rot_x(euler[0])
    tris = (tris * size) @ R.T + np.asarray(center, float)
    cols = np.tile(np.asarray(color, float), (len(tris), 1))
    return tris, cols


def combine(parts):
    return (np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]))


# --------------------------------------------------------------------------- #
# Scenes   ->  (triangles, colors, double_sided)
# --------------------------------------------------------------------------- #
def scene_intersecting_cubes():
    """Two cubes that pass THROUGH each other. Painter's cannot sort this."""
    a = make_cube((-0.55, 0.0, 0.0), 2.0, (0.5, 0.6, 0.2), (220, 90, 80))
    b = make_cube((0.55, 0.25, 0.25), 2.0, (0.2, 0.9, 0.7), (80, 150, 230))
    tris, cols = combine([a, b])
    return tris, cols, False


def scene_cyclic_overlap():
    """Three bars, each in front of the next -> a true depth cycle (no valid order)."""
    r, L, w, s = 0.8, 1.9, 0.6, 0.38
    colors = [(235, 90, 90), (90, 210, 120), (90, 140, 240)]
    tris, cols = [], []
    for i in range(3):
        R = rot_z(math.radians(120 * i))
        corners = []
        for u, v in [(-L, -w), (L, -w), (L, w), (-L, w)]:
            p = R @ np.array([r + v, u, 0.0])
            p[2] = s * u            # tilt the bar: one end near, one end far
            corners.append(p)
        a, b, c, d = corners
        tris += [[a, b, c], [a, c, d]]
        cols += [colors[i]] * 2
    return np.array(tris), np.array(cols, float), True


def scene_stress(n_cubes, seed=7):
    """Many randomly placed cubes - used to compare performance as triangle count grows."""
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(n_cubes):
        center = rng.uniform([-3.2, -2.2, -2.2], [3.2, 2.2, 2.2])
        size = rng.uniform(0.6, 1.2)
        euler = rng.uniform(0, math.tau, 3)
        color = rng.integers(70, 245, 3)
        parts.append(make_cube(center, size, euler, color))
    tris, cols = combine(parts)
    return tris, cols, False


# --------------------------------------------------------------------------- #
# Geometry pipeline (shared by both algorithms)
# --------------------------------------------------------------------------- #
def transform(tris, cols, yaw, pitch, double_sided, cull):
    R = rot_x(pitch) @ rot_y(yaw)
    v = tris @ R.T
    v[..., 2] += CAM_DIST

    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    facing = np.einsum("ij,ij->i", n, v[:, 0])          # < 0  => faces the camera
    keep = np.all(v[..., 2] > NEAR, axis=1)
    if cull and not double_sided:
        keep &= facing < 0
    n = np.where((facing > 0)[:, None], -n, n)           # light the visible side

    v, n, cols = v[keep], n[keep], cols[keep]
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    shade = 0.30 + 0.70 * np.clip(n @ LIGHT, 0, 1)
    colors = np.clip(cols * shade[:, None], 0, 255).astype(np.uint8)

    z = v[..., 2]
    scr = np.empty((len(v), 3, 2))
    scr[..., 0] = W / 2 + FOCAL * v[..., 0] / z
    scr[..., 1] = H / 2 - FOCAL * v[..., 1] / z
    return {"scr": scr, "inv_z": 1.0 / z, "z_mean": z.mean(axis=1),
            "z_max": z.max(axis=1), "col": colors, "n": len(v)}


# --------------------------------------------------------------------------- #
# Rasterizer
# --------------------------------------------------------------------------- #
def raster_triangle(fb, zb, p, iz, color, stats):
    """Fill one triangle.  zb is None for the Painter's algorithm (no depth test)."""
    x0 = max(int(math.floor(p[:, 0].min())), 0)
    x1 = min(int(math.ceil(p[:, 0].max())), W - 1)
    y0 = max(int(math.floor(p[:, 1].min())), 0)
    y1 = min(int(math.ceil(p[:, 1].max())), H - 1)
    if x0 > x1 or y0 > y1:
        return
    (ax, ay), (bx, by), (cx, cy) = p
    den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(den) < 1e-9:
        return

    px, py = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
    w0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
    w1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
    w2 = 1.0 - w0 - w1
    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)

    window = (slice(y0, y1 + 1), slice(x0, x1 + 1))
    if zb is None:                                    # Painter's: blind overwrite
        fb[window][inside] = color
        stats["writes"] += int(inside.sum())
    else:                                             # Z-buffer: depth test
        z = w0 * iz[0] + w1 * iz[1] + w2 * iz[2]      # interpolate 1/z (perspective-correct)
        zwin = zb[window]
        visible = inside & (z > zwin)                 # larger 1/z == closer
        zwin[visible] = z[visible]
        fb[window][visible] = color
        stats["tests"] += int(inside.sum())
        stats["writes"] += int(visible.sum())


def render_zbuffer(g):
    fb = np.empty((H, W, 3), np.uint8)
    fb[:] = BG
    zb = np.zeros((H, W), np.float32)                 # 0 == infinitely far
    stats = {"writes": 0, "tests": 0, "sort_ms": 0.0, "mem_kb": zb.nbytes / 1024}
    t = time.perf_counter()
    for i in range(g["n"]):
        raster_triangle(fb, zb, g["scr"][i], g["inv_z"][i], g["col"][i], stats)
    stats["ms"] = (time.perf_counter() - t) * 1000
    return fb, zb, stats


def render_painter(g, key="mean"):
    fb = np.empty((H, W, 3), np.uint8)
    fb[:] = BG
    stats = {"writes": 0, "tests": 0, "mem_kb": 0.0}
    t = time.perf_counter()
    depth = g["z_mean"] if key == "mean" else g["z_max"]
    order = np.argsort(-depth)                        # farthest first
    stats["sort_ms"] = (time.perf_counter() - t) * 1000
    for i in order:
        raster_triangle(fb, None, g["scr"][i], None, g["col"][i], stats)
    stats["ms"] = (time.perf_counter() - t) * 1000
    return fb, stats


def mismatch_percent(fb_painter, fb_zbuf):
    """Share of pixels where Painter's differs from the (correct) Z-buffer image."""
    return float((fb_painter != fb_zbuf).any(axis=2).mean() * 100)


def overdraw(fb, stats):
    covered = int((fb != BG).any(axis=2).sum())
    return stats["writes"] / covered if covered else 0.0


# --------------------------------------------------------------------------- #
# Benchmark mode (no window needed)
# --------------------------------------------------------------------------- #
def run_benchmark(frames=8):
    counts = [2, 5, 10, 20, 40, 60, 80]
    rows = []
    print(f"\nRender resolution {W}x{H}, {frames} frames per row, back-face culling ON\n")
    print(f"{'cubes':>5} {'tris':>5} | {'Z-buf ms':>9} {'Paint ms':>9} {'(sort)':>7} | "
          f"{'Z-buf ovrdr':>11} {'Paint ovrdr':>11} | {'Painter err %':>13}")
    print("-" * 88)
    for n in counts:
        tris, cols, ds = scene_stress(n)
        zt, pt, ps, zo, po, er, ntri = [], [], [], [], [], [], 0
        for f in range(frames):
            g = transform(tris, cols, f * 0.45, 0.35, ds, True)
            ntri = g["n"]
            fz, _, sz = render_zbuffer(g)
            fp, sp = render_painter(g)
            zt.append(sz["ms"]); pt.append(sp["ms"]); ps.append(sp["sort_ms"])
            zo.append(overdraw(fz, sz)); po.append(overdraw(fp, sp))
            er.append(mismatch_percent(fp, fz))
        row = dict(cubes=n, triangles=ntri, zbuffer_ms=np.mean(zt), painter_ms=np.mean(pt),
                   painter_sort_ms=np.mean(ps), zbuffer_overdraw=np.mean(zo),
                   painter_overdraw=np.mean(po), painter_error_pct=np.mean(er))
        rows.append(row)
        print(f"{n:>5} {ntri:>5} | {row['zbuffer_ms']:>9.1f} {row['painter_ms']:>9.1f} "
              f"{row['painter_sort_ms']:>7.2f} | {row['zbuffer_overdraw']:>11.2f} "
              f"{row['painter_overdraw']:>11.2f} | {row['painter_error_pct']:>13.2f}")

    with open("benchmark_results.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("\nSaved benchmark_results.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib to also get benchmark_chart.png)")
        return
    x = [r["triangles"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(x, [r["zbuffer_ms"] for r in rows], "o-", label="Z-Buffer")
    ax[0].plot(x, [r["painter_ms"] for r in rows], "s-", label="Painter's (incl. sort)")
    ax[0].set(xlabel="Triangles drawn", ylabel="Frame time (ms)", title="Performance")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(x, [r["zbuffer_overdraw"] for r in rows], "o-", label="Z-Buffer")
    ax[1].plot(x, [r["painter_overdraw"] for r in rows], "s-", label="Painter's")
    ax[1].set(xlabel="Triangles drawn", ylabel="Pixel writes / visible pixel",
              title="Overdraw (wasted work)")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.suptitle("Z-Buffer vs Painter's Algorithm")
    fig.tight_layout()
    fig.savefig("benchmark_chart.png", dpi=150)
    print("Saved benchmark_chart.png")


# --------------------------------------------------------------------------- #
# Interactive application
# --------------------------------------------------------------------------- #
SCENE_NAMES = {1: "Intersecting cubes", 2: "Cyclic overlap", 3: "Stress test"}


def run_gui(start_scene=1, n_cubes=20, max_frames=None):
    import pygame

    pygame.init()
    panel_w, panel_h = W * SCALE, H * SCALE
    hud_h = 200
    screen = pygame.display.set_mode((panel_w * 2, panel_h + hud_h + 56))
    pygame.display.set_caption("Comparative 3D Depth Rendering - Z-Buffer vs Painter's")
    font = pygame.font.Font(None, 22)
    big = pygame.font.Font(None, 30)
    clock = pygame.time.Clock()

    state = dict(scene=start_scene, cubes=n_cubes, yaw=0.6, pitch=0.35, paused=False,
                 cull=True, key="mean", depth_view=False)
    cache = {}

    def get_scene():
        k = (state["scene"], state["cubes"])
        if cache.get("k") != k:
            if state["scene"] == 1:
                cache["data"] = scene_intersecting_cubes()
            elif state["scene"] == 2:
                cache["data"] = scene_cyclic_overlap()
            else:
                cache["data"] = scene_stress(state["cubes"])
            cache["k"] = k
        return cache["data"]

    hist_z, hist_p = deque(maxlen=160), deque(maxlen=160)
    ema = {"z": None, "p": None}
    shot_id, frame_no, running = 0, 0, True

    def to_surface(img):
        s = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
        return pygame.transform.scale(s, (panel_w, panel_h))

    def text(msg, pos, color=(225, 228, 235), f=font):
        screen.blit(f.render(msg, True, color), pos)

    def graph(x, y, w, h, data, color, vmax):
        pygame.draw.rect(screen, (30, 33, 44), (x, y, w, h))
        if len(data) > 1:
            pts = [(x + i * w / 160, y + h - min(v / vmax, 1) * h) for i, v in enumerate(data)]
            pygame.draw.lines(screen, color, False, pts, 2)

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif k in (pygame.K_1, pygame.K_2, pygame.K_3):
                    state["scene"] = {pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3}[k]
                    state["yaw"], state["pitch"] = 0.0, 0.0 if state["scene"] == 2 else 0.35
                elif k == pygame.K_SPACE:
                    state["paused"] = not state["paused"]
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    state["cubes"] = min(state["cubes"] + 5, 150)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    state["cubes"] = max(state["cubes"] - 5, 1)
                elif k == pygame.K_c:
                    state["cull"] = not state["cull"]
                elif k == pygame.K_s:
                    state["key"] = "max" if state["key"] == "mean" else "mean"
                elif k == pygame.K_d:
                    state["depth_view"] = not state["depth_view"]
                elif k == pygame.K_p:
                    name = f"depth_render_{shot_id:04d}.png"
                    pygame.image.save(screen, name)
                    shot_id += 1
                    print("saved", name)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:  state["yaw"] -= 0.04
        if keys[pygame.K_RIGHT]: state["yaw"] += 0.04
        if keys[pygame.K_UP]:    state["pitch"] -= 0.03
        if keys[pygame.K_DOWN]:  state["pitch"] += 0.03
        if not state["paused"]:
            state["yaw"] += 0.015

        tris, cols, double_sided = get_scene()
        g = transform(tris, cols, state["yaw"], state["pitch"], double_sided, state["cull"])
        fz, zb, sz = render_zbuffer(g)
        fp, sp = render_painter(g, state["key"])
        err = mismatch_percent(fp, fz)

        for name, ms in (("z", sz["ms"]), ("p", sp["ms"])):
            ema[name] = ms if ema[name] is None else 0.9 * ema[name] + 0.1 * ms
        hist_z.append(sz["ms"]); hist_p.append(sp["ms"])

        # ---- draw ----
        screen.fill((12, 13, 18))
        left = fz
        if state["depth_view"]:
            d = zb / (zb.max() + 1e-9)
            d = (d * 255).astype(np.uint8)
            left = np.stack([d, d, d], axis=2)
        screen.blit(to_surface(left), (0, 28))
        screen.blit(to_surface(fp), (panel_w, 28))
        title_l = "Z-BUFFER" + ("  (depth map)" if state["depth_view"] else "")
        text(title_l, (10, 5), (120, 220, 255), big)
        text("PAINTER'S ALGORITHM", (panel_w + 10, 5), (255, 190, 110), big)
        pygame.draw.line(screen, (70, 75, 95), (panel_w, 0), (panel_w, panel_h + 28), 2)

        y = panel_h + 36
        fps_z, fps_p = 1000 / max(ema["z"], 1e-3), 1000 / max(ema["p"], 1e-3)
        text(f"Frame: {sz['ms']:6.1f} ms   ({fps_z:5.1f} FPS)", (10, y))
        text(f"Depth tests: {sz['tests']:,}   Pixel writes: {sz['writes']:,}", (10, y + 22))
        text(f"Overdraw: {overdraw(fz, sz):.2f}x   Extra memory: {sz['mem_kb']:.0f} KB (depth buffer)",
             (10, y + 44))
        text("Sorting: none needed   |   Correct at pixel level", (10, y + 66), (130, 230, 150))
        graph(10, y + 92, 320, 80, hist_z, (120, 220, 255), 100)

        x = panel_w + 10
        text(f"Frame: {sp['ms']:6.1f} ms   ({fps_p:5.1f} FPS)", (x, y))
        text(f"Sort: {sp['sort_ms']:.2f} ms ({'centroid' if state['key']=='mean' else 'farthest-vertex'} key)"
             f"   Pixel writes: {sp['writes']:,}", (x, y + 22))
        text(f"Overdraw: {overdraw(fp, sp):.2f}x   Extra memory: 0 KB (no depth buffer)", (x, y + 44))
        col = (255, 110, 110) if err > 0.05 else (130, 230, 150)
        text(f"Pixels differing from Z-buffer: {err:.2f}%", (x, y + 66), col)
        graph(x, y + 92, 320, 80, hist_p, (255, 190, 110), 100)
        text("graph: 0-100 ms/frame", (x + 330, y + 150), (120, 125, 140))

        info = (f"Scene {state['scene']}: {SCENE_NAMES[state['scene']]}"
                f"   | triangles drawn: {g['n']}"
                f"   | culling: {'ON' if state['cull'] and not double_sided else 'OFF'}"
                f"   | [1-3] scene  [SPACE] pause  [arrows] rotate  [+/-] cubes  "
                f"[C] cull  [S] sort key  [D] depth map  [P] screenshot")
        text(info, (10, panel_h + hud_h + 34), (150, 155, 175))
        pygame.display.flip()
        clock.tick(60)

        frame_no += 1
        if max_frames and frame_no >= max_frames:
            if os.environ.get("DEPTH_SHOT"):
                pygame.image.save(screen, os.environ["DEPTH_SHOT"])
            running = False
    pygame.quit()


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Z-Buffer vs Painter's Algorithm comparison")
    ap.add_argument("--benchmark", action="store_true", help="run headless benchmark")
    ap.add_argument("--scene", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--cubes", type=int, default=20, help="cube count for scene 3")
    ap.add_argument("--frames", type=int, default=None, help="auto-quit after N frames")
    args = ap.parse_args()
    if args.benchmark:
        run_benchmark()
    else:
        run_gui(args.scene, args.cubes, args.frames)