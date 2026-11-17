"""
Open3D point cloud viewer of step 6 (started by stripe_depth_app.py "Open in Open3D"; also usable on its own with
any .ply / .pcd / .xyz file).

The clouds are in the rectified left camera frame: x right, y down, z forward (metres). The viewer starts looking from
the camera position, so the first view matches the depth image.

Usage
    python cloud_viewer.py                                # newest .ply under output/
    python cloud_viewer.py output/cw_20us/*.ply           # several files: N / P switch between them
    python cloud_viewer.py a.ply b.ply --overlay          # all files at once, one colour per file
    python cloud_viewer.py a.ply --color plane --voxel 0.002 --denoise
    python cloud_viewer.py a.ply --screenshot view.png    # render once without a window and exit

Keys (in addition to Open3D's own: mouse to rotate / pan / zoom, - and = for point size, H for its help)
    C       next colour mode: file colours -> depth (Z) -> height (y) -> distance from plane
    N / P   next / previous file (when several are given and not overlaid)
    A       show / hide the camera axes and frustum
    R       reset to the camera's viewpoint
    K       save a screenshot next to the file (<name>_view.png)
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit(f"open3d is not installed in {sys.executable}: pip install open3d   (or: uv pip install open3d)")

DEFAULT_DIR = Path(__file__).resolve().parent / "output"
MODES = ("file", "depth", "height", "plane")
OVERLAY_COLOURS = np.array([[0.12, 0.47, 0.71], [1.0, 0.5, 0.05], [0.17, 0.63, 0.17], [0.84, 0.15, 0.16],
                            [0.58, 0.40, 0.74], [0.55, 0.34, 0.29], [0.89, 0.47, 0.76], [0.5, 0.5, 0.5]])


# ----------------------------------------------------------------------------------------------------------------------
# Colour maps (no matplotlib needed)
# ----------------------------------------------------------------------------------------------------------------------
_VIRIDIS = np.array([[0.267, 0.005, 0.329], [0.283, 0.141, 0.458], [0.254, 0.265, 0.530], [0.207, 0.372, 0.553],
                     [0.164, 0.471, 0.558], [0.128, 0.567, 0.551], [0.135, 0.659, 0.518], [0.267, 0.749, 0.441],
                     [0.478, 0.821, 0.318], [0.741, 0.873, 0.150], [0.993, 0.906, 0.144]])
_DIVERGING = np.array([[0.02, 0.19, 0.38], [0.26, 0.58, 0.76], [0.97, 0.97, 0.97], [0.84, 0.38, 0.30],
                       [0.40, 0.0, 0.12]])


def colormap(v: np.ndarray, lo: float, hi: float, table: np.ndarray) -> np.ndarray:
    u = np.clip((v - lo) / max(hi - lo, 1e-12), 0, 1) * (len(table) - 1)
    i = np.minimum(u.astype(int), len(table) - 2)
    f = (u - i)[:, None]
    return table[i] * (1 - f) + table[i + 1] * f


# ----------------------------------------------------------------------------------------------------------------------
# Loading and analysis
# ----------------------------------------------------------------------------------------------------------------------
def load(path: Path, voxel: float, denoise: bool) -> o3d.geometry.PointCloud:
    pc = o3d.io.read_point_cloud(str(path))
    if pc.is_empty():
        raise ValueError(f"no points in {path}")
    n0 = len(pc.points)
    if voxel > 0:
        pc = pc.voxel_down_sample(voxel)
    if denoise:
        pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    p = np.asarray(pc.points)
    z = p[:, 2]
    print(f"{path.name}: {n0:,} points" + (f" -> {len(p):,} after filtering" if len(p) != n0 else "")
          + f", Z {np.percentile(z, 1) * 1e3:.0f}..{np.percentile(z, 99) * 1e3:.0f} mm (1-99 %),"
          f" median {np.median(z) * 1e3:.0f} mm")
    return pc


def plane_distance(pc: o3d.geometry.PointCloud, thr: float = 0.01):
    """Signed distance (m) of every point to a RANSAC plane, with the normal pointing away from the camera."""
    _, inl = pc.segment_plane(distance_threshold=thr, ransac_n=3, num_iterations=1000)
    p = np.asarray(pc.points)
    for _ in range(3):  # Least-squares refinement on the inliers
        q = p[inl]
        c0 = q.mean(0)
        n = np.linalg.svd(q - c0, full_matrices=False)[2][-1]
        inl = np.abs((p - c0) @ n) < thr
    if n[2] < 0:
        n = -n
    d = -n @ c0
    e = p @ n + d
    tilt = np.degrees(np.arccos(abs(n[2])))
    mad = 1.4826 * np.median(np.abs(e - np.median(e)))
    print(f"   plane: normal ({n[0]:+.3f}, {n[1]:+.3f}, {n[2]:+.3f}), {tilt:.1f} deg from facing the camera,"
          f" distance {-d * 1e3:.0f} mm; points within {thr * 1e3:.0f} mm: {inl.mean():.1%},"
          f" robust spread (1.48 MAD) {mad * 1e3:.2f} mm")
    return e


class CloudSet:
    """The loaded clouds with their original colours and the per-mode colours computed on demand."""

    def __init__(self, paths, voxel, denoise):
        self.paths = paths
        self.clouds = [load(p, voxel, denoise) for p in paths]
        self.original = [np.asarray(c.colors).copy() if c.has_colors() else None for c in self.clouds]
        self._plane = {}

    def colours(self, i: int, mode: str, overlay: bool) -> np.ndarray:
        pc = self.clouds[i]
        p = np.asarray(pc.points)
        if mode == "file":
            if overlay:
                return np.tile(OVERLAY_COLOURS[i % len(OVERLAY_COLOURS)], (len(p), 1))
            if self.original[i] is not None:
                return self.original[i]
            mode = "depth"
        if mode == "depth":
            lo, hi = np.percentile(p[:, 2], [1, 99])
            return colormap(p[:, 2], hi, lo, _VIRIDIS)  # near = yellow, far = purple
        if mode == "height":
            lo, hi = np.percentile(p[:, 1], [1, 99])
            return colormap(p[:, 1], hi, lo, _VIRIDIS)
        if i not in self._plane:
            print(self.paths[i].name)
            self._plane[i] = plane_distance(pc)
        return colormap(self._plane[i], -0.01, 0.01, _DIVERGING)  # blue = in front of the plane, red = behind, +-10 mm


def camera_markers(size: float, extent: float):
    """Axes at the camera centre (x red, y green, z blue) and a small frustum looking along +z."""
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    w, h, z = 0.64 * extent, 0.36 * extent, extent
    pts = np.array([[0, 0, 0], [-w, -h, z], [w, -h, z], [w, h, z], [-w, h, z]])
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    fr = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts), o3d.utility.Vector2iVector(lines))
    fr.paint_uniform_color([0.4, 0.4, 0.4])
    return [axes, fr]


# ----------------------------------------------------------------------------------------------------------------------
# Viewer
# ----------------------------------------------------------------------------------------------------------------------
class Viewer:
    def __init__(self, cs: CloudSet, mode: str, overlay: bool, point_size: float, show_camera: bool):
        self.cs, self.mode, self.overlay = cs, mode, overlay
        self.point_size, self.show_camera = point_size, show_camera
        self.current = 0
        allp = np.concatenate([np.asarray(c.points) for c in cs.clouds])
        self.centre = np.median(allp, 0)
        dist = float(np.linalg.norm(self.centre))
        self.markers = camera_markers(0.03 * dist, 0.08 * dist)

    def shown(self):
        return list(range(len(self.cs.clouds))) if self.overlay else [self.current]

    def paint(self):
        for i in self.shown():
            self.cs.clouds[i].colors = o3d.utility.Vector3dVector(self.cs.colours(i, self.mode, self.overlay))

    def title(self):
        names = "  +  ".join(self.cs.paths[i].name for i in self.shown())
        return f"{names}  [colour: {self.mode}]"

    def reset_view(self, vis):
        ctr = vis.get_view_control()
        ctr.set_lookat(self.centre)
        ctr.set_front([0.0, 0.0, -1.0])  # Looking along +z from the camera side
        ctr.set_up([0.0, -1.0, 0.0])  # y points down in the camera frame
        ctr.set_zoom(0.6)
        return False

    def rebuild(self, vis, keep_view=True):
        params = vis.get_view_control().convert_to_pinhole_camera_parameters() if keep_view else None
        vis.clear_geometries()
        self.paint()
        for i in self.shown():
            vis.add_geometry(self.cs.clouds[i], reset_bounding_box=not keep_view)
        if self.show_camera:
            for m in self.markers:  # Added without enlarging the bounding box, so the view stays on the cloud
                vis.add_geometry(m, reset_bounding_box=False)
        if params is not None:
            vis.get_view_control().convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        print(self.title())
        return True

    def setup(self, vis):
        opt = vis.get_render_option()
        opt.point_size = self.point_size
        opt.background_color = np.array([1.0, 1.0, 1.0])
        self.rebuild(vis, keep_view=False)
        vis.poll_events()  # The view control only takes effect after the first render
        vis.update_renderer()
        self.reset_view(vis)

    def run(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=f"Point cloud - {self.cs.paths[self.current].name}", width=1400, height=850)
        self.setup(vis)

        def next_mode(v):
            self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
            return self.rebuild(v)

        def step(k):
            def f(v):
                if self.overlay or len(self.cs.clouds) == 1:
                    return False
                self.current = (self.current + k) % len(self.cs.clouds)
                return self.rebuild(v)
            return f

        def toggle_camera(v):
            self.show_camera = not self.show_camera
            return self.rebuild(v)

        def screenshot(v):
            out = self.cs.paths[self.current].with_name(self.cs.paths[self.current].stem + "_view.png")
            v.capture_screen_image(str(out), do_render=True)
            print(f"saved {out}")
            return False

        for key, fn in (("C", next_mode), ("N", step(1)), ("P", step(-1)), ("A", toggle_camera),
                        ("R", self.reset_view), ("K", screenshot)):
            vis.register_key_callback(ord(key), fn)
        print("Keys: C colour mode, N/P next/previous file, A camera axes, R reset view, K screenshot, Q quit")
        vis.run()
        vis.destroy_window()

    def screenshot(self, out: Path):
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1400, height=850, visible=False)
        self.setup(vis)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(out), do_render=True)
        vis.destroy_window()
        print(f"saved {out}")


def resolve(args) -> list:
    if not args:
        files = sorted(DEFAULT_DIR.rglob("*.ply"), key=lambda p: p.stat().st_mtime)
        if not files:
            sys.exit(f"no .ply files in {DEFAULT_DIR}; pass a file name")
        return [files[-1]]
    out = []
    for a in args:  # Expand globs (Windows shells do not) and folders
        p = Path(a)
        if p.is_dir():
            out += sorted(p.glob("*.ply"))
        else:
            out += [Path(g) for g in sorted(glob.glob(a))] or [p]
    missing = [p for p in out if not p.exists()]
    if missing:
        sys.exit("not found: " + ", ".join(map(str, missing)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help=".ply/.pcd files, globs or folders (default: newest under output/)")
    ap.add_argument("--color", choices=MODES, default="file", help="initial colour mode (default: colours in the file)")
    ap.add_argument("--overlay", action="store_true", help="show all files together, one colour per file")
    ap.add_argument("--voxel", type=float, default=0.0, help="voxel down-sampling size in m (e.g. 0.002)")
    ap.add_argument("--denoise", action="store_true", help="remove statistical outliers (20 neighbours, 2 sigma)")
    ap.add_argument("--point-size", type=float, default=2.0)
    ap.add_argument("--no-camera", action="store_true", help="hide the camera axes and frustum")
    ap.add_argument("--screenshot", type=Path, help="render to this PNG without opening a window, then exit")
    a = ap.parse_args()

    cs = CloudSet(resolve(a.files), a.voxel, a.denoise)
    v = Viewer(cs, a.color, a.overlay, a.point_size, not a.no_camera)
    if a.screenshot:
        v.screenshot(a.screenshot)
    else:
        v.run()


if __name__ == "__main__":
    main()
