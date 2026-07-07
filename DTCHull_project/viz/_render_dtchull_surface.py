"""Render DTCHull surface (col[3] near 0) with pressure colormap.
Inputs from results/<save_name>/{x_raw,y_raw,out_raw}_<id>.npy
Outputs PNG per sample under /tmp/dtchull_render/
"""
import numpy as np
import os
import sys
import pyvista as pv

pv.OFF_SCREEN = True
pv.global_theme.background = "white"
pv.global_theme.window_size = [1200, 900]

# results now live in DTCHull_project/results (this script is in DTCHull_project/viz)
_RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
GEOPT_DIR = os.path.join(_RES, "hull_geopt_transolver_new_data_correct")
SCRATCH_DIR = os.path.join(_RES, "dtchull_transolver_8layers_normal_cond")
OUT = "/tmp/dtchull_render"
os.makedirs(OUT, exist_ok=True)
SURFACE_THRESH = 0.02  # col[3] (distance) <= this -> surface

def load_triple(d, i):
    x = np.load(f"{d}/x_raw_{i}.npy")[0]  # (N, 7)
    y = np.load(f"{d}/y_raw_{i}.npy")[0]  # (N, 4)
    o = np.load(f"{d}/out_raw_{i}.npy")[0]  # (N, 4)
    return x, y, o

def render(points, scalars, title, fname, vmin, vmax, cmap="coolwarm"):
    cloud = pv.PolyData(points)
    cloud["val"] = scalars
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(cloud, scalars="val", cmap=cmap, point_size=4,
                     render_points_as_spheres=True, clim=(vmin, vmax),
                     scalar_bar_args={"title": "Pressure", "fmt": "%.0f"})
    plotter.add_title(title, font_size=10)
    plotter.show_axes()
    plotter.camera_position = [(8, -8, 5), (0.5, 0, 0), (0, 0, 1)]
    plotter.screenshot(fname)
    plotter.close()
    print(f"  saved {fname}")

def process_sample(i):
    print(f"=== sample {i} ===")
    xg, yg, og = load_triple(GEOPT_DIR, i)
    xs, ys, os_ = load_triple(SCRATCH_DIR, i)
    # surface extraction by col[3] near 0
    mask_g = xg[:, 3] < SURFACE_THRESH
    mask_s = xs[:, 3] < SURFACE_THRESH
    print(f"  GeoPT surface points:  {mask_g.sum()}/{len(mask_g)}")
    print(f"  scratch surface points: {mask_s.sum()}/{len(mask_s)}")
    if mask_g.sum() < 100:
        # threshold may be wrong - fall back to using all points
        print(f"  WARN: few surface points; lowering threshold... trying < 0.1")
        mask_g = xg[:, 3] < 0.1
        mask_s = xs[:, 3] < 0.1
        print(f"  GeoPT: {mask_g.sum()}, scratch: {mask_s.sum()}")
    pts_g_pos = xg[mask_g, :3]; pts_s_pos = xs[mask_s, :3]
    p_gt_g = yg[mask_g, 0]; p_gt_s = ys[mask_s, 0]
    p_pred_g = og[mask_g, 0]; p_pred_s = os_[mask_s, 0]
    err_g = p_pred_g - p_gt_g; err_s = p_pred_s - p_gt_s
    # common color scale based on GT range
    vmin = min(p_gt_g.min(), p_gt_s.min())
    vmax = max(p_gt_g.max(), p_gt_s.max())
    err_abs = max(abs(err_g).max(), abs(err_s).max())
    print(f"  pressure range: [{vmin:.1f}, {vmax:.1f}]")
    print(f"  error abs max: {err_abs:.1f}")
    # 6 PNG: GT (shared), pred_geopt, pred_scratch, err_geopt, err_scratch
    render(pts_g_pos, p_gt_g, f"GT (sample {i})", f"{OUT}/sample{i}_gt.png", vmin, vmax)
    render(pts_g_pos, p_pred_g, f"GeoPT pred (sample {i})", f"{OUT}/sample{i}_pred_geopt.png", vmin, vmax)
    render(pts_s_pos, p_pred_s, f"scratch pred (sample {i})", f"{OUT}/sample{i}_pred_scratch.png", vmin, vmax)
    render(pts_g_pos, err_g, f"GeoPT error (sample {i})", f"{OUT}/sample{i}_err_geopt.png", -err_abs, err_abs)
    render(pts_s_pos, err_s, f"scratch error (sample {i})", f"{OUT}/sample{i}_err_scratch.png", -err_abs, err_abs)

if __name__ == "__main__":
    samples = [1, 2, 3] if len(sys.argv) < 2 else [int(x) for x in sys.argv[1:]]
    for i in samples:
        try:
            process_sample(i)
        except Exception as e:
            print(f"sample {i} failed: {e}")
