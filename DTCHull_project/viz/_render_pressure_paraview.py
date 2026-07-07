"""Render DTCHull surface pressure distribution with ParaView (headless pvbatch/pvpython).

For each test sample it builds a .vtp point cloud (surface only) carrying:
  - p_gt   : ground-truth pressure (y_raw col0)
  - p_pred : model-predicted pressure (out_raw col0)
  - p_err  : p_pred - p_gt
Then renders PNGs colored by each field with a shared color scale.

Data source: results/<save_name>/{x_raw,y_raw,out_raw}_<id>.npy   (shape (1,N,C))
  x cols: 0-2 xyz, 3 distance-to-surface, 4-6 normal
  y/out cols: 0 pressure, 1-3 shear/velocity components

Outputs:
  results/paraview_vtp/*.vtp         (open these in ParaView GUI)
  results/paraview_png/*.png         (rendered figures)
"""
import os, sys
import numpy as np
import vtk
from paraview.simple import *

BASE = os.path.dirname(os.path.abspath(__file__))
# results now live in DTCHull_project/results (this script is in DTCHull_project/viz)
_RES = os.path.join(BASE, "..", "results")
GEOPT_DIR = os.path.join(_RES, "hull_geopt_transolver_new_data_correct")
SCRATCH_DIR = os.path.join(_RES, "dtchull_transolver_8layers_normal_cond")
VTP_OUT = os.path.join(_RES, "paraview_vtp")
PNG_OUT = os.path.join(_RES, "paraview_png")
os.makedirs(VTP_OUT, exist_ok=True)
os.makedirs(PNG_OUT, exist_ok=True)
SURFACE_THRESH = 0.02


def load_triple(d, i):
    x = np.load(f"{d}/x_raw_{i}.npy")[0]
    y = np.load(f"{d}/y_raw_{i}.npy")[0]
    o = np.load(f"{d}/out_raw_{i}.npy")[0]
    return x, y, o


def build_vtp(points, p_gt, p_pred, path):
    pts = vtk.vtkPoints()
    pts.SetData(vtk.util.numpy_support.numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64)))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    # vertex cells so ParaView renders every point
    verts = vtk.vtkCellArray()
    n = points.shape[0]
    ids = np.empty((n, 2), dtype=np.int64)
    ids[:, 0] = 1
    ids[:, 1] = np.arange(n)
    ca = vtk.util.numpy_support.numpy_to_vtkIdTypeArray(ids.ravel())
    verts.SetCells(n, ca)
    poly.SetVerts(verts)
    for name, arr in (("p_gt", p_gt), ("p_pred", p_pred), ("p_err", p_pred - p_gt)):
        va = vtk.util.numpy_support.numpy_to_vtk(np.ascontiguousarray(arr, dtype=np.float64))
        va.SetName(name)
        poly.GetPointData().AddArray(va)
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(path)
    w.SetInputData(poly)
    w.Write()


def render(vtp_path, array, clim, out_png, title, preset="Cool to Warm"):
    src = XMLPolyDataReader(FileName=[vtp_path])
    view = CreateView("RenderView")
    view.ViewSize = [1200, 900]
    try:
        view.UseColorPaletteForBackground = 0
        view.BackgroundColorMode = "Single Color"
    except Exception:
        pass
    view.Background = [1, 1, 1]
    view.OrientationAxesVisibility = 1
    disp = Show(src, view)
    disp.SetRepresentationType("Point Gaussian")
    disp.GaussianRadius = 0.012
    ColorBy(disp, ("POINTS", array))
    lut = GetColorTransferFunction(array)
    lut.ApplyPreset(preset, True)
    lut.RescaleTransferFunction(clim[0], clim[1])
    disp.RescaleTransferFunctionToDataRange(False, True)
    lut.RescaleTransferFunction(clim[0], clim[1])
    bar = GetScalarBar(lut, view)
    bar.Title = title
    bar.ComponentTitle = ""
    bar.TitleColor = [0, 0, 0]
    bar.LabelColor = [0, 0, 0]
    disp.SetScalarBarVisibility(view, True)
    view.ResetCamera()
    cam = GetActiveCamera()
    cam.SetPosition(8, -9, 5)
    cam.SetFocalPoint(0.5, 0, 0)
    cam.SetViewUp(0, 0, 1)
    view.ResetCamera()
    Render(view)
    SaveScreenshot(out_png, view, ImageResolution=[1200, 900])
    Delete(src)
    del src
    Delete(view)
    del view
    print(f"  saved {out_png}")


def process(i):
    print(f"=== sample {i} ===")
    xg, yg, og = load_triple(GEOPT_DIR, i)
    xs, ys, osr = load_triple(SCRATCH_DIR, i)
    mg = xg[:, 3] < SURFACE_THRESH
    ms = xs[:, 3] < SURFACE_THRESH
    if mg.sum() < 100:
        mg = xg[:, 3] < 0.1
        ms = xs[:, 3] < 0.1
    print(f"  surface pts geopt={int(mg.sum())} scratch={int(ms.sum())}")
    pg, pgtg, ppredg = xg[mg, :3], yg[mg, 0], og[mg, 0]
    ps, pgts, ppreds = xs[ms, :3], ys[ms, 0], osr[ms, 0]
    vg = os.path.join(VTP_OUT, f"geopt_sample{i}.vtp")
    vs = os.path.join(VTP_OUT, f"scratch_sample{i}.vtp")
    build_vtp(pg, pgtg, ppredg, vg)
    build_vtp(ps, pgts, ppreds, vs)
    # shared pressure scale from GT; symmetric error scale
    vmin = float(min(pgtg.min(), pgts.min()))
    vmax = float(max(pgtg.max(), pgts.max()))
    eabs = float(max(np.abs(ppredg - pgtg).max(), np.abs(ppreds - pgts).max()))
    print(f"  pressure clim=[{vmin:.1f},{vmax:.1f}] err_abs={eabs:.1f}")
    render(vg, "p_gt", (vmin, vmax), f"{PNG_OUT}/sample{i}_gt.png", "Pressure (GT)")
    render(vg, "p_pred", (vmin, vmax), f"{PNG_OUT}/sample{i}_pred_geopt.png", "Pressure (GeoPT)")
    render(vs, "p_pred", (vmin, vmax), f"{PNG_OUT}/sample{i}_pred_scratch.png", "Pressure (scratch)")
    render(vg, "p_err", (-eabs, eabs), f"{PNG_OUT}/sample{i}_err_geopt.png", "Error (GeoPT)")
    render(vs, "p_err", (-eabs, eabs), f"{PNG_OUT}/sample{i}_err_scratch.png", "Error (scratch)")


if __name__ == "__main__":
    import vtk.util.numpy_support  # noqa
    samples = [1, 2, 3] if len(sys.argv) < 2 else [int(a) for a in sys.argv[1:]]
    for i in samples:
        try:
            process(i)
        except Exception as e:
            print(f"sample {i} FAILED: {e}")
