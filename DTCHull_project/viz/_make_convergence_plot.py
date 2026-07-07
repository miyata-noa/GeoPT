import re, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def parse(p):
    t = Path(p).read_text()
    return np.array([(int(e), float(tr), float(r)) for e, tr, r in re.findall(r"Epoch (\d+) Train loss : ([\d.]+)\s*\nrel_err:([\d.]+)", t)])

gd = parse("logs/geopt_dtchull_20260511_085719.log")
sd = parse("logs/transolver_dtchull_scratch_20260511_113944.log")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

ax1.plot(gd[:,0], gd[:,2], label="GeoPT 200ep [with pre-training]", color="C0", linewidth=2)
ax1.plot(sd[:,0], sd[:,2], label="scratch 200ep [Transolver]", color="C3", linewidth=2)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Test rel_err (lower is better)")
ax1.set_title("DTCHull: full convergence curves")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 0.7)

ax2.plot(gd[:,0], gd[:,2], label="GeoPT", color="C0", linewidth=2)
ax2.plot(sd[:,0], sd[:,2], label="scratch", color="C3", linewidth=2)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Test rel_err")
ax2.set_title("Speedup annotation (10-ep continuous stable)")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.10, 0.35)

window = 10
for g_ep in [20, 40, 60, 100]:
    g_rel = float(gd[gd[:,0]==g_ep, 2][0])
    stable = None
    for i in range(len(sd)-window+1):
        if all(sd[i+j,2] < g_rel for j in range(window)):
            stable = int(sd[i,0])
            break
    if stable is None:
        continue
    speedup = stable / g_ep
    ax2.axhline(y=g_rel, color="gray", ls=":", alpha=0.5)
    ax2.plot([g_ep], [g_rel], "o", color="C0", markersize=11, zorder=5)
    ax2.plot([stable], [g_rel], "s", color="C3", markersize=11, zorder=5)
    ax2.annotate(
        f"GeoPT ep{g_ep} = {g_rel:.3f}\nscratch reaches at ep{stable} ({speedup:.1f}x slower)",
        xy=(stable, g_rel), xytext=(stable+8, g_rel+0.015),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )

plt.suptitle("GeoPT vs scratch on DTCHull (epochs=200, ntrain=100, batch_size=1)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("/tmp/dtchull_convergence.png", dpi=150, bbox_inches="tight")
print("Saved /tmp/dtchull_convergence.png")
print(f"GeoPT final rel_err = {gd[-1,2]:.5f}")
print(f"scratch final rel_err = {sd[-1,2]:.5f}")
