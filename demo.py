"""
Python demo for Sparse Projection Pursuit Analysis (SPPA).

Mirrors matlab/demo.m — applies SPPA to the Salmon NMR blood plasma dataset + adds a comparison with PCA, t-SNE, and UMAP.

Data: NMR spectra of salmon blood plasma (74 samples, 1117 variables, 5 classes,
      ~15 samples per class).

Requirements
------------
    pip install scipy matplotlib scikit-learn  # core dependencies
    pip install umap-learn                     # optional, for UMAP comparison
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt

from sppa import sppa

# ---------------------------------------------------------------------------
# Load Salmon.mat
# ---------------------------------------------------------------------------

MAT_PATH = pathlib.Path(__file__).parent / "matlab" / "Salmon.mat"
COMPARE = True  # also plot PCA and UMAP scores for comparison

def _load_mat(path: pathlib.Path) -> dict:
    """Load a .mat file using scipy.io."""
    import scipy.io
    raw = scipy.io.loadmat(str(path))
    return {
        "X": np.asarray(raw["X"], dtype=float),
        "class": np.asarray(raw["class"], dtype=int).ravel(),
        "chemshift": np.asarray(raw["chemshift"], dtype=float).ravel(),
    }


data = _load_mat(MAT_PATH)
X          = data["X"]           # (74, 1117)  rows=samples, cols=NMR variables
classes    = data["class"]       # (74,)        integer labels 1–5
chemshift  = data["chemshift"]   # (1117,)      ppm axis

print(f"X shape      : {X.shape}")
print(f"Classes      : {np.unique(classes)}  ({len(classes)} samples)")
print(f"Chemshift    : {chemshift.min():.2f} – {chemshift.max():.2f} ppm")

# ---------------------------------------------------------------------------
# Run SPPA  (dim=2, nvars=5, multivariate kurtosis — same as demo.m)
# ---------------------------------------------------------------------------

print("\nRunning SPPA (dim=2, nvars=5, meth='mul') …")
T, V, Var, kurt = sppa(
    X,
    dim=2,
    nvars=5,
    meth="mul",   # multivariate kurtosis (matches demo.m)
)

print(f"\nSelected variables (0-based) : {Var}")
print(f"Selected chemshifts (ppm)    : {chemshift[Var.ravel()]}")
print(f"Kurtosis                     : {kurt}")

# ---------------------------------------------------------------------------
# Plot — colour-coded scatter of Score 1 vs Score 2  (mirrors demo.m figure)
# ---------------------------------------------------------------------------

# Colours matching MATLAB's  colorvec = 'rgbyk'
COLORS = ["tab:red", "tab:green", "tab:blue", "tab:olive", "black"]
CLASS_LABELS = [f"Class {i}" for i in range(1, 6)]

fig, ax = plt.subplots(figsize=(7, 6))

for cls_idx, (color, label) in enumerate(zip(COLORS, CLASS_LABELS), start=1):
    mask = classes == cls_idx
    ax.scatter(
        T[mask, 0], T[mask, 1],
        facecolors=color,
        edgecolors="none",
        alpha=0.75,
        s=60,
        label=label,
        zorder=3,
    )

ax.set_xlabel("Score 1", fontsize=14, fontweight="bold")
ax.set_ylabel("Score 2", fontsize=14, fontweight="bold")
ax.set_title("SPPA — Salmon NMR (dim=2, nvars=5, meth='mul')", fontsize=13)
ax.tick_params(width=2, labelsize=11)
for spine in ax.spines.values():
    spine.set_linewidth(2)
ax.legend(loc="best", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.4)

lim = max(abs(T[:, 0]).max(), abs(T[:, 1]).max()) * 1.1
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)

plt.tight_layout()

# ---------------------------------------------------------------------------
# Plot spectra with selected variables highlighted per dimension
# ---------------------------------------------------------------------------

dimvar = Var.shape[0]  # 1 for meth='mul', dim for meth='uni'
dim_labels = (
    ["Multivariate"] if dimvar == 1
    else [f"Dimension {i + 1}" for i in range(dimvar)]
)

mean_spectrum = X.mean(axis=0)
HIGHLIGHT_COLORS = ["tab:red", "tab:green", "tab:blue"]

fig_spec, axes_spec = plt.subplots(dimvar, 1, figsize=(12, 5), sharex=True)
if dimvar == 1:
    axes_spec = [axes_spec]

for d, (ax_s, dlabel) in enumerate(zip(axes_spec, dim_labels)):
    ax_s.plot(chemshift, mean_spectrum, color="steelblue", linewidth=0.8, label="Mean spectrum")
    sel_vars = Var[d]
    for i, idx in enumerate(sel_vars):
        ax_s.axvline(
            chemshift[idx],
            color=HIGHLIGHT_COLORS[d % len(HIGHLIGHT_COLORS)],
            linewidth=1.5,
            alpha=0.8,
            label=f"Selected vars ({dlabel})" if i == 0 else None,
        )
    ax_s.set_ylabel("Intensity", fontsize=12)
    ax_s.set_title(f"NMR Spectra — Selected Variables ({dlabel})", fontsize=12)
    ax_s.legend(loc="upper right", fontsize=10)
    ax_s.grid(True, linestyle="--", alpha=0.3)

axes_spec[-1].set_xlabel("Chemical shift (ppm)", fontsize=12)
fig_spec.tight_layout()

# ---------------------------------------------------------------------------
# Optional: comparison with PCA, t-SNE, UMAP
# ---------------------------------------------------------------------------

if COMPARE:
    from sklearn.decomposition import PCA

    X_mc = X - X.mean(axis=0)
    T_pca = PCA(n_components=2).fit_transform(X_mc)

    T_umap = None
    try:
        from umap import UMAP
        print("\nFitting UMAP …")
        T_umap = UMAP(n_components=2, random_state=42).fit_transform(X)
    except ImportError:
        print("umap-learn not installed — skipping UMAP.  pip install umap-learn")

    from sklearn.manifold import TSNE
    print("Fitting t-SNE …")
    T_tsne = TSNE(n_components=2, random_state=42).fit_transform(X)

    methods = [("SPPA", T, True, "PP"), ("PCA", T_pca, True, "PC"),
               ("t-SNE", T_tsne, False, "t-SNE")]
    if T_umap is not None:
        methods.append(("UMAP", T_umap, False, "UMAP"))

    fig_cmp, axes_cmp = plt.subplots(2, 2, figsize=(7, 6))
    axes_flat = axes_cmp.flatten()

    def _scatter_scores(ax_c, scores, title, equal_axes=True, prefix="Component"):
        for cls_idx, (color, label) in enumerate(zip(COLORS, CLASS_LABELS), start=1):
            mask = classes == cls_idx
            ax_c.scatter(
                scores[mask, 0], scores[mask, 1],
                facecolors=color, edgecolors="none",
                alpha=0.75, s=60, label=label, zorder=3,
            )
        if equal_axes:
            lim = max(abs(scores[:, 0]).max(), abs(scores[:, 1]).max()) * 1.1
            ax_c.set_xlim(-lim, lim)
            ax_c.set_ylim(-lim, lim)
        else:
            pad_x = (scores[:, 0].max() - scores[:, 0].min()) * 0.05
            pad_y = (scores[:, 1].max() - scores[:, 1].min()) * 0.05
            ax_c.set_xlim(scores[:, 0].min() - pad_x, scores[:, 0].max() + pad_x)
            ax_c.set_ylim(scores[:, 1].min() - pad_y, scores[:, 1].max() + pad_y)
        ax_c.set_xlabel(f"{prefix}1", fontsize=12, fontweight="bold")
        ax_c.set_ylabel(f"{prefix}2", fontsize=12, fontweight="bold")
        ax_c.set_title(title, fontsize=13)
        ax_c.tick_params(width=2, labelsize=11)
        for spine in ax_c.spines.values():
            spine.set_linewidth(2)
        ax_c.grid(True, linestyle="--", alpha=0.4)

    for ax_c, (name, scores, eq, prefix) in zip(axes_flat, methods):
        _scatter_scores(ax_c, scores, name, equal_axes=eq, prefix=prefix)

    # single shared legend in the figure
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig_cmp.legend(handles, labels, loc="lower center", ncol=len(CLASS_LABELS),
                   fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.0))

    fig_cmp.suptitle("Score comparison — Salmon NMR", fontsize=14)
    fig_cmp.tight_layout(rect=[0, 0.05, 1, 1])

plt.show()
