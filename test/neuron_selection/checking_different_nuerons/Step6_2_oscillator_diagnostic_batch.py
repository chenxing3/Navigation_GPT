"""
Batch classification of neurons in a layer by spatial-tuning type.

Two-pass smoothing strategy
---------------------------
Neurons in this model have a universal fine-scale 'landmark speckle'
that can mask slower spatial structure. To see past it:

  Pass 1 — heavy smoothing (default σ = 10 bins = 100 m).
           Kills the speckle. Place fields, multi-field patterns,
           and border strips survive. Classifies most neurons.

  Pass 2 — light smoothing (default σ = 1 bin = 10 m), only for
           neurons the first pass left Unclassified. At this scale
           the speckle is preserved, so 'landmark speckle' neurons
           get correctly labeled here.

The CSV records `smooth_sigma_used` so you can tell which pass
classified each neuron.

Categories
----------
  1. Place cell           — single localized field
  2. Multi-field cell     — 2-3 localized fields
  3. Border cell          — elevated firing along a wall strip
  4. Landmark speckle     — dense fine-scale peaks, low per-peak amplitude
  5. Unclassified         — no match in either pass
"""

import os
import sys
import glob
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label as ndi_label, maximum_filter
from scipy.signal import fftconvolve
from tqdm import tqdm


# =============================================================================
# Config
# =============================================================================

INPUT_GLOB   = "./ratemaps_v4token_no_generation_16layer_freeze_1m/ratemap_data_neuron_*.npy"
OUTPUT_DIR   = "./ratemaps_v4token_no_generation_16layer_freeze_1m_classification/"
SUMMARY_CSV  = os.path.join(OUTPUT_DIR, "neuron_classification.csv")

# Outer crop (matches the crop you've been using)
CROP_Y = (80, 1000)
CROP_X = (80, 800)
# CROP_Y = (60, 1000)
# CROP_X = (60, 800)


# Additional inner edge strip before scoring
EDGE_CROP_FRAC = 0.05

# Two-pass smoothing — earlier values run first
SMOOTH_SIGMA_PASSES = [30.0, 1.0]

METERS_PER_BIN = 10.0

# Place-cell thresholds
PLACE_MIN_FIELD_SIZE_BINS = 40
PLACE_PEAK_TO_MEDIAN_MIN  = 1.24
PLACE_MAX_N_FIELDS        = 3

# Border thresholds
BORDER_STRIP_FRAC        = 0.10
BORDER_ELEV_RATIO        = 1.5
BORDER_MIN_WALL_COVERAGE = 0.25

# Speckle thresholds
SPECKLE_MIN_PEAK_DENSITY   = 3.0
SPECKLE_MAX_PEAK_TO_MEDIAN = 2.0

# Field-detection defaults
FIELD_PEAK_THRESH_FRAC = 0.5
FIELD_MIN_SIZE_BINS    = 20
PEAK_MIN_DIST_BINS     = 5

# Plotting
SAVE_PLOTS = True


# =============================================================================
# Data container
# =============================================================================

@dataclass
class NeuronResult:
    neuron_id: str
    cell_type: str
    smooth_sigma_used: float
    pass_number: int
    n_fields: int
    peak_to_median: float
    peak_density_per_km2: float
    border_elevation: float
    border_coverage: float
    field_radius_m: float
    autocorr_has_secondary_peak: bool
    notes: str


# =============================================================================
# Ratemap helpers
# =============================================================================

def crop_inner(rm: np.ndarray, inner_frac: float) -> np.ndarray:
    h, w = rm.shape
    d = int(inner_frac * min(h, w))
    if d == 0:
        return rm
    return rm[d:h - d, d:w - d]


def preprocess(rm_raw: np.ndarray, sigma: float) -> np.ndarray:
    """Outer crop → inner crop → Gaussian smoothing."""
    rm = rm_raw[CROP_Y[0]:CROP_Y[1], CROP_X[0]:CROP_X[1]]
    rm = crop_inner(rm, EDGE_CROP_FRAC)
    rm = gaussian_filter(np.nan_to_num(rm, nan=0.0), sigma=sigma)
    return rm


def find_fields(rm: np.ndarray,
                peak_thresh_frac: float = FIELD_PEAK_THRESH_FRAC,
                min_size: int = FIELD_MIN_SIZE_BINS) -> List[np.ndarray]:
    med = np.nanmedian(rm)
    top = np.nanmax(rm) - med
    if top <= 0:
        return []
    thresh = med + peak_thresh_frac * top
    labeled, n = ndi_label(rm > thresh)
    return [labeled == lbl for lbl in range(1, n + 1)
            if (labeled == lbl).sum() >= min_size]


def count_peaks(rm: np.ndarray,
                peak_thresh_frac: float = 0.3,
                min_peak_dist: int = PEAK_MIN_DIST_BINS) -> int:
    med = np.nanmedian(rm)
    top = np.nanmax(rm) - med
    if top <= 0:
        return 0
    thr = med + peak_thresh_frac * top
    foot = np.ones((2 * min_peak_dist + 1, 2 * min_peak_dist + 1))
    local_max = (rm == maximum_filter(rm, footprint=foot))
    return int((local_max & (rm > thr)).sum())


def peak_to_median_ratio(rm: np.ndarray) -> float:
    med = np.nanmedian(rm)
    top = np.nanmax(rm)
    if abs(med) < 1e-6:
        return float("inf") if top > 0 else 0.0
    return float(top / abs(med))


def border_elevation(rm: np.ndarray,
                     strip_frac: float = BORDER_STRIP_FRAC
                     ) -> Tuple[float, float]:
    """Return (max_wall_elevation_ratio, best_wall_coverage)."""
    h, w = rm.shape
    sh = max(1, int(strip_frac * h))
    sw = max(1, int(strip_frac * w))
    rm_shifted = rm - np.nanmin(rm)
    interior = rm_shifted[sh:h - sh, sw:w - sw]
    interior_mean = max(float(np.nanmean(interior)), 1e-6)
    med = np.nanmedian(rm_shifted)
    top = np.nanmax(rm_shifted)
    thr = med + 0.5 * (top - med)

    walls = {
        "top":    rm_shifted[:sh, :],
        "bottom": rm_shifted[h - sh:, :],
        "left":   rm_shifted[:, :sw],
        "right":  rm_shifted[:, w - sw:],
    }
    best_ratio = 0.0
    best_cov = 0.0
    for strip in walls.values():
        ratio = float(np.nanmean(strip)) / interior_mean
        best_ratio = max(best_ratio, ratio)
        cov = float((strip > thr).sum()) / max(strip.size, 1)
        best_cov = max(best_cov, cov)
    return best_ratio, best_cov


def autocorrelogram_radial(rm: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rm_zm = np.nan_to_num(rm - np.nanmean(rm))
    ac = fftconvolve(rm_zm, rm_zm[::-1, ::-1], mode="same")
    c = ac[ac.shape[0] // 2, ac.shape[1] // 2]
    if c > 0:
        ac = ac / c
    h, w = ac.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - cy, xx - cx).astype(int)
    r_max = int(r.max())
    profile = np.zeros(r_max + 1)
    counts = np.zeros(r_max + 1)
    for v, rv in zip(ac.ravel(), r.ravel()):
        profile[rv] += v
        counts[rv] += 1
    counts = np.maximum(counts, 1)
    return ac, np.arange(r_max + 1), profile / counts


def field_halfmax_radius(profile: np.ndarray) -> float:
    below = np.where(profile < 0.5)[0]
    if len(below) == 0:
        return float("nan")
    return float(below[0] * METERS_PER_BIN)


def autocorr_has_secondary_peak(profile: np.ndarray) -> bool:
    """
    True if there's a secondary bump after the first local min, which would
    indicate multi-field or periodic structure (rules out 'simple place').
    """
    if len(profile) < 20:
        return False
    for i in range(5, len(profile) - 5):
        if profile[i] < profile[i - 1] and profile[i] < profile[i + 1]:
            rest = profile[i:]
            return bool(rest.max() - rest.min() > 0.08)
    return False


def smooth_single_peak(profile: np.ndarray) -> bool:
    """Negation of has_secondary_peak — place-cell AC signature."""
    return not autocorr_has_secondary_peak(profile)


# =============================================================================
# Classification at a single smoothing scale
# =============================================================================

def classify_at_scale(rm_raw: np.ndarray,
                      neuron_id: str,
                      sigma: float,
                      pass_num: int) -> NeuronResult:
    rm = preprocess(rm_raw, sigma)
    h, w = rm.shape
    area_km2 = (h * w * METERS_PER_BIN ** 2) / 1e6

    fields = find_fields(rm)
    n_fields = len(fields)
    ptm = peak_to_median_ratio(rm)
    n_peaks = count_peaks(rm, peak_thresh_frac=0.3)
    peak_density = n_peaks / max(area_km2, 1e-6)
    border_ratio, border_cov = border_elevation(rm)

    _, _, profile = autocorrelogram_radial(rm)
    has_secondary = autocorr_has_secondary_peak(profile)
    field_r = field_halfmax_radius(profile)

    cell_type = "Unclassified"
    notes = ""

    print("n_fields: ", n_fields)
    print("peak_to_median: ", ptm)
    print("peak_density_per_km2: ", peak_density)
    print("border_elevation: ", border_ratio)
    print("border_coverage: ", border_cov)
    print("field_radius_m: ", field_r)
    print("autocorr_has_secondary_peak: ", has_secondary)

    if (1 <= n_fields <= PLACE_MAX_N_FIELDS
            and ptm >= PLACE_PEAK_TO_MEDIAN_MIN
            and not has_secondary
            and not np.isnan(field_r)):
        cell_type = "Place cell" if n_fields == 1 else "Multi-field cell"
        notes = (f"{n_fields} field(s), ptm={ptm:.2f}, "
                 f"field_r={field_r:.0f} m")
    elif (border_ratio >= BORDER_ELEV_RATIO
          and border_cov >= BORDER_MIN_WALL_COVERAGE):
        cell_type = "Border cell"
        notes = (f"border_ratio={border_ratio:.2f}, coverage={border_cov:.2f}")
    elif (peak_density >= SPECKLE_MIN_PEAK_DENSITY
          and ptm <= SPECKLE_MAX_PEAK_TO_MEDIAN):
        cell_type = "Landmark speckle"
        notes = (f"peak_density={peak_density:.1f}/km², ptm={ptm:.2f}")
    else:
        notes = (f"n_fields={n_fields}, ptm={ptm:.2f}, "
                 f"density={peak_density:.1f}/km², "
                 f"border={border_ratio:.2f}")

    return NeuronResult(
        neuron_id=neuron_id,
        cell_type=cell_type,
        smooth_sigma_used=sigma,
        pass_number=pass_num,
        n_fields=n_fields,
        peak_to_median=ptm,
        peak_density_per_km2=peak_density,
        border_elevation=border_ratio,
        border_coverage=border_cov,
        field_radius_m=field_r,
        autocorr_has_secondary_peak=has_secondary,
        notes=notes,
    )


def classify_two_pass(rm_raw: np.ndarray, neuron_id: str) -> NeuronResult:
    """
    Run classify_at_scale with each sigma in SMOOTH_SIGMA_PASSES until the
    result is not 'Unclassified'. Return the first decisive classification,
    or the last attempt if all passes fail.
    """
    last_result = None
    for pass_num, sigma in enumerate(SMOOTH_SIGMA_PASSES, start=1):
        result = classify_at_scale(rm_raw, neuron_id, sigma, pass_num)
        last_result = result
        if result.cell_type != "Unclassified":
            return result
    return last_result


# =============================================================================
# Plotting
# =============================================================================

def plot_neuron_summary(rm_raw: np.ndarray,
                        result: NeuronResult,
                        out_path: str):
    rm = preprocess(rm_raw, result.smooth_sigma_used)
    ac, radii_bins, profile = autocorrelogram_radial(rm)
    radii_m = radii_bins * METERS_PER_BIN

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    im = axs[0].imshow(rm, cmap="jet", origin="lower")
    axs[0].set_title(f"Ratemap  (σ={result.smooth_sigma_used:g})")
    plt.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)

    im = axs[1].imshow(ac, cmap="seismic", origin="lower", vmin=-0.5, vmax=0.5)
    axs[1].set_title("Autocorrelogram")
    plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

    axs[2].plot(radii_m, profile, lw=1.5)
    axs[2].axhline(0, color="k", lw=0.5)
    axs[2].axhline(0.5, color="gray", ls=":", lw=0.7)
    axs[2].set_xlim(0, radii_m.max() * 0.7)
    axs[2].set_title(f"Radial AC  —  pass {result.pass_number}\n{result.notes}")
    axs[2].set_xlabel("Distance (m)")
    axs[2].set_ylabel("Autocorrelation")

    fig.suptitle(f"{result.cell_type} — Neuron {result.neuron_id}",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for cat in ("place", "multifield", "border", "speckle", "unclassified"):
        os.makedirs(os.path.join(OUTPUT_DIR, cat), exist_ok=True)

    files = sorted(glob.glob(INPUT_GLOB))
    if not files:
        sys.exit(f"No files matched: {INPUT_GLOB}")
    print(f"Classifying {len(files)} neurons with two-pass smoothing "
          f"σ = {SMOOTH_SIGMA_PASSES}")

    results: List[NeuronResult] = []
    for fp in tqdm(files, desc="Neurons"):
        neuron_id = (os.path.basename(fp)
                     .replace("ratemap_data_neuron_", "")
                     .replace(".npy", ""))
        try:
            rm = np.load(fp)
        except Exception as e:
            print(f"  {neuron_id}: load failed ({e})")
            continue
        try:
            r = classify_two_pass(rm, neuron_id)
            results.append(r)

            print(r)

            if SAVE_PLOTS:
                folder = {
                    "Place cell":       "place",
                    "Multi-field cell": "multifield",
                    "Border cell":      "border",
                    "Landmark speckle": "speckle",
                }.get(r.cell_type, "unclassified")
                plot_neuron_summary(
                    rm, r,
                    os.path.join(OUTPUT_DIR, folder,
                                 f"neuron_{neuron_id}.png"),
                )
        except Exception as e:
            print(f"  {neuron_id}: classify failed ({e})")

    # Summary CSV
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nWrote summary: {SUMMARY_CSV}")

    # Distribution
    print("\n=== Cell-type distribution ===")
    counts = df["cell_type"].value_counts()
    total = counts.sum()
    for ct, n in counts.items():
        print(f"  {ct:20s} {n:5d}  ({100 * n / total:5.1f}%)")
    print(f"  {'TOTAL':20s} {total:5d}")

    # Pass breakdown — how many were decided by σ=10 vs σ=1
    print("\n=== Pass breakdown ===")
    pass_counts = df.groupby(["pass_number", "cell_type"]).size().unstack(
        fill_value=0
    )
    print(pass_counts)
    pass_counts.to_csv(os.path.join(OUTPUT_DIR, "pass_breakdown.csv"))

    # Distribution plot
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot.bar(ax=ax)
    ax.set_title(f"Cell-type distribution ({total} neurons)")
    ax.set_ylabel("Number of neurons")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "distribution.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()