import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

# --- Configuration ---
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 18})

# Define your file paths
file1 = "./Step2_different_results/1535_traj_and_echo_and_cls.feather"
file3 = "./Step2_different_results/1535_only_echo.feather" 

files = [
    [file1, "+ qc and trajectory"],
    [file3, "only echo"],
]

OUTPUT_FILENAME = "proportion_accuracy_comparison.png"

colors = {
    '+ qc and trajectory': '#1f77b4', 
    'only echo': '#2ca02c'            
}

# --- 1. Process Data ---
location_stats_dict = {}

for filepath, label in files:
    print(f"\nProcessing: {label}...")
    try:
        df_results = pd.read_feather(filepath)
    except FileNotFoundError:
        print(f"  Error: File not found at '{filepath}'. Skipping.")
        continue

    if 'predicted_location' not in df_results.columns or 'location' not in df_results.columns:
        print(f"  Error: Missing required columns in {filepath}. Skipping.")
        continue

    df_results['predicted_class_index'] = df_results['predicted_location']
    df_results['true_class_index'] = df_results['location']

    df_valid = df_results.dropna(subset=['true_class_index', 'predicted_class_index']).copy()
    if df_valid.empty:
        print("  Warning: No valid samples found.")
        continue

    df_valid['true_class_index'] = df_valid['true_class_index'].astype(int)
    df_valid['predicted_class_index'] = df_valid['predicted_class_index'].astype(int)

    # Calculate exact-match accuracy
    df_valid['sample_accuracy'] = (df_valid['predicted_class_index'] == df_valid['true_class_index']).astype(int)

    # Group by unique location
    df_location_stats = df_valid.groupby('true_class_index')['sample_accuracy'].mean().reset_index()
    df_location_stats.columns = ['location_id', 'accuracy']
    
    location_stats_dict[label] = df_location_stats

    # --- DIAGNOSTIC PRINT ---
    print(f"  -> Raw samples processed: {len(df_valid)}")
    print(f"  -> Unique locations evaluated (n): {len(df_location_stats)}")

if not location_stats_dict:
    print("No data was successfully loaded. Exiting.")
    sys.exit(1)

# --- 1.5 Ensure All Locations are Included (Union with Zero Fill) ---
print("\nAligning data to include all locations across both models (filling missing with 0)...")

# Find the union of all locations (present in ANY file)
all_locations = set()
for df_stats in location_stats_dict.values():
    all_locations.update(df_stats['location_id'])

print(f"  -> Total unique locations across all models: {len(all_locations)}")

# Reindex to ensure both models have all locations, filling missing with 0 accuracy
model_accuracies = {}
for label, df_stats in location_stats_dict.items():
    df_reindexed = df_stats.set_index('location_id').reindex(list(all_locations), fill_value=0.0)
    model_accuracies[label] = df_reindexed['accuracy'].values
    print(f"  -> {label}: expanded to {len(df_reindexed)} locations (filled {len(df_reindexed) - len(df_stats)} missing with 0.0)")

# --- 2. Calculate Binned Proportions, Plot Curves, and Medians ---
print("\nGenerating smoothed probability plot...")
fig, ax = plt.subplots(figsize=(12, 7))

bin_width = 0.02
bins = np.arange(-0.01, 1.1 + bin_width, bin_width)
x_grid = (bins[:-1] + bins[1:]) / 2 
SMOOTHING_SIGMA = 2

for label, acc_values in model_accuracies.items():
    if len(acc_values) == 0: continue
    
    # 2a. Calculate and plot the smoothed distribution
    counts, _ = np.histogram(acc_values, bins=bins)
    proportions = counts / len(acc_values)
    smoothed_proportions = gaussian_filter1d(proportions, sigma=SMOOTHING_SIGMA)
    ax.plot(x_grid, smoothed_proportions, color=colors[label], linewidth=2.5)

    # 2b. Calculate median and find the exact height of the curve at that point
    median_val = np.median(acc_values)
    median_y_height = np.interp(median_val, x_grid, smoothed_proportions)
    
    # Plot the dashed line only up to the curve's height
    ax.vlines(x=median_val, ymin=0, ymax=median_y_height, color=colors[label], linestyle='--', linewidth=2.5, alpha=0.8)
    
    print(f"  -> {label} Median Accuracy: {median_val:.3f}")

# --- 3. Formatting and Custom Legend ---
ax.set_xlabel('Accuracy per Location', fontsize=20)
ax.set_ylabel('Proportion of Locations', fontsize=20)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

ax.set_xlim([-0.1, 1.0])
ax.set_ylim(bottom=0)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1.5)

custom_lines = [
    Line2D([0], [0], color=colors['+ qc and trajectory'], lw=2.5, label='Full model'),
    Line2D([0], [0], color=colors['only echo'], lw=2.5, label='Only echo'),
    Line2D([0], [0], color='gray', lw=2.5, linestyle='--', label='Median accuracy')
]
ax.legend(handles=custom_lines, loc="upper right", fontsize=14, frameon=True, edgecolor='black')

plt.tight_layout()
plt.savefig(OUTPUT_FILENAME, dpi=200)
print(f"Saved figure to: {OUTPUT_FILENAME}")
plt.show()