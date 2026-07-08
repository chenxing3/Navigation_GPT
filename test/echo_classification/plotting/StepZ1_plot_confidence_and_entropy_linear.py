import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from matplotlib.colors import LogNorm
from scipy.stats import linregress
from matplotlib.ticker import FuncFormatter

# --- Configuration ---
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 16})

# Define your file paths here
RESULTS_FILENAMEs = [
    "./Step2_different_results/1535_traj_and_echo_and_cls.feather",
    "./Step2_different_results/hidden_states_far_60_120_conf0.00_0.50_result.feather",
    "./Step2_different_results/hidden_states_near_0_80_conf0.00_0.50_result.feather",
]

# --- 1. Load Data ---
print(f"Loading results from {len(RESULTS_FILENAMEs)} files...")
try:
    df_results = pd.concat([pd.read_feather(f) for f in RESULTS_FILENAMEs], ignore_index=True)
    print(f"Successfully loaded {len(df_results)} samples.")
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# --- 2. Process Data ---
print("Calculating means per location...")
grouped_data = []
groups = df_results.groupby("location")

for name, group in groups:
    mean_confidence = group['confidence'].mean()
    mean_entropy = group['mean_entropy'].mean()
    
    total = len(group)
    correct = (group['predicted_location'] == group['location']).sum()
    rate = correct / total if total > 0 else 0

    grouped_data.append({
        'location': name,
        'confidence': mean_confidence,
        'entropy': mean_entropy,
        'accuracy': rate
    })

df_plot = pd.DataFrame(grouped_data)
df_plot = df_plot.dropna(subset=['entropy', 'confidence', 'accuracy'])
df_valid_acc = df_plot[df_plot['accuracy'] > 0].copy()

# Add the transformed column to the DataFrame
df_plot['ln_entropy'] = np.log(df_plot['entropy'])

# --- 3. Statistical Fitting ---

# A. Linear Fit on Transformed Axis: ln(Entropy) vs Confidence
slope_log, intercept_log, r_log, p_log, _ = linregress(df_plot['ln_entropy'], df_plot['confidence'])
log_r_squared = r_log**2

# B. Linear Fit: Entropy vs Log10(Accuracy)
log_acc = np.log(df_valid_acc['accuracy'])
slope_ent, intercept_ent, r_ent, p_ent, _ = linregress(df_valid_acc['entropy'], log_acc)

# C. Linear Fit: Confidence vs Raw Accuracy
slope_conf, intercept_conf, r_conf, p_conf, _ = linregress(df_valid_acc['confidence'], df_valid_acc['accuracy'])


# --- 4. Generate Plot ---
print("Generating single-panel plot...")
fig, ax = plt.subplots(figsize=(10, 8))

# Safeguard for LogNorm: Clip minimum to 0.001 (0.1%)
c_data = np.clip(df_plot['accuracy'], 1e-3, 1.0)

# Scatter plot using ln(Entropy) for the X-axis
scatter = ax.scatter(
    df_plot['ln_entropy'], 
    df_plot['confidence'], 
    c=c_data, 
    cmap='viridis', 
    norm=LogNorm(vmin=c_data.min(), vmax=1.0), 
    s=20, 
    alpha=0.8, 
    edgecolors='w', 
    linewidth=0.5
)

# Plot the straight linear fit line in this transformed space
x_line = np.linspace(df_plot['ln_entropy'].min(), df_plot['ln_entropy'].max(), 100)
y_line = slope_log * x_line + intercept_log 

ax.plot(
    x_line, y_line, 
    color='red', 
    linewidth=2.5, 
    label='Linear Fit' 
)

# --- Add Mean Entropy Vertical Line ---
# We calculate the mean in raw space, but plot it in ln() space
mean_entropy_val_raw = df_plot['entropy'].mean()
mean_entropy_val_ln = np.log(mean_entropy_val_raw)
pct_below_mean = (df_plot['entropy'] < mean_entropy_val_raw).mean() * 100

ax.axvline(
    x=mean_entropy_val_ln, 
    color='black', 
    linestyle='--', 
    linewidth=2, 
    label=f'Mean Entropy (Raw: {mean_entropy_val_raw:.2f})'
)

# Text annotation for the mean line
y_text_pos = df_plot['confidence'].max() * 0.98  
ax.text(
    mean_entropy_val_ln + 0.05, y_text_pos, 
    f'{np.round(pct_below_mean)}% of locations < mean', 
    horizontalalignment='left', 
    verticalalignment='top',
    fontsize=16,
    color='black',
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
)

# Axis labels and title
ax.set_xlabel('ln(Mean Echo Entropy)') # Updated label
ax.set_ylabel('Mean Model Confidence')
ax.grid(True, linestyle='--', alpha=0.5) 
ax.legend(loc='lower right')

# --- Format the Colorbar ---
cbar = fig.colorbar(scatter, ax=ax, format=FuncFormatter(lambda y, _: f'{y*100:g}%'))
cbar.set_label('Classification Accuracy (Log Scale)', rotation=270, labelpad=20)

plt.tight_layout()
plt.show()

# --- Print stats to the console ---
print("\n" + "="*50)
print("STATISTICS FOR MANUSCRIPT LEGEND")
print("="*50)

print(f"[1] Linear Fit (Confidence vs ln(Entropy)):")
print(f"    Formula: y = {slope_log:.3f}x + {intercept_log:.3f}")
print(f"    R^2 = {log_r_squared:.3f}, p-value = {p_log:.1e}")

print(f"\n[2] Linear Fit (Log(Accuracy) vs Entropy):")
print(f"    Formula: log(Acc) = {slope_ent:.3f} * Entropy + {intercept_ent:.3f}")
print(f"    R = {r_ent:.3f}, p-value = {p_ent:.1e}")

print(f"\n[3] Linear Fit (Raw Accuracy vs Confidence):")
print(f"    Formula: Acc = {slope_conf:.3f} * Confidence + {intercept_conf:.3f}")
print(f"    R = {r_conf:.3f}, p-value = {p_conf:.1e}")
print("="*50 + "\n")