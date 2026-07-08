import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.formula.api as smf
import sys, os
from scipy.stats import mannwhitneyu

# Load the dataframe from the feather file
file = "step13_echo_classification_entropy_echo_attention_all_finetue_final_12000.feather"
file2 = "location_new_echo_attention_v2_ckpt12000_5031.feather"
df = pd.read_feather(file)

print("df : ", df, df.columns)
# sys.exit(1)


df2 = pd.read_feather(file2)

# random select 10%
df2 = df2.sample(frac=0.01, random_state=42).reset_index(drop=True)

true_tokens = []
for i in df2['location_id']:
    true_token = f"LOC_{str(i).zfill(5)}_"
    true_tokens.append(true_token)
df2["true_token"] = true_tokens
df2["echo_attention_scores"] = df2["attention_weight"] 



# print("df2: ", df2, df2.columns)
# sys.exit(1)
# random generate a matrix for echoes column, the size is (6, 5000)
# df['echoes'] = [np.random.rand(6, 5000) for _ in range(len(df))]

# true_tokens = []
# for i in df['true_token']:
#     i = i.replace("[", "").replace("]", "_")
#     true_tokens.append(i)

# df['true_token'] = true_tokens
# print(df, df.columns)

# ATTENTION_FILE = "/Users/xingchen/Downloads/new_start/checking_results_place_grid_direction/direction_and_angle/echo_attention/location_ALL_echo_attention_v2_ckpt8000_7101.feather"

ATTENTION_FILE = "location_new_echo_attention_v2_ckpt1500_only_cls_5891.feather"



df_control = pd.read_feather(ATTENTION_FILE)

# print("df_control: ", df_control, df_control.columns)
# sys.exit(1)
print(f"Successfully loaded attention data from: {ATTENTION_FILE}")
# print("\nDataFrame columns:", df.columns)

def fancy_amplitude_envelope(signal, frame_size, hop_length):
    """Fancier Python code to calculate the amplitude envelope of a signal with a given frame size."""
    return np.array([max(signal[i:i+frame_size]) for i in range(0, len(signal), hop_length)])

def calculate_smoothed_attention(attention_weight_values, window_size=5):
    if len(attention_weight_values) == 0: return pd.Series([], dtype=float)
    attention_weight_flat_list = [aw.tolist() for aw in attention_weight_values]
    if not attention_weight_flat_list: return pd.Series([], dtype=float)
    try:
        attention_weight_np = np.array(attention_weight_flat_list)
    except ValueError:
        return pd.Series([], dtype=float)
    attn_rescaleds_list = []
    for i in attention_weight_np:
        max_val, min_val = np.max(i), np.min(i)
        attn_rescaled = np.zeros_like(i)
        if max_val > 0 and (max_val - min_val) > 1e-12:
            attn_rescaled = (i - min_val) / (max_val - min_val)
        attn_rescaleds_list.append(attn_rescaled)
    if not attn_rescaleds_list: return pd.Series([], dtype=float)
    attn_summed = np.array(attn_rescaleds_list).sum(axis=0)
    if attn_summed.ndim != 1 or attn_summed.size == 0: return pd.Series([], dtype=float)
    return pd.Series(attn_summed).rolling(window=window_size, center=True, min_periods=1).mean()

# --- 1. Overall Attention ---
print("\nProcessing overall attention...")
# smoothed_attention_overall_orig_len = calculate_smoothed_attention([i[0] for i in df_control['echo_attention_scores']])


smoothed_attention_overall_orig_len = calculate_smoothed_attention(df_control['attention_weight'].values)
# print("smoothed_attention_overall_orig_len: ", np.array(smoothed_attention_overall_orig_len).shape)

# sys.exit(1)

# --- 2. Data Preparation and Merging ---
print("Preparing and merging landscape data...")
file_landscape = "/Users/xingchen/Downloads/submit_paper/Navigation_LLM/analysis/lanscape/Landscape_annotation4.xlsx"
df_landscape = pd.read_excel(file_landscape)
landscape_feature_cols = ['Hill', 'Forest', 'Rural settlement', 'Water', 'Orchards', 'Field_corp', 'Road', 'River', 'Swamp']
df_landscape['landscape'] = df_landscape[landscape_feature_cols].idxmax(axis=1)
df_landscape_to_merge = df_landscape[['target_loc', 'landscape']]
df_merged = pd.merge(df, df_landscape_to_merge, left_on='true_token', right_on='target_loc', how='left').drop(columns=['target_loc'])
df_merged2 = pd.merge(df2, df_landscape_to_merge, left_on='true_token', right_on='target_loc', how='left').drop(columns=['target_loc'])
print("\nMerge successful.")

# --- 3. Data Aggregation ---
window_size = 5
groups = df_merged.groupby("landscape")
attention_arrays = []
echo_arrays = []
for landscape, group in groups:
    for record_attention, record_echo in zip(group['echo_attention_scores'], group['echoes']):
        for i in record_attention:
            i_tmp = pd.Series(i.tolist()).rolling(window=window_size, center=True, min_periods=1).mean()
            attention_arrays.append(i_tmp.tolist())
        for i in record_echo:
            i_tmp2 = fancy_amplitude_envelope(i, 100, 50)
            i_tmp2 = pd.Series(i_tmp2.tolist()).rolling(window=window_size, center=True, min_periods=1).mean()
            echo_arrays.append(i_tmp2.tolist())

groups2 = df_merged2.groupby("landscape")
# attention_arrays = []
for landscape, group in groups2:
    for record_attention in group['echo_attention_scores']:
        # for i in record_attention:
            i_tmp = pd.Series(record_attention.tolist()).rolling(window=window_size, center=True, min_periods=1).mean()
            attention_arrays.append(i_tmp.tolist())


# print("attention_arrays: ", attention_arrays)
# sys.exit(1)

mean_echo_array = np.mean(echo_arrays, axis=0)
mean_attention_both_tasks_array = np.mean(attention_arrays, axis=0)
smoothed_attention_overall_orig_len = np.array(smoothed_attention_overall_orig_len)


print("Data Aggregation succeed!!")


# --- 4. Normalization ---
def normalize_array(data_array):
    min_val, max_val = np.min(data_array), np.max(data_array)
    return (data_array - min_val) / (max_val - min_val) if max_val - min_val else np.zeros_like(data_array)

mean_echo_array_normalized = normalize_array(mean_echo_array)
mean_attention_both_tasks_array_normalized = normalize_array(mean_attention_both_tasks_array)
smoothed_attention_overall_orig_len_normalized = normalize_array(smoothed_attention_overall_orig_len)
x_axis = np.linspace(0, 12.5, num=len(mean_echo_array_normalized))


# ### MODIFICATION START: Replaced Plotly function with Matplotlib ###
def plot_signal_comparison_matplotlib(x_data, echo_data, attention_both_data, attention_sensory_data):
    """
    Generates a line plot using Matplotlib comparing the three signal arrays.
    """
    # --- Setup plot style ---
    plt.style.use('default') # Reset style to avoid conflicts
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.set_facecolor('white')
    ax.set_facecolor('white')

    # --- Plot each line ---
    # 1. Echo Signal Envelope
    ax.plot(x_data, echo_data, color='gray', linewidth=2, linestyle=':', label='Echo Signal Envelope')
    
    # 2. Model - Both Tasks
    ax.plot(x_data, attention_both_data, color='crimson', linewidth=2, linestyle='-', label='Model (Sensory + Trajectory)')

    # 3. Model - Sensory Only
    ax.plot(x_data, attention_sensory_data, color='dodgerblue', linewidth=2, linestyle='--', label='Model (Sensory Only)')

    # --- Apply professional styling ---
    # Labels and Title
    ax.set_xlabel("Distance (m)", fontsize=20, fontfamily='Arial')
    ax.set_ylabel("Normalized amp/attn", fontsize=20, fontfamily='Arial')

    # Legend
    legend = ax.legend(
        loc='upper right',
        facecolor='white',
        framealpha=0.8,
        edgecolor='black',
        frameon=True,
        fontsize=14
    )
    legend.get_frame().set_linewidth(1.0)
    
    # Axes and Grid
    ax.grid(True, which='both', color='lightgrey', linestyle='-', linewidth=1)
    ax.set_axisbelow(True) # Ensure grid is behind the plot lines

    # Set spines (the lines bounding the plot area)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', width=1, length=6, labelsize=16)
    # ax.tick_params(axis='x', top=True) # Add ticks to the top
    # ax.tick_params(axis='y', right=True) # Add ticks to the right
    
    plt.tight_layout()
    plt.show()
# ### MODIFICATION END ###


def perform_statistical_tests(model_both_data, model_sensory_data):
    """
    Performs a Mann-Whitney U test to compare the two model output arrays.
    """
    print("\n--- Statistical Comparison ---")
    u_statistic, p_value = mannwhitneyu(model_both_data, model_sensory_data, alternative='two-sided')
    # print(f"Comparing 'Model (Sensory + Trajectory)' vs. 'Model (Sensory Only)':")
    print(f"  Mann-Whitney U-statistic: {u_statistic:.1f}")
    print(f"  P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("  Result: There is a statistically significant difference.")
    else:
        print("  Result: No statistically significant difference.")

# --- 5. Main Execution ---
if __name__ == "__main__":
    # Call the new Matplotlib plotting function
    plot_signal_comparison_matplotlib(
        x_axis,
        mean_echo_array_normalized,
        mean_attention_both_tasks_array_normalized,
        smoothed_attention_overall_orig_len_normalized
    )

    # Perform statistical tests on the ORIGINAL, unscaled data
    perform_statistical_tests(
        mean_attention_both_tasks_array,
        smoothed_attention_overall_orig_len
    )