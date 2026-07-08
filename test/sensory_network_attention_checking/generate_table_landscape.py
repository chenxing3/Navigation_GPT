import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.formula.api as smf # Added for statistical modeling
import sys, os
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial', # Specify Arial as the sans-serif font
    'font.size': 18,            # Base font size for all text
    'axes.labelsize': 20,       # Font size for x and y labels
    'axes.titlesize': 22,       # Font size for subplot titles
    'legend.fontsize': 14       # Font size for legend
})
# Load the dataframe from the feather file
file = "step13_echo_classification_entropy_echo_attention_all_finetue_final.feather"
df = pd.read_feather(file)




def fancy_amplitude_envelope(signal, frame_size, hop_length):
	"""Fancier Python code to calculate the amplitude envelope of a signal with a given frame size."""
	return np.array([max(signal[i:i+frame_size]) for i in range(0, len(signal), hop_length)])



# --- Function to process attention weights (YOUR EXACT ORIGINAL) ---
def calculate_smoothed_attention(attention_weight_values, window_size=7):
    if len(attention_weight_values) == 0: return pd.Series([], dtype=float)

    # print("attention_weight_values: ", attention_weight_values)
    attention_weight_flat_list = []
    for array_group in attention_weight_values:
        # print("array_group: ", len(array_group))
        # for single_attention_array in array_group:
            # print("single_attention_array: ", len(single_attention_array))

            attention_weight_flat_list.append(array_group.tolist())
    if not attention_weight_flat_list: return pd.Series([], dtype=float)


    # print("len attention_weight_flat_list: ", len(attention_weight_flat_list))
    try: attention_weight_np = np.array(attention_weight_flat_list)
    except ValueError: return pd.Series([], dtype=float)
    attn_rescaleds_list = []

    # print("attention_weight_np: ", attention_weight_np, attention_weight_np.shape)
    # sys.exit(1)
    for i in attention_weight_np:
        max_val, min_val = np.max(i), np.min(i)
        attn_rescaled = np.zeros_like(i)
        if max_val > 0 and (max_val - min_val) > 1e-12:
            attn_rescaled = (i - min_val) / (max_val - min_val)
        attn_rescaleds_list.append(attn_rescaled)
    # print("attn_rescaleds_list: ", attn_rescaleds_list)
    # sys.exit(1)
    
    if not attn_rescaleds_list: return pd.Series([], dtype=float)
    attn_summed = np.array(attn_rescaleds_list).sum(axis=0)
    # print("attn_summed: ", attn_summed)

    if attn_summed.ndim != 1 or attn_summed.size == 0: return pd.Series([], dtype=float)

    # print("attn_summed: ", attn_summed)
    return pd.Series(attn_summed).rolling(window=window_size, center=True, min_periods=1).mean()



# add the landscape type to the dataframe

file_landscape = "/Users/xingchen/Downloads/submit_paper/Navigation_LLM/analysis/lanscape/Landscape_annotation3.xlsx"

df_landscape = pd.read_excel(file_landscape)

# --- 2. MERGE THE DATASETS ---
# --- 2. Data Preparation and Merging ---

print("Preparing and merging landscape data...")

# Define the columns that represent landscape types
landscape_feature_cols = ['Hill', 'Forest', 'Rural settlement', 'Water', 'Orchards', 'Field_corp', 'Road', 'River', 'Swamp']

# Check if all expected landscape columns exist
missing_cols = [col for col in landscape_feature_cols if col not in df_landscape.columns]
if missing_cols:
    print(f"Warning: The following landscape columns were not found in the CSV and will be ignored: {missing_cols}")
    landscape_feature_cols = [col for col in landscape_feature_cols if col in df_landscape.columns]

# Reshape df_landscape from wide to long format.
# idxmax(axis=1) finds the column name with the maximum value (the '1') for each row.
# This efficiently converts the one-hot encoded format to a single column.
df_landscape['landscape'] = df_landscape[landscape_feature_cols].idxmax(axis=1)





# --- 1. Overall Attention (YOUR ORIGINAL) ---
print("\nProcessing overall attention...")

ATTENTION_FILE = "/Users/xingchen/Downloads/new_start/checking_results_place_grid_direction/direction_and_angle/echo_attention/location_ALL_echo_attention_v2_ckpt8000_7101.feather"

df_control = pd.read_feather(ATTENTION_FILE)
print(f"Successfully loaded attention data from: {ATTENTION_FILE}")
print("\nDataFrame columns:",df_control,  df_control.columns)

# print("df['attention_weight'].values ", df_control['attention_weight'].values)
# smoothed_attention_overall_orig_len = calculate_smoothed_attention(df_control['attention_weight'].values)
# smoothed_attention_overall_orig_len = smoothed_attention_overall_orig_len/max(smoothed_attention_overall_orig_len)
# print("smoothed_attention_overall_orig_len: ", smoothed_attention_overall_orig_len, len(smoothed_attention_overall_orig_len))
# sys.exit(1)


     


dict_idx_to_landscape = {}
for i in df_landscape.index:
    id = int(df_landscape['target_loc'][i].split("_")[1])
    dict_idx_to_landscape[id] = df_landscape['landscape'][i]
df_control['landscape']= [dict_idx_to_landscape[x] for x in df_control["location_id"]]


landscape_to_id = {
    'Water': 0, 'Crop field': 1, 'River': 2, 'Road': 3,
    'Rural settlement': 4, 'Swamp': 5, #'Hill': 6,
    'Hill': 6,'Grove': 7, 'Orchards': 8, 
}

rename_mapping = {
    'Field_corp': 'Crop field',
    'Forest': 'Grove'
}

cmap = plt.get_cmap('rainbow')
max_id = max(landscape_to_id.values())

landscape_colors = {}
for name, id_val in landscape_to_id.items():
    # Normalize ID to get color: (value / max_value)
    landscape_colors[name] = cmap(id_val / max_id) if max_id > 0 else cmap(0)



# Apply the renaming to the 'landscape' column
df_landscape['landscape'] = df_landscape['landscape'].replace(rename_mapping)
df_control['landscape'] = df_control['landscape'].replace(rename_mapping)

# Keep only the necessary columns for the merge to keep it clean
df_landscape_to_merge = df_landscape[['target_loc', 'landscape']]

# Merge the main dataframe with the landscape dataframe
# We use a 'left' merge to ensure all original rows from `df` are kept.
# If a `true_token` in `df` doesn't have a match in `df_landscape`, its `landscape` will be NaN.
df_merged = pd.merge(
    df,
    df_landscape_to_merge,
    left_on='true_token',
    right_on='target_loc',
    how='left'
)

# Optional: Clean up by dropping the redundant key column from the right dataframe
df_merged = df_merged.drop(columns=['target_loc'])

print("\nMerge successful. Here's the head of the new merged DataFrame:")

print(df_merged, df_merged.columns)
print(f"\nNumber of rows in original df: {len(df)}")
print(f"Number of rows in merged df: {len(df_merged)}")
print(f"Number of rows with missing landscape info (if any): {df_merged['landscape'].isnull().sum()}")

window_size = 1
groups = df_merged.groupby("landscape")

fig = plt.figure()


pools = []
current_amplitude_arrays_all = []
for landscape, group in groups:
    
    current_attention_arrays = []
    for record in group['echo_attention_scores']:
        records = [i.tolist() for i in record]
        current_attention_arrays.extend(records)

    mean_current_attention_arrays = np.mean(current_attention_arrays, axis=0)
    mean_current_attention_arrays = mean_current_attention_arrays.tolist()
    mean_current_attention_smooth = pd.Series(mean_current_attention_arrays).rolling(window=window_size, center=True, min_periods=1).mean()
    # print("mean_current_attention_smooth: ", mean_current_attention_smooth.shape)

    # mean_current_attention_smooth = mean_current_attention_smooth/max(mean_current_attention_smooth)

    x_axis = np.linspace(0, 12.5, num=len(mean_current_attention_smooth))
    # plt.plot(x_axis, mean_current_attention_smooth, label=landscape)
    c = landscape_colors.get(landscape, 'black')
    plt.plot(x_axis, mean_current_attention_smooth, label=landscape, color=c)



    current_amplitude_arrays = []
    for record in group['echoes']:
        records = [i.tolist() for i in record]
        current_amplitude_arrays.extend(records)
        # current_amplitude_arrays_all.extend(records)

    mean_current_amp_arrays = np.mean(current_amplitude_arrays, axis=0)
    mean_current_amp_smoothed = fancy_amplitude_envelope(mean_current_amp_arrays, 100,50)
    current_amplitude_arrays_all.append(mean_current_amp_smoothed.tolist())
    # mean_current_amp_smoothed = mean_current_amp_smoothed/max(mean_current_amp_smoothed)

    # print("mean_current_amp_smoothed: ", mean_current_amp_smoothed.shape)
    # plt.plot(x_axis, mean_current_amp_smoothed, label=landscape, color=c)
    # sys.exit(1)


    for idx, (i, j) in enumerate(zip(mean_current_attention_smooth, mean_current_amp_smoothed)):
        pools.append([idx, i, j, landscape, landscape_to_id[landscape], 1])


plt.xlabel('Distance (m)')


# plt.ylabel('Amplitude')# / Scaled Contribution')
plt.ylabel('Attention score')
# plt.legend()

# handles, labels = plt.gca().get_legend_handles_labels()
# sorted_pairs = sorted(zip(handles, labels), key=lambda t: landscape_to_id.get(t[1], 999))

# # 3. Unzip them back into lists
# if sorted_pairs:
#     handles, labels = zip(*sorted_pairs)
#     plt.legend(handles, labels)
# else:
#     plt.legend()

plt.show()

mean_current_amp_average_arrays = np.mean(current_amplitude_arrays_all, axis=0)
mean_current_amp_average_arrays = mean_current_amp_average_arrays/max(mean_current_amp_average_arrays)

# print("mean_current_amp_average_arrays: ", mean_current_amp_average_arrays, mean_current_amp_average_arrays.shape)


# # sys.exit(1)
# for idx, (i, j) in enumerate(zip(smoothed_attention_overall_orig_len.tolist(), mean_current_amp_average_arrays.tolist())):
#     pools.append([idx, i, j, "control"])

groups_control = df_control.groupby("landscape")
for landscape, group in groups_control:
    # print("group: ", group['attention_weight'])
    
    current_attention_arrays = []
    for record in group['attention_weight']:
        records = [i.tolist() for i in record]
        current_attention_arrays.append(records)

    control_mean_current_attention_arrays = np.mean(current_attention_arrays, axis=0)
    control_mean_current_attention_arrays = control_mean_current_attention_arrays.tolist()
    control_mean_current_attention_smooth = pd.Series(control_mean_current_attention_arrays).rolling(window=window_size, center=True, min_periods=1).mean()

    # print("control_mean_current_attention_smooth: ", control_mean_current_attention_smooth)
    # sys.exit(1)

    # get group for according to the landscape

    amplitude_group = groups.get_group(landscape)
    current_amplitude_arrays = []
    for record in amplitude_group['echoes']:
        records = [i.tolist() for i in record]
        current_amplitude_arrays.extend(records)
        # current_amplitude_arrays_all.extend(records)

    mean_current_amp_arrays = np.mean(current_amplitude_arrays, axis=0)
    mean_current_amp_smoothed = fancy_amplitude_envelope(mean_current_amp_arrays, 100,50)
    current_amplitude_arrays_all.append(mean_current_amp_smoothed.tolist())
    mean_current_amp_smoothed = mean_current_amp_smoothed/max(mean_current_amp_smoothed)


    x_axis = np.linspace(0, 12.5, num=len(control_mean_current_attention_smooth))
    plt.plot(x_axis, control_mean_current_attention_smooth, label=landscape)

    for idx, (i, j) in enumerate(zip(control_mean_current_attention_smooth, mean_current_amp_smoothed)):
        pools.append([idx, i, j, landscape, landscape_to_id[landscape], 0])






plt.xlabel('Distance (m)')


# plt.ylabel('Amplitude')# / Scaled Contribution')
plt.ylabel('Attention')
plt.legend()

plt.show()



df_sta = pd.DataFrame(pools, columns= ["Distance", "attention", "amplitude", "landscape", "landscape_id", "exp_type"])

df_sta.to_csv("attention_amplitude_control_landscape.csv", index=False)


# sys.exit(1)




import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# only use the df_sta when the Distance > 5




# df_sta = df_sta[df_sta['Distance'] > 40]
df_sta['attention_log'] = [ np.log(i*100) for i in df_sta['attention']]
# from scipy.stats import shapiro
# # Perform the Shapiro-Wilk test
# stat, p_value = shapiro(df_sta['attention'].values)

# print(f"Shapiro-Wilk Test Statistic: {stat:.4f}")
# print(f"P-value: {p_value:.4f}")

# sys.exit(1)



formula = 'attention_log ~ C(landscape):Distance + Distance + amplitude'

# 2. Fit the model using GLM instead of OLS
# We specify the Gamma family and the log link function.
glm_family = sm.families.Gamma() # link=sm.families.links.log()
model = smf.glm(formula=formula, data=df_sta, family=glm_family).fit()
print("summary: ", model.summary())



# formula = 'attention_log ~ C(landscape_id) + Distance + amplitude + C(landscape_id):amplitude'
# # For anova_lm, we fit with OLS, which is equivalent to GLM with Gaussian family
# model = smf.ols(formula=formula, data=df_sta).fit()
# print("summary: ", model.summary())

# p_entropy = model.pvalues
# anova_results = anova_lm(model, typ=3)

# print("anova_results: ", anova_results)

sys.exit(1)




# save_file = "attention_amplitude_control.csv"

# df_save = pd.DataFrame(pools, columns= ["postion", "attention", "amplitude", "landscape"])

# df_save.to_csv(save_file)



sys.exit(1)

