import numpy as np
import pandas as pd
import os, sys, glob
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math # For ceil
from tqdm import tqdm

print(f"PyTorch version: {torch.__version__}")
print(f"Pandas version: {pd.__version__}")

# --- Configuration ---
PATH_MATCH1 = "./smart_1535loc_hidden_states_ckpt15000/hidden_states_far_60_120_conf0.50_1.00.feather" 
PATH_MATCH2 = "./smart_1535loc_hidden_states_ckpt15000/hidden_states_near_0_80_conf0.50_1.00.feather" 
PATH_MATCH1_TEST_ONLY = "./smart_1535loc_hidden_states_ckpt15000/hidden_states_far_60_120_conf0.00_0.50.feather" 
PATH_MATCH2_TEST_ONLY = "./smart_1535loc_hidden_states_ckpt15000/hidden_states_near_0_80_conf0.00_0.50.feather" 

OUTPUT_RESULTS_FILE = "./1535_multi_task_avg_pred_results_v4token_500_0_120_all_landdmark_all_confidence.feather"
TEST_SPLIT_RATIO = 0.6 
RANDOM_SEED = 42
SUB_SEQUENCE_FRACTION = 2/5 

# --- Hyperparameters ---
LR_CLASSIFICATION = 0.003
EPOCHS_CLASSIFICATION = 12000

LR_AZIMUTH = 0.001
EPOCHS_AZIMUTH = 12000

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# --- Helper function for angular metrics ---
def calculate_angular_metrics(true_az_deg, pred_az_deg):
    true_az_deg = np.asarray(true_az_deg)
    pred_az_deg = np.asarray(pred_az_deg)
    abs_diff_deg = np.abs(true_az_deg - pred_az_deg)
    angular_abs_diff_deg = np.minimum(abs_diff_deg, 360.0 - abs_diff_deg)
    mae_deg = np.mean(angular_abs_diff_deg)
    mse_deg_sq = np.mean(angular_abs_diff_deg**2)
    return mae_deg, mse_deg_sq

# --- 1. Load Data ---
print("Loading hidden state data (expecting sequences)...")
input_feather_file1 = PATH_MATCH1
# all_data_df = pd.read_feather(input_feather_file1)
# print("all_data_df: ", all_data_df, all_data_df.columns)
# sys.exit(1)



input_feather_file2 = PATH_MATCH2
print(f"Loading data from: {input_feather_file1}")
try:
    all_data_df = pd.read_feather(input_feather_file1)
    all_data_df_2 = pd.read_feather(input_feather_file2)
except Exception as e: print(f"Error reading feather file: {e}"); sys.exit(1)




all_data_df_3 = pd.read_feather(PATH_MATCH1_TEST_ONLY)
all_data_df_4 = pd.read_feather(PATH_MATCH2_TEST_ONLY)


# combine the two df into all_data_df
all_data_df = pd.concat([all_data_df, all_data_df_2,all_data_df_3,all_data_df_4], ignore_index=True)
# print("all_data_df: ", all_data_df, all_data_df.columns)
# sys.exit(1)

# only use the rows with confidence > 0.6
# all_data_df = all_data_df[all_data_df['confidence'] > 0.50].reset_index(drop=True)


print(f"Loaded {len(all_data_df)} records. Columns: {all_data_df.columns.tolist()}")

all_data_df = all_data_df.rename(columns={
    "location_id": "location",
    "hidden_state_sequence": "array", 
    "mean_azimuth": "azimuth",
    "mean_coordinate_x": "coord_x",
    "mean_coordinate_y": "coord_y"
})
if 'array' not in all_data_df.columns:
    print("Error: 'array' (hidden_state_sequence) column not found after renaming. Check input file.")
    sys.exit(1)
if 'coord_x' not in all_data_df.columns or 'coord_y' not in all_data_df.columns:
    print("Error: 'coord_x' or 'coord_y' column not found after renaming. These are needed for proximity filtering.")
    sys.exit(1)


# --- NEW: Landscape INCLUSION Logic ---
landscape_file = "./Landscape_annotation4_all.csv"
if os.path.exists(landscape_file):
    print(f"\nLoading landscape annotation from: {landscape_file} for filtering.")
    try:
        landscape_df = pd.read_csv(landscape_file)
        
        if not all(col in landscape_df.columns for col in ['target_loc', 'Water', 'Field_corp']):
            print(f"Error: Landscape file must contain 'target_loc', 'Water', and 'Field_corp' columns.")
            sys.exit(1)

        # 1. Parse the integer ID from 'target_loc' (e.g., 'LOC_00001_' -> 1)
        landscape_df['loc_id_int'] = landscape_df['target_loc'].str.extract(r'(\d+)').astype(int)

        # 2. Define the condition for locations you want to KEEP
        mask_to_keep = (landscape_df['Water'] == 2) | (landscape_df['Field_corp'] == 2) # one is to filter, other value > 1 means we didn't filter anythings 
        included_locations = landscape_df.loc[~mask_to_keep, 'loc_id_int'].unique()
        
        print(f"Identified {len(included_locations)} unique locations to KEEP (Water=1 or Field_corp=1).")
        
        # Ensure the main dataframe 'location' column is treated as integers for a perfect match
        all_data_df['location'] = all_data_df['location'].astype(int)

        # 3. Filter the main dataframe to KEEP ONLY those locations
        original_len = len(all_data_df)
        all_data_df = all_data_df[all_data_df['location'].isin(included_locations)]
        
        print(f"Filtered out {original_len - len(all_data_df)} data points based on landscape.")
        print(f"Dataset size after keeping specific landscapes: {len(all_data_df)}")
        print(f"Number of unique location_ids after landscape filtering: {all_data_df['location'].nunique()}") # ADDED LINE
        all_data_df = all_data_df.reset_index(drop=True)
        
    except Exception as e:
        print(f"Error reading or processing landscape file '{landscape_file}': {e}")
        sys.exit(1)
else:
    print(f"\nWarning: Landscape file '{landscape_file}' not found. Skipping landscape filtering.")
# --- END OF Landscape INCLUSION Logic ---


# --- NEW: Proximity Filtering Logic ---
hex_centers_file_path = "../ops/hex_centers_50_0_extend_no_GL_version2.csv"
PROXIMITY_FILTER_DISTANCE_METERS = 30.1 

if PROXIMITY_FILTER_DISTANCE_METERS is not None and PROXIMITY_FILTER_DISTANCE_METERS >= 0:
    print(f"\nLoading hex centers data from: {hex_centers_file_path} for proximity filtering.")
    try:
        hex_df = pd.read_csv(hex_centers_file_path)
        if not all(col in hex_df.columns for col in ['Type', 'x', 'y']):
            print(f"Error: Hex centers file {hex_centers_file_path} must contain 'Type', 'x', 'y' columns.")
            sys.exit(1)
    except Exception as e:
        print(f"Error reading hex centers CSV file '{hex_centers_file_path}': {e}"); sys.exit(1)

    ref_locs_df = hex_df[hex_df['Type'] == 'LOC'].copy()
    if ref_locs_df.empty:
        print("Error: No locations of Type 'LOC' found in hex centers file. Cannot perform proximity filtering."); sys.exit(1)

    ref_locs_xy = ref_locs_df[['x', 'y']].values
    print(f"Found {len(ref_locs_xy)} reference LOCations for filtering.")

    print(f"Filtering main dataset based on proximity to reference LOCations (within {PROXIMITY_FILTER_DISTANCE_METERS} meters)...")
    data_coords_np = all_data_df[['coord_x', 'coord_y']].values

    if len(data_coords_np) > 0: # Ensure we still have data after landscape filtering
        squared_distances_to_ref_locs = np.sum(
            (data_coords_np[:, np.newaxis, :] - ref_locs_xy[np.newaxis, :, :])**2,
            axis=2
        )
        min_squared_distances = np.nanmin(squared_distances_to_ref_locs, axis=1)

        if PROXIMITY_FILTER_DISTANCE_METERS == 0:
            proximity_mask = min_squared_distances == 0.0
            print("Note: Filtering for exact coordinate matches (distance = 0m). This might result in few or no data points.")
        else:
            dist_min_sq = 0.0 ** 2
            dist_max_sq = 120.0 ** 2
            proximity_mask = (min_squared_distances >= dist_min_sq) & (min_squared_distances < dist_max_sq)

        original_count = len(all_data_df)
        all_data_df = all_data_df[proximity_mask]
        filtered_count = len(all_data_df)
        print(f"Dataset size after proximity filtering (within {PROXIMITY_FILTER_DISTANCE_METERS}m of a LOC): {filtered_count}")
        print(f"Number of unique location_ids after proximity filtering: {all_data_df['location'].nunique()}") # ADDED LINE

        if filtered_count == 0 and PROXIMITY_FILTER_DISTANCE_METERS > 0 : 
            print("Warning: No data points left after proximity filtering. Check filter criteria, data, or reference locations.");
        all_data_df = all_data_df.reset_index(drop=True)
    else:
        print("Skipping proximity filtering because dataset is empty after landscape filtering.")
else:
    print("\nProximity filtering is disabled.")
# --- END OF NEW Proximity Filtering Logic ---


# --- 2. Data Cleaning and Preparation ---
print("\nCleaning and preparing data...")
valid_sequences = []
sequence_length_S = -1
hidden_dim_H = -1
problematic_items_indices = [] 

for item_idx, item in tqdm(enumerate(all_data_df['array']), desc="Validating sequences", total=len(all_data_df)):

    item = [i.tolist() for i in item]
    if isinstance(item, list): 
        if not item: 
            valid_sequences.append(None)
            problematic_items_indices.append(item_idx)
            continue
        try:
            sequence_np = np.asarray(item, dtype=np.float32)
        except ValueError as e: 
            valid_sequences.append(None)
            problematic_items_indices.append(item_idx)
            continue
        except TypeError as e: 
            valid_sequences.append(None)
            problematic_items_indices.append(item_idx)
            continue
    elif isinstance(item, np.ndarray):
        sequence_np = item.astype(np.float32) 
    else:
        valid_sequences.append(None)
        problematic_items_indices.append(item_idx)
        continue

    if sequence_np.ndim != 2:
        valid_sequences.append(None)
        problematic_items_indices.append(item_idx)
        continue
    
    if sequence_length_S == -1:
        if sequence_np.shape[0] > 0 and sequence_np.shape[1] > 0: 
            sequence_length_S = sequence_np.shape[0]
            hidden_dim_H = sequence_np.shape[1]
            print(f"Detected sequence length S = {sequence_length_S}, hidden_dim H = {hidden_dim_H}")
        else:
            valid_sequences.append(None)
            problematic_items_indices.append(item_idx)
            continue
    elif sequence_np.shape[0] != sequence_length_S or sequence_np.shape[1] != hidden_dim_H:
        valid_sequences.append(None)
        problematic_items_indices.append(item_idx)
        continue
    valid_sequences.append(sequence_np)


all_data_df['array'] = valid_sequences
if problematic_items_indices:
    print(f"Number of problematic items skipped during sequence validation: {len(problematic_items_indices)}")
all_data_df.dropna(subset=['array'], inplace=True) 


required_targets = ['location', 'azimuth'] # Removed coord targets from required
all_data_df.dropna(subset=required_targets, inplace=True) 
print(f"Records after NaN drop & sequence validation: {len(all_data_df)}")
if not len(all_data_df): print("Error: No data after cleaning."); sys.exit(1)

# --- INsERT NEW METHODOLOGY: Filter locations with insufficient samples ---
MIN_SAMPLES = 20
print(f"\nEvaluating class distribution for minimum threshold (N >= {MIN_SAMPLES})...")

# Calculate the frequency of each unique location
loc_counts = all_data_df['location'].value_counts()

# Identify locations that meet the statistical threshold
valid_locs = loc_counts[loc_counts >= MIN_SAMPLES].index
dropped_classes_count = len(loc_counts) - len(valid_locs)

# Apply the filter to the dataframe
original_record_count = len(all_data_df)
all_data_df = all_data_df[all_data_df['location'].isin(valid_locs)].reset_index(drop=True)



if sequence_length_S == -1 or hidden_dim_H == -1:
    if len(all_data_df) > 0 and all_data_df['array'].iloc[0] is not None:
        first_valid_array = all_data_df['array'].iloc[0]
        if first_valid_array.ndim == 2 and first_valid_array.shape[0] > 0 and first_valid_array.shape[1] > 0:
            sequence_length_S = first_valid_array.shape[0]
            hidden_dim_H = first_valid_array.shape[1]
            print(f"Re-detected sequence length S = {sequence_length_S}, hidden_dim H = {hidden_dim_H} from cleaned data.")
        else:
            print("Error: First valid array in cleaned data is not usable for S, H detection."); sys.exit(1)
    else:
        print("Error: No valid sequences in cleaned data to determine S and H."); sys.exit(1)

all_data_df = all_data_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# Determine number of sub-states to use
num_sub_states_to_use_k = math.ceil(sequence_length_S * SUB_SEQUENCE_FRACTION)
if num_sub_states_to_use_k == 0 and sequence_length_S > 0: 
    num_sub_states_to_use_k = 1
if num_sub_states_to_use_k > sequence_length_S: 
    num_sub_states_to_use_k = sequence_length_S
print(f"Using last k = {num_sub_states_to_use_k} states from each sequence of length S = {sequence_length_S}.")


# --- Prepare Expanded Features (X_expanded) and Targets for Training ---
X_orig_np = np.stack(all_data_df['array'].values) 
X_selected_np = X_orig_np[:, -num_sub_states_to_use_k:, :]
X_expanded_np = X_selected_np.reshape(-1, hidden_dim_H)
X_expanded_tensor = torch.tensor(X_expanded_np, dtype=torch.float32)
input_dim_model = hidden_dim_H 
print(f"Original X shape: {X_orig_np.shape}")
print(f"Selected k states X shape: {X_selected_np.shape}")
print(f"Expanded X_train shape (N_samples*k, H): {X_expanded_tensor.shape}, Model input_dim: {input_dim_model}")


# --- Prepare Expanded Targets ---
# Classification
original_location_ids = all_data_df['location'].astype(int).values
y_loc_expanded = np.repeat(original_location_ids, num_sub_states_to_use_k) 
unique_locations = sorted(np.unique(y_loc_expanded)) 
num_loc_classes = len(unique_locations)
print("len num_loc_classes: ", num_loc_classes)
if num_loc_classes <= 1:
    print(f"Error: Only {num_loc_classes} unique location(s) found. Need at least 2 for classification.")
    sys.exit(1)
location_to_index = {loc_id: i for i, loc_id in enumerate(unique_locations)}
index_to_location = {i: loc_id for loc_id, i in location_to_index.items()}
y_loc_indices_expanded = np.array([location_to_index[loc_id] for loc_id in y_loc_expanded])
y_loc_expanded_tensor = torch.tensor(y_loc_indices_expanded, dtype=torch.long)

# Azimuth Regression (sin/cos)
y_azimuth_deg_orig = all_data_df['azimuth'].values.astype(np.float32)
y_azimuth_rad_orig = np.deg2rad(y_azimuth_deg_orig)
y_azimuth_sin_orig = np.sin(y_azimuth_rad_orig)
y_azimuth_cos_orig = np.cos(y_azimuth_rad_orig)
y_azimuth_sincos_orig_np = np.stack((y_azimuth_sin_orig, y_azimuth_cos_orig), axis=-1) 
y_azimuth_sincos_expanded_np = np.repeat(y_azimuth_sincos_orig_np, num_sub_states_to_use_k, axis=0) 
y_azimuth_sincos_expanded_tensor = torch.tensor(y_azimuth_sincos_expanded_np, dtype=torch.float32)

# --- 3. Split Data ---
original_sample_indices_expanded = np.repeat(np.arange(len(all_data_df)), num_sub_states_to_use_k)

(X_train_exp, X_test_exp_placeholder, 
 y_loc_train_exp, y_loc_test_exp_placeholder,
 y_az_sincos_train_exp, y_az_sincos_test_exp_placeholder,
 train_indices_exp, test_indices_exp_placeholder 
 ) = train_test_split(
    X_expanded_tensor,
    y_loc_expanded_tensor,
    y_azimuth_sincos_expanded_tensor,
    np.arange(len(X_expanded_tensor)), 
    test_size=TEST_SPLIT_RATIO, 
    random_state=RANDOM_SEED,
    stratify=y_loc_expanded_tensor 
)

print(f"Expanded train set size (for model training): {X_train_exp.shape[0]}")

stratify_labels_original = all_data_df['location'].map(location_to_index).values
if np.isnan(stratify_labels_original).any(): 
    print("Warning: NaNs found in stratify_labels_original. This might indicate new/unseen location IDs.")
    _ , test_original_indices = train_test_split(
        all_data_df.index, 
        test_size=TEST_SPLIT_RATIO,
        random_state=RANDOM_SEED
    )
else:
    _ , test_original_indices = train_test_split(
        all_data_df.index, 
        test_size=TEST_SPLIT_RATIO,
        random_state=RANDOM_SEED,
        stratify=stratify_labels_original
    )

df_test_set_original_samples = all_data_df.loc[test_original_indices].reset_index(drop=True)
X_test_orig_sequences = np.stack(df_test_set_original_samples['array'].values)[:, -num_sub_states_to_use_k:, :]
print(f"Number of original test samples: {len(df_test_set_original_samples)}")
print(f"Shape of X_test_orig_sequences (for final eval): {X_test_orig_sequences.shape}")

# --- 4. Define Models, Loss, Optimizers ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X_train_exp = X_train_exp.to(device)
y_loc_train_exp = y_loc_train_exp.to(device)
y_az_sincos_train_exp = y_az_sincos_train_exp.to(device)
X_test_orig_sequences_tensor = torch.tensor(X_test_orig_sequences, dtype=torch.float32).to(device)

# 4.1 Location Classification Model
class LocationClassifierModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim // 2) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3) 
        self.linear2 = nn.Linear(input_dim // 2, num_classes)
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)

model_loc_clf = LocationClassifierModel(input_dim_model, num_loc_classes).to(device)
criterion_loc_clf = nn.CrossEntropyLoss()
optimizer_loc_clf = optim.Adam(model_loc_clf.parameters(), lr=LR_CLASSIFICATION)

# 4.2 Azimuth Regression Model
class AzimuthRegressorModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3) 
        self.linear2 = nn.Linear(input_dim, input_dim // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3) 
        self.linear3 = nn.Linear(input_dim // 2, 2) 
    def forward(self, x):
        x = self.relu1(self.linear1(x))
        x = self.dropout1(x)
        x = self.relu2(self.linear2(x))
        x = self.dropout2(x)
        return self.linear3(x)

model_az_reg = AzimuthRegressorModel(input_dim_model).to(device)
criterion_az_reg = nn.MSELoss()
optimizer_az_reg = optim.Adam(model_az_reg.parameters(), lr=LR_AZIMUTH)

# --- 5. Training Loops ---
print("\n--- Training Location Classifier ---")
for epoch in range(EPOCHS_CLASSIFICATION):
    model_loc_clf.train()
    optimizer_loc_clf.zero_grad()
    outputs = model_loc_clf(X_train_exp)
    loss = criterion_loc_clf(outputs, y_loc_train_exp)
    loss.backward(); optimizer_loc_clf.step()
    if (epoch + 1) % (max(1,EPOCHS_CLASSIFICATION // 10)) == 0: print(f'Epoch [{epoch+1}/{EPOCHS_CLASSIFICATION}], Loc Loss: {loss.item():.4f}')

print("\n--- Training Azimuth Regressor (sin/cos) ---")
for epoch in range(EPOCHS_AZIMUTH):
    model_az_reg.train()
    optimizer_az_reg.zero_grad()
    outputs = model_az_reg(X_train_exp)
    loss = criterion_az_reg(outputs, y_az_sincos_train_exp)
    loss.backward(); optimizer_az_reg.step()
    if (epoch + 1) % (max(1,EPOCHS_AZIMUTH // 10)) == 0: print(f'Epoch [{epoch+1}/{EPOCHS_AZIMUTH}], Az (sin/cos) Loss: {loss.item():.4f}')


# --- 6. Evaluation with Averaging for Test Set ---
print("\n--- Evaluating Models on Original Test Samples (with averaging) ---")
results_data = df_test_set_original_samples.copy() 

final_pred_loc_ids = []
final_pred_az_degs = []

true_loc_indices_test_orig = df_test_set_original_samples['location'].map(location_to_index).values
true_az_deg_test_orig = df_test_set_original_samples['azimuth'].values

with torch.no_grad():
    model_loc_clf.eval()
    model_az_reg.eval()

    for i in tqdm(range(len(df_test_set_original_samples)), desc="Evaluating test samples"):
        current_sample_sub_sequences = X_test_orig_sequences_tensor[i]

        # Location Classification
        loc_probs_k = torch.softmax(model_loc_clf(current_sample_sub_sequences), dim=1) 
        avg_loc_probs = torch.mean(loc_probs_k, dim=0) 
        final_pred_loc_idx = torch.argmax(avg_loc_probs).item()
        final_pred_loc_ids.append(index_to_location.get(final_pred_loc_idx, -1)) 

        # Azimuth Regression (sin/cos)
        pred_sincos_k = model_az_reg(current_sample_sub_sequences) 
        avg_sincos = torch.mean(pred_sincos_k, dim=0).cpu().numpy() 
        avg_sin, avg_cos = avg_sincos[0], avg_sincos[1]
        pred_az_rad = np.arctan2(avg_sin, avg_cos)
        pred_az_deg = (np.rad2deg(pred_az_rad) + 360) % 360
        final_pred_az_degs.append(pred_az_deg)

results_data['predicted_location'] = final_pred_loc_ids
results_data['predicted_azimuth'] = final_pred_az_degs


# print("results_data: ", results_data.columns)
# sys.exit(1)

predicted_loc_indices_test = results_data['predicted_location'].map(location_to_index).fillna(-1).astype(int).values
valid_comparison_mask = true_loc_indices_test_orig != np.nan 

if np.sum(~valid_comparison_mask) > 0:
    print(f"Warning: {np.sum(~valid_comparison_mask)} true location IDs could not be mapped to indices for metric calculation.")

final_accuracy = accuracy_score(true_loc_indices_test_orig[valid_comparison_mask], predicted_loc_indices_test[valid_comparison_mask])
print(f"\nFINAL Location Classifier - Averaged Test Accuracy: {final_accuracy*100:.2f}%")

mae_az_final, mse_az_final = calculate_angular_metrics(true_az_deg_test_orig, final_pred_az_degs)
rmse_az_final = np.sqrt(mse_az_final)
print(f"FINAL Azimuth Regressor - Averaged MAE: {mae_az_final:.2f} deg, MSE: {mse_az_final:.2f} deg^2, RMSE: {rmse_az_final:.2f} deg")

# Save results
try:
    # Drop the complex array column before saving
    results_to_save = results_data.drop(columns=['array'], errors='ignore')
    
    # NEW: Convert entropy_list to a string so Feather doesn't crash!
    if 'entropy_list' in results_to_save.columns:
        results_to_save['entropy_list'] = results_to_save['entropy_list'].astype(str)
    
    results_to_save.to_feather(OUTPUT_RESULTS_FILE)
    print(f"\nSaved prediction results to: {OUTPUT_RESULTS_FILE}")
except Exception as e:
    print(f"\nError saving results to feather file: {e}")
    # Also drop for CSV just in case
    results_to_save.to_csv(OUTPUT_RESULTS_FILE.replace(".feather", ".csv"), index=False)
    print(f"Saved prediction results as CSV instead: {OUTPUT_RESULTS_FILE.replace('.feather', '.csv')}")

print("\nScript finished.")



# --- 7. Model Serialization ---
import json
print("\n=== PHASE 5: Model and Taxonomy Serialization ===")
SAVE_DIR = f"./saved_navigation_models_epochs_15000_51"
os.makedirs(SAVE_DIR, exist_ok=True)

# Define export paths
loc_model_path = os.path.join(SAVE_DIR, "model_loc_clf.pth")
az_model_path = os.path.join(SAVE_DIR, "model_az_reg.pth")
mapping_path = os.path.join(SAVE_DIR, "location_mapping.json")

# 1. Serialize Neural Network Weights
torch.save(model_loc_clf.state_dict(), loc_model_path)
torch.save(model_az_reg.state_dict(), az_model_path)


# 2. Serialize Categorical Taxonomy (Critical for Inference)
# JSON strictly requires string keys, so we cast the index keys to strings
mapping_to_save = {str(k): int(v) for k, v in index_to_location.items()}
with open(mapping_path, 'w') as f:
    json.dump(mapping_to_save, f)

print(f"Network state dictionaries serialized to: {SAVE_DIR}")
print(f"Categorical translation matrix saved to: {mapping_path}")
print("Pipeline execution complete.")

