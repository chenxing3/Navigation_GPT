import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys

# --- Set Matplotlib global font properties ---
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# --- Configuration ---
file_location_coords = "/Users/xingchen/Downloads/new_start/checking_results_place_grid_direction/direction_and_angle/batch_statistic_azimuth/ops/step3_add_more_locations_cluster_all.csv"
file_hidden_states = '../evaluation_hidden_states_third_round_7325_Pos_04_0_38.feather'

# --- Load Data ---
print(f"Loading location coordinates from: {file_location_coords}")
df_loc_coords_original = pd.read_csv(file_location_coords)

print(f"Loading hidden states from: {file_hidden_states}")
df_hidden_states = pd.read_feather(file_hidden_states)
df_hidden_states['location'] = df_hidden_states['loc_key']
df_hidden_states['array'] = df_hidden_states['hidden_state']



print("df_hidden_states: ", df_hidden_states)

# sys.exit(1)



df_hidden_states.reset_index(drop=True, inplace=True)

if df_hidden_states.empty:
    raise ValueError("Main dataframe 'df_hidden_states' is empty.")

# ---- Color Generation Logic based on Coordinates (RGB Mapping) ----
if df_loc_coords_original.empty:
    raise ValueError("'df_loc_coords_original' is empty.")
if not all(col in df_loc_coords_original.columns for col in ['x', 'y']):
    raise KeyError("'df_loc_coords_original' must contain 'x', and 'y' columns.")

df_loc_for_color = df_loc_coords_original.copy()
df_loc_for_color['location_id_from_index'] = df_loc_for_color.index

try:
    df_hidden_states['location'] = df_hidden_states['location'].astype(int)
except ValueError as e:
    raise ValueError(f"Could not convert df_hidden_states['location'] to int: {e}.")

unique_locs_in_data = df_hidden_states['location'].unique()
df_loc_relevant = df_loc_for_color[df_loc_for_color['location_id_from_index'].isin(unique_locs_in_data)].copy()

if df_loc_relevant.empty:
    raise ValueError("No relevant locations found for the data in 'df_hidden_states'.")

min_x, max_x = df_loc_relevant['x'].min(), df_loc_relevant['x'].max()
min_y, max_y = df_loc_relevant['y'].min(), df_loc_relevant['y'].max()
delta_x = max_x - min_x if max_x > min_x else 1.0
delta_y = max_y - min_y if max_y > min_y else 1.0
df_loc_relevant['r_channel'] = ((df_loc_relevant['y'] - min_y) / delta_y * 255).astype(int)
df_loc_relevant['g_channel'] = ((df_loc_relevant['x'] - min_x) / delta_x * 255).astype(int)
BLUE_CHANNEL_CONSTANT = 128
df_loc_relevant['hex_color'] = [
    f'#{r:02x}{g:02x}{b:02x}' for r, g, b in zip(
        df_loc_relevant['r_channel'],
        df_loc_relevant['g_channel'],
        [BLUE_CHANNEL_CONSTANT] * len(df_loc_relevant)
    )
]
location_to_hex_color = pd.Series(
    df_loc_relevant['hex_color'].values,
    index=df_loc_relevant['location_id_from_index']
).to_dict()

# ---- Prepare data for t-SNE and Plotting ----
df_hidden_states['hex_color'] = df_hidden_states['location'].map(location_to_hex_color)
if df_hidden_states['hex_color'].isnull().any():
    missing_color_locs = df_hidden_states[df_hidden_states['hex_color'].isnull()]['location'].unique()
    print(f"Warning: Locations {missing_color_locs} had no color. They will be colored by default.")
    df_hidden_states['hex_color'].fillna('#808080', inplace=True)
hidden_states_np = np.vstack(df_hidden_states['array'].values)

# --- Run TSNE loop ---
for perplexity_value in [30]:
    print(f"Running t-SNE with perplexity: {perplexity_value}")
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value,
                learning_rate='auto', init='pca')
    embeddings_2d = tsne.fit_transform(hidden_states_np)

    df_plot = df_hidden_states.copy()
    df_plot['tsne_dim1'] = embeddings_2d[:, 0]
    df_plot['tsne_dim2'] = embeddings_2d[:, 1]
    
    df_centroids = df_plot.groupby('location').agg(
        mean_tsne_dim1=('tsne_dim1', 'mean'),
        mean_tsne_dim2=('tsne_dim2', 'mean')
    ).reset_index()
    df_centroids['label_text_centroid'] = df_centroids['location'].astype(str)

    # ---- Matplotlib Plotting ----
    
    # 1. Create figure and axes objects. Adjusted size for legend.
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 2. Create the scatter plot
    ax.scatter(
        x=df_plot['tsne_dim1'],
        y=df_plot['tsne_dim2'],
        c=df_plot['hex_color'],
        s=16,
        edgecolors='DarkSlateGrey',
        linewidths=0.5
    )

    # 3. Add text labels at centroids
    for _, row in df_centroids.iterrows():
        ax.text(
            x=row['mean_tsne_dim1'],
            y=row['mean_tsne_dim2'] + 1,
            s=row['label_text_centroid'],
            fontdict={'size': 12, 'color': 'black'},
            ha='center',
            va='bottom'
        )

    # ==================== LEGEND INSERTION START ====================
    # Create an inset axis for the 2D color legend.
    # The parameters are [left, bottom, width, height] in figure coordinates.
    ax_legend = fig.add_axes([0.75, 0.7, 0.1, 0.2])
    
    # Generate the color grid for the legend.
    N = 100  # Resolution of the legend grid
    g_channel = np.linspace(0, 255, N).astype(np.uint8) # Corresponds to X-coordinate
    r_channel = np.linspace(0, 255, N).astype(np.uint8) # Corresponds to Y-coordinate
    color_grid = np.zeros((N, N, 3), dtype=np.uint8)
    
    color_grid[:, :, 0] = r_channel[:, np.newaxis] # Red channel varies along Y-axis
    color_grid[:, :, 1] = g_channel[np.newaxis, :] # Green channel varies along X-axis
    color_grid[:, :, 2] = BLUE_CHANNEL_CONSTANT      # Blue channel is constant
    
    ax_legend.imshow(color_grid, origin='lower', aspect='auto')
    ax_legend.set_title('Color legend (B=128)', fontsize=10)
    ax_legend.set_xlabel('Longitude', fontsize=8)
    ax_legend.set_ylabel('Latitude', fontsize=8)
    ax_legend.set_xticks([]) # Hide tick numbers for a cleaner look
    ax_legend.set_yticks([]) # Hide tick numbers
    # ===================== LEGEND INSERTION END =====================

    # 4. Configure plot appearance
    ax.set_xlabel('t-SNE Dim 1')
    ax.set_ylabel('t-SNE Dim 2')
    ax.set_facecolor('white')
    ax.grid(True, color='lightgrey', linestyle='--', linewidth=1)
    
    # 5. Configure borders (spines)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
        
    # 6. Display the plot
    plt.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust layout to prevent legend overlap
    plt.show()