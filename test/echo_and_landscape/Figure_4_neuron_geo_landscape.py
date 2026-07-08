import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # For creating custom legends
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting
from sklearn.manifold import TSNE
import sys



# --- MODIFICATION: Set Matplotlib global font properties ---
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 20  # Font size for x and y labels
plt.rcParams['xtick.labelsize'] = 16 # Font size for x tick labels
plt.rcParams['ytick.labelsize'] = 16 # Font size for y tick labels

# --- Configuration ---
file_hidden_states = '../evaluation_hidden_states_third_round_7325_Pos_04_0_38.feather'
file_landscape_annotations = "/Users/xingchen/Downloads/submit_paper/Navigation_LLM/analysis/lanscape/Landscape_annotation3.xlsx"

# --- Load and Process Data (No changes in this section) ---
print("Loading and processing data...")
df_hidden_states = pd.read_feather(file_hidden_states)

df_hidden_states.reset_index(drop=True, inplace=True)
df_hidden_states['location'] = df_hidden_states['loc_key']
df_hidden_states['array'] = df_hidden_states['hidden_state']


df_landscapes = pd.read_excel(file_landscape_annotations)

if df_hidden_states.empty or df_landscapes.empty:
    raise ValueError("One of the dataframes is empty.")

landscape_columns = ['Hill', 'Forest', 'Rural settlement', 'Water', 'Orchards', 'Field_corp', 'Road', 'River', 'Swamp']
if 'target_loc' not in df_landscapes.columns:
    raise KeyError("'target_loc' column not found in landscape annotations.")

df_landscapes_indexed = df_landscapes.set_index('target_loc')
location_to_landscape_map = df_landscapes_indexed[landscape_columns].idxmax(axis=1)
all_false_mask = (df_landscapes_indexed[landscape_columns] == False).all(axis=1)
location_to_landscape_map[all_false_mask] = 'None'
location_to_landscape_map_dict = location_to_landscape_map.to_dict()

df_hidden_states['target_loc'] = "LOC_" + df_hidden_states['location'].astype(str).str.zfill(5) + "_"
df_hidden_states['landscape_type'] = df_hidden_states['target_loc'].map(location_to_landscape_map_dict)
df_hidden_states['landscape_type'].fillna('Unknown', inplace=True)
print("Data processing complete.")

# --- Prepare data for t-SNE ---
hidden_states_np = np.vstack(df_hidden_states['array'].values)

# --- MODIFICATION: Create a consistent color map for landscapes ---
# This ensures that 'Forest' is the same color in every plot.
unique_landscapes = sorted(df_hidden_states['landscape_type'].unique())
# Using a qualitative colormap suitable for categorical data
colors = plt.cm.get_cmap('tab10', len(unique_landscapes))
landscape_to_color_map = {landscape: colors(i) for i, landscape in enumerate(unique_landscapes)}


# ---- MATPLOTLIB 3D t-SNE and Plotting Loop ----
for perplexity_value in [5]:
    print(f"\nRunning 3D t-SNE with perplexity: {perplexity_value}")
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=perplexity_value,
                   learning_rate='auto', init='pca')
    embeddings_3d = tsne_3d.fit_transform(hidden_states_np)

    df_plot = df_hidden_states.copy()
    df_plot['tsne_dim1'] = embeddings_3d[:, 0]
    df_plot['tsne_dim2'] = embeddings_3d[:, 1]
    df_plot['tsne_dim3'] = embeddings_3d[:, 2]
    df_plot['plot_color'] = df_plot['landscape_type'].map(landscape_to_color_map)

    df_centroids = df_plot.groupby(['target_loc']).agg(
        mean_tsne_dim1=('tsne_dim1', 'mean'),
        mean_tsne_dim2=('tsne_dim2', 'mean'),
        mean_tsne_dim3=('tsne_dim3', 'mean')
    ).reset_index()

    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8.5))
    ax = fig.add_subplot(111, projection='3d')

    # Create the scatter plot
    ax.scatter(df_plot['tsne_dim1'], df_plot['tsne_dim2'], df_plot['tsne_dim3'],
               c=df_plot['plot_color'],
               s=20, # Equivalent to Plotly's size=4
            #    edgecolors='DarkSlateGrey',
               linewidths=0.5)

    # Add text labels (optional, can be cluttered in 3D)
    # for _, row in df_centroids.iterrows():
    #     label = row['target_loc'].replace("LOC_000", "").replace("_", "")
    #     ax.text(row['mean_tsne_dim1'], row['mean_tsne_dim2'], row['mean_tsne_dim3'], label, size=8)

    # Create a custom legend
    legend_patches = [mpatches.Patch(color=color, label=label)
                      for label, color in landscape_to_color_map.items()]
    ax.legend(handles=legend_patches, title='Landscape Type', fontsize=10, bbox_to_anchor=(1.15, 1))

    # Styling
    ax.set_title(f'3D t-SNE by Landscape (Perplexity: {perplexity_value})', fontsize=16)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_zlabel('t-SNE Dimension 3', fontsize=12)
    ax.view_init(elev=20., azim=45) # Set camera angle
    fig.tight_layout()
    
    # Save and show
    save_path = f'tsne_3d_landscape_perplexity_{perplexity_value}.png'
    plt.savefig(save_path, dpi=300)
    plt.show()


# ---- MATPLOTLIB 2D t-SNE and Plotting Loop ----
for perplexity_value in [30]:
    print(f"\nRunning 2D t-SNE with perplexity: {perplexity_value}")
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=perplexity_value,
                   learning_rate='auto', init='pca')
    embeddings_2d = tsne_2d.fit_transform(hidden_states_np)

    df_plot = df_hidden_states.copy()
    df_plot['tsne_dim1'] = embeddings_2d[:, 0]
    df_plot['tsne_dim2'] = embeddings_2d[:, 1]
    df_plot['plot_color'] = df_plot['landscape_type'].map(landscape_to_color_map)

    df_centroids = df_plot.groupby('target_loc').agg(
        mean_tsne_dim1=('tsne_dim1', 'mean'),
        mean_tsne_dim2=('tsne_dim2', 'mean')
    ).reset_index()

    # Create the 2D plot
    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Create the scatter plot
    ax.scatter(df_plot['tsne_dim1'], df_plot['tsne_dim2'],
               c=df_plot['plot_color'],
               s=15, # Equivalent to Plotly's size=8
            #    edgecolors='DarkSlateGrey',
               linewidths=0.5)

    # Add text labels at centroids
    for _, row in df_centroids.iterrows():
        label = row['target_loc'].replace("LOC_000", "").replace("_", "")
        ax.text(row['mean_tsne_dim1'], row['mean_tsne_dim2']+1.5, label,
                ha='center', va='bottom', fontsize=10, color='black')

        
    # Create a custom legend
    legend_patches = [mpatches.Patch(color=color, label=label)
                      for label, color in landscape_to_color_map.items()]
    ax.legend(handles=legend_patches, fontsize=12)

    # Styling
    # ax.set_title(f'2D t-SNE by Landscape (Perplexity: {perplexity_value})', fontsize=16)
    # 4. Configure plot appearance
    ax.set_xlabel('t-SNE Dim 1')
    ax.set_ylabel('t-SNE Dim 2')
    ax.set_facecolor('white')
    ax.grid(True, color='lightgrey', linestyle='--', linewidth=1)
    
    # ax.set_xlim(-30, 30)

    # 5. Configure borders (spines)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)


    # Save and show
    save_path = f'tsne_2d_landscape_perplexity_{perplexity_value}.png'
    plt.savefig(save_path, dpi=300)
    plt.show()

print("Processing complete.")