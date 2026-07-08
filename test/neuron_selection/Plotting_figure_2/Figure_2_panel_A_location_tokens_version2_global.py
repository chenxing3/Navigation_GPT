import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.manifold import TSNE
import os

import pyproj
from shapely.geometry import Point
from shapely.ops import transform
from pyproj import Transformer

# --- Configuration ---
# Path to the CSV file with original token coordinates (x, y)
file_location_coords = './hex_centers_50_0_extend_no_GL_version2.csv' # ADJUST AS NEEDED

# Path to the Feather file with hidden states
file_hidden_states = './global_tokens/layer_24_hidden_states.feather' # Updated target
output_2d_plot_filename = f"tsne_2D_Layer24_RoostLocal800m——24_local_more.html"



# For 2D t-SNE
perplexity_2d = 50
tsne_n_iter_2d = 1000
tsne_random_state_2d = 42

# # For 3D t-SNE (Adding missing declarations from original script)
# perplexity_3d_loc = 50
# tsne_random_state_3d_loc = 42

# Constant value for the Blue color channel (0-255)
BLUE_CHANNEL_CONSTANT = 128

# Set Roost Coordinates and Filter Radius
roost_lon, roost_lat = 35.583259619555800, 33.111815850574700
buffer_meters = 800

# Projection Setup
proj_meters = pyproj.CRS('EPSG:6991')
proj_latlng = pyproj.CRS('EPSG:4326')
project_to_meters = Transformer.from_crs(proj_latlng, proj_meters, always_xy=True).transform

def coordinate_to_xy(lon, lat):
    pt_meters = transform(project_to_meters, Point(lon, lat))
    return pt_meters.x, pt_meters.y

# --- 1. Load Data ---
print(f"Loading location coordinates from: {file_location_coords}")
if not os.path.exists(file_location_coords):
    raise FileNotFoundError(f"Coordinate file not found: {file_location_coords}")
df_loc_coords_original = pd.read_csv(file_location_coords)

print(f"Loading hidden states from: {file_hidden_states}")
if not os.path.exists(file_hidden_states):
    raise FileNotFoundError(f"Hidden states file not found: {file_hidden_states}")
df_hidden_states = pd.read_feather(file_hidden_states)




location_ids = []
for token in df_hidden_states['token']:
    if token.startswith('[LOC_'):
        try:
            loc_id = int(token.split('_')[1].rstrip(']'))
            location_ids.append(loc_id)
        except ValueError:
            location_ids.append(-1)  # Mark invalid tokens with -1
    else:
        location_ids.append(-1)  # Non-location tokens also marked with -1

df_hidden_states['location_id'] = location_ids
# only select the location < 1616 and >= 0
# df_hidden_states = df_hidden_states[(df_hidden_states['location_id'] >= 0) & (df_hidden_states['location_id'] < 1616)].copy()



# --- 2. Validate and Prepare Data ---
df_loc_coords_original['Token'] = df_loc_coords_original['text'].astype(str)
df_hidden_states['token'] = df_hidden_states['token'].astype(str)

# --- 3. Merge DataFrames ---
df_merged = pd.merge(
    df_hidden_states,
    df_loc_coords_original[['Token', 'x', 'y', 'Type']],
    left_on='token',
    right_on='Token',
    how='left'
)

if df_merged['Type'].isnull().any():
    print("Warning: Some tokens have missing 'Type'. Inferring from token name.")
    def infer_type_from_token(token_str):
        if pd.isna(token_str): return 'UNKNOWN'
        if token_str.startswith('LOC_'): return 'LOC'
        if token_str.startswith('GRID_'): return 'GRID'
        return 'UNKNOWN'
    df_merged['Type'] = df_merged.apply(lambda row: infer_type_from_token(row['token']) if pd.isna(row['Type']) else row['Type'], axis=1)

# --- NEW 3.5: Geographic Filtering (800m around the roost) ---
roost_x, roost_y = coordinate_to_xy(roost_lon, roost_lat)
df_merged['dist_to_roost'] = np.sqrt((df_merged['x'] - roost_x)**2 + (df_merged['y'] - roost_y)**2)
df_merged = df_merged[df_merged['dist_to_roost'] <= buffer_meters].copy()
print(f"Filtered data to {len(df_merged)} locations within {buffer_meters}m of the roost.")

# --- 4. Generate Direct RGB Color Mapping ---
print("Generating RGB color values from coordinates...")
df_for_color_calc = df_merged.dropna(subset=['x', 'y'])

if df_for_color_calc.empty:
    df_merged['rgb_color'] = 'rgb(128, 128, 128)'
else:
    min_x, max_x = df_for_color_calc['x'].min(), df_for_color_calc['x'].max()
    min_y, max_y = df_for_color_calc['y'].min(), df_for_color_calc['y'].max()
    
    delta_x = max_x - min_x if max_x > min_x else 1.0
    delta_y = max_y - min_y if max_y > min_y else 1.0

    norm_y_255 = ((df_merged['y'] - min_y) / delta_y * 255).fillna(128).astype(int) 
    norm_x_255 = ((df_merged['x'] - min_x) / delta_x * 255).fillna(128).astype(int) 

    df_merged['rgb_color'] = [
        f'rgb({r},{g},{b})' for r, g, b in zip(
            norm_y_255,                 
            norm_x_255,                 
            [BLUE_CHANNEL_CONSTANT] * len(df_merged) 
        )
    ]

# --- 5. Prepare for 2D t-SNE ---
if not df_merged.empty:
    hidden_states_all_np = np.vstack(df_merged['hidden_state_array'].values)
    
    # Adjust perplexity if the number of samples is smaller than the requested perplexity
    current_perplexity = min(perplexity_2d, len(df_merged) - 1) if len(df_merged) > 1 else 1
    
    print(f"Running 2D t-SNE for filtered data (Perplexity: {current_perplexity})")
    tsne_2d_all = TSNE(n_components=2, random_state=tsne_random_state_2d, perplexity=current_perplexity,
                       learning_rate='auto', init='pca', n_jobs=-1)
    embeddings_2d_all = tsne_2d_all.fit_transform(hidden_states_all_np)

    df_plot_2d = df_merged.copy()
    df_plot_2d['tsne_dim1'] = embeddings_2d_all[:, 0]
    df_plot_2d['tsne_dim2'] = embeddings_2d_all[:, 1]

    df_plot_grid_2d = df_plot_2d[df_plot_2d['Type'] == 'GRID'].copy()
    df_plot_loc_2d = df_plot_2d[df_plot_2d['Type'] == 'LOC'].copy()

# # --- 6. Prepare and run 3D t-SNE for LOC tokens ONLY ---
# df_loc_only_for_3d = df_merged[df_merged['Type'] == 'LOC'].copy()
# if not df_loc_only_for_3d.empty and len(df_loc_only_for_3d) > 3:
#     hidden_states_loc_np = np.vstack(df_loc_only_for_3d['hidden_state_array'].values)
#     current_perplexity_3d = min(perplexity_3d_loc, len(df_loc_only_for_3d) - 1)
    
#     print(f"Running 3D t-SNE for LOC data (Perplexity: {current_perplexity_3d})")
#     tsne_3d_loc = TSNE(n_components=3, random_state=tsne_random_state_3d_loc, perplexity=current_perplexity_3d,
#                        learning_rate='auto', init='pca', n_jobs=-1)
#     embeddings_3d_loc = tsne_3d_loc.fit_transform(hidden_states_loc_np)

#     df_loc_only_for_3d['tsne_3d_dim1'] = embeddings_3d_loc[:, 0]
#     df_loc_only_for_3d['tsne_3d_dim2'] = embeddings_3d_loc[:, 1]
#     df_loc_only_for_3d['tsne_3d_dim3'] = embeddings_3d_loc[:, 2]
# else:
#     print("Not enough LOC tokens found to generate a 3D plot.")

# --- 7. Generate 2D Subplot (LOC and GRID) ---
print("Generating 2D subplot visualization...")
fig_2d_subplots = make_subplots(
    rows=1, cols=1,
    subplot_titles=(f" ", f"2D t-SNE of GRID Tokens (P={perplexity_2d})"),
    shared_yaxes=True
)

hovertemplate_2d_str = ("<b>Token: %{customdata[0]}</b><br>" +
                        "Original X: %{customdata[1]:.2f}<br>" +
                        "Original Y: %{customdata[2]:.2f}<br>" +
                        "Type: %{customdata[3]}<br>" +
                        "Color: %{marker.color}<br>" + 
                        "t-SNE Dim1: %{x:.2f}<br>" +
                        "t-SNE Dim2: %{y:.2f}<extra></extra>")

if 'df_plot_loc_2d' in locals() and not df_plot_loc_2d.empty:
    fig_2d_subplots.add_trace(
        go.Scatter(
            x=df_plot_loc_2d['tsne_dim1'], y=df_plot_loc_2d['tsne_dim2'], mode='markers',
            marker=dict(
                color=df_plot_loc_2d['rgb_color'], 
                size=8, # Slightly increased size for better visibility on local map
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            customdata=np.stack((df_plot_loc_2d['token'], df_plot_loc_2d['x'], df_plot_loc_2d['y'], df_plot_loc_2d['Type']), axis=-1),
            hovertemplate=hovertemplate_2d_str, name='LOC Tokens (2D)'
        ), row=1, col=1
    )

fig_2d_subplots.update_layout(
    title_text=f'2D t-SNE of Hidden States (Layer 23, Local 800m Zone)', title_font_size=16, title_x=0.5,
    width=600, height=600, plot_bgcolor='white', showlegend=False
)
# fig_2d_subplots.update_xaxes(title_text='t-SNE Dim 1', showline=True, linewidth=1, linecolor='black', showgrid=True, gridwidth=0.5, gridcolor='lightgrey', zeroline=False, mirror=True)
# fig_2d_subplots.update_yaxes(title_text='t-SNE Dim 2', showline=True, linewidth=1, linecolor='black', showgrid=True, gridwidth=0.5, gridcolor='lightgrey', zeroline=False, mirror=True)
fig_2d_subplots.update_layout(
    xaxis_title='t-SNE Dim 1',
    yaxis_title='t-SNE Dim 2',
    xaxis_title_font_size=36,
    yaxis_title_font_size=36,
    font_size=28,
    width=800,
    height=700,
    plot_bgcolor='white',
    xaxis=dict(showline=True, linewidth=2, linecolor='black', showgrid=True, gridwidth=1, gridcolor='lightgrey', zeroline=False, mirror=True),
    yaxis=dict(showline=True, linewidth=2, linecolor='black', showgrid=True, gridwidth=1, gridcolor='lightgrey', zeroline=False, mirror=True),
    coloraxis_colorbar=dict(title_font_size=16, tickfont_size=14, len=0.6, thickness=20, y=1.0, yanchor='top', x=1.02, xanchor='left'),
    legend=dict(title_font_size=16, font_size=14, x=1.02, y=(1.0 - 0.6 - 0.05), xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.7)', bordercolor='Black', borderwidth=1)
)


fig_2d_subplots.write_html(output_2d_plot_filename)
print(f"2D Subplot saved to {output_2d_plot_filename}")

# # --- 8. Generate Separate 3D Plot for LOC Tokens ---
# if not df_loc_only_for_3d.empty and 'tsne_3d_dim1' in df_loc_only_for_3d.columns:
#     print("Generating 3D plot for LOC tokens...")
#     fig_3d_loc = go.Figure()
#     fig_3d_loc.add_trace(
#         go.Scatter3d(
#             x=df_loc_only_for_3d['tsne_3d_dim1'],
#             y=df_loc_only_for_3d['tsne_3d_dim2'],
#             z=df_loc_only_for_3d['tsne_3d_dim3'],
#             mode='markers',
#             marker=dict(
#                 size=6,
#                 color=df_loc_only_for_3d['rgb_color'], 
#                 opacity=0.8
#             ),
#             customdata=np.stack((
#                 df_loc_only_for_3d['token'],
#                 df_loc_only_for_3d['x'],
#                 df_loc_only_for_3d['y'],
#                 df_loc_only_for_3d['Type']
#             ), axis=-1),
#             hovertemplate=("<b>Token: %{customdata[0]}</b><br>" +
#                            "Original X: %{customdata[1]:.2f}<br>" +
#                            "Original Y: %{customdata[2]:.2f}<br>" +
#                            "Type: %{customdata[3]}<br>" +
#                            "Color: %{marker.color}<br>" + 
#                            "Dim1: %{x:.2f}, Dim2: %{y:.2f}, Dim3: %{z:.2f}<extra></extra>")
#         )
#     )
#     fig_3d_loc.update_layout(
#         title=f'3D t-SNE of LOC Tokens (Layer 23, Local 800m Zone)',
#         width=1000, height=800,
#         margin=dict(l=0, r=0, b=0, t=40),
#         scene=dict(
#             xaxis_title='t-SNE Dimension 1',
#             yaxis_title='t-SNE Dimension 2',
#             zaxis_title='t-SNE Dimension 3',
#             aspectmode='cube'
#         ),
#         scene_camera=dict(
#             eye=dict(x=1.8, y=1.8, z=0.5)
#         )
#     )
#     output_3d_plot_filename = f"tsne_3D_Layer24_RoostLocal800m——2.html"
#     fig_3d_loc.write_html(output_3d_plot_filename)
#     print(f"3D LOC Plot saved to {output_3d_plot_filename}")