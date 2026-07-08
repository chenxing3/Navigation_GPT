import pandas as pd
import numpy as np
import math
import ast
import os
import sys

import pyproj
from shapely.geometry import Point
from shapely.ops import transform
from pyproj import Transformer

import plotly.graph_objects as go
import plotly.colors

# --- Configuration for Multiple Trajectories ---
# Adjust paths and row indices as needed
TRAJECTORY_CONFIGS = [
    {
        'file': '../dataset/trajectory/full_model/homing_trajectory/trajectories_iterative_with_echo_to_home_20251217_142826.feather',
        'row_indices': [2], # Which rows to plot from this file
        'color': plotly.colors.qualitative.Plotly[0], # Example: 'blue',
        'name_prefix': 'File1'
    },
    {
        'file': '../dataset/trajectory/full_model/homing_trajectory/trajectories_iterative_with_echo_to_home_20251217_142956.feather',
        'row_indices': [48], # Which rows to plot from this file
        'color': plotly.colors.qualitative.Plotly[1], # Example: 'blue',
        'name_prefix': 'File1'
    },
    {
        'file': '../dataset/trajectory/full_model/homing_trajectory/trajectories_iterative_with_echo_to_home_20251217_143316.feather',
        'row_indices': [15], # Which rows to plot from this file
        'color': plotly.colors.qualitative.Plotly[2], # Example: 'blue',
        'name_prefix': 'File1'
    },
    {
        'file': '../dataset/trajectory/full_model/homing_trajectory/trajectories_iterative_with_echo_to_home_20251217_142455.feather',
        'row_indices': [24],
        'color': plotly.colors.qualitative.Plotly[4], # Example: 'green',
        'name_prefix': 'File2'
    },
    {
        'file': '../dataset/trajectory/full_model/homing_trajectory/trajectories_iterative_with_echo_20260416_153756.feather',
        'row_indices': [9],
        'color': plotly.colors.qualitative.Plotly[5], # Example: 'green',
        'name_prefix': 'File2'
    },
    {
        'file': '../dataset/trajectory/full_model/homing_trajectory/trajectories_iterative_with_echo_to_home_20251217_142458.feather',
        'row_indices': [7],
        'color': plotly.colors.qualitative.Plotly[6], # Example: 'green',
        'name_prefix': 'File2'
    }
]

HEX_CENTERS_CSV = '../ops/token_info.csv'
AZIMUTH_STEP_DISTANCE = 5

# (Keep your CRS functions and other helpers as they are)
geodesic = pyproj.Geod(ellps='WGS84')
proj_meters = pyproj.CRS('EPSG:6991')
proj_latlng = pyproj.CRS('EPSG:4326')
project_to_coords = Transformer.from_crs(proj_meters, proj_latlng, always_xy=True).transform
project_to_meters = Transformer.from_crs(proj_latlng, proj_meters, always_xy=True).transform

def coordinate_to_xy(lon, lat):
    pt_meters = transform(project_to_meters, Point(lon, lat))
    return pt_meters.x, pt_meters.y

def xy_to_coordinate(x, y):
    if x is None or y is None or math.isnan(x) or math.isnan(y): return None, None
    try:
        pt_latlng = transform(project_to_coords, Point(x, y))
        return pt_latlng.x, pt_latlng.y
    except Exception: return None, None

def calculate_trajectory_coordinates(x_start, y_start, azimuth_sequence, distance):
    try:
        azimuth_sequence = [float(az) for az in azimuth_sequence if str(az).strip()]
    except (ValueError, TypeError) as e:
        # print(f"Error converting azimuths: {e}. Seq: {azimuth_sequence}")
        return [], []
    azimuth_sequence_rad = [math.radians(azimuth) for azimuth in azimuth_sequence]
    x_curr, y_curr = x_start, y_start
    x_traj, y_traj = [x_start], [y_start]
    for azimuth_rad in azimuth_sequence_rad:
        delta_x = distance * math.sin(azimuth_rad)
        delta_y = distance * math.cos(azimuth_rad)
        x_curr += delta_x
        y_curr += delta_y
        x_traj.append(x_curr)
        y_traj.append(y_curr)
    return x_traj, y_traj

# --- Function to add a single trajectory to a regular Plotly chart ---
def add_trajectory_to_chart_fig(fig, results_file, hex_centers_lookup, row_index, trajectory_color='red', name_prefix='Trajectory', first_of_kind=False):
    plotted_x_coords, plotted_y_coords = [], []
    if not os.path.exists(results_file):
        # print(f"Error: Results file not found: {results_file} for row {row_index}")
        return plotted_x_coords, plotted_y_coords
    df_results = pd.read_feather(results_file)
    if row_index < 0 or row_index >= len(df_results):
        # print(f"Error: Row index {row_index} for {name_prefix} out of bounds.")
        return plotted_x_coords, plotted_y_coords
    trajectory_data = df_results.iloc[row_index]


    print("trajectory_data initial_start_loc_text: ", trajectory_data['initial_start_loc_text'])

    # sys.exit(1)
    try:
        loc1_x = trajectory_data['loc1_x']; loc1_y = trajectory_data['loc1_y']
        loc2_x = trajectory_data['loc2_x']; loc2_y = trajectory_data['loc2_y']
        azimuths_data_str = trajectory_data['azimuths_tokens_str']
        middle_loc_texts_str = trajectory_data.get('middle_locations_texts', "")
        if pd.isna(middle_loc_texts_str): middle_loc_texts_str = ""
    except KeyError as e:
        # print(f"Error: Missing column for {name_prefix} Row {row_index}: {e}")
        return plotted_x_coords, plotted_y_coords
    if pd.isna(loc1_x) or pd.isna(loc1_y) or pd.isna(loc2_x) or pd.isna(loc2_y):
        # print(f"Error: Start/End XY missing/NaN for {name_prefix} Row {row_index}.")
        return plotted_x_coords, plotted_y_coords
    plotted_x_coords.extend([loc1_x, loc2_x]); plotted_y_coords.extend([loc1_y, loc2_y])
    if not isinstance(azimuths_data_str, str): azimuths_data_str = str(azimuths_data_str)
    azimuths = [i.replace("Azi_", "").replace("_", "") for i in azimuths_data_str.split(" ") if i.strip()]
    traj_path_x, traj_path_y = calculate_trajectory_coordinates(loc1_x, loc1_y, azimuths, AZIMUTH_STEP_DISTANCE)
    if traj_path_x: plotted_x_coords.extend(traj_path_x); plotted_y_coords.extend(traj_path_y)
    middle_loc_ids = []
    if isinstance(middle_loc_texts_str, str) and middle_loc_texts_str.strip():
        middle_loc_ids = [loc_id.replace(" ", "").strip() for loc_id in middle_loc_texts_str.split(',') if loc_id.strip()]
    middle_loc_plot_x, middle_loc_plot_y, middle_loc_plot_labels = [], [], []
    if middle_loc_ids and hex_centers_lookup:
        for loc_id_token in middle_loc_ids:
            found = False
            if loc_id_token in hex_centers_lookup:
                coords = hex_centers_lookup[loc_id_token]
                middle_loc_plot_x.append(coords['x']); middle_loc_plot_y.append(coords['y'])
                middle_loc_plot_labels.append(loc_id_token); found = True
            else:
                cleaned_loc_id = loc_id_token.rstrip('_')
                if cleaned_loc_id != loc_id_token and cleaned_loc_id in hex_centers_lookup:
                    coords = hex_centers_lookup[cleaned_loc_id]
                    middle_loc_plot_x.append(coords['x']); middle_loc_plot_y.append(coords['y'])
                    middle_loc_plot_labels.append(loc_id_token); found = True
            # if not found: print(f"Warn: Mid Loc ID '{loc_id_token}' for {name_prefix} R{row_index} not found.")
    if middle_loc_plot_x: plotted_x_coords.extend(middle_loc_plot_x); plotted_y_coords.extend(middle_loc_plot_y)

    # Plot Azimuth Trajectory Path
    if traj_path_x:
        fig.add_trace(go.Scatter(
            x=traj_path_x, y=traj_path_y, mode='lines', # Only lines for actual path
            line=dict(width=4, color=trajectory_color),
            hoverinfo='skip', # Skip hover for these repeated traces
            showlegend=False # No individual legend for each path
        ))
    # Plot Start and End Locations
    fig.add_trace(go.Scatter(
        x=[loc1_x], y=[loc1_y], mode='markers',
        marker=dict(size=24, color="black", symbol='star'), # Symbol for start
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=[loc2_x], y=[loc2_y], mode='markers',
        marker=dict(size=24, color="black", symbol='x'),    # Symbol for end
        showlegend=False, hoverinfo='skip'
    ))

    if middle_loc_plot_x:
        fig.add_trace(go.Scatter(
            x=middle_loc_plot_x, y=middle_loc_plot_y, mode='markers',
            marker=dict(
                size=14, 
                color="black", 
                symbol='circle', 
                opacity=1.0,                         # Increased to 1.0 for solid contrast
                line=dict(color='white', width=2)    # <-- THIS ADDS THE WHITE BORDER
            ), 
            textfont=dict(size=9, color=trajectory_color) if first_of_kind else None,
            showlegend=False, hoverinfo='skip'
        ))


    return plotted_x_coords, plotted_y_coords
# --- Main Plotting Logic for Multiple Trajectories on a Transparent Chart ---
def plot_all_trajectories_on_transparent_chart(trajectory_configs, hex_centers_csv_path, output_filename="transparent_trajectories.png"):
    fig = go.Figure()
    all_x_coords, all_y_coords = [], []

    hex_centers_lookup = {}
    if os.path.exists(hex_centers_csv_path):
        try:
            df_hex_centers = pd.read_csv(hex_centers_csv_path)
            if all(col in df_hex_centers.columns for col in ['text', 'x', 'y']):
                hex_centers_lookup = df_hex_centers.set_index('text').to_dict('index')
        except Exception as e: print(f"Error loading {hex_centers_csv_path}: {e}.")

    # --- Add Dummy Traces for Simplified Legend ---
    dummy_x = [-1e9]; dummy_y = [-1e9]
    legend_item_color = 'black' # Color of the symbols/lines in the legend
    legend_marker_size = 44     # Increased marker size for legend
    legend_line_width = 3       # Increased line width for legend

    fig.add_trace(go.Scatter(x=dummy_x, y=dummy_y, mode='lines',
                             line=dict(color=legend_item_color, width=legend_line_width), 
                             name='Trajectory'))
    fig.add_trace(go.Scatter(x=dummy_x, y=dummy_y, mode='markers',
                             marker=dict(color=legend_item_color, size=legend_marker_size, symbol='star'), 
                             name='Origin'))
    fig.add_trace(go.Scatter(x=dummy_x, y=dummy_y, mode='markers',
                             marker=dict(color=legend_item_color, size=legend_marker_size, symbol='x'), 
                             name='Destination (roost)'))

    fig.add_trace(go.Scatter(
        x=dummy_x, 
        y=dummy_y, 
        mode='markers',
        marker=dict(
            color='black', 
            size=legend_marker_size,
            symbol='circle', 
            opacity=1.0,
            line=dict(color='#d3d3d3', width=3) # Changed width from 20 to 3
        ), 
        name='Sensory location'
    ))

    # --- End Dummy Traces ---

    # (Loop through configs and call add_trajectory_to_chart_fig - same as before)
    first_trajectory_processed = True 
    for config_idx, config in enumerate(trajectory_configs):
        results_file = config['file']; color = config['color']; name_prefix = config['name_prefix']
        for row_idx_in_config, row_idx_val in enumerate(config['row_indices']):
            is_first_middle_point_overall = (config_idx == 0 and row_idx_in_config == 0)
            x_coords, y_coords = add_trajectory_to_chart_fig(
                fig, results_file, hex_centers_lookup, row_idx_val,
                trajectory_color=color, name_prefix=name_prefix,
                first_of_kind=is_first_middle_point_overall 
            )
            all_x_coords.extend(x_coords)
            all_y_coords.extend(y_coords)
            if x_coords: 
                first_trajectory_processed = False
    
    # (Axis range calculations - same as before)
    if not all_x_coords or not all_y_coords:
        min_x, max_x, min_y, max_y = 0, 1000, 0, 1000
    else:
        min_x, max_x = min(all_x_coords), max(all_x_coords)
        min_y, max_y = min(all_y_coords), max(all_y_coords)
    x_padding = (max_x - min_x) * 0.05 if (max_x - min_x) > 0 else 50
    y_padding = (max_y - min_y) * 0.05 if (max_y - min_y) > 0 else 50
    plot_min_x, plot_max_x = min_x - x_padding, max_x + x_padding
    plot_min_y, plot_max_y = min_y - y_padding, max_y + y_padding

    # (Scale Bar addition - same as before)
    if all_x_coords and all_y_coords:
        data_width_m = plot_max_x - plot_min_x
        target_scale_m = data_width_m / 7
        possible_scales = [2000, 5000, 10000, 20000]
        scale_bar_length_m = min(possible_scales, key=lambda x: abs(x - target_scale_m) if x <= target_scale_m * 1.5 else float('inf'))
        if scale_bar_length_m == 0 and target_scale_m > 0:
             scale_bar_length_m = min(s for s in possible_scales if s > 0)
        margin_factor = 0.05
        bar_x_end = plot_max_x - (plot_max_x - plot_min_x) * margin_factor
        bar_x_start = bar_x_end - scale_bar_length_m
        bar_y = plot_min_y + (plot_max_y - plot_min_y) * margin_factor
        if bar_x_start < plot_min_x + (plot_max_x - plot_min_x) * margin_factor:
            bar_x_start = plot_min_x + (plot_max_x - plot_min_x) * margin_factor
            bar_x_end = bar_x_start + scale_bar_length_m
            if bar_x_end > plot_max_x - (plot_max_x - plot_min_x) * margin_factor:
                scale_bar_length_m = (plot_max_x - (plot_max_x - plot_min_x) * margin_factor) - bar_x_start
                bar_x_end = bar_x_start + scale_bar_length_m
        bar_thickness_data_units = (plot_max_y - plot_min_y) * 0.01
        fig.add_shape(type="line", x0=bar_x_start, y0=bar_y, x1=bar_x_end, y1=bar_y,
                      line=dict(color="black", width=3), xref="x", yref="y")
        cap_height = bar_thickness_data_units * 1.5
        fig.add_shape(type="line", x0=bar_x_start, y0=bar_y - cap_height/2, x1=bar_x_start, y1=bar_y + cap_height/2,
                      line=dict(color="black", width=3), xref="x", yref="y")
        fig.add_shape(type="line", x0=bar_x_end, y0=bar_y - cap_height/2, x1=bar_x_end, y1=bar_y + cap_height/2,
                      line=dict(color="black", width=3), xref="x", yref="y")
        label_text = f"{scale_bar_length_m} m"
        if scale_bar_length_m >= 1000: label_text = f"{scale_bar_length_m/1000:.1f} km".replace(".0","")
        fig.add_annotation(
            x=(bar_x_start + bar_x_end) / 2, 
            y=bar_y + bar_thickness_data_units * 1.5,
            text=label_text, 
            showarrow=False, 
            font=dict(
                family="Arial, sans-serif", # Specify Arial, with sans-serif as fallback
                size=30, 
                color="black"
            ),
            xanchor="center", 
            yanchor="bottom", 
            xref="x", yref="y"
        )


    fig.update_layout(
        title=None,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(scaleanchor="x", scaleratio=1, visible=False, range=[plot_min_y, plot_max_y]),
        xaxis=dict(visible=False, range=[plot_min_x, plot_max_x]),
        showlegend=False,
        legend=dict(
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor="rgba(0,0,0,0.5)",
            orientation="v",
            yanchor="top", y=0.98,
            xanchor="right", x=0.76,
            font=dict(
                family="Arial, sans-serif",
                size=23,  # Increased font size for legend text
                color="black"  # Set legend font color to black
            )
        ),
        margin={"r":5,"t":5,"l":5,"b":5},
    )

    # fig.show()
    try:
        fig.write_image(output_filename, width=1200, height=900, scale=2) # Keep scale=2 for better resolution image
        print(f"Transparent chart saved to {output_filename}")
    except Exception as e:
        print(f"Error saving image: {e}. Ensure 'kaleido' is installed (`pip install kaleido`).")

# The rest of the script (create_dummy_files_if_needed, main execution) remains the same.
# Make sure to include them when you run the full script.
# --- Main Execution ---
if __name__ == '__main__':

    plot_all_trajectories_on_transparent_chart(
        TRAJECTORY_CONFIGS,
        HEX_CENTERS_CSV,
        output_filename="trajectories_final_overlay.png"
    )


