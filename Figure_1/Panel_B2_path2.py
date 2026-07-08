

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import ast
import os, re
import sys
from matplotlib.lines import Line2D 
from scipy.spatial import cKDTree  # <--- Added for distance checking
from matplotlib.legend_handler import HandlerTuple


# add font as arial and size of 14
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

# --- Configuration ---
RESULTS_FEATHER_FILE = '../dataset/trajectory/full_model/homing_trajectory/trajectories_iterative_with_echo_20260416_153756.feather'
HEX_CENTERS_CSV = '../ops/token_info.csv'
ADDITIONAL_TRAJECTORIES_FILE = '../dataset/trained_data/example5.txt'
ADDITIONAL_TRAJECTORIES_FILE2 ='../dataset/trained_data/example5_reverse.txt'

ROW_INDEX_TO_PLOT = 9
AZIMUTH_STEP_DISTANCE = 5.0
SOURCE_CRS_EPSG = "EPSG:6991"

# --- Helper functions ---

def calculate_trajectory_coordinates(x_start, y_start, azimuth_sequence, distance):
    try:
        azimuth_sequence = [float(az) for az in azimuth_sequence if str(az).strip()]
    except (ValueError, TypeError) as e:
        print(f"Error converting azimuths to float: {e}")
        return [], []
    
    azimuth_sequence_rad = [math.radians(azimuth) for azimuth in azimuth_sequence]
    x_current, y_current = x_start, y_start
    x_trajectory = [x_start]
    y_trajectory = [y_start]
    
    for azimuth_rad in azimuth_sequence_rad:
        delta_x = distance * math.sin(azimuth_rad)
        delta_y = distance * math.cos(azimuth_rad)
        x_current += delta_x
        y_current += delta_y
        x_trajectory.append(x_current)
        y_trajectory.append(y_current)
        
    return x_trajectory, y_trajectory

def filter_and_cut_trajectory(path_x, path_y, target_x, target_y, threshold_meters=100.0):
    """
    Checks if the trajectory hits the target (< threshold).
    Returns (cut_path_x, cut_path_y) if it hits, otherwise (None, None).
    """
    if not path_x or not path_y:
        return None, None

    # Combine x and y into points
    path_points = np.column_stack((path_x, path_y))
    
    # Build tree and query nearest point to target
    tree = cKDTree(path_points)
    dist, idx = tree.query([target_x, target_y], k=1)
    
    if dist < threshold_meters:
        # It's a hit! Cut the path at this point to make it look clean
        return path_x[:idx+1], path_y[:idx+1]
    else:
        # It's a miss. Return None to filter it out.
        return None, None

def load_additional_trajectories(file_path, start_x, start_y, target_x, target_y, step_distance):
    """
    Loads trajectories and filters them based on whether they reach the target.
    """
    additional_paths_data = []
    if not os.path.exists(file_path):
        print(f"Warning: Additional trajectories file not found: {file_path}")
        return additional_paths_data
        
    total_count = 0
    kept_count = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue

            total_count += 1
            pattern = re.compile(r'(\d{3})_')
            azimuth_strs = pattern.findall(line)
            
            # 1. Calculate Full Path
            path_x, path_y = calculate_trajectory_coordinates(start_x, start_y, azimuth_strs, step_distance)
            
            # 2. Filter (Check if it hits target)
            if path_x and path_y:
                valid_x, valid_y = filter_and_cut_trajectory(
                    path_x, path_y, 
                    target_x, target_y, 
                    threshold_meters=100.0
                )
                
                if valid_x is not None:
                    additional_paths_data.append((valid_x, valid_y))
                    kept_count += 1
                    
    print(f"File {os.path.basename(file_path)}: Kept {kept_count} trajectories out of {total_count} (filtered by target proximity).")
    return additional_paths_data


# --- Matplotlib Plotting Function ---
def draw_transparent_plot(
    loc1_x, loc1_y, loc2_x, loc2_y,
    azimuth_path_x_proj,
    azimuth_path_y_proj,
    grid_path_x_proj,
    grid_path_y_proj,
    middle_loc_plot_x,
    middle_loc_plot_y,
    additional_traj_paths_data1,
    additional_traj_paths_data2,
    output_filename="trajectory.png"
):
    print("\n--- Starting Matplotlib Plot Generation ---")
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    # 1. Plot Forward Trajectories (loc1 -> loc2) - darkgrey
    if additional_traj_paths_data1:
        for path_x, path_y in additional_traj_paths_data1:
            ax.plot(path_x, path_y, color='lightgray', linewidth=5, alpha=0.9)

    # 2. Plot Reverse Trajectories (loc2 -> loc1) - dimgray dashed
    if additional_traj_paths_data2:
        for path_x, path_y in additional_traj_paths_data2:
            ax.plot(path_x, path_y, color='lightgray', linewidth=5, linestyle='--', alpha=0.9)

    # 3. Plot Generated Trajectory Path
    if azimuth_path_x_proj:
        ax.plot(azimuth_path_x_proj, azimuth_path_y_proj, color='#18d3f3', linewidth=6, solid_capstyle='round')

    # 4. Plot Selected Grid Points
    if grid_path_x_proj:
        colors = np.arange(len(grid_path_x_proj))
        ax.scatter(grid_path_x_proj, grid_path_y_proj, c=colors, cmap='rainbow', s=30, zorder=3)

    # # 5. Plot Sensory Locations
    # if middle_loc_plot_x:
    #     ax.scatter(middle_loc_plot_x, middle_loc_plot_y, marker='o', s=100, color='black',
    #                edgecolor='black', linewidth=1, zorder=4)
# 5. Plot Sensory Locations - Enlarged, black fill, white border
    if middle_loc_plot_x:
        # Tripled 's' (size), added white edgecolors with a thicker linewidths
        ax.scatter(middle_loc_plot_x, middle_loc_plot_y, marker='o', s=300, c='black',
                   edgecolors='white', linewidths=3, zorder=5)
        


    # 6. Plot Start (LOC1) and Target (LOC2)
    ax.plot(loc1_x, loc1_y, marker='*', markersize=36, color='red', linestyle='None')
    ax.plot(loc2_x, loc2_y, marker='X', markersize=36, color='black', linestyle='None')

    # 7. Final plot adjustments
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')






    # # 8. *** Custom Legend Definition and Styling ***
    # legend_elements = [
    #     Line2D([0], [0], color='darkgrey', lw=2.5, label='Trained trajectory (go home)'),
    #     Line2D([0], [0], color='darkgrey', lw=2.5, linestyle='--', label='Trained trajectory (forward)'),
    #     Line2D([0], [0], color='#ffa15a', lw=3, label='Generated trajectory (Path 1)'),
    #     Line2D([0], [0], color='#18d3f3', lw=3, label='Generated trajectory (Path 2)'),
    #     Line2D([0], [0], marker='o', color='w', label='Sensory location',
    #            markerfacecolor='black', markeredgecolor='black', markersize=10),
    #     Line2D([0], [0], marker='*', color='w', label='Origin',
    #            markerfacecolor='red', markeredgecolor='none', markersize=15),
    #     Line2D([0], [0], marker='X', color='w', label='Destination',
    #            markerfacecolor='black', markeredgecolor='none', markersize=12),
    #     Line2D([0], [0], marker='o', color='w', label='Path plan (colorful dots)',
    #            markerfacecolor='red', markeredgecolor='none', markersize=6)
    # ]

    legend_elements = [
        # Updated line colors to match the plot changes
        Line2D([0], [0], color='white', lw=3, label='Trained trajectory (go home)'),
        Line2D([0], [0], color='white', lw=3, linestyle='--', label='Trained trajectory (forward)'),
        Line2D([0], [0], color='#ffa15a', lw=3, label='Generated trajectory (Path 1)'),
        Line2D([0], [0], color='#18d3f3', lw=3, label='Generated trajectory (Path 2)'),
        
        # Updated sensory location: black fill, white edge, increased size
        Line2D([0], [0], marker='o', color='w', label='Corrected sensory location',
            markerfacecolor='black', markeredgecolor='white', markeredgewidth=1.5, markersize=12),
            
        Line2D([0], [0], marker='*', color='w', label='Origin',
            markerfacecolor='red', markeredgecolor='none', markersize=15),
        Line2D([0], [0], marker='X', color='w', label='Destination',
            markerfacecolor='black', markeredgecolor='none', markersize=12)
    ]

    # Define two distinct markers for the path plan to show color variation

    path_line_1 = Line2D([0], [0], color='white', lw=5, marker='o', 
                        markerfacecolor='red', markeredgecolor='none', markersize=7)
    path_line_2 = Line2D([0], [0], color='white', lw=5, marker='o', 
                        markerfacecolor='blue', markeredgecolor='none', markersize=7)



    handles = legend_elements + [(path_line_1, path_line_2)]
    labels = [h.get_label() for h in legend_elements] + ['Path plan (colorful dots)']

    # ax.legend(handles=legend_elements,
    #           loc='upper left',
    #           bbox_to_anchor=(0.1, 0.3), # Position: 2% from left, 98% from bottom
    #           frameon=True,
    #           framealpha=0.75, # Semi-transparent background
    #           edgecolor='gray',
    #           fancybox=True, # Rounded corners
    #           prop={'family': 'sans-serif', 'size': 12} # Font properties
    #          )

    ax.legend(handles=handles,
            labels=labels,
            loc='upper left',
            bbox_to_anchor=(0.1, 0.3), 
            frameon=True,
            framealpha=0.75, 
            edgecolor='gray',
            fancybox=True, 
            prop={'family': 'sans-serif', 'size': 12}, 
            # The handler_map is required to draw the tuple markers side-by-side
            handler_map={tuple: HandlerTuple(ndivide=None)} 
            )

    # 9. Save the figure
    plt.savefig(output_filename, transparent=True, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"--- Matplotlib plot saved to {output_filename} ---")


# --- Main Logic ---
def generate_plot_from_data(results_file, hex_centers_file, additional_traj_src_file, additional_traj_src_file2, row_index):
    print(f"Loading results from: {results_file}")
    if not os.path.exists(results_file):
        print(f"Error: Results file not found"); return
    df_results = pd.read_feather(results_file)
    df_results['grid_ids'] = df_results['generated_grids_str']

    hex_centers_lookup = {}
    if os.path.exists(hex_centers_file):
        try:
            df_hex_centers = pd.read_csv(hex_centers_file)
            if all(col in df_hex_centers.columns for col in ['text', 'x', 'y']):
                hex_centers_lookup = df_hex_centers.set_index('text').to_dict('index')
        except Exception: pass

    if row_index < 0 or row_index >= len(df_results):
        print(f"Error: Row index {row_index} out of bounds."); return

    print(f"\n--- Processing data from row index: {row_index} ---")
    trajectory_data = df_results.iloc[row_index]
    print("trajectory_data: ", trajectory_data)


    try:
        loc1_x, loc1_y = trajectory_data['loc1_x'], trajectory_data['loc1_y']
        loc2_x, loc2_y = trajectory_data['loc2_x'], trajectory_data['loc2_y']
        azimuths_data_str = str(trajectory_data['azimuths_tokens_str'])
        middle_loc_texts_str = str(trajectory_data.get('middle_locations_texts', ""))

        full_seq = trajectory_data['full_generated_sequence']
        print("full_seq: ", len(full_seq.split(" ")))
        # sys.exit(1)
        if pd.isna(middle_loc_texts_str): middle_loc_texts_str = ""
    except KeyError: return
    
    if any(pd.isna(c) for c in [loc1_x, loc1_y, loc2_x, loc2_y]):
        print(f"Error: Coordinates missing."); return
    
    # 1. Main Azimuth Path
    azimuths = [i for i in azimuths_data_str.split(" ") if i.strip()]
    azimuth_path_x_proj, azimuth_path_y_proj = calculate_trajectory_coordinates(
        loc1_x, loc1_y, azimuths, AZIMUTH_STEP_DISTANCE
    )

    # 2. Grid Path
    grid_ids = [str(g) for g in str(trajectory_data['grid_ids']).split(" ")]
    original_grid_path_x_temp, original_grid_path_y_temp = [], []
    for grid_id_val in grid_ids:
        coords = hex_centers_lookup.get(str(grid_id_val).strip())
        if coords:
            original_grid_path_x_temp.append(coords['x'])
            original_grid_path_y_temp.append(coords['y'])
        else:
            original_grid_path_x_temp.append(-1)
            original_grid_path_y_temp.append(-1)
    
    grid_path_x_proj = [x for x, y in zip(original_grid_path_x_temp, original_grid_path_y_temp) if x != -1]
    grid_path_y_proj = [y for x, y in zip(original_grid_path_x_temp, original_grid_path_y_temp) if x != -1]
    
    # 3. Middle Locations
    middle_loc_ids = [loc_id.strip() for loc_id in middle_loc_texts_str.split(',') if loc_id.strip()]
    middle_loc_plot_x, middle_loc_plot_y = [], []
    if middle_loc_ids and hex_centers_lookup:
        for loc_id_token in middle_loc_ids:
            coords = hex_centers_lookup.get(loc_id_token, hex_centers_lookup.get(loc_id_token.rstrip('_')))
            if coords:
                middle_loc_plot_x.append(coords['x'])
                middle_loc_plot_y.append(coords['y'])

    # 4. Load Additional Trajectories (WITH FILTERING)
    
    # Forward File: Start=LOC1, Target=LOC2
    print("Processing Forward Trajectories...")
    additional_traj_paths_data1 = load_additional_trajectories(
        additional_traj_src_file, 
        loc1_x, loc1_y,          # Start
        loc2_x, loc2_y,          # Target
        AZIMUTH_STEP_DISTANCE
    )
    
    # Reverse File: Start=LOC2, Target=LOC1
    print("Processing Reverse Trajectories...")
    additional_traj_paths_data2 = load_additional_trajectories(
        additional_traj_src_file2, 
        loc2_x, loc2_y,          # Start
        loc1_x, loc1_y,          # Target
        AZIMUTH_STEP_DISTANCE
    )

    draw_transparent_plot(
        loc1_x, loc1_y, loc2_x, loc2_y,
        azimuth_path_x_proj, azimuth_path_y_proj,
        grid_path_x_proj, grid_path_y_proj,
        middle_loc_plot_x, middle_loc_plot_y,
        additional_traj_paths_data1, additional_traj_paths_data2,
        output_filename=f"trajectory_row_{row_index}.png"
    )

if __name__ == '__main__':
    plot_row = ROW_INDEX_TO_PLOT
    if len(sys.argv) > 1:
        try:
            plot_row = int(sys.argv[1])
        except ValueError: pass

    generate_plot_from_data(
        RESULTS_FEATHER_FILE, HEX_CENTERS_CSV,
        ADDITIONAL_TRAJECTORIES_FILE, ADDITIONAL_TRAJECTORIES_FILE2,
        plot_row
    )