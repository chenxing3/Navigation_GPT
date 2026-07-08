import os
import sys
import math
import ast
import numpy as np
import pandas as pd
import re
from collections import defaultdict
from scipy.stats import norm, gaussian_kde
from scipy.spatial import cKDTree
import matplotlib.colors

import plotly.graph_objects as go
import plotly.io as pio

# --- Configuration ---
WINDOW_SIZE = 50
CONFIDENCE_LEVEL = 0.95
PLOT_SCATTER = True
SCATTER_ALPHA = 0.05
SCATTER_SIZE = 4
GROUP_BY_COLOR = True
KDE_BANDWIDTH_FACTOR = 0.4
TRAJECTORY_STEP_DISTANCE = 5 # The distance for each step in the trajectory

HEX_CENTERS_CSV = '../ops/token_info.csv'
df_hex_centers = pd.read_csv(HEX_CENTERS_CSV)
hex_centers_lookup = {}
if all(col in df_hex_centers.columns for col in ['text', 'x', 'y']):
    hex_centers_lookup = df_hex_centers.set_index('text').to_dict('index')
else:
    print(f"Warning: '{HEX_CENTERS_CSV}' is missing 'text', 'x', or 'y' columns. Middle point analysis will likely fail.")




FILES_TO_PLOT = [

    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251203_205445.feather', 'orange', '-.', 'with_echo (high, 3.8+6.7)'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251203_230917.feather', 'orange', '-.', 'with_echo (high, 3.8+6.7)'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251204_004800.feather', 'orange', '-.', 'with_echo (high, 3.8+6.7)'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251204_030131.feather', 'orange', '-.', 'with_echo (high, 3.8+6.7)'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251217_105031.feather', 'orange', '-.', 'with_echo (high, 3.8+6.7)'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251217_110922.feather', 'orange', '-.', 'with_echo (high, 3.8+6.7)'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251217_141425.feather', 'orange', '-.', 'with_echo (high, 3.8+6.7)'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251217_141527.feather', 'orange', '-.', 'with_echo (high, 3.8+6.7)'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251217_141721.feather', 'orange', '-.', 'with_echo (high, 3.8+6.7)'],


]


# --- Helper Functions ---
def calculate_trajectory_coordinates(x_start, y_start, azimuth_sequence, distance):
    """Reconstructs a trajectory path from a start point and azimuths."""
    try:
        azimuth_sequence = [float(az) for az in azimuth_sequence]
    except (ValueError, TypeError):
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

# (All other helper functions and plotting functions remain the same)
def generate_label_from_filename(filepath):
    basename = os.path.basename(filepath)
    label = re.sub(r'^\d+_', '', basename)
    label = label.replace('.feather', '')
    label = label.replace('_', ' ').title()
    match = re.search(r'V\d+ (\d+)', label, re.IGNORECASE)
    if match:
        return f"Data {match.group(1)}"
    return label

def mcolor_to_plotly_rgba(mcolor, alpha):
    r, g, b, _ = matplotlib.colors.to_rgba(mcolor)
    return f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {alpha})'

def plot_error_vs_distance_plotly(plot_data_list, max_dist_overall, units_label):
    # This function is unchanged from your provided code
    fig = go.Figure()
    linestyle_map = {':': 'dot', '--': 'dash', '-.': 'dashdot', '-': 'solid'}

    for data_item in plot_data_list:
        label = data_item['label']
        distances = data_item['distances']
        error_values = data_item['error_values']
        rolling_mean = data_item['rolling_mean']
        lower_bound = data_item['lower_bound']
        upper_bound = data_item['upper_bound']
        color = data_item['color']
        plotly_linestyle = linestyle_map.get(data_item['raw_linestyle'], 'solid')
        ci_label = f'{data_item["CONFIDENCE_LEVEL"]*100:.0f}% CI' # Simplified CI label for legend
        mean_label = 'Mean' # Simplified Mean label

        valid_indices_mean = ~rolling_mean.isna()
        dist_mean_valid = distances[valid_indices_mean]
        roll_mean_valid = rolling_mean[valid_indices_mean]

        valid_indices_ci = ~lower_bound.isna() & ~upper_bound.isna() & valid_indices_mean
        dist_ci_valid = distances[valid_indices_ci]
        low_b_valid = lower_bound[valid_indices_ci]
        up_b_valid = upper_bound[valid_indices_ci]

        if PLOT_SCATTER and len(error_values.dropna()) > 0:
            valid_scatter_indices = ~error_values.isna()
            fig.add_trace(go.Scatter(
                x=distances[valid_scatter_indices],
                y=error_values[valid_scatter_indices],
                mode='markers',
                marker=dict(color=color, size=SCATTER_SIZE, opacity=SCATTER_ALPHA),
                name=f'{label} (Data)',
                legendgroup=label,
                showlegend=False,
                customdata=np.stack((distances[valid_scatter_indices], error_values[valid_scatter_indices]), axis=-1),
                hovertemplate="<b>Actual Dist:</b> %{customdata[0]:.2f}<br>" + "<b>Error:</b> %{customdata[1]:.2f}<extra></extra>"
            ))
        if len(dist_ci_valid) > 1:
            fig.add_trace(go.Scatter(
                x=np.concatenate((dist_ci_valid, dist_ci_valid[::-1])),
                y=np.concatenate((up_b_valid, low_b_valid[::-1])),
                fill='toself',
                fillcolor=mcolor_to_plotly_rgba(color, 0.2),
                line=dict(width=0),
                hoverinfo='skip',
                name=ci_label,
                legendgroup=label,
                legendgrouptitle_text=label
            ))
        if len(dist_mean_valid) > 0:
            fig.add_trace(go.Scatter(
                x=dist_mean_valid,
                y=roll_mean_valid,
                mode='lines',
                line=dict(color=color, dash=plotly_linestyle, width=2.5),
                name=mean_label,
                legendgroup=label,
                customdata=np.stack((dist_mean_valid, roll_mean_valid), axis=-1),
                hovertemplate="<b>Group:</b> " + label + "<br>" + "<b>Actual Dist:</b> %{customdata[0]:.2f}<br>" + "<b>Mean Error:</b> %{customdata[1]:.2f}<extra></extra>"
            ))

    fig.update_layout(
        title_text='',
        xaxis_title_text=f'Bee line distance from origin locations to destinations {units_label}',
        yaxis_title_text=f'Predicted distance error {units_label}',
        plot_bgcolor='white',
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.8)', bordercolor='rgba(0,0,0,0.5)', borderwidth=1, traceorder="grouped", groupclick="toggleitem", font=dict(size=14), title_font_size=16),
        xaxis=dict(showline=True, linewidth=1.5, linecolor='black', mirror=True, showgrid=True, gridwidth=0.5, gridcolor='lightgrey', zeroline=False, range=[0, min(10000, max_dist_overall * 1.05) if max_dist_overall > 0 else 10000], title_font=dict(size=30), tickfont=dict(size=16)),
        yaxis=dict(showline=True, linewidth=1.5, linecolor='black', mirror=True, showgrid=True, gridwidth=0.5, gridcolor='lightgrey', zeroline=False, range=[0, 5200], title_font=dict(size=30), tickfont=dict(size=16)),
        hovermode='closest',
        width = 900, height= 600,
        font=dict(family="Arial")
    )
    output_filename = 'error_vs_actual_distance_plotly.html'
    fig.write_html(output_filename)
    print(f"\nPlotly Error plot saved to {output_filename}")
    fig.show()


import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import numpy as np

def plot_middle_point_distribution_plotly(norm_positions, unknown_hex_codes, kde_bandwidth_factor=0.5):
    """
    Generates a normalized histogram and KDE plot for the positions.
    The y-axis is scaled so the highest peak in the density is 1.0.
    """
    if not norm_positions:
        print("\n--- Middle Point Distribution (Plotly) ---")
        print("No valid middle points found to plot their distribution.")
        return

    print(f"\n--- Plotting Middle Point Distribution (Plotly Histogram + KDE) ---")
    print(f"Total normalized middle points collected: {len(norm_positions)}")
    
    min_val_norm = min(norm_positions) if norm_positions else 0
    max_val_norm = max(norm_positions) if norm_positions else 1
    print(f"  Min normalized value: {min_val_norm:.2f}, Max normalized value: {max_val_norm:.2f}")

    # --- Data Preparation for Normalization ---

    # 1. Manually calculate histogram densities to enable scaling
    bin_size = (max_val_norm - min_val_norm) / 50 if (max_val_norm - min_val_norm) > 0 else 0.1
    # Ensure at least one bin is created
    num_bins = max(1, int((max_val_norm - min_val_norm) / bin_size)) if bin_size > 0 else 50
    counts, bin_edges = np.histogram(norm_positions, bins=num_bins, range=(min_val_norm, max_val_norm))
    bin_widths = np.diff(bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Calculate true probability density
    hist_density = counts / (np.sum(counts) * bin_widths) if np.sum(counts) > 0 and np.any(bin_widths > 0) else np.zeros_like(counts)

    # 2. Calculate KDE densities
    y_kde_plot = np.array([])
    x_kde_plot = np.linspace(min_val_norm - 0.5, max_val_norm + 0.5, 200)
    if len(set(norm_positions)) > 1:
        kde = gaussian_kde(norm_positions)
        try:
            kde.set_bandwidth(bw_method=kde.factor * kde_bandwidth_factor)
        except Exception as e:
            print(f"Could not set bandwidth with factor {kde_bandwidth_factor}, using default. Error: {e}")
            kde.set_bandwidth(bw_method='scott')
        y_kde_plot = kde(x_kde_plot)

    # 3. Find the overall maximum density value to use for normalization
    max_hist_density = np.max(hist_density) if len(hist_density) > 0 else 0
    max_kde_density = np.max(y_kde_plot) if len(y_kde_plot) > 0 else 0
    overall_max_density = max(max_hist_density, max_kde_density)
    
    # Avoid division by zero if there's no density
    if overall_max_density < 1e-9:
        overall_max_density = 1.0

    # 4. Scale both datasets by the overall maximum
    scaled_hist_density = hist_density / overall_max_density
    scaled_y_kde_plot = y_kde_plot / overall_max_density

    # --- Plotting the Normalized Data ---
    fig_dist = go.Figure()

    # Plot scaled histogram using go.Bar
    fig_dist.add_trace(go.Bar(
        x=bin_centers,
        y=scaled_hist_density,
        name='Histogram',
        marker_color='royalblue',
        opacity=0.7,
        width=bin_widths
    ))

    # Plot scaled KDE
    if len(y_kde_plot) > 0:
        fig_dist.add_trace(go.Scatter(
            x=x_kde_plot,
            y=scaled_y_kde_plot,
            mode='lines',
            name='KDE',
            line=dict(color='darkblue', width=2.5)
        ))

    # Update layout
    fig_dist.update_layout(
        title_text='',
        xaxis_title_text='Normalized position (0=Ori, 1=Dest)',
        yaxis_title_text='Normalized density',
        plot_bgcolor='white',
        showlegend=False,
        bargap=0, # Make bars touch to resemble a histogram
        xaxis=dict(
            showline=True, linewidth=1.5, linecolor='black', mirror=True,
            showgrid=True, gridwidth=0.5, gridcolor='lightgrey', zeroline=True,
            range=[min(-0.75, min_val_norm - 0.2), max(1.75, max_val_norm + 0.2)],
            title_font=dict(size=28), tickfont=dict(size=28)
        ),
        yaxis=dict(
            showline=True, linewidth=1.5, linecolor='black', mirror=True,
            showgrid=True, gridwidth=0.5, gridcolor='lightgrey', zeroline=False,
            range=[0, 1.05], # Set range to 0-1 with a little headroom
            title_font=dict(size=28), tickfont=dict(size=28)
        ),
        # shapes=[
        #     dict(type="line", yref="paper", y0=0, y1=1, xref="x", x0=0, x1=0, line=dict(color="SlateGray", width=1.5, dash="dash")),
        #     dict(type="line", yref="paper", y0=0, y1=1, xref="x", x0=1, x1=1, line=dict(color="SlateGray", width=1.5, dash="dash"))
        # ],
        hovermode='closest',
        width=400, height=500,
        font=dict(family="Arial")
    )
    
    output_filename = 'middle_point_distribution_normalized.html'
    fig_dist.write_html(output_filename)
    print(f"Normalized middle point plot saved to {output_filename}")
    fig_dist.show()

    if unknown_hex_codes:
        print("\n--- Hex Codes from Trajectories Not Found in Lookup ---")
        print(f"The following {len(unknown_hex_codes)} unique hex code(s) were present in trajectory data "
              f"but not found in the lookup and were skipped:")
        MAX_TO_PRINT = 10
        for i, code in enumerate(list(unknown_hex_codes)):
            if i < MAX_TO_PRINT:
                print(f"  - '{code}'")
            elif i == MAX_TO_PRINT:
                print(f"  ... and {len(unknown_hex_codes) - MAX_TO_PRINT} more.")
                break



# --- Main Processing Function ---
def run():
    all_normalized_middle_point_positions = []
    unknown_hex_codes_in_trajectories = set()
    
    # (Grouping and setup logic is unchanged)
    plot_items_to_process = []
    if GROUP_BY_COLOR:
        groups = defaultdict(list)
        group_styles, group_base_labels = {}, {}
        for config in FILES_TO_PLOT:
            file, color, style, label_hint = config[0], config[1], config[2], (config[3] if len(config) > 3 else None)
            groups[color].append(file)
            if color not in group_styles:
                group_styles[color] = style
                group_base_labels[color] = label_hint if label_hint and label_hint != '_nolabel_' else f"{color.capitalize()} Group"
            elif (label_hint and label_hint != '_nolabel_' and (group_base_labels.get(color, "").endswith(" Group") or not group_base_labels.get(color))):
                group_base_labels[color] = label_hint
        for color_key, file_list in groups.items():
            plot_items_to_process.append({'files': file_list, 'color': color_key, 'style': group_styles.get(color_key, '-'), 'label': group_base_labels.get(color_key, f"{color_key.capitalize()} Group")})
    else:
        for config in FILES_TO_PLOT:
            file_path, color, style = config[0], config[1], config[2]
            label = (config[3] if len(config) > 3 and config[3] and config[3] != '_nolabel_' else generate_label_from_filename(file_path))
            plot_items_to_process.append({'files': [file_path], 'color': color, 'style': style, 'label': label})

    error_stats_results = []
    DISTANCE_THRESHOLD_KM = 2
    DISTANCE_THRESHOLD_METERS = DISTANCE_THRESHOLD_KM * 1000
    UNITS_LABEL = "(m)"
    max_actual_distance_overall = 0
    plotly_error_plot_data_list = []

    for item_config in plot_items_to_process:
        files_in_item = item_config['files']
        current_item_label = item_config['label']
        print(f"\nProcessing Group '{current_item_label}'...")

        item_dfs_for_error_plot = []
        for file_path in files_in_item:
            # try:
                df_single = pd.read_feather(file_path)
                # for az in df_single['azimuths_tokens_str']:
                #     print("az: ", az, )

                # # print("df_single: ", df_single)
                # sys.exit(1)

                if "azimuths_tokens_str" in df_single.columns:
                    azimuths = []
                    for az in df_single['azimuths_tokens_str']:
                        try: 
                            tmp = [int(j) for j in az.split(" ")]
                        except:
                            print("What is error: ", az, "!!")#[int(j) for j in az.split(" ")])
                            # sys.exit(1)
                        azimuths.append(tmp)
                    df_single['azimuths'] = azimuths

                    # print("azimuths: ", azimuths)
                    # sys.exit(1)
                    # try:
                    #     df_single['azimuths'] = [int(j.split("_")[1]) for i in df_single['azimuths_tokens_str'] for j in i.split(" ")]
                    # except:
                    #     print([int(j.split("_")[1]) for i in df_single['azimuths_tokens_str'] for j in i.split(" ")])
                    #     sys.exit(1)

                # print("azimuths: ", df_single['azimuths'])
                # sys.exit(1)
                # --- NEW: Trajectory Index Normalization Logic ---
                required_cols = {'loc1_x', 'loc1_y', 'azimuths'}
                if not required_cols.issubset(df_single.columns):
                    print(f"  Warning: File {os.path.basename(file_path)} is missing columns for trajectory analysis: {required_cols - set(df_single.columns)}. Skipping.")
                    continue

                if "middle_locations_texts" in df_single.columns:
                    df_single['middle_location'] = df_single['middle_locations_texts']
                elif "generated_locs_str" in df_single.columns:
                    df_single['middle_location'] = df_single['generated_locs_str']



                # print("df_single: ", df_single, df_single.columns)

                # sys.exit(1)

                for index, row in df_single.iterrows():
                    # 1. Reconstruct the full trajectory for this row
                    # print(df_single.columns)
                    azimuth_sequence = row.get('azimuths')
                    start_x, start_y = row['loc1_x'], row['loc1_y']

                    # if pd.isna(any(azimuths_str)): continue
                    # try:
                    #     azimuth_sequence = ast.literal_eval(azimuths_str)
                    #     if not isinstance(azimuth_sequence, list): continue
                    # except (ValueError, SyntaxError):
                    #     continue

                    x_traj, y_traj = calculate_trajectory_coordinates(start_x, start_y, azimuth_sequence, TRAJECTORY_STEP_DISTANCE)

                    if len(x_traj) < 2: continue # Need at least start and end point

                    trajectory_points = np.c_[x_traj, y_traj]


                    # print("trajectory_points: ", trajectory_points)
                    # sys.exit(1)



                    num_trajectory_points = len(trajectory_points)



                    # 2. Build a cKDTree for fast lookup on this trajectory
                    trajectory_tree = cKDTree(trajectory_points)

                    # 3. Get the "middle" hex codes to query
                    middle_location_str = row.get('middle_location')

                    print("middle_location_str: ", middle_location_str)


                    # sys.exit(1)


                    if pd.isna(middle_location_str): continue
                    # try:
                    #     middle_hex_codes = ast.literal_eval(middle_location_str)
                    #     if not isinstance(middle_hex_codes, list): continue
                    # except (ValueError, SyntaxError):
                    #     continue

                    middle_hex_codes = middle_location_str.split(", ")

                    
                    # unique_middle_hex_codes = list(set(middle_hex_codes))


                    # print("middle_hex_codes: ", middle_hex_codes)

                    # 4. Find the coordinates of the middle points
                    query_points = []
                    for hex_code in middle_hex_codes:
                        if hex_code in hex_centers_lookup:
                            coords = hex_centers_lookup[hex_code]
                            query_points.append([coords['x'], coords['y']])
                        else:
                            unknown_hex_codes_in_trajectories.add(hex_code)
                    
                    if not query_points: continue

                    # 5. Query the tree to find the closest indices on the trajectory
                    _, closest_indices = trajectory_tree.query(query_points)

                    # print("closest_indices: ", closest_indices)
                    # sys.exit(1)

                    # 6. Normalize the indices and store them
                    for idx in closest_indices:
                        normalized_position = idx / (num_trajectory_points - 1)
                        all_normalized_middle_point_positions.append(normalized_position)
                
                # (The error plot logic remains the same, processing error data)
                error_cols = {'actual_distance', 'predict_nearest_distance', 'predict_nearest_distance_to_target'}
                if any(c in df_single.columns for c in error_cols):
                    if "predict_nearest_distance_to_target" in df_single.columns and "predict_nearest_distance" not in df_single.columns:
                        df_single["predict_nearest_distance"] = df_single['predict_nearest_distance_to_target']
                    if all(c in df_single.columns for c in ['actual_distance', 'predict_nearest_distance']):
                        item_dfs_for_error_plot.append(df_single[['actual_distance', 'predict_nearest_distance']].copy())

            # except Exception as e:
            #     print(f"  Error reading or processing file {file_path}: {e}. Skipping.")

        if item_dfs_for_error_plot:
            # (This entire block for calculating and preparing error plot data is unchanged)
            combined_df = pd.concat(item_dfs_for_error_plot, ignore_index=True)
            if not combined_df.empty:
                max_dist = combined_df['actual_distance'].max(skipna=True)
                if pd.notna(max_dist): max_actual_distance_overall = max(max_actual_distance_overall, max_dist)
                
                # Perform stats calculations... (code omitted for brevity, it is unchanged)
                z_score = norm.ppf(1 - (1 - CONFIDENCE_LEVEL) / 2)
                df_sorted = combined_df.sort_values(by='actual_distance').reset_index(drop=True)
                distances = df_sorted['actual_distance']
                error_values = pd.to_numeric(df_sorted['predict_nearest_distance'], errors='coerce')
                effective_window = min(WINDOW_SIZE, len(error_values.dropna()))
                if effective_window >= 2:
                    rolling_mean = error_values.rolling(window=effective_window, center=True, min_periods=max(1, effective_window//2)).mean()
                    rolling_std = error_values.rolling(window=effective_window, center=True, min_periods=max(2, effective_window//2)).std()
                    rolling_count = error_values.rolling(window=effective_window, center=True, min_periods=max(2, effective_window//2)).count()
                    rolling_sem = rolling_std / np.sqrt(rolling_count)
                    confidence_interval = z_score * rolling_sem
                    upper_bound = rolling_mean + confidence_interval
                    lower_bound = (rolling_mean - confidence_interval).clip(lower=0)
                else:
                    rolling_mean, lower_bound, upper_bound = pd.Series(), pd.Series(), pd.Series()

                plotly_error_plot_data_list.append({
                    'label': current_item_label, 'distances': distances, 'error_values': error_values,
                    'rolling_mean': rolling_mean, 'lower_bound': lower_bound, 'upper_bound': upper_bound,
                    'color': item_config['color'], 'raw_linestyle': item_config['style'], 'CONFIDENCE_LEVEL': CONFIDENCE_LEVEL
                })

    # --- Generate Plots ---
    if plotly_error_plot_data_list:
        plot_error_vs_distance_plotly(plotly_error_plot_data_list, max_actual_distance_overall, UNITS_LABEL)
    else:
        print("\nNo data available to generate the Error vs. Actual Distance plot.")


    # print("all_normalized_middle_point_positions: ", all_normalized_middle_point_positions)

    # sys.exit(1)

    plot_middle_point_distribution_plotly(all_normalized_middle_point_positions,
                                          unknown_hex_codes_in_trajectories,
                                          kde_bandwidth_factor=KDE_BANDWIDTH_FACTOR)

if __name__ == "__main__":
    if not os.path.exists(HEX_CENTERS_CSV):
        print(f"FATAL ERROR: Hex centers file not found at '{HEX_CENTERS_CSV}'.")
        sys.exit(1)
    if not hex_centers_lookup:
         print(f"FATAL ERROR: Hex centers lookup could not be created from '{HEX_CENTERS_CSV}'.")
         sys.exit(1)
    run()



