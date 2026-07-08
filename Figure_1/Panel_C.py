import os
import sys
import numpy as np
import pandas as pd
import re
from collections import defaultdict
from scipy.stats import norm, gaussian_kde, mannwhitneyu
import matplotlib.colors
import colorsys

# Plotly imports
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio
from plotly.subplots import make_subplots

# --- Configuration ---
# (Rest of your configurations remain the same)
WINDOW_SIZE = 100
CONFIDENCE_LEVEL = 0.95
PLOT_SCATTER = True
SCATTER_ALPHA = 0.05
SCATTER_SIZE = 4 # Adjusted for Plotly
GROUP_BY_COLOR = True
KDE_BANDWIDTH_FACTOR = 0.4 # Adjust for KDE smoothness (0.1 very jagged, 1.0 very smooth)

HEX_CENTERS_CSV = '../ops/token_info.csv'
df_hex_centers = pd.read_csv(HEX_CENTERS_CSV)
hex_centers_lookup = {}
if all(col in df_hex_centers.columns for col in ['text', 'x', 'y']):
    hex_centers_lookup = df_hex_centers.set_index('text').to_dict('index')
else:
    print(f"Warning: '{HEX_CENTERS_CSV}' is missing 'text', 'x', or 'y' columns. Middle point analysis will likely fail.")

FILES_TO_PLOT = [

    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251203_205445.feather', 'blue', '-', 'with_echo (6.7 Gm)'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251203_230917.feather', 'blue', '-', 'with_echo (6.7 Gm)'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251204_004800.feather', 'blue', '-', 'with_echo (6.7 Gm)'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251204_030131.feather', 'blue', '-', 'with_echo (6.7 Gm)'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251217_105031.feather', 'blue', '-', 'with_echo (6.7 Gm)'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251217_110922.feather', 'blue', '-', 'with_echo (6.7 Gm)'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251217_141425.feather', 'blue', '-', 'with_echo (6.7 Gm)'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251217_141527.feather', 'blue', '-', 'with_echo (6.7 Gm)'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251217_141721.feather', 'blue', '-', 'with_echo (6.7 Gm)'],

    ['../dataset/trajectory/no_echo/trajectories_iterative_with_echo_20251217_114131.feather', 'red', '--', 'no_echo (6.7 Gm)'],
    ['../dataset/trajectory/no_echo/trajectories_iterative_with_echo_20251217_114553.feather', 'red', '--', 'no_echo (6.7 Gm)'],
    ['../dataset/trajectory/no_echo/trajectories_iterative_with_echo_20251217_175530.feather', 'red', '--', 'no_echo (6.7 Gm)'],
    ['../dataset/trajectory/no_echo/trajectories_iterative_with_echo_20251217_184457.feather', 'red', '--', 'no_echo (6.7 Gm)'],

    ['../dataset/trajectory/1mM/trajectories_iterative_with_echo_20260102_171247.feather', 'yellow', '-', 'partial_training (1 Mm)'],
    ['../dataset/trajectory/1mM/trajectories_iterative_with_echo_20260102_172754.feather', 'yellow', '-', 'partial_training (1 Mm)'],
    ['../dataset/trajectory/1mM/trajectories_iterative_with_echo_20260102_173619.feather', 'yellow', '-', 'partial_training (1 Mm)'],
    ['../dataset/trajectory/1mM/trajectories_iterative_with_echo_20260102_185303.feather', 'yellow', '-', 'partial_training (1 Mm)'],
    ['../dataset/trajectory/1mM/trajectories_iterative_with_echo_20260102_191226.feather', 'yellow', '-', 'partial_training (1 Mm)'],
    ['../dataset/trajectory/1mM/trajectories_iterative_with_echo_20260102_191407.feather', 'yellow', '-', 'partial_training (1 Mm)'],
    ['../dataset/trajectory/1mM/trajectories_iterative_with_echo_20260102_192810.feather', 'yellow', '-', 'partial_training (1 Mm)'],
    ['../dataset/trajectory/1mM/trajectories_iterative_with_echo_20260102_205905.feather', 'yellow', '-', 'partial_training (1 Mm)'],
    ['../dataset/trajectory/1mM/trajectories_iterative_with_echo_20260102_211506.feather', 'yellow', '-', 'partial_training (1 Mm)'],
    ['../dataset/trajectory/1mM/trajectories_iterative_with_echo_20260102_215222.feather', 'yellow', '-', 'partial_training (1 Mm)'],
    ['../dataset/trajectory/1mM/trajectories_iterative_with_echo_20260102_222845.feather', 'yellow', '-', 'partial_training (1 Mm)'],
]


# --- Helper Functions (unchanged) ---
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

def process_error_dataframe(df, window_size, z_score, label_for_log=""):
    if df.empty:
        return None
    df_sorted = df.sort_values(by='actual_distance').reset_index(drop=True)
    distances = df_sorted['actual_distance']
    error_values = df_sorted['predict_nearest_distance']
    effective_window = min(window_size, len(error_values.dropna()))
    if effective_window < 2:
        return {
            'distances': distances, 'error_values': error_values,
            'rolling_mean': pd.Series([np.nan] * len(distances)),
            'lower_bound': pd.Series([np.nan] * len(distances)),
            'upper_bound': pd.Series([np.nan] * len(distances))
        }
    min_p_mean = max(1, effective_window // 2)
    min_p_std_sem = max(2, effective_window // 2)
    rolling_mean = error_values.rolling(window=effective_window, center=True, min_periods=min_p_mean).mean()
    rolling_std = error_values.rolling(window=effective_window, center=True, min_periods=min_p_std_sem).std()
    rolling_count = error_values.rolling(window=effective_window, center=True, min_periods=min_p_std_sem).count()
    rolling_sem = rolling_std / np.sqrt(rolling_count)
    rolling_sem[rolling_count < 2] = np.nan
    confidence_interval = z_score * rolling_sem
    upper_bound = rolling_mean + confidence_interval
    lower_bound = (rolling_mean - confidence_interval).clip(lower=0)
    return {
        'distances': distances, 'error_values': error_values,
        'rolling_mean': rolling_mean, 'lower_bound': lower_bound, 'upper_bound': upper_bound
    }

def adjust_color_brightness(color, factor):
    try:
        r, g, b = matplotlib.colors.to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        l = max(0, min(1, l * factor))
        r_new, g_new, b_new = colorsys.hls_to_rgb(h, l, s)
        return matplotlib.colors.to_hex((r_new, g_new, b_new))
    except ValueError:
        return color

# --- Final Combined Panel Plotting Function ---
def plot_combined_panels_plotly(trained_data, untrained_data, units_label):
    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        # --- MODIFICATION: Increased spacing to prevent label overlap ---
        horizontal_spacing=0.05,
        subplot_titles=("Trained", "Untrained"),
        column_widths=[8000, 8000]
    )
    linestyle_map = {':': 'dot', '--': 'dash', '-.': 'dashdot', '-': 'solid'}
    
    added_legend_groups = set()

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(color='lightgray', size=15, symbol='square'),
        name='CI regions'
    ), row=1, col=1)

    def add_traces_to_panel(panel_data, col_index):
        for data_item in panel_data:
            group_label = data_item['group_label']
            distances = data_item['distances']
            error_values = data_item['error_values']
            rolling_mean = data_item['rolling_mean']
            lower_bound = data_item['lower_bound']
            upper_bound = data_item['upper_bound']
            color = data_item['color']
            plotly_linestyle = linestyle_map.get(data_item['raw_linestyle'], 'solid')
            
            show_in_legend = group_label not in added_legend_groups
            if show_in_legend:
                added_legend_groups.add(group_label)
            
            if PLOT_SCATTER and not error_values.dropna().empty:
                fig.add_trace(go.Scatter(
                    x=distances[~error_values.isna()], y=error_values.dropna(),
                    mode='markers', legendgroup=group_label, showlegend=False,
                    marker=dict(color=color, size=SCATTER_SIZE, opacity=SCATTER_ALPHA),
                    hovertemplate="Dist: %{x:.0f} m<br>Error: %{y:.0f} m<extra></extra>"
                ), row=1, col=col_index)
            
            valid_ci = ~lower_bound.isna() & ~upper_bound.isna()
            if valid_ci.any():
                fig.add_trace(go.Scatter(
                    x=np.concatenate((distances[valid_ci], distances[valid_ci][::-1])),
                    y=np.concatenate((upper_bound[valid_ci], lower_bound[valid_ci][::-1])),
                    fill='toself', fillcolor='rgba(211, 211, 211, 0.5)',
                    line=dict(width=0), hoverinfo='skip', showlegend=False,
                    legendgroup=group_label
                ), row=1, col=col_index)
                
            valid_mean = ~rolling_mean.isna()
            if valid_mean.any():
                fig.add_trace(go.Scatter(
                    x=distances[valid_mean], y=rolling_mean[valid_mean],
                    mode='lines', name=group_label, legendgroup=group_label,
                    showlegend=show_in_legend,
                    line=dict(color=color, dash=plotly_linestyle, width=3),
                    hovertemplate="<b>%{text}</b><br>Dist: %{x:.0f} m<br>Mean Error: %{y:.0f} m<extra></extra>",
                    text=[group_label] * len(distances[valid_mean])
                ), row=1, col=col_index)

    add_traces_to_panel(trained_data, 1)
    add_traces_to_panel(untrained_data, 2)
    
    fig.update_layout(
        height=500, width=800,
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family="Arial"),
        legend=dict(
            traceorder="normal", x=0.01, y=0.99, xanchor='left', yanchor='top',
            bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1.5,
            font=dict(size=20)
        ),
        margin=dict(t=50, b=100)
    )

    # --- MODIFICATION: Y-axis title is automatically handled by shared_yaxes, this styles it. ---
    # First, apply styling that is common to both y-axes (range, lines, grids)
    fig.update_yaxes(
        range=[0, 2100],
        showline=True, linewidth=2, linecolor='black', mirror=True,
        showgrid=True, gridwidth=1, gridcolor='lightgrey',
        tickfont=dict(size=24)
    )

    # Second, apply the title ONLY to the y-axis of the first panel (row=1, col=1)
    fig.update_yaxes(
        title_text=f'Distance error {units_label}',
        title_font=dict(size=28),  # not work here!!
        row=1, col=1
    )
    # --- MODI

    fig.update_yaxes(
        range=[0, 2000],
        showline=True, linewidth=2, linecolor='black', mirror=True,
        showgrid=True, gridwidth=1, gridcolor='lightgrey',
        tickfont=dict(size=24)
    )

    fig.update_xaxes(
        showline=True, linewidth=2, linecolor='black', mirror=True,
        showgrid=True, gridwidth=1, gridcolor='lightgrey',
        tickfont=dict(size=24), 
        zeroline=False
    )
    
    fig.update_xaxes(range=[0, 7500], 
        tickvals=[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
        ticktext=['0', '1k', '2k', '3k', '4k', '5k', '6k', '7k', '8k'], 
        
        row=1, col=1)
    fig.update_xaxes(
        range=[2000, 10000],
        tickvals=[2000, 4000, 6000, 8000, 10000],
        ticktext=['2k', '4k', '6k', '8k', '10k'],
        row=1, col=2
    )


    fig.update_annotations(font=dict(size=26))
    
    fig.add_annotation(
        text=f'Bee line distance from origin locations to destinations {units_label}',
        align='center', showarrow=False,
        xref='paper', yref='paper',
        x=0.5, y=-0.2,
        font=dict(size=28)
    )
    
    output_filename = 'error_vs_distance_combined_panels.html'
    fig.write_html(output_filename)
    print(f"\nCombined panel plot saved to {output_filename}")
    fig.show()


# --- Main Processing Function ---
def run():
    z_score = norm.ppf(1 - (1 - CONFIDENCE_LEVEL) / 2)
    plot_items_to_process = []
    
    # Initialize the global list to capture data for the final aggregate statistics
    global_df_list = []


# --- NEW: Dictionary to store statistical groups for Mann-Whitney Test ---
    stat_comparison_data = {
        'with_echo': {},
        'no_echo': {}
    }
    if GROUP_BY_COLOR:
        groups = defaultdict(list)
        group_styles, group_base_labels = {}, {}
        for config in FILES_TO_PLOT:
            file, color, style, label_hint = config[0], config[1], config[2], (config[3] if len(config) > 3 else 'Group')
            groups[color].append(file)
            if color not in group_styles:
                group_styles[color], group_base_labels[color] = style, label_hint
        for color_key, file_list in groups.items():
            plot_items_to_process.append({
                'files': file_list, 'color': color_key, 'style': group_styles[color_key], 'label': group_base_labels[color_key]
            })

    UNITS_LABEL = "(m)"
    plotly_error_plot_data_list = []
    
    # Define the threshold for the statistical split
    THRESHOLD_M = 7500

    # Helper function to print stats cleanly
    def print_stats(df_subset, label):
        valid_errors = df_subset['predict_nearest_distance'].dropna()
        if not valid_errors.empty:
            mean_val = valid_errors.mean()
            std_val = valid_errors.std()
            median_val = valid_errors.median()
            n_val = len(valid_errors)
            print(f"    {label:<20} -> Mean: {mean_val:>7.2f} m | SD: {std_val:>7.2f} m | Median: {median_val:>7.2f} m | (n={n_val})")
        else:
            print(f"    {label:<20} -> No data available")

    print("\n" + "="*85)
    print("INDIVIDUAL CATEGORY STATISTICAL SUMMARY")
    print("="*85)

    for item_config in plot_items_to_process:
        files_in_item = item_config['files']
        color = item_config['color']
        raw_line_style = item_config['style']
        current_item_label = item_config['label']
        item_dfs_for_error_plot = []
        
        for file_path in files_in_item:
            try:
                df_single = pd.read_feather(file_path)
                if "predict_nearest_distance_to_target" in df_single.columns and "predict_nearest_distance" not in df_single.columns:
                    df_single["predict_nearest_distance"] = df_single['predict_nearest_distance_to_target']
                required_cols = {'actual_distance', 'predict_nearest_distance'}
                if not required_cols.issubset(df_single.columns): continue
                cols_to_keep = ['actual_distance', 'predict_nearest_distance']
                if 'trained_binary' in df_single.columns:
                    cols_to_keep.append('trained_binary')
                item_dfs_for_error_plot.append(df_single[cols_to_keep].copy())
            except Exception as e:
                print(f"  Warning: Could not process {file_path}. Error: {e}")

        if item_dfs_for_error_plot:
            combined_df = pd.concat(item_dfs_for_error_plot, ignore_index=True)
            if combined_df.empty: continue
            combined_df['predict_nearest_distance'] = pd.to_numeric(combined_df['predict_nearest_distance'], errors='coerce')

            # Append to global list for final aggregate stats
            global_df_list.append(combined_df)

            # --- INDIVIDUAL CATEGORY STATS PRINTOUT ---
            print(f"\n[{current_item_label.upper()}]")
            data_to_plot = []


            # --- NEW: Identify if the current group is 'with_echo' or 'no_echo' ---
            group_key = None
            if 'with_echo' in current_item_label.lower():
                group_key = 'with_echo'
            elif 'no_echo' in current_item_label.lower():
                group_key = 'no_echo'



            if 'trained_binary' in combined_df.columns:
                df_trained = combined_df[combined_df['trained_binary'] == 1]
                df_untrained = combined_df[combined_df['trained_binary'] == 0]
                
                print("  -- Trained Data --")
                print_stats(df_trained[df_trained['actual_distance'] < THRESHOLD_M], f"< {THRESHOLD_M}m")
                print_stats(df_trained[df_trained['actual_distance'] >= THRESHOLD_M], f">= {THRESHOLD_M}m")
                
                print("  -- Untrained Data --")
                print_stats(df_untrained[df_untrained['actual_distance'] < THRESHOLD_M], f"< {THRESHOLD_M}m")
                print_stats(df_untrained[df_untrained['actual_distance'] >= THRESHOLD_M], f">= {THRESHOLD_M}m")



# --- NEW: Save data to dictionary for statistical testing later ---
                if group_key:
                    stat_comparison_data[group_key]['trained_lt'] = df_trained[df_trained['actual_distance'] < THRESHOLD_M]['predict_nearest_distance'].dropna()
                    stat_comparison_data[group_key]['trained_ge'] = df_trained[df_trained['actual_distance'] >= THRESHOLD_M]['predict_nearest_distance'].dropna()
                    stat_comparison_data[group_key]['untrained_lt'] = df_untrained[df_untrained['actual_distance'] < THRESHOLD_M]['predict_nearest_distance'].dropna()
                    stat_comparison_data[group_key]['untrained_ge'] = df_untrained[df_untrained['actual_distance'] >= THRESHOLD_M]['predict_nearest_distance'].dropna()


                if not df_trained.empty:
                    data_to_plot.append({'df': df_trained, 'label_suffix': ' (Trained)', 'linestyle': raw_line_style})
                if not df_untrained.empty:
                    data_to_plot.append({'df': df_untrained, 'label_suffix': ' (Untrained)', 'linestyle': raw_line_style})
            else:
                print("  -- Overall Data (No 'trained_binary' column found) --")
                print_stats(combined_df[combined_df['actual_distance'] < THRESHOLD_M], f"< {THRESHOLD_M}m")
                print_stats(combined_df[combined_df['actual_distance'] >= THRESHOLD_M], f">= {THRESHOLD_M}m")
                data_to_plot.append({'df': combined_df, 'label_suffix': ' (Untrained)', 'linestyle': raw_line_style})

            # Prepare data for plotting
            for subset in data_to_plot:
                processed_data = process_error_dataframe(subset['df'], WINDOW_SIZE, z_score, f"{current_item_label}{subset['label_suffix']}")
                if processed_data:
                    plot_color = adjust_color_brightness(color, 0.6) if 'Trained' in subset['label_suffix'] else color
                    plotly_error_plot_data_list.append({
                        'label': f"{current_item_label}{subset['label_suffix']}",
                        'group_label': current_item_label,
                        'distances': processed_data['distances'],
                        'error_values': processed_data['error_values'],
                        'rolling_mean': processed_data['rolling_mean'],
                        'lower_bound': processed_data['lower_bound'],
                        'upper_bound': processed_data['upper_bound'],
                        'color': color, 
                        'raw_linestyle': subset['linestyle']
                    })


# --- NEW: Calculate and print Mann-Whitney U test p-values ---
    print("\n" + "="*85)
    print("MANN-WHITNEY U TEST: WITH_ECHO vs NO_ECHO")
    print("="*85)
    
    if stat_comparison_data['with_echo'] and stat_comparison_data['no_echo']:
        # Tuples mapping label to the dictionary key
        subsets = [
            (f'Trained < {THRESHOLD_M}m', 'trained_lt'),
            (f'Trained >= {THRESHOLD_M}m', 'trained_ge'),
            (f'Untrained < {THRESHOLD_M}m', 'untrained_lt'),
            (f'Untrained >= {THRESHOLD_M}m', 'untrained_ge')
        ]
        
        for label, key in subsets:
            data_with = stat_comparison_data['with_echo'].get(key, pd.Series())
            data_no = stat_comparison_data['no_echo'].get(key, pd.Series())
            
            if len(data_with) > 0 and len(data_no) > 0:
                stat, p_val = mannwhitneyu(data_with, data_no, alternative='two-sided')
                print(f"  {label:<25} -> p-value = {p_val:.4e} (n_with_echo={len(data_with)}, n_no_echo={len(data_no)})")
            else:
                print(f"  {label:<25} -> Not enough data for comparison")
    else:
        print("  Missing 'with_echo' or 'no_echo' data required for comparison.")
    print("="*85 + "\n")
    # -------------------------------------------------------------

    if global_df_list:
        global_df = pd.concat(global_df_list, ignore_index=True)
        print("\n" + "="*85)
        print("GLOBAL AGGREGATE STATISTICAL SUMMARY (All Categories Combined)")
        print("="*85)
        print("  -- All Trained Data --")
        if 'trained_binary' in global_df.columns:
             df_global_trained = global_df[global_df['trained_binary'] == 1]
             print_stats(df_global_trained[df_global_trained['actual_distance'] < THRESHOLD_M], f"< {THRESHOLD_M}m")
             print_stats(df_global_trained[df_global_trained['actual_distance'] >= THRESHOLD_M], f">= {THRESHOLD_M}m")
             
             print("  -- All Untrained Data --")
             df_global_untrained = global_df[global_df['trained_binary'] == 0]
             print_stats(df_global_untrained[df_global_untrained['actual_distance'] < THRESHOLD_M], f"< {THRESHOLD_M}m")
             print_stats(df_global_untrained[df_global_untrained['actual_distance'] >= THRESHOLD_M], f">= {THRESHOLD_M}m")
        print("="*85 + "\n")

    if plotly_error_plot_data_list:
        trained_data = [d for d in plotly_error_plot_data_list if '(Trained)' in d['label']]
        untrained_data = [d for d in plotly_error_plot_data_list if '(Untrained)' in d['label']]

        if trained_data or untrained_data:
            print("\n--- Generating Combined Panel Plot for Trajectories ---")
            plot_combined_panels_plotly(trained_data, untrained_data, UNITS_LABEL)
        else:
            print("\nNo data was categorized as 'Trained' or 'Untrained'.")
    else:
        print("\nNo data available to generate the Error vs. Actual Distance plot.")

if __name__ == "__main__":
    if not os.path.exists(HEX_CENTERS_CSV):
        print(f"FATAL ERROR: Hex centers file not found at '{HEX_CENTERS_CSV}'.")
        sys.exit(1)
    if not hex_centers_lookup:
        print(f"FATAL ERROR: Hex centers lookup table could not be created from '{HEX_CENTERS_CSV}'.")
        sys.exit(1)
    run()

