import pandas as pd
import numpy as np
import math
import os
import sys

import pyproj
from shapely.geometry import Point
from shapely.ops import transform
from pyproj import Transformer

import plotly.graph_objects as go
from scipy.stats import mannwhitneyu, f_oneway # Added f_oneway for ANOVA

import matplotlib.pyplot as plt
# Using a font that more reliably supports Unicode characters like '→'
plt.rcParams['font.family'] = ['Arial']


# --- User-Provided Setup Code ---
geodesic = pyproj.Geod(ellps='WGS84')
proj_meters = pyproj.CRS('EPSG:6991')
proj_latlng = pyproj.CRS('EPSG:4326')
project_to_coords = Transformer.from_crs(proj_meters, proj_latlng, always_xy=True).transform
project_to_meters = Transformer.from_crs(proj_latlng, proj_meters, always_xy=True).transform

def coordinate_to_xy(lon, lat):
    pt_meters = transform(project_to_meters, Point(lon, lat))
    return pt_meters.x, pt_meters.y

def xy_to_coordinate(x, y):
    if x is None or y is None or math.isnan(x) or math.isnan(y):
        return None, None
    try:
        pt_latlng = transform(project_to_coords, Point(x, y))
        return pt_latlng.x, pt_latlng.y
    except Exception as e:
        print(f"Error in xy_to_coordinate for x={x}, y={y}: {e}")
        return None, None

FILES_TO_PLOT = [


    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251203_205445.feather'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251203_230917.feather'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251204_004800.feather'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251204_030131.feather'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251217_105031.feather'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251217_110922.feather'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251217_141425.feather'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251217_141527.feather'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251217_141721.feather'],

    ['../dataset/trajectory/no_echo/trajectories_iterative_with_echo_20251217_114131.feather'],
    ['../dataset/trajectory/no_echo/trajectories_iterative_with_echo_20251217_114553.feather'],
    ['../dataset/trajectory/no_echo/trajectories_iterative_with_echo_20251217_175530.feather'],
    ['../dataset/trajectory/no_echo/trajectories_iterative_with_echo_20251217_184457.feather'],


]

# --- Solution Code ---

# 1. Define map boundaries and quadrants
lon_min = 35.5635052799; lon_max = 35.6561130473
lat_min = 33.0655435585; lat_max = 33.1683021567
bl_x, bl_y = coordinate_to_xy(lon_min, lat_min)
tr_x, tr_y = coordinate_to_xy(lon_max, lat_max)
mid_x = (bl_x + tr_x) / 2
mid_y = (bl_y + tr_y) / 2

def get_phase(x, y):
    if not (bl_x <= x <= tr_x and bl_y <= y <= tr_y): return None
    is_top = y > mid_y; is_left = x < mid_x
    if is_top and is_left: return 1
    elif is_top and not is_left: return 2
    elif not is_top and is_left: return 3
    elif not is_top and not is_left: return 4
    return None

# 2. Process ALL files and aggregate data
all_dfs = [pd.read_feather(fp[0]) for fp in FILES_TO_PLOT if os.path.exists(fp[0])]
if not all_dfs:
    print("Error: No data files found at the specified paths. Exiting."); sys.exit()
df_combined = pd.concat(all_dfs, ignore_index=True)
df_combined['phase1'] = df_combined.apply(lambda r: get_phase(r['loc1_x'], r['loc1_y']), axis=1)
df_combined['phase2'] = df_combined.apply(lambda r: get_phase(r['loc2_x'], r['loc2_y']), axis=1)
df_combined.dropna(subset=['phase1', 'phase2', 'predict_nearest_distance_to_target'], inplace=True)
df_combined = df_combined[df_combined['predict_nearest_distance_to_target'] > 0]

distance_data_by_category = {}
for (p1, p2), group_df in df_combined.groupby(['phase1', 'phase2']):
    # MODIFIED: Use a clearer Unicode arrow
    category_name = f"{int(p1)} → {int(p2)}"
    distance_data_by_category[category_name] = group_df['predict_nearest_distance_to_target'].tolist()

# 3. Define plotting order and pairs
# MODIFIED: Use the Unicode arrow for consistency
plot_order = [
    '1 → 1', '2 → 2', '3 → 3', '4 → 4',
    ('1 → 2', '2 → 1'), ('1 → 3', '3 → 1'), ('1 → 4', '4 → 1'),
    ('2 → 3', '3 → 2'), ('2 → 4', '4 → 2'), ('3 → 4', '4 → 3'),
]
colors = {'base': 'black', 'reciprocal': 'gray'}

# MODIFIED: Updated function to show "ns" for non-significant p-values
def format_p_value(p):
    if p >= 0.05: return "ns"
    if p < 0.001: return "p < 0.001"
    if p < 0.01: return "p < 0.01"
    if p < 0.05: return "p < 0.05"
    return f"p = {p:.3f}" # Fallback, should not be hit with the logic above

# 4. Generate the plot
fig = go.Figure()
x_axis_labels = []
x_pos_counter = 0

# --- PRE-CALCULATE Y-AXIS RANGE TO INCLUDE ANNOTATIONS ---
y_max_for_annotations = 0
y_min_for_annotations = float('inf')

for cat in distance_data_by_category:
    if distance_data_by_category[cat]:
        y_max_for_annotations = max(y_max_for_annotations, max(distance_data_by_category[cat]))
        y_min_for_annotations = min(y_min_for_annotations, min(distance_data_by_category[cat]))

y_axis_top = y_max_for_annotations * 20

# 5. Build the plot with annotations
for item in plot_order:
    if isinstance(item, str):
        if item in distance_data_by_category:
            y_data = distance_data_by_category[item]
            fig.add_trace(go.Box(y=y_data, name=item, boxpoints='outliers', fillcolor='rgba(0,0,0,0)', line=dict(color='darkgrey', width=2)))
            fig.add_shape(type="line", x0=x_pos_counter-0.23, y0=np.median(y_data), x1=x_pos_counter+0.23, y1=np.median(y_data), line=dict(color="red", width=2))
            x_axis_labels.append(item)
            x_pos_counter += 1
            
    elif isinstance(item, tuple):
        cat1, cat2 = item
        if cat1 in distance_data_by_category and cat2 in distance_data_by_category:
            y_data1, y_data2 = distance_data_by_category[cat1], distance_data_by_category[cat2]
            fig.add_trace(go.Box(y=y_data1, name=cat1, boxpoints='outliers', fillcolor='rgba(0,0,0,0)', line=dict(color=colors['base'], width=2)))
            fig.add_shape(type="line", x0=x_pos_counter-0.23, y0=np.median(y_data1), x1=x_pos_counter+0.23, y1=np.median(y_data1), line=dict(color="red", width=3))
            x_pos_counter += 1
            fig.add_trace(go.Box(y=y_data2, name=cat2, boxpoints='outliers', fillcolor='rgba(0,0,0,0)', line=dict(color=colors['reciprocal'], width=2)))
            fig.add_shape(type="line", x0=x_pos_counter-0.23, y0=np.median(y_data2), x1=x_pos_counter+0.23, y1=np.median(y_data2), line=dict(color="red", width=3))
            x_pos_counter += 1
            x_axis_labels.extend([cat1, cat2])
            
            stat, p_value = mannwhitneyu(y_data1, y_data2, alternative='two-sided')
            y_max_pair = max(max(y_data1), max(y_data2))
            y_pos = y_max_pair * 1.05
            x_center = x_pos_counter - 1.5
            fig.add_shape(type="line", x0=x_center - 0.4, y0=y_pos, x1=x_center + 0.4, y1=y_pos, line=dict(color="black", width=1))
            fig.add_annotation(x=x_center, y=np.log10(y_pos), text=format_p_value(p_value), showarrow=False, yshift=10, font=dict(size=20, color='black')) # Adjusted y-positioning slightly for log scale

# 6. Perform and Annotate ANOVA for intra-quadrant groups
intra_quadrant_cats = ['1 → 1', '2 → 2', '3 → 3', '4 → 4'] # MODIFIED
anova_data = [np.array(distance_data_by_category[cat]) for cat in intra_quadrant_cats if cat in distance_data_by_category]
if len(anova_data) >= 2:
    f_stat, p_value_anova = f_oneway(*anova_data)
    y_max_anova_groups = max(max(d) for d in anova_data if len(d) > 0)
    y_pos_anova = y_max_anova_groups * 1.2
    fig.add_shape(type="line", x0=-0.25, y0=y_pos_anova, x1=len(anova_data)-0.75, y1=y_pos_anova, line=dict(color="black", width=2))
    # This will now display "ANOVA: ns" given the p-value of 0.466
    fig.add_annotation(x=(len(anova_data)-1)/2, y=np.log10(y_pos_anova), text=f"{format_p_value(p_value_anova)}", showarrow=False, yshift=10, font=dict(size=24, color='black'))

# 7. Finalize layout
fig.update_layout(
    # MODIFIED: Changed X-axis title for clarity
    xaxis_title='Inter/intra-quadrant trajectory (Origin → Destination)',
    yaxis_title='Predicted distance error (m)',
    showlegend=False,
    xaxis=dict(categoryorder='array', categoryarray=x_axis_labels, tickangle=-45),
    plot_bgcolor='white',
    font=dict(family="Arial, sans-serif", size=28, color="black"),
    yaxis_type="log",
)
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True, gridcolor='lightgrey')

fig.show()