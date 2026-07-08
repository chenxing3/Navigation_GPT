import pandas as pd
import numpy as np
import math
import os
import sys
import glob # Added to find real trajectory files

import pyproj
from shapely.geometry import Point
from shapely.ops import transform
from pyproj import Transformer

import plotly.graph_objects as go
from scipy.stats import mannwhitneyu

# --- User-Provided Setup Code ---
geodesic = pyproj.Geod(ellps='WGS84')
proj_meters = pyproj.CRS('EPSG:6991') # A suitable projected CRS for distance measurement
proj_latlng = pyproj.CRS('EPSG:4326') # Standard lat/lon
project_to_coords = Transformer.from_crs(proj_meters, proj_latlng, always_xy=True).transform
project_to_meters = Transformer.from_crs(proj_latlng, proj_meters, always_xy=True).transform

def coordinate_to_xy(lon, lat):
    """Converts longitude and latitude to x, y meters."""
    try:
        pt_meters = transform(project_to_meters, Point(lon, lat))
        return pt_meters.x, pt_meters.y
    except Exception:
        return None, None

def xy_to_coordinate(x, y):
    """Converts x, y meters back to longitude and latitude."""
    if x is None or y is None or math.isnan(x) or math.isnan(y):
        return None, None
    try:
        pt_latlng = transform(project_to_coords, Point(x, y))
        return pt_latlng.x, pt_latlng.y
    except Exception as e:
        print(f"Error in xy_to_coordinate for x={x}, y={y}: {e}")
        return None, None


# --- Helper function to recalculate trajectory ---
def calculate_trajectory_coordinates(x_start, y_start, azimuth_sequence, distance):
    try:
        azimuth_sequence = [float(az) for az in azimuth_sequence if str(az).strip()] # Ensure az is string before strip
    except (ValueError, TypeError) as e:
        print(f"Error converting azimuths to float: {e}. Azimuth sequence: {azimuth_sequence}")
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




# --- Data File Paths ---
Model_generated_trajectory_files = [

    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251203_205445.feather'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251203_230917.feather'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251204_004800.feather'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251204_030131.feather'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251217_105031.feather'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251217_110922.feather'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251217_141425.feather'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251217_141527.feather'],
    ['../dataset/trajectory/full_model/trajectories_iterative_with_echo_20251217_141721.feather'],
]

# Use glob to find all real trajectory files in the specified directory
Real_trajectory_files = glob.glob("../dataset/real_bats_traj/*.csv")
# --- Constants for Calculation ---
AZIMUTH_STEP_DISTANCE = 5  # The distance (in meters) for each step in the generated model
MAX_SEGMENT_GAP = 100      # The maximum distance (in meters) between two points before splitting a real trajectory
# --- Main Processing ---
model_straightness_indices = []
real_straightness_indices = []

print("--- Processing Model-Generated Trajectories ---")
flat_model_files = [file for sublist in Model_generated_trajectory_files for file in sublist]

for file_path in flat_model_files:
    if not os.path.exists(file_path):
        print(f"Warning: Model file not found, skipping: {file_path}")
        continue
    
    print(f"Processing model file: {file_path}")
    df = pd.read_feather(file_path)
    for index, trajectory_data in df.iterrows():
        loc1_x, loc1_y = trajectory_data['loc1_x'], trajectory_data['loc1_y']

        
        azimuths_data = trajectory_data['azimuths_tokens_str']
        if isinstance(azimuths_data, bytes):
            azimuths_data = azimuths_data.decode('utf-8')
        azimuths = [i.replace("Azi_", "").replace("_", "") for i in azimuths_data.split(" ") if i.strip()]
        num_steps = len(azimuths) + 1
        actual_length = AZIMUTH_STEP_DISTANCE * num_steps

        traj_path_x, traj_path_y = calculate_trajectory_coordinates(
            loc1_x, loc1_y, azimuths, AZIMUTH_STEP_DISTANCE
        )
        loc2_x = traj_path_x[-1]
        loc2_y = traj_path_y[-1]


        bee_line_distance = math.hypot(loc2_x - loc1_x, loc2_y - loc1_y)
        if actual_length > 0:
            straightness_index = bee_line_distance / actual_length
            model_straightness_indices.append(straightness_index)

print(f"\nFinished processing {len(model_straightness_indices)} generated trajectories.")
print("\n--- Processing Real Bat Trajectories ---")
for file_path in Real_trajectory_files:
    print(f"Processing real trajectory file: {file_path}")
    df = pd.read_csv(file_path)
    if len(df) < 2: continue
    if 'lon' not in df.columns or 'lat' not in df.columns: continue
    coords = [coordinate_to_xy(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]
    df['x'] = [c[0] for c in coords]
    df['y'] = [c[1] for c in coords]
    df.dropna(subset=['x', 'y'], inplace=True)
    if len(df) < 2: continue
    df['dist_to_prev'] = np.sqrt(df['x'].diff()**2 + df['y'].diff()**2)
    df['segment_id'] = (df['dist_to_prev'] > MAX_SEGMENT_GAP).cumsum()
    for seg_id, segment in df.groupby('segment_id'):
        if len(segment) < 2: continue
        start_point, end_point = segment.iloc[0], segment.iloc[-1]
        bee_line_distance = math.hypot(end_point['x'] - start_point['x'], end_point['y'] - start_point['y'])
        actual_length = segment['dist_to_prev'].sum()
        if actual_length > 0:
            straightness_index = bee_line_distance / actual_length
            real_straightness_indices.append(straightness_index)

print(f"\nFinished processing {len(real_straightness_indices)} real trajectory segments.")

# --- Analysis and Visualization ---

if not model_straightness_indices or not real_straightness_indices:
    print("\nCould not perform analysis: one or both data sources resulted in zero valid trajectories.")
    sys.exit()
    
print("\n--- Comparison of Trajectory Straightness ---")
print("\n--- Summary Statistics ---")
print(f"LLM Generated Trajectories:\n  Count: {len(model_straightness_indices)}\n  Mean Straightness: {np.mean(model_straightness_indices):.4f}\n  Std Dev: {np.std(model_straightness_indices):.4f}\n")
print(f"Real Bat Trajectories:\n  Count: {len(real_straightness_indices)}\n  Mean Straightness: {np.mean(real_straightness_indices):.4f}\n  Std Dev: {np.std(real_straightness_indices):.4f}\n")
print("\n--- Statistical Test (Mann-Whitney U) ---")
u_stat, p_value = mannwhitneyu(model_straightness_indices, real_straightness_indices, alternative='two-sided')
print(f"Statistic: {u_stat:.2f}, P-value: {p_value:.4f}")
if p_value < 0.05:
    print("Result: The difference between the straightness of LLM-generated and real trajectories is statistically significant (p < 0.05).")
else:
    print("Result: There is no statistically significant difference between the straightness of the two groups (p >= 0.05).")

# --- Visualization (Box Plot) with Custom Styling ---
print("\n--- Generating Visualization ---")
# --- Helper function to format p-values for the plot ---
def format_p_value(p):
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return "p < 0.01"
    elif p < 0.05:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.3f}" # or return "n.s." for not significant

# --- Visualization with Statistical Annotation ---
print("\n--- Generating Visualization ---")
fig = go.Figure()

data_groups = {'Model': model_straightness_indices, 'Real bat': real_straightness_indices}

for i, (name, data) in enumerate(data_groups.items()):
    fig.add_trace(go.Box(y=data, name=name, boxpoints='outliers', fillcolor='rgba(0,0,0,0)', line=dict(color='black', width=2)))
    median_val = np.median(data)
    fig.add_shape(type="line", x0=i - 0.24, y0=median_val, x1=i + 0.24, y1=median_val, line=dict(color="red", width=3))

# --- Add Statistical Annotation ---
# Determine the y-position for the annotation bracket
y_max = max(np.max(model_straightness_indices), np.max(real_straightness_indices))
y_range = y_max - min(np.min(model_straightness_indices), np.min(real_straightness_indices))
bracket_y = y_max + y_range * 0.1  # Position bracket 10% of the range above the max point
text_y = y_max + y_range * 0.18    # Position text slightly above the bracket

# Create the annotation text
annotation_text = f"ns" 
# print(f"{u_stat:,.0f}<br>{format_p_value(p_value)}")

# Add the bracket lines
fig.add_shape(type="line", x0=0, y0=bracket_y, x1=1, y1=bracket_y, line=dict(color="black", width=1.5)) # Horizontal line
fig.add_shape(type="line", x0=0, y0=bracket_y-y_range*0.03, x1=0, y1=bracket_y, line=dict(color="black", width=1.5)) # Left tick
fig.add_shape(type="line", x0=1, y0=bracket_y-y_range*0.03, x1=1, y1=bracket_y, line=dict(color="black", width=1.5)) # Right tick

# Add the text annotation
fig.add_annotation(
    x=0.5, # Midpoint between the two boxes (at x=0 and x=1)
    y=text_y,
    text=annotation_text,
    showarrow=False,
    font=dict(size=20, color="black")
)

# --- Update Layout ---
fig.update_layout(
    # title_text='Comparison of Trajectory Straightness (Generated vs. Real)',
    yaxis_title='Straightness Index',
    xaxis_title='Data Source',
    plot_bgcolor='white',
    font=dict(family="Arial, sans-serif", size=24, color='black'),
    paper_bgcolor='white',
    showlegend=False,
    # Increase the top margin to make space for the annotation
    margin=dict(t=100), 
    yaxis_range=[0, y_max + y_range * 0.3] # Extend y-axis to fit annotation
)

fig.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')

print("Displaying interactive plot...")
fig.show()
